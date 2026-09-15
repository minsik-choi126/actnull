"""Checks for the one-pass exact-null sensitivity grid."""
from __future__ import annotations

import math
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import actnull.null as act
from scripts.exact_overlap import exact_block_null, projector_block_masses
from scripts.exact_sensitivity import (
    Aggregate,
    coordinate_block_masses,
    module_sensitivity_rows,
    parse_unique_floats,
    parse_unique_ints,
    public_checkpoint_id,
    read_subspace_cache,
    setting_tag,
    summary_rows,
    write_subspace_cache,
)


def haar_frame(dimension: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    return act.gaussian_haar_frame(
        torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
    )


class ExactSensitivityTest(unittest.TestCase):
    def test_grid_parsing_is_sorted_and_unique(self) -> None:
        self.assertEqual(parse_unique_ints("32,4,8,4", "--k"), (4, 8, 32))
        self.assertEqual(parse_unique_floats("3,1.5,1", "--z"), (1.0, 1.5, 3.0))
        self.assertEqual(setting_tag(64, 1.5), "m64_z1p5")
        self.assertEqual(
            public_checkpoint_id(
                "/tmp/hf/hub/models--openai--clip-vit-base-patch16/snapshots/abc123"
            ),
            "openai/clip-vit-base-patch16@abc123",
        )

    def test_cached_coordinate_masses_match_reference_implementation(self) -> None:
        generator = torch.Generator().manual_seed(41)
        d_in, max_m, max_k = 17, 9, 5
        ambient = haar_frame(d_in, d_in, generator)
        basis = ambient[:, :max_m].T
        frame = haar_frame(d_in, max_k, generator)
        coordinates = basis @ frame
        blocks = [[0], [1, 2], [3, 4, 5]]
        m, k = 6, 3
        got_masses, got_dimensions = coordinate_block_masses(
            coordinates, k, blocks, m, d_in
        )
        expected_masses, expected_dimensions = projector_block_masses(
            frame[:, :k], basis[:m], blocks, d_in
        )
        self.assertEqual(got_dimensions, expected_dimensions)
        for got, expected in zip(got_masses, expected_masses):
            self.assertAlmostEqual(got, expected, places=12)

    def test_one_pass_grid_matches_repeated_exact_overlap_calls(self) -> None:
        generator = torch.Generator().manual_seed(2027)
        d_in, max_m, max_k = 15, 8, 4
        ambient = haar_frame(d_in, d_in, generator)
        basis = ambient[:, :max_m].T
        subspaces = {
            name: haar_frame(d_in, max_k, generator) for name in ("a", "b", "c")
        }
        eigvals = torch.tensor(
            [9.0, 7.5, 7.2, 4.0, 3.8, 2.0, 1.9, 0.7], dtype=torch.float64
        )
        standard_errors = torch.tensor(
            [0.1, 0.2, 0.2, 0.1, 0.15, 0.1, 0.1, 0.05], dtype=torch.float64
        )
        k_values = (2, 4)
        m_values = (4, 8)
        z_values = (1.0, 2.0)
        rows, counts = module_sensitivity_rows(
            "layer", d_in, subspaces, basis, eigvals, standard_errors,
            k_values, m_values, z_values,
        )
        self.assertEqual(len(rows), math.comb(3, 2) * len(k_values))
        self.assertEqual(set(counts), {setting_tag(m, z) for m in m_values for z in z_values})

        for row in rows:
            a, b, k = str(row["a"]), str(row["b"]), int(row["k"])
            self.assertAlmostEqual(
                float(row["raw"]),
                act.overlap(subspaces[a][:, :k], subspaces[b][:, :k]),
                places=13,
            )
            for m in m_values:
                for z in z_values:
                    blocks = act.resolvable_blocks(eigvals[:m], standard_errors[:m], z)
                    mass_a, dimensions = projector_block_masses(
                        subspaces[a][:, :k], basis[:m], blocks, d_in
                    )
                    mass_b, dimensions_b = projector_block_masses(
                        subspaces[b][:, :k], basis[:m], blocks, d_in
                    )
                    self.assertEqual(dimensions, dimensions_b)
                    expected = exact_block_null(mass_a, mass_b, dimensions, k)
                    self.assertAlmostEqual(
                        float(row[f"null_{setting_tag(m, z)}"]), expected, places=12
                    )

    def test_summary_uses_ratio_of_sums_not_mean_row_fraction(self) -> None:
        aggregate = Aggregate()
        aggregate.add(raw=0.3, isotropic=0.1, null=0.2)   # fraction .5
        aggregate.add(raw=0.9, isotropic=0.1, null=0.3)   # fraction .25
        rows = summary_rows(
            {(8, 64, 2.0): aggregate},
            base_spec="base",
            task_count=2,
            module_count=1,
            block_counts={"layer": {"m64_z2": 5}},
        )
        # Ratio of sums is .3 / 1.0 = .3, unlike mean(.5, .25) = .375.
        self.assertAlmostEqual(float(rows[0]["fraction_explained"]), 0.3, places=14)
        self.assertEqual(rows[0]["weighting"], "equal_cell_ratio_of_sums")

    def test_subspace_cache_round_trip_and_identity_guard(self) -> None:
        generator = torch.Generator().manual_seed(3)
        frames = {
            "a": haar_frame(10, 4, generator).float(),
            "b": haar_frame(10, 4, generator).float(),
        }
        norms = {"a": 0.1, "b": 0.2}
        identity = {"module": "layer", "sources": ["snapshot-a", "snapshot-b"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer.pt"
            write_subspace_cache(path, identity, frames, norms, max_k=4)
            loaded = read_subspace_cache(
                path, identity, ("a", "b"), d_in=10, max_k=3, device="cpu"
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            loaded_frames, loaded_norms = loaded
            self.assertEqual(loaded_frames["a"].shape, (10, 3))
            self.assertEqual(loaded_norms, norms)
            self.assertIsNone(
                read_subspace_cache(
                    path, {"module": "changed"}, ("a", "b"), 10, 3, "cpu"
                )
            )

    def test_cli_writes_compact_cells_summary_and_explicit_scope(self) -> None:
        generator = torch.Generator().manual_seed(19)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir, cov_dir, fold_dir = root / "base", root / "cov", root / "fold"
            base_dir.mkdir()
            cov_dir.mkdir()
            fold_dir.mkdir()
            key = "layer.weight"
            base_weight = torch.randn(10, 10, generator=generator)
            torch.save({key: base_weight}, base_dir / "weights.bin")
            expert_args = []
            for name in ("a", "b", "c"):
                expert_dir = root / name
                expert_dir.mkdir()
                delta = 0.1 * torch.randn(10, 10, generator=generator)
                torch.save({key: base_weight + delta}, expert_dir / "weights.bin")
                expert_args.append(f"{name}={expert_dir}")

            ambient = haar_frame(10, 10, generator).float()
            eigenvalues = torch.linspace(5.0, 0.5, 8)
            torch.save(
                {"eigvecs": ambient[:, :8].T, "eigvals": eigenvalues, "d": 10},
                cov_dir / "layer.pt",
            )
            (cov_dir / "manifest.json").write_text(
                json.dumps({"m_out": 8, "modules": {"layer": {}}})
            )
            for fold in range(4):
                target = fold_dir / f"f{fold}"
                target.mkdir()
                torch.save(
                    {"eigvals": eigenvalues + 0.01 * fold}, target / "layer.pt"
                )

            cells, summary, metadata = root / "cells.csv", root / "summary.csv", root / "meta.json"
            script = Path(__file__).resolve().parents[1] / "scripts" / "exact_sensitivity.py"
            command = [
                sys.executable, str(script), "--base", str(base_dir), "--experts", *expert_args,
                "--cov", str(cov_dir), "--fold-cov", str(fold_dir),
                "--k-values", "2,4", "--null-m-values", "4,8",
                "--block-z-values", "1,2", "--expect-tasks", "3",
                "--expect-modules", "1", "--out", str(cells),
                "--summary", str(summary), "--metadata", str(metadata),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, msg=completed.stderr + completed.stdout)
            with cells.open() as handle:
                cell_rows = list(csv.DictReader(handle))
            with summary.open() as handle:
                summary_grid = list(csv.DictReader(handle))
            scope = json.loads(metadata.read_text())["scope"]
            self.assertEqual(len(cell_rows), math.comb(3, 2) * 2)
            self.assertEqual(len(summary_grid), 2 * 2 * 2)
            self.assertEqual(scope["n_tasks"], 3)
            self.assertEqual(scope["n_modules"], 1)
            self.assertEqual(scope["definition"], "all unordered task pairs x all retained modules")
            # A distributable result must not expose the machine's temporary/workspace path.
            self.assertNotIn(str(root), metadata.read_text())
            self.assertNotIn(str(root), summary.read_text())


if __name__ == "__main__":
    unittest.main()
