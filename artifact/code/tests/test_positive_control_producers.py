"""Synthetic checks for the portable positive-control producers."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import actnull.null as act
from actnull.experiment import parse_named_paths, write_json_new
from actnull.positive_control_utils import (
    initial_coupling_destroying_null,
    materialize_in_adapter_row_space,
    plant_right_directions,
)


ROOT = Path(__file__).resolve().parents[1]


def frame(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    return act.gaussian_haar_frame(
        torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    ).float()


class PositiveControlUtilityTest(unittest.TestCase):
    def test_plant_preserves_spectrum_and_installs_requested_directions(self) -> None:
        generator = torch.Generator().manual_seed(2027)
        reference = torch.randn(7, 12, generator=generator)
        candidate = torch.randn(7, 12, generator=generator)
        planted = plant_right_directions(reference, candidate, 2, 4, generator)
        torch.testing.assert_close(
            torch.linalg.svdvals(planted), torch.linalg.svdvals(candidate),
            rtol=3e-5, atol=2e-6,
        )
        requested = act.top_right(reference, 2)
        delivered = act.top_right(planted, 2)
        torch.testing.assert_close(
            requested @ requested.T, delivered @ delivered.T,
            rtol=3e-5, atol=3e-6,
        )

    def test_materialized_lora_plant_is_attainable(self) -> None:
        generator = torch.Generator().manual_seed(19)
        adapter_a = torch.randn(4, 11, generator=generator)
        proposed = torch.randn(7, 11, generator=generator)
        materialized, adapter_b = materialize_in_adapter_row_space(
            proposed, adapter_a, 2.0
        )
        torch.testing.assert_close(materialized, 2.0 * adapter_b @ adapter_a)
        projector = torch.linalg.pinv(adapter_a) @ adapter_a
        torch.testing.assert_close(
            materialized, materialized @ projector, rtol=3e-5, atol=3e-6
        )

    def test_initial_activation_null_preserves_row_gram(self) -> None:
        generator = torch.Generator().manual_seed(5)
        update = torch.randn(5, 13, generator=generator)
        activation_basis = frame(13, 5, generator).T
        randomized = initial_coupling_destroying_null(
            update, activation_basis, 0, generator, tail_rank=4,
            blocks=[[0], [1, 2], [3, 4]],
        )
        torch.testing.assert_close(
            randomized @ randomized.T, update @ update.T, rtol=3e-5, atol=3e-5
        )

    def test_expert_parser_and_exclusive_output(self) -> None:
        self.assertEqual(parse_named_paths(("a=/x", "b=repo/id")),
                         {"a": "/x", "b": "repo/id"})
        with self.assertRaises(ValueError):
            parse_named_paths(("missing-equals", "b=/x"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            write_json_new(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json_new(target, {"ok": False})
            self.assertEqual(json.loads(target.read_text()), {"ok": True})


class PortableProducerCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.covariance = self.root / "cov"
        self.folds = self.root / "folds"
        self.base.mkdir()
        self.covariance.mkdir()
        self.folds.mkdir()
        generator = torch.Generator().manual_seed(71)
        self.key = "vision_model.layer.weight"
        self.base_weight = torch.randn(6, 10, generator=generator)
        torch.save({self.key: self.base_weight}, self.base / "weights.bin")
        self.experts = []
        for name in ("a", "b"):
            directory = self.root / f"expert-{name}"
            directory.mkdir()
            weight = self.base_weight + 0.1 * torch.randn(6, 10, generator=generator)
            torch.save({self.key: weight}, directory / "weights.bin")
            self.experts.append(f"{name}={directory}")

        activation_basis = frame(10, 4, generator).T
        eigenvalues = torch.tensor([5.0, 3.1, 1.8, 0.7])
        torch.save(
            {
                "eigvecs": activation_basis,
                "eigvals": eigenvalues,
                "d": 10,
                "trace_full": 12.0,
            },
            self.covariance / "layer.pt",
        )
        (self.covariance / "manifest.json").write_text(
            json.dumps({"modules": {"layer": {"participation_ratio": 3.0}}})
        )
        for index in range(4):
            directory = self.folds / f"f{index}"
            directory.mkdir()
            torch.save(
                {"eigvals": eigenvalues + index * 0.01}, directory / "layer.pt"
            )

        self.adapters = []
        for name in ("a", "b"):
            directory = self.root / f"adapter-{name}"
            directory.mkdir()
            (directory / "adapter_config.json").write_text(
                json.dumps({"r": 4, "lora_alpha": 4})
            )
            adapter_a = torch.randn(4, 10, generator=generator)
            adapter_b = torch.randn(6, 4, generator=generator)
            save_file(
                {
                    "base_model.model.layer.lora_A.weight": adapter_a,
                    "base_model.model.layer.lora_B.weight": adapter_b,
                },
                directory / "adapter_model.safetensors",
            )
            self.adapters.append(f"{name}={directory}")

    def run_script(self, script: str, arguments: list[str]) -> dict:
        output = self.root / f"{script}.json"
        command = [
            sys.executable, str(ROOT / "scripts" / f"{script}.py"), *arguments,
            "--out", str(output),
        ]
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
        self.assertTrue(output.is_file())
        self.assertNotIn(str(self.root), output.read_text())
        return json.loads(output.read_text())

    def common_activation_arguments(self) -> list[str]:
        return [
            "--arch", "toy", "--cov", str(self.covariance),
            "--fold-cov", str(self.folds), "--base", str(self.base),
            "--experts", *self.experts, "--modules", "layer",
            "--k", "2", "--draws", "1",
        ]

    def test_activation_power_cli(self) -> None:
        payload = self.run_script(
            "power_activation",
            self.common_activation_arguments()
            + ["--planted", "0,1", "--null-draws", "1"],
        )
        self.assertEqual([row["planted"] for row in payload["rows"]], [0, 1])
        self.assertIn("delivered_overlap", payload["rows"][1])
        base = payload["rows"][0]["raw"]
        self.assertAlmostEqual(payload["rows"][1]["truth"], 0.5 * (1 - base))

    def test_lora_power_cli_reads_local_adapter_snapshots(self) -> None:
        payload = self.run_script(
            "power_lora",
            [
                "--arch", "toy", "--base", str(self.base),
                "--experts", *self.adapters, "--modules", "layer",
                "--k", "2", "--planted", "0,1", "--draws", "1",
            ],
        )
        self.assertEqual(payload["tasks"], ["a", "b"])
        self.assertIsNotNone(payload["rows"][1]["delivered_overlap"])

    def test_ablation_cli(self) -> None:
        payload = self.run_script(
            "ablation", self.common_activation_arguments() + ["--planted", "1"]
        )
        self.assertEqual(len(payload), 4)
        self.assertEqual(
            [row["variant"] for row in payload],
            [
                "initial (neither correction)",
                "singleton signs only",
                "full complement only",
                "corrected (both)",
            ],
        )
        for row in payload:
            self.assertEqual(
                set(row),
                {
                    "variant", "raw0", "null0", "excess0", "raw1", "null1",
                    "excess1", "delivered_overlap", "recovery",
                },
            )
            self.assertAlmostEqual(row["excess0"], row["raw0"] - row["null0"])
            self.assertAlmostEqual(row["excess1"], row["raw1"] - row["null1"])
            self.assertAlmostEqual(
                row["delivered_overlap"], row["raw1"] - row["raw0"]
            )
            self.assertAlmostEqual(
                row["recovery"],
                100.0 * (row["excess1"] - row["excess0"])
                / row["delivered_overlap"],
            )

    def test_baseline_cli(self) -> None:
        payload = self.run_script("baseline_levels", self.common_activation_arguments())
        self.assertEqual(set(payload["level"]), {
            "orthogonality", "isotropic", "regmean", "tikhonov",
            "generative", "permutation", "ours",
        })
        metadata = payload["metadata"]
        self.assertEqual(metadata["schema_version"], 3)
        self.assertAlmostEqual(
            metadata["raw_overlap_over_isotropic_chance"],
            payload["level"]["orthogonality"] / payload["iso"],
        )
        self.assertTrue(
            metadata["ratio_definition"].startswith(
                "raw_null_pair_overlap / isotropic_chance"
            )
        )


if __name__ == "__main__":
    unittest.main()
