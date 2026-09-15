#!/usr/bin/env python3
"""Synthetic regression tests for overlap_bootstrap.py."""
from __future__ import annotations

import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_bootstrap as ob  # noqa: E402


FIELDNAMES = (
    "module",
    "a",
    "b",
    "raw",
    "null_iso",
    "null_exact",
    "null_kind",
    "frac_num",
    "frac_den",
    "excess_exact",
    "frac_exact",
)


def write_balanced_csv(
    path: Path,
    *,
    tasks: tuple[str, ...],
    layers: tuple[int, ...],
    module_types: tuple[str, ...] = ("q_proj",),
) -> None:
    rows = []
    for ai, a in enumerate(tasks):
        for bi, b in enumerate(tasks[ai + 1 :], start=ai + 1):
            pair_effect = 0.025 * (ai + bi - len(tasks) + 1)
            for layer in layers:
                layer_effect = 0.035 * (layer - np.mean(layers))
                for module_type in module_types:
                    iso = 0.1
                    raw = 1.1
                    null = 0.55 + pair_effect + layer_effect
                    rows.append(
                        {
                            "module": f"encoder.layers.{layer}.self_attn.{module_type}",
                            "a": a,
                            "b": b,
                            "raw": raw,
                            "null_iso": iso,
                            "null_exact": null,
                            "null_kind": "exact_haar_block",
                            "frac_num": null - iso,
                            "frac_den": raw - iso,
                            "excess_exact": raw - null,
                            "frac_exact": (null - iso) / (raw - iso),
                        }
                    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.csv"

    def test_point_is_ratio_of_means_and_exact_fields_are_checked(self) -> None:
        write_balanced_csv(
            self.path, tasks=("a", "b", "c", "d"), layers=(0, 1, 2)
        )
        data = ob.load_csv(self.path, label="toy")
        point = ob.point_estimate(data)
        # Symmetric construction: mean null=.55, iso=.1, raw=1.1.
        np.testing.assert_allclose(point, (45.0, 0.55), atol=1e-12)
        self.assertEqual(data.null_column, "null_exact")
        self.assertEqual(data.n_rows, 18)

    def test_gzip_input_matches_plain_csv(self) -> None:
        write_balanced_csv(
            self.path, tasks=("a", "b", "c", "d"), layers=(0, 1, 2)
        )
        compressed = self.path.with_suffix(".csv.gz")
        with self.path.open("rb") as source, gzip.open(compressed, "wb") as target:
            target.write(source.read())
        plain = ob.point_estimate(ob.load_csv(self.path))
        zipped = ob.point_estimate(ob.load_csv(compressed))
        np.testing.assert_allclose(zipped, plain, atol=0.0)

    def test_crossed_weight_is_product_of_endpoint_and_layer_counts(self) -> None:
        write_balanced_csv(self.path, tasks=("a", "b", "c"), layers=(0, 1))
        data = ob.load_csv(self.path)
        task_counts = np.asarray([[2, 1, 0]])
        layer_counts = np.asarray([[0, 2]])
        weights = ob.cell_weights(
            data, task_counts, layer_counts, "joint_task_layer"
        )[0]
        for cell, weight in enumerate(weights):
            pair = data.pairs[data.cell_pair[cell]]
            layer = data.layers[data.cell_layer[cell]]
            expected = (
                task_counts[0, data.tasks.index(pair[0])]
                * task_counts[0, data.tasks.index(pair[1])]
                * layer_counts[0, data.layers.index(layer)]
            )
            self.assertEqual(weight, expected)

    def test_one_layer_joint_equals_task_margin(self) -> None:
        write_balanced_csv(
            self.path, tasks=("a", "b", "c", "d", "e"), layers=(0,)
        )
        result = ob.bootstrap(
            ob.load_csv(self.path), draws=2_000, seed=ob.DEFAULT_SEED, batch_size=97
        )
        intervals = result["percentile_95_intervals"]
        self.assertEqual(
            intervals["joint_task_layer"], intervals["task_only"]
        )

    def test_crossed_interval_reflects_both_synthetic_cluster_effects(self) -> None:
        write_balanced_csv(
            self.path,
            tasks=("a", "b", "c", "d", "e", "f"),
            layers=(0, 1, 2, 3, 4),
        )
        result = ob.bootstrap(
            ob.load_csv(self.path), draws=5_000, seed=ob.DEFAULT_SEED, batch_size=211
        )
        comparison = result["one_way_comparison"]
        for metric in ob.METRICS:
            self.assertGreater(
                comparison[metric]["joint_width"],
                comparison[metric]["task_only_width"],
            )
            self.assertGreater(
                comparison[metric]["joint_width"],
                comparison[metric]["layer_only_width"],
            )

    def test_seed_is_deterministic(self) -> None:
        write_balanced_csv(
            self.path, tasks=("a", "b", "c", "d"), layers=(0, 1, 2)
        )
        data = ob.load_csv(self.path)
        first = ob.bootstrap(data, draws=300, seed=7, batch_size=64)
        second = ob.bootstrap(data, draws=300, seed=7, batch_size=64)
        self.assertEqual(
            first["percentile_95_intervals"], second["percentile_95_intervals"]
        )

    def test_inconsistent_exact_field_fails_closed(self) -> None:
        write_balanced_csv(self.path, tasks=("a", "b", "c"), layers=(0, 1))
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["frac_num"] = "123"
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ob.DataError):
            ob.load_csv(self.path)


if __name__ == "__main__":
    unittest.main()
