from __future__ import annotations

import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pool_lora_mc_batches import pool  # noqa: E402


FIELDS = [
    "module", "raw", "cond", "null_act_raw", "null_act_cond",
    "excess_raw", "excess_cond", "excess_raw_corrected",
]


def write_batch(path: Path, raw_null: float, cond_null: float,
                module: str = "encoder.layers.0.self_attn.q_proj") -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "module": module,
            "raw": "0.7",
            "cond": "0.6",
            "null_act_raw": str(raw_null),
            "null_act_cond": str(cond_null),
            "excess_raw": str(0.7 - raw_null),
            "excess_cond": str(0.6 - cond_null),
            "excess_raw_corrected": str(0.7 - raw_null),
        })


class PoolLoRABatchesTest(unittest.TestCase):
    def test_pool_accepts_gzip_and_recomputes_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv.gz"
            second = root / "second.csv"
            output = root / "pooled.csv"
            write_batch(first, 0.4, 0.3)
            write_batch(second, 0.5, 0.4)
            pool([first, second], output)
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["null_act_raw"]), 0.45)
            self.assertEqual(float(row["null_act_cond"]), 0.35)
            self.assertAlmostEqual(float(row["excess_raw"]), 0.25)
            self.assertAlmostEqual(float(row["excess_cond"]), 0.25)
            self.assertEqual(row["excess_raw_corrected"], row["excess_raw"])

    def test_pool_refuses_static_cell_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            write_batch(first, 0.4, 0.3)
            write_batch(second, 0.5, 0.4, module="different.module")
            with self.assertRaisesRegex(ValueError, "static value mismatch"):
                pool([first, second], root / "pooled.csv")


if __name__ == "__main__":
    unittest.main()
