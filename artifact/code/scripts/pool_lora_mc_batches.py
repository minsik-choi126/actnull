#!/usr/bin/env python3
"""Pool independent LoRA-null batch means into one row-level MC estimate.

Each input must contain the same cells in the same order.  Columns unrelated to
the Monte Carlo null must be byte-for-byte identical across inputs.  The two
null columns are averaged and their residual columns are recomputed.  Both CSV
and CSV.gz inputs are accepted; the output format follows its suffix.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from contextlib import ExitStack
from pathlib import Path
from typing import IO


NULL_COLUMNS = ("null_act_raw", "null_act_cond")
DERIVED_COLUMNS = ("excess_raw", "excess_cond", "excess_raw_corrected")
VARYING_COLUMNS = frozenset(NULL_COLUMNS + DERIVED_COLUMNS)


def _open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def pool(inputs: list[Path], output: Path) -> None:
    if len(inputs) < 2:
        raise ValueError("at least two independent batch files are required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input paths must be distinct")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = [stack.enter_context(_open_text(path, "rt")) for path in inputs]
        readers = [csv.DictReader(handle) for handle in handles]
        fieldnames = readers[0].fieldnames
        if not fieldnames:
            raise ValueError(f"empty or headerless input: {inputs[0]}")
        required = {"raw", "cond", *NULL_COLUMNS, *DERIVED_COLUMNS}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")
        for path, reader in zip(inputs[1:], readers[1:]):
            if reader.fieldnames != fieldnames:
                raise ValueError(f"column mismatch: {path}")

        output_handle = stack.enter_context(_open_text(output, "xt"))
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        row_number = 1
        while True:
            batch_rows = [next(reader, None) for reader in readers]
            present = [row is not None for row in batch_rows]
            if not any(present):
                break
            row_number += 1
            if not all(present):
                raise ValueError(f"input row-count mismatch at CSV row {row_number}")
            rows = [row for row in batch_rows if row is not None]
            reference = rows[0]
            for batch_index, row in enumerate(rows[1:], 2):
                for column in fieldnames:
                    if column not in VARYING_COLUMNS and row[column] != reference[column]:
                        raise ValueError(
                            f"static value mismatch at CSV row {row_number}, "
                            f"batch {batch_index}, column {column!r}"
                        )

            pooled = dict(reference)
            for column in NULL_COLUMNS:
                values = [float(row[column]) for row in rows]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f"non-finite {column!r} at CSV row {row_number}"
                    )
                pooled[column] = str(sum(values) / len(values))
            pooled["excess_raw"] = str(
                float(reference["raw"]) - float(pooled["null_act_raw"])
            )
            pooled["excess_cond"] = str(
                float(reference["cond"]) - float(pooled["null_act_cond"])
            )
            pooled["excess_raw_corrected"] = pooled["excess_raw"]
            writer.writerow(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="independent per-row batch CSV/CSV.gz files")
    parser.add_argument("--out", required=True, help="new pooled CSV/CSV.gz file")
    args = parser.parse_args()
    pool([Path(path) for path in args.inputs], Path(args.out))


if __name__ == "__main__":
    main()
