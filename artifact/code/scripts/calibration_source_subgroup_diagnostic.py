#!/usr/bin/env python3
"""Audit overlap by the number of pair endpoints used as calibration sources.

The calculation is read-only.  Within each architecture and endpoint-count
subgroup, rows are weighted equally; raw, isotropic, and fitted-null values are
averaged before computing residual and the reproduced fraction.  Output paths
are created exclusively and never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics as stats
from pathlib import Path


CALIBRATION_SOURCES = (
    "cifar10", "dtd", "eurosat", "gtsrb", "resisc45", "stl10", "sun397"
)
ARMS = {
    "ViT-B/16": "v7_exact_clip-vit-base-patch16.csv.gz",
    "ViT-B/32": "v7_exact_clip-vit-base-patch32.csv.gz",
    "ViT-L/14 q/v": "v7_exact_clip-vit-large-patch14.csv.gz",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    raw = stats.fmean(float(row["raw"]) for row in rows)
    isotropic = stats.fmean(float(row["null_iso"]) for row in rows)
    fitted = stats.fmean(float(row["null_exact"]) for row in rows)
    denominator = raw - isotropic
    pairs = {tuple(sorted((row["a"], row["b"]))) for row in rows}
    modules = {row["module"] for row in rows}
    return {
        "rows": len(rows),
        "unique_task_pairs": len(pairs),
        "modules": len(modules),
        "mean_raw": raw,
        "mean_isotropic": isotropic,
        "mean_activation_null": fitted,
        "residual_raw_minus_null": raw - fitted,
        "reproduced_fraction_percent": 100.0 * (fitted - isotropic) / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_set = set(CALIBRATION_SOURCES)
    architectures = {}
    for label, filename in ARMS.items():
        rows = read_rows(args.results_root / filename)
        grouped = {0: [], 1: [], 2: []}
        for row in rows:
            endpoints = int(row["a"] in source_set) + int(row["b"] in source_set)
            grouped[endpoints].append(row)
        architectures[label] = {
            "source_row_file": filename,
            "all_tasks": sorted({row[key] for row in rows for key in ("a", "b")}),
            "groups": {
                str(endpoint_count): summarize(grouped[endpoint_count])
                for endpoint_count in (0, 1, 2)
            },
        }

    payload = {
        "schema_version": 1,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": {
            "group": "number of unordered task-pair endpoints in calibration_source_tasks",
            "aggregation": "equal row weights within group; average raw, isotropic, and activation-null values before residual and ratio",
            "residual": "mean(raw) - mean(activation_null)",
            "reproduced_fraction_percent": "100 * (mean(activation_null) - mean(isotropic)) / (mean(raw) - mean(isotropic))",
        },
        "calibration_source_tasks": list(CALIBRATION_SOURCES),
        "architectures": architectures,
        "path_policy": "Only artifact-relative row filenames are recorded; no local paths.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output: {args.output}")


if __name__ == "__main__":
    main()
