#!/usr/bin/env python3
"""Rebuild the calibration image set used to estimate C_H.

⛔ THE CALIBRATION SET IS PART OF THE MEASUREMENT. C_H is an expectation over some input
distribution, so two runs on different images are two different measurements even when the
manifests look alike. The artifact's source-resampling maps record the exact file names and
draw assignments; this script regenerates their pinned source pools.

⛔ AND IT MUST NOT BE ONE TASK'S DATA. Estimating C_H on a single dataset makes it the base
model's geometry ON THAT TASK'S INPUTS, which reintroduces exactly the conditioning the null
exists to remove. We draw round-robin across several of the benchmark's own sources.

Usage:  python scripts/fetch_calibration_images.py \
          --revision-record ../upstream_revisions.json --out <dir> --per_source 143
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = ["eurosat", "dtd", "gtsrb", "resisc45", "sun397", "cifar10", "stl10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=143)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument(
        "--revision-record", required=True,
        help="artifact upstream_revisions.json; dataset revisions are never inferred from HEAD",
    )
    args = ap.parse_args()

    from datasets import load_dataset
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"[refuse] output already exists: {out}")
    revision_record = json.loads(Path(args.revision_record).read_text(encoding="utf-8"))
    calibration = revision_record["calibration_datasets"]
    template = calibration["repository_template"]
    revisions = calibration["revision_by_source"]
    total = 0
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        if name not in revisions:
            raise SystemExit(f"[refuse] no pinned calibration revision for {name!r}")
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        try:
            repository = template.format(source=name)
            ds = load_dataset(repository, revision=revisions[name],
                              split="train", streaming=True)
        except Exception as e:
            raise SystemExit(
                f"[refuse] failed to load pinned source {name}: {type(e).__name__}"
            ) from e
        got = 0
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            img.convert("RGB").save(d / f"{got:04d}.png")
            got += 1
            if got >= args.per_source:
                break
        total += got
        if got != args.per_source:
            raise SystemExit(
                f"[refuse] pinned source {name} yielded {got}, expected {args.per_source}"
            )
        print(f"[ok] {name}@{revisions[name]}: {got}")
    print(f"wrote {total} images to {out}")
    print("Compare file names against the shipped calibration resampling maps.")


if __name__ == "__main__":
    main()
