#!/usr/bin/env python3
"""Rebuild the calibration image set used to estimate C_H.

⛔ THE CALIBRATION SET IS PART OF THE MEASUREMENT. C_H is an expectation over some input
distribution, so two runs on different images are two different measurements even when the
manifests look alike. results/act_cov/calibration_images.json records the exact file list, the
per-source counts and a hash; this script regenerates it.

⛔ AND IT MUST NOT BE ONE TASK'S DATA. Estimating C_H on a single dataset makes it the base
model's geometry ON THAT TASK'S INPUTS, which reintroduces exactly the conditioning the null
exists to remove. We draw round-robin across several of the benchmark's own sources.

Usage:  python analysis/fetch_calibration_images.py --out <dir> --per_source 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

SOURCES = ["eurosat", "dtd", "gtsrb", "resisc45", "sun397", "cifar10", "stl10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=32)
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()

    from datasets import load_dataset
    out = Path(args.out)
    total = 0
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        try:
            ds = load_dataset(f"tanganke/{name}", split="train", streaming=True)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}")
            continue
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
        print(f"[ok] {name}: {got}")
    print(f"wrote {total} images to {out}")
    print("Compare against results/act_cov/calibration_images.json before using a new set.")


if __name__ == "__main__":
    main()
