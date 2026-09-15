#!/usr/bin/env python3
"""Create a source-stratified bootstrap calibration sample as relative symlinks.

Each source is sampled independently with replacement.  Occurrences, rather than
unique image targets, are striped round-robin into disjoint fold directories so a
target selected twice remains two bootstrap observations.  The JSON record is
path-sanitized and sufficient to reconstruct every selected occurrence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCES = (
    "cifar10",
    "dtd",
    "eurosat",
    "gtsrb",
    "resisc45",
    "stl10",
    "sun397",
)


def digest_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, start=link.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--per-source", type=int, default=143)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root)
    if out_root.exists() or out_root.is_symlink():
        raise SystemExit(f"[refuse] output already exists: {out_root}")
    if args.per_source < 1 or args.folds < 2:
        raise SystemExit("[refuse] --per-source must be positive and --folds >= 2")
    if len(set(args.sources)) != len(args.sources):
        raise SystemExit("[refuse] --sources contains duplicates")

    pools: dict[str, list[Path]] = {}
    inventory_lines: list[str] = []
    for source in args.sources:
        source_dir = source_root / source
        files = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(files) < 2:
            raise SystemExit(f"[refuse] source {source!r} has only {len(files)} images")
        if len({path.name for path in files}) != len(files):
            raise SystemExit(f"[refuse] duplicate filenames in source {source!r}")
        pools[source] = files
        inventory_lines.extend(
            f"{source}/{path.name}\t{path.stat().st_size}" for path in files
        )

    out_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    fold_sizes = Counter()
    selected_by_source: dict[str, list[int]] = {}

    # A stable source-specific seed makes a source's draw invariant to source ordering.
    for source_index, source in enumerate(args.sources):
        source_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{source}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(source_seed)
        selected_by_source[source] = [
            rng.randrange(len(pools[source])) for _ in range(args.per_source)
        ]

    # Match the original calibration protocol: round-robin sources, then position mod folds.
    for draw_index in range(args.per_source):
        for source_index, source in enumerate(args.sources):
            pool_index = selected_by_source[source][draw_index]
            target = pools[source][pool_index]
            ordinal = draw_index * len(args.sources) + source_index
            fold = ordinal % args.folds
            link_name = f"draw_{draw_index:03d}__pool_{pool_index:03d}{target.suffix.lower()}"
            image_link = out_root / "images" / source / link_name
            fold_link = out_root / "folds" / f"f{fold}" / source / link_name
            relative_symlink(target, image_link)
            relative_symlink(target, fold_link)
            fold_sizes[f"f{fold}"] += 1
            records.append(
                {
                    "ordinal": ordinal,
                    "source": source,
                    "draw_index": draw_index,
                    "pool_index": pool_index,
                    "source_file": target.name,
                    "link_name": link_name,
                    "fold": f"f{fold}",
                }
            )

    selected_lines = [
        f"{r['ordinal']}\t{r['source']}\t{r['draw_index']}\t{r['pool_index']}\t"
        f"{r['source_file']}\t{r['fold']}" for r in records
    ]
    unique_counts = {
        source: len(set(indices)) for source, indices in selected_by_source.items()
    }
    manifest = {
        "schema_version": 1,
        "method": "source_stratified_nonparametric_bootstrap_with_replacement",
        "rng": "Python random.Random (MT19937), SHA256-derived source-specific seeds",
        "seed": args.seed,
        "source_root_basename": source_root.name,
        "sources": list(args.sources),
        "per_source": args.per_source,
        "n_items": len(records),
        "fold_rule": "ordinal = draw_index * n_sources + source_index; fold = ordinal mod n_folds",
        "n_folds": args.folds,
        "fold_sizes": dict(sorted(fold_sizes.items())),
        "pool_sizes": {source: len(pool) for source, pool in pools.items()},
        "unique_targets_per_source": unique_counts,
        "duplicate_occurrences_per_source": {
            source: args.per_source - unique_counts[source] for source in args.sources
        },
        "source_pool_name_size_sha256": digest_lines(sorted(inventory_lines)),
        "selected_occurrences_sha256": digest_lines(selected_lines),
        "records": records,
    }
    (out_root / "resample_map.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[resample] seed={args.seed} n={len(records)} folds={dict(sorted(fold_sizes.items()))}"
    )
    print(f"[resample] unique targets per source: {unique_counts}")
    print(f"[resample] wrote {out_root}")


if __name__ == "__main__":
    main()
