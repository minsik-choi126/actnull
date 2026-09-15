#!/usr/bin/env python3
"""Materialize a deterministic 1,001-image MNIST/Cars/SVHN calibration control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from PIL import __version__ as pillow_version
import datasets


SOURCES = (
    {
        "name": "mnist",
        "dataset_id": "ylecun/mnist",
        "config": "mnist",
        "split": "test",
        "builder_fingerprint": "77f3279092a1c1579b2250db8eafed0ad422088c",
        "count": 334,
        "files": ("mnist-test.arrow",),
        "file_sha256": (
            "c1271fa343652fa634d0fb45544b526d89d6a552d18b14a3a05980899a7141fb",
        ),
        "license": "MIT (dataset-card metadata)",
    },
    {
        "name": "stanford_cars",
        "dataset_id": "tanganke/stanford_cars",
        "config": "default",
        "split": "test",
        "builder_fingerprint": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "count": 334,
        "files": (
            "stanford_cars-test-00000-of-00002.arrow",
            "stanford_cars-test-00001-of-00002.arrow",
        ),
        "file_sha256": (
            "f93eaec0f687268607af546a7cc0d36b496d66485ea0f5a6681b1393eb91ad13",
            "f120f8dedf8ac0fec9416b7426c80cd8574b45d147a4627cbc8605b6821f19d4",
        ),
        "license": "dataset-card field empty; local academic/non-commercial use only; do not redistribute images",
    },
    {
        "name": "svhn",
        "dataset_id": "ufldl-stanford/svhn",
        "config": "cropped_digits",
        "split": "test",
        "builder_fingerprint": "f9e1717d73324ebbc1ece84267ccd80d0aca0690",
        "count": 333,
        "files": ("svhn-test.arrow",),
        "file_sha256": (
            "e636de302053f47422fac68d0bfa71385507f30f2328d2c9ff84cfd3fda5615d",
        ),
        "license": "non-commercial use only (dataset card)",
    },
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def stable_rng(seed: int, source: str) -> random.Random:
    source_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()[:8], "big"
    )
    return random.Random(source_seed)


def source_directory(cache_root: Path, source: dict) -> Path:
    prefix = source["dataset_id"].replace("/", "___")
    return (
        cache_root
        / prefix
        / source["config"]
        / "0.0.0"
        / source["builder_fingerprint"]
    )


def relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, start=link.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-cache", required=True)
    parser.add_argument("--revision-record", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--folds", type=int, default=8)
    args = parser.parse_args()

    cache_root = Path(args.datasets_cache).resolve()
    out_root = Path(args.out_root)
    if out_root.exists() or out_root.is_symlink():
        raise SystemExit(f"[refuse] output already exists: {out_root}")
    if args.folds != 8:
        raise SystemExit("[refuse] this control is pre-registered for exactly 8 folds")

    revision_record = json.loads(
        Path(args.revision_record).read_text(encoding="utf-8")
    )["calibration_composition_datasets"]
    if set(revision_record) != {source["name"] for source in SOURCES}:
        raise SystemExit("[refuse] composition dataset revision record has the wrong sources")
    for source in SOURCES:
        pinned = revision_record[source["name"]]
        expected = {
            "repository": source["dataset_id"],
            "config": source["config"],
            "split": source["split"],
            "revision": source["builder_fingerprint"],
        }
        if pinned != expected:
            raise SystemExit(
                f"[refuse] composition revision mismatch for {source['name']}: "
                f"{pinned!r} != {expected!r}"
            )

    loaded = {}
    source_provenance = {}
    selected_indices = {}
    for source in SOURCES:
        directory = source_directory(cache_root, source)
        paths = [directory / filename for filename in source["files"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit("[refuse] missing cached Arrow shard(s):\n" + "\n".join(missing))
        for path, expected_sha256 in zip(paths, source["file_sha256"]):
            observed_sha256 = sha256(path)
            if observed_sha256 != expected_sha256:
                raise SystemExit(
                    f"[refuse] Arrow shard hash mismatch for {source['name']}/{path.name}: "
                    f"{observed_sha256} != {expected_sha256}"
                )
        shards = [Dataset.from_file(str(path)) for path in paths]
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        if dataset.column_names != ["image", "label"]:
            raise SystemExit(f"[refuse] unexpected schema for {source['name']}: {dataset.column_names}")
        count = source["count"]
        rng = stable_rng(args.seed, source["name"])
        indices = rng.sample(range(len(dataset)), count)
        loaded[source["name"]] = dataset
        selected_indices[source["name"]] = indices
        source_provenance[source["name"]] = {
            "dataset_id": source["dataset_id"],
            "config": source["config"],
            "split": source["split"],
            "builder_fingerprint": source["builder_fingerprint"],
            "arrow_shards": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in paths
            ],
            "split_rows": len(dataset),
            "selected_without_replacement": count,
            "license_note": source["license"],
            "features": {key: repr(value) for key, value in dataset.features.items()},
        }

    out_root.mkdir(parents=True)
    records = []
    fold_sizes = Counter()
    ordinal = 0
    max_count = max(source["count"] for source in SOURCES)
    for draw_index in range(max_count):
        for source in SOURCES:
            name, count = source["name"], source["count"]
            if draw_index >= count:
                continue
            source_index = selected_indices[name][draw_index]
            row = loaded[name][source_index]
            image = row["image"].convert("RGB")
            label = int(row["label"])
            image_name = f"draw_{draw_index:03d}__index_{source_index:06d}__label_{label}.png"
            image_path = out_root / "images" / name / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="PNG", optimize=False, compress_level=6)
            fold = ordinal % args.folds
            fold_path = out_root / "folds" / f"f{fold}" / name / image_name
            relative_symlink(image_path, fold_path)
            records.append({
                "ordinal": ordinal,
                "source": name,
                "draw_index": draw_index,
                "source_index": source_index,
                "label": label,
                "image_file": image_name,
                "image_sha256": sha256(image_path),
                "fold": f"f{fold}",
            })
            fold_sizes[f"f{fold}"] += 1
            ordinal += 1

    assert ordinal == 1001
    manifest = {
        "schema_version": 1,
        "experiment": "alternative_three_source_calibration_composition_control",
        "method": "source-balanced deterministic simple random sample without replacement",
        "seed": args.seed,
        "rng": "Python random.Random (MT19937), SHA256-derived source-specific seeds",
        "source_order": [source["name"] for source in SOURCES],
        "source_counts": {source["name"]: source["count"] for source in SOURCES},
        "n_items": ordinal,
        "n_folds": args.folds,
        "fold_rule": "round-robin available sources by draw_index; fold=ordinal mod 8",
        "fold_sizes": dict(sorted(fold_sizes.items())),
        "software": {"datasets": datasets.__version__, "pillow": pillow_version},
        "redistribution": "No source images are intended for artifact redistribution.",
        "sources": source_provenance,
        "records": records,
    }
    manifest_path = out_root / "composition_map.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    forbidden = ("/" + "private" + "/", "/" + "Users" + "/", "scratch" + "pad")
    if any(marker in manifest_path.read_text(encoding="utf-8") for marker in forbidden):
        raise AssertionError("manifest is not path-sanitized")
    print(f"[composition] wrote {ordinal} images; folds={dict(sorted(fold_sizes.items()))}")
    print(f"[composition] source counts={manifest['source_counts']}")
    print(f"[composition] wrote {out_root}")


if __name__ == "__main__":
    main()
