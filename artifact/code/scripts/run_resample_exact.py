#!/usr/bin/env python3
"""Run exact_overlap from pinned local checkpoints for one calibration refit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--revision-record", required=True)
    parser.add_argument("--hf-home", required=True,
                        help="Hugging Face cache root containing hub/models--... snapshots")
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    code_root = Path(args.code_root).resolve()
    hf_home = Path(args.hf_home).resolve()
    seed_root = Path(args.seed_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"[refuse] output already exists: {out}")

    revisions = json.loads(Path(args.revision_record).read_text(encoding="utf-8"))
    specialist = revisions["full_finetuning_specialists"]["clip-vit-base-patch16"]
    experts: list[str] = []
    for task, revision in specialist["revision_by_task"].items():
        repo_id = specialist["repository_template"].format(task=task)
        spec = f"{repo_id}@{revision}"
        cache_name = "models--" + repo_id.replace("/", "--")
        snapshot = hf_home / "hub" / cache_name / "snapshots" / revision
        if not (snapshot / "model.safetensors").is_file():
            raise SystemExit(f"[refuse] missing pinned checkpoint for {task}: {spec}")
        experts.append(f"{task}={snapshot}")

    fold_root = seed_root / "fold_cov"
    required = [seed_root / "cov/manifest.json"] + [
        fold_root / f"f{i}/manifest.json" for i in range(8)
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("[refuse] missing covariance inputs:\n" + "\n".join(missing))

    command = [
        sys.executable,
        str(code_root / "scripts/exact_overlap.py"),
        "--base", str(
            hf_home / "hub/models--openai--clip-vit-base-patch16/snapshots"
            / revisions["base_models"]["openai/clip-vit-base-patch16"]
            ["task_vector_weights_revision"]
        ),
        "--experts", *experts,
        "--cov", str(seed_root / "cov"),
        "--fold_cov", str(fold_root),
        "--k", "8",
        "--null_m", "64",
        "--block_z", "2",
        "--key_prefix", "vision_model.",
        "--device", "cpu",
        "--out", str(out),
    ]
    subprocess.run(command, cwd=code_root, check=True)


if __name__ == "__main__":
    main()
