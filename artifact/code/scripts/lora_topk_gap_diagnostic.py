#!/usr/bin/env python3
"""Audit relative singular-value gaps at the top-k boundary for PRISM LoRA arms.

This script reads the public repository revisions in the artifact pin record and
locates already-materialized Hugging Face snapshots under a supplied hub root.
It does not download or modify checkpoints.  For each task/module cell it reports

    gap = (s_k - s_{k+1}) / s_k

for A and BA.  It also redraws the factor-conditioned B spectrum with Haar W and
reports the resulting BA gaps.  The output contains public repository@revision
labels but never local filesystem paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
THRESHOLDS = (1e-4, 1e-3, 1e-2, 5e-2)


def haar_orthogonal(size: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    frame, triangular = torch.linalg.qr(matrix)
    phase = torch.where(torch.diag(triangular) >= 0, 1.0, -1.0)
    return frame * phase.unsqueeze(0)


def relative_gap_from_squared_singular_values(values: torch.Tensor, k: int) -> float:
    values = torch.sort(torch.clamp(values, min=0), descending=True).values
    singular = torch.sqrt(values)
    if singular.numel() <= k or float(singular[k - 1]) == 0.0:
        raise ValueError(f"cannot evaluate a rank-{k} boundary from {singular.numel()} values")
    return float((singular[k - 1] - singular[k]) / singular[k - 1])


def positive_semidefinite_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * torch.clamp(values, min=0).sqrt()) @ vectors.T


def summarize(values: list[float]) -> dict[str, object]:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(tensor, torch.tensor(QUANTILES, dtype=torch.float64))
    return {
        "n": len(values),
        "quantiles": {
            f"{probability:g}": float(value)
            for probability, value in zip(QUANTILES, quantiles)
        },
        "threshold_counts": {
            f"lt_{threshold:g}": int((tensor < threshold).sum())
            for threshold in THRESHOLDS
        },
    }


def adapter_file(snapshot: Path, linearized: bool) -> Path:
    filename = "linearized_adapter_model.safetensors" if linearized else "adapter_model.safetensors"
    path = snapshot / filename
    if not path.is_file():
        raise FileNotFoundError(f"required pinned adapter file is not materialized: {filename}")
    return path


def factor_keys(keys: list[str], linearized: bool) -> list[tuple[str, str]]:
    if linearized:
        suffix_a, suffix_b = "lora_A.default.weight", "lora_B.default.weight"
    else:
        suffix_a, suffix_b = ".lora_A.weight", ".lora_B.weight"
    pairs = []
    for key in keys:
        if key.endswith(suffix_a):
            pairs.append((key, key[: -len(suffix_a)] + suffix_b))
    return pairs


def audit_family(
    *,
    hub: Path,
    pins: dict[str, object],
    family: str,
    repository_suffix: str,
    linearized: bool,
    k: int,
    seeds: tuple[int, ...],
    draws_per_seed: int,
) -> dict[str, object]:
    observed_a: list[float] = []
    observed_ba: list[float] = []
    cells: list[tuple[torch.Tensor, torch.Tensor]] = []
    snapshots = []
    revisions = pins["adapters"][family]["revision_by_task"]
    for task, revision in revisions.items():
        repository = f"tanganke/clip-vit-base-patch16_{task}_{repository_suffix}"
        snapshots.append(f"{repository}@{revision}")
        snapshot = (
            hub
            / repository.replace("/", "--").replace("tanganke--", "models--tanganke--")
            / "snapshots"
            / revision
        )
        path = adapter_file(snapshot, linearized)
        with safe_open(str(path), framework="pt") as handle:
            for a_key, b_key in factor_keys(list(handle.keys()), linearized):
                adapter_a = handle.get_tensor(a_key).double()
                adapter_b = handle.get_tensor(b_key).double()
                a_gram = adapter_a @ adapter_a.T
                b_gram = adapter_b.T @ adapter_b
                b_sqrt = positive_semidefinite_sqrt(b_gram)
                observed_a.append(
                    relative_gap_from_squared_singular_values(torch.linalg.eigvalsh(a_gram), k)
                )
                observed_ba.append(
                    relative_gap_from_squared_singular_values(
                        torch.linalg.eigvalsh(b_sqrt @ a_gram @ b_sqrt), k
                    )
                )
                cells.append((a_gram, torch.linalg.svdvals(adapter_b)))

    factor_null_ba: list[float] = []
    factor_rank = cells[0][0].shape[0]
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        for a_gram, b_singular in cells:
            diagonal = torch.diag(b_singular)
            for _ in range(draws_per_seed):
                rotation = haar_orthogonal(factor_rank, generator)
                squared = diagonal @ rotation.T @ a_gram @ rotation @ diagonal
                factor_null_ba.append(
                    relative_gap_from_squared_singular_values(torch.linalg.eigvalsh(squared), k)
                )

    return {
        "family": family,
        "scope": {
            "tasks": len(revisions),
            "modules_per_task": len(cells) // len(revisions),
            "observed_cells": len(cells),
            "factor_rank": factor_rank,
            "snapshots": snapshots,
        },
        "observed_A": summarize(observed_a),
        "observed_BA": summarize(observed_ba),
        "factor_null_BA": summarize(factor_null_ba),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-record", type=Path, required=True)
    parser.add_argument("--hf-hub", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seeds", default="0,1000003,2000003,3000017")
    parser.add_argument("--draws-per-seed", type=int, default=16)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    pins = json.loads(args.pin_record.read_text())
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    families = [
        audit_family(
            hub=args.hf_hub,
            pins=pins,
            family="rank16_lora",
            repository_suffix="lora-16",
            linearized=False,
            k=args.k,
            seeds=seeds,
            draws_per_seed=args.draws_per_seed,
        ),
        audit_family(
            hub=args.hf_hub,
            pins=pins,
            family="rank16_linearized_lora",
            repository_suffix="l-lora-16",
            linearized=True,
            k=args.k,
            seeds=seeds,
            draws_per_seed=args.draws_per_seed,
        ),
    ]
    payload = {
        "schema_version": 1,
        "implementation_sha256": script_sha256,
        "gap_definition": "(s_k - s_{k+1}) / s_k using linear singular values",
        "k": args.k,
        "factor_null": {
            "seeds": list(seeds),
            "draws_per_seed_per_task_module_cell": args.draws_per_seed,
            "construction": "B'=U diag(S_B) W^T; U omitted because it cancels the BA singular spectrum and right geometry; W is sign-corrected Gaussian-QR Haar",
        },
        "thresholds": list(THRESHOLDS),
        "families": families,
        "path_policy": "No local paths are recorded; snapshot labels are public repository@revision strings.",
    }
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
