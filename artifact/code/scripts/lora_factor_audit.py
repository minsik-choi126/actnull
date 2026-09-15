#!/usr/bin/env python3
"""Reproduce the PRISM LoRA factor audit on all 24 targeted modules.

The primary audit uses the eight pinned ViT-B/16 adapter repositories, all 28
unordered task pairs, and q_proj/v_proj at all 12 encoder layers (24 modules).
The historical six-module selection at layers 2, 6, and 10 is retained only as
an explicitly labelled secondary provenance diagnostic.  For top-k orthonormal
bases Q_i and Q_j the script reports

    overlap(Q_i, Q_j) = ||Q_i^T Q_j||_F^2 / k

for the right singular subspace of A, the left singular subspace of B, and the
right singular subspace of BA.  BA is evaluated through an exact compact
factorization, so no dense 768 x 768 product or randomized SVD is needed.

This script reads already-materialized Hugging Face snapshots under ``--hf-hub``;
it does not download or modify checkpoints.  Its JSON output records public
repository@revision labels and content hashes, but never local filesystem paths.
The two optional released product CSVs provide an independent row-by-row check of
the BA column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


TASKS = (
    "dtd",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "stanford-cars",
    "sun397",
    "svhn",
)
PRIMARY_LAYERS = tuple(range(12))
LEGACY_LAYERS = (2, 6, 10)
PROJECTIONS = ("q_proj", "v_proj")


def modules_for(layers: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        f"encoder.layers.{layer}.self_attn.{projection}"
        for layer in layers
        for projection in PROJECTIONS
    )


PRIMARY_MODULES = modules_for(PRIMARY_LAYERS)
LEGACY_MODULES = modules_for(LEGACY_LAYERS)
METRICS = ("A_right", "B_left", "BA_right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_cache_directory(hub: Path, repository: str) -> Path:
    owner, name = repository.split("/", maxsplit=1)
    return hub / f"models--{owner}--{name}"


def factor_suffixes(linearized: bool) -> tuple[str, str]:
    if linearized:
        return ".lora_A.default.weight", ".lora_B.default.weight"
    return ".lora_A.weight", ".lora_B.weight"


def normalize_module(a_key: str, a_suffix: str) -> str:
    stem = a_key[: -len(a_suffix)]
    prefix = "base_model.model."
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if stem.endswith(".model"):
        stem = stem[: -len(".model")]
    return stem


def load_selected_factors(
    path: Path, *, linearized: bool
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    a_suffix, b_suffix = factor_suffixes(linearized)
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for a_key in sorted(key for key in keys if key.endswith(a_suffix)):
            module = normalize_module(a_key, a_suffix)
            if module not in PRIMARY_MODULES:
                continue
            b_key = a_key[: -len(a_suffix)] + b_suffix
            if b_key not in keys:
                raise KeyError(f"missing B factor paired with {a_key}")
            adapter_a = handle.get_tensor(a_key).double()
            adapter_b = handle.get_tensor(b_key).double()
            if adapter_a.ndim != 2 or adapter_b.ndim != 2:
                raise ValueError(f"LoRA factors must be matrices: {module}")
            if adapter_b.shape[1] != adapter_a.shape[0]:
                raise ValueError(
                    f"incompatible factors for {module}: "
                    f"A={tuple(adapter_a.shape)}, B={tuple(adapter_b.shape)}"
                )
            factors[module] = (adapter_a, adapter_b)
    missing = sorted(set(PRIMARY_MODULES) - set(factors))
    extra = sorted(set(factors) - set(PRIMARY_MODULES))
    if missing or extra:
        raise ValueError(f"selected-module mismatch: missing={missing}, extra={extra}")
    return factors


def right_basis(matrix: torch.Tensor, k: int) -> torch.Tensor:
    if k > min(matrix.shape):
        raise ValueError(f"k={k} exceeds matrix shape {tuple(matrix.shape)}")
    return torch.linalg.svd(matrix, full_matrices=False).Vh[:k].T.contiguous()


def left_basis(matrix: torch.Tensor, k: int) -> torch.Tensor:
    if k > min(matrix.shape):
        raise ValueError(f"k={k} exceeds matrix shape {tuple(matrix.shape)}")
    return torch.linalg.svd(matrix, full_matrices=False).U[:, :k].contiguous()


def product_right_basis(
    adapter_a: torch.Tensor, adapter_b: torch.Tensor, k: int
) -> torch.Tensor:
    """Return BA's top-k right basis without materializing the dense product.

    With A^T = QR, BA = (B R^T) Q^T.  If B R^T = U S V^T, the right
    singular basis of BA is therefore QV.  This is algebraically identical to a
    dense SVD and is substantially faster for rank-16 factors.
    """

    frame, triangular = torch.linalg.qr(adapter_a.T, mode="reduced")
    compact_product = adapter_b @ triangular.T
    compact_right = torch.linalg.svd(compact_product, full_matrices=False).Vh[:k].T
    return (frame @ compact_right).contiguous()


def overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError(f"basis shape mismatch: {first.shape} != {second.shape}")
    k = first.shape[1]
    value = float((first.T @ second).square().sum() / k)
    tolerance = 1e-10
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"subspace overlap outside [0,1]: {value}")
    return min(1.0, max(0.0, value))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return math.fsum(values) / len(values)


def summarize(values: list[float]) -> dict[str, object]:
    average = mean(values)
    return {
        "n": len(values),
        "mean": average,
        "paper_rounded_4": f"{average:.4f}",
        "minimum": min(values),
        "maximum": max(values),
    }


def audit_shared_a(
    factors: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]],
    modules: tuple[str, ...],
) -> dict[str, object]:
    identical = True
    maximum_difference = 0.0
    by_module: dict[str, object] = {}
    for module in modules:
        module_identical = True
        module_maximum = 0.0
        for first, second in itertools.combinations(TASKS, 2):
            a_first = factors[first][module][0]
            a_second = factors[second][module][0]
            module_identical = module_identical and torch.equal(a_first, a_second)
            module_maximum = max(
                module_maximum, float((a_first - a_second).abs().max())
            )
        identical = identical and module_identical
        maximum_difference = max(maximum_difference, module_maximum)
        by_module[module] = {
            "bit_identical_across_tasks": module_identical,
            "maximum_absolute_element_difference": module_maximum,
        }
    return {
        "bit_identical_across_all_tasks_and_modules": identical,
        "maximum_absolute_element_difference": maximum_difference,
        "by_module": by_module,
    }


def product_csv_crosscheck(
    path: Path,
    product_cells: dict[tuple[str, str, str], float],
) -> dict[str, object]:
    order = {task: index for index, task in enumerate(TASKS)}
    selected_modules = {key[2] for key in product_cells}
    released: dict[tuple[str, str, str], float] = {}
    total_rows = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            if row["module"] not in selected_modules:
                continue
            if row["a"] not in order or row["b"] not in order:
                continue
            first, second = sorted((row["a"], row["b"]), key=order.__getitem__)
            key = (first, second, row["module"])
            if key in released:
                raise ValueError(f"duplicate selected row in product CSV: {key}")
            released[key] = float(row["raw"])
    if set(released) != set(product_cells):
        missing = sorted(set(product_cells) - set(released))
        extra = sorted(set(released) - set(product_cells))
        raise ValueError(
            f"product CSV cell mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    differences = [abs(product_cells[key] - released[key]) for key in released]
    released_mean = mean(released.values())
    factor_mean = mean(product_cells.values())
    return {
        "file": path.name,
        "sha256": sha256(path),
        "total_rows": total_rows,
        "selected_rows": len(released),
        "released_raw_mean": released_mean,
        "released_raw_paper_rounded_4": f"{released_mean:.4f}",
        "factor_recomputation_mean": factor_mean,
        "maximum_absolute_cell_difference": max(differences),
        "mean_absolute_cell_difference": mean(differences),
        "paper_rounding_matches": f"{released_mean:.4f}" == f"{factor_mean:.4f}",
    }


def audit_family(
    *,
    hub: Path,
    pins: dict[str, object],
    family: str,
    repository_suffix: str,
    checkpoint_filename: str,
    linearized: bool,
    k: int,
    product_csv: Path | None,
) -> dict[str, object]:
    revisions = pins["adapters"][family]["revision_by_task"]
    if set(revisions) != set(TASKS):
        raise ValueError(
            f"{family} task pin mismatch: expected={list(TASKS)}, "
            f"observed={sorted(revisions)}"
        )

    factors: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    checkpoints = []
    for task in TASKS:
        revision = revisions[task]
        repository = f"tanganke/clip-vit-base-patch16_{task}_{repository_suffix}"
        snapshot = repository_cache_directory(hub, repository) / "snapshots" / revision
        checkpoint = snapshot / checkpoint_filename
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"required pinned checkpoint is not materialized: "
                f"{repository}@{revision}/{checkpoint_filename}"
            )
        factors[task] = load_selected_factors(checkpoint, linearized=linearized)
        checkpoints.append(
            {
                "task": task,
                "repository_at_revision": f"{repository}@{revision}",
                "file": checkpoint_filename,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            }
        )

    factor_shapes = sorted(
        {
            (tuple(adapter_a.shape), tuple(adapter_b.shape))
            for task in TASKS
            for adapter_a, adapter_b in factors[task].values()
        }
    )
    if len(factor_shapes) != 1:
        raise ValueError(f"inconsistent factor shapes: {factor_shapes}")

    bases: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for task in TASKS:
        bases[task] = {}
        for module in PRIMARY_MODULES:
            adapter_a, adapter_b = factors[task][module]
            bases[task][module] = (
                right_basis(adapter_a, k),
                left_basis(adapter_b, k),
                product_right_basis(adapter_a, adapter_b, k),
            )

    by_module = {
        module: {metric: [] for metric in METRICS}
        for module in PRIMARY_MODULES
    }
    product_cells: dict[tuple[str, str, str], float] = {}
    for first, second in itertools.combinations(TASKS, 2):
        for module in PRIMARY_MODULES:
            cell_values = tuple(
                overlap(bases[first][module][index], bases[second][module][index])
                for index in range(3)
            )
            for metric, value in zip(METRICS, cell_values):
                by_module[module][metric].append(value)
            product_cells[(first, second, module)] = cell_values[2]

    def scope_aggregate(modules: tuple[str, ...]) -> dict[str, object]:
        return {
            metric: summarize(
                [value for module in modules for value in by_module[module][metric]]
            )
            for metric in METRICS
        }

    legacy_product_cells = {
        key: value for key, value in product_cells.items() if key[2] in LEGACY_MODULES
    }

    payload: dict[str, object] = {
        "family": family,
        "checkpoint_filename": checkpoint_filename,
        "factor_shapes": {
            "A": list(factor_shapes[0][0]),
            "B": list(factor_shapes[0][1]),
        },
        "checkpoints": checkpoints,
        "primary_all_24_modules": {
            "A_identity_audit": audit_shared_a(factors, PRIMARY_MODULES),
            "aggregate": scope_aggregate(PRIMARY_MODULES),
        },
        "secondary_legacy_six_modules": {
            "status": (
                "historical layers 2/6/10 diagnostic retained for provenance; "
                "not the primary factor-audit estimand"
            ),
            "modules": list(LEGACY_MODULES),
            "A_identity_audit": audit_shared_a(factors, LEGACY_MODULES),
            "aggregate": scope_aggregate(LEGACY_MODULES),
        },
        "by_module": {
            module: {
                metric: summarize(metric_values)
                for metric, metric_values in module_values.items()
            }
            for module, module_values in by_module.items()
        },
    }
    if product_csv is not None:
        payload["primary_all_24_modules"]["released_product_csv_crosscheck"] = (
            product_csv_crosscheck(product_csv, product_cells)
        )
        payload["secondary_legacy_six_modules"]["released_product_csv_crosscheck"] = (
            product_csv_crosscheck(product_csv, legacy_product_cells)
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-record", type=Path, required=True)
    parser.add_argument("--hf-hub", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-product-csv", type=Path)
    parser.add_argument("--linearized-product-csv", type=Path)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    pins = json.loads(args.pin_record.read_text())
    script_sha256 = sha256(Path(__file__))
    families = [
        audit_family(
            hub=args.hf_hub,
            pins=pins,
            family="rank16_lora",
            repository_suffix="lora-16",
            checkpoint_filename="adapter_model.safetensors",
            linearized=False,
            k=args.k,
            product_csv=args.lora_product_csv,
        ),
        audit_family(
            hub=args.hf_hub,
            pins=pins,
            family="rank16_linearized_lora",
            repository_suffix="l-lora-16",
            checkpoint_filename="linearized_adapter_model.safetensors",
            linearized=True,
            k=args.k,
            product_csv=args.linearized_product_csv,
        ),
    ]
    payload = {
        "schema_version": 2,
        "implementation_sha256": script_sha256,
        "input_pin_record": {
            "file": args.pin_record.name,
            "sha256": sha256(args.pin_record),
        },
        "scope": {
            "backbone": "openai/clip-vit-base-patch16",
            "tasks": list(TASKS),
            "unordered_task_pairs": math.comb(len(TASKS), 2),
            "k": args.k,
            "ambient_dimension": 768,
            "primary_all_24_modules": {
                "layers": list(PRIMARY_LAYERS),
                "projections": list(PROJECTIONS),
                "modules": list(PRIMARY_MODULES),
                "modules_per_task": len(PRIMARY_MODULES),
                "cells_per_family": math.comb(len(TASKS), 2) * len(PRIMARY_MODULES),
            },
            "secondary_legacy_six_modules": {
                "status": "secondary provenance diagnostic, not primary",
                "layers": list(LEGACY_LAYERS),
                "projections": list(PROJECTIONS),
                "modules": list(LEGACY_MODULES),
                "modules_per_task": len(LEGACY_MODULES),
                "cells_per_family": math.comb(len(TASKS), 2) * len(LEGACY_MODULES),
            },
        },
        "definitions": {
            "overlap": "||Q_i^T Q_j||_F^2 / k for orthonormal top-k bases",
            "A_right": "top-k right singular subspace of the stored LoRA A factor",
            "B_left": "top-k left singular subspace of the stored LoRA B factor",
            "BA_right": "top-k right singular subspace of BA; invariant to the LoRA scalar",
            "BA_compact_algorithm": "A^T=QR; SVD(BR^T)=USV^T; BA right basis=QV",
            "primary_aggregation": (
                "unweighted arithmetic mean over 28 task pairs x 24 modules"
            ),
            "secondary_legacy_aggregation": (
                "unweighted arithmetic mean over 28 task pairs x the historical "
                "six-module layers-2/6/10 subset"
            ),
            "isotropic_expectation": args.k / 768,
        },
        "numeric_environment": {
            "dtype": "float64",
            "device": "cpu",
            "torch": torch.__version__,
            "safetensors": version("safetensors"),
            "torch_threads": args.torch_threads,
        },
        "families": families,
        "path_policy": (
            "No local paths are recorded; inputs are identified by public "
            "repository@revision labels, basenames, byte sizes, and SHA-256 hashes."
        ),
    }
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
