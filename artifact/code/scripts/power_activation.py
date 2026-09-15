#!/usr/bin/env python3
"""Positive-control power study for the activation-conditioned null.

A current-null draw supplies a zero-coupling partner. We replace its leading
right singular directions with ``s`` directions from a real reference update,
then ask the current or initial null how much of the delivered overlap it
removes. Both comparison arms see exactly the same planted pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actnull.null as act
from actnull.experiment import (
    covariance_directories,
    fold_eigenvalues,
    parse_csv_ints,
    parse_csv_strings,
    parse_named_paths,
    public_sources,
    write_json_new,
)
from actnull.positive_control_utils import (
    initial_coupling_destroying_null,
    plant_right_directions,
)


DEFAULT_MODULES = (
    "encoder.layers.4.self_attn.q_proj",
    "encoder.layers.4.mlp.fc1",
    "encoder.layers.8.self_attn.v_proj",
    "encoder.layers.8.mlp.fc2",
)


def normalized_version(value: str) -> str:
    return "initial" if value == "shipped" else value


def study(
    *,
    covariance: Path,
    fold_covariance: Path,
    base_path: str,
    experts: dict[str, str],
    modules: tuple[str, ...],
    null_version: str = "current",
    planted_values: tuple[int, ...] = (0, 1, 2, 4),
    k: int = 8,
    draws: int = 10,
    null_draws: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict[int, tuple[float, float, float, float, float]]:
    """Return raw/null/isotropic/excess/planted fraction for each planted count."""
    if 0 not in planted_values:
        raise ValueError("planted values must include 0 for level centering")
    if any(value < 0 or value > k for value in planted_values):
        raise ValueError(f"planted values must lie in [0, k={k}]")
    if draws < 1 or null_draws < 1:
        raise ValueError("draws and null_draws must be positive")

    version = normalized_version(null_version)
    if version not in {"current", "initial"}:
        raise ValueError(f"unknown null version {null_version!r}")
    test_null = (
        initial_coupling_destroying_null
        if version == "initial"
        else act.coupling_destroying_null
    )
    manifest = json.loads((covariance / "manifest.json").read_text())
    missing = [module for module in modules if module not in manifest["modules"]]
    if missing:
        raise ValueError(f"modules absent from covariance manifest: {missing}")

    base = act.LazyCheckpoint(base_path, "vision_model.")
    checkpoints = {
        name: act.LazyCheckpoint(path, "vision_model.") for name, path in experts.items()
    }
    names = sorted(checkpoints)
    output = {}
    for planted in planted_values:
        cells: list[tuple[float, float, int]] = []
        for module in modules:
            blob = torch.load(
                covariance / f"{module.replace('.', '__')}.pt",
                map_location=device,
                weights_only=False,
            )
            activation_basis = blob["eigvecs"].to(device)
            eigenvalues = blob["eigvals"].to(device)
            folds = fold_eigenvalues(fold_covariance, module)
            length = min(len(values) for values in folds)
            standard_error = (
                torch.stack([values[:length] for values in folds]).std(0)
                / len(folds) ** 0.5
            ).float().to(device)
            blocks = act.resolvable_blocks(eigenvalues[:length], standard_error, 2.0)
            key = module + ".weight"
            null_generator = torch.Generator(device=device).manual_seed(seed)
            partner_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
            plant_generator = torch.Generator(device=device).manual_seed(seed + 20_000)
            for draw in range(draws):
                left = names[draw % len(names)]
                right = names[(draw + 1 + draw // len(names)) % len(names)]
                if left == right:
                    continue
                reference = act.load_delta(base, checkpoints[left], key, device)
                candidate = act.load_delta(base, checkpoints[right], key, device)
                if float(reference.norm()) == 0 or float(candidate.norm()) == 0:
                    continue
                partner = act.coupling_destroying_null(
                    candidate, activation_basis, 0, partner_generator, 128, blocks=blocks
                )
                planted_matrix = plant_right_directions(
                    reference, partner, planted, k, plant_generator
                )
                raw = act.overlap(act.top_right(reference, k), act.top_right(planted_matrix, k))
                null_level = st.mean(
                    act.overlap(
                        act.top_right(
                            test_null(
                                reference, activation_basis, 0, null_generator, 128,
                                blocks=blocks,
                            ),
                            k,
                        ),
                        act.top_right(
                            test_null(
                                planted_matrix, activation_basis, 0, null_generator, 128,
                                blocks=blocks,
                            ),
                            k,
                        ),
                    )
                    for _ in range(null_draws)
                )
                cells.append((raw, null_level, int(blob["d"])))
        if not cells:
            raise RuntimeError(f"no live positive-control cells for planted={planted}")
        raw = st.mean(value for value, _, _ in cells)
        null_level = st.mean(value for _, value, _ in cells)
        isotropic = st.mean(k / dimension for _, _, dimension in cells)
        output[planted] = (
            raw, null_level, isotropic, raw - null_level, planted / k
        )
    return output


def result_rows(
    results: dict[int, tuple[float, float, float, float, float]], k: int
) -> list[dict[str, float | int | None]]:
    base_raw, base_null = results[0][0], results[0][1]
    base_excess = base_raw - base_null
    rows = []
    for planted, (raw, null_level, isotropic, excess, _planted_fraction) in results.items():
        # Replacing s of k directions can only add overlap not already present at
        # s=0. The nominal increment is therefore (s/k)(1-O0), exactly as in the
        # LoRA positive control, rather than the unattainable s/k ceiling.
        truth = (planted / k) * (1 - base_raw)
        delivered_overlap = raw - base_raw
        if truth == 0:
            delivery = recovery = leak = None
        else:
            delivery = 100 * delivered_overlap / truth
            if abs(delivered_overlap) <= 1e-15:
                recovery = leak = None
            else:
                recovery = 100 * (excess - base_excess) / delivered_overlap
                leak = 100 * (null_level - base_null) / delivered_overlap
        rows.append(
            {
                "planted": planted,
                "k": k,
                "truth": truth,
                "raw": raw,
                "null": null_level,
                "iso": isotropic,
                "excess": excess,
                "delivered_overlap": None if truth == 0 else delivered_overlap,
                "delivered": delivery,
                "recovery": recovery,
                "leak": leak,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="clip-vit-base-patch16")
    parser.add_argument("--work-root", default=os.environ.get("ACTNULL_WORK"))
    parser.add_argument("--cov")
    parser.add_argument("--fold-cov")
    parser.add_argument("--base", default=os.environ.get("ACTNULL_BASE"))
    parser.add_argument("--experts", nargs="+", required=True, help="NAME=PATH ...")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--null-version", choices=("current", "initial", "shipped"),
                        default=os.environ.get("NULL_VERSION", "current"))
    parser.add_argument("--planted", default="0,1,2,4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--null-draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True,
                        help="new JSON path; existing files are never overwritten")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.base:
        parser.error("--base is required (or set ACTNULL_BASE)")
    try:
        experts = parse_named_paths(args.experts)
        modules = parse_csv_strings(args.modules, DEFAULT_MODULES)
        planted_values = parse_csv_ints(args.planted)
        covariance, folds = covariance_directories(
            arch=args.arch, work_root=args.work_root,
            covariance=args.cov, fold_covariance=args.fold_cov,
        )
        results = study(
            covariance=covariance, fold_covariance=folds, base_path=args.base,
            experts=experts, modules=modules, null_version=args.null_version,
            planted_values=planted_values, k=args.k, draws=args.draws,
            null_draws=args.null_draws, seed=args.seed, device=args.device,
        )
        rows = result_rows(results, args.k)
        target = write_json_new(
            args.out,
            {
                "arch": args.arch,
                "null_version": normalized_version(args.null_version),
                "modules": list(modules),
                "experts": public_sources(experts),
                "k": args.k,
                "draws": args.draws,
                "null_draws": args.null_draws,
                "seed": args.seed,
                "rows": rows,
            },
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print(f"  {'planted':>9s} {'nominal':>9s} {'raw':>8s} {'null':>8s} "
          f"{'excess':>9s} {'recovery':>9s} {'delivery':>9s}")
    for row in rows:
        recovery = "-" if row["recovery"] is None else f"{row['recovery']:.1f}%"
        delivery = "-" if row["delivered"] is None else f"{row['delivered']:.1f}%"
        print(f"  {row['planted']:5d}/{args.k:<3d} {row['truth']:9.4f} {row['raw']:8.4f} "
              f"{row['null']:8.4f} {row['excess']:+9.4f} {recovery:>9s} {delivery:>9s}")
    print(f"  wrote {target}")


if __name__ == "__main__":
    main()
