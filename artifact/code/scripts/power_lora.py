#!/usr/bin/env python3
"""Positive-control power study for the shared-initialisation LoRA null."""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actnull.null as act
from actnull.experiment import (
    parse_csv_ints,
    parse_csv_strings,
    parse_named_paths,
    public_sources,
    write_json_new,
)
from actnull.positive_control_utils import (
    initial_lora_shared_init_null,
    materialize_in_adapter_row_space,
    plant_right_directions,
)


DEFAULT_MODULES = (
    "encoder.layers.4.self_attn.q_proj",
    "encoder.layers.4.self_attn.v_proj",
    "encoder.layers.8.self_attn.q_proj",
    "encoder.layers.8.self_attn.v_proj",
)


def normalized_version(value: str) -> str:
    return "initial" if value == "shipped" else value


def run(
    *,
    base_path: str,
    experts: dict[str, str],
    modules: tuple[str, ...],
    null_version: str = "current",
    planted_values: tuple[int, ...] = (0, 1, 2, 4),
    k: int = 8,
    draws: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[int, tuple[float, float]]:
    if 0 not in planted_values:
        raise ValueError("planted values must include 0 for level centering")
    if any(value < 0 or value > k for value in planted_values):
        raise ValueError(f"planted values must lie in [0, k={k}]")
    if draws < 1:
        raise ValueError("draws must be positive")
    version = normalized_version(null_version)
    if version not in {"current", "initial"}:
        raise ValueError(f"unknown null version {null_version!r}")
    test_null = (
        initial_lora_shared_init_null
        if version == "initial"
        else act.lora_shared_init_null
    )

    base = act.LazyCheckpoint(base_path, "vision_model.")
    checkpoints = {name: act.AdapterCheckpoint(path) for name, path in experts.items()}
    common_targets = set.intersection(*(checkpoint.targets for checkpoint in checkpoints.values()))
    missing = [module for module in modules if module not in common_targets]
    if missing:
        raise ValueError(f"modules not targeted by every adapter: {missing}")
    names = sorted(checkpoints)

    output = {}
    for planted in planted_values:
        raw_values: list[float] = []
        null_values: list[float] = []
        for module in modules:
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
                if float(reference.norm()) == 0:
                    continue
                right_data = checkpoints[right].mods[module]
                adapter_a = right_data["lora_A"].to(device=device, dtype=torch.float32)
                partner = act.lora_shared_init_null(
                    adapter_a,
                    right_data["lora_B"].to(device=device, dtype=torch.float32),
                    checkpoints[right].scale,
                    partner_generator,
                )
                proposed = plant_right_directions(
                    reference, partner, planted, k, plant_generator
                )
                # A planted right direction outside row(A_b) is not implementable by
                # adapter b. Score the matrix the adapter can actually materialise.
                materialized, materialized_b = materialize_in_adapter_row_space(
                    proposed, adapter_a, checkpoints[right].scale
                )
                raw_values.append(
                    act.overlap(act.top_right(reference, k), act.top_right(materialized, k))
                )
                left_data = checkpoints[left].mods[module]
                left_null = test_null(
                    left_data["lora_A"].to(device=device, dtype=torch.float32),
                    left_data["lora_B"].to(device=device, dtype=torch.float32),
                    checkpoints[left].scale,
                    null_generator,
                )
                right_null = test_null(
                    adapter_a, materialized_b, checkpoints[right].scale, null_generator
                )
                null_values.append(
                    act.overlap(act.top_right(left_null, k), act.top_right(right_null, k))
                )
        if not raw_values or not null_values:
            raise RuntimeError(f"no live LoRA positive-control cells for planted={planted}")
        output[planted] = (st.mean(raw_values), st.mean(null_values))
    return output


def result_rows(
    results: dict[int, tuple[float, float]], k: int
) -> list[dict[str, float | int | None]]:
    base_raw, base_null = results[0]
    base_excess = base_raw - base_null
    rows = []
    for planted, (raw, null_level) in results.items():
        truth = (planted / k) * (1 - base_raw)
        excess = raw - null_level
        delivered_overlap = raw - base_raw
        if planted == 0:
            delivery = recovery = leak = None
        else:
            delivery = 100 * delivered_overlap / truth if truth else None
            if abs(delivered_overlap) <= 1e-15:
                recovery = leak = None
            else:
                recovery = 100 * (excess - base_excess) / delivered_overlap
                leak = 100 * (null_level - base_null) / delivered_overlap
        rows.append(
            {
                "planted": planted,
                "k": k,
                "O0": base_raw,
                "truth": truth,
                "raw": raw,
                "null": null_level,
                "excess": excess,
                "delivered_overlap": None if planted == 0 else delivered_overlap,
                "delivered": delivery,
                "recovery": recovery,
                "leak": leak,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="clip-vit-base-patch16")
    parser.add_argument("--base", default=os.environ.get("ACTNULL_BASE"))
    parser.add_argument("--experts", nargs="+", required=True,
                        help="NAME=adapter-repo-or-local-snapshot ...")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--null-version", choices=("current", "initial", "shipped"),
                        default=os.environ.get("NULL_VERSION", "current"))
    parser.add_argument("--planted", default="0,1,2,4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--draws", type=int, default=12)
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
        results = run(
            base_path=args.base, experts=experts, modules=modules,
            null_version=args.null_version, planted_values=planted_values,
            k=args.k, draws=args.draws, seed=args.seed, device=args.device,
        )
        rows = result_rows(results, args.k)
        target = write_json_new(
            args.out,
            {
                "arch": args.arch,
                "null_version": normalized_version(args.null_version),
                "modules": list(modules),
                "tasks": public_sources(experts),
                "k": args.k,
                "draws": args.draws,
                "seed": args.seed,
                "rows": rows,
            },
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print(f"  baseline overlap O_0 = {results[0][0]:.4f}")
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
