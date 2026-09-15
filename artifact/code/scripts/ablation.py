#!/usr/bin/env python3
"""Ablate singleton randomisation and full-complement randomisation."""
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
    parse_csv_strings,
    parse_named_paths,
    write_json_new,
)
from actnull.positive_control_utils import plant_right_directions


DEFAULT_MODULES = (
    "encoder.layers.4.self_attn.q_proj",
    "encoder.layers.4.mlp.fc1",
    "encoder.layers.8.self_attn.v_proj",
    "encoder.layers.8.mlp.fc2",
)
VARIANTS = (
    ("initial (neither correction)", False, False),
    ("singleton signs only", False, True),
    ("full complement only", True, False),
    ("corrected (both)", True, True),
)


def null_variant(
    dW: torch.Tensor,
    activation_basis: torch.Tensor,
    generator: torch.Generator,
    blocks: list[list[int]],
    full_complement: bool,
    flip_singletons: bool,
    tail_rank: int = 128,
) -> torch.Tensor:
    """Current null with either of its two substantive corrections disabled."""
    _, d_in = dW.shape
    m = activation_basis.shape[0]
    basis = activation_basis.to(dW)
    coordinates = dW @ basis.T
    residual = dW - coordinates @ basis
    for block in blocks:
        index = torch.tensor(block, device=dW.device)
        if len(block) < 2:
            if flip_singletons:
                sign = 1.0 - 2.0 * float(
                    torch.randint(0, 2, (1,), generator=generator, device=dW.device)
                )
                coordinates[:, index] = coordinates[:, index] * sign
            continue
        rotation = act.gaussian_haar_frame(
            torch.randn(
                len(block), len(block), generator=generator,
                device=dW.device, dtype=torch.float32,
            )
        )
        coordinates[:, index] = coordinates[:, index] @ rotation.to(dW)

    if full_complement:
        rank = int(min(residual.shape[0], max(d_in - m, 0)))
        if rank >= 2 and float(residual.norm()) > 0:
            try:
                left, singular, _ = torch.linalg.svd(residual.double(), full_matrices=False)
                factor = left[:, :rank] * singular[:rank]
            except torch._C._LinAlgError:
                gram = residual.double() @ residual.double().T
                eigenvalues, left = torch.linalg.eigh(gram)
                eigenvalues = eigenvalues.flip(0)[:rank].clamp(min=0.0)
                left = left.flip(1)[:, :rank]
                factor = left * eigenvalues.sqrt()
            frame = torch.randn(
                d_in, rank, generator=generator,
                device=dW.device, dtype=torch.float32,
            ).double()
            frame = frame - basis.T.double() @ (basis.double() @ frame)
            frame = act.gaussian_haar_frame(frame)
            residual = (factor @ frame.T).to(dW)
    else:
        width = min(tail_rank, max(d_in - m, 0))
        if width >= 2:
            frame = torch.randn(
                d_in, width, generator=generator,
                device=dW.device, dtype=torch.float32,
            )
            frame = frame - basis.T.float() @ (basis.float() @ frame)
            frame = act.gaussian_haar_frame(frame).to(dW)
            rotation = act.gaussian_haar_frame(
                torch.randn(
                    width, width, generator=generator,
                    device=dW.device, dtype=torch.float32,
                )
            ).to(dW)
            projected = residual @ frame
            residual = residual - projected @ frame.T + (projected @ rotation) @ frame.T
    return coordinates @ basis + residual


def study(
    *,
    covariance: Path,
    fold_covariance: Path,
    base_path: str,
    experts: dict[str, str],
    modules: tuple[str, ...],
    planted: int = 2,
    k: int = 8,
    draws: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, dict[str, list[float]]]:
    if planted < 0 or planted > k:
        raise ValueError(f"planted={planted} must lie in [0, k={k}]")
    if draws < 1:
        raise ValueError("draws must be positive")
    manifest = json.loads((covariance / "manifest.json").read_text())
    missing = [module for module in modules if module not in manifest["modules"]]
    if missing:
        raise ValueError(f"modules absent from covariance manifest: {missing}")
    base = act.LazyCheckpoint(base_path, "vision_model.")
    checkpoints = {
        name: act.LazyCheckpoint(path, "vision_model.") for name, path in experts.items()
    }
    names = sorted(checkpoints)
    accumulation = {label: {"raw": [], "null": []} for label, _, _ in VARIANTS}
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
        for label, full_complement, flip_singletons in VARIANTS:
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
                partner = null_variant(
                    candidate, activation_basis, partner_generator, blocks, True, True
                )
                planted_matrix = plant_right_directions(
                    reference, partner, planted, k, plant_generator
                )
                accumulation[label]["raw"].append(
                    act.overlap(act.top_right(reference, k), act.top_right(planted_matrix, k))
                )
                accumulation[label]["null"].append(
                    act.overlap(
                        act.top_right(
                            null_variant(
                                reference, activation_basis, null_generator, blocks,
                                full_complement, flip_singletons,
                            ),
                            k,
                        ),
                        act.top_right(
                            null_variant(
                                planted_matrix, activation_basis, null_generator, blocks,
                                full_complement, flip_singletons,
                            ),
                            k,
                        ),
                    )
                )
    for label in accumulation:
        if not accumulation[label]["raw"] or not accumulation[label]["null"]:
            raise RuntimeError(f"no live ablation cells for {label}")
    return accumulation


def study_level_centered(
    *, planted: int = 2, **study_arguments: object
) -> dict[int, dict[str, dict[str, list[float]]]]:
    """Run zero-signal and planted-signal levels with common RNG seeds.

    A recovery percentage is meaningful only after subtracting finite-sample
    excess at zero planted directions. Calling :func:`study` separately at the
    two levels resets every generator to the same seed and supplies the
    common-random-number pairing used by the released level-centred result.
    """
    k = int(study_arguments.get("k", 8))
    if planted < 1 or planted > k:
        raise ValueError(f"planted={planted} must lie in [1, k={k}]")
    return {
        level: study(planted=level, **study_arguments)
        for level in (0, planted)
    }


def summarize(
    accumulation: dict[int, dict[str, dict[str, list[float]]]],
    planted: int,
) -> list[dict[str, float | str]]:
    """Return the released two-level, level-centred ablation schema."""
    rows = []
    for label, _, _ in VARIANTS:
        raw0 = st.mean(accumulation[0][label]["raw"])
        null0 = st.mean(accumulation[0][label]["null"])
        raw_planted = st.mean(accumulation[planted][label]["raw"])
        null_planted = st.mean(accumulation[planted][label]["null"])
        excess0 = raw0 - null0
        excess_planted = raw_planted - null_planted
        delivered = raw_planted - raw0
        if delivered <= 0:
            raise RuntimeError(
                f"non-positive delivered overlap for {label}: {delivered:g}"
            )
        rows.append(
            {
                "variant": label,
                "raw0": raw0,
                "null0": null0,
                "excess0": excess0,
                f"raw{planted}": raw_planted,
                f"null{planted}": null_planted,
                f"excess{planted}": excess_planted,
                "delivered_overlap": delivered,
                "recovery": 100.0 * (excess_planted - excess0) / delivered,
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
    parser.add_argument("--planted", type=int, default=2)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--draws", type=int, default=10)
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
        covariance, folds = covariance_directories(
            arch=args.arch, work_root=args.work_root,
            covariance=args.cov, fold_covariance=args.fold_cov,
        )
        rows = summarize(
            study_level_centered(
                covariance=covariance, fold_covariance=folds, base_path=args.base,
                experts=experts, modules=modules, planted=args.planted,
                k=args.k, draws=args.draws, seed=args.seed, device=args.device,
            ),
            args.planted,
        )
        # Preserve the published artifact's list-of-rows schema so fig_power.py and
        # check_numbers.py can consume a reviewed rerun without a conversion step.
        target = write_json_new(args.out, rows)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))

    print(f"  level-centred planted {args.planted}/{args.k}")
    print(f"  {'variant':>30s} {'raw0':>8s} {'raw+':>8s} {'recovery':>10s}")
    for row in rows:
        print(
            f"  {row['variant']:>30s} {row['raw0']:8.4f} "
            f"{row[f'raw{args.planted}']:8.4f} {row['recovery']:9.2f}%"
        )
    print(f"  wrote {target}")


if __name__ == "__main__":
    main()
