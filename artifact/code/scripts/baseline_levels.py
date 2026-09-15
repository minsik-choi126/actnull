#!/usr/bin/env python3
"""Measure baseline-reported excess on activation-conditioned null pairs.

Each input task vector is independently passed through PRISM's corrected
activation-conditioned randomisation before any baseline is evaluated. The resulting pair
retains each update's spectrum and activation-mass profile but has no cross-task directional
agreement, so its target excess is zero. This is an external level check, not an additive
correction to the observed-data estimate.

Example:

    python scripts/baseline_levels.py \\
      --arch clip-vit-base-patch16 --work-root WORK_ROOT \\
      --base openai/clip-vit-base-patch16 \\
      --experts dtd=tanganke/clip-vit-base-patch16_dtd \\
                eurosat=tanganke/clip-vit-base-patch16_eurosat \\
      --out results/baseline_levels.json

Use --cov and --fold-cov instead of --work-root when those inputs do not share the standard
k1000_cov_<arch> / k1000_fold_<arch> layout. By default every ninth manifest module is
evaluated, matching the paper artifact's original arm.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import actnull.null as act
from actnull.experiment import (
    covariance_directories,
    fold_eigenvalues,
    parse_named_paths,
    public_sources,
    write_json_new,
)


LABELS = (
    ("orthogonality", "exact orthogonality"),
    ("isotropic", "isotropic $k/d$"),
    ("regmean", "whitened at participation ratio"),
    ("tikhonov", "whitened, all stored"),
    ("generative", "generative null"),
    ("permutation", "coordinate permutation"),
    ("ours", "activation-conditioned (ours, circular here)"),
)


def select_modules(
    manifest_modules: Sequence[str], specifications: Sequence[str] | None, stride: int
) -> list[str]:
    """Select exact module names or all modules with a requested terminal name."""
    available = sorted(manifest_modules)
    if not specifications:
        return available[::stride]
    selectors: list[str] = []
    for specification in specifications:
        selectors.extend(piece.strip() for piece in specification.split(",") if piece.strip())
    if not selectors:
        raise ValueError("--modules was supplied but contained no module names")
    selected: set[str] = set()
    for selector in selectors:
        if selector in available:
            selected.add(selector)
            continue
        matches = [module for module in available if module.rsplit(".", 1)[-1] == selector]
        if not matches:
            raise ValueError(
                f"module selector {selector!r} matches no entry in the covariance manifest"
            )
        selected.update(matches)
    return [module for module in available if module in selected]


def make_generator(device: str, seed: int) -> torch.Generator:
    """Construct a generator on the tensor device used by the corrected null."""
    try:
        return torch.Generator(device=torch.device(device)).manual_seed(seed)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"cannot construct a random generator on --device {device!r}") from error


def run(
    *,
    covariance: Path,
    fold_covariance: Path,
    modules: Sequence[str],
    base_path: str,
    expert_paths: Mapping[str, str],
    k: int = 8,
    draws: int = 20,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[dict[str, list[float]], float, dict]:
    """Evaluate all historical baselines on corrected-null pairs."""
    manifest = json.loads((covariance / "manifest.json").read_text(encoding="utf-8"))
    base = act.LazyCheckpoint(base_path, "vision_model.")
    experts = {
        name: act.LazyCheckpoint(checkpoint, "vision_model.")
        for name, checkpoint in expert_paths.items()
    }
    names = sorted(experts)
    accumulator = {key: [] for key, _ in LABELS}
    isotropic_levels: list[float] = []
    skipped_missing: list[str] = []
    skipped_degenerate = 0
    fold_counts: set[int] = set()

    for module_index, module in enumerate(modules):
        key = module + ".weight"
        if key not in base or any(key not in expert for expert in experts.values()):
            skipped_missing.append(module)
            continue
        tensor_file = covariance / f"{module.replace('.', '__')}.pt"
        if not tensor_file.is_file():
            raise FileNotFoundError(f"missing covariance tensor for {module}: {tensor_file}")
        blob = torch.load(tensor_file, map_location=device, weights_only=False)
        eigenvectors = blob["eigvecs"].to(device)
        eigenvalues = blob["eigvals"].to(device)
        d_in = int(blob["d"])
        trace_full = blob.get("trace_full")
        if trace_full is None:
            raise ValueError(
                f"{module}: covariance predates trace_full; rerun the current extractor"
            )
        if k > d_in:
            raise ValueError(f"{module}: --k {k} exceeds input dimension {d_in}")

        folds = fold_eigenvalues(fold_covariance, module, device="cpu")
        fold_counts.add(len(folds))
        stored_rank = int(eigenvectors.shape[0])
        available = min(len(values) for values in folds)
        if available < stored_rank:
            raise ValueError(
                f"{module}: folds store {available} eigenvalues but the covariance stores "
                f"{stored_rank} eigenvectors"
            )
        stacked = torch.stack([values[:stored_rank] for values in folds])
        standard_error = (stacked.std(0) / math.sqrt(len(folds))).float().to(device)
        blocks = act.resolvable_blocks(eigenvalues[:stored_rank], standard_error, 2.0)
        participation_ratio = manifest["modules"][module]["participation_ratio"]
        m_pr = max(1, min(int(participation_ratio), stored_rank))
        generator = make_generator(device, seed)

        # Keep only the current pair resident. This preserves LazyCheckpoint's bounded-memory
        # contract for pools whose individual tensors are large.
        for draw in range(draws):
            delta_a = act.load_delta(
                base, experts[names[draw % len(names)]], key, device
            )
            delta_b = act.load_delta(
                base, experts[names[(draw + 1) % len(names)]], key, device
            )
            if (
                not bool(torch.isfinite(delta_a).all())
                or not bool(torch.isfinite(delta_b).all())
                or float(delta_a.norm()) == 0.0
                or float(delta_b.norm()) == 0.0
            ):
                skipped_degenerate += 1
                continue

            # Only the corrected implementation is used: sign-corrected Haar frames,
            # singleton sign flips, and an exact full-complement frame.
            null_a = act.coupling_destroying_null(
                delta_a, eigenvectors, 0, generator, 128, blocks=blocks
            )
            null_b = act.coupling_destroying_null(
                delta_b, eigenvectors, 0, generator, 128, blocks=blocks
            )
            observed = act.overlap(act.top_right(null_a, k), act.top_right(null_b, k))
            isotropic = k / d_in
            isotropic_levels.append(isotropic)
            accumulator["orthogonality"].append(observed)
            accumulator["isotropic"].append(observed - isotropic)

            for label, retained in (("regmean", m_pr), ("tikhonov", stored_rank)):
                whitened_a = act.top_right(
                    act.truncated_whiten(null_a, eigenvalues, eigenvectors, retained), k
                )
                whitened_b = act.top_right(
                    act.truncated_whiten(null_b, eigenvalues, eigenvectors, retained), k
                )
                accumulator[label].append(act.overlap(whitened_a, whitened_b) - isotropic)

            generated_a = act.act_matched_null(
                null_a, eigenvalues, eigenvectors, float(trace_full), generator
            )
            generated_b = act.act_matched_null(
                null_b, eigenvalues, eigenvectors, float(trace_full), generator
            )
            accumulator["generative"].append(
                observed
                - act.overlap(act.top_right(generated_a, k), act.top_right(generated_b, k))
            )
            permutation = torch.randperm(d_in, generator=generator, device=device)
            accumulator["permutation"].append(
                observed
                - act.overlap(
                    act.top_right(null_a[:, permutation], k),
                    act.top_right(null_b[:, permutation], k),
                )
            )
            rerandomized_a = act.coupling_destroying_null(
                null_a, eigenvectors, 0, generator, 128, blocks=blocks
            )
            rerandomized_b = act.coupling_destroying_null(
                null_b, eigenvectors, 0, generator, 128, blocks=blocks
            )
            accumulator["ours"].append(
                observed
                - act.overlap(
                    act.top_right(rerandomized_a, k), act.top_right(rerandomized_b, k)
                )
            )

        print(
            f"[baseline] {module_index + 1}/{len(modules)} {module}: "
            f"{len(accumulator['ours'])} valid samples total",
            flush=True,
        )

    if not accumulator["ours"] or not isotropic_levels:
        raise ValueError("no valid baseline samples were produced")
    diagnostics = {
        "skipped_missing_modules": skipped_missing,
        "skipped_degenerate_draws": skipped_degenerate,
        "fold_counts": sorted(fold_counts),
    }
    return accumulator, st.mean(isotropic_levels), diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch", required=True, help="architecture label used in output")
    parser.add_argument(
        "--work-root",
        help="root containing k1000_cov_<arch> and k1000_fold_<arch>",
    )
    parser.add_argument("--cov", help="explicit activation covariance directory")
    parser.add_argument(
        "--fold-cov",
        help="explicit per-fold covariance root containing f0, f1, ...",
    )
    parser.add_argument("--base", required=True, help="base checkpoint repo or local directory")
    parser.add_argument("--experts", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--modules",
        nargs="+",
        help="exact manifest names or terminal names (comma-separated values also accepted)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=9,
        help="take every Nth manifest module when --modules is omitted (default: 9)",
    )
    parser.add_argument("--k", type=int, default=8, help="read-subspace dimension")
    parser.add_argument("--draws", type=int, default=20, help="null pairs per selected module")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, help="new JSON result path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.draws < 1:
        parser.error("--draws must be positive")
    if args.stride < 1:
        parser.error("--stride must be positive")
    try:
        experts = parse_named_paths(args.experts)
        covariance, fold_covariance = covariance_directories(
            arch=args.arch,
            work_root=args.work_root,
            covariance=args.cov,
            fold_covariance=args.fold_cov,
        )
        manifest = json.loads(
            (covariance / "manifest.json").read_text(encoding="utf-8")
        )
        modules = select_modules(manifest["modules"], args.modules, args.stride)
        if not modules:
            raise ValueError("module selection is empty")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"[baseline] indexing {len(experts)} experts", flush=True)
    accumulator, isotropic, diagnostics = run(
        covariance=covariance,
        fold_covariance=fold_covariance,
        modules=modules,
        base_path=args.base,
        expert_paths=experts,
        k=args.k,
        draws=args.draws,
        seed=args.seed,
        device=args.device,
    )
    levels = {key: st.mean(accumulator[key]) for key, _ in LABELS}
    print(
        f"\n  Excess reported on null pairs (truth = 0); isotropic level k/d = {isotropic:.5f}"
    )
    raw_overlap = levels["orthogonality"]
    raw_over_chance = raw_overlap / isotropic
    print(f"  modules {len(modules)}, samples {len(accumulator['ours'])}")
    print(f"  raw null-pair overlap / isotropic chance = {raw_overlap:.5f} / "
          f"{isotropic:.5f} = {raw_over_chance:.3f}x\n")
    print("  %-34s %14s" % ("baseline", "native excess"))
    for key, label in LABELS:
        value = levels[key]
        print("  %-34s %+14.5f" % (label, value))

    # Keep the original six top-level fields. Metadata records public expert labels only:
    # cache/checkpoint paths may be local and must not leak into a portable artifact.
    payload = {
        "arch": args.arch,
        "modules": modules,
        "k": args.k,
        "iso": isotropic,
        "n": len(accumulator["ours"]),
        "level": levels,
        "metadata": {
            "schema_version": 3,
            "labels": dict(LABELS),
            "level_definitions": {
                "orthogonality": "mean raw overlap; the native center is zero",
                "isotropic": "mean raw overlap minus k/d",
                "regmean": "mean participation-ratio-whitened overlap minus k/d",
                "tikhonov": "mean all-stored-whitened overlap minus k/d",
                "generative": "mean raw overlap minus mean generative-null overlap",
                "permutation": "mean raw overlap minus mean common-permutation overlap",
                "ours": "mean raw overlap minus mean rerandomized corrected-null overlap",
            },
            "raw_null_pair_overlap": raw_overlap,
            "isotropic_chance": isotropic,
            "raw_overlap_over_isotropic_chance": raw_over_chance,
            "ratio_definition": "raw_null_pair_overlap / isotropic_chance",
            "expert_names": public_sources(experts),
            "draws_per_module": args.draws,
            "seed": args.seed,
            "device": args.device,
            "module_stride": None if args.modules else args.stride,
            "block_z": 2.0,
            "null_implementation": "actnull.null.coupling_destroying_null",
            **diagnostics,
        },
    }
    output = write_json_new(args.out, payload)
    print(f"\n  wrote {output}")


if __name__ == "__main__":
    main()
