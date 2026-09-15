#!/usr/bin/env python3
"""Compute the analytic Haar block-null expectation for read-subspace overlap.

For a top-``k`` read projector ``P_i`` and mutually orthogonal activation blocks
``P_b`` of dimensions ``d_b``, let ``t_ib = tr(P_i P_b)``.  Independently Haar
rotating each task inside every block gives the exact expectation

    E[O(i,j)] = (1 / k) * sum_b t_ib * t_jb / d_b.

The blocks include every statistically resolvable block in the stored activation
eigenbasis and one block for its full orthogonal complement.  Thus this script has
no Monte Carlo null draws and no finite-draw error.  It materialises each observed
task vector only long enough to obtain its top-k right subspace, using the same
checkpoint and covariance readers as :mod:`actnull.null`.

Example::

    python scripts/exact_overlap.py \
      --base openai/clip-vit-base-patch16 --key_prefix vision_model. \
      --experts cifar10=tanganke/clip-vit-base-patch16_cifar10 dtd=... \
      --cov "$ACTNULL_WORK/cov_b16" --fold_cov "$ACTNULL_WORK/fold_b16" \
      --k 8 --device cpu --out results/exact_b16.csv

``null_act_raw`` and ``excess_raw`` are aliases retained for the existing
bootstrap/checking scripts.  Aggregate ``frac_num`` and ``frac_den`` separately
before taking their ratio; averaging row-wise fractions is not the paper's
estimand.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Sequence

import torch


# Running ``python scripts/exact_overlap.py`` puts scripts/, rather than the
# repository root, on sys.path.  Resolve the package relative to this file so the
# artifact is relocatable and never depends on an author's checkout path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import actnull.null as act


def contiguous_blocks(size: int, count: int) -> list[list[int]]:
    """Return the fixed-count fallback used when no fold covariance is supplied."""
    if size < 1:
        return []
    if count < 1:
        raise ValueError("block count must be positive")
    edges = torch.linspace(0, size, min(size, count) + 1).round().long().tolist()
    return [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def validate_blocks(blocks: Sequence[Sequence[int]], stored_rank: int) -> None:
    """Refuse overlapping, missing, or out-of-range stored directions."""
    flat = [int(i) for block in blocks for i in block]
    if any(len(block) == 0 for block in blocks):
        raise ValueError("activation blocks must be non-empty")
    if sorted(flat) != list(range(stored_rank)):
        raise ValueError(
            "activation blocks must partition each stored direction exactly once; "
            f"got {sorted(flat)} for rank {stored_rank}"
        )


def projector_block_masses(
    subspace: torch.Tensor,
    stored_basis: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    d_in: int,
) -> tuple[list[float], list[int]]:
    """Return ``tr(P_i P_b)`` and ``d_b`` for stored blocks plus complement.

    ``subspace`` has shape ``(d_in, k)`` with orthonormal columns, while
    ``stored_basis`` has shape ``(m, d_in)`` with orthonormal rows.  The complement
    mass is computed as ``k - sum(stored masses)``; no dense ``d_in x d_in``
    projector is formed.
    """
    if subspace.ndim != 2 or stored_basis.ndim != 2:
        raise ValueError("subspace and stored_basis must be matrices")
    if subspace.shape[0] != d_in or stored_basis.shape[1] != d_in:
        raise ValueError(
            f"dimension mismatch: subspace={tuple(subspace.shape)}, "
            f"stored_basis={tuple(stored_basis.shape)}, d_in={d_in}"
        )
    m = stored_basis.shape[0]
    if m > d_in:
        raise ValueError(f"stored rank {m} exceeds d_in={d_in}")
    validate_blocks(blocks, m)

    # These checks catch a mismatched/corrupt covariance before its coordinates are
    # interpreted as mutually orthogonal Haar blocks.
    k = subspace.shape[1]
    q_err = float(
        (subspace.T @ subspace - torch.eye(k, device=subspace.device, dtype=subspace.dtype))
        .abs()
        .max()
    )
    if q_err > 5e-3:
        raise ValueError(f"read subspace is not orthonormal (max Gram error {q_err:.3g})")
    if m:
        v_err = float(
            (stored_basis @ stored_basis.T
             - torch.eye(m, device=stored_basis.device, dtype=stored_basis.dtype))
            .abs()
            .max()
        )
        if v_err > 5e-3:
            raise ValueError(
                f"stored activation eigenvectors are not orthonormal "
                f"(max Gram error {v_err:.3g})"
            )
        coordinates = stored_basis @ subspace
    else:
        coordinates = subspace.new_empty((0, k))

    masses = [float(coordinates[list(block)].square().sum()) for block in blocks]
    stored_mass = math.fsum(masses)
    complement_mass = float(k) - stored_mass
    # Float32 eigenspaces can overshoot k by a few ulps.  A material negative mass,
    # however, means the purported stored vectors are not a valid orthogonal block basis.
    tolerance = 5e-4 * max(1, k)
    if complement_mass < -tolerance:
        raise ValueError(
            f"negative complement projector mass {complement_mass:.6g}; "
            "the stored activation basis is not orthogonal to numerical tolerance"
        )
    complement_mass = max(0.0, complement_mass)
    dims = [len(block) for block in blocks]
    if d_in > m:
        masses.append(complement_mass)
        dims.append(d_in - m)
    elif abs(complement_mass) > tolerance:
        raise ValueError(
            f"stored basis spans d_in but leaves projector mass {complement_mass:.6g}"
        )
    return masses, dims


def exact_block_null(
    masses_a: Sequence[float],
    masses_b: Sequence[float],
    block_dims: Sequence[int],
    k: int,
) -> float:
    """Evaluate ``(1/k) sum_b t_ab t_bb / d_b`` with input validation."""
    if k < 1:
        raise ValueError("k must be positive")
    if not (len(masses_a) == len(masses_b) == len(block_dims)):
        raise ValueError("the two mass vectors and block_dims must have equal lengths")
    if any(d < 1 for d in block_dims):
        raise ValueError("all block dimensions must be positive")
    if any(x < -1e-8 or not math.isfinite(x) for x in (*masses_a, *masses_b)):
        raise ValueError("projector block masses must be finite and non-negative")
    return math.fsum(a * b / d for a, b, d in zip(masses_a, masses_b, block_dims)) / k


def fold_blocks(
    module: str,
    eigvals: torch.Tensor,
    stored_rank: int,
    fold_cov: Path,
    block_z: float,
    device: str,
) -> list[list[int]]:
    """Build the same uncertainty-resolved blocks as :mod:`actnull.null`."""
    filename = f"{module.replace('.', '__')}.pt"
    fold_eigvals = []
    for fold_dir in sorted(fold_cov.glob("f*")):
        candidate = fold_dir / filename
        if candidate.exists():
            fold_eigvals.append(
                torch.load(candidate, map_location="cpu", weights_only=False)["eigvals"].double()
            )
    if len(fold_eigvals) < 4:
        raise SystemExit(
            f"[refuse] {module}: only {len(fold_eigvals)} folds under {fold_cov}; "
            "eigenvalue standard errors need at least four"
        )
    available = min(len(x) for x in fold_eigvals)
    if available < stored_rank:
        raise SystemExit(
            f"[refuse] {module}: folds store {available} eigenvalues but the exact null "
            f"needs {stored_rank} to partition the retained activation basis"
        )
    stack = torch.stack([x[:stored_rank] for x in fold_eigvals])
    se = (stack.std(0) / len(fold_eigvals) ** 0.5).float().to(device)
    blocks = act.resolvable_blocks(eigvals[:stored_rank], se, block_z)
    validate_blocks(blocks, stored_rank)
    return blocks


def parse_experts(specs: Sequence[str], adapters: bool, key_prefix: str):
    experts = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"[refuse] expert must be NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise SystemExit(f"[refuse] expert must be NAME=path, got {spec!r}")
        if name in experts:
            raise SystemExit(f"[refuse] duplicate expert name {name!r}")
        print(f"[exact] indexing expert {name}", flush=True)
        experts[name] = (
            act.AdapterCheckpoint(path) if adapters else act.LazyCheckpoint(path, key_prefix)
        )
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--experts", nargs="+", required=True, help="NAME=path ...")
    parser.add_argument("--cov", required=True, help="activation covariance directory")
    parser.add_argument(
        "--manifest", default="",
        help="covariance manifest JSON; defaults to <cov>/manifest.json",
    )
    parser.add_argument(
        "--fold_cov", "--fold-cov", default="",
        help="per-fold covariance root (f0, f1, ...); strongly recommended",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--null_m", "--null-m", type=int, default=0,
        help="retained activation eigenvectors; 0 uses every stored direction",
    )
    parser.add_argument("--block_z", "--block-z", type=float, default=2.0)
    parser.add_argument(
        "--null_blocks", "--null-blocks", type=int, default=8,
        help="fixed-count fallback used only when --fold_cov is absent",
    )
    parser.add_argument(
        "--only_modules", "--only-modules", default="",
        help="comma-separated terminal module names, e.g. q_proj,v_proj",
    )
    parser.add_argument(
        "--experts_are_adapters", "--experts-are-adapters", "--adapters",
        action="store_true", dest="adapters",
        help="read experts as PEFT adapters rather than full checkpoints; this still computes "
        "the activation-block null, not the separate shared-A LoRA factor null",
    )
    parser.add_argument("--key_prefix", "--key-prefix", default="")
    parser.add_argument("--min_rel_norm", "--min-rel-norm", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.k < 1:
        raise SystemExit("[refuse] --k must be positive")
    if args.null_m < 0:
        raise SystemExit("[refuse] --null_m cannot be negative")
    if args.block_z < 0:
        raise SystemExit("[refuse] --block_z cannot be negative")

    cov_dir = Path(args.cov)
    manifest_path = Path(args.manifest) if args.manifest else cov_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"[refuse] missing covariance manifest: {manifest_path}; pass --manifest when "
            "the tensors and manifest are stored separately"
        )
    manifest = json.loads(manifest_path.read_text())

    print(f"[exact] indexing base {args.base}", flush=True)
    base = act.LazyCheckpoint(args.base, args.key_prefix)
    experts = parse_experts(args.experts, args.adapters, args.key_prefix)

    modules = sorted(manifest["modules"])
    if args.only_modules:
        keep = {x.strip() for x in args.only_modules.split(",") if x.strip()}
        modules = [module for module in modules if module.split(".")[-1] in keep]
        print(f"[exact] --only_modules {sorted(keep)}: {len(modules)} retained", flush=True)

    fold_cov = Path(args.fold_cov) if args.fold_cov else None
    rows: list[dict] = []
    degenerate: dict[tuple[str, str], float] = {}
    skipped_modules: list[str] = []

    for module_index, module in enumerate(modules):
        key = module + ".weight"
        if key not in base or any(key not in expert for expert in experts.values()):
            continue
        blob = torch.load(
            cov_dir / f"{module.replace('.', '__')}.pt",
            map_location=args.device,
            weights_only=False,
        )
        eigvals = blob["eigvals"].to(args.device)
        basis = blob["eigvecs"].to(args.device)
        d_in = int(blob["d"])
        stored_rank = args.null_m or basis.shape[0]
        if stored_rank > basis.shape[0]:
            raise SystemExit(
                f"[refuse] {module}: --null_m={stored_rank} exceeds the "
                f"{basis.shape[0]} stored activation directions"
            )
        basis = basis[:stored_rank]
        eigvals = eigvals[:stored_rank]
        blocks = (
            fold_blocks(
                module, eigvals, stored_rank, fold_cov, args.block_z, args.device
            )
            if fold_cov
            else contiguous_blocks(stored_rank, args.null_blocks)
        )
        validate_blocks(blocks, stored_rank)

        base_weight = base.get(key, args.device)
        base_norm = float(base_weight.norm())
        subspaces: dict[str, torch.Tensor] = {}
        masses: dict[str, list[float]] = {}
        block_dims: list[int] | None = None
        for name, expert in experts.items():
            # Reuse the tested adapter/full-checkpoint dispatch.  Full checkpoints may
            # read the base tensor once more internally; that small I/O cost keeps this
            # path identical to the producer used for the archived CSVs.
            delta = act.load_delta(base, expert, key, args.device)
            if delta.shape[1] != d_in:
                raise SystemExit(
                    f"[refuse] {module}: task vector d_in={delta.shape[1]} but covariance "
                    f"d_in={d_in}"
                )
            rel_norm = float(delta.norm()) / max(base_norm, 1e-30)
            if not math.isfinite(rel_norm) or rel_norm < args.min_rel_norm:
                degenerate[(module, name)] = rel_norm
                continue
            if args.k > min(delta.shape):
                raise SystemExit(
                    f"[refuse] {module}: --k={args.k} exceeds task-vector rank bound "
                    f"min(shape)={min(delta.shape)}"
                )
            subspace = act.top_right(delta, args.k)
            task_masses, dims = projector_block_masses(
                subspace, basis, blocks, d_in
            )
            if block_dims is None:
                block_dims = dims
            elif dims != block_dims:
                raise AssertionError("block dimensions changed within one module")
            subspaces[name] = subspace
            masses[name] = task_masses
            del delta

        live = sorted(subspaces)
        if len(live) < 2:
            skipped_modules.append(module)
            continue
        assert block_dims is not None

        for a, b in itertools.combinations(live, 2):
            raw = act.overlap(subspaces[a], subspaces[b])
            null_exact = exact_block_null(masses[a], masses[b], block_dims, args.k)
            null_iso = args.k / d_in
            excess_exact = raw - null_exact
            frac_num = null_exact - null_iso
            frac_den = raw - null_iso
            frac_exact = frac_num / frac_den if frac_den != 0 else float("nan")
            rows.append(
                dict(
                    module=module,
                    d_in=d_in,
                    a=a,
                    b=b,
                    k=args.k,
                    m=stored_rank,
                    null_kind="exact_haar_block",
                    # Keep n_blocks compatible with actnull.null: it counts the
                    # uncertainty-resolved stored blocks, excluding the complement.
                    n_blocks=len(blocks),
                    n_total_blocks=len(block_dims),
                    complement_dim=d_in - stored_rank,
                    raw=raw,
                    null_iso=null_iso,
                    null_exact=null_exact,
                    null_act_raw=null_exact,
                    excess_over_iso=raw - null_iso,
                    excess_exact=excess_exact,
                    excess_raw=excess_exact,
                    h0_excess=0.0,
                    excess_raw_corrected=excess_exact,
                    frac_num=frac_num,
                    frac_den=frac_den,
                    frac_exact=frac_exact,
                )
            )
        if (module_index + 1) % 12 == 0:
            print(f"[exact] {module_index + 1}/{len(modules)} modules", flush=True)

    if degenerate:
        counts: dict[str, int] = {}
        for _, name in degenerate:
            counts[name] = counts.get(name, 0) + 1
        print(
            f"[exact] dropped {len(degenerate)} degenerate (module, expert) cells; "
            f"per expert: {dict(sorted(counts.items()))}",
            flush=True,
        )
    if skipped_modules:
        print(
            f"[exact] skipped {len(skipped_modules)} modules with fewer than two live experts",
            flush=True,
        )
    if not rows:
        raise SystemExit(
            "[refuse] no module produced a pair; check checkpoint namespaces, covariance, "
            "adapter targets, and --only_modules"
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean_raw = st.mean(row["raw"] for row in rows)
    mean_iso = st.mean(row["null_iso"] for row in rows)
    mean_null = st.mean(row["null_exact"] for row in rows)
    fraction = (mean_null - mean_iso) / (mean_raw - mean_iso)
    print(f"[exact] wrote {len(rows)} rows to {output}")
    print(f"  mean raw          {mean_raw:+.6f}")
    print(f"  mean null_iso     {mean_iso:+.6f}")
    print(f"  mean null_exact   {mean_null:+.6f}")
    print(f"  mean excess_exact {mean_raw - mean_null:+.6f}")
    print(f"  explained         {100 * fraction:.3f}%")


if __name__ == "__main__":
    main()
