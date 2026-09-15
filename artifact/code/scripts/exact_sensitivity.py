#!/usr/bin/env python3
"""One-pass full-pool sensitivity grid for the analytic activation-block null.

This is the efficient companion to :mod:`scripts.exact_overlap`.  A naive grid invokes
``exact_overlap.py`` once per setting and consequently reads every checkpoint and computes
every task-vector SVD 36 times.  This producer instead computes a task/module's top-``max(k)``
right singular frame once, optionally caches it on disk, and obtains all nested smaller
subspaces by slicing that frame.

The default grid is the paper audit requested for the full fine-tuning pool::

    k       = 4, 8, 16, 32
    null_m  = 16, 32, 64
    block_z = 1, 2, 3

It writes three files:

* a compact cell CSV with one row per (module, task pair, k) and nine exact-null columns;
* a 36-row summary CSV using the ratio of sums (never the mean of row-wise ratios); and
* a JSON record of the exact task/module scope, block counts, cache use, and inputs.

The exact expectation is imported from ``scripts/exact_overlap.py``.  No Monte Carlo null
draw is made, so the grid has no QR dependency or finite-draw error.  The only expensive
operation left is the observed checkpoint SVD, once per (module, task), regardless of grid
size.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import actnull.null as act
from scripts.exact_overlap import exact_block_null, validate_blocks


FORMAT_VERSION = 1


def parse_unique_ints(value: str, option: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer grid, preserving sorted uniqueness."""
    try:
        values = tuple(sorted({int(piece.strip()) for piece in value.split(",") if piece.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError(f"{option} must contain positive integers")
    return values


def parse_unique_floats(value: str, option: str) -> tuple[float, ...]:
    """Parse a comma-separated finite non-negative float grid."""
    try:
        values = tuple(sorted({float(piece.strip()) for piece in value.split(",") if piece.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated numbers") from exc
    if not values or any(number < 0 or not math.isfinite(number) for number in values):
        raise argparse.ArgumentTypeError(f"{option} must contain finite non-negative numbers")
    return values


def number_tag(value: float) -> str:
    """Return a stable CSV-safe spelling (e.g. 1.5 -> ``1p5``)."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def setting_tag(null_m: int, block_z: float) -> str:
    return f"m{null_m}_z{number_tag(block_z)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def public_checkpoint_id(spec: str) -> str:
    """Return a path-free checkpoint identifier, including a cached HF revision when known."""
    path = Path(spec)
    parts = path.parts
    if "snapshots" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("snapshots")
        if index > 0 and index + 1 < len(parts):
            model_dir = parts[index - 1]
            revision = parts[index + 1]
            if model_dir.startswith("models--"):
                repo_id = model_dir[len("models--"):].replace("--", "/")
                return f"{repo_id}@{revision}"
    if not path.exists() and "/" in spec and not spec.startswith("/"):
        return spec
    return f"local:{path.name}"


def portable_path_label(path: Path) -> str:
    """Describe an input/output without serializing a workstation-specific absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        if resolved.name == "manifest.json":
            return f"{resolved.parent.name}/{resolved.name}"
        if resolved.parent.name in {"exact_subspaces", "results"}:
            return f"{resolved.parent.name}/{resolved.name}"
        return resolved.name


def checkpoint_source(checkpoint: act.LazyCheckpoint, key: str) -> dict:
    """Describe the exact local tensor source used to validate a subspace cache."""
    resolved = checkpoint.resolve(key)
    if resolved is None:
        raise KeyError(key)
    source = checkpoint.map.get(resolved)
    record = {"resolved_key": resolved}
    if source is None:
        # Dense .bin checkpoints do not retain the shard path in LazyCheckpoint.  The cache
        # identity still includes the user-facing checkpoint spec, but callers should not
        # silently trust that weaker signature.
        record["cache_safe"] = False
        record["source"] = "dense_bin_loaded_in_memory"
        return record
    path = Path(source).resolve()
    stat = path.stat()
    record.update(
        cache_safe=True,
        source_file=path.name,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
    return record


def module_cache_identity(
    module: str,
    key: str,
    d_in: int,
    base_spec: str,
    expert_specs: Mapping[str, str],
    base: act.LazyCheckpoint,
    experts: Mapping[str, act.LazyCheckpoint],
) -> tuple[dict, bool]:
    """Build a checkpoint-file-aware identity for one cached module."""
    base_source = checkpoint_source(base, key)
    expert_sources = {name: checkpoint_source(experts[name], key) for name in sorted(experts)}
    identity = {
        "format": "exact_sensitivity_subspace_cache",
        "format_version": FORMAT_VERSION,
        "module": module,
        "key": key,
        "d_in": d_in,
        "base_spec": public_checkpoint_id(base_spec),
        "expert_specs": {
            name: public_checkpoint_id(expert_specs[name]) for name in sorted(expert_specs)
        },
        "base_source": base_source,
        "expert_sources": expert_sources,
    }
    cache_safe = bool(base_source["cache_safe"]) and all(
        bool(source["cache_safe"]) for source in expert_sources.values()
    )
    return identity, cache_safe


def cache_path(cache_dir: Path, module: str, identity: Mapping) -> Path:
    safe_module = module.replace(".", "__")
    return cache_dir / f"{safe_module}.{json_digest(identity)[:16]}.pt"


def frame_error(frame: torch.Tensor, k: int) -> float:
    gram = frame[:, :k].T @ frame[:, :k]
    eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
    return float((gram - eye).abs().max())


def read_subspace_cache(
    path: Path,
    identity: Mapping,
    task_names: Sequence[str],
    d_in: int,
    max_k: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float]] | None:
    """Load a complete validated cache, or return ``None`` and force recomputation."""
    if not path.exists():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format_version") != FORMAT_VERSION:
            return None
        if blob.get("identity_digest") != json_digest(identity):
            return None
        if int(blob.get("max_k", 0)) < max_k:
            return None
        frames = blob["subspaces"]
        if sorted(frames) != sorted(task_names):
            return None
        checked: dict[str, torch.Tensor] = {}
        for name in task_names:
            frame = frames[name]
            if frame.ndim != 2 or frame.shape[0] != d_in or frame.shape[1] < max_k:
                return None
            if not bool(torch.isfinite(frame[:, :max_k]).all()):
                return None
            if frame_error(frame, max_k) > 5e-3:
                return None
            # A top-k frame is commonly a transposed view into the full Vh returned by SVD.
            # Materialise only the requested thin frame; otherwise torch serialization keeps
            # the entire backing Vh storage alive (gigabytes across a full pool).
            checked[name] = frame[:, :max_k].to(device).contiguous()
        relative_norms = {name: float(blob["relative_norms"][name]) for name in task_names}
        return checked, relative_norms
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def write_subspace_cache(
    path: Path,
    identity: Mapping,
    subspaces: Mapping[str, torch.Tensor],
    relative_norms: Mapping[str, float],
    max_k: int,
) -> None:
    """Atomically store CPU frames so an interrupted full-pool run is resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format_version": FORMAT_VERSION,
        "identity_digest": json_digest(identity),
        "identity": identity,
        "max_k": max_k,
        # ``top_right`` returns a view into Vh.  Clone the contiguous thin frame so torch.save
        # does not serialize Vh's much larger backing storage.
        "subspaces": {
            name: frame.detach().cpu().contiguous().clone()
            for name, frame in subspaces.items()
        },
        "relative_norms": dict(relative_norms),
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(blob, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fold_standard_errors(
    module: str,
    fold_cov: Path,
    required_rank: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    """Load fold eigenvalues once and reproduce the paper's SD/sqrt(F) standard error."""
    filename = f"{module.replace('.', '__')}.pt"
    fold_values = []
    for fold_dir in sorted(fold_cov.glob("f*")):
        candidate = fold_dir / filename
        if candidate.exists():
            fold_values.append(
                torch.load(candidate, map_location="cpu", weights_only=False)["eigvals"].double()
            )
    if len(fold_values) < 4:
        raise SystemExit(
            f"[refuse] {module}: found only {len(fold_values)} fold covariances under "
            f"{fold_cov}; at least four are required"
        )
    available = min(len(values) for values in fold_values)
    if available < required_rank:
        raise SystemExit(
            f"[refuse] {module}: fold covariances store {available} eigenvalues but "
            f"max(null_m)={required_rank}"
        )
    stack = torch.stack([values[:required_rank] for values in fold_values])
    standard_errors = stack.std(0).div(math.sqrt(len(fold_values))).float().to(device)
    if not bool(torch.isfinite(standard_errors).all()) or bool((standard_errors < 0).any()):
        raise SystemExit(f"[refuse] {module}: fold eigenvalue standard errors are invalid")
    return standard_errors, len(fold_values)


def coordinate_block_masses(
    coordinates: torch.Tensor,
    k: int,
    blocks: Sequence[Sequence[int]],
    null_m: int,
    d_in: int,
) -> tuple[list[float], list[int]]:
    """Compute projector block masses from cached ``V_m @ U_k`` coordinates."""
    if coordinates.ndim != 2 or coordinates.shape[0] < null_m or coordinates.shape[1] < k:
        raise ValueError("coordinate matrix is smaller than the requested (null_m, k)")
    if null_m > d_in:
        raise ValueError("null_m cannot exceed d_in")
    validate_blocks(blocks, null_m)
    view = coordinates[:null_m, :k]
    masses = [float(view[list(block)].square().sum()) for block in blocks]
    stored_mass = math.fsum(masses)
    complement_mass = float(k) - stored_mass
    tolerance = 5e-4 * max(1, k)
    if complement_mass < -tolerance:
        raise ValueError(
            f"negative complement projector mass {complement_mass:.6g}; activation basis "
            "or cached frame is not orthonormal"
        )
    complement_mass = max(0.0, complement_mass)
    dimensions = [len(block) for block in blocks]
    if d_in > null_m:
        masses.append(complement_mass)
        dimensions.append(d_in - null_m)
    elif abs(complement_mass) > tolerance:
        raise ValueError("full activation basis leaves nonzero complement projector mass")
    return masses, dimensions


def module_sensitivity_rows(
    module: str,
    d_in: int,
    subspaces: Mapping[str, torch.Tensor],
    basis: torch.Tensor,
    eigvals: torch.Tensor,
    standard_errors: torch.Tensor,
    k_values: Sequence[int],
    null_m_values: Sequence[int],
    block_z_values: Sequence[float],
) -> tuple[list[dict], dict[str, int]]:
    """Evaluate every task-pair cell for one module without another checkpoint SVD."""
    max_k, max_m = max(k_values), max(null_m_values)
    if basis.ndim != 2 or basis.shape[0] < max_m or basis.shape[1] != d_in:
        raise ValueError(
            f"{module}: activation basis shape {tuple(basis.shape)} cannot serve max_m={max_m}, "
            f"d_in={d_in}"
        )
    if eigvals.shape[0] < max_m or standard_errors.shape[0] < max_m:
        raise ValueError(f"{module}: eigenvalue arrays cannot serve max_m={max_m}")
    basis = basis[:max_m]
    gram_error = float(
        (basis @ basis.T - torch.eye(max_m, dtype=basis.dtype, device=basis.device)).abs().max()
    )
    if gram_error > 5e-3:
        raise ValueError(f"{module}: activation basis Gram error is {gram_error:.3g}")

    names = sorted(subspaces)
    if len(names) < 2:
        raise ValueError(f"{module}: fewer than two task subspaces")
    coordinates: dict[str, torch.Tensor] = {}
    for name in names:
        frame = subspaces[name]
        if frame.shape[0] != d_in or frame.shape[1] < max_k:
            raise ValueError(f"{module}/{name}: frame cannot serve max(k)={max_k}")
        error = frame_error(frame, max_k)
        if error > 5e-3:
            raise ValueError(f"{module}/{name}: read-frame Gram error is {error:.3g}")
        coordinates[name] = basis @ frame[:, :max_k]

    blocks_by_setting: dict[tuple[int, float], list[list[int]]] = {}
    block_counts: dict[str, int] = {}
    for null_m, block_z in itertools.product(null_m_values, block_z_values):
        blocks = act.resolvable_blocks(
            eigvals[:null_m], standard_errors[:null_m], float(block_z)
        )
        validate_blocks(blocks, null_m)
        blocks_by_setting[(null_m, block_z)] = blocks
        block_counts[setting_tag(null_m, block_z)] = len(blocks)

    masses: dict[tuple[str, int, int, float], tuple[list[float], list[int]]] = {}
    for name, k, null_m, block_z in itertools.product(
        names, k_values, null_m_values, block_z_values
    ):
        masses[(name, k, null_m, block_z)] = coordinate_block_masses(
            coordinates[name], k, blocks_by_setting[(null_m, block_z)], null_m, d_in
        )

    rows: list[dict] = []
    for a, b in itertools.combinations(names, 2):
        for k in k_values:
            raw = act.overlap(subspaces[a][:, :k], subspaces[b][:, :k])
            null_iso = k / d_in
            if raw < -1e-5 or raw > 1.0 + 1e-5:
                raise ValueError(f"{module}/{a}/{b}/k={k}: overlap {raw} is outside [0,1]")
            row = {
                "module": module,
                "d_in": d_in,
                "a": a,
                "b": b,
                "k": k,
                "raw": raw,
                "null_iso": null_iso,
                "frac_den": raw - null_iso,
            }
            for null_m, block_z in itertools.product(null_m_values, block_z_values):
                mass_a, dimensions_a = masses[(a, k, null_m, block_z)]
                mass_b, dimensions_b = masses[(b, k, null_m, block_z)]
                if dimensions_a != dimensions_b:
                    raise AssertionError("block dimensions changed across tasks")
                null = exact_block_null(mass_a, mass_b, dimensions_a, k)
                if null < -1e-5 or null > 1.0 + 1e-5:
                    raise ValueError(
                        f"{module}/{a}/{b}/k={k}: exact null {null} is outside [0,1]"
                    )
                row[f"null_{setting_tag(null_m, block_z)}"] = null
            rows.append(row)
    return rows, block_counts


@dataclass
class KahanSum:
    total: float = 0.0
    compensation: float = 0.0

    def add(self, value: float) -> None:
        corrected = value - self.compensation
        updated = self.total + corrected
        self.compensation = (updated - self.total) - corrected
        self.total = updated


@dataclass
class Aggregate:
    count: int = 0
    raw: KahanSum = None  # type: ignore[assignment]
    isotropic: KahanSum = None  # type: ignore[assignment]
    null: KahanSum = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.raw = KahanSum()
        self.isotropic = KahanSum()
        self.null = KahanSum()

    def add(self, raw: float, isotropic: float, null: float) -> None:
        self.count += 1
        self.raw.add(raw)
        self.isotropic.add(isotropic)
        self.null.add(null)


def summary_rows(
    aggregates: Mapping[tuple[int, int, float], Aggregate],
    base_spec: str,
    task_count: int,
    module_count: int,
    block_counts: Mapping[str, Mapping[str, int]],
) -> list[dict]:
    """Create ratio-of-sums summaries with explicit equal-cell weighting."""
    result = []
    for (k, null_m, block_z), aggregate in sorted(aggregates.items()):
        if aggregate.count < 1:
            raise ValueError("cannot summarize an empty sensitivity setting")
        raw_sum = aggregate.raw.total
        iso_sum = aggregate.isotropic.total
        null_sum = aggregate.null.total
        numerator = null_sum - iso_sum
        denominator = raw_sum - iso_sum
        fraction = numerator / denominator if denominator else float("nan")
        tag = setting_tag(null_m, block_z)
        counts = [int(module_counts[tag]) for module_counts in block_counts.values()]
        result.append(
            {
                "scope": "all_task_pairs_x_all_retained_modules",
                "weighting": "equal_cell_ratio_of_sums",
                "base": public_checkpoint_id(base_spec),
                "n_tasks": task_count,
                "n_modules": module_count,
                "n_cells": aggregate.count,
                "k": k,
                "null_m": null_m,
                "block_z": block_z,
                "mean_raw": raw_sum / aggregate.count,
                "mean_null_iso": iso_sum / aggregate.count,
                "mean_null_exact": null_sum / aggregate.count,
                "mean_excess_exact": (raw_sum - null_sum) / aggregate.count,
                "fraction_explained": fraction,
                "fraction_explained_pct": 100.0 * fraction,
                "frac_num_sum": numerator,
                "frac_den_sum": denominator,
                "n_blocks_min": min(counts),
                "n_blocks_median": statistics.median(counts),
                "n_blocks_max": max(counts),
            }
        )
    return result


def parse_experts(specs: Sequence[str], key_prefix: str):
    expert_specs: dict[str, str] = {}
    experts: dict[str, act.LazyCheckpoint] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"[refuse] expert must be NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise SystemExit(f"[refuse] expert must be NAME=path, got {spec!r}")
        if name in experts:
            raise SystemExit(f"[refuse] duplicate expert name {name!r}")
        print(f"[sensitivity] indexing expert {name}", flush=True)
        expert_specs[name] = path
        experts[name] = act.LazyCheckpoint(path, key_prefix)
    return expert_specs, experts


def output_path(value: str, cells: Path, suffix: str) -> Path:
    if value:
        return Path(value)
    return cells.with_name(f"{cells.stem}_{suffix}{cells.suffix if suffix == 'summary' else '.json'}")


def atomic_write_csv(path: Path, rows: Sequence[Mapping], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"[refuse] output exists: {path}; pass --overwrite deliberately")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"[refuse] output exists: {path}; pass --overwrite deliberately")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--experts", nargs="+", required=True, help="NAME=path ...")
    parser.add_argument("--cov", required=True)
    parser.add_argument("--manifest", default="", help="defaults to <cov>/manifest.json")
    parser.add_argument("--fold_cov", "--fold-cov", required=True)
    parser.add_argument("--k_values", "--k-values", default="4,8,16,32")
    parser.add_argument("--null_m_values", "--null-m-values", default="16,32,64")
    parser.add_argument("--block_z_values", "--block-z-values", default="1,2,3")
    parser.add_argument("--only_modules", "--only-modules", default="")
    parser.add_argument("--key_prefix", "--key-prefix", default="")
    parser.add_argument("--min_rel_norm", "--min-rel-norm", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf_home", "--hf-home", default="")
    parser.add_argument(
        "--cache_dir", "--cache-dir", default="",
        help="optional per-module top-max(k) cache; makes interrupted/rerun grids cheap",
    )
    parser.add_argument("--out", required=True, help="compact wide cell CSV")
    parser.add_argument("--summary", default="", help="default: <out stem>_summary.csv")
    parser.add_argument("--metadata", default="", help="default: <out stem>_metadata.json")
    parser.add_argument("--expect_tasks", "--expect-tasks", type=int, default=0)
    parser.add_argument("--expect_modules", "--expect-modules", type=int, default=0)
    parser.add_argument(
        "--allow_incomplete_pool", "--allow-incomplete-pool", action="store_true",
        help="allow a module to omit missing/degenerate tasks; off by default",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        k_values = parse_unique_ints(args.k_values, "--k-values")
        null_m_values = parse_unique_ints(args.null_m_values, "--null-m-values")
        block_z_values = parse_unique_floats(args.block_z_values, "--block-z-values")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.min_rel_norm < 0 or not math.isfinite(args.min_rel_norm):
        parser.error("--min-rel-norm must be finite and non-negative")
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).resolve())

    cells_path = Path(args.out)
    summary_path = output_path(args.summary, cells_path, "summary")
    metadata_path = output_path(args.metadata, cells_path, "metadata")
    if len({cells_path.resolve(), summary_path.resolve(), metadata_path.resolve()}) != 3:
        raise SystemExit("[refuse] --out, --summary, and --metadata must be distinct files")
    for path in (cells_path, summary_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"[refuse] output exists: {path}; pass --overwrite deliberately")

    cov_dir = Path(args.cov)
    fold_cov = Path(args.fold_cov)
    manifest_path = Path(args.manifest) if args.manifest else cov_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"[refuse] missing covariance manifest: {manifest_path}")
    if not fold_cov.is_dir():
        raise SystemExit(f"[refuse] missing fold covariance directory: {fold_cov}")
    manifest = json.loads(manifest_path.read_text())
    if max(null_m_values) > int(manifest.get("m_out", 0)):
        raise SystemExit(
            f"[refuse] max(null_m)={max(null_m_values)} exceeds manifest m_out="
            f"{manifest.get('m_out')}"
        )

    print(f"[sensitivity] indexing base {args.base}", flush=True)
    base = act.LazyCheckpoint(args.base, args.key_prefix)
    expert_specs, experts = parse_experts(args.experts, args.key_prefix)
    task_names = sorted(experts)
    if args.expect_tasks and len(task_names) != args.expect_tasks:
        raise SystemExit(
            f"[refuse] expected {args.expect_tasks} tasks but received {len(task_names)}"
        )

    modules = sorted(manifest["modules"])
    if args.only_modules:
        keep = {piece.strip() for piece in args.only_modules.split(",") if piece.strip()}
        modules = [module for module in modules if module.split(".")[-1] in keep]
        print(f"[sensitivity] --only-modules {sorted(keep)}: {len(modules)} retained", flush=True)
    if args.expect_modules and len(modules) != args.expect_modules:
        raise SystemExit(
            f"[refuse] manifest/filter selected {len(modules)} modules, expected "
            f"{args.expect_modules}"
        )
    if not modules:
        raise SystemExit("[refuse] no modules selected")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    max_k, max_m = max(k_values), max(null_m_values)
    setting_pairs = tuple(itertools.product(null_m_values, block_z_values))
    fieldnames = ["module", "d_in", "a", "b", "k", "raw", "null_iso", "frac_den"] + [
        f"null_{setting_tag(null_m, block_z)}" for null_m, block_z in setting_pairs
    ]
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    partial_cells = cells_path.with_name(cells_path.name + ".partial")
    aggregates = {
        (k, null_m, block_z): Aggregate()
        for k, null_m, block_z in itertools.product(k_values, null_m_values, block_z_values)
    }
    block_counts: dict[str, dict[str, int]] = {}
    relative_norms: dict[str, dict[str, float]] = {}
    fold_counts: dict[str, int] = {}
    retained_modules: list[str] = []
    cache_hits = 0
    cache_misses = 0
    cache_bypassed = 0
    rows_written = 0

    with partial_cells.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for module_index, module in enumerate(modules):
            key = module + ".weight"
            missing = []
            if key not in base:
                missing.append(f"base ({base.why_missing(key)})")
            for name, expert in experts.items():
                if key not in expert:
                    missing.append(f"{name} ({expert.why_missing(key)})")
            if missing:
                message = f"{module}: missing checkpoint tensors: " + "; ".join(missing)
                if args.allow_incomplete_pool:
                    print(f"[sensitivity] skip {message}", flush=True)
                    continue
                raise SystemExit(f"[refuse] {message}")

            covariance_path = cov_dir / f"{module.replace('.', '__')}.pt"
            if not covariance_path.exists():
                raise SystemExit(f"[refuse] missing covariance tensor: {covariance_path}")
            blob = torch.load(
                covariance_path, map_location=args.device, weights_only=False
            )
            basis = blob["eigvecs"].to(args.device)
            eigvals = blob["eigvals"].to(args.device)
            d_in = int(blob["d"])
            if basis.shape[0] < max_m:
                raise SystemExit(
                    f"[refuse] {module}: covariance stores {basis.shape[0]} directions, "
                    f"below max(null_m)={max_m}"
                )
            standard_errors, fold_count = fold_standard_errors(
                module, fold_cov, max_m, args.device
            )
            fold_counts[module] = fold_count

            identity, cache_safe = module_cache_identity(
                module, key, d_in, args.base, expert_specs, base, experts
            )
            cached = None
            module_cache = None
            if cache_dir and cache_safe:
                module_cache = cache_path(cache_dir, module, identity)
                cached = read_subspace_cache(
                    module_cache, identity, task_names, d_in, max_k, args.device
                )
            elif cache_dir:
                cache_bypassed += 1
            if cached is not None:
                subspaces, module_norms = cached
                cache_hits += 1
                print(f"[sensitivity] cache hit {module}", flush=True)
            else:
                cache_misses += 1
                base_weight = base.get(key, args.device)
                base_norm = float(base_weight.norm())
                subspaces: dict[str, torch.Tensor] = {}
                module_norms: dict[str, float] = {}
                for name, expert in experts.items():
                    delta = expert.get(key, args.device) - base_weight
                    relative_norm = float(delta.norm()) / max(base_norm, 1e-30)
                    if not math.isfinite(relative_norm) or relative_norm < args.min_rel_norm:
                        if args.allow_incomplete_pool:
                            print(
                                f"[sensitivity] drop {module}/{name}: relative norm "
                                f"{relative_norm:.3g}", flush=True,
                            )
                            continue
                        raise SystemExit(
                            f"[refuse] {module}/{name}: relative task-vector norm "
                            f"{relative_norm:.3g} is below {args.min_rel_norm:.3g}"
                        )
                    if max_k > min(delta.shape):
                        raise SystemExit(
                            f"[refuse] {module}/{name}: max(k)={max_k} exceeds task-vector "
                            f"rank bound {min(delta.shape)}"
                        )
                    subspaces[name] = act.top_right(delta, max_k)
                    module_norms[name] = relative_norm
                    del delta
                del base_weight
                if len(subspaces) < 2:
                    if args.allow_incomplete_pool:
                        print(f"[sensitivity] skip {module}: fewer than two live tasks", flush=True)
                        continue
                    raise SystemExit(f"[refuse] {module}: fewer than two live tasks")
                if module_cache is not None and len(subspaces) == len(task_names):
                    write_subspace_cache(
                        module_cache, identity, subspaces, module_norms, max_k
                    )

            if not args.allow_incomplete_pool and sorted(subspaces) != task_names:
                raise SystemExit(f"[refuse] {module}: incomplete task pool")
            module_rows, module_block_counts = module_sensitivity_rows(
                module=module,
                d_in=d_in,
                subspaces=subspaces,
                basis=basis,
                eigvals=eigvals,
                standard_errors=standard_errors,
                k_values=k_values,
                null_m_values=null_m_values,
                block_z_values=block_z_values,
            )
            writer.writerows(module_rows)
            rows_written += len(module_rows)
            retained_modules.append(module)
            block_counts[module] = module_block_counts
            relative_norms[module] = module_norms
            for row in module_rows:
                k = int(row["k"])
                raw = float(row["raw"])
                isotropic = float(row["null_iso"])
                for null_m, block_z in setting_pairs:
                    aggregates[(k, null_m, block_z)].add(
                        raw,
                        isotropic,
                        float(row[f"null_{setting_tag(null_m, block_z)}"]),
                    )
            print(
                f"[sensitivity] {module_index + 1}/{len(modules)} modules; "
                f"{len(subspaces)} tasks, {len(module_rows)} compact cells",
                flush=True,
            )

    if args.expect_modules and len(retained_modules) != args.expect_modules:
        raise SystemExit(
            f"[refuse] retained {len(retained_modules)} modules, expected {args.expect_modules}; "
            f"partial cells remain at {partial_cells}"
        )
    if not retained_modules:
        raise SystemExit("[refuse] no module produced sensitivity cells")
    if not args.allow_incomplete_pool:
        expected_cells = len(retained_modules) * math.comb(len(task_names), 2) * len(k_values)
        if rows_written != expected_cells:
            raise SystemExit(
                f"[refuse] wrote {rows_written} compact cells, expected {expected_cells}"
            )
        expected_per_setting = len(retained_modules) * math.comb(len(task_names), 2)
        for setting, aggregate in aggregates.items():
            if aggregate.count != expected_per_setting:
                raise SystemExit(
                    f"[refuse] setting {setting} has {aggregate.count} cells, expected "
                    f"{expected_per_setting}"
                )

    summaries = summary_rows(
        aggregates, args.base, len(task_names), len(retained_modules), block_counts
    )
    metadata = {
        "format": "exact_activation_null_full_pool_sensitivity",
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "definition": "all unordered task pairs x all retained modules",
            "weighting": "equal cell; fraction is ratio of summed numerators/denominators",
            "base": public_checkpoint_id(args.base),
            "tasks": task_names,
            "expert_specs": {
                name: public_checkpoint_id(spec) for name, spec in expert_specs.items()
            },
            "requested_modules": modules,
            "retained_modules": retained_modules,
            "n_tasks": len(task_names),
            "n_modules": len(retained_modules),
            "n_compact_cells": rows_written,
            "allow_incomplete_pool": args.allow_incomplete_pool,
        },
        "grid": {
            "k": list(k_values),
            "null_m": list(null_m_values),
            "block_z": list(block_z_values),
            "n_settings": len(k_values) * len(null_m_values) * len(block_z_values),
        },
        "estimand": {
            "null": "(1/k) sum_b tr(P_i P_b) tr(P_j P_b) / dim(P_b)",
            "fraction": "sum(null_exact-null_iso) / sum(raw-null_iso)",
            "monte_carlo_draws": 0,
        },
        "inputs": {
            "cov": portable_path_label(cov_dir),
            "fold_cov": portable_path_label(fold_cov),
            "manifest": portable_path_label(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "key_prefix": args.key_prefix,
            "only_modules": args.only_modules,
            "min_rel_norm": args.min_rel_norm,
            "device": args.device,
        },
        "implementation_sha256": {
            "exact_sensitivity.py": sha256_file(Path(__file__).resolve()),
            "exact_overlap.py": sha256_file(ROOT / "scripts" / "exact_overlap.py"),
            "actnull/null.py": sha256_file(ROOT / "actnull" / "null.py"),
        },
        "cache": {
            "directory": portable_path_label(cache_dir) if cache_dir else None,
            "hits": cache_hits,
            "misses": cache_misses,
            "bypassed_unsafe_dense_checkpoint": cache_bypassed,
            "cached_object": f"top-{max_k} observed right singular frame per task/module",
        },
        "fold_count_by_module": fold_counts,
        "stored_block_count_by_module": block_counts,
        "relative_task_vector_norm_by_module": relative_norms,
        "outputs": {
            "cells": portable_path_label(cells_path),
            "summary": portable_path_label(summary_path),
            "metadata": portable_path_label(metadata_path),
        },
        "uncertainty_note": (
            "This grid is a deterministic conditional sensitivity analysis. It does not add "
            "a confidence interval; bootstrap the compact cells with task and layer clusters "
            "if an uncertainty interval is required."
        ),
    }

    atomic_write_csv(summary_path, summaries, args.overwrite)
    atomic_write_json(metadata_path, metadata, args.overwrite)
    os.replace(partial_cells, cells_path)

    print(f"[sensitivity] wrote {rows_written} compact cells to {cells_path}")
    print(f"[sensitivity] wrote {len(summaries)} settings to {summary_path}")
    print(f"[sensitivity] wrote scope metadata to {metadata_path}")
    for row in summaries:
        if row["k"] == 8 and row["null_m"] == 64 and float(row["block_z"]) == 2.0:
            print(
                f"[sensitivity] reference k=8,m=64,z=2: "
                f"{float(row['fraction_explained_pct']):.3f}% explained"
            )


if __name__ == "__main__":
    main()
