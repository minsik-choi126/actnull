"""Small, dependency-light helpers shared by experiment entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


def parse_named_paths(specifications: Sequence[str]) -> dict[str, str]:
    """Parse ``NAME=PATH`` arguments without interpreting either side."""
    parsed: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"expert must be NAME=PATH, got {specification!r}")
        name, path = specification.split("=", 1)
        if not name or not path:
            raise ValueError(f"expert must be NAME=PATH, got {specification!r}")
        if name in parsed:
            raise ValueError(f"duplicate expert name {name!r}")
        parsed[name] = path
    if len(parsed) < 2:
        raise ValueError("at least two experts are required")
    return parsed


def parse_csv_strings(value: str, defaults: Sequence[str] = ()) -> tuple[str, ...]:
    """Return a nonempty, order-preserving comma-separated string list."""
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    return values or tuple(defaults)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Return unique comma-separated integers in the order supplied."""
    parsed: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        number = int(item)
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise ValueError("expected at least one integer")
    return tuple(parsed)


def covariance_directories(
    *,
    arch: str,
    work_root: str | Path | None,
    covariance: str | Path | None,
    fold_covariance: str | Path | None,
) -> tuple[Path, Path]:
    """Resolve explicit covariance paths, or derive both from a work root."""
    root = Path(work_root).expanduser() if work_root else None
    cov = Path(covariance).expanduser() if covariance else None
    folds = Path(fold_covariance).expanduser() if fold_covariance else None
    if cov is None:
        if root is None:
            raise ValueError("pass --cov or --work-root")
        cov = root / f"k1000_cov_{arch}"
    if folds is None:
        if root is None:
            raise ValueError("pass --fold-cov or --work-root")
        folds = root / f"k1000_fold_{arch}"
    if not (cov / "manifest.json").is_file():
        raise FileNotFoundError(f"missing covariance manifest: {cov / 'manifest.json'}")
    if not folds.is_dir():
        raise FileNotFoundError(f"missing fold covariance directory: {folds}")
    return cov, folds


def fold_eigenvalues(
    fold_covariance: str | Path,
    module: str,
    *,
    device: str | torch.device = "cpu",
    minimum: int = 4,
) -> list[torch.Tensor]:
    """Load the available fold eigenvalue vectors for one module."""
    filename = f"{module.replace('.', '__')}.pt"
    values = []
    for directory in sorted(Path(fold_covariance).glob("f*")):
        path = directory / filename
        if path.is_file():
            values.append(
                torch.load(path, map_location=device, weights_only=False)["eigvals"].double()
            )
    if len(values) < minimum:
        raise ValueError(
            f"{module}: found {len(values)} fold covariances under {fold_covariance}; "
            f"need at least {minimum}"
        )
    return values


def write_json_new(path: str | Path, payload: Mapping | Sequence) -> Path:
    """Write JSON while refusing to replace an existing result."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing result: {target}; choose a new --out"
        ) from error
    return target


def public_sources(experts: Mapping[str, str]) -> list[str]:
    """Record pool labels only; local checkpoint paths do not belong in artifacts."""
    return sorted(experts)
