#!/usr/bin/env python3
"""Crossed task-by-layer bootstrap for PRISM overlap estimates.

For a row belonging to task pair ``(a,b)`` and encoder layer ``l``, one
crossed (pigeonhole) bootstrap replicate assigns weight

    w_(a,b,l) = C_a C_b L_l,

where ``C`` and ``L`` are independent multinomial multiplicities obtained by
resampling the observed task endpoints and encoder layers, respectively. All
module types in a selected encoder layer are retained. The task-only and
layer-only margins use ``C_a C_b`` and ``L_l`` and are reported for comparison.

Within every replicate, rows are averaged before computing

    reproduced (%) = 100 * (mean(selected_null) - mean(isotropic))
                           / (mean(raw) - mean(isotropic))
    residual       = mean(raw - selected_null).

Thus this bootstraps the reported ratio-of-means estimator, not a mean of
row-wise ratios. Percentile intervals condition on the realised rows and
selected per-row null values. "Crossed" means that both clustering dimensions are
resampled in the same replicate; it does not claim simultaneous family-wise
coverage for the two reported statistics.

Inputs can be bare paths or ``LABEL=PATH``. ``null_exact`` is preferred when
``--null-column auto`` (the default), with ``null_act_raw`` retained only for
backward-compatible auditing of older CSVs.

Examples:
  python analysis/overlap_bootstrap.py B16=results/v7_exact_b16.csv
  python analysis/overlap_bootstrap.py B16=a.csv B32=b.csv \
      --draws 50000 --seed 20260915 --output bootstrap.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_SEED = 20_260_915
DEFAULT_DRAWS = 50_000
DEFAULT_LAYER_REGEX = r"encoder\.layers\.(\d+)"
SCHEMES = ("joint_task_layer", "task_only", "layer_only")
EXACT_FRACTION_METRIC = "exact_reproduced_fraction_percent"
FITTED_FRACTION_METRIC = "reproduced_fraction_percent"
# Backward-compatible public constant used by the exact-null regression tests.
METRICS = (EXACT_FRACTION_METRIC, "residual")


class DataError(ValueError):
    """Raised when an input CSV cannot support the requested estimator."""


@dataclass(frozen=True)
class BootstrapData:
    """Cell-aggregated data and cluster indices for one CSV."""

    label: str
    source: str
    sha256: str
    null_column: str
    null_kinds: tuple[str, ...]
    tasks: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    layers: tuple[int, ...]
    modules: tuple[str, ...]
    cell_pair: np.ndarray
    cell_layer: np.ndarray
    # Columns are sums of raw, isotropic, fitted/exact null, and row count.
    cell_sums: np.ndarray
    n_rows: int
    n_unique_cells: int
    n_observed_pair_module_cells: int
    n_expected_pair_module_cells: int


def parse_input(spec: str) -> tuple[str, Path]:
    """Parse PATH or LABEL=PATH without requiring labels for single inputs."""

    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if not label or not raw_path:
            raise DataError(f"invalid labelled input {spec!r}; expected LABEL=PATH")
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_csv(path: Path):
    """Open plain or gzip-compressed CSV input in text mode."""

    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="")
    return path.open(newline="")


def _as_float(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"row {row_number}: invalid {column!r}") from exc
    if not math.isfinite(value):
        raise DataError(f"row {row_number}: non-finite {column!r}")
    return value


def _close(got: float, want: float, *, atol: float = 2e-8) -> bool:
    return math.isclose(got, want, rel_tol=2e-7, abs_tol=atol)


def load_csv(
    path: Path | str,
    *,
    label: str | None = None,
    null_column: str = "auto",
    layer_regex: str = DEFAULT_LAYER_REGEX,
    validate_exact_fields: bool = True,
) -> BootstrapData:
    """Read and aggregate a result CSV by (task pair, encoder layer)."""

    path = Path(path)
    if not path.is_file():
        raise DataError(f"input does not exist: {path}")
    try:
        layer_pattern = re.compile(layer_regex)
    except re.error as exc:
        raise DataError(f"invalid --layer-regex: {exc}") from exc

    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        base_required = {"module", "a", "b", "raw", "null_iso"}
        missing = base_required.difference(fieldnames)
        if missing:
            raise DataError(f"{path}: missing columns {sorted(missing)}")
        if null_column == "auto":
            if "null_exact" in fieldnames:
                selected_null = "null_exact"
            elif "null_act_raw" in fieldnames:
                selected_null = "null_act_raw"
            else:
                raise DataError(
                    f"{path}: neither 'null_exact' nor 'null_act_raw' is present"
                )
        else:
            selected_null = null_column
            if selected_null not in fieldnames:
                raise DataError(f"{path}: missing null column {selected_null!r}")
        rows = list(reader)

    if not rows:
        raise DataError(f"{path}: empty CSV")

    parsed: list[tuple[tuple[str, str], int, str, float, float, float]] = []
    null_kinds: set[str] = set()
    tasks: set[str] = set()
    modules: set[str] = set()
    pair_modules: set[tuple[tuple[str, str], str]] = set()
    for row_number, row in enumerate(rows, start=2):
        a, b = row["a"], row["b"]
        if not a or not b:
            raise DataError(f"row {row_number}: empty task endpoint")
        if a == b:
            raise DataError(f"row {row_number}: self-pair {a!r} is unsupported")
        pair = tuple(sorted((a, b)))
        match = layer_pattern.search(row["module"])
        if match is None:
            raise DataError(
                f"row {row_number}: cannot extract layer from {row['module']!r}"
            )
        layer = int(match.group(1))
        raw = _as_float(row, "raw", row_number)
        iso = _as_float(row, "null_iso", row_number)
        null = _as_float(row, selected_null, row_number)

        if validate_exact_fields and selected_null == "null_exact":
            optional_checks = (
                ("frac_num", null - iso),
                ("frac_den", raw - iso),
                ("excess_exact", raw - null),
            )
            for column, expected in optional_checks:
                if column in row and row[column] not in (None, ""):
                    got = _as_float(row, column, row_number)
                    if not _close(got, expected):
                        raise DataError(
                            f"row {row_number}: {column}={got} is inconsistent "
                            f"with recomputed value {expected}"
                        )
            if "frac_exact" in row and row["frac_exact"] not in (None, ""):
                denominator = raw - iso
                if abs(denominator) > 1e-14:
                    got = _as_float(row, "frac_exact", row_number)
                    expected = (null - iso) / denominator
                    if not _close(got, expected):
                        raise DataError(
                            f"row {row_number}: frac_exact={got} is inconsistent "
                            f"with recomputed value {expected}"
                        )

        parsed.append((pair, layer, row["module"], raw, iso, null))
        tasks.update(pair)
        modules.add(row["module"])
        pair_modules.add((pair, row["module"]))
        if row.get("null_kind"):
            null_kinds.add(row["null_kind"])

    task_tuple = tuple(sorted(tasks))
    pair_tuple = tuple(sorted({item[0] for item in parsed}))
    layer_tuple = tuple(sorted({item[1] for item in parsed}))
    module_tuple = tuple(sorted(modules))
    pair_index = {pair: i for i, pair in enumerate(pair_tuple)}
    layer_index = {layer: i for i, layer in enumerate(layer_tuple)}

    # Aggregate module types within each pair x encoder-layer cell. They share
    # the same crossed-bootstrap weight but each still contributes one row.
    cells: dict[tuple[int, int], np.ndarray] = {}
    for pair, layer, _module, raw, iso, null in parsed:
        key = (pair_index[pair], layer_index[layer])
        if key not in cells:
            cells[key] = np.zeros(4, dtype=np.float64)
        cells[key] += (raw, iso, null, 1.0)

    keys = sorted(cells)
    cell_pair = np.asarray([key[0] for key in keys], dtype=np.int64)
    cell_layer = np.asarray([key[1] for key in keys], dtype=np.int64)
    cell_sums = np.stack([cells[key] for key in keys])

    return BootstrapData(
        label=label or path.stem,
        source=str(path),
        sha256=_sha256(path),
        null_column=selected_null,
        null_kinds=tuple(sorted(null_kinds)),
        tasks=task_tuple,
        pairs=pair_tuple,
        layers=layer_tuple,
        modules=module_tuple,
        cell_pair=cell_pair,
        cell_layer=cell_layer,
        cell_sums=cell_sums,
        n_rows=len(parsed),
        n_unique_cells=len(keys),
        n_observed_pair_module_cells=len(pair_modules),
        n_expected_pair_module_cells=len(pair_tuple) * len(module_tuple),
    )


def _pair_endpoint_indices(data: BootstrapData) -> tuple[np.ndarray, np.ndarray]:
    task_index = {task: i for i, task in enumerate(data.tasks)}
    return (
        np.asarray([task_index[a] for a, _ in data.pairs], dtype=np.int64),
        np.asarray([task_index[b] for _, b in data.pairs], dtype=np.int64),
    )


def cell_weights(
    data: BootstrapData,
    task_counts: np.ndarray,
    layer_counts: np.ndarray,
    scheme: str,
) -> np.ndarray:
    """Return row-cell weights for supplied multiplicity matrices.

    ``task_counts`` and ``layer_counts`` have shape ``(replicates, clusters)``.
    This public helper makes the precise resampling rule directly testable.
    """

    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    task_counts = np.asarray(task_counts)
    layer_counts = np.asarray(layer_counts)
    if task_counts.ndim != 2 or task_counts.shape[1] != len(data.tasks):
        raise ValueError("task_counts has the wrong shape")
    if layer_counts.ndim != 2 or layer_counts.shape[1] != len(data.layers):
        raise ValueError("layer_counts has the wrong shape")
    if task_counts.shape[0] != layer_counts.shape[0]:
        raise ValueError("task_counts and layer_counts use different replicate counts")

    left, right = _pair_endpoint_indices(data)
    pair_weight = task_counts[:, left] * task_counts[:, right]
    if scheme == "joint_task_layer":
        return pair_weight[:, data.cell_pair] * layer_counts[:, data.cell_layer]
    if scheme == "task_only":
        return pair_weight[:, data.cell_pair]
    return layer_counts[:, data.cell_layer]


def statistics_from_weights(
    weights: np.ndarray,
    cell_sums: np.ndarray,
    *,
    denominator_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ratio-of-means statistics and a validity mask by replicate."""

    # Some NumPy/Accelerate builds emit spurious floating-point matmul warnings
    # even when these small nonnegative weights yield finite outputs. We always
    # apply an explicit finite mask below, so suppress the backend warning here.
    with np.errstate(all="ignore"):
        weighted = np.asarray(weights, dtype=np.float64) @ cell_sums
    n_rows = weighted[:, 3]
    fraction_denominator = weighted[:, 0] - weighted[:, 1]
    valid = (
        np.isfinite(weighted).all(axis=1)
        & (n_rows > 0)
        & (np.abs(fraction_denominator) > denominator_eps)
    )
    values = np.full((weighted.shape[0], 2), np.nan, dtype=np.float64)
    values[valid, 0] = (
        100.0
        * (weighted[valid, 2] - weighted[valid, 1])
        / fraction_denominator[valid]
    )
    values[valid, 1] = (
        weighted[valid, 0] - weighted[valid, 2]
    ) / n_rows[valid]
    valid &= np.isfinite(values).all(axis=1)
    return values, valid


def point_estimate(
    data: BootstrapData, *, denominator_eps: float = 1e-12
) -> np.ndarray:
    weights = np.ones((1, data.n_unique_cells), dtype=np.float64)
    values, valid = statistics_from_weights(
        weights, data.cell_sums, denominator_eps=denominator_eps
    )
    if not valid[0]:
        raise DataError(
            f"{data.source}: raw-minus-isotropic denominator is zero or non-finite"
        )
    return values[0]


def metric_keys(data: BootstrapData) -> tuple[str, str]:
    """Name the fraction honestly for analytic versus sampled fitted nulls."""

    fraction = (
        EXACT_FRACTION_METRIC
        if data.null_column == "null_exact"
        else FITTED_FRACTION_METRIC
    )
    return fraction, "residual"


def _append_valid(
    chunks: list[np.ndarray],
    values: np.ndarray,
    valid: np.ndarray,
    remaining: int,
) -> tuple[int, int]:
    """Append up to remaining valid rows and return attempts and rejections used."""

    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return len(valid), len(valid)
    chosen = indices[:remaining]
    chunks.append(values[chosen])
    last = int(chosen[-1])
    attempted = last + 1
    rejected = attempted - len(chosen)
    return attempted, rejected


def bootstrap(
    data: BootstrapData,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    batch_size: int = 512,
    denominator_eps: float = 1e-12,
) -> dict[str, object]:
    """Run crossed and one-way percentile bootstraps for one dataset."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(data.tasks) < 2:
        raise DataError("at least two tasks are required")
    if not data.layers:
        raise DataError("at least one encoder layer is required")

    point = point_estimate(data, denominator_eps=denominator_eps)
    metrics = metric_keys(data)
    rng = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {scheme: [] for scheme in SCHEMES}
    accepted = {scheme: 0 for scheme in SCHEMES}
    attempted = {scheme: 0 for scheme in SCHEMES}
    rejected = {scheme: 0 for scheme in SCHEMES}

    task_probability = np.full(len(data.tasks), 1.0 / len(data.tasks))
    layer_probability = np.full(len(data.layers), 1.0 / len(data.layers))
    while min(accepted.values()) < draws:
        unfinished = [scheme for scheme in SCHEMES if accepted[scheme] < draws]
        needed = max(draws - accepted[scheme] for scheme in unfinished)
        size = min(batch_size, max(1, needed))
        task_counts = rng.multinomial(
            len(data.tasks), task_probability, size=size
        )
        layer_counts = rng.multinomial(
            len(data.layers), layer_probability, size=size
        )
        for scheme in unfinished:
            weights = cell_weights(data, task_counts, layer_counts, scheme)
            values, valid = statistics_from_weights(
                weights, data.cell_sums, denominator_eps=denominator_eps
            )
            use_attempted, use_rejected = _append_valid(
                chunks[scheme], values, valid, draws - accepted[scheme]
            )
            attempted[scheme] += use_attempted
            rejected[scheme] += use_rejected
            accepted[scheme] += use_attempted - use_rejected

        if max(attempted.values()) > max(100_000, 100 * draws):
            raise DataError(
                "too many invalid bootstrap replicates; inspect the ratio denominator"
            )

    replicates = {
        scheme: np.concatenate(chunks[scheme], axis=0)[:draws]
        for scheme in SCHEMES
    }
    intervals = {
        scheme: np.quantile(values, (0.025, 0.975), axis=0, method="linear")
        for scheme, values in replicates.items()
    }
    legacy_envelope = np.stack(
        (
            np.minimum(intervals["task_only"][0], intervals["layer_only"][0]),
            np.maximum(intervals["task_only"][1], intervals["layer_only"][1]),
        )
    )

    def interval_object(bounds: np.ndarray) -> dict[str, list[float]]:
        return {
            metric: [float(bounds[0, i]), float(bounds[1, i])]
            for i, metric in enumerate(metrics)
        }

    interval_json = {
        scheme: interval_object(bounds) for scheme, bounds in intervals.items()
    }
    interval_json["legacy_one_way_envelope"] = interval_object(legacy_envelope)

    widths = {
        scheme: intervals[scheme][1] - intervals[scheme][0]
        for scheme in SCHEMES
    }
    envelope_width = legacy_envelope[1] - legacy_envelope[0]
    comparison: dict[str, dict[str, float | None]] = {}
    for metric_index, metric in enumerate(metrics):
        joint_width = widths["joint_task_layer"][metric_index]

        def ratio(denominator: float) -> float | None:
            return float(joint_width / denominator) if denominator > 0 else None

        comparison[metric] = {
            "joint_width": float(joint_width),
            "task_only_width": float(widths["task_only"][metric_index]),
            "layer_only_width": float(widths["layer_only"][metric_index]),
            "legacy_envelope_width": float(envelope_width[metric_index]),
            "joint_to_task_only_width_ratio": ratio(
                widths["task_only"][metric_index]
            ),
            "joint_to_layer_only_width_ratio": ratio(
                widths["layer_only"][metric_index]
            ),
            "joint_to_legacy_envelope_width_ratio": ratio(
                envelope_width[metric_index]
            ),
        }

    return {
        "label": data.label,
        "input": data.source,
        "sha256": data.sha256,
        "null_column": data.null_column,
        "null_kinds": list(data.null_kinds),
        "metric_keys": {"fraction": metrics[0], "residual": metrics[1]},
        "sample": {
            "rows": data.n_rows,
            "tasks": len(data.tasks),
            "task_pairs": len(data.pairs),
            "layers": len(data.layers),
            "modules": len(data.modules),
            "pair_layer_cells": data.n_unique_cells,
            "balanced_pair_module_grid": (
                data.n_rows == data.n_expected_pair_module_cells
                and data.n_observed_pair_module_cells
                == data.n_expected_pair_module_cells
            ),
        },
        "point_estimate": {
            metric: float(point[i]) for i, metric in enumerate(metrics)
        },
        "percentile_95_intervals": interval_json,
        "one_way_comparison": comparison,
        "replicates": {
            scheme: {
                "valid": draws,
                "attempted": attempted[scheme],
                "rejected": rejected[scheme],
            }
            for scheme in SCHEMES
        },
    }


def make_report(
    results: Sequence[dict[str, object]],
    *,
    draws: int,
    seed: int,
    denominator_eps: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": {
            "name": "crossed task-endpoint by encoder-layer pigeonhole bootstrap",
            "seed_reset_for_each_input": seed,
            "valid_replicates_per_scheme": draws,
            "task_resampling": (
                "sample T endpoints with replacement from the T observed tasks; "
                "row (a,b) receives endpoint weight C_a*C_b"
            ),
            "layer_resampling": (
                "sample L encoder-layer blocks with replacement from the L "
                "observed layers; all module types in layer l receive weight L_l"
            ),
            "crossed_weight": "C_a*C_b*L_l",
            "point_and_replicate_aggregation": (
                "weighted rows are averaged before taking the reproduced-fraction ratio"
            ),
            "fraction": "100*(mean(null)-mean(isotropic))/(mean(raw)-mean(isotropic))",
            "residual": "mean(raw-null)",
            "interval": "2.5th and 97.5th percentiles (NumPy linear quantiles)",
            "ratio_denominator_absolute_epsilon": denominator_eps,
            "conditioning": (
                "conditions on realised task-pair rows, modules, activation "
                "calibration, and selected per-row null values; null_exact is "
                "analytic, whereas sampled fitted-null columns are realised "
                "Monte Carlo means"
            ),
            "coverage_note": (
                "crossed/joint refers to cluster dimensions, not family-wise "
                "simultaneous coverage across the two metrics"
            ),
        },
        "datasets": list(results),
    }


def _fmt_interval(bounds: Iterable[float], suffix: str = "") -> str:
    lo, hi = bounds
    return f"[{lo:.6f}, {hi:.6f}]{suffix}"


def print_result(result: dict[str, object]) -> None:
    point = result["point_estimate"]
    intervals = result["percentile_95_intervals"]
    sample = result["sample"]
    keys = result["metric_keys"]
    fraction_metric, residual_metric = keys["fraction"], keys["residual"]
    print(
        f"{result['label']}: rows={sample['rows']} tasks={sample['tasks']} "
        f"pairs={sample['task_pairs']} layers={sample['layers']} "
        f"modules={sample['modules']} null={result['null_column']}"
    )
    print(
        "  point: fraction="
        f"{point[fraction_metric]:.6f}% residual={point[residual_metric]:+.8f}"
    )
    for scheme in (*SCHEMES, "legacy_one_way_envelope"):
        bounds = intervals[scheme]
        print(
            f"  {scheme:25s} fraction="
            f"{_fmt_interval(bounds[fraction_metric], '%')} residual="
            f"{_fmt_interval(bounds[residual_metric])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crossed task-by-layer bootstrap for overlap CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="[LABEL=]CSV", help="one or more result CSVs"
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--null-column",
        default="auto",
        help="column used as the fitted null; auto prefers null_exact",
    )
    parser.add_argument("--layer-regex", default=DEFAULT_LAYER_REGEX)
    parser.add_argument("--denominator-eps", type=float, default=1e-12)
    parser.add_argument(
        "--skip-exact-field-validation",
        action="store_true",
        help="do not cross-check frac_num/frac_den/excess_exact/frac_exact",
    )
    parser.add_argument(
        "--output", type=Path, help="write the complete machine-readable JSON report"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.denominator_eps < 0:
        raise SystemExit("--denominator-eps must be nonnegative")

    results = []
    labels: set[str] = set()
    try:
        for spec in args.inputs:
            label, path = parse_input(spec)
            if label in labels:
                raise DataError(f"duplicate input label {label!r}")
            labels.add(label)
            data = load_csv(
                path,
                label=label,
                null_column=args.null_column,
                layer_regex=args.layer_regex,
                validate_exact_fields=not args.skip_exact_field_validation,
            )
            result = bootstrap(
                data,
                draws=args.draws,
                seed=args.seed,
                batch_size=args.batch_size,
                denominator_eps=args.denominator_eps,
            )
            results.append(result)
            print_result(result)
    except (DataError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = make_report(
        results,
        draws=args.draws,
        seed=args.seed,
        denominator_eps=args.denominator_eps,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
