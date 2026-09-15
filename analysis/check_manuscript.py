#!/usr/bin/env python3
"""Run the v7 artifact audit and check semantic claims in the live manuscript.

This is deliberately not a global "does this number occur somewhere?" checker.
Every enforced claim is tied to a named artifact-derived metric and distinctive
manuscript context.  The paper entry point is mandatory; its actual ``\input``
and ``\include`` graph is resolved and printed before checking.

Usage:

    python3 analysis/check_manuscript.py \
        --paper iclr2027_conference.tex --artifact artifact

The semantic registry covers the current headline and table claims backed by
the compact v7 artifact.  A conservative heuristic separately reports
result-like TeX literals outside the registry.  Those are coverage warnings,
not a claim that every paper number is already backed by this artifact.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics as stats
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Metric:
    value: float
    source: str


class MetricStore:
    """Flat semantic metric namespace with provenance and duplicate protection."""

    def __init__(self) -> None:
        self._items: dict[str, Metric] = {}

    def add(self, key: str, value: float | int, source: str) -> None:
        if key in self._items:
            raise ValueError(f"duplicate semantic metric {key!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite semantic metric {key!r}: {number}")
        self._items[key] = Metric(number, source)

    def value(self, key: str) -> float:
        try:
            return self._items[key].value
        except KeyError as exc:
            raise KeyError(f"semantic metric {key!r} was not materialised") from exc

    def source(self, key: str) -> str:
        return self._items[key].source

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class Document:
    path: Path
    relative: str
    text: str
    normalized: str


@dataclass(frozen=True)
class Claim:
    claim_id: str
    file: str
    fragment: str
    count: int = 1


def _strip_tex_comments(text: str) -> str:
    """Strip unescaped TeX comments while retaining line boundaries."""
    cleaned = []
    for line in text.splitlines():
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        cleaned.append(line[:cut])
    return "\n".join(cleaned)


def _normalize_tex(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_tex_comments(text)).strip()


_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")


def resolve_input_graph(entrypoint: Path) -> tuple[Path, dict[str, Document]]:
    """Resolve the TeX include graph, confined to the entrypoint directory."""
    paper = entrypoint.expanduser().resolve()
    if not paper.is_file():
        raise FileNotFoundError(f"paper entrypoint does not exist: {paper}")
    root = paper.parent
    documents: dict[str, Document] = {}
    visiting: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"TeX input escapes paper root {root}: {resolved}") from exc
        if resolved in visiting:
            raise ValueError(f"cyclic TeX input involving {resolved}")
        if relative in documents:
            return
        if not resolved.is_file():
            raise FileNotFoundError(f"missing TeX input: {resolved}")
        visiting.add(resolved)
        text = resolved.read_text(encoding="utf-8")
        stripped = _strip_tex_comments(text)
        documents[relative] = Document(
            path=resolved,
            relative=relative,
            text=text,
            normalized=_normalize_tex(text),
        )
        for match in _INPUT_RE.finditer(stripped):
            raw = match.group(1).strip()
            if not raw or "\\" in raw:
                raise ValueError(
                    f"dynamic TeX input is not auditable in {relative}: {raw!r}"
                )
            candidate = root / raw
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            if not candidate.exists():
                nested = resolved.parent / raw
                if nested.suffix == "":
                    nested = nested.with_suffix(".tex")
                candidate = nested
            visit(candidate)
        visiting.remove(resolved)

    visit(paper)
    return root, documents


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"cannot average empty column {key!r}")
    return stats.fmean(values)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = stats.fmean(left)
    right_mean = stats.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    if denominator == 0.0:
        raise ValueError("cannot correlate a constant vector")
    return numerator / denominator


def build_metrics(artifact: Path) -> MetricStore:
    """Materialise manuscript-facing metrics from artifact records, not prose."""
    root = artifact.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    results = root / "results"
    metrics = MetricStore()
    arm_rows: dict[str, list[dict[str, str]]] = {}
    arms = {
        "B16": ("v7_exact_clip-vit-base-patch16.csv.gz", "null_exact"),
        "B32": ("v7_exact_clip-vit-base-patch32.csv.gz", "null_exact"),
        "L14": ("v7_exact_clip-vit-large-patch14.csv.gz", "null_exact"),
        "FT8": ("v7_exact_b16_8task_q_v_fullft.csv.gz", "null_exact"),
        "LoRA16": (
            "v7_mc64_b16_8task_q_v_lora-16_loranull.csv",
            "null_act_raw",
        ),
        "LinearizedLoRA16": (
            "v7_mc64_b16_8task_q_v_l-lora-16_loranull.csv",
            "null_act_raw",
        ),
    }
    for label, (filename, null_key) in arms.items():
        source = f"results/{filename}"
        rows = _read_csv(results / filename)
        arm_rows[label] = rows
        raw = _mean(rows, "raw")
        iso = _mean(rows, "null_iso")
        fitted = _mean(rows, null_key)
        residual = raw - fitted
        fraction = 100.0 * (fitted - iso) / (raw - iso)
        for suffix, value in (
            ("raw", raw),
            ("iso", iso),
            ("null", fitted),
            ("residual", residual),
            ("fraction", fraction),
            ("residual_fraction", 100.0 - fraction),
        ):
            metrics.add(f"arm.{label}.{suffix}", value, source)
        metrics.add(
            f"arm.{label}.modules",
            len({row["module"] for row in rows}),
            source,
        )

    role_types = {
        "residual": {"q_proj", "k_proj", "v_proj", "fc1"},
        "internal": {"out_proj", "fc2"},
    }
    for role, suffixes in role_types.items():
        rows = [
            row
            for row in arm_rows["B16"]
            if row["module"].split(".")[-1] in suffixes
        ]
        raw = _mean(rows, "raw")
        iso = _mean(rows, "null_iso")
        fitted = _mean(rows, "null_exact")
        metrics.add(
            f"role.B16.{role}.fraction",
            100.0 * (fitted - iso) / (raw - iso),
            "results/v7_exact_clip-vit-base-patch16.csv.gz",
        )

    for label in ("B16", "B32"):
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in arm_rows[label]:
            pair = tuple(sorted((row["a"], row["b"])))
            grouped.setdefault(pair, []).append(row)
        pairs = sorted(grouped)
        raw = [_mean(grouped[pair], "raw") for pair in pairs]
        residual = [_mean(grouped[pair], "excess_exact") for pair in pairs]
        rho = _pearson(_average_ranks(raw), _average_ranks(residual))
        raw_top = set(
            sorted(range(len(raw)), key=raw.__getitem__, reverse=True)[:20]
        )
        residual_top = set(
            sorted(range(len(residual)), key=residual.__getitem__, reverse=True)[:20]
        )
        source = f"results/{arms[label][0]}"
        metrics.add(f"rank.{label}.spearman", rho, source)
        metrics.add(
            f"rank.{label}.shared_top20", len(raw_top & residual_top), source
        )

    for filename in (
        "bootstrap_exact_joint_50000.json",
        "bootstrap_lora_joint_50000.json",
    ):
        report = json.loads((root / filename).read_text())
        for entry in report["datasets"]:
            label = entry["label"]
            metric_key = entry["metric_keys"]["fraction"]
            interval = entry["percentile_95_intervals"]["joint_task_layer"]
            for suffix, value in zip(("lo", "hi"), interval[metric_key]):
                metrics.add(
                    f"bootstrap.{label}.fraction.{suffix}", value, filename
                )
            for suffix, value in zip(("lo", "hi"), interval["residual"]):
                metrics.add(
                    f"bootstrap.{label}.residual.{suffix}", value, filename
                )

    power_files = {
        "activation.current": "v7_power_activation_current.json",
        "activation.initial": "v7_power_activation_initial.json",
        "lora.current": "v7_power_lora_current.json",
        "lora.initial": "v7_power_lora_initial.json",
    }
    for family, filename in power_files.items():
        source = f"results/{filename}"
        report = json.loads((results / filename).read_text())
        for row in report["rows"]:
            planted = int(row["planted"])
            for field in (
                "raw",
                "null",
                "excess",
                "truth",
                "delivered",
                "recovery",
            ):
                if row.get(field) is not None:
                    metrics.add(
                        f"power.{family}.s{planted}.{field}", row[field], source
                    )

    ablation_source = "results/v7_ablation_level_centered.json"
    variant_keys = {
        "initial (neither correction)": "neither",
        "singleton signs only": "singleton",
        "full complement only": "complement",
        "corrected (both)": "both",
    }
    for row in json.loads((root / ablation_source).read_text()):
        metrics.add(
            f"ablation.{variant_keys[row['variant']]}.recovery",
            row["recovery"],
            ablation_source,
        )

    baseline_source = "results/v7_baseline_levels_corrected_haar.json"
    baseline = json.loads((root / baseline_source).read_text())
    for key, value in baseline["level"].items():
        metrics.add(f"baseline.{key}.excess", value, baseline_source)
    metrics.add(
        "baseline.raw", baseline["metadata"]["raw_null_pair_overlap"], baseline_source
    )
    metrics.add(
        "baseline.raw_over_chance",
        baseline["metadata"]["raw_overlap_over_isotropic_chance"],
        baseline_source,
    )

    projector_files = {
        "FT": "v7_projector_b16_fullft.csv",
        "LoRA": "v7_projector_b16_lora16.csv",
    }
    for family, filename in projector_files.items():
        source = f"results/{filename}"
        rows = _read_csv(results / filename)
        for field in (
            "projector_survives_null",
            "chance",
            "projector_in_top_activation",
            "activation_chance",
        ):
            metrics.add(f"projector.{family}.{field}", _mean(rows, field), source)

    merge_source = "results/v7_merge_b32_accuracy.json"
    merge = json.loads((root / merge_source).read_text())
    for method, by_task in merge.items():
        metrics.add(
            f"merge.{method}.mean", stats.fmean(by_task.values()), merge_source
        )

    sensitivity_files = {
        "B16": "v7_exact_sensitivity_clip-vit-base-patch16_summary.csv",
        "B32": "v7_exact_sensitivity_clip-vit-base-patch32_summary.csv",
        "L14": "v7_exact_sensitivity_clip-vit-large-patch14_summary.csv",
    }
    maximum_z_spread = 0.0
    for label, filename in sensitivity_files.items():
        source = f"results/{filename}"
        rows = _read_csv(results / filename)
        values = [float(row["fraction_explained_pct"]) for row in rows]
        k8 = [
            float(row["fraction_explained_pct"])
            for row in rows
            if int(row["k"]) == 8
        ]
        metrics.add(f"sensitivity.{label}.all.lo", min(values), source)
        metrics.add(f"sensitivity.{label}.all.hi", max(values), source)
        metrics.add(f"sensitivity.{label}.k8.lo", min(k8), source)
        metrics.add(f"sensitivity.{label}.k8.hi", max(k8), source)
        grouped: dict[tuple[int, int], list[float]] = {}
        for row in rows:
            grouped.setdefault(
                (int(row["k"]), int(row["null_m"])), []
            ).append(float(row["fraction_explained_pct"]))
        maximum_z_spread = max(
            maximum_z_spread,
            max(max(group) - min(group) for group in grouped.values()),
        )
    metrics.add(
        "sensitivity.max_z_spread",
        maximum_z_spread,
        "three v7_exact_sensitivity_*_summary.csv files",
    )

    h0_files = {
        "FT8": "v7_b16_8task_q_v_fullft_mc_h0.csv.gz",
        "LoRA": "v7_b16_8task_q_v_lora16_actnull_h0.csv.gz",
        "LinearizedLoRA": "v7_b16_8task_q_v_llora16_actnull_h0.csv.gz",
    }
    for family, filename in h0_files.items():
        source = f"results/{filename}"
        rows = _read_csv(results / filename)
        metrics.add(f"h0.{family}", _mean(rows, "h0_excess"), source)
        if family == "LoRA":
            raw = _mean(rows, "raw")
            iso = _mean(rows, "null_iso")
            fitted = _mean(rows, "null_act_raw")
            metrics.add("mismatch.LoRA.residual", raw - fitted, source)
            metrics.add(
                "mismatch.LoRA.residual_fraction",
                100.0 * (raw - fitted) / (raw - iso),
                source,
            )

    refit_source = "results/calibration_resample_b16/summary.json"
    refit = json.loads((root / refit_source).read_text())
    for seed, entry in refit["bootstrap_refits"].items():
        metrics.add(
            f"calibration.refit.{seed}.fraction",
            entry["headline"]["activation_explained_percent"],
            refit_source,
        )
    metrics.add(
        "calibration.refit.max_delta",
        refit["observed_robustness"][
            "maximum_absolute_change_from_reference_percentage_points"
        ],
        refit_source,
    )

    composition_source = "results/calibration_composition_b16/summary.json"
    composition = json.loads((root / composition_source).read_text())
    metrics.add(
        "calibration.composition.fraction",
        composition["headline"]["activation_explained_percent"],
        composition_source,
    )
    metrics.add(
        "calibration.composition.delta",
        composition["reference"]["change_percentage_points"],
        composition_source,
    )
    metrics.add(
        "calibration.composition.null",
        composition["headline"]["mean_null_activation"],
        composition_source,
    )
    metrics.add(
        "calibration.composition.residual",
        composition["headline"]["mean_residual"],
        composition_source,
    )
    intervals = composition[
        "conditional_crossed_task_layer_percentile_95_interval"
    ]
    for suffix, value in zip(
        ("lo", "hi"), intervals["exact_reproduced_fraction_percent"]
    ):
        metrics.add(
            f"calibration.composition.fraction_ci.{suffix}",
            value,
            composition_source,
        )
    for suffix, value in zip(("lo", "hi"), intervals["residual"]):
        metrics.add(
            f"calibration.composition.residual_ci.{suffix}",
            value,
            composition_source,
        )
    metrics.add(
        "calibration.composition.pr_median",
        composition["covariance"]["full"]["participation_ratio"]["median"],
        composition_source,
    )
    metrics.add(
        "calibration.composition.modules_below64",
        composition["covariance"]["full"]["participation_ratio"][
            "modules_below_64"
        ],
        composition_source,
    )

    subgroup_source = "results/v7_calibration_source_subgroup_diagnostic.json"
    subgroup = json.loads((root / subgroup_source).read_text())
    subgroup_labels = {
        "ViT-B/16": "B16",
        "ViT-B/32": "B32",
        "ViT-L/14 q/v": "L14",
    }
    for manuscript_label, key in subgroup_labels.items():
        groups = subgroup["architectures"][manuscript_label]["groups"]
        for endpoints in ("0", "1", "2"):
            metrics.add(
                f"calibration.subgroup.{key}.{endpoints}.fraction",
                groups[endpoints]["reproduced_fraction_percent"],
                subgroup_source,
            )

    factor_source = "results/v7_lora_factor_audit.json"
    factor = json.loads((root / factor_source).read_text())
    factor_labels = {
        "rank16_lora": "LoRA",
        "rank16_linearized_lora": "LinearizedLoRA",
    }
    for family in factor["families"]:
        label = factor_labels[family["family"]]
        aggregate = family["primary_all_24_modules"]["aggregate"]
        for field in ("A_right", "B_left", "BA_right"):
            metrics.add(
                f"factor.{label}.{field}",
                aggregate[field]["mean"],
                factor_source,
            )
    metrics.add(
        "factor.iso",
        factor["definitions"]["isotropic_expectation"],
        factor_source,
    )

    return metrics


def _render(value: float, decimals: int, *, sign: bool = False) -> str:
    """Conventional ROUND_HALF_UP rendering used by the paper tables."""
    quantum = Decimal("1").scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return format(rounded, f"{'+' if sign else ''}.{decimals}f")


def build_claims(metrics: MetricStore) -> list[Claim]:
    """Build contextual manuscript expectations from artifact-derived metrics."""
    value = metrics.value
    shown = lambda key, digits: _render(value(key), digits)
    signed = lambda key, digits: _render(value(key), digits, sign=True)
    claims: list[Claim] = []

    def add(claim_id: str, file: str, fragment: str, count: int = 1) -> None:
        claims.append(Claim(claim_id, file, _normalize_tex(fragment), count))

    # Abstract and duplicated paper-level headlines.
    add(
        "abstract.complete_vitb.rounded_range",
        "sections/abstract.tex",
        f"activation null reproduces ${shown('arm.B32.fraction', 0)}$--"
        f"${shown('arm.B16.fraction', 0)}\\%$ of above-isotropic overlap",
    )
    add(
        "abstract.lora.rounded",
        "sections/abstract.tex",
        f"factor null reproduces about ${shown('arm.LoRA16.fraction', 0)}\\%$",
    )
    add(
        "introduction.complete_vitb",
        "sections/introduction.tex",
        f"null reproduces ${shown('arm.B16.fraction', 1)}\\%$ and "
        f"${shown('arm.B32.fraction', 1)}\\%$ of above-isotropic top-$8$ overlap",
    )
    add(
        "introduction.lora",
        "sections/introduction.tex",
        f"conditioned null reproduces ${shown('arm.LoRA16.fraction', 1)}\\%$ "
        "of the product update's above-isotropic overlap",
    )
    add(
        "discussion.headlines",
        "sections/conclusion.tex",
        f"The null reproduces ${shown('arm.B16.fraction', 1)}\\%$ and "
        f"${shown('arm.B32.fraction', 1)}\\%$ in the complete ViT-B runs; "
        f"the factor-conditioned LoRA value is "
        f"${shown('arm.LoRA16.fraction', 1)}\\%$",
    )

    # Main experiment point estimates and crossed intervals.
    add(
        "experiments.exact.complete_vitb",
        "sections/experiments.tex",
        f"${shown('arm.B16.fraction', 4)}\\%$ "
        f"($[{shown('bootstrap.B16.fraction.lo', 4)},"
        f"{shown('bootstrap.B16.fraction.hi', 4)}]$) and "
        f"${shown('arm.B32.fraction', 4)}\\%$ "
        f"($[{shown('bootstrap.B32.fraction.lo', 4)},"
        f"{shown('bootstrap.B32.fraction.hi', 4)}]$)",
    )
    add(
        "experiments.exact.l14",
        "sections/experiments.tex",
        f"gives ${shown('arm.L14.fraction', 4)}\\%$ "
        f"($[{shown('bootstrap.L14.fraction.lo', 4)},"
        f"{shown('bootstrap.L14.fraction.hi', 4)}]$)",
    )
    add(
        "experiments.exact.residual_intervals",
        "sections/experiments.tex",
        f"intervals are $[{signed('bootstrap.B16.residual.lo', 5)},"
        f"{signed('bootstrap.B16.residual.hi', 5)}]$, "
        f"$[{signed('bootstrap.B32.residual.lo', 5)},"
        f"{signed('bootstrap.B32.residual.hi', 5)}]$, and "
        f"$[{signed('bootstrap.L14.residual.lo', 5)},"
        f"{signed('bootstrap.L14.residual.hi', 5)}]$",
    )
    for label, prose in (
        ("LoRA16", "factor null reproduces"),
        ("LinearizedLoRA16", "linearized pool gives"),
        ("FT8", "gives"),
    ):
        add(
            f"experiments.{label}.point_interval",
            "sections/experiments.tex",
            f"{prose} ${shown(f'arm.{label}.fraction', 4)}\\%$ "
            f"($[{shown(f'bootstrap.{label}.fraction.lo', 4)},"
            f"{shown(f'bootstrap.{label}.fraction.hi', 4)}]$)",
        )

    # Exact scalar decomposition table. Row labels make every number semantic.
    for label, row_label, modules in (
        ("B16", "ViT-B/16", 72),
        ("B32", "ViT-B/32", 72),
        ("L14", "ViT-L/14 $q/v$", 48),
        ("FT8", "matched FT8 $q/v$", 24),
    ):
        add(
            f"table.main.{label}",
            "sections/appendix.tex",
            f"{row_label} & ${modules}$ & ${shown(f'arm.{label}.raw', 6)}$ & "
            f"${shown(f'arm.{label}.iso', 6)}$ & "
            f"${shown(f'arm.{label}.null', 6)}$ & "
            f"${shown(f'arm.{label}.residual', 6)}$ & "
            f"${shown(f'arm.{label}.fraction', 4)}\\%$",
        )

    # ViT-B/16 role split in both reporting locations.
    add(
        "experiments.role_split",
        "sections/experiments.tex",
        f"${shown('role.B16.residual.fraction', 4)}\\%$ for modules reading "
        f"the residual stream and ${shown('role.B16.internal.fraction', 4)}\\%$ "
        "for modules reading a representation built within the block",
    )
    add(
        "appendix.role_split",
        "sections/appendix.tex",
        f"${shown('role.B16.residual.fraction', 4)}\\%$ for residual-reading "
        "modules (\\texttt{q}, \\texttt{k}, \\texttt{v}, \\texttt{fc1}) and "
        f"${shown('role.B16.internal.fraction', 4)}\\%$ for block-internal "
        "readers (\\texttt{out\\_proj}, \\texttt{fc2})",
    )

    # LoRA factor audit: all 24 targeted modules, not the legacy six-module subset.
    add(
        "experiments.factor_audit",
        "sections/experiments.tex",
        f"learned-from-zero $B$ overlaps at "
        f"${shown('factor.LoRA.B_left', 4)}$--"
        f"${shown('factor.LinearizedLoRA.B_left', 4)}$, well below $A$'s "
        f"${shown('factor.LoRA.A_right', 4)}$--"
        f"${shown('factor.LinearizedLoRA.A_right', 4)}$",
    )
    for family, row_label in (
        ("LoRA", "LoRA-16"),
        ("LinearizedLoRA", "lin-LoRA"),
    ):
        add(
            f"table.factor_audit.{family}",
            "sections/appendix.tex",
            f"{row_label} & ${shown(f'factor.{family}.A_right', 4)}$ & "
            f"${shown(f'factor.{family}.B_left', 4)}$ & "
            f"${shown(f'factor.{family}.BA_right', 4)}$",
        )
    add(
        "table.factor_audit.isotropic",
        "sections/appendix.tex",
        "isotropic & "
        + " & ".join(f"${shown('factor.iso', 4)}$" for _ in range(3)),
    )
    add(
        "introduction.factor_audit",
        "sections/introduction.tex",
        f"right singular subspaces overlap at "
        f"${shown('factor.LoRA.A_right', 2)}$, and at "
        f"${shown('factor.LinearizedLoRA.A_right', 2)}$ when linearisation "
        f"keeps $A$ bit-identical, while the learned $B$ factor's top-$8$ left "
        f"singular subspaces overlap at ${shown('factor.LoRA.B_left', 2)}$ "
        f"against ${shown('factor.iso', 2)}$ isotropic chance",
    )

    # Positive-control ranges and activation response table.
    initial_recovery = [
        value(f"power.activation.initial.s{s}.recovery") for s in (1, 2, 4)
    ]
    current_recovery = [
        value(f"power.activation.current.s{s}.recovery") for s in (1, 2, 4)
    ]
    initial_range = (
        f"${_render(min(initial_recovery), 1)}$--"
        f"${_render(max(initial_recovery), 1)}\\%$"
    )
    current_range = (
        f"${_render(min(current_recovery), 1)}$--"
        f"${_render(max(current_recovery), 1)}\\%$"
    )
    add(
        "validation.activation_power",
        "sections/validating.tex",
        f"recovers only {initial_range} of delivered overlap while returning "
        f"${signed('power.activation.initial.s0.excess', 5)}$ at $s=0$, "
        f"which looks well behaved under the offset check; the corrected null "
        f"recovers {current_range} and has "
        f"${signed('power.activation.current.s0.excess', 6)}$ at $s=0$",
    )
    add(
        "validation.activation_power_cost",
        "sections/validating.tex",
        f"recovery rises from {initial_range} to {current_range} while the "
        f"unplanted offset moves from "
        f"${signed('power.activation.initial.s0.excess', 5)}$ to "
        f"${signed('power.activation.current.s0.excess', 6)}$",
    )
    add(
        "introduction.activation_power",
        "sections/introduction.tex",
        f"{initial_range} to {current_range}",
    )
    add(
        "experiments.activation_power",
        "sections/experiments.tex",
        f"{initial_range} instead of the corrected null's {current_range}",
    )
    add(
        "table.activation_power.s0",
        "sections/appendix.tex",
        f"$0/8$ & ${shown('power.activation.current.s0.raw', 5)}$ & --- & "
        f"$X_0={signed('power.activation.initial.s0.excess', 5)}$ & "
        f"$X_0={signed('power.activation.current.s0.excess', 6)}$",
    )
    for planted in (1, 2, 4):
        add(
            f"table.activation_power.s{planted}",
            "sections/appendix.tex",
            f"${planted}/8$ & "
            f"${shown(f'power.activation.current.s{planted}.raw', 5)}$ & "
            f"${shown(f'power.activation.current.s{planted}.delivered', 2)}\\%$ & "
            f"${shown(f'power.activation.initial.s{planted}.recovery', 1)}\\%$ & "
            f"${shown(f'power.activation.current.s{planted}.recovery', 1)}\\%$",
        )

    # LoRA positive control: ranges, main-text list, and table.
    deliveries = [value(f"power.lora.current.s{s}.delivered") for s in (1, 2, 4)]
    recoveries = [value(f"power.lora.current.s{s}.recovery") for s in (1, 2, 4)]
    delivery_range = (
        f"${_render(min(deliveries), 1)}$--"
        f"${_render(max(deliveries), 1)}\\%$"
    )
    recovery_range = (
        f"${_render(min(recoveries), 1)}$--"
        f"${_render(max(recoveries), 1)}\\%$"
    )
    for claim_id, file in (
        ("introduction.lora_power", "sections/introduction.tex"),
        ("method.lora_power", "sections/method.tex"),
        ("appendix.lora_power_limit", "sections/appendix.tex"),
    ):
        possessive = "its " if file != "sections/introduction.tex" else "the "
        add(
            claim_id,
            file,
            f"{delivery_range} of {possessive}nominal increment",
        )
        add(claim_id + ".recovery", file, recovery_range)
    add(
        "validation.lora_power.delivery",
        "sections/validating.tex",
        "It delivers "
        + ", ".join(f"${_render(x, 3)}\\%$" for x in deliveries[:-1])
        + f", and ${_render(deliveries[-1], 3)}\\%$",
    )
    add(
        "validation.lora_power.recovery",
        "sections/validating.tex",
        "gives "
        + ", ".join(f"${_render(x, 3)}\\%$" for x in recoveries[:-1])
        + f", and ${_render(recoveries[-1], 3)}\\%$ recovery",
    )
    add(
        "experiments.lora_power",
        "sections/experiments.tex",
        "$"
        + "/".join(_render(x, 2) for x in deliveries)
        + "\\%$ of its nominal increment, while level-centred recovery is $"
        + "/".join(_render(x, 2) for x in recoveries)
        + "\\%$ at $s=1/2/4$",
    )
    for planted in (0, 1, 2, 4):
        if planted == 0:
            fragment = (
                f"$0/8$ & ${shown('power.lora.current.s0.raw', 4)}$ & --- & "
                f"${shown('power.lora.initial.s0.excess', 4)}$ & "
                f"${shown('power.lora.current.s0.excess', 4)}$ & ---"
            )
        else:
            fragment = (
                f"${planted}/8$ & "
                f"${shown(f'power.lora.current.s{planted}.raw', 4)}$ & "
                f"${shown(f'power.lora.current.s{planted}.delivered', 3)}\\%$ & "
                f"${signed(f'power.lora.initial.s{planted}.excess', 4)}$ & "
                f"${signed(f'power.lora.current.s{planted}.excess', 4)}$ & "
                f"${shown(f'power.lora.current.s{planted}.recovery', 3)}\\%$"
            )
        add(
            f"table.lora_power.s{planted}",
            "sections/appendix.tex",
            fragment,
        )

    # Ablation table and prose are tied to named interventions.
    for key, prefix in (
        ("neither", "        &        "),
        ("singleton", "        & \\ding{51}"),
        ("complement", "\\ding{51} &        "),
        ("both", "\\ding{51} & \\ding{51}"),
    ):
        add(
            f"table.ablation.{key}",
            "sections/appendix.tex",
            f"{prefix} & ${shown(f'ablation.{key}.recovery', 1)}\\%$",
        )
    add(
        "appendix.ablation.prose",
        "sections/appendix.tex",
        f"recovery from ${shown('ablation.neither.recovery', 1)}\\%$ to "
        f"${shown('ablation.singleton.recovery', 1)}\\%$, the full complement "
        f"alone to ${shown('ablation.complement.recovery', 1)}\\%$, and both "
        f"to ${shown('ablation.both.recovery', 1)}\\%$",
    )

    # Common-reference baseline table and prose, including the generative row.
    for key, row_label in (
        ("orthogonality", "exact orthogonality~\\citep{gargiulo2025task}"),
        ("isotropic", "isotropic $k/d$"),
        ("regmean", "whitened at the participation ratio"),
        ("tikhonov", "whitened, all stored directions"),
        ("generative", "generative activation model"),
        ("permutation", "coordinate permutation (paired action)"),
        ("ours", "activation-conditioned (ours, circular here)"),
    ):
        add(
            f"table.baseline.{key}",
            "sections/validating.tex",
            f"{row_label} & ${signed(f'baseline.{key}.excess', 4)}$",
        )
    add(
        "validation.baseline.prose",
        "sections/validating.tex",
        f"raw overlap is ${shown('baseline.raw', 4)}$, or "
        f"${shown('baseline.raw_over_chance', 1)}$ times isotropic chance; "
        f"subtracting the isotropic comparator reports "
        f"${signed('baseline.isotropic.excess', 4)}$ excess",
    )
    add(
        "introduction.baseline",
        "sections/introduction.tex",
        f"isotropic and participation-ratio-whitened baselines report "
        f"${signed('baseline.isotropic.excess', 4)}$ and "
        f"${signed('baseline.regmean.excess', 4)}$ apparent excess",
    )
    add(
        "appendix.generative_baseline",
        "sections/appendix.tex",
        f"generator reports native raw-minus-null excess "
        f"${signed('baseline.generative.excess', 4)}$, whereas the "
        f"observed-profile-preserving null reports "
        f"${signed('baseline.ours.excess', 4)}$",
    )

    # Projector paragraph and row-labeled appendix table.
    add(
        "experiments.projector",
        "sections/experiments.tex",
        f"Observed--null basis overlap is "
        f"${shown('projector.LoRA.projector_survives_null', 4)}$ versus "
        f"${shown('projector.LoRA.chance', 4)}$ chance for LoRA and "
        f"${shown('projector.FT.projector_survives_null', 4)}$ versus "
        f"${shown('projector.FT.chance', 4)}$ for full fine-tuning. The "
        f"corresponding basis mass in the top activation subspace is "
        f"${shown('projector.LoRA.projector_in_top_activation', 4)}$ and "
        f"${shown('projector.FT.projector_in_top_activation', 4)}$ "
        f"(chance ${shown('projector.LoRA.activation_chance', 4)}$)",
    )
    for family, row_label in (("FT", "full fine-tuning"), ("LoRA", "LoRA-16")):
        add(
            f"table.projector.{family}",
            "sections/appendix.tex",
            f"{row_label} & "
            f"${shown(f'projector.{family}.projector_survives_null', 4)}$ & "
            f"${shown(f'projector.{family}.chance', 4)}$ & "
            f"${shown(f'projector.{family}.projector_in_top_activation', 4)}$ & "
            f"${shown(f'projector.{family}.activation_chance', 4)}$",
        )
    add(
        "appendix.projector.percentage",
        "sections/appendix.tex",
        "the $"
        f"{_render(100.0 * value('projector.LoRA.projector_survives_null'), 1)}"
        "\\%$ "
        "projector-basis fraction",
    )

    # Merge table uses conventional half-up rounding, not Python banker's rounding.
    add(
        "table.merge.references",
        "sections/appendix.tex",
        f"Zero-shot is ${shown('merge.zeroshot.mean', 1)}\\%$ and "
        f"uncompressed task arithmetic is ${shown('merge.full.mean', 1)}\\%$",
    )
    ranks = (8, 16, 32, 64, 128, 256)
    add(
        "table.merge.tau",
        "sections/appendix.tex",
        "$\\tau$ basis & "
        + " & ".join(
            f"${shown(f'merge.tau_{rank}.mean', 1)}$" for rank in ranks
        ),
    )
    add(
        "table.merge.cross",
        "sections/appendix.tex",
        "cross-term basis & "
        + " & ".join(
            f"${shown(f'merge.cross_{rank}.mean', 1)}$" for rank in ranks
        ),
    )

    # Rank claims are explicitly architecture-bound.
    add(
        "appendix.pair_rank",
        "sections/appendix.tex",
        f"Across the $190$ ViT-B/32 pairs, raw and calibrated overlap have "
        f"Spearman ${shown('rank.B32.spearman', 3)}$ and share "
        f"${shown('rank.B32.shared_top20', 0)}$ of the top $20$ pairs; "
        f"ViT-B/16 gives ${shown('rank.B16.spearman', 3)}$ and shares "
        f"${shown('rank.B16.shared_top20', 0)}$ of the top $20$",
    )
    add(
        "experiments.pair_rank",
        "sections/experiments.tex",
        f"Spearman ${shown('rank.B16.spearman', 3)}$--"
        f"${shown('rank.B32.spearman', 3)}$",
    )

    # Artifact-backed sensitivity and external-H0 tables.
    for label, row_label in (
        ("B16", "ViT-B/16"),
        ("B32", "ViT-B/32"),
        ("L14", "ViT-L/14 $q/v$"),
    ):
        add(
            f"table.sensitivity.{label}",
            "sections/appendix.tex",
            f"{row_label} & ${shown(f'sensitivity.{label}.k8.lo', 2)}$--"
            f"${shown(f'sensitivity.{label}.k8.hi', 2)}\\%$ & "
            f"${shown(f'sensitivity.{label}.all.lo', 2)}$--"
            f"${shown(f'sensitivity.{label}.all.hi', 2)}\\%$",
        )
    add(
        "appendix.sensitivity.max_z",
        "sections/appendix.tex",
        "the largest observed change from varying $z$ is "
        f"${shown('sensitivity.max_z_spread', 3)}$ percentage points",
    )
    for family, row_label in (
        ("FT8", "matched FT8, activation null"),
        ("LoRA", "LoRA-16, activation null"),
        ("LinearizedLoRA", "lin-LoRA, activation null"),
    ):
        add(
            f"table.h0.{family}",
            "sections/appendix.tex",
            f"{row_label} & ${shown(f'h0.{family}', 5)}$",
        )

    # The mismatch claim is intentionally strict enough to catch stale rounding.
    add(
        "experiments.regime_mismatch",
        "sections/experiments.tex",
        f"activation null instead leaves "
        f"${shown('mismatch.LoRA.residual_fraction', 1)}\\%$ of the same LoRA "
        f"pool's above-chance overlap as residual, versus "
        f"${shown('arm.LoRA16.residual_fraction', 1)}\\%$ under the "
        f"observed-$A$-preserving factor null "
        f"(${signed('mismatch.LoRA.residual', 4)}$ versus "
        f"${signed('arm.LoRA16.residual', 4)}$)",
    )

    return claims


def audit_claims(
    documents: dict[str, Document], claims: list[Claim]
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    errors: list[str] = []
    covered: dict[str, list[tuple[int, int]]] = {
        name: [] for name in documents
    }
    for claim in claims:
        document = documents.get(claim.file)
        if document is None:
            errors.append(
                f"{claim.claim_id}: required input {claim.file!r} is absent "
                "from the paper graph"
            )
            continue
        starts = []
        cursor = 0
        while True:
            start = document.normalized.find(claim.fragment, cursor)
            if start < 0:
                break
            starts.append(start)
            covered[claim.file].append((start, start + len(claim.fragment)))
            cursor = start + max(1, len(claim.fragment))
        if len(starts) != claim.count:
            errors.append(
                f"{claim.claim_id}: found {len(starts)} occurrence(s) in "
                f"{claim.file}, expected {claim.count}; expected context: "
                f"{claim.fragment[:180]!r}"
            )
    return errors, covered


_RESULT_LIKE_RE = re.compile(
    r"\$[^$\n]{0,100}(?:\\%|(?<![A-Za-z])(?:[+-]?\d+\.\d+))[^$\n]{0,100}\$"
)


def find_uncovered_result_like(
    documents: dict[str, Document],
    covered: dict[str, list[tuple[int, int]]],
) -> list[tuple[str, str]]:
    """Conservatively inventory result-like tokens outside registered spans."""
    findings: list[tuple[str, str]] = []
    for relative, document in sorted(documents.items()):
        spans = covered.get(relative, [])
        for match in _RESULT_LIKE_RE.finditer(document.normalized):
            if any(
                match.start() < stop and match.end() > start
                for start, stop in spans
            ):
                continue
            token = match.group(0)
            if len(token) > 70 and "\\%" not in token:
                continue
            context_start = max(0, match.start() - 55)
            context_stop = min(len(document.normalized), match.end() + 55)
            findings.append(
                (relative, document.normalized[context_start:context_stop])
            )
    return findings


def run_v7_audit(artifact: Path) -> tuple[bool, str]:
    checker = Path(__file__).with_name("check_paper_numbers.py").resolve()
    if not checker.is_file():
        return False, f"missing v7 numerical checker: {checker}"
    completed = subprocess.run(
        [sys.executable, str(checker), str(artifact.resolve())],
        cwd=str(checker.parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        type=Path,
        required=True,
        help="main TeX entrypoint whose input graph is authoritative",
    )
    parser.add_argument(
        "--artifact", type=Path, required=True, help="v7 artifact root"
    )
    parser.add_argument(
        "--strict-uncovered",
        action="store_true",
        help="fail on result-like literals outside the semantic registry",
    )
    parser.add_argument(
        "--max-uncovered",
        type=int,
        default=60,
        help="maximum uncovered candidates to print (default: 60)",
    )
    args = parser.parse_args(argv)

    paper = args.paper.expanduser().resolve()
    artifact = args.artifact.expanduser().resolve()
    print("=== Resolved audit targets ===")
    print(f"paper entrypoint: {paper}")
    print(f"artifact root:    {artifact}")
    try:
        _root, documents = resolve_input_graph(paper)
    except (OSError, ValueError) as exc:
        print(f"FAILED to resolve manuscript: {exc}")
        return 1
    print("compiled TeX graph:")
    for relative in sorted(documents):
        print(f"  - {relative}")

    print("\n=== Existing v7 numerical/artifact audit ===")
    audit_ok, output = run_v7_audit(artifact)
    print(output.rstrip())
    if not audit_ok:
        print(
            "\nFAILED: semantic checks were not trusted because the v7 audit failed."
        )
        return 1

    print("\n=== Semantic manuscript audit ===")
    try:
        metrics = build_metrics(artifact)
        claims = build_claims(metrics)
        errors, covered = audit_claims(documents, claims)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAILED to build semantic metrics/claims: {exc}")
        return 1
    print(f"materialised {len(metrics)} semantic metrics")
    print(f"enforced {len(claims)} contextual claims")
    if errors:
        print("\nSEMANTIC CLAIM FAILURES")
        for error in errors:
            print(f"- {error}")

    uncovered = find_uncovered_result_like(documents, covered)
    print("\n=== Coverage inventory (non-authoritative warnings) ===")
    if uncovered:
        print(
            f"{len(uncovered)} result-like TeX occurrence(s) are outside the "
            "current semantic registry. They may be design constants, "
            "mathematical examples, literature values, or empirical claims "
            "requiring an additional compact artifact record."
        )
        for relative, context in uncovered[: max(0, args.max_uncovered)]:
            print(f"- {relative}: {context}")
        if len(uncovered) > args.max_uncovered:
            print(
                f"- ... {len(uncovered) - args.max_uncovered} additional occurrence(s)"
            )
    else:
        print("No result-like TeX occurrences remain outside the registry.")

    if errors or (args.strict_uncovered and uncovered):
        print("\nFAILED")
        return 1
    print("\nPASSED: v7 artifact audit and registered semantic claims agree.")
    if uncovered:
        print(
            "Coverage is intentionally partial; review the inventory before "
            "claiming that every empirical literal is checked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
