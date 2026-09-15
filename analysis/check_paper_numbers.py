#!/usr/bin/env python3
"""Audit the compact PRISM v7 result bundle against paper-facing numbers.

The checker uses only the Python standard library. It recomputes row-level
ratio-of-means estimates from the shipped gzip CSVs, checks the crossed
bootstrap reports, sensitivity grid, positive controls, ablation, projector
and external-H0 diagnostics, LoRA top-k gaps, downstream merge-table aggregation,
exact-arm leave-one-task/layer-out influence,
producer source provenance, upstream revisions, environment pins, file checksums,
and anonymity hygiene.

Run from any directory:

    python3 analysis/check_paper_numbers.py
    python3 analysis/check_paper_numbers.py /path/to/artifact
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import random
import statistics as stats
from pathlib import Path, PurePosixPath


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifact"
EXACT_LABELS = ("B16", "B32", "FT8", "L14")

ARMS = {
    "B16": {
        "file": "v7_exact_clip-vit-base-patch16.csv.gz",
        "null": "null_exact", "null_kind": "exact_haar_block",
        "rows": 13680, "tasks": 20, "pairs": 190, "modules": 72,
        "raw": 0.104573648205648, "iso": 0.00911458333333333,
        "fitted": 0.0333087948281544, "residual": 0.0712648533774935,
        "fraction": 25.3451168070660,
    },
    "B32": {
        "file": "v7_exact_clip-vit-base-patch32.csv.gz",
        "null": "null_exact", "null_kind": "exact_haar_block",
        "rows": 13680, "tasks": 20, "pairs": 190, "modules": 72,
        "raw": 0.0953126490612416, "iso": 0.00911458333333333,
        "fitted": 0.0284195730629921, "residual": 0.0668930759982495,
        "fraction": 22.3960822863435,
    },
    "FT8": {
        "file": "v7_exact_b16_8task_q_v_fullft.csv.gz",
        "null": "null_exact", "null_kind": "exact_haar_block",
        "rows": 672, "tasks": 8, "pairs": 28, "modules": 24,
        "raw": 0.0899303229324730, "iso": 0.0104166666666667,
        "fitted": 0.0233328612264667, "residual": 0.0665974617060063,
        "fraction": 16.2439952661999,
    },
    "L14": {
        "file": "v7_exact_clip-vit-large-patch14.csv.gz",
        "null": "null_exact", "null_kind": "exact_haar_block",
        "rows": 9120, "tasks": 20, "pairs": 190, "modules": 48,
        "raw": 0.0714015473421201, "iso": 0.0078125,
        "fitted": 0.0153771341961227, "residual": 0.0560244131459975,
        "fraction": 11.8961275759072,
    },
    "LoRA16": {
        "file": "v7_mc64_b16_8task_q_v_lora-16_loranull.csv",
        "null": "null_act_raw", "null_kind": "lora",
        "rows": 672, "tasks": 8, "pairs": 28, "modules": 24,
        "raw": 0.607530054813694, "iso": 0.0104166666666667,
        "fitted": 0.489964168584612, "residual": 0.117565886229083,
        "fraction": 80.3109612742205,
    },
    "LinearizedLoRA16": {
        "file": "v7_mc64_b16_8task_q_v_l-lora-16_loranull.csv",
        "null": "null_act_raw", "null_kind": "lora",
        "rows": 672, "tasks": 8, "pairs": 28, "modules": 24,
        "raw": 0.664891910384453, "iso": 0.0104166666666667,
        "fitted": 0.508284363352383, "residual": 0.156607547032070,
        "fraction": 76.0712802301808,
    },
}

BOOTSTRAP = {
    "B16": ((19.3032940672202, 31.80102213675881),
             (0.04539694804998023, 0.10567610594404583)),
    "B32": ((15.551625148194512, 29.88496628736618),
             (0.042240847549262196, 0.10085059137669078)),
    "FT8": ((9.233261730324305, 27.481184677192694),
             (0.0314908079069092, 0.1135981105245895)),
    "L14": ((8.106414053582531, 16.489547018536598),
             (0.03476116710000898, 0.08251974336675953)),
    "LoRA16": ((75.72049668799923, 85.06808051806232),
                (0.08367044051478117, 0.15511742154317798)),
    "LinearizedLoRA16": ((72.19743197530852, 80.24834212447841),
                          (0.12253427443223543, 0.19191310385306445)),
}

LEGACY_LORA_FILES = {
    "LoRA16": "v7_legacy16_b16_8task_q_v_lora-16_loranull.csv.gz",
    "LinearizedLoRA16": "v7_legacy16_b16_8task_q_v_l-lora-16_loranull.csv.gz",
}

H0_ARMS = {
    "FT8 activation MC": {
        "file": "v7_b16_8task_q_v_fullft_mc_h0.csv.gz",
        "raw": 0.0899303229324730, "null": 0.0233027671633791,
        "h0": -0.000402660109102726,
    },
    "LoRA16 activation": {
        "file": "v7_b16_8task_q_v_lora16_actnull_h0.csv.gz",
        "raw": 0.607530054813694, "null": 0.0108859320913210,
        "h0": -0.000373389261464278,
    },
    "LinearizedLoRA16 activation": {
        "file": "v7_b16_8task_q_v_llora16_actnull_h0.csv.gz",
        "raw": 0.664891910384453, "null": 0.0105252213758095,
        "h0": -0.000366558320820332,
    },
}

SENSITIVITY = {
    "clip-vit-base-patch16": {
        "label": "B16", "tasks": 20, "modules": 72, "cells": 13680,
        "all_range": (13.283342366144499, 30.372422671363573),
        "k8_range": (20.22123926764494, 25.608528376432655),
    },
    "clip-vit-base-patch32": {
        "label": "B32", "tasks": 20, "modules": 72, "cells": 13680,
        "all_range": (13.309797147212404, 28.509535746372144),
        "k8_range": (17.937483803402092, 22.554316142771047),
    },
    "clip-vit-large-patch14": {
        "label": "L14", "tasks": 20, "modules": 48, "cells": 9120,
        "all_range": (5.36443304660206, 19.010343005376374),
        "k8_range": (5.835268652112722, 12.002171359852557),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_content_sha256(path: Path) -> str:
    """Hash CSV bytes independent of transparent gzip packaging."""
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{key} contains no values or non-finite values")
    return stats.fmean(values)


def close(got: float, want: float, tolerance: float, label: str,
          errors: list[str]) -> None:
    if not math.isfinite(got) or abs(got - want) > tolerance:
        errors.append(f"{label}: got {got:.12g}, expected {want:.12g}")


def equal(got: object, want: object, label: str, errors: list[str]) -> None:
    if got != want:
        errors.append(f"{label}: got {got!r}, expected {want!r}")


def check_arm(root: Path, label: str, expected: dict[str, object],
              errors: list[str]) -> dict[str, float]:
    path = root / "results" / str(expected["file"])
    rows = read_csv(path)
    equal(len(rows), expected["rows"], f"{label} rows", errors)
    tasks = {row[key] for row in rows for key in ("a", "b")}
    pairs = {tuple(sorted((row["a"], row["b"]))) for row in rows}
    modules = {row["module"] for row in rows}
    equal(len(tasks), expected["tasks"], f"{label} tasks", errors)
    equal(len(pairs), expected["pairs"], f"{label} task pairs", errors)
    equal(len(modules), expected["modules"], f"{label} modules", errors)
    equal({row["null_kind"] for row in rows}, {expected["null_kind"]},
          f"{label} null kind", errors)

    raw = mean(rows, "raw")
    iso = mean(rows, "null_iso")
    fitted = mean(rows, str(expected["null"]))
    residual = raw - fitted
    fraction = 100.0 * (fitted - iso) / (raw - iso)
    for got, key in ((raw, "raw"), (iso, "iso"), (fitted, "fitted"),
                     (residual, "residual"), (fraction, "fraction")):
        close(got, float(expected[key]), 5e-12 if key != "fraction" else 5e-9,
              f"{label} {key}", errors)

    if expected["null"] == "null_exact":
        for row_number, row in enumerate(rows, 2):
            raw_i = float(row["raw"])
            iso_i = float(row["null_iso"])
            null_i = float(row["null_exact"])
            for key, want in (("frac_num", null_i - iso_i),
                              ("frac_den", raw_i - iso_i),
                              ("excess_exact", raw_i - null_i),
                              ("null_act_raw", null_i)):
                close(float(row[key]), want, 2e-9,
                      f"{label} row {row_number} {key}", errors)
        close(mean(rows, "h0_excess"), 0.0, 1e-15,
              f"{label} exact-null H0 field", errors)

    print(f"{label:16s} fraction={fraction:9.4f}% residual={residual:+.8f} "
          f"rows={len(rows)}")
    return {"raw": raw, "iso": iso, "fitted": fitted,
            "residual": residual, "fraction": fraction}


def _average_ranks(values: list[float]) -> list[float]:
    """One-based average ranks, including deterministic handling of exact ties."""
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
    numerator = sum((x - left_mean) * (y - right_mean)
                    for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    if denominator == 0.0:
        raise ValueError("rank correlation has a constant input")
    return numerator / denominator


def check_pair_rank_preservation(root: Path, errors: list[str]) -> None:
    specifications = {
        "B16": ("v7_exact_clip-vit-base-patch16.csv.gz",
                0.994000708577727, 19),
        "B32": ("v7_exact_clip-vit-base-patch32.csv.gz",
                0.9948824941935992, 17),
    }
    for label, (filename, expected_spearman, expected_shared_top20) in \
            specifications.items():
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in read_csv(root / "results" / filename):
            pair = tuple(sorted((row["a"], row["b"])))
            grouped.setdefault(pair, []).append(row)
        pairs = sorted(grouped)
        equal(len(pairs), 190, f"{label} rank-preservation pairs", errors)
        raw = [mean(grouped[pair], "raw") for pair in pairs]
        calibrated = [mean(grouped[pair], "excess_exact") for pair in pairs]
        spearman = _pearson(_average_ranks(raw), _average_ranks(calibrated))
        raw_top = set(sorted(range(len(raw)), key=raw.__getitem__, reverse=True)[:20])
        calibrated_top = set(sorted(range(len(calibrated)),
                                    key=calibrated.__getitem__, reverse=True)[:20])
        shared_top20 = len(raw_top & calibrated_top)
        close(spearman, expected_spearman, 5e-13,
              f"{label} raw/calibrated pair Spearman", errors)
        equal(shared_top20, expected_shared_top20,
              f"{label} raw/calibrated shared top-20", errors)
        print(f"rank preservation {label} Spearman={spearman:.6f}, "
              f"shared top-20={shared_top20}")


def check_bootstraps(root: Path, arm_values: dict[str, dict[str, float]],
                     errors: list[str]) -> None:
    reports = [root / "bootstrap_exact_joint_50000.json",
               root / "bootstrap_lora_joint_50000.json"]
    seen: set[str] = set()
    for path in reports:
        report = json.loads(path.read_text())
        method = report["method"]
        equal(method["seed_reset_for_each_input"], 20260915,
              f"{path.name} bootstrap seed", errors)
        equal(method["valid_replicates_per_scheme"], 50000,
              f"{path.name} bootstrap draws", errors)
        for result in report["datasets"]:
            label = result["label"]
            seen.add(label)
            expected_file = str(ARMS[label]["file"])
            input_text = str(result["input"])
            equal(Path(input_text).name, expected_file,
                  f"{label} bootstrap input basename", errors)
            if Path(input_text).is_absolute() or ".." in PurePosixPath(input_text).parts:
                errors.append(f"{label} bootstrap input is not a safe relative path")
            source = root / "results" / expected_file
            equal(result["sha256"], sha256(source),
                  f"{label} bootstrap source SHA-256", errors)
            metric = result["metric_keys"]["fraction"]
            want_metric = ("exact_reproduced_fraction_percent"
                           if ARMS[label]["null"] == "null_exact"
                           else "reproduced_fraction_percent")
            equal(metric, want_metric, f"{label} bootstrap metric key", errors)
            point = result["point_estimate"]
            close(float(point[metric]), arm_values[label]["fraction"], 5e-10,
                  f"{label} bootstrap point fraction", errors)
            close(float(point["residual"]), arm_values[label]["residual"], 5e-12,
                  f"{label} bootstrap point residual", errors)
            joint = result["percentile_95_intervals"]["joint_task_layer"]
            for got, want, suffix in zip(joint[metric], BOOTSTRAP[label][0], ("lo", "hi")):
                close(float(got), want, 5e-10,
                      f"{label} crossed fraction CI {suffix}", errors)
            for got, want, suffix in zip(joint["residual"], BOOTSTRAP[label][1],
                                         ("lo", "hi")):
                close(float(got), want, 5e-12,
                      f"{label} crossed residual CI {suffix}", errors)
            for scheme in ("joint_task_layer", "task_only", "layer_only"):
                counts = result["replicates"][scheme]
                equal(counts["valid"], 50000, f"{label} {scheme} valid", errors)
                equal(counts["rejected"], 0, f"{label} {scheme} rejected", errors)
    equal(seen, set(ARMS), "bootstrap dataset labels", errors)
    print("bootstrap        6/6 point estimates, crossed CIs, and source hashes checked")


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_artifact_result(root: Path, relative: object,
                          label: str, errors: list[str]) -> Path:
    text = str(relative)
    parts = PurePosixPath(text).parts
    if (not parts or parts[0] != "results" or ".." in parts
            or PurePosixPath(text).is_absolute()):
        errors.append(f"{label} is not a safe artifact-relative result path: {text!r}")
    return root / text


def check_mc64(root: Path, errors: list[str]) -> None:
    report = json.loads(
        (root / "results" / "v7_mc64_lora_factor_null_convergence.json").read_text()
    )
    design = report["design"]
    equal(design["batch_seeds"], [0, 1000003, 2000003, 3000017],
          "MC64 batch seeds", errors)
    equal(design["draws_per_batch"], 16, "MC64 draws/batch", errors)
    equal(design["independent_batches"], 4, "MC64 batches", errors)
    equal(design["total_draws_per_row"], 64, "MC64 total draws/row", errors)
    equal(design["all_batch_row_files_included"], True,
          "MC64 batch inclusion", errors)
    settings = json.loads((root / "run_settings.json").read_text())["runs"]
    configured = settings["lora_factor_null"]
    equal(configured["independent_batch_seeds"], design["batch_seeds"],
          "MC64 run-settings seeds", errors)
    equal(configured["draws_per_batch"], design["draws_per_batch"],
          "MC64 run-settings draws/batch", errors)
    equal(configured["independent_batches"], design["independent_batches"],
          "MC64 run-settings batches", errors)
    equal(configured["total_null_draws_per_row"], design["total_draws_per_row"],
          "MC64 run-settings total draws", errors)
    equal(configured["primary_rows"],
          [f"results/{ARMS[label]['file']}"
           for label in ("LoRA16", "LinearizedLoRA16")],
          "MC64 run-settings primary rows", errors)
    equal(configured["legacy_seed0_rows"],
          [f"results/{LEGACY_LORA_FILES[label]}"
           for label in ("LoRA16", "LinearizedLoRA16")],
          "MC64 run-settings legacy rows", errors)
    equal(settings["legacy_lora_crossed_bootstrap"]["record"],
          "bootstrap_lora_legacy16_joint_50000.json",
          "legacy LoRA bootstrap run-settings record", errors)

    arm_names = {"LoRA16": "LoRA-16", "LinearizedLoRA16": "Linearized-LoRA-16"}
    for label, report_label in arm_names.items():
        arm = report["arms"][report_label]
        batches = {int(entry["seed"]): entry for entry in arm["batch_files"]}
        equal(set(batches), set(design["batch_seeds"]),
              f"{label} MC64 batch seeds", errors)
        batch_rows = {}
        batch_points = {int(entry["seed"]): entry
                        for entry in arm["batch_point_estimates"]}
        for seed in design["batch_seeds"]:
            entry = batches[seed]
            path = _safe_artifact_result(root, entry["path"],
                                         f"{label} seed {seed}", errors)
            equal(csv_content_sha256(path), entry["sha256"],
                  f"{label} seed {seed} decompressed SHA-256", errors)
            rows = read_csv(path)
            equal(len(rows), 672, f"{label} seed {seed} rows", errors)
            keyed = {(row["module"], row["a"], row["b"]): row for row in rows}
            equal(len(keyed), 672, f"{label} seed {seed} unique cells", errors)
            batch_rows[seed] = keyed
            point = batch_points[seed]
            raw = mean(rows, "raw")
            iso = mean(rows, "null_iso")
            fitted = mean(rows, "null_act_raw")
            computed = {
                "raw": raw,
                "cond": mean(rows, "cond"),
                "null_iso": iso,
                "null_act_raw": fitted,
                "null_act_cond": mean(rows, "null_act_cond"),
                "residual": raw - fitted,
                "reproduced_fraction_percent": 100.0 * (fitted - iso) / (raw - iso),
            }
            for key, value in computed.items():
                close(float(point[key]), value, 5e-12,
                      f"{label} seed {seed} {key}", errors)

        pooled_record = arm["pooled_64_draw_file"]
        pooled_path = _safe_artifact_result(root, pooled_record["path"],
                                            f"{label} pooled64", errors)
        equal(str(pooled_record["path"]), f"results/{ARMS[label]['file']}",
              f"{label} canonical pooled path", errors)
        equal(csv_content_sha256(pooled_path), pooled_record["sha256"],
              f"{label} pooled64 SHA-256", errors)
        pooled_rows = read_csv(pooled_path)
        equal(len(pooled_rows), 672, f"{label} pooled64 rows", errors)
        pooled = {(row["module"], row["a"], row["b"]): row for row in pooled_rows}
        equal(len(pooled), 672, f"{label} pooled64 unique cells", errors)
        reference_keys = set(batch_rows[0])
        equal(set(pooled), reference_keys, f"{label} pooled cell keys", errors)
        for seed, keyed in batch_rows.items():
            equal(set(keyed), reference_keys,
                  f"{label} seed {seed} cell keys", errors)
        for key, row in pooled.items():
            for column in ("raw", "cond", "null_iso"):
                close(float(row[column]), float(batch_rows[0][key][column]), 2e-15,
                      f"{label} pooled {key} {column}", errors)
            raw_null = stats.fmean(float(batch_rows[seed][key]["null_act_raw"])
                                   for seed in design["batch_seeds"])
            cond_null = stats.fmean(float(batch_rows[seed][key]["null_act_cond"])
                                    for seed in design["batch_seeds"])
            close(float(row["null_act_raw"]), raw_null, 2e-15,
                  f"{label} pooled {key} raw null", errors)
            close(float(row["null_act_cond"]), cond_null, 2e-15,
                  f"{label} pooled {key} conditional null", errors)
            close(float(row["excess_raw"]), float(row["raw"]) - raw_null, 2e-15,
                  f"{label} pooled {key} raw residual", errors)
            close(float(row["excess_cond"]), float(row["cond"]) - cond_null, 2e-15,
                  f"{label} pooled {key} conditional residual", errors)
            close(float(row["excess_raw_corrected"]), float(row["excess_raw"]), 0.0,
                  f"{label} pooled {key} compatibility alias", errors)

        point = arm["pooled_point_estimate"]
        raw = mean(pooled_rows, "raw")
        iso = mean(pooled_rows, "null_iso")
        fitted = mean(pooled_rows, "null_act_raw")
        pooled_values = {
            "raw": raw, "cond": mean(pooled_rows, "cond"), "null_iso": iso,
            "null_act_raw": fitted,
            "null_act_cond": mean(pooled_rows, "null_act_cond"),
            "residual": raw - fitted,
            "reproduced_fraction_percent": 100.0 * (fitted - iso) / (raw - iso),
        }
        for key, value in pooled_values.items():
            close(float(point[key]), value, 5e-12,
                  f"{label} pooled point {key}", errors)

        legacy = arm["legacy_16_draw_file"]
        equal(legacy["identical_to_seed0"], True,
              f"{label} legacy seed-zero identity flag", errors)
        equal(str(legacy["path"]), f"results/{LEGACY_LORA_FILES[label]}",
              f"{label} legacy path", errors)
        legacy_path = _safe_artifact_result(root, legacy["path"],
                                            f"{label} legacy16", errors)
        equal(csv_content_sha256(legacy_path), legacy["sha256"],
              f"{label} legacy decompressed SHA-256", errors)
        equal(legacy["sha256"], batches[0]["sha256"],
              f"{label} legacy equals seed-zero hash", errors)

        difference = arm["difference_from_legacy_16_draw"]
        seed_zero = batch_points[0]
        for field in ("null_act_raw", "residual", "reproduced_fraction_percent"):
            close(float(difference[field]), float(point[field]) - float(seed_zero[field]),
                  5e-12, f"{label} pooled-minus-legacy {field}", errors)

        diagnostics = arm["batch_mc_diagnostic"]
        batch_fractions = [float(batch_points[seed]["reproduced_fraction_percent"])
                           for seed in design["batch_seeds"]]
        batch_nulls = [float(batch_points[seed]["null_act_raw"])
                       for seed in design["batch_seeds"]]
        for field, values in (("fraction", batch_fractions), ("null_mean", batch_nulls)):
            sample_sd = stats.stdev(values)
            prefix = ("fraction_sample_sd_percentage_points" if field == "fraction"
                      else "null_mean_sample_sd")
            se_key = ("fraction_pooled_mc_se_percentage_points" if field == "fraction"
                      else "null_mean_pooled_mc_se")
            close(float(diagnostics[prefix]), sample_sd, 5e-12,
                  f"{label} {field} batch sample SD", errors)
            close(float(diagnostics[se_key]), sample_sd / 2.0, 5e-12,
                  f"{label} {field} pooled MC SE", errors)
        close(float(diagnostics["residual_pooled_mc_se"]),
              float(diagnostics["null_mean_pooled_mc_se"]), 2e-15,
              f"{label} residual/null MC SE identity", errors)

        for column, diagnostic_key in (
                ("null_act_raw", "rowwise_raw_null_pooled_mc_se"),
                ("null_act_cond", "rowwise_cond_null_pooled_mc_se")):
            rowwise = [stats.stdev(float(batch_rows[seed][key][column])
                                   for seed in design["batch_seeds"]) / 2.0
                       for key in sorted(reference_keys)]
            expected = diagnostics[diagnostic_key]
            for summary_key, value in (
                    ("min", min(rowwise)), ("max", max(rowwise)),
                    ("median", stats.median(rowwise)),
                    ("p95", _linear_quantile(rowwise, 0.95))):
                close(float(expected[summary_key]), value, 5e-12,
                      f"{label} {column} rowwise SE {summary_key}", errors)

    legacy_report = json.loads(
        (root / "bootstrap_lora_legacy16_joint_50000.json").read_text()
    )
    legacy_seen = set()
    for result in legacy_report["datasets"]:
        label = result["label"]
        legacy_seen.add(label)
        expected_path = f"results/{LEGACY_LORA_FILES[label]}"
        equal(result["input"], expected_path,
              f"{label} legacy bootstrap input", errors)
        source = root / expected_path
        equal(result["sha256"], sha256(source),
              f"{label} legacy bootstrap source SHA-256", errors)
        seed_zero = report["arms"][arm_names[label]]["batch_point_estimates"][0]
        metric = result["metric_keys"]["fraction"]
        close(float(result["point_estimate"][metric]),
              float(seed_zero["reproduced_fraction_percent"]), 5e-10,
              f"{label} legacy bootstrap point fraction", errors)
        close(float(result["point_estimate"]["residual"]),
              float(seed_zero["residual"]), 5e-12,
              f"{label} legacy bootstrap point residual", errors)
    equal(legacy_seen, set(arm_names), "legacy LoRA bootstrap labels", errors)
    print("LoRA MC64        4x16 draws/row pooled; batch MC SE and legacy seed-0 checked")


def check_lora_topk_gaps(root: Path, errors: list[str]) -> None:
    path = root / "results" / "v7_lora_topk_gap_diagnostic.json"
    report = json.loads(path.read_text())
    equal(report["schema_version"], 1, "LoRA top-k gap schema", errors)
    equal(report["gap_definition"],
          "(s_k - s_{k+1}) / s_k using linear singular values",
          "LoRA top-k gap definition", errors)
    equal(report["k"], 8, "LoRA top-k gap boundary", errors)
    equal(report["thresholds"], [0.0001, 0.001, 0.01, 0.05],
          "LoRA top-k gap thresholds", errors)
    factor = report["factor_null"]
    equal(factor["seeds"], [0, 1000003, 2000003, 3000017],
          "LoRA top-k gap null seeds", errors)
    equal(factor["draws_per_seed_per_task_module_cell"], 16,
          "LoRA top-k gap draws/seed", errors)
    if "No local paths" not in report["path_policy"]:
        errors.append("LoRA top-k gap diagnostic lacks path policy")

    pins = json.loads((root / "upstream_revisions.json").read_text())["adapters"]
    expected_counts = {
        "rank16_lora": {
            "observed_A": [0, 0, 63, 192],
            "observed_BA": [0, 0, 1, 25],
            "factor_null_BA": [0, 3, 68, 1963],
        },
        "rank16_linearized_lora": {
            "observed_A": [0, 0, 72, 192],
            "observed_BA": [0, 0, 2, 31],
            "factor_null_BA": [0, 1, 91, 1909],
        },
    }
    families = {entry["family"]: entry for entry in report["families"]}
    equal(set(families), set(expected_counts), "LoRA top-k gap families", errors)
    null_below_1e3 = 0
    null_below_1e4 = 0
    for family, diagnostic_counts in expected_counts.items():
        entry = families[family]
        scope = entry["scope"]
        equal({key: scope[key] for key in
               ("tasks", "modules_per_task", "observed_cells", "factor_rank")},
              {"tasks": 8, "modules_per_task": 24,
               "observed_cells": 192, "factor_rank": 16},
              f"{family} top-k gap scope", errors)
        family_pins = pins[family]
        expected_snapshots = [
            f"{family_pins['repository_template'].format(task=task)}@{revision}"
            for task, revision in family_pins["revision_by_task"].items()
        ]
        equal(scope["snapshots"], expected_snapshots,
              f"{family} top-k gap snapshots", errors)
        for diagnostic, counts in diagnostic_counts.items():
            record = entry[diagnostic]
            expected_n = 12288 if diagnostic == "factor_null_BA" else 192
            equal(record["n"], expected_n, f"{family} {diagnostic} n", errors)
            equal(list(record["threshold_counts"].values()), counts,
                  f"{family} {diagnostic} threshold counts", errors)
            quantiles = [float(value) for value in record["quantiles"].values()]
            if quantiles != sorted(quantiles) or not all(0.0 <= value <= 1.0
                                                        for value in quantiles):
                errors.append(f"{family} {diagnostic} quantiles are invalid")
        equal(entry["observed_A"]["threshold_counts"]["lt_0.001"], 0,
              f"{family} observed A boundary gaps below 1e-3", errors)
        equal(entry["observed_BA"]["threshold_counts"]["lt_0.001"], 0,
              f"{family} observed BA boundary gaps below 1e-3", errors)
        null_below_1e3 += entry["factor_null_BA"]["threshold_counts"]["lt_0.001"]
        null_below_1e4 += entry["factor_null_BA"]["threshold_counts"]["lt_0.0001"]
    equal(null_below_1e3, 4, "LoRA null BA gaps below 1e-3", errors)
    equal(null_below_1e4, 0, "LoRA null BA gaps below 1e-4", errors)
    provenance = json.loads((root / "code" / "SOURCE_PROVENANCE.json").read_text())
    equal(report["implementation_sha256"],
          provenance["archived_execution_sha256"]["scripts/lora_topk_gap_diagnostic.py"],
          "LoRA top-k gap implementation hash", errors)
    settings = json.loads((root / "run_settings.json").read_text())["runs"][
        "lora_topk_boundary_gap_diagnostic"
    ]
    equal(settings["record"], "results/v7_lora_topk_gap_diagnostic.json",
          "LoRA top-k gap run-settings record", errors)
    equal(settings["factor_null_cells"], 24576,
          "LoRA top-k gap run-settings null cells", errors)
    print("LoRA top-k gaps  observed A/BA: 0/384 <1e-3; null BA: 4/24576")


def check_lora_factor_audit(root: Path, errors: list[str]) -> None:
    """Check the all-24-module A/B/BA audit and its released-product cross-check."""
    path = root / "results" / "v7_lora_factor_audit.json"
    report = json.loads(path.read_text())
    equal(report["schema_version"], 2, "LoRA factor-audit schema", errors)
    implementation = root / "code" / "scripts" / "lora_factor_audit.py"
    equal(report["implementation_sha256"], sha256(implementation),
          "LoRA factor-audit implementation SHA-256", errors)
    pins_path = root / "upstream_revisions.json"
    equal(report["input_pin_record"]["sha256"], sha256(pins_path),
          "LoRA factor-audit pin-record SHA-256", errors)

    scope = report["scope"]
    equal(scope["unordered_task_pairs"], 28, "LoRA factor-audit pairs", errors)
    equal(scope["k"], 8, "LoRA factor-audit k", errors)
    equal(scope["ambient_dimension"], 768,
          "LoRA factor-audit ambient dimension", errors)
    primary_scope = scope["primary_all_24_modules"]
    equal(primary_scope["layers"], list(range(12)),
          "LoRA factor-audit primary layers", errors)
    equal(primary_scope["projections"], ["q_proj", "v_proj"],
          "LoRA factor-audit primary projections", errors)
    equal(primary_scope["modules_per_task"], 24,
          "LoRA factor-audit primary modules", errors)
    equal(primary_scope["cells_per_family"], 672,
          "LoRA factor-audit primary cells", errors)

    expected = {
        "rank16_lora": {
            "A_right": 0.8758931204598783,
            "B_left": 0.05399135030964878,
            "BA_right": 0.6075299837546246,
            "csv": "v7_mc64_b16_8task_q_v_lora-16_loranull.csv",
        },
        "rank16_linearized_lora": {
            "A_right": 0.9999999999999998,
            "B_left": 0.057049144648505606,
            "BA_right": 0.6648918856110606,
            "csv": "v7_mc64_b16_8task_q_v_l-lora-16_loranull.csv",
        },
    }
    families = {entry["family"]: entry for entry in report["families"]}
    equal(set(families), set(expected), "LoRA factor-audit families", errors)
    for family, values in expected.items():
        entry = families[family]
        primary = entry["primary_all_24_modules"]
        aggregates = primary["aggregate"]
        for metric in ("A_right", "B_left", "BA_right"):
            record = aggregates[metric]
            equal(record["n"], 672,
                  f"{family} factor-audit {metric} cells", errors)
            close(float(record["mean"]), float(values[metric]), 5e-15,
                  f"{family} factor-audit {metric} mean", errors)
            equal(record["paper_rounded_4"], f"{float(values[metric]):.4f}",
                  f"{family} factor-audit {metric} paper rounding", errors)

        crosscheck = primary["released_product_csv_crosscheck"]
        equal(crosscheck["file"], values["csv"],
              f"{family} factor-audit product CSV", errors)
        csv_path = root / "results" / str(values["csv"])
        equal(crosscheck["sha256"], sha256(csv_path),
              f"{family} factor-audit product CSV SHA-256", errors)
        rows = read_csv(csv_path)
        equal(len(rows), 672, f"{family} factor-audit released rows", errors)
        released_mean = mean(rows, "raw")
        close(float(crosscheck["released_raw_mean"]), released_mean, 5e-15,
              f"{family} factor-audit released raw mean", errors)
        equal(crosscheck["selected_rows"], 672,
              f"{family} factor-audit selected rows", errors)
        if not crosscheck["paper_rounding_matches"]:
            errors.append(f"{family} factor-audit BA paper rounding does not match CSV")

    print("LoRA factors     all-24-module A/B/BA audit and product rows checked")


def check_merge_accuracy(root: Path, errors: list[str]) -> None:
    path = root / "results" / "v7_merge_b32_accuracy.json"
    result = json.loads(path.read_text())
    tasks = ["dtd", "eurosat", "gtsrb", "mnist", "resisc45",
             "stanford-cars", "sun397", "svhn"]
    ranks = [8, 16, 32, 64, 128, 256]
    expected_settings = ({"zeroshot", "full"}
                         | {f"{kind}_{rank}" for kind in ("tau", "cross")
                            for rank in ranks})
    equal(set(result), expected_settings, "merge accuracy settings", errors)
    for setting, values in result.items():
        equal(list(values), tasks, f"merge {setting} task order", errors)
        for task, value in values.items():
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
                errors.append(f"merge {setting}/{task} accuracy is invalid")
            close(numeric * 5.0, round(numeric * 5.0), 2e-12,
                  f"merge {setting}/{task} 500-example grid", errors)

    expected_means = {
        "zeroshot": 38.875,
        "full": 59.475,
        **{f"tau_{rank}": value for rank, value in zip(
            ranks, (45.85, 49.9, 52.775, 55.7, 57.575, 58.95))},
        **{f"cross_{rank}": value for rank, value in zip(
            ranks, (45.25, 48.825, 51.375, 53.45, 54.775, 55.65))},
    }
    for setting, want in expected_means.items():
        close(stats.fmean(float(value) for value in result[setting].values()),
              want, 2e-12, f"merge {setting} mean", errors)
    tau_wins = sum(float(result["tau_256"][task]) >
                   float(result["cross_256"][task]) for task in tasks)
    equal(tau_wins, 8, "merge tau-vs-cross r256 task wins", errors)

    settings = json.loads((root / "run_settings.json").read_text())["runs"][
        "merge_accuracy_evaluation"
    ]
    equal(settings["record"], "results/v7_merge_b32_accuracy.json",
          "merge run-settings record", errors)
    equal(settings["model"], "openai/clip-vit-base-patch32",
          "merge run-settings model", errors)
    equal(settings["tasks"], tasks, "merge run-settings tasks", errors)
    equal(settings["ranks"], ranks, "merge run-settings ranks", errors)
    equal(settings["test_examples_per_task"], 500,
          "merge run-settings examples", errors)
    equal(settings["seed"], 0, "merge run-settings seed", errors)
    close(float(settings["lambda_scale"]), 0.3, 0.0,
          "merge run-settings lambda", errors)
    print("merge accuracy   14 settings x 8 tasks; means and tau r256 8/8 wins checked")


def check_sensitivity(root: Path, arm_values: dict[str, dict[str, float]],
                      errors: list[str]) -> None:
    expected_grid = set(itertools.product((4, 8, 16, 32), (16, 32, 64),
                                          (1.0, 2.0, 3.0)))
    for architecture, spec in SENSITIVITY.items():
        path = root / "results" / f"v7_exact_sensitivity_{architecture}_summary.csv"
        rows = read_csv(path)
        grid = {(int(row["k"]), int(row["null_m"]), float(row["block_z"]))
                for row in rows}
        equal(len(rows), 36, f"{architecture} sensitivity rows", errors)
        equal(grid, expected_grid, f"{architecture} sensitivity grid", errors)
        equal({int(row["n_tasks"]) for row in rows}, {spec["tasks"]},
              f"{architecture} sensitivity tasks", errors)
        equal({int(row["n_modules"]) for row in rows}, {spec["modules"]},
              f"{architecture} sensitivity modules", errors)
        equal({int(row["n_cells"]) for row in rows}, {spec["cells"]},
              f"{architecture} sensitivity cells", errors)
        values = [float(row["fraction_explained_pct"]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{architecture} sensitivity has non-finite values")
        k8 = [float(row["fraction_explained_pct"]) for row in rows
              if int(row["k"]) == 8]
        for got, want, suffix in zip((min(values), max(values)), spec["all_range"],
                                     ("min", "max")):
            close(got, want, 5e-10, f"{architecture} all-grid {suffix}", errors)
        for got, want, suffix in zip((min(k8), max(k8)), spec["k8_range"],
                                     ("min", "max")):
            close(got, want, 5e-10, f"{architecture} k=8 {suffix}", errors)

        by_key = {(int(row["k"]), int(row["null_m"]), float(row["block_z"])): row
                  for row in rows}
        headline = by_key[(8, 64, 2.0)]
        arm = arm_values[str(spec["label"])]
        for column, key in (("mean_raw", "raw"), ("mean_null_iso", "iso"),
                            ("mean_null_exact", "fitted"),
                            ("mean_excess_exact", "residual"),
                            ("fraction_explained_pct", "fraction")):
            close(float(headline[column]), arm[key], 5e-11,
                  f"{architecture} headline {column}", errors)

        # Reported qualitative trends: increasing m and decreasing z.
        for k in (4, 8, 16, 32):
            for z in (1.0, 2.0, 3.0):
                seq = [float(by_key[(k, m, z)]["fraction_explained_pct"])
                       for m in (16, 32, 64)]
                if any(right + 1e-10 < left for left, right in zip(seq, seq[1:])):
                    errors.append(f"{architecture}: fraction is not monotone in m at k={k}, z={z}")
            for m in (16, 32, 64):
                seq = [float(by_key[(k, m, z)]["fraction_explained_pct"])
                       for z in (1.0, 2.0, 3.0)]
                if any(right > left + 1e-10 for left, right in zip(seq, seq[1:])):
                    errors.append(f"{architecture}: fraction is not monotone in z at k={k}, m={m}")
        print(f"sensitivity {spec['label']:4s} all=[{min(values):.2f},{max(values):.2f}]% "
              f"k=8=[{min(k8):.2f},{max(k8):.2f}]%")


def check_calibration_source_subgroups(root: Path, errors: list[str]) -> None:
    report_path = root / "results" / "v7_calibration_source_subgroup_diagnostic.json"
    report = json.loads(report_path.read_text())
    equal(report["schema_version"], 1,
          "calibration-source subgroup schema", errors)
    sources = ["cifar10", "dtd", "eurosat", "gtsrb",
               "resisc45", "stl10", "sun397"]
    equal(report["calibration_source_tasks"], sources,
          "calibration-source subgroup task set", errors)
    if "no local paths" not in report["path_policy"].lower():
        errors.append("calibration-source subgroup report lacks path policy")
    specifications = {
        "ViT-B/16": ("v7_exact_clip-vit-base-patch16.csv.gz", 72,
                     (23.059243818353952, 26.563756913314517, 28.3274809902238)),
        "ViT-B/32": ("v7_exact_clip-vit-base-patch32.csv.gz", 72,
                     (20.382082842480493, 23.454617254480798, 24.545561566122547)),
        "ViT-L/14 q/v": ("v7_exact_clip-vit-large-patch14.csv.gz", 48,
                         (10.197176815202043, 12.639446810331139,
                          14.471765007293058)),
    }
    equal(set(report["architectures"]), set(specifications),
          "calibration-source subgroup architectures", errors)
    source_set = set(sources)
    for label, (filename, module_count, expected_fractions) in specifications.items():
        record = report["architectures"][label]
        equal(record["source_row_file"], filename,
              f"{label} subgroup source file", errors)
        if PurePosixPath(filename).name != filename:
            errors.append(f"{label} subgroup source is not a safe filename")
        rows = read_csv(root / "results" / filename)
        all_tasks = sorted({row[key] for row in rows for key in ("a", "b")})
        equal(record["all_tasks"], all_tasks, f"{label} subgroup tasks", errors)
        grouped = {count: [] for count in (0, 1, 2)}
        for row in rows:
            endpoints = int(row["a"] in source_set) + int(row["b"] in source_set)
            grouped[endpoints].append(row)
        expected_pairs = (78, 91, 21)
        for count, expected_fraction, pair_count in zip(
                (0, 1, 2), expected_fractions, expected_pairs):
            subset = grouped[count]
            raw = mean(subset, "raw")
            isotropic = mean(subset, "null_iso")
            fitted = mean(subset, "null_exact")
            pairs = {tuple(sorted((row["a"], row["b"]))) for row in subset}
            modules = {row["module"] for row in subset}
            computed = {
                "rows": len(subset),
                "unique_task_pairs": len(pairs),
                "modules": len(modules),
                "mean_raw": raw,
                "mean_isotropic": isotropic,
                "mean_activation_null": fitted,
                "residual_raw_minus_null": raw - fitted,
                "reproduced_fraction_percent":
                    100.0 * (fitted - isotropic) / (raw - isotropic),
            }
            stored = record["groups"][str(count)]
            equal(stored["rows"], computed["rows"],
                  f"{label} subgroup {count} rows", errors)
            equal(stored["unique_task_pairs"], pair_count,
                  f"{label} subgroup {count} pairs", errors)
            equal(stored["modules"], module_count,
                  f"{label} subgroup {count} modules", errors)
            for key in ("mean_raw", "mean_isotropic", "mean_activation_null",
                        "residual_raw_minus_null", "reproduced_fraction_percent"):
                close(float(stored[key]), float(computed[key]),
                      5e-10 if "percent" in key else 5e-12,
                      f"{label} subgroup {count} {key}", errors)
            close(float(stored["reproduced_fraction_percent"]), expected_fraction,
                  5e-10, f"{label} subgroup {count} headline", errors)

    provenance = json.loads((root / "code" / "SOURCE_PROVENANCE.json").read_text())
    implementation = "scripts/calibration_source_subgroup_diagnostic.py"
    equal(report["implementation_sha256"],
          provenance["archived_execution_sha256"][implementation],
          "calibration-source subgroup implementation hash", errors)
    equal(sha256(root / "code" / implementation), report["implementation_sha256"],
          "calibration-source subgroup shipped-source hash", errors)
    settings = json.loads((root / "run_settings.json").read_text())["runs"][
        "calibration_source_endpoint_subgroups"
    ]
    equal(settings["record"],
          "results/v7_calibration_source_subgroup_diagnostic.json",
          "calibration-source subgroup run-settings record", errors)
    equal(settings["calibration_source_tasks"], sources,
          "calibration-source subgroup run-settings tasks", errors)
    equal(settings["groups"], [0, 1, 2],
          "calibration-source subgroup run-settings groups", errors)
    print("calibration groups B16=23.06/26.56/28.33%; "
          "B32=20.38/23.45/24.55%; L14=10.20/12.64/14.47%")


def _lines_sha256(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_calibration_resampling(root: Path,
                                 arm_values: dict[str, dict[str, float]],
                                 errors: list[str]) -> None:
    experiment_root = root / "results" / "calibration_resample_b16"
    summary = json.loads((experiment_root / "summary.json").read_text())
    equal(summary["schema_version"], 1, "calibration-refit schema", errors)
    equal(summary["status"], "complete", "calibration-refit status", errors)
    configuration = summary["calibration_resampling"]
    sources = ["cifar10", "dtd", "eurosat", "gtsrb",
               "resisc45", "stl10", "sun397"]
    seeds = [20260916, 20260917]
    equal(configuration["sources"], sources, "calibration-refit sources", errors)
    equal(configuration["seeds"], seeds, "calibration-refit seeds", errors)
    equal(configuration["draws_per_source_with_replacement"], 143,
          "calibration-refit draws/source", errors)
    equal(configuration["source_pool_size"], 143,
          "calibration-refit source pool size", errors)
    equal(configuration["total_images_per_draw"], 1001,
          "calibration-refit total images", errors)
    equal(configuration["folds"], 8, "calibration-refit folds", errors)
    equal(summary["conditional_interval"]["seed"], 20260918,
          "calibration-refit conditional bootstrap seed", errors)
    if "not a calibration-resampling confidence interval" not in \
            summary["conditional_interval"]["warning"].lower():
        errors.append("calibration-refit conditional interval lacks scope warning")

    configured = json.loads((root / "run_settings.json").read_text())["runs"][
        "calibration_resampling_robustness"
    ]
    equal(configured["sources"], sources,
          "calibration-refit run-settings sources", errors)
    equal(configured["refit_seeds"], seeds,
          "calibration-refit run-settings seeds", errors)
    equal(configured["source_pool_size_each"], 143,
          "calibration-refit run-settings pool size", errors)
    equal(configured["draws_per_source_each_refit"], 143,
          "calibration-refit run-settings draws/source", errors)
    equal(configured["total_images_each_refit"], 1001,
          "calibration-refit run-settings total images", errors)
    equal(configured["folds"], 8,
          "calibration-refit run-settings folds", errors)
    equal(configured["conditional_crossed_bootstrap_draws"], 50000,
          "calibration-refit run-settings conditional draws", errors)
    equal(configured["conditional_crossed_bootstrap_seed"], 20260918,
          "calibration-refit run-settings conditional seed", errors)
    equal(configured["record"], "results/calibration_resample_b16/summary.json",
          "calibration-refit run-settings record", errors)

    provenance = summary["provenance"]
    equal(provenance["path_sanitized"], True,
          "calibration-refit path sanitation flag", errors)
    equal(provenance["artifact_root"], "results/calibration_resample_b16",
          "calibration-refit artifact root", errors)
    upstream = json.loads((root / "upstream_revisions.json").read_text())
    specialist = upstream["full_finetuning_specialists"]["clip-vit-base-patch16"]
    expected_experts = {
        task: f"{specialist['repository_template'].format(task=task)}@{revision}"
        for task, revision in specialist["revision_by_task"].items()
    }
    equal(provenance["expert_specs"], expected_experts,
          "calibration-refit expert revisions", errors)
    for name, digest in provenance["implementations_sha256"].items():
        if (not name or not isinstance(digest, str) or len(digest) != 64
                or not all(character in "0123456789abcdef" for character in digest)):
            errors.append(f"calibration-refit implementation hash is malformed: {name!r}")

    reference = summary["reference"]
    equal(reference["logical_path"],
          "results/v7_exact_clip-vit-base-patch16.csv.gz",
          "calibration-refit reference path", errors)
    reference_path = root / reference["logical_path"]
    equal(csv_content_sha256(reference_path), reference["sha256"],
          "calibration-refit reference SHA-256", errors)
    close(float(reference["activation_explained_percent"]),
          arm_values["B16"]["fraction"], 5e-9,
          "calibration-refit reference fraction", errors)
    reference_rows = {
        (row["module"], row["a"], row["b"]): row
        for row in read_csv(reference_path)
    }

    expected_intervals = {
        20260916: ((19.169601007907367, 31.538477332599527),
                   (0.04546998877077615, 0.10591661894124046)),
        20260917: ((19.305410540628895, 31.75874509305432),
                   (0.045349028814769606, 0.1056358413116072)),
    }
    fractions = []
    refits = summary["bootstrap_refits"]
    for seed in seeds:
        label = f"seed_{seed}"
        record = refits[label]
        equal(record["seed"], seed, f"calibration refit {seed} seed", errors)
        csv_name = f"refit_seed{seed}.csv.gz"
        map_name = f"resample_map_seed{seed}.json"
        csv_path = experiment_root / csv_name
        map_path = experiment_root / map_name
        declared = provenance["artifact_sha256"]
        equal(csv_content_sha256(csv_path), declared[csv_name],
              f"calibration refit {seed} CSV SHA-256", errors)
        equal(sha256(map_path), declared[map_name],
              f"calibration refit {seed} map SHA-256", errors)

        occurrence_map = json.loads(map_path.read_text())
        equal(occurrence_map["schema_version"], 1,
              f"calibration map {seed} schema", errors)
        equal(occurrence_map["seed"], seed,
              f"calibration map {seed} seed", errors)
        equal(occurrence_map["sources"], sources,
              f"calibration map {seed} sources", errors)
        equal(occurrence_map["per_source"], 143,
              f"calibration map {seed} draws/source", errors)
        equal(occurrence_map["n_items"], 1001,
              f"calibration map {seed} items", errors)
        equal(occurrence_map["n_folds"], 8,
              f"calibration map {seed} folds", errors)
        equal(occurrence_map["pool_sizes"], {source: 143 for source in sources},
              f"calibration map {seed} pool sizes", errors)
        map_rows = occurrence_map["records"]
        equal(len(map_rows), 1001, f"calibration map {seed} records", errors)
        equal({int(row["ordinal"]) for row in map_rows}, set(range(1001)),
              f"calibration map {seed} ordinals", errors)
        fold_sizes = {f"f{fold}": 0 for fold in range(8)}
        selected: dict[str, list[int]] = {source: [] for source in sources}
        selected_lines = []
        for row in map_rows:
            source = str(row["source"])
            draw_index = int(row["draw_index"])
            source_index = sources.index(source)
            pool_index = int(row["pool_index"])
            ordinal = draw_index * len(sources) + source_index
            fold = f"f{ordinal % 8}"
            equal(int(row["ordinal"]), ordinal,
                  f"calibration map {seed} ordinal {ordinal}", errors)
            equal(row["fold"], fold,
                  f"calibration map {seed} fold {ordinal}", errors)
            if not (0 <= draw_index < 143 and 0 <= pool_index < 143):
                errors.append(f"calibration map {seed} has out-of-range draw/pool index")
            fold_sizes[fold] += 1
            selected[source].append(pool_index)
            selected_lines.append(
                f"{ordinal}\t{source}\t{draw_index}\t{pool_index}\t"
                f"{row['source_file']}\t{fold}"
            )
        equal(fold_sizes, occurrence_map["fold_sizes"],
              f"calibration map {seed} fold sizes", errors)
        unique = {source: len(set(indices)) for source, indices in selected.items()}
        duplicates = {source: 143 - count for source, count in unique.items()}
        equal(unique, occurrence_map["unique_targets_per_source"],
              f"calibration map {seed} unique targets", errors)
        equal(duplicates, occurrence_map["duplicate_occurrences_per_source"],
              f"calibration map {seed} duplicate occurrences", errors)
        equal(unique, record["unique_targets_per_source"],
              f"calibration summary {seed} unique targets", errors)
        equal(duplicates, record["duplicate_occurrences_per_source"],
              f"calibration summary {seed} duplicates", errors)
        equal(_lines_sha256(selected_lines), occurrence_map["selected_occurrences_sha256"],
              f"calibration map {seed} selected-occurrence digest", errors)
        equal(occurrence_map["selected_occurrences_sha256"],
              record["selected_occurrences_sha256"],
              f"calibration summary {seed} occurrence digest", errors)
        for source in sources:
            source_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()[:8],
                "big",
            )
            generator = random.Random(source_seed)
            expected_draws = [generator.randrange(143) for _ in range(143)]
            equal(selected[source], expected_draws,
                  f"calibration map {seed} RNG replay {source}", errors)

        rows = read_csv(csv_path)
        equal(len(rows), 13680, f"calibration refit {seed} rows", errors)
        tasks = {row[key] for row in rows for key in ("a", "b")}
        pairs = {tuple(sorted((row["a"], row["b"]))) for row in rows}
        modules = {row["module"] for row in rows}
        equal(len(tasks), 20, f"calibration refit {seed} tasks", errors)
        equal(len(pairs), 190, f"calibration refit {seed} pairs", errors)
        equal(len(modules), 72, f"calibration refit {seed} modules", errors)
        equal({row["null_kind"] for row in rows}, {"exact_haar_block"},
              f"calibration refit {seed} null kind", errors)
        keyed = {(row["module"], row["a"], row["b"]): row for row in rows}
        equal(set(keyed), set(reference_rows),
              f"calibration refit {seed} cell keys", errors)
        for key, row in keyed.items():
            reference_row = reference_rows[key]
            close(float(row["raw"]), float(reference_row["raw"]), 0.0,
                  f"calibration refit {seed} raw cell {key}", errors)
            close(float(row["null_iso"]), float(reference_row["null_iso"]), 0.0,
                  f"calibration refit {seed} isotropic cell {key}", errors)
            raw_i = float(row["raw"])
            iso_i = float(row["null_iso"])
            fitted_i = float(row["null_exact"])
            close(float(row["frac_num"]), fitted_i - iso_i, 2e-9,
                  f"calibration refit {seed} numerator {key}", errors)
            close(float(row["frac_den"]), raw_i - iso_i, 2e-9,
                  f"calibration refit {seed} denominator {key}", errors)
            close(float(row["excess_exact"]), raw_i - fitted_i, 2e-9,
                  f"calibration refit {seed} residual {key}", errors)

        raw = mean(rows, "raw")
        iso = mean(rows, "null_iso")
        fitted = mean(rows, "null_exact")
        residual = raw - fitted
        fraction = 100.0 * (fitted - iso) / (raw - iso)
        fractions.append(fraction)
        headline = record["headline"]
        for key, got in (("mean_raw", raw), ("mean_null_isotropic", iso),
                         ("mean_null_activation", fitted),
                         ("mean_residual", residual),
                         ("activation_explained_percent", fraction),
                         ("activation_explained_fraction", fraction / 100.0)):
            close(float(headline[key]), got, 5e-10 if "percent" in key else 5e-12,
                  f"calibration refit {seed} {key}", errors)
        blocks = [int(row["n_blocks"]) for row in rows]
        equal(headline["n_blocks"], {
            "min": min(blocks), "median": stats.median(blocks), "max": max(blocks)
        }, f"calibration refit {seed} block counts", errors)
        covariance = record["covariance"]
        equal(covariance["full"]["n_items"], 1001,
              f"calibration refit {seed} covariance items", errors)
        equal(covariance["full"]["n_modules"], 72,
              f"calibration refit {seed} covariance modules", errors)
        equal([fold["n_items"] for fold in covariance["folds"]],
              [126, 125, 125, 125, 125, 125, 125, 125],
              f"calibration refit {seed} covariance fold sizes", errors)
        interval = record["conditional_crossed_task_layer_percentile_95_interval"]
        for got, want, suffix in zip(interval["exact_reproduced_fraction_percent"],
                                     expected_intervals[seed][0], ("lo", "hi")):
            close(float(got), want, 5e-10,
                  f"calibration refit {seed} fraction CI {suffix}", errors)
        for got, want, suffix in zip(interval["residual"],
                                     expected_intervals[seed][1], ("lo", "hi")):
            close(float(got), want, 5e-12,
                  f"calibration refit {seed} residual CI {suffix}", errors)

    observed = summary["observed_robustness"]
    reference_fraction = arm_values["B16"]["fraction"]
    computed_observed = {
        "activation_explained_percent_min": min(fractions),
        "activation_explained_percent_max": max(fractions),
        "mean_across_two_refits_percent": stats.fmean(fractions),
        "range_width_percentage_points": max(fractions) - min(fractions),
        "maximum_absolute_change_from_reference_percentage_points":
            max(abs(value - reference_fraction) for value in fractions),
    }
    for key, value in computed_observed.items():
        close(float(observed[key]), value, 5e-9,
              f"calibration-refit robustness {key}", errors)
    for got, want, suffix in zip(configured["observed_fraction_percent_range"],
                                 (min(fractions), max(fractions)), ("minimum", "maximum")):
        close(float(got), want, 5e-9,
              f"calibration-refit run-settings range {suffix}", errors)
    close(float(configured["maximum_absolute_change_from_reference_percentage_points"]),
          computed_observed["maximum_absolute_change_from_reference_percentage_points"],
          5e-9, "calibration-refit run-settings maximum change", errors)
    print(f"calibration refit B16 range=[{min(fractions):.3f},{max(fractions):.3f}]% "
          f"max |delta|={computed_observed['maximum_absolute_change_from_reference_percentage_points']:.3f}pp")


def check_calibration_composition(root: Path,
                                  arm_values: dict[str, dict[str, float]],
                                  errors: list[str]) -> None:
    experiment_root = root / "results" / "calibration_composition_b16"
    summary = json.loads((experiment_root / "summary.json").read_text())
    occurrence_map = json.loads((experiment_root / "composition_map.json").read_text())
    rows_path = experiment_root / "exact_b16_20task_72module.csv.gz"
    equal(summary["schema_version"], 1, "composition-control schema", errors)
    equal(summary["status"], "complete", "composition-control status", errors)
    equal(summary["experiment"], "B/16 alternative calibration-composition control",
          "composition-control experiment", errors)
    if "not uncertainty over possible calibration compositions" not in \
            summary["interval_warning"].lower():
        errors.append("composition-control conditional interval lacks scope warning")
    if "not a bootstrap draw" not in summary["separation_from_bootstrap_refits"].lower():
        errors.append("composition control is not separated from calibration refits")

    sources = ["mnist", "stanford_cars", "svhn"]
    source_counts = {"mnist": 334, "stanford_cars": 334, "svhn": 333}
    source_details = {
        "mnist": {
            "dataset_id": "ylecun/mnist", "config": "mnist", "split": "test",
            "builder_fingerprint": "77f3279092a1c1579b2250db8eafed0ad422088c",
            "split_rows": 10000,
            "arrow_shards": [
                ("mnist-test.arrow", 2949616,
                 "c1271fa343652fa634d0fb45544b526d89d6a552d18b14a3a05980899a7141fb")
            ],
        },
        "stanford_cars": {
            "dataset_id": "tanganke/stanford_cars", "config": "default",
            "split": "test",
            "builder_fingerprint": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
            "split_rows": 8041,
            "arrow_shards": [
                ("stanford_cars-test-00000-of-00002.arrow", 501769016,
                 "f93eaec0f687268607af546a7cc0d36b496d66485ea0f5a6681b1393eb91ad13"),
                ("stanford_cars-test-00001-of-00002.arrow", 493171160,
                 "f120f8dedf8ac0fec9416b7426c80cd8574b45d147a4627cbc8605b6821f19d4"),
            ],
        },
        "svhn": {
            "dataset_id": "ufldl-stanford/svhn", "config": "cropped_digits",
            "split": "test",
            "builder_fingerprint": "f9e1717d73324ebbc1ece84267ccd80d0aca0690",
            "split_rows": 26032,
            "arrow_shards": [
                ("svhn-test.arrow", 44549912,
                 "e636de302053f47422fac68d0bfa71385507f30f2328d2c9ff84cfd3fda5615d")
            ],
        },
    }

    sampling = summary["sampling"]
    equal(sampling["seed"], 20260919, "composition-control seed", errors)
    equal(sampling["counts"], source_counts, "composition-control counts", errors)
    equal(sampling["total"], 1001, "composition-control sample size", errors)
    equal(sampling["folds"], 8, "composition-control folds", errors)
    if "without replacement" not in sampling["method"].lower():
        errors.append("composition-control sampling is not without replacement")

    settings = json.loads((root / "run_settings.json").read_text())["runs"][
        "calibration_composition_control"
    ]
    equal(settings["sources"], sources, "composition-control run-settings sources", errors)
    equal(settings["source_counts"], source_counts,
          "composition-control run-settings counts", errors)
    equal(settings["total_images"], 1001,
          "composition-control run-settings sample size", errors)
    equal(settings["folds"], 8, "composition-control run-settings folds", errors)
    equal(settings["seed"], 20260919, "composition-control run-settings seed", errors)
    equal(settings["summary_record"],
          "results/calibration_composition_b16/summary.json",
          "composition-control summary record", errors)
    equal(settings["map_record"],
          "results/calibration_composition_b16/composition_map.json",
          "composition-control map record", errors)
    equal(settings["row_record"],
          "results/calibration_composition_b16/exact_b16_20task_72module.csv.gz",
          "composition-control row record", errors)
    equal(settings["conditional_crossed_bootstrap_seed"], 20260915,
          "composition-control bootstrap seed", errors)
    equal(settings["conditional_crossed_bootstrap_record"],
          "results/calibration_composition_b16/bootstrap_task_layer_50000.json",
          "composition-control bootstrap record", errors)
    equal(settings["images_in_bundle"], False,
          "composition-control image redistribution setting", errors)

    equal(occurrence_map["schema_version"], 1,
          "composition-control map schema", errors)
    equal(occurrence_map["seed"], 20260919,
          "composition-control map seed", errors)
    equal(occurrence_map["source_order"], sources,
          "composition-control map source order", errors)
    equal(occurrence_map["source_counts"], source_counts,
          "composition-control map source counts", errors)
    equal(occurrence_map["n_items"], 1001,
          "composition-control map items", errors)
    equal(occurrence_map["n_folds"], 8,
          "composition-control map folds", errors)
    equal(occurrence_map["fold_sizes"],
          {"f0": 126, **{f"f{index}": 125 for index in range(1, 8)}},
          "composition-control map fold sizes", errors)
    equal(occurrence_map["software"], {"datasets": "5.0.1", "pillow": "12.3.0"},
          "composition-control map software", errors)
    if "no source images" not in occurrence_map["redistribution"].lower():
        errors.append("composition-control map lacks image redistribution warning")
    if summary["license_scope"]["images_redistributed"] is not False:
        errors.append("composition-control summary says images were redistributed")

    pins = json.loads((root / "upstream_revisions.json").read_text())[
        "calibration_composition_datasets"
    ]
    for source in sources:
        expected = source_details[source]
        pin = pins[source]
        equal(pin, {
            "repository": expected["dataset_id"], "config": expected["config"],
            "split": expected["split"], "revision": expected["builder_fingerprint"],
        }, f"composition-control upstream pin {source}", errors)
        for container_label, container in (
                ("map", occurrence_map["sources"]),
                ("summary", summary["source_provenance"])):
            record = container[source]
            for key in ("dataset_id", "config", "split", "builder_fingerprint",
                        "split_rows"):
                equal(record[key], expected[key],
                      f"composition-control {container_label} {source} {key}", errors)
            equal(record["selected_without_replacement"], source_counts[source],
                  f"composition-control {container_label} {source} selected count", errors)
            shards = [(entry["name"], entry["bytes"], entry["sha256"])
                      for entry in record["arrow_shards"]]
            equal(shards, expected["arrow_shards"],
                  f"composition-control {container_label} {source} Arrow shards", errors)

    map_rows = occurrence_map["records"]
    equal(len(map_rows), 1001, "composition-control map record count", errors)
    selected: dict[str, list[int]] = {source: [] for source in sources}
    expected_schedule = [
        (source, draw_index)
        for draw_index in range(max(source_counts.values()))
        for source in sources
        if draw_index < source_counts[source]
    ]
    equal(len(expected_schedule), 1001,
          "composition-control expected schedule size", errors)
    for ordinal, (row, (expected_source, expected_draw)) in enumerate(
            zip(map_rows, expected_schedule)):
        equal(row["ordinal"], ordinal,
              f"composition-control map ordinal {ordinal}", errors)
        equal(row["source"], expected_source,
              f"composition-control map source {ordinal}", errors)
        equal(row["draw_index"], expected_draw,
              f"composition-control map draw {ordinal}", errors)
        equal(row["fold"], f"f{ordinal % 8}",
              f"composition-control map fold {ordinal}", errors)
        source_index = int(row["source_index"])
        if not 0 <= source_index < source_details[expected_source]["split_rows"]:
            errors.append(f"composition-control source index is out of range at {ordinal}")
        selected[expected_source].append(source_index)
        label = int(row["label"])
        if expected_source in {"mnist", "svhn"} and not 0 <= label <= 9:
            errors.append(f"composition-control digit label is out of range at {ordinal}")
        if expected_source == "stanford_cars" and not 0 <= label <= 195:
            errors.append(f"composition-control car label is out of range at {ordinal}")
        expected_name = (
            f"draw_{expected_draw:03d}__index_{source_index:06d}__label_{label}.png"
        )
        equal(row["image_file"], expected_name,
              f"composition-control image name {ordinal}", errors)
        image_digest = row["image_sha256"]
        if (not isinstance(image_digest, str) or len(image_digest) != 64
                or not all(character in "0123456789abcdef" for character in image_digest)):
            errors.append(f"composition-control image hash is malformed at {ordinal}")
    for source in sources:
        generator = random.Random(int.from_bytes(
            hashlib.sha256(f"20260919:{source}".encode("utf-8")).digest()[:8], "big"
        ))
        expected_indices = generator.sample(
            range(source_details[source]["split_rows"]), source_counts[source]
        )
        equal(selected[source], expected_indices,
              f"composition-control RNG replay {source}", errors)
        equal(len(set(selected[source])), source_counts[source],
              f"composition-control without-replacement uniqueness {source}", errors)

    provenance = summary["provenance"]
    equal(provenance["path_sanitized"], True,
          "composition-control path sanitation flag", errors)
    equal(provenance["artifact_root"], "results/calibration_composition_b16",
          "composition-control artifact root", errors)
    declared_hashes = provenance["artifact_sha256"]
    equal(sha256(experiment_root / "composition_map.json"),
          declared_hashes["composition_map.json"],
          "composition-control map SHA-256", errors)
    equal(csv_content_sha256(rows_path),
          declared_hashes["exact_b16_20task_72module.csv.gz"],
          "composition-control row SHA-256", errors)
    specialists = json.loads((root / "upstream_revisions.json").read_text())[
        "full_finetuning_specialists"
    ]["clip-vit-base-patch16"]
    expected_experts = {
        task: f"{specialists['repository_template'].format(task=task)}@{revision}"
        for task, revision in specialists["revision_by_task"].items()
    }
    equal(provenance["expert_specs"], expected_experts,
          "composition-control expert revisions", errors)
    source_provenance = json.loads((root / "code" / "SOURCE_PROVENANCE.json").read_text())
    archived = source_provenance["archived_execution_sha256"]
    implementation_map = {
        "actnull/covariance.py": "actnull/covariance.py",
        "actnull/null.py": "actnull/null.py",
        "materialize_composition_control.py": "scripts/materialize_composition_control.py",
        "run_resample_exact.py": "scripts/run_resample_exact.py",
        "scripts/bootstrap_overlap.py": "scripts/bootstrap_overlap.py",
        "scripts/exact_overlap.py": "scripts/exact_overlap.py",
    }
    equal(set(provenance["implementations_sha256"]), set(implementation_map),
          "composition-control implementation provenance keys", errors)
    for result_name, source_name in implementation_map.items():
        equal(provenance["implementations_sha256"][result_name], archived[source_name],
              f"composition-control implementation hash {source_name}", errors)

    reference = summary["reference"]
    equal(reference["logical_path"], "results/v7_exact_clip-vit-base-patch16.csv.gz",
          "composition-control reference path", errors)
    equal(csv_content_sha256(root / reference["logical_path"]), reference["sha256"],
          "composition-control reference SHA-256", errors)

    rows = read_csv(rows_path)
    equal(len(rows), 13680, "composition-control row count", errors)
    keyed = {(row["module"], row["a"], row["b"]): row for row in rows}
    equal(len(keyed), 13680, "composition-control unique cells", errors)
    tasks = {row[key] for row in rows for key in ("a", "b")}
    pairs = {tuple(sorted((row["a"], row["b"]))) for row in rows}
    modules = {row["module"] for row in rows}
    equal((len(tasks), len(pairs), len(modules)), (20, 190, 72),
          "composition-control task/pair/module scope", errors)
    equal({row["null_kind"] for row in rows}, {"exact_haar_block"},
          "composition-control null kind", errors)
    reference_rows = {
        (row["module"], row["a"], row["b"]): row
        for row in read_csv(root / reference["logical_path"])
    }
    equal(set(keyed), set(reference_rows),
          "composition-control reference cell keys", errors)
    for key, row in keyed.items():
        base_row = reference_rows[key]
        close(float(row["raw"]), float(base_row["raw"]), 0.0,
              f"composition-control raw cell {key}", errors)
        close(float(row["null_iso"]), float(base_row["null_iso"]), 0.0,
              f"composition-control isotropic cell {key}", errors)
        raw_i = float(row["raw"])
        iso_i = float(row["null_iso"])
        null_i = float(row["null_exact"])
        close(float(row["frac_num"]), null_i - iso_i, 2e-9,
              f"composition-control numerator {key}", errors)
        close(float(row["frac_den"]), raw_i - iso_i, 2e-9,
              f"composition-control denominator {key}", errors)
        close(float(row["excess_exact"]), raw_i - null_i, 2e-9,
              f"composition-control residual {key}", errors)

    raw = mean(rows, "raw")
    iso = mean(rows, "null_iso")
    fitted = mean(rows, "null_exact")
    residual = raw - fitted
    fraction = 100.0 * (fitted - iso) / (raw - iso)
    expected_values = {
        "mean_raw": 0.10457364820564787,
        "mean_null_isotropic": 0.009114583333333334,
        "mean_null_activation": 0.02962087574161484,
        "mean_residual": 0.07495277246403304,
        "activation_explained_percent": 21.481765441255057,
        "activation_explained_fraction": 0.21481765441255055,
    }
    headline = summary["headline"]
    computed = {
        "mean_raw": raw, "mean_null_isotropic": iso,
        "mean_null_activation": fitted, "mean_residual": residual,
        "activation_explained_percent": fraction,
        "activation_explained_fraction": fraction / 100.0,
    }
    for key, want in expected_values.items():
        close(computed[key], want, 5e-10 if "percent" in key else 5e-12,
              f"composition-control recomputed {key}", errors)
        close(float(headline[key]), computed[key],
              5e-10 if "percent" in key else 5e-12,
              f"composition-control headline {key}", errors)
    equal(headline["n_blocks"], {
        "min": min(int(row["n_blocks"]) for row in rows),
        "median": stats.median(int(row["n_blocks"]) for row in rows),
        "max": max(int(row["n_blocks"]) for row in rows),
    }, "composition-control block counts", errors)
    interval = summary["conditional_crossed_task_layer_percentile_95_interval"]
    expected_interval = {
        "exact_reproduced_fraction_percent": [16.897537228735672, 26.268987267650733],
        "residual": [0.04860552331363323, 0.10901518198309416],
    }
    for metric, want_pair in expected_interval.items():
        for got, want, suffix in zip(interval[metric], want_pair, ("lo", "hi")):
            close(float(got), want, 5e-10 if "percent" in metric else 5e-12,
                  f"composition-control crossed {metric} {suffix}", errors)
    close(float(reference["activation_explained_percent"]),
          arm_values["B16"]["fraction"], 5e-9,
          "composition-control reference fraction", errors)
    change = fraction - float(reference["activation_explained_percent"])
    close(float(reference["change_percentage_points"]), change, 5e-9,
          "composition-control reference change", errors)
    close(float(settings["activation_explained_percent"]), fraction, 5e-9,
          "composition-control run-settings fraction", errors)
    close(float(settings["change_from_reference_percentage_points"]), change, 5e-9,
          "composition-control run-settings change", errors)
    for got, want, suffix in zip(settings["conditional_crossed_fraction_percent_interval"],
                                 expected_interval["exact_reproduced_fraction_percent"],
                                 ("lo", "hi")):
        close(float(got), want, 5e-10,
              f"composition-control run-settings CI {suffix}", errors)

    bootstrap_metadata = summary["conditional_crossed_task_layer_bootstrap"]
    equal(bootstrap_metadata["record"], "bootstrap_task_layer_50000.json",
          "composition-control summary bootstrap record", errors)
    equal(bootstrap_metadata["seed"], 20260915,
          "composition-control summary bootstrap seed", errors)
    equal(bootstrap_metadata["draws"], 50000,
          "composition-control summary bootstrap draws", errors)
    equal(bootstrap_metadata["rejected_draws"], 0,
          "composition-control summary bootstrap rejects", errors)
    bootstrap_path = experiment_root / bootstrap_metadata["record"]
    equal(sha256(bootstrap_path), bootstrap_metadata["sha256"],
          "composition-control bootstrap SHA-256", errors)
    equal(sha256(rows_path), bootstrap_metadata["input_compressed_sha256"],
          "composition-control bootstrap input SHA-256", errors)
    bootstrap = json.loads(bootstrap_path.read_text())
    equal(bootstrap["schema_version"], 1,
          "composition-control bootstrap schema", errors)
    equal(bootstrap["method"]["seed_reset_for_each_input"], 20260915,
          "composition-control bootstrap report seed", errors)
    equal(bootstrap["method"]["valid_replicates_per_scheme"], 50000,
          "composition-control bootstrap report draws", errors)
    equal(len(bootstrap["datasets"]), 1,
          "composition-control bootstrap dataset count", errors)
    bootstrap_dataset = bootstrap["datasets"][0]
    equal(bootstrap_dataset["label"], "composition_control",
          "composition-control bootstrap label", errors)
    equal(bootstrap_dataset["input"],
          "artifact/results/calibration_composition_b16/exact_b16_20task_72module.csv.gz",
          "composition-control bootstrap input", errors)
    equal(bootstrap_dataset["sha256"], sha256(rows_path),
          "composition-control bootstrap source hash", errors)
    close(float(bootstrap_dataset["point_estimate"]["exact_reproduced_fraction_percent"]),
          fraction, 5e-9, "composition-control bootstrap point fraction", errors)
    close(float(bootstrap_dataset["point_estimate"]["residual"]), residual, 5e-12,
          "composition-control bootstrap point residual", errors)
    bootstrap_joint = bootstrap_dataset["percentile_95_intervals"]["joint_task_layer"]
    for metric, want_pair in expected_interval.items():
        for got, want, suffix in zip(bootstrap_joint[metric], want_pair, ("lo", "hi")):
            close(float(got), want, 5e-10 if "percent" in metric else 5e-12,
                  f"composition-control bootstrap {metric} {suffix}", errors)
    for scheme in ("joint_task_layer", "task_only", "layer_only"):
        equal(bootstrap_dataset["replicates"][scheme],
              {"attempted": 50000, "rejected": 0, "valid": 50000},
              f"composition-control bootstrap {scheme} replicates", errors)

    covariance = summary["covariance"]
    equal(covariance["full"]["exact_covariance"], True,
          "composition-control exact covariance", errors)
    equal(covariance["full"]["n_items"], 1001,
          "composition-control covariance items", errors)
    equal(covariance["full"]["n_modules"], 72,
          "composition-control covariance modules", errors)
    equal(covariance["full"]["stored_eigendirections"], 64,
          "composition-control stored eigendirections", errors)
    equal(covariance["full"]["participation_ratio"]["modules_below_64"], 71,
          "composition-control modules below PR=64", errors)
    equal([fold["n_items"] for fold in covariance["folds"]],
          [126, 125, 125, 125, 125, 125, 125, 125],
          "composition-control covariance fold sizes", errors)
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    bundled_images = [path for path in experiment_root.rglob("*")
                      if path.is_file() and path.suffix.lower() in image_suffixes]
    equal(bundled_images, [], "composition-control redistributed images", errors)
    print(f"composition B16 fraction={fraction:.3f}% residual={residual:.6f} "
          f"CI=[{interval['exact_reproduced_fraction_percent'][0]:.3f},"
          f"{interval['exact_reproduced_fraction_percent'][1]:.3f}]%")


def _power_rows(path: Path) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    data = json.loads(path.read_text())
    return data, {int(row["planted"]): row for row in data["rows"]}


def check_power(root: Path, errors: list[str]) -> None:
    configurations = (
        ("activation", "v7_power_activation_current.json", -0.0006233223990420832,
         {1: (93.45005621505851, 99.59252623489648),
          2: (96.52965923946289, 99.36101036904356),
          4: (98.29573535952777, 99.3486358599417)}),
        ("activation initial", "v7_power_activation_initial.json", 3.9816720527597466e-05,
         {1: (93.45005621505851, 45.629427967989855),
          2: (96.52965923946289, 49.45844101225911),
          4: (98.29573535952777, 51.729348026222105)}),
        ("LoRA", "v7_power_lora_current.json", -0.007260120784242929,
         {1: (37.38603293069146, 101.11692431590735),
          2: (49.789918523629005, 101.07784580041537),
          4: (57.04254212072588, 100.70126093464108)}),
        ("LoRA initial", "v7_power_lora_initial.json", -0.14845554033915204,
         {1: (37.38603293069146, 99.99985110071883),
          2: (49.789918523629005, 99.99995180826254),
          4: (57.04254212072588, 99.99995709426948)}),
    )
    for label, filename, base_level, expected in configurations:
        _data, rows = _power_rows(root / "results" / filename)
        equal(set(rows), {0, 1, 2, 4}, f"{label} planted ranks", errors)
        base = rows[0]
        close(float(base["excess"]), base_level, 5e-13,
              f"{label} s=0 level", errors)
        o0 = float(base.get("O0", base["raw"]))
        for planted, (want_delivery, want_recovery) in expected.items():
            row = rows[planted]
            k = int(row["k"])
            truth = (planted / k) * (1.0 - o0)
            delivered_overlap = float(row["raw"]) - float(base["raw"])
            delivery = 100.0 * delivered_overlap / truth
            recovery = (100.0 * (float(row["excess"]) - float(base["excess"]))
                        / delivered_overlap)
            close(float(row["truth"]), truth, 5e-13,
                  f"{label} s={planted} nominal increment", errors)
            close(float(row["delivered_overlap"]), delivered_overlap, 5e-13,
                  f"{label} s={planted} delivered overlap", errors)
            close(float(row["delivered"]), delivery, 5e-10,
                  f"{label} s={planted} stored delivery", errors)
            close(float(row["recovery"]), recovery, 5e-10,
                  f"{label} s={planted} stored recovery", errors)
            close(delivery, want_delivery, 5e-9,
                  f"{label} s={planted} delivery", errors)
            close(recovery, want_recovery, 5e-9,
                  f"{label} s={planted} recovery", errors)
        print(f"power {label:18s} X0={float(base['excess']):+.6f}")


def check_ablation(root: Path, errors: list[str]) -> None:
    rows = json.loads((root / "results" / "v7_ablation_level_centered.json").read_text())
    expected = {
        "initial (neither correction)": 49.00242396481453,
        "singleton signs only": 77.73091536957362,
        "full complement only": 89.21708993662038,
        "corrected (both)": 99.30397760794425,
    }
    equal({row["variant"] for row in rows}, set(expected), "ablation variants", errors)
    for row in rows:
        delivered = float(row["raw2"]) - float(row["raw0"])
        recovery = 100.0 * ((float(row["excess2"]) - float(row["excess0"]))
                            / delivered)
        close(float(row["delivered_overlap"]), delivered, 5e-13,
              f"ablation {row['variant']} delivered overlap", errors)
        close(recovery, expected[row["variant"]], 5e-9,
              f"ablation {row['variant']} recovery", errors)
        close(float(row["recovery"]), recovery, 5e-10,
              f"ablation {row['variant']} stored recovery", errors)
    print("ablation         recovery=49.00/77.73/89.22/99.30%")


def check_baselines(root: Path, errors: list[str]) -> None:
    data = json.loads(
        (root / "results" / "v7_baseline_levels_corrected_haar.json").read_text()
    )
    equal(data["k"], 8, "baseline k", errors)
    equal(data["n"], 160, "baseline observations", errors)
    equal(len(data["modules"]), 8, "baseline modules", errors)
    expected = {
        "orthogonality": 0.027510759243159554,
        "isotropic": 0.01709409257649289,
        "regmean": 0.052185766534724586,
        "tikhonov": 0.05540781267530595,
        "generative": -0.8959326509298989,
        "permutation": -2.843444235622883e-09,
        "ours": -0.000676566400215961,
    }
    equal(set(data["level"]), set(expected), "baseline methods", errors)
    for key, want in expected.items():
        close(float(data["level"][key]), want, 5e-13,
              f"baseline {key}", errors)
    metadata = data["metadata"]
    close(float(metadata["raw_overlap_over_isotropic_chance"]),
          float(data["level"]["orthogonality"]) / float(data["iso"]), 5e-13,
          "baseline raw/chance ratio", errors)
    equal(metadata["draws_per_module"], 20, "baseline draws/module", errors)
    equal(metadata["skipped_degenerate_draws"], 0,
          "baseline skipped draws", errors)
    print("baselines         raw/chance=2.6410x; corrected-null level=-.00068")


def check_projector(root: Path, errors: list[str]) -> None:
    expected = {
        "v7_projector_b16_fullft.csv":
            (12, 0.259407215234306, 0.166666666666667,
             0.261191739390294, 0.0833333333333333),
        "v7_projector_b16_lora16.csv":
            (24, 0.874952667703231, 0.0208333333333333,
             0.135585801986357, 0.0833333333333333),
    }
    for filename, (n, survive, chance, in_activation, activation_chance) in expected.items():
        rows = read_csv(root / "results" / filename)
        equal(len(rows), n, f"{filename} rows", errors)
        for key, want in (("projector_survives_null", survive), ("chance", chance),
                          ("projector_in_top_activation", in_activation),
                          ("activation_chance", activation_chance)):
            close(mean(rows, key), want, 5e-12, f"{filename} {key}", errors)
    print("projector        FT observed/null=.2594/.1667; LoRA=.8750/.0208")


def check_h0(root: Path, arm_values: dict[str, dict[str, float]],
             errors: list[str]) -> None:
    for label, expected in H0_ARMS.items():
        rows = read_csv(root / "results" / str(expected["file"]))
        equal(len(rows), 672, f"{label} rows", errors)
        raw = mean(rows, "raw")
        fitted = mean(rows, "null_act_raw")
        h0 = mean(rows, "h0_excess")
        close(raw, float(expected["raw"]), 5e-12, f"{label} raw", errors)
        close(fitted, float(expected["null"]), 5e-12, f"{label} null", errors)
        close(h0, float(expected["h0"]), 5e-12, f"{label} H0", errors)
        # The compatibility alias must not add the external diagnostic.
        close(mean(rows, "excess_raw_corrected"), mean(rows, "excess_raw"), 1e-15,
              f"{label} H0 kept separate", errors)
        print(f"H0 {label:27s} null={fitted:.5f} H0={h0:+.5f}")
    mc_difference = (float(H0_ARMS["FT8 activation MC"]["null"])
                     - arm_values["FT8"]["fitted"])
    close(mc_difference, -3.00940630876e-05, 5e-13,
          "FT8 MC versus exact null difference", errors)


def _layer_index(module: str) -> int:
    parts = module.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot extract encoder layer from {module!r}") from exc


def _exact_summary(rows: list[dict[str, str]]) -> tuple[float, float]:
    raw = mean(rows, "raw")
    iso = mean(rows, "null_iso")
    fitted = mean(rows, "null_exact")
    denominator = raw - iso
    if abs(denominator) <= 1e-12:
        raise ValueError("leave-one-out subset has a degenerate denominator")
    return 100.0 * (fitted - iso) / denominator, raw - fitted


def _influence_dimension(rows: list[dict[str, str]], dimension: str) -> dict[str, object]:
    if dimension == "task":
        units: list[object] = sorted({row[key] for row in rows for key in ("a", "b")})
        keep = lambda row, unit: row["a"] != unit and row["b"] != unit
    elif dimension == "layer":
        units = sorted({_layer_index(row["module"]) for row in rows})
        keep = lambda row, unit: _layer_index(row["module"]) != unit
    else:
        raise ValueError(f"unknown influence dimension {dimension!r}")

    values = []
    for unit in units:
        retained = [row for row in rows if keep(row, unit)]
        fraction, residual = _exact_summary(retained)
        values.append({
            "omitted": unit,
            "retained_rows": len(retained),
            "fraction_percent": fraction,
            "residual": residual,
        })

    minimum_fraction = min(values, key=lambda row: row["fraction_percent"])
    maximum_fraction = max(values, key=lambda row: row["fraction_percent"])
    minimum_residual = min(values, key=lambda row: row["residual"])
    maximum_residual = max(values, key=lambda row: row["residual"])
    return {
        "n_units": len(units),
        "fraction_percent": {
            "minimum": minimum_fraction["fraction_percent"],
            "minimum_omitted": minimum_fraction["omitted"],
            "maximum": maximum_fraction["fraction_percent"],
            "maximum_omitted": maximum_fraction["omitted"],
        },
        "residual": {
            "minimum": minimum_residual["residual"],
            "minimum_omitted": minimum_residual["omitted"],
            "maximum": maximum_residual["residual"],
            "maximum_omitted": maximum_residual["omitted"],
        },
        "values": values,
    }


def make_influence_report(root: Path) -> dict[str, object]:
    arms = {}
    for label in EXACT_LABELS:
        filename = str(ARMS[label]["file"])
        rows = read_csv(root / "results" / filename)
        arms[label] = {
            "source": f"results/{filename}",
            "leave_one_task_out": _influence_dimension(rows, "task"),
            "leave_one_layer_out": _influence_dimension(rows, "layer"),
        }
    return {
        "schema_version": 1,
        "estimand": {
            "fraction_percent": "100 * (mean(null_exact) - mean(null_iso)) / (mean(raw) - mean(null_iso))",
            "residual": "mean(raw) - mean(null_exact)",
            "task_omission": "remove every row whose unordered endpoint pair contains the task",
            "layer_omission": "remove every module row from the encoder-layer index",
        },
        "arms": arms,
    }


def check_influence(root: Path, errors: list[str]) -> None:
    path = root / "influence_exact_leave_one_out.json"
    stored = json.loads(path.read_text())
    recomputed = make_influence_report(root)
    equal(stored, recomputed, "exact-arm leave-one-out report", errors)
    for label in EXACT_LABELS:
        arm = recomputed["arms"][label]
        task = arm["leave_one_task_out"]
        layer = arm["leave_one_layer_out"]
        if task["residual"]["minimum"] <= 0 or layer["residual"]["minimum"] <= 0:
            errors.append(f"{label} leave-one-out residual is not uniformly positive")
        print(
            f"influence {label:4s} LOTO=[{task['fraction_percent']['minimum']:.2f},"
            f"{task['fraction_percent']['maximum']:.2f}]% "
            f"LOLO=[{layer['fraction_percent']['minimum']:.2f},"
            f"{layer['fraction_percent']['maximum']:.2f}]%"
        )


def _is_revision(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 40
            and all(character in "0123456789abcdef" for character in value))


def check_reproducibility_pins(root: Path, errors: list[str]) -> None:
    settings = json.loads((root / "run_settings.json").read_text())
    pins = json.loads((root / "upstream_revisions.json").read_text())
    equal(pins["schema_version"], 1, "upstream revision schema", errors)
    equal(pins["resolution_evidence"]["local_paths_included"], False,
          "upstream revision path policy", errors)

    bases = pins["base_models"]
    expected_bases = {
        "openai/clip-vit-base-patch16": "5ef227a78de3f75873f373246dac80def63b0003",
        "openai/clip-vit-base-patch32": "c237dc49a33fc61debc9276459120b7eac67e7ef",
        "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    }
    equal(set(bases), set(expected_bases), "base repository pins", errors)
    for repository, revision in expected_bases.items():
        values = [value for key, value in bases[repository].items()
                  if key.endswith("revision")]
        if revision not in values:
            errors.append(f"{repository} task-vector revision is not pinned")
        for value in values:
            if not _is_revision(value):
                errors.append(f"{repository} has malformed revision {value!r}")
    frozen = {entry.rsplit("@", 1)[0]: entry.rsplit("@", 1)[1]
              for entry in settings["frozen_bases"]}
    equal(frozen, expected_bases, "run-settings base revisions", errors)

    for architecture in ("openai/clip-vit-base-patch16",
                         "openai/clip-vit-base-patch32"):
        audit = bases[architecture]["split_snapshot_audit"]
        equal(audit["compared_tensors"], 400,
              f"{architecture} split-snapshot tensor count", errors)
        equal(audit["key_sets_equal"], True,
              f"{architecture} split-snapshot keys", errors)
        equal(audit["all_tensors_exactly_equal"], True,
              f"{architecture} split-snapshot values", errors)
        close(float(audit["maximum_absolute_difference"]), 0.0, 0.0,
              f"{architecture} split-snapshot max difference", errors)

    tasks_20 = set(settings["tasks_20"])
    tasks_8 = set(settings["tasks_8"])
    specialists = pins["full_finetuning_specialists"]
    equal(set(specialists), set(SENSITIVITY), "specialist architectures", errors)
    specialist_revisions = []
    for architecture, record in specialists.items():
        revisions = record["revision_by_task"]
        equal(set(revisions), tasks_20, f"{architecture} specialist tasks", errors)
        specialist_revisions.extend(revisions.values())
        if "{task}" not in record["repository_template"]:
            errors.append(f"{architecture} specialist repository template lacks task slot")

    adapters = pins["adapters"]
    equal(set(adapters), {"rank16_lora", "rank16_linearized_lora"},
          "adapter families", errors)
    adapter_revisions = []
    for family, record in adapters.items():
        revisions = record["revision_by_task"]
        equal(set(revisions), tasks_8, f"{family} adapter tasks", errors)
        adapter_revisions.extend(revisions.values())
        if "{task}" not in record["repository_template"]:
            errors.append(f"{family} adapter repository template lacks task slot")

    calibration = pins["calibration_datasets"]
    equal(set(calibration["revision_by_source"]),
          {"cifar10", "dtd", "eurosat", "gtsrb", "resisc45", "stl10", "sun397"},
          "calibration dataset sources", errors)
    calibration_revisions = list(calibration["revision_by_source"].values())
    composition = pins["calibration_composition_datasets"]
    expected_composition = {
        "mnist": ("ylecun/mnist", "mnist", "test",
                  "77f3279092a1c1579b2250db8eafed0ad422088c"),
        "stanford_cars": ("tanganke/stanford_cars", "default", "test",
                          "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"),
        "svhn": ("ufldl-stanford/svhn", "cropped_digits", "test",
                 "f9e1717d73324ebbc1ece84267ccd80d0aca0690"),
    }
    equal(set(composition), set(expected_composition),
          "composition-control dataset sources", errors)
    composition_revisions = []
    for source, (repository, config, split, revision) in expected_composition.items():
        equal(composition[source], {
            "repository": repository, "config": config,
            "split": split, "revision": revision,
        }, f"composition-control dataset pin {source}", errors)
        composition_revisions.append(composition[source]["revision"])
    for label, revisions, expected_count in (
            ("specialist", specialist_revisions, 60),
            ("adapter", adapter_revisions, 16),
            ("primary calibration dataset", calibration_revisions, 7),
            ("composition calibration dataset", composition_revisions, 3)):
        equal(len(revisions), expected_count, f"{label} revision count", errors)
        if not all(_is_revision(revision) for revision in revisions):
            errors.append(f"{label} revisions include a malformed commit")
    equal(pins["counts"], {
        "base_repositories": 3,
        "full_finetuning_specialist_repositories": 60,
        "adapter_repositories": 16,
        "calibration_dataset_repositories": 10,
        "primary_calibration_dataset_repositories": 7,
        "composition_control_dataset_repositories": 3,
    }, "upstream revision declared counts", errors)
    equal(len(pins["remaining_unknowns"]), 1,
          "upstream provenance caveat count", errors)

    pyproject = (root / "environment" / "pyproject.toml").read_text()
    lock = (root / "environment" / "uv.lock").read_text()
    freeze_lines = [line.strip() for line in
                    (root / "environment" / "runtime-freeze.txt").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    freeze = {}
    for line in freeze_lines:
        if line.count("==") != 1:
            errors.append(f"runtime freeze entry is not exact: {line!r}")
            continue
        package, version = line.split("==", 1)
        freeze[package] = version
    equal(len(freeze), 35, "runtime freeze distribution count", errors)
    critical = {
        "huggingface-hub": "1.31.0", "numpy": "2.4.6", "pillow": "12.3.0",
        "safetensors": "0.8.0", "torch": "2.14.0", "transformers": "5.17.0",
    }
    for package, version in critical.items():
        equal(freeze.get(package), version, f"runtime pin {package}", errors)
        if f'name = "{package}"' not in lock or f'version = "{version}"' not in lock:
            errors.append(f"uv.lock does not contain runtime pin {package}=={version}")
    for dependency in ("torch>=2.0", "transformers>=4.40", "safetensors",
                       "huggingface_hub", "numpy", "pillow"):
        if f'"{dependency}"' not in pyproject:
            errors.append(f"pyproject is missing dependency {dependency!r}")
    for marker in ('version = 1', 'revision = 3', 'requires-python = \">=3.10\"'):
        if marker not in lock:
            errors.append(f"uv.lock is missing header {marker!r}")
    materialization = json.loads(
        (root / "environment" / "composition-materialization-runtime.json").read_text()
    )
    equal(materialization["schema_version"], 1,
          "composition materialization runtime schema", errors)
    observation = materialization["execution_observation"]
    equal(observation["imported_versions"],
          {"datasets": "5.0.1", "pillow": "12.3.0"},
          "composition materialization observed imports", errors)
    equal(observation["full_installed_distribution_freeze_captured"], False,
          "composition materialization freeze caveat", errors)
    equal(materialization["numerical_runtime_record"], {
        "path": "runtime-freeze.txt", "python": "3.11.16",
        "distributions": 35, "includes_optional_data_extra": False,
    }, "separate numerical runtime record", errors)
    locked_data = materialization["locked_reconstruction_environment"]["lock_contains"]
    expected_locked_data = {
        "datasets": "5.0.1", "dill": "0.4.1", "multiprocess": "0.70.19",
        "pyarrow": "25.0.1", "requests": "2.34.2", "xxhash": "4.0.1",
        "pillow": "12.3.0",
    }
    equal(locked_data, expected_locked_data,
          "composition materialization locked data environment", errors)
    for package, version in expected_locked_data.items():
        if f'name = "{package}"\nversion = "{version}"' not in lock:
            errors.append(f"uv.lock lacks data-stage pin {package}=={version}")
    print("reproducibility  3 bases, 60 specialists, 16 adapters, 10 datasets; "
          "35 numeric-runtime pins + locked data extra")


def check_source_bundle(root: Path, errors: list[str]) -> None:
    code = root / "code"
    provenance = json.loads((code / "SOURCE_PROVENANCE.json").read_text())
    equal(provenance["schema_version"], 1, "source provenance schema", errors)
    snapshot = provenance["snapshot"]
    equal(snapshot["local_paths_included"], False,
          "source snapshot path policy", errors)
    equal(snapshot["producer_repository_identifier_included"], False,
          "source snapshot double-blind repository policy", errors)
    if "RESULTS.sha256" not in snapshot["authoritative_identity"]:
        errors.append("source snapshot does not name its authoritative checksum record")
    if "no repository commit is claimed" not in snapshot["working_tree_status"]:
        errors.append("source snapshot working-tree caveat is missing")

    included = provenance["included"]
    declared = set(itertools.chain.from_iterable(included.values()))
    actual = {
        str(path.relative_to(code)) for path in code.rglob("*")
        if path.is_file() and path.suffix == ".py"
    }
    equal(declared, actual, "source provenance Python-file coverage", errors)
    if any("__pycache__" in path.parts or path.suffix == ".pyc"
           for path in code.rglob("*")):
        errors.append("source bundle contains bytecode or __pycache__")

    archived = provenance["archived_execution_sha256"]
    calibration = json.loads(
        (root / "results" / "calibration_resample_b16" / "summary.json").read_text()
    )["provenance"]["implementations_sha256"]
    calibration_to_source = {
        "actnull/covariance.py": "actnull/covariance.py",
        "actnull/null.py": "actnull/null.py",
        "calibration_resample.py": "scripts/calibration_resample.py",
        "run_resample_exact.py": "scripts/run_resample_exact.py",
        "scripts/bootstrap_overlap.py": "scripts/bootstrap_overlap.py",
        "scripts/exact_overlap.py": "scripts/exact_overlap.py",
    }
    equal(set(calibration), set(calibration_to_source),
          "calibration implementation provenance keys", errors)
    for calibration_name, source_name in calibration_to_source.items():
        equal(archived[source_name], calibration[calibration_name],
              f"archived execution hash {source_name}", errors)

    adjustments = {entry["path"]: entry
                   for entry in provenance["release_adjustments"]}
    equal(set(adjustments), {
        "scripts/ablation.py",
        "actnull/covariance.py", "scripts/run_resample_exact.py",
        "scripts/fetch_calibration_images.py", "scripts/pool_lora_mc_batches.py",
        "actnull/null.py", "scripts/materialize_composition_control.py",
        "scripts/lora_topk_gap_diagnostic.py", "scripts/merge_eval.py",
        "scripts/lora_factor_audit.py", "scripts/projector_rebuild.py",
        "scripts/whitening_calibration.py",
    }, "source release-adjustment paths", errors)
    for path_text, record in adjustments.items():
        equal(sha256(code / path_text), record["shipped_sha256"],
              f"shipped source hash {path_text}", errors)
        if "archived_execution_sha256" in record:
            equal(record["archived_execution_sha256"], archived[path_text],
                  f"adjusted source archived hash {path_text}", errors)
    for path_text in ("scripts/bootstrap_overlap.py",
                      "scripts/calibration_resample.py",
                      "scripts/calibration_source_subgroup_diagnostic.py",
                      "scripts/exact_overlap.py"):
        equal(sha256(code / path_text), archived[path_text],
              f"unchanged executed source hash {path_text}", errors)

    equal((code / "pyproject.toml").read_text(),
          (root / "environment" / "pyproject.toml").read_text(),
          "source/environment pyproject identity", errors)
    fetch_source = (code / "scripts" / "fetch_calibration_images.py").read_text()
    for marker in ("--revision-record", "revision=revisions[name]",
                   "expected {args.per_source}"):
        if marker not in fetch_source:
            errors.append(f"calibration fetch helper lacks {marker!r}")
    wrapper_source = (code / "scripts" / "run_resample_exact.py").read_text()
    for marker in ("--revision-record", "--hf-home", "--seed-root",
                   '"scripts/exact_overlap.py"'):
        if marker not in wrapper_source:
            errors.append(f"calibration refit wrapper lacks {marker!r}")
    pool_source = (code / "scripts" / "pool_lora_mc_batches.py").read_text()
    for marker in ("static value mismatch", "null_act_raw", "null_act_cond",
                   "excess_raw_corrected"):
        if marker not in pool_source:
            errors.append(f"LoRA pooling helper lacks {marker!r}")
    ablation_source = (code / "scripts" / "ablation.py").read_text()
    for marker in ("initial (neither correction)", "singleton signs only",
                   "full complement only", "corrected (both)",
                   "study_level_centered", "delivered_overlap", "recovery"):
        if marker not in ablation_source:
            errors.append(f"ablation producer lacks {marker!r}")
    gap_source = (code / "scripts" / "lora_topk_gap_diagnostic.py").read_text()
    for marker in ('args.output.open("x")', "--pin-record", "--hf-hub"):
        if marker not in gap_source:
            errors.append(f"LoRA top-k gap producer lacks {marker!r}")
    merge_source = (code / "scripts" / "merge_eval.py").read_text()
    for marker in ("--revision-record", "snapshot_download", "revision=revision",
                   'args.out.open("x"', "fixed ranks"):
        if marker not in merge_source:
            errors.append(f"merge evaluator lacks {marker!r}")
    subgroup_source = (code / "scripts" /
                       "calibration_source_subgroup_diagnostic.py").read_text()
    for marker in ('args.output.open("x"', "CALIBRATION_SOURCES",
                   "reproduced_fraction_percent"):
        if marker not in subgroup_source:
            errors.append(f"calibration-source subgroup producer lacks {marker!r}")
    print(f"producer source  {len(actual)} Python files, execution hashes and release deltas checked")


def check_checksums(root: Path, errors: list[str]) -> None:
    manifest = root / "RESULTS.sha256"
    project = root.parent
    entries = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            digest, relative = line.split(None, 1)
        except ValueError:
            errors.append(f"RESULTS.sha256 line {line_number} is malformed")
            continue
        relative = relative.lstrip("*")
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] != "artifact" or ".." in parts:
            errors.append(f"RESULTS.sha256 line {line_number} escapes artifact: {relative}")
            continue
        path = project / relative
        if not path.is_file():
            errors.append(f"RESULTS.sha256 missing file: {relative}")
            continue
        equal(sha256(path), digest, f"checksum {relative}", errors)
        entries.append(relative)
    expected = {
        str(path.relative_to(project)) for path in root.rglob("*")
        if path.is_file() and path != root / "RESULTS.sha256"
    }
    equal(set(entries), expected, "checksum manifest coverage", errors)
    print(f"checksums        {len(entries)} files verified")


def check_anonymity(root: Path, errors: list[str]) -> None:
    # Construct generic local prefixes so the audit source itself contains no
    # machine- or author-specific identifier that a release scanner could flag.
    banned = ("/" + "Users" + "/", "/" + "private" + "/" + "tmp" + "/",
              "/" + "home" + "/", "/" + "131_data" + "/",
              "C:" + "\\\\" + "Users" + "\\\\")
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt") as handle:
                    text = handle.read()
            else:
                text = path.read_text()
        except UnicodeDecodeError:
            continue
        for token in banned:
            if token.lower() in text.lower():
                hits.append(f"{path.relative_to(root)} contains {token!r}")
    if hits:
        errors.extend(hits)
    print("anonymity        no local absolute paths or machine identifiers found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT,
                        help="artifact directory (default: repository artifact/)")
    parser.add_argument("--write-influence", action="store_true",
                        help="regenerate influence_exact_leave_one_out.json and exit")
    args = parser.parse_args()
    root = args.artifact.resolve()
    if args.write_influence:
        destination = root / "influence_exact_leave_one_out.json"
        destination.write_text(json.dumps(make_influence_report(root), indent=2,
                                          sort_keys=True) + "\n")
        print(f"wrote {destination}")
        return 0
    errors: list[str] = []
    arm_values: dict[str, dict[str, float]] = {}
    try:
        for label, expected in ARMS.items():
            arm_values[label] = check_arm(root, label, expected, errors)
        check_pair_rank_preservation(root, errors)
        check_bootstraps(root, arm_values, errors)
        check_mc64(root, errors)
        check_lora_factor_audit(root, errors)
        check_lora_topk_gaps(root, errors)
        check_merge_accuracy(root, errors)
        check_sensitivity(root, arm_values, errors)
        check_calibration_source_subgroups(root, errors)
        check_calibration_resampling(root, arm_values, errors)
        check_calibration_composition(root, arm_values, errors)
        check_power(root, errors)
        check_ablation(root, errors)
        check_baselines(root, errors)
        check_projector(root, errors)
        check_h0(root, arm_values, errors)
        check_influence(root, errors)
        check_reproducibility_pins(root, errors)
        check_source_bundle(root, errors)
        check_checksums(root, errors)
        check_anonymity(root, errors)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"checker could not read bundle: {exc}")

    if errors:
        print("\nFAILED")
        print("\n".join(f"- {error}" for error in errors[:80]))
        if len(errors) > 80:
            print(f"- ... {len(errors) - 80} additional errors")
        return 1
    print("\nAll checked v7 results passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
