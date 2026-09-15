# PRISM v7 compact result artifact

This compact bundle contains the row-level data needed to recompute every headline
overlap estimate and crossed bootstrap interval, plus compact outputs for the
sensitivity, positive-control, ablation, baseline, projector, and external-H0
diagnostics, the downstream B/32 merge-accuracy comparison, the all-24-module LoRA
factor audit, and the LoRA top-k boundary-gap audit. The LoRA rows additionally include all four independent 16-draw
batches, their pooled 64-draw primary estimate, and a Monte Carlo convergence
record. The bundle also includes two source-stratified B/16 calibration refits,
the sanitized producer source and tests, immutable upstream revisions, and the
locked v7 software environment. It deliberately omits model weights, activation
covariance tensors, fold tensors, and sensitivity cell CSVs; the large
calibration intermediates are fingerprinted in `calibration_link.json`.

The bundle contains the public upstream repository identifiers required to
resolve the inputs, but no local paths, usernames, or machine-specific metadata.

## One-command audit

From the project root, run:

```bash
python3 analysis/check_paper_numbers.py
```

The checker uses only the Python standard library. It independently recomputes
the six row-level ratio-of-means estimates, validates all 50,000-draw crossed
intervals and their source hashes, checks MC64 pooling and Monte Carlo error,
replays the calibration-resampling and composition-control maps and refit
estimates, checks the complete 36-setting sensitivity grids, recomputes
exact-arm leave-one-task/layer-out
influence, power and ablation normalizations, recomputes the merge table from
per-task accuracies, checks LoRA boundary gaps, validates source/upstream/environment
pins, checks the all-24-module LoRA factor audit against the released product rows,
recomputes the calibration-source endpoint subgroups, and verifies every
entry in `RESULTS.sha256`. It also recomputes the B/16 and B/32 pair-level
Spearman correlations and shared top-20 counts from the exact rows.

To additionally verify that the live TeX input graph contains the corresponding
values in the correct semantic contexts, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 analysis/check_manuscript.py \
  --paper iclr2027_conference.tex --artifact artifact
```

This first runs the numerical audit above and then checks a context-anchored claim
registry. It also prints result-like TeX literals outside the current registry as
coverage warnings; it deliberately does not claim that an identical number found
elsewhere validates a manuscript claim.

To check file integrity alone:

```bash
shasum -a 256 -c artifact/RESULTS.sha256
```

## Bundle contents

- `results/v7_exact_*.csv.gz`: analytic scalar-Haar activation-null rows for
  ViT-B/16, ViT-B/32, ViT-L/14, and the matched eight-task full-FT arm.
- `results/v7_mc64_*loranull.csv`: primary standard and linearized LoRA rows,
  formed by pooling four independent 16-draw null batches row by row.
- `results/v7_mc64_*seed*.csv.gz` and `v7_legacy16_*.csv.gz`: all component
  batches. The explicitly named legacy files are the seed-0 batches used by the
  earlier 16-draw audit and are not primary results.
- `results/v7_mc64_lora_factor_null_convergence.json`: batch estimates,
  pooled-minus-legacy deltas, and across-batch Monte Carlo standard errors.
- `results/v7_lora_factor_audit.json`: checkpoint-derived $A$-right, $B$-left,
  and $BA$-right overlap summaries over all 24 q/v modules (the primary scope),
  with the historical layers-2/6/10 six-module result retained only as an
  explicitly labelled secondary provenance diagnostic. It includes every pinned
  adapter hash and a 672-row cross-check against each released product CSV.
- `results/v7_lora_topk_gap_diagnostic.json`: observed-A, observed-BA, and all
  24,576 factor-null top-k boundary-gap summaries for both LoRA arms.
- `results/v8_merge_b32_accuracy.json`: the reported merge table. All eight per-task
  accuracies, binomial standard errors, and counts for zero-shot, uncompressed, and
  both rank-constrained bases at six ranks, on 2,000 test images per task. Zero-shot
  averages 48.3% and uncompressed task arithmetic 70.3%, both within a point of the
  published values for this benchmark.
- `results/v7_merge_b32_accuracy.json`: the superseded first run of that table. Its
  zero-shot heads used a three-template generic ensemble instead of the benchmark's
  per-dataset ensembles, which put both baselines about ten points low. Retained
  because the comparison between the two is what establishes that the prompt protocol,
  not the bases being compared, produced the gap.
- `results/calibration_resample_b16/`: two complete source-stratified calibration
  refit row files, all 1,001 source/draw/fold assignments per refit, and a compact
  method/result summary. The observed headline range is 25.102--25.302%, at most
  0.243 percentage points from the original B/16 estimate.
- `results/calibration_composition_b16/`: a deliberately different
  MNIST/Stanford-Cars/SVHN calibration-composition control (334/334/333 images),
  with all sampled indices and image hashes but no redistributed images. It yields
  21.482% and a conditional crossed interval of [16.898, 26.269]%.
- `results/v7_calibration_source_subgroup_diagnostic.json`: B/16, B/32, and
  L/14 results stratified by whether 0, 1, or 2 task-pair endpoints also appear
  among the seven activation-calibration source tasks.
- `bootstrap_*_joint_50000.json`: crossed task-endpoint-by-layer percentile
  intervals (seed 20260915). `bootstrap_lora_joint_50000.json` uses the primary
  pooled-64 rows; `bootstrap_lora_legacy16_joint_50000.json` is retained only as
  an explicitly labeled seed-0 audit.
- `results/*sensitivity*_summary.csv`: all 36 combinations of
  `k={4,8,16,32}`, `m={16,32,64}`, and `z={1,2,3}` for each architecture.
- `results/v7_power_*.json` and `v7_ablation_level_centered.json`: corrected
  activation and LoRA positive controls, initial implementations, and the
  singleton-sign/full-complement ablation.
- `results/v7_baseline_levels_corrected_haar.json`: common-reference null levels
  from the final sign-correct Haar implementation.
- `results/v7_projector_*.csv`: observed-null basis-overlap diagnostics.
- `results/*actnull_h0.csv.gz`: matched activation-null Monte Carlo rows with
  external H0 recorded as a separate diagnostic, never an additive correction.
- `influence_exact_leave_one_out.json`: exact-arm leave-one-task-out and
  leave-one-layer-out values and ranges, recomputed from the row files.
- `upstream_revisions.json`: immutable revisions for all 3 bases, 60 full-FT
  specialists, 16 adapters, 7 primary-calibration datasets, and 3
  composition-control datasets, including content hashes and the split-base
  equivalence audit.
- `environment/`: the producer dependency specification, exact resolver lock,
  35-package numerical-runtime freeze, and the separately scoped composition
  materialization version record.
- `code/`: the sanitized `actnull` producer package, portable reconstruction
  helpers, tests, and source-provenance record. See `code/README.md`.
- `run_settings.json`: estimands, task/module scopes, seeds, and draw counts.

## Recompute crossed intervals

`analysis/overlap_bootstrap.py` accepts both `.csv` and `.csv.gz`. For example:

```bash
python3 analysis/overlap_bootstrap.py \
  B16=artifact/results/v7_exact_clip-vit-base-patch16.csv.gz \
  B32=artifact/results/v7_exact_clip-vit-base-patch32.csv.gz \
  FT8=artifact/results/v7_exact_b16_8task_q_v_fullft.csv.gz \
  L14=artifact/results/v7_exact_clip-vit-large-patch14.csv.gz \
  --draws 50000 --seed 20260915 --output /tmp/bootstrap_exact.json
```

The bootstrap conditions on the realized rows, calibration, one checkpoint per
task, and selected per-row null. It resamples task endpoints and encoder layers
independently in the same replicate, weights row `(a,b,l)` by `C_a C_b L_l`,
averages weighted rows, and then forms the ratio. These brackets are finite-pool
resampling/stability intervals: they do not claim superpopulation coverage or
simultaneous family-wise coverage across metrics.

## Recompute leave-one-out influence

The same checker regenerates the path-free influence record directly from the
four exact-arm row files:

```bash
python3 analysis/check_paper_numbers.py --write-influence
```

Each task omission removes every pair containing that endpoint. Each layer
omission removes all retained module rows at that encoder-layer index. The
headline ratio and residual are then recomputed on the remaining equal-weight
cells; this is a deterministic influence diagnostic, not a confidence interval.

## Environment and upstream inputs

`upstream_revisions.json` resolves every upstream repository used in v7 to a
40-character commit. ViT-B/16 and ViT-B/32 used weights available in two snapshot
layouts; the retained legacy binary and safetensors versions have identical keys
and bitwise-identical values across all 400 tensors. The historical B16
covariance manifest did not preserve which of those two equivalent snapshot
labels it used, so that label remains explicitly marked unknown while its
numerical weight content is pinned by SHA-256.

With `uv` installed, validate the dependency lock with:

```bash
uv lock --check --project artifact/environment
```

To materialize the locked environment and run the producer tests, follow
`code/README.md`. No third-party dependency is needed for the numerical checker.

## Scope and release note

This directory is a compact numerical audit and producer-source bundle, not a
stand-alone training release. It includes immutable checkpoint/dataset revisions,
the exact v7 dependency records, and the producer package, but not the upstream
weights themselves or the large derived activation covariance and fold tensors.
Reconstructing those large intermediates therefore requires downloading the
pinned public inputs and running the shipped producer code. The six primary
overlap estimates, downstream merge-table aggregation, and the sensitivity,
resampling, positive-control, ablation, baseline-level, projector, H0, factor,
gap, and influence diagnostics listed above can be audited fully offline without
them, except that recomputing factor singular spaces requires the pinned adapters.
Re-running the merge model forward passes still requires the pinned public
weights and datasets; the compact result records per-task accuracies rather than
individual-example logits.

This compact audit does not validate cross-term energy-capture values or the
separate whitening/generative evaluation diagnostics. Those remain outside the
checker's stated coverage.
