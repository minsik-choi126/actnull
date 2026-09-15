# PRISM v7 producer source

This directory is the sanitized source snapshot for the numerical artifact.  It
contains the `actnull` package, the v7 producer scripts, and their tests.  Exact
source identity is given by the `artifact/code/` entries in
`artifact/RESULTS.sha256`; `SOURCE_PROVENANCE.json` distinguishes archived
execution hashes from the few release-only portability changes.

The snapshot included working-tree changes beyond the producer repository's last
commit, so no commit alone is claimed to identify v7.  The repository identifier
is omitted during double-blind review because the complete, checksummed source
needed here is included directly.

## Install and test

From the project root, create the exact locked environment and expose this source
tree without installing a second copy:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/prism-v7-env \
  uv sync --project artifact/environment --all-extras --no-install-project --locked
export PYTHONPATH="$PWD/artifact/code"
PYTHONDONTWRITEBYTECODE=1 /tmp/prism-v7-env/bin/python -m unittest discover \
  -s artifact/code/tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /tmp/prism-v7-env/bin/python \
  artifact/code/tests/test_haar_null.py
```

The release-level audits remain dependency-free.  `check_paper_numbers.py` audits
the v7 results on their own; `check_manuscript.py` additionally ties registered
manuscript claims to those results and verifies the checksum manifest, so it is
the one to run before a release:

```bash
python3 analysis/check_paper_numbers.py
PYTHONDONTWRITEBYTECODE=1 python3 analysis/check_manuscript.py \
  --paper iclr2027_conference.tex --artifact artifact
```

Run both from the project root, never from inside `artifact/code/`.  A build or
test invoked in-tree writes `.venv`, `uv.lock`, and `__pycache__` next to the
sources, which `check_manuscript.py` reports as a manifest-coverage failure.

## Reproduce the MC64 pooled LoRA rows

All four independent 16-draw batches are shipped.  Seed 0 has the explicit
`legacy16` name; it is one component of the pooled result, not a second result.
This command recreates the standard-LoRA primary CSV byte for byte:

```bash
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/pool_lora_mc_batches.py \
  --inputs \
    artifact/results/v7_legacy16_b16_8task_q_v_lora-16_loranull.csv.gz \
    artifact/results/v7_mc64_b16_8task_q_v_lora-16_loranull_seed1000003.csv.gz \
    artifact/results/v7_mc64_b16_8task_q_v_lora-16_loranull_seed2000003.csv.gz \
    artifact/results/v7_mc64_b16_8task_q_v_lora-16_loranull_seed3000017.csv.gz \
  --out /tmp/v7_mc64_lora.csv
cmp /tmp/v7_mc64_lora.csv \
  artifact/results/v7_mc64_b16_8task_q_v_lora-16_loranull.csv
```

Use the corresponding `l-lora` files to recreate the linearized-LoRA arm.  The
same rows can be passed to `scripts/bootstrap_overlap.py` with `--draws 50000
--seed 20260915` to regenerate the crossed finite-pool resampling intervals.

To generate the four component batches from pinned adapters, run
`python -m actnull.null` four times with seeds `0`, `1000003`, `2000003`, and
`3000017`, together with:

```text
--experts_are_adapters --only_modules q_proj,v_proj --key_prefix vision_model.
--k 8 --null lora --null_draws 16 --h0_draws 0 --device cpu
```

The base and all eight adapter snapshot revisions for each arm are in
`artifact/upstream_revisions.json`.

## Reproduce the LoRA factor audit

With the 16 pinned adapter snapshots materialized in a Hugging Face hub cache,
the following command recomputes the primary all-24-module factor audit and
cross-checks every $BA$ cell against the released product rows:

```bash
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/lora_factor_audit.py \
  --pin-record artifact/upstream_revisions.json \
  --hf-hub <hugging-face-hub-directory> \
  --lora-product-csv \
    artifact/results/v7_mc64_b16_8task_q_v_lora-16_loranull.csv \
  --linearized-product-csv \
    artifact/results/v7_mc64_b16_8task_q_v_l-lora-16_loranull.csv \
  --output /tmp/v7_lora_factor_audit.json
cmp /tmp/v7_lora_factor_audit.json \
  artifact/results/v7_lora_factor_audit.json
```

The primary scope is all 12 encoder layers times q/v (24 modules, 672
task-pair/module cells per arm). The historical layers-2/6/10 subset is emitted
only under `secondary_legacy_six_modules`. The result records checkpoint content
hashes and public repository revisions, but no cache paths.

## Rebuild exact activation-null rows

After downloading the pinned base and specialist snapshots and rebuilding the
activation covariance plus its eight folds, invoke:

```bash
/tmp/prism-v7-env/bin/python artifact/code/scripts/exact_overlap.py \
  --base <pinned-base-snapshot> \
  --experts <task-1>=<pinned-specialist-1> <task-2>=<pinned-specialist-2> \
  --cov <covariance-directory> --fold_cov <fold-covariance-root> \
  --k 8 --null_m 64 --block_z 2 --key_prefix vision_model. \
  --device cpu --out <new-row-csv>
```

Pass all specialists named in `upstream_revisions.json`; the abbreviated two-task
argument above only shows the `NAME=path` syntax.  For the LoRA-matched full-FT
arm add `--only_modules q_proj,v_proj` and use the eight-task subset recorded in
`run_settings.json`.  `scripts/exact_sensitivity.py` evaluates the full
`k={4,8,16,32}`, `m={16,32,64}`, `z={1,2,3}` grid without reloading each task
vector for every setting.

## Rebuild the calibration-resampling refits

First materialize the seven immutable dataset revisions and create one
source-stratified draw:

```bash
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/fetch_calibration_images.py \
  --revision-record artifact/upstream_revisions.json \
  --out <source-pool> --per_source 143
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/calibration_resample.py \
  --source-root <source-pool> --out-root <seed-root> \
  --seed 20260916 --per-source 143 --folds 8
```

For `<seed-root>/images` and each `<seed-root>/folds/f0` through `f7`, run the
exact covariance extractor.  Store the full result as `<seed-root>/cov` and the
fold results as `<seed-root>/fold_cov/f0` through `f7`:

```bash
/tmp/prism-v7-env/bin/python -m actnull.covariance \
  --arch clip_vision --model <pinned-base-snapshot> \
  --images <image-directory> --out <covariance-output> \
  --exact_cov --m_out 64 --max_seqs 2000 --batch 4 \
  --device cpu --dtype float32
```

Then the portable wrapper resolves all 20 specialist revisions from the pin
record and rebuilds the 13,680 exact rows:

```bash
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/run_resample_exact.py \
  --revision-record artifact/upstream_revisions.json \
  --hf-home <hugging-face-cache> --seed-root <seed-root> \
  --out <refit-row-csv>
```

Repeat with seed `20260917`.  The shipped occurrence maps specify all 1,001
draws, including duplicates and fold assignment, and the checker replays their
RNG construction.  The two conditional crossed intervals hold each fitted
calibration covariance fixed; they are finite-pool stability intervals, not
calibration-resampling or superpopulation confidence intervals.

`scripts/calibration_source_subgroup_diagnostic.py` is a read-only compact
diagnostic over the shipped exact row files. It recomputes the 0/1/2
calibration-source endpoint strata without requiring model weights or covariance
tensors.

The deliberately different MNIST/Stanford-Cars/SVHN composition control uses
immutable revisions and exact cached Arrow-shard hashes.  Materialize its
334/334/333 sample with:

```bash
/tmp/prism-v7-env/bin/python \
  artifact/code/scripts/materialize_composition_control.py \
  --revision-record artifact/upstream_revisions.json \
  --datasets-cache <datasets-cache> --out-root <composition-root> \
  --seed 20260919 --folds 8
```

Run the same full/fold covariance and `run_resample_exact.py` commands on that
root.  The compact control in `results/calibration_composition_b16/` records all
1,001 selected indices and image hashes, and yields 21.482% with a conditional
crossed interval of [16.898, 26.269]%.  No source image is redistributed.

## Reproduce the B/32 merge-accuracy comparison

The released evaluator resolves the base, eight specialists, and eight test
datasets through `upstream_revisions.json`; no repository HEAD is accepted. It
uses the recorded seed, 500 examples per task, scale 0.3, and ranks
`8,16,32,64,128,256`, and refuses to overwrite an existing result:

```bash
/tmp/prism-v7-env/bin/python artifact/code/scripts/merge_eval.py \
  --revision-record artifact/upstream_revisions.json \
  --out /tmp/merge_b32.json
```

`scripts/lora_topk_gap_diagnostic.py` similarly reads pinned, locally
materialized adapter snapshots and reproduces the compact boundary-gap audit.

## Scope

Weights, covariance tensors, fold tensors, and sensitivity cell caches are too
large for this compact bundle.  They are reconstructible from the pinned public
inputs and these scripts.  The shipped rows, maps, summaries, and checker are
sufficient for an entirely offline audit of the overlap estimates and listed
robustness/diagnostic outputs. The per-task merge accuracies are shipped for an
offline table audit; repeating its forward passes requires the pinned upstream
weights and test datasets.

The old internal number checker and figure-rendering helpers are intentionally
not shipped: they contained superseded values or were unnecessary for numerical
reconstruction.  The shipped checkers are `analysis/check_paper_numbers.py` and
`analysis/check_manuscript.py`.
