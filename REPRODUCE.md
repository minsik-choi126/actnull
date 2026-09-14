# Reproducing the results

Every number in the paper comes from `results/`, which is committed. This document is for
rebuilding those files from scratch.

All scripts read two environment variables:

```bash
export ACTNULL_WORK=work          # scratch: images, covariances, HF cache
export ACTNULL_RESULTS=results    # where result files are read and written
```

Everything runs on CPU. The covariances are the expensive part.

## 0. Install

```bash
pip install -e '.[data,figures]'
```

## 1. The calibration set (2 min, needs network)

`C_H` is an expectation over some input distribution, so the calibration set is part of the
measurement. We draw 1001 images round-robin across seven of the benchmark's own datasets.

```bash
python scripts/fetch_calibration_images.py --out $ACTNULL_WORK/images --per_source 143
```

`results/calibration/calibration_1001.json` records the exact file list we used, the per-source
counts, a sha256 of the list, and the assignment of each image to one of eight jackknife folds.
Check against it if you want the identical set.

Stripe the round-robin order into eight folds by position mod 8. The folds are what the
eigenvalue standard errors, and therefore the block structure, are computed from.

## 2. Activation covariances (6 min per base-size encoder, 25 min for ViT-L/14)

```bash
python -m actnull.covariance --arch clip_vision --model openai/clip-vit-base-patch16 \
  --images $ACTNULL_WORK/images_flat --max_seqs 2000 --exact_cov --m_out 64 \
  --batch 32 --device cpu --dtype float32 --out $ACTNULL_WORK/cov_b16
```

Repeat for each of the eight folds into `$ACTNULL_WORK/fold_b16/f0 ... f7`.

These come to about 26 GB across the three encoders and are not distributed. The per-module
participation ratios and condition numbers we measured are in
`results/calibration/*_manifest_1001.json` if you want to compare without rebuilding.

## 3. The overlap measurement (80 min per base-size encoder)

```bash
python -m actnull.null --base openai/clip-vit-base-patch16 --key_prefix vision_model. \
  --experts cifar10=tanganke/clip-vit-base-patch16_cifar10 ... \
  --cov $ACTNULL_WORK/cov_b16 --fold_cov $ACTNULL_WORK/fold_b16 \
  --k 8 --m auto --null coupling --null_draws 12 --h0_draws 24 --device cpu \
  --out $ACTNULL_RESULTS/v6_clip-vit-base-patch16.csv
```

`--fold_cov` is what makes the block structure data-driven; without it the null falls back to a
fixed block count, which is a free parameter the method does not otherwise have.

## 4. The checks

These are the point of the paper and they are cheap.

```bash
python scripts/baseline_levels.py     # what the baselines in use report where truth is zero
python scripts/power_activation.py    # planting protocol, activation null
python scripts/power_lora.py          # planting protocol, LoRA null
python scripts/ablation.py            # which correction did the work
```

`power_activation.py` and `power_lora.py` take `NULL_VERSION=shipped` to score the null as first
written, for the comparison in Figure 1. Both arms score the same planted pairs.

## 5. Downstream arms

```bash
python scripts/projector_rebuild.py --cov ... --fold_cov ... --rp 128 --draws 6 --stride 6
python scripts/merge_eval.py --arch clip-vit-base-patch32 --rp 8 16 32 64 128 256 --n_test 500
python scripts/whitening_calibration.py
```

`merge_eval.py` downloads the eight task test sets on first run.

## 6. Figures and verification

```bash
python scripts/fig_power.py
python scripts/fig_main.py
python scripts/check_numbers.py
```

`check_numbers.py` exits non-zero if any percentage in the paper is not reproduced by `results/`
or declared with its source.

## Notes

- Set `HF_HUB_DISABLE_XET=1` if downloads fail with a 403 from the CDN.
- The complement rotation runs in float64. The float32 path fails to converge on some residuals.
- Expect the block count to move with the calibration budget: 16 blocks at 224 images, 37 at
  1001. That is a resource-precision trade-off, not a hidden hyperparameter, and the appendix of
  the paper reports how far it moves the answer.
