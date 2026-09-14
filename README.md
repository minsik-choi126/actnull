# Calibrated null models for task-vector subspace overlap

> **Under review.** The accompanying paper is under review at ICLR 2027. This repository is the
> code and the measured results behind it.

Model merging research reports that task vectors from independently fine-tuned models share
input-side subspaces, and compares that overlap against an isotropic random subspace. That
baseline is wrong. A fine-tuning update is a sum of outer products with layer input activations,
so its row space lies inside a span every model descended from one checkpoint inherits for free.

This repository provides a null that preserves the inherited geometry and destroys only whether
two specialists chose the same directions inside it, and a protocol for checking whether any such
null actually does what it claims.

## What the protocol says about the baselines in use

On a pair of updates that share the base model's activation geometry and nothing else, so the
truth is zero:

| baseline | reported excess | x chance |
|---|---:|---:|
| exact orthogonality | +0.0278 | 3.7 |
| isotropic *k/d* | +0.0174 | 2.7 |
| whitened at the participation ratio | +0.0521 | 6.0 |
| whitened, all stored directions | +0.0567 | 6.4 |
| coordinate permutation | +0.0000 | 1.0 |

Whitening the task vectors first, a transform introduced to reduce interference inside a merge,
more than doubles the false signal when the same object is read as evidence.

Run it yourself with `scripts/baseline_levels.py` (see [REPRODUCE.md](REPRODUCE.md)).

## Why level is not enough

The usual check on a null is that it returns zero excess on data built to satisfy the null
hypothesis. Writing `N` for the null and `(A0, B0)` for a matched true pair, that statistic is

```
E[ O(A0, B0) - O(N A0, N B0) ]
```

which is exactly zero for `N = id`. A null that destroys nothing earns a perfect score, and so
does one that destroys a tenth of what it claims to. Under-destruction is the failure that makes
real structure look inherited, and the standard check cannot see it.

So measure power instead: plant a known number of shared read directions into an otherwise null
pair and ask how many the test recovers. Run on our own null this found three defects the level
check had passed, and correcting them raised recovery from 44% to 91%.

![power](figures/fig_power.pdf)

## Layout

```
actnull/
  null.py         the overlap statistic, the activation-conditioned null, the LoRA null,
                  resolvable_blocks, and the checkpoint readers
  covariance.py   the activation second moment C_H of a frozen base model, exact or two-pass
scripts/
  fetch_calibration_images.py   build the calibration set
  power_activation.py           planting protocol on the activation null
  power_lora.py                 planting protocol on the LoRA null
  baseline_levels.py            level of the baselines this literature uses
  ablation.py                   which correction did the work
  projector_rebuild.py          rebuild a published projector from null-passed task vectors
  merge_eval.py                 merged accuracy by rank budget
  whitening_calibration.py      type-I study for the whitening truncation
  check_numbers.py              verify every number in the paper against the results
results/        the measured outputs, committed
figures/        the two figures in the paper
```

## Install

```bash
pip install -e .
```

Optional extras: `pip install -e '.[data,figures]'` for the calibration-set builder and the
figure scripts.

## Reproducing

See [REPRODUCE.md](REPRODUCE.md). The short version: the calibration covariances come to 26 GB
and are not distributed, but `results/calibration/calibration_1001.json` records the exact image
list, the per-source counts, a hash, and the fold assignment, which is enough to rebuild them.

## Verifying the numbers

```bash
python scripts/check_numbers.py
```

This reads `results/` and recomputes every percentage the paper asserts, failing on any that no
arm produces and is not declared with its source. It is the guard against text and data drifting
apart, which happened repeatedly while the paper was being written.

## License

MIT. See [LICENSE](LICENSE).
