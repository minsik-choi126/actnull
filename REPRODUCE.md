# Reproducing the results

Every headline number is recomputable from the committed `artifact/` bundle without a GPU, model
weights, or network access. Rebuilding the bundle from upstream checkpoints is a separate, much
longer path described at the end.

**Run every command from the repository root.** A build or test invoked from inside
`artifact/code/` writes `.venv`, `uv.lock`, and `__pycache__` next to the shipped sources, and
`analysis/check_manuscript.py` reports that as a manifest-coverage failure.

## 1. Audit the committed results

No install required. Both checkers use only the Python standard library.

```bash
shasum -a 256 -c artifact/RESULTS.sha256

python3 analysis/check_paper_numbers.py

PYTHONDONTWRITEBYTECODE=1 python3 analysis/check_manuscript.py \
  --paper iclr2027_conference.tex --artifact artifact
```

The first recomputes the six row-level ratio-of-means estimates, the 50,000-draw crossed
bootstrap intervals, the sensitivity grids, leave-one-task-out and leave-one-layer-out influence,
the power and ablation normalisations, the merge table, the LoRA factor audit against the
released product rows, and every manifest entry. The second runs that audit and then checks the
manuscript's registered claims in their stated context.

## 2. Run the producer tests

This creates the locked environment outside the tree, which is what keeps the audit clean.

```bash
UV_PROJECT_ENVIRONMENT=/tmp/prism-v7-env \
  uv sync --project artifact/environment --all-extras --no-install-project --locked

export PYTHONPATH="$PWD/artifact/code"
PYTHONDONTWRITEBYTECODE=1 /tmp/prism-v7-env/bin/python -m unittest discover \
  -s artifact/code/tests -p 'test_*.py' -v
```

## 3. Rebuild the figures

```bash
python3 figures/make_fig_power.py
python3 figures/make_fig_main.py
```

## 4. Recompute from upstream checkpoints

`artifact/code/scripts/` holds the producer entry points, and `artifact/upstream_revisions.json`
pins the exact checkpoint and dataset revisions each one reads. The order is: fetch calibration
images, estimate the activation covariance, then run the overlap, power, ablation, baseline,
projector, and merge scripts.

Two costs dominate. The covariance estimation reads 1001 calibration images through three CLIP
encoders. The merge evaluation needs the pinned upstream weights and test datasets, which is why
per-task accuracies are shipped as a table instead.

`artifact/code/README.md` gives the per-script invocations, including the byte-for-byte
reconstruction of the pooled MC64 LoRA rows. `artifact/calibration_link.json` fingerprints the
large calibration intermediates that the bundle deliberately omits.
