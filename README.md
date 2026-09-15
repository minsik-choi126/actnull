# Conditional nulls and positive controls for task-vector subspace overlap

> **Under review.** The accompanying paper is under review at ICLR 2027. This repository holds the
> manuscript source, the measured results behind it, the producer code, and the checkers that tie
> the three together.

Model merging research reports that task vectors from independently fine-tuned models share
input-side subspaces, and compares that overlap against an isotropic random subspace. A
fine-tuning update is a sum of outer products with layer input activations, so its row space lies
inside a span that every model descended from one checkpoint inherits for free. The isotropic
comparator conditions on none of that.

This repository provides a conditional null that preserves each update's spectrum and blockwise
occupancy in frozen-base activation geometry while randomising cross-task directional agreement,
a closed form for its expected overlap, and a positive control that checks whether any such null
actually destroys what it claims to.

## Headline results

On the complete ViT-B benchmarks, the corrected activation null reproduces **25.3%** (B/16) and
**22.4%** (B/32) of above-isotropic overlap. Residual alignment stays positive in every fully
fine-tuned pool. For shared-start LoRA a regime-matched factor null reproduces **80.3%**, but a one-line count from
the shared rank already gives **82.0%**: under a shared start the admissible row space, not a
learned basis, accounts for nearly all observed overlap.

The positive control is what makes those numbers reportable. Our first implementation recovered
only **45.6–51.7%** of planted, delivered signal while returning `+0.00004` at zero plant, which
looks well behaved under a level check alone. After correction the same arm recovers
**99.3–99.6%**.

Module role moves the fraction far more than encoder scale does: within ViT-B/16 the null
reproduces 16.6% for modules reading the residual stream against 39.0% for modules reading a
representation built inside the block. The apparent gradient across encoders is mostly module
selection, and collapses from 13.5 points to 2.8 once every pool is restricted to the `q`/`v`
modules all three share.

## What the protocol says about baselines in use

On a reference pair that shares the base model's activation geometry and nothing else, so the
truth is zero:

| baseline | reported excess |
|---|---:|
| exact orthogonality | +0.0275 |
| isotropic *k/d* | +0.0171 |
| whitened at the participation ratio | +0.0522 |
| whitened, all stored directions | +0.0554 |
| coordinate permutation (paired action) | −0.0000 |
| activation-conditioned (ours, circular here) | −0.0007 |

The untransformed pair's raw overlap is 0.0275, or 2.6 times isotropic chance. Whitening, a
transform introduced to reduce interference inside a merge, reports still larger native excess
when the same object is read as evidence. Entries share overlap units but not an estimand;
our own row is circular here and is shown only as an implementation check.

## Layout

```
artifact/          the v7 result bundle: row-level data, producer source, tests,
                   locked environment, upstream pins, RESULTS.sha256
analysis/          the two release checkers and their tests
sections/          manuscript source
figures/           the two paper figures and their generators
iclr2027_*.tex     manuscript entry point and ICLR style files
iclr2027_*.pdf     the compiled manuscript
```

`artifact/README.md` documents the bundle contents and every reproduction path.
`artifact/code/README.md` documents the producer source and its tests.

## Verify

Run from the repository root.

```bash
# file integrity
shasum -a 256 -c artifact/RESULTS.sha256

# numerical audit of the v7 results, standard library only
python3 analysis/check_paper_numbers.py

# the same audit plus a context-anchored check of manuscript claims
PYTHONDONTWRITEBYTECODE=1 python3 analysis/check_manuscript.py \
  --paper iclr2027_conference.tex --artifact artifact
```

`check_manuscript.py` also verifies manifest coverage, so it fails if a build or test writes
files into the shipped tree. Never invoke one from inside `artifact/code/`; see
[REPRODUCE.md](REPRODUCE.md).

The coverage inventory it prints is deliberately partial. It checks registered empirical claims
in their stated context rather than every numeric literal in the TeX, and design constants,
mathematical examples, and literature values appear there as warnings.

## Scope

All quantitative conclusions concern CLIP vision encoders. The decomposition is conditional on a
stated reference model, not causal: it reports how much observed overlap survives holding each
update's own occupancy of a common activation coordinate system fixed. It does not locate where
the agreement originates.

## License

MIT. See [LICENSE](LICENSE).
