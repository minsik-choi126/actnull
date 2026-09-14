#!/usr/bin/env python3
"""Type-I calibration for activation-conditioned whitening. RUN THIS BEFORE ANY REAL DATA.

⛔ THE FINDING THAT MOTIVATES THIS FILE. Whitening two task vectors by a SHARED activation
covariance manufactures subspace alignment that is not there. Under a null where the two
task vectors are drawn independently, so the true read-subspace overlap is chance (k/d):

    no whitening                         0.0391   (chance 0.0400)
    Tikhonov lambda = 1e-4 * mean(w)     0.7139   <- 18x chance, pure artifact
    Tikhonov lambda = 1e-2 * mean(w)     0.2859
    Tikhonov lambda = 1e-1 * mean(w)     0.0996
    Tikhonov lambda = 1     * mean(w)    0.0496

The mechanism is that C_H^{-1/2} amplifies the SMALL-eigenvalue directions of C_H. At
lambda = 1e-4 * mean(w), 96.5% of the whitened read subspace's mass lands on C_H's 20
smallest eigendirections. Both task vectors get pushed onto the same few directions, and
they then look aligned because the whitening put them there, not because they share
anything. lambda is therefore NOT a nuisance parameter. It sets the false positive rate of
the entire test, and a conventional "small ridge for numerical stability" choice makes the
test report a large effect on pure noise.

THE FIX THAT WORKS. Whiten only the top-m eigendirections of C_H and leave the complement
untouched. Type-I returns to nominal and stays there over a wide range of m:

    top-8 directions whitened            0.0397
    top-32                               0.0395
    top-64                               0.0412
    top-128 (of d=200)                   0.0640   <- degrades as m -> d

m must be CHOSEN BY CALIBRATION on the null, per (layer, submodule), and reported. Do not
inherit a lambda from RegMean / ACTMat / ACE-Merging: those use whitening to build a
merge, where an inflated shared direction costs a little accuracy, not to test a
hypothesis, where it costs the conclusion. None of those papers runs a null model at all.

⛔ AND DO NOT WHITEN BY THE TASK VECTOR'S OWN GRAM. With dW = U S V^T,

    dW (dW^T dW + lambda I)^{-1/2} = U S (S^2 + lambda I)^{-1/2} V^T

whose right singular vectors are EXACTLY V for every lambda > 0. Verified numerically to
1e-10 at lambda >= 1e-3. So the data-free covariance estimators of ACTMat (C_hat =
Delta^T Delta) and ACE-Merging (a mean-centred version of the same) are an exact no-op on
the read subspace this test measures: the null test would run on unwhitened task vectors
while appearing to run on whitened ones. A real, data-derived, task-INDEPENDENT C_H is
required, which keeps activation extraction on the critical path.

Usage:
  python analysis/act_whitening_calibration.py                  # the null study above
  python analysis/act_whitening_calibration.py --power          # add the signal case
"""
from __future__ import annotations

import argparse
import numpy as np

np.seterr(all="ignore")


def top_read(A: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular (read-side) subspace, as an orthonormal basis."""
    return np.linalg.svd(A, full_matrices=False)[2][:k].T


def overlap(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Mean squared canonical correlation between the two read subspaces.

    Equals k/d in expectation for independent isotropic subspaces, which is the chance
    level every number below is compared against."""
    return float((np.linalg.svd(top_read(A, k).T @ top_read(B, k), compute_uv=False) ** 2).mean())


def eig_desc(C: np.ndarray):
    w, Q = np.linalg.eigh(C)
    i = np.argsort(w)[::-1]
    return np.clip(w[i], 0, None), Q[:, i]


def whiten_tikhonov(D, w, Q, lam):
    return D @ (Q @ np.diag(1.0 / np.sqrt(w + lam)) @ Q.T)


def whiten_regmean(D, C, alpha):
    """RegMean's actual regularizer: shrink the OFF-diagonal entries toward the diagonal.

        G~ = alpha * G + (1 - alpha) * diag(G)

    Jin et al., ICLR 2023, Eq. (2) and the transformer implementation, published default
    alpha = 0.9. This is included because it is the regularizer the merging literature
    actually uses, and at its own default it does NOT hold type-I: the null lands at 0.0920
    against a chance level of 0.0400. Shrinking off-diagonals reduces the condition number
    but sets no floor on the smallest eigenvalue, which is the quantity that drives the
    inflation."""
    G = alpha * C + (1 - alpha) * np.diag(np.diag(C))
    w, Q = eig_desc(G)
    return D @ (Q @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ Q.T)


def whiten_truncated(D, w, Q, m):
    """Whiten the top-m eigendirections only; leave the complement at unit scale.

    This is the estimator that holds type-I. Scaling inside the block is relative to the
    block's own mean eigenvalue so the transform is unit-free."""
    dinv = np.ones(len(w))
    blk = w[:m]
    dinv[:m] = 1.0 / np.sqrt(np.maximum(blk / blk.mean(), 1e-12))
    return D @ (Q @ np.diag(dinv) @ Q.T)


def lowrank(rng, d_out, d_in, r):
    return rng.normal(size=(d_out, r)) @ rng.normal(size=(r, d_in))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d_in", type=int, default=200)
    ap.add_argument("--d_out", type=int, default=60)
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--k", type=int, default=8, help="read subspace budget")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--power", action="store_true", help="also report the signal case")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d_in, d_out, r, k = args.d_in, args.d_out, args.rank, args.k
    chance = k / d_in

    H = rng.normal(size=(d_in, d_in))
    C_H = H @ H.T / d_in
    w, Q = eig_desc(C_H)
    print(f"C_H condition number {w[0]/max(w[-1],1e-12):.1e}   chance level k/d = {chance:.4f}\n")

    def sweep(make_pair, title, flag_inflation=True):
        print(f"  {title}")
        print(f"    {'estimator':34s} {'mean':>8s} {'95%ile':>8s}")

        def run(f):
            v = [overlap(f(a), f(b), k) for a, b in (make_pair() for _ in range(args.trials))]
            return np.mean(v), np.percentile(v, 95)

        m, p = run(lambda D: D)
        print(f"    {'no whitening':34s} {m:8.4f} {p:8.4f}")
        for lf in (1e-4, 1e-2, 1e-1, 1.0):
            m, p = run(lambda D, l=lf * w.mean(): whiten_tikhonov(D, w, Q, l))
            tag = "  <- inflated" if flag_inflation and m > 2 * chance else ""
            print(f"    {'Tikhonov lam=%.0e*mean(w)' % lf:34s} {m:8.4f} {p:8.4f}{tag}")
        for a in (1.0, 0.9, 0.5, 0.1):
            m, p = run(lambda D, aa=a: whiten_regmean(D, C_H, aa))
            tag = "  <- inflated" if flag_inflation and m > 2 * chance else ""
            lab = "RegMean shrink alpha=%.2f%s" % (a, " (published default)" if a == 0.9 else "")
            print(f"    {lab:34s} {m:8.4f} {p:8.4f}{tag}")
        for mm in (8, 32, 64, 128):
            if mm > d_in:
                continue
            m, p = run(lambda D, q=mm: whiten_truncated(D, w, Q, q))
            tag = "  <- inflated" if flag_inflation and m > 2 * chance else ""
            print(f"    {'truncated, top-%d' % mm:34s} {m:8.4f} {p:8.4f}{tag}")
        print()

    print("NULL: two independently drawn task vectors. Anything above chance is type-I error.")
    sweep(lambda: (lowrank(rng, d_out, d_in, r), lowrank(rng, d_out, d_in, r)), "independent pair")

    if args.power:
        print("SIGNAL: a shared read direction beyond the activation span. Should survive.")
        S = rng.normal(size=(d_in, r))

        def pair():
            return (rng.normal(size=(d_out, r)) @ S.T + 0.3 * lowrank(rng, d_out, d_in, r),
                    rng.normal(size=(d_out, r)) @ S.T + 0.3 * lowrank(rng, d_out, d_in, r))
        sweep(pair, "shared-signal pair", flag_inflation=False)

    print("READING THIS: pick the estimator whose NULL row sits at chance, then read its")
    print("SIGNAL row. An estimator that inflates the null cannot be rescued by a large")
    print("signal number, because the signal number is inflated by the same mechanism.")


if __name__ == "__main__":
    main()
