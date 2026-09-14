#!/usr/bin/env python3
"""How much of a published merging method's projector basis is inherited rather than learned?

WHY. Correcting an effect size is a weaker claim than showing that a live construction rests on
the part being corrected. NUFILT (ICLR 2026) filters each incoming task vector through
P = I - V V^T where V holds the top-r_p RIGHT singular vectors of the cumulative update
tau = theta_merged - theta_0. That basis is exactly the cross-task read-subspace structure this
paper argues is largely inherited, so it can be measured rather than argued about.

WHAT IT COMPUTES. Build tau from a real pool, take its top-r_p right singular subspace V_real.
Then rebuild tau from the SAME task vectors after each has been passed through the
coupling-destroying null -- same spectra, same read-mass profiles, no cross-task coupling -- and
take V_null. The overlap between V_real and V_null is the fraction of the projector's basis that
survives having the shared capability removed, i.e. the fraction that never depended on it.

⛔ THIS IS NOT A CLAIM THAT THE METHOD DOES NOT WORK. NUFILT reports 4-7 point gains and nothing
here contradicts that. The claim is narrower and it is about attribution: if the basis is
reproduced by task vectors that share only their activation geometry, then the reported gains
are not evidence that the basis captures shared task structure.

Usage:
  python analysis/projector_inheritance.py --cov <dir> --base <ckpt> --key_prefix vision_model. \
      --experts a=... b=... --rp 128 --draws 8
"""
from __future__ import annotations

import argparse, itertools, json, statistics as st
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import actnull.null as act



def top_right(M, k):
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T


def overlap(A, B):
    return float((torch.linalg.svdvals(A.T @ B) ** 2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--key_prefix", default="")
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--experts_are_adapters", action="store_true")
    ap.add_argument("--null", default="coupling", choices=["coupling", "lora"])
    ap.add_argument("--rp", type=int, default=128, help="NUFILT's projector rank")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fold_cov", default="",
                    help="directory of per-fold covariances (f0, f1, ...). Given this, the "
                         "rotation blocks are the runs of eigendirections the calibration set "
                         "cannot separate, exactly as in act_test.py, and --null_blocks is "
                         "unused. Without it the null falls back to a fixed block count, which "
                         "is a free parameter the paper does not otherwise have.")
    ap.add_argument("--null_blocks", type=int, default=16)
    ap.add_argument("--block_z", type=float, default=2.0)
    ap.add_argument("--tail_rank", type=int, default=128)
    ap.add_argument("--stride", type=int, default=6, help="score every Nth module")
    ap.add_argument("--only_modules", default="",
                    help="comma-separated submodule names, e.g. q_proj,v_proj. Required for a "
                         "LoRA pool, whose adapters touch only some modules; without it most "
                         "sampled modules have a zero task vector and are skipped.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cov = Path(args.cov)
    man = json.loads((cov / "manifest.json").read_text())
    base = act.LazyCheckpoint(args.base, args.key_prefix)
    experts = {}
    for spec in args.experts:
        n, p = spec.split("=", 1)
        experts[n] = (act.AdapterCheckpoint(p) if args.experts_are_adapters
                      else act.LazyCheckpoint(p, args.key_prefix))

    gen = torch.Generator(device=args.device).manual_seed(0)
    rows = []
    mods = sorted(man["modules"])
    if args.only_modules:
        keep = tuple(x.strip() for x in args.only_modules.split(",") if x.strip())
        mods = [m for m in mods if m.split(".")[-1] in keep]
    mods = mods[::args.stride]
    for mi, mod in enumerate(mods):
        key = mod + ".weight"
        blob = torch.load(cov / f"{mod.replace('.', '__')}.pt", map_location=args.device,
                          weights_only=False)
        V = blob["eigvecs"].to(args.device)
        d_in = blob["d"]
        blocks = None
        if args.fold_cov:
            fw = []
            for fd in sorted(Path(args.fold_cov).glob("f*")):
                q = fd / f"{mod.replace('.', '__')}.pt"
                if q.exists():
                    fw.append(torch.load(q, map_location="cpu",
                                         weights_only=False)["eigvals"].double())
            if len(fw) < 4:
                raise SystemExit(f"[refuse] {mod}: only {len(fw)} folds under {args.fold_cov}. "
                                 f"Eigenvalue standard errors need at least four.")
            L = min(len(x) for x in fw)
            se = (torch.stack([x[:L] for x in fw]).std(0) / len(fw) ** 0.5).float().to(args.device)
            blocks = act.resolvable_blocks(blob["eigvals"][:L].to(args.device), se, args.block_z)
        D = {}
        for n, e in experts.items():
            dW = act.load_delta(base, e, key, args.device)
            if float(dW.norm()) > 0:
                D[n] = dW
        if len(D) < 2:
            continue
        rp = min(args.rp, d_in - 1)

        tau = sum(D.values())
        V_real = top_right(tau, rp)

        # the same projector, rebuilt from task vectors with the coupling removed
        ov = []
        for _ in range(args.draws):
            if args.null == "lora":
                R = {n: experts[n].randomized(mod, args.device, gen) for n in D}
            else:
                R = {n: act.coupling_destroying_null(D[n], V, args.null_blocks, gen,
                                                     args.tail_rank, blocks=blocks) for n in D}
            ov.append(overlap(V_real, top_right(sum(R.values()), rp)))

        # for reference, how much of the projector sits in the leading activation directions
        m = V.shape[0]
        in_act = float((V @ V_real).pow(2).sum() / rp)
        rows.append(dict(module=mod, d_in=d_in, rp=rp, n_experts=len(D),
                         n_blocks=(len(blocks) if blocks is not None else args.null_blocks),
                         projector_survives_null=st.mean(ov),
                         chance=rp / d_in,
                         projector_in_top_activation=in_act,
                         activation_chance=m / d_in))
        if (mi + 1) % 6 == 0:
            print(f"[proj] {mi+1}/{len(mods)}", flush=True)

    if not rows:
        raise SystemExit("[refuse] no module had two live task vectors. For a LoRA pool pass "
                         "--only_modules q_proj,v_proj; the adapters touch nothing else.")
    if args.out:
        import csv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out}")

    print(f"\n  modules scored          {len(rows)}")
    print(f"  projector rank r_p      {rows[0]['rp']}")
    print(f"  survives the null       {st.mean(r['projector_survives_null'] for r in rows):.4f}"
          f"   (chance {st.mean(r['chance'] for r in rows):.4f})")
    print(f"  inside top activation   {st.mean(r['projector_in_top_activation'] for r in rows):.4f}"
          f"   (chance {st.mean(r['activation_chance'] for r in rows):.4f})")
    print("\n  A projector basis that survives removing the cross-task coupling did not depend"
          "\n  on it. This measures attribution, not whether the method works.")


if __name__ == "__main__":
    main()
