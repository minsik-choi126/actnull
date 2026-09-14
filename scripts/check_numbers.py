#!/usr/bin/env python3
"""Verify every number the paper asserts against the CSVs it came from.

⛔ WHY. A round-1 reviewer found the draft claiming a regime gap of +48.5 points where the
paper's own table entries give 49.00, and the headline explained fraction moving 3.7 points
depending on which pairs the null was averaged over. Neither is visible in a clean LaTeX build:
both compile, both look right, and both are wrong. Text and data drift apart every time a
measurement is re-run, and re-runs are frequent here.

This reads results/act/*.csv, recomputes the quantities the sections quote, and reports any
disagreement beyond a tolerance. It is not a linter for prose. It answers one question: does the
paper still say what the data says?

Usage:  python analysis/check_paper_numbers.py            # check
        python analysis/check_paper_numbers.py --print    # show current values, to paste in
"""
from __future__ import annotations

import argparse, csv, math, os, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = Path(os.environ.get("ACTNULL_RESULTS", ROOT / "results"))
# The manuscript is not in this repository. Point ACTNULL_SECTIONS at its sections/ directory to
# check the text against the data; without it the script verifies the results files alone.
SEC = Path(os.environ.get("ACTNULL_SECTIONS", ROOT / "sections"))


def load(name):
    p = ACT / f"{name}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for c in r:
            if c not in ("module", "a", "b", "null_kind"):
                r[c] = float(r[c])
    return rows


def summary(rows):
    raw = st.mean(r["raw"] for r in rows)
    iso = st.mean(r["null_iso"] for r in rows)
    nul = st.mean(r["null_act_raw"] for r in rows)
    h0 = st.mean(r.get("h0_excess", 0.0) for r in rows)
    return dict(pairs=len({(r["a"], r["b"]) for r in rows}), rows=len(rows),
                raw=raw, iso=iso, null=nul, h0=h0,
                excess=raw - nul, excess_corrected=raw - nul - h0,
                explained=100 * (nul - iso) / (raw - iso) if raw > iso else float("nan"),
                explained_h0=100 * (nul + h0 - iso) / (raw - iso) if raw > iso else float("nan"))


RESIDUAL = ("q_proj", "k_proj", "v_proj", "fc1")   # read the residual stream
INTERNAL = ("out_proj", "fc2")                     # read a representation built inside the block
# q and v are the only modules ViT-L/14 was run on, so the cross-architecture comparison in
# Table 1's prose is made on that subset and the checker has to know it.


def power_values():
    """Recovery percentages from the planted-structure study, which lives in its own JSON."""
    import json
    out = set()
    for tag in ("shipped", "current", "lora_shipped", "lora_current"):
        p = ACT / f"power_{tag}.json"
        if p.exists():
            for r in json.loads(p.read_text())["rows"]:
                for key in ("recovery", "delivered", "leak"):
                    if r.get(key) is not None:
                        out.add(round(r[key], 1))
    for tag in ("lora_shipped", "lora_current"):
        p = ACT / f"power_{tag}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())["rows"]
        base = {r["planted"]: r for r in rows}[0]
        for r in rows:
            if r["truth"] > 0:
                rec = ((r["raw"] - base["raw"]) - (r["null"] - base["null"])) / r["truth"]
                out.add(round(100 * rec, 1))
    # The paper also quotes recovery divided by delivery, the null's own pass-through, so the
    # ratio has to be a legitimate value too.
    import json as _j
    q = ACT / "power_current.json"
    if q.exists():
        for r in _j.loads(q.read_text())["rows"]:
            if r.get("recovery") and r.get("delivered"):
                out.add(round(100 * r["recovery"] / r["delivered"], 1))
    return out


def baseline_values():
    """Level of the baselines in use, from the JSON that Appendix~\\ref{app:baselines} reports."""
    import json
    q = ACT / "baseline_levels.json"
    if not q.exists():
        return set()
    d = json.loads(q.read_text())
    iso = d["iso"]
    return {round((iso + v) / iso, 1) for v in d["level"].values()}


def ablation_values():
    """Recovery per defect-fix combination, from the ablation JSON."""
    import json
    q = ACT / "ablation.json"
    if not q.exists():
        return set()
    rows = json.loads(q.read_text())
    O0 = json.loads((ACT / "power_current.json").read_text())["rows"][0]["raw"]
    target = (2 / 8) * (1 - O0)
    out = set()
    base = next(r["excess"] for r in rows if "shipped" in r["variant"])
    for r in rows:
        out.add(round(100 * r["excess"] / target, 1))
        out.add(round(100 * (r["excess"] - base) / target, 1))
    return out


def projector_values():
    """Survival fractions from the projector rebuild, which is its own CSV shape."""
    out = set()
    for tag in ("fullft", "lora16"):
        q = ACT / f"v6_projector_b16_{tag}.csv"
        if not q.exists():
            continue
        rs = list(csv.DictReader(open(q)))
        for col in ("projector_survives_null", "chance", "projector_in_top_activation"):
            if rs and col in rs[0]:
                out.add(round(100 * st.mean(float(r[col]) for r in rs), 1))
    return out


def group_explained(rows):
    """Explained fractions per module type and per module group.

    ⛔ WHY. Table 1 reports the split because the whole-arm average mixes two populations that
    differ by a factor of two. A checker that only knows whole-arm averages would flag every
    entry of that split as unbacked, and the tempting fix, declaring them as sourced elsewhere,
    would turn the check into a rubber stamp. So compute them here instead.
    """
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["module"].split(".")[-1]].append(r)
    out = {}
    def frac(rs, use_h0=True):
        raw = st.mean(r["raw"] for r in rs)
        iso = st.mean(r["null_iso"] for r in rs)
        nul = st.mean(r["null_act_raw"] for r in rs)
        h0 = st.mean(r.get("h0_excess", 0.0) for r in rs) if use_h0 else 0.0
        return 100 * (nul + h0 - iso) / (raw - iso) if raw > iso else float("nan")
    for t, rs in by.items():
        out[t] = frac(rs)
    for name, types in (("residual", RESIDUAL), ("internal", INTERNAL),
                        ("qv", ("q_proj", "v_proj"))):
        rs = [r for t in types for r in by.get(t, [])]
        if rs:
            out[name] = frac(rs)
            # The paper reports the split both with and without the H0 term, so both are
            # legitimate values for a number in the text to be.
            out[name + "_noh0"] = frac(rs, use_h0=False)
    return out


def spearman(x, y):
    def rank(z):
        o = sorted(range(len(z)), key=lambda i: z[i]); R = [0] * len(z)
        for i, j in enumerate(o):
            R[j] = i
        return R
    a, b = rank(x), rank(y); ma, mb = st.mean(a), st.mean(b)
    num = sum((i - ma) * (j - mb) for i, j in zip(a, b))
    den = math.sqrt(sum((i - ma) ** 2 for i in a) * sum((j - mb) ** 2 for j in b))
    return num / den if den else float("nan")


def rank_stability(rows):
    byp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        byp[(r["a"], r["b"])]["raw"].append(r["raw"])
        byp[(r["a"], r["b"])]["ex"].append(r.get("excess_raw_corrected", r["excess_raw"]))
    P = sorted(byp)
    X = [st.mean(byp[p]["raw"]) for p in P]
    Y = [st.mean(byp[p]["ex"]) for p in P]
    k = min(20, max(2, len(P) // 2))
    top = len(set(sorted(range(len(P)), key=lambda i: -X[i])[:k])
              & set(sorted(range(len(P)), key=lambda i: -Y[i])[:k]))
    return spearman(X, Y), top, k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="absolute tolerance in percentage points for the explained fraction")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="also list the percentages sourced outside results/act/*.csv")
    args = ap.parse_args()

    if not SEC.is_dir():
        print(f"  [skip] no manuscript at {SEC}; set ACTNULL_SECTIONS to check the text too.")
        tex = ""
    else:
        tex = "".join((SEC / f).read_text() for f in sorted(p.name for p in SEC.glob("*.tex")))
    problems, checked = [], 0

    ARMS = [("v6_clip-vit-base-patch32", "ViT-B/32"),
            ("v6_clip-vit-base-patch16", "ViT-B/16"),
            ("v6_clip-vit-large-patch14", "ViT-L/14"),
            ("v6_b16_8task_q_v_fullft", "full FT 8task"),
            ("v6_b16_8task_q_v_lora16", "LoRA-16 act null"),
            ("v6_b16_8task_q_v_llora16", "lin-LoRA act null"),
            ("v6_b16_8task_q_v_lora-16_loranull", "LoRA-16"),
            ("v6_b16_8task_q_v_l-lora-16_loranull", "lin-LoRA"),
            # the m sweep of Appendix B: the null's protected/destroyed boundary
            ("msweep_clip-vit-base-patch16_m16", "m=16"),
            ("msweep_clip-vit-base-patch16_m32", "m=32"),
            ("msweep_clip-vit-base-patch16_m64", "m=64")]

    for name, label in ARMS:
        rows = load(name)
        if rows is None:
            problems.append(f"MISSING  {name}.csv  (referenced as {label})")
            continue
        s = summary(rows)
        rho, top, k = rank_stability(rows)
        if args.show:
            print(f"  {label:16s} pairs={s['pairs']:4d} rows={s['rows']:6d} "
                  f"raw={s['raw']:.5f} iso={s['iso']:.5f} null={s['null']:.5f} "
                  f"H0={s['h0']:+.5f} expl={s['explained']:.1f}% "
                  f"expl_h0={s['explained_h0']:.1f}% rho={rho:.4f} top={top}/{k}")
        # every "NN.N%" in the text must exist in some arm, within tolerance
        checked += 1

    # Percentages the paper takes from measurements that do not live in results/act/*.csv.
    # Declaring the source here is the point: an undeclared number that no arm reproduces is a
    # drift bug, and without this registry every such number would look like one.
    ELSEWHERE = {
        19.7: "analysis/act_test.py lora_shared_init_null docstring: singular values move "
              "19.7% under B -> BQ, which is why that construction was rejected",
        2.0: "Table 3: the isotropic chance level r_p/d for the LoRA projector, 16/768",
        4.0: "analysis/act_test.py coupling_destroying_null: the shipped null rotated 4% of the "
             "complement at d_in=3072",
        5.2: "analysis/act_test.py act_matched_null docstring: the four leading activation "
             "directions of layers.0.mlp.fc1 hold 5.2% of the real read mass",
        48.0: "appendix and method: the rotation-rank sweep on the superseded fixed-block null, "
              "quoted as evidence that the knob had to go",
        67.0: "results/act/k1000_clip-vit-base-patch16.csv: the headline before the defects of "
              "Section 4.3 were corrected, quoted as history",
        71.0: "the singleton blocks' share of the stored eigenvalue mass on ViT-B/16, from the "
              "block structure the fold covariances resolve",
        38.9: "scratchpad/merge_b32.json: zero-shot accuracy averaged over the eight tasks",
        59.5: "scratchpad/merge_b32.json: uncompressed task-arithmetic accuracy, eight tasks",
        63.7: "analysis/act_test.py act_matched_null docstring: the four leading activation "
              "directions of layers.0.mlp.fc1 hold 63.7% of the activation energy. This is an "
              "energy share, not an explained fraction, and it coincided with ViT-B/32's "
              "explained fraction under a superseded null",
        74.7: "analysis/act_test.py coupling_destroying_null: a coordinate permutation on the "
              "complement leaks 74.7% of ||R|| into span(V)",
        96.5: "analysis/act_whitening_calibration.py: whitened read mass on the twenty "
              "smallest stored eigendirections at lambda = 1e-4 wbar",
    }

    if not args.show:
        # ⛔ An earlier pattern was \d{2}\.\d, which saw only the percentages written with a
        # decimal. That is 21 of the 39 in the text, and the ones it skipped included every
        # headline in the abstract. Integers count too.
        # An integer in the text is a rounded value, so it matches anything within half a
        # point. A value written with a decimal is a claim to that precision and gets --tol.
        raw_pct = re.findall(r"\$(\d{1,3}(?:\.\d)?)\\%\$", tex)
        pct = {(float(x), 0.5 if "." not in x else args.tol) for x in raw_pct}
        have = set()
        for name, _ in ARMS:
            rows = load(name)
            if rows:
                s = summary(rows)
                have |= {round(s["explained"], 1), round(s["explained_h0"], 1),
                         round(100 - s["explained_h0"], 1)}
                have |= {round(v, 1) for v in group_explained(rows).values()
                         if v == v}
        have |= power_values()
        have |= projector_values()
        have |= ablation_values()
        have |= baseline_values()
        for v, tol in sorted(pct):
            if any(abs(v - h) <= tol for h in have):
                continue
            if v in ELSEWHERE:
                if args.verbose:
                    print(f"  declared  {v}%  <- {ELSEWHERE[v]}")
                continue
            problems.append(f"UNBACKED  the text asserts {v}% and no arm produces it, and it is "
                            f"not declared in ELSEWHERE (arms give {sorted(have)})")
        for v in sorted(ELSEWHERE if tex else []):
            if not any(abs(v - q) <= max(tol, args.tol) for q, tol in pct):
                problems.append(f"STALE     ELSEWHERE declares {v}% but no section quotes it "
                                f"any more; drop the entry")

    # ⛔ rank_stability() used to be computed and printed but never compared with the text, and
    # a stale Spearman from a superseded arm sat in the paper through two review rounds because
    # of it. Compare it.
    for name, label in ARMS:
        rows = load(name)
        if rows is None:
            continue
        rho, top, k = rank_stability(rows)
        for want in re.findall(r"Spearman \$(\d\.\d+)\$", tex):
            if abs(float(want) - rho) <= 0.0005:
                break
        else:
            continue
        if not re.search(rf"\${top}\$ of the top \${k}\$", tex):
            problems.append(f"RANK      {label}: rho {rho:.4f} matches the text but the text "
                            f"does not say {top} of the top {k} are shared")

    print(f"\n  checked {checked} arms, {len(problems)} problems")
    for p in problems:
        print(f"  ⛔ {p}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
