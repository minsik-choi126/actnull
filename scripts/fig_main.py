"""Figure 2: 등방 초과의 분해와 모듈 역할별 분할."""
import os
import csv, json, sys, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R = os.environ.get("ACTNULL_RESULTS", "results") + "/"
RES = ("q_proj","k_proj","v_proj","fc1"); INT = ("out_proj","fc2")
ARCH = [("v6_clip-vit-base-patch16.csv","ViT-B/16"),
        ("v6_clip-vit-base-patch32.csv","ViT-B/32"),
        ("v6_clip-vit-large-patch14.csv","ViT-L/14")]

def agg(rows):
    f = lambda c: st.mean(float(x[c]) for x in rows)
    raw, iso, nul, h0 = f("raw"), f("null_iso"), f("null_act_raw"), f("h0_excess")
    return raw, iso, (nul + h0 - iso), (raw - nul - h0)      # 등방, 상속분, 잔존

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.05))

labs, inh, sur = [], [], []
for fn, lab in ARCH:
    r = list(csv.DictReader(open(R + fn)))
    raw, iso, i_, s_ = agg(r)
    labs.append(lab); inh.append(i_); sur.append(s_)
y = np.arange(3)
ax[0].barh(y, inh, color="#92C5DE", label="explained by activation geometry")
ax[0].barh(y, sur, left=inh, color="#2166AC", label="survives the null")
for k in range(3):
    ax[0].text(inh[k]/2, y[k], f"{100*inh[k]/(inh[k]+sur[k]):.0f}%", ha="center",
               va="center", fontsize=8, color="#08306B")
ax[0].set_yticks(y); ax[0].set_yticklabels(labs, fontsize=8.5)
ax[0].invert_yaxis()
ax[0].set_xlabel("above-chance overlap", labelpad=2)
ax[0].set_title("What the isotropic null counts as structure", fontsize=9.5)
ax[0].legend(fontsize=7.2, frameon=False, loc="upper center",
             bbox_to_anchor=(.5, -.30), ncol=2, handlelength=1.4)
ax[0].grid(axis="x", alpha=.25, lw=.5)

w = 0.36
for j, (grp, col, nm) in enumerate([(RES, "#4393C3", "reads the residual stream"),
                                    (INT, "#B2182B", "reads a block-internal representation")]):
    vals = []
    for fn, lab in ARCH:
        rows = [x for x in csv.DictReader(open(R + fn)) if x["module"].split(".")[-1] in grp]
        if not rows: vals.append(np.nan); continue
        raw, iso, i_, s_ = agg(rows)
        vals.append(100 * i_ / (i_ + s_))
    ax[1].bar(np.arange(3) + (j - .5) * w, vals, width=w, color=col, label=nm)
    for k, v in enumerate(vals):
        if v == v: ax[1].text(k + (j - .5) * w, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.6)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labs, fontsize=8.5)
ax[1].set_ylabel("% explained by activation geometry")
ax[1].set_ylim(0, 62)
ax[1].set_title("The split follows the module's role", fontsize=9.5)
ax[1].legend(fontsize=7.2, frameon=False, loc="upper center",
             bbox_to_anchor=(.5, -.30), ncol=1, handlelength=1.4)
ax[1].grid(axis="y", alpha=.25, lw=.5)
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=.6)
out = "figures/fig_main.pdf"
fig.savefig(out, bbox_inches="tight"); print(f"  wrote {out}")
print("  좌 (상속/잔존):", [(l, round(a,4), round(b,4)) for l,a,b in zip(labs,inh,sur)])
