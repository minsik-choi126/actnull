"""Figure 1: 심은 구조를 널이 얼마나 회수하는가. 논문의 중심 결과."""
import os
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.environ.get("ACTNULL_RESULTS", "results") + "/"
sh = json.load(open(R + "power_shipped.json"))["rows"]
cu = json.load(open(R + "power_current.json"))["rows"]
lo = json.load(open(R + "power_lora_current.json"))["rows"]
ab = json.load(open(R + "ablation.json"))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.75))
x = [r["planted"] / 8 for r in cu]
ideal = [r["truth"] for r in cu]
ax[0].plot(x, ideal, "k--", lw=1.1, label="perfect recovery", zorder=1)
ax[0].plot(x, [r["excess"] for r in cu], "o-", color="#2166AC", lw=1.8, ms=5,
           label="corrected null", zorder=3)
ax[0].plot(x, [r["excess"] for r in sh], "s-", color="#B2182B", lw=1.8, ms=5,
           label="null as first written", zorder=2)
ax[0].set_xlabel("fraction of read directions planted, $s/k$")
ax[0].set_ylabel("measured excess")
ax[0].set_title("Recovery of planted structure", fontsize=9.5)
ax[0].legend(fontsize=7.4, frameon=False, loc="upper left")
ax[0].grid(alpha=.25, lw=.5)

O0 = cu[0]["raw"]; tgt = (2 / 8) * (1 - O0)
base = next(r["excess"] for r in ab if "shipped" in r["variant"])
labs = ["neither", "singleton\nflip only", "full complement\nonly", "both"]
keys = ["shipped (둘 다 꺼짐)", "싱글턴 부호만", "여집합 전체만", "corrected (둘 다)"]
vals = [100 * next(r["excess"] for r in ab if r["variant"] == k) / tgt for k in keys]
cols = ["#B2182B", "#D6604D", "#4393C3", "#2166AC"]
ax[1].bar(range(4), vals, color=cols, width=.62)
for i, v in enumerate(vals):
    ax[1].text(i, v + 1.6, f"{v:.0f}%", ha="center", fontsize=8)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labs, fontsize=7.4)
ax[1].set_ylabel("recovery at $s/k = 1/4$")
ax[1].set_ylim(0, 108)
ax[1].set_title("Which correction did the work", fontsize=9.5)
ax[1].grid(axis="y", alpha=.25, lw=.5)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=.6)
out = "figures/fig_power.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"  wrote {out}")
print(f"  좌: 이상 {ideal}  교정 {[round(r['excess'],4) for r in cu]}")
print(f"  우: {[round(v,1) for v in vals]}")
