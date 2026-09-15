"""Rebuild fig_main.pdf from the aggregate values reported in Appendix Table K."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


models = ["ViT-B/16", "ViT-B/32", "ViT-L/14"]
raw = np.array([0.104573648206, 0.095312649061, 0.071401547342])
isotropic = np.array([0.009114583333, 0.009114583333, 0.007812500000])
exact_null = np.array([0.033308794828, 0.028419573063, 0.015377134196])
above_iso = raw - isotropic
reproduced = exact_null - isotropic
survives = above_iso - reproduced
role_labels = ["residual-reading", "block-internal"]
role_reproduced = np.array([16.5568, 38.9609])

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.15), constrained_layout=True)

ax = axes[0]
y = np.arange(len(models))
ax.barh(y, reproduced, color="#92c5de", label="null-reproduced")
ax.barh(y, survives, left=reproduced, color="#2166ac", label="residual")
ax.set_yticks(y, models)
ax.invert_yaxis()
# Headroom so the legend clears the longest bar.
ax.set_xlim(0, 0.118)
ax.set_xlabel("above-isotropic overlap")
ax.set_title("Conditional decomposition")
ax.grid(axis="x", alpha=0.22, linewidth=0.6)
ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=7)

# Keep each share label inside its own segment, and push it out to the right when the
# segment is too narrow to hold it.  ViT-L/14 is narrow enough that a centred label
# would run off the axis and collide with the tick label.
fig.canvas.draw()
for idx, frac in enumerate(100 * reproduced / above_iso):
    label = ax.text(
        reproduced[idx] / 2, idx, f"{frac:.1f}%", ha="center", va="center", fontsize=8
    )
    text_w = label.get_window_extent(fig.canvas.get_renderer()).width
    bar_w = ax.transData.transform((reproduced[idx], 0))[0] - ax.transData.transform((0, 0))[0]
    if text_w > 0.88 * bar_w:
        # Sits just past the boundary, on the residual segment, so it stays next to the
        # share it labels.  White, because the residual colour is dark.
        label.set_position((reproduced[idx] + 0.0022, idx))
        label.set_ha("left")
        label.set_color("white")

ax = axes[1]
x = np.arange(len(role_labels))
bars = ax.bar(x, role_reproduced, 0.56, color=["#4393c3", "#b2182b"])
for bar, value in zip(bars, role_reproduced):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, role_labels)
ax.set_ylim(0, 50)
ax.set_ylabel("null-reproduced (%)")
ax.set_title("ViT-B/16 module grouping")
ax.grid(axis="y", alpha=0.22, linewidth=0.6)

fig.savefig(Path(__file__).with_name("fig_main.pdf"), bbox_inches="tight")
plt.close(fig)
