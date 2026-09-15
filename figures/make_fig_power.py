"""Rebuild fig_power.pdf from the values reported in Appendix Tables I and J."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


raw = np.array([0.027714916167, 0.141290035844, 0.262350785732, 0.505572302639])
initial_excess = np.array([0.000039816721, 0.051863494143, 0.116087059863, 0.247232327238])
corrected_excess = np.array([-0.000623322399, 0.112489008461, 0.232513248289, 0.474121472417])

delivered = raw - raw[0]
initial_response = initial_excess - initial_excess[0]
corrected_response = corrected_excess - corrected_excess[0]

ablation_labels = ["neither", "singleton\nflip", "full\ncomplement", "both"]
ablation_recovery = np.array([49.0024, 77.7309, 89.2171, 99.3040])
ablation_colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]

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

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.65), constrained_layout=True)

ax = axes[0]
limit = 0.5
ax.plot([0, limit], [0, limit], "--", color="black", linewidth=1.1, label="full recovery")
ax.plot(
    delivered,
    corrected_response,
    "o-",
    color="#2166ac",
    linewidth=1.8,
    markersize=4,
    label="corrected null",
)
ax.plot(
    delivered,
    initial_response,
    "s-",
    color="#b2182b",
    linewidth=1.8,
    markersize=4,
    label="null as first written",
)
ax.set_xlim(-0.01, limit)
ax.set_ylim(-0.01, limit)
ax.set_xlabel(r"delivered overlap $D_s$")
ax.set_ylabel(r"level-centred excess $X_s-X_0$")
ax.set_title("Response to planted directions")
ax.grid(alpha=0.22, linewidth=0.6)
ax.legend(frameon=False, loc="upper left")

ax = axes[1]
bars = ax.bar(np.arange(4), ablation_recovery, color=ablation_colors, width=0.72)
ax.set_xticks(np.arange(4), ablation_labels)
ax.set_ylim(0, 108)
ax.set_ylabel("delivered-normalized recovery (%)")
ax.set_title("Effect of each implementation fix")
ax.grid(axis="y", alpha=0.22, linewidth=0.6)
for bar, value in zip(bars, ablation_recovery):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.5,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

out = Path(__file__).with_name("fig_power.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
