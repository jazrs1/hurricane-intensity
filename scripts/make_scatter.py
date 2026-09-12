import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PAPER = "#EDEDE7"
INK = "#22201C"
MUTED = "#6B6455"
RULE = "#B9B3A4"
AMBER = "#8A6A3C"

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "heldout_predictions.csv"
OUT = ROOT / "site" / "figures" / "heldout_scatter.png"

df = pd.read_csv(CSV)
fix = df[~df["is_interpolated"]]
interp = df[df["is_interpolated"]]

plt.rcParams["font.family"] = "DejaVu Sans Mono"

fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=200)
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)

lim = (15, 150)
ax.plot(lim, lim, color=RULE, lw=1.0, zorder=1)

ax.scatter(interp["truth"], interp["pred"], s=13, facecolors="none",
           edgecolors=MUTED, linewidths=0.7, alpha=0.55, zorder=2,
           label="interpolated label")
ax.scatter(fix["truth"], fix["pred"], s=17, color=AMBER, zorder=3,
           label="true best-track fix")

ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")

ticks = [25, 50, 75, 100, 125, 150]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.6)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(RULE)
    ax.spines[side].set_linewidth(0.6)

ax.set_xlabel("best-track wind  (kt)", color=MUTED, fontsize=9.5, labelpad=10)
ax.set_ylabel("predicted wind  (kt)", color=MUTED, fontsize=9.5, labelpad=10)

ax.grid(True, color=RULE, lw=0.35, alpha=0.45, zorder=0)
ax.set_axisbelow(True)

leg = ax.legend(loc="upper left", frameon=False, fontsize=9,
                labelcolor=MUTED, handletextpad=0.5, borderpad=0)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
print(f"wrote {OUT.relative_to(ROOT)}")
