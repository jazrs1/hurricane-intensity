import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PAPER = "#EDEDE7"
INK = "#22201C"
MUTED = "#6B6455"
RULE = "#B9B3A4"
# Darkened from #8A6A3C, which sat at 4.25:1 on the paper and missed the
# 4.5:1 floor for data marks.
AMBER = "#7D5F33"

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "heldout_predictions.csv"
OUT_DIR = ROOT / "site" / "figures"

df = pd.read_csv(CSV)
fix = df[~df["is_interpolated"]]
interp = df[df["is_interpolated"]]

plt.rcParams["font.family"] = "DejaVu Sans Mono"

# One PNG cannot serve both widths: type sized to stay legible at the 316px
# phone column renders oversized at the 560px desktop column. So the figure is
# drawn twice, each with type scaled for the width it is actually displayed at,
# and the page picks between them with <picture media>.
VARIANTS = [
    # name,               figsize, dpi, tick, label, legend, s_interp, s_fix, note
    ("heldout_scatter",       6.2, 200,  9.0,   9.5,    9.0,       13,    17, "desktop, shown at 560px"),
    ("heldout_scatter_narrow", 5.0, 126, 13.0,  13.5,   12.5,      22,    26, "phone, shown at ~316px"),
]


def render(name, figsize, dpi, fs_tick, fs_label, fs_legend, s_interp, s_fix, note):
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    lim = (15, 150)
    ax.plot(lim, lim, color=RULE, lw=1.0, zorder=1)

    # Names the diagonal on the figure itself, rotated onto the line -- the
    # axes are equal-aspect and square, so 45 degrees is exact. It rides the
    # stretch where the cloud has dropped below the line, and the paper-colored
    # backing keeps it legible over the few points it crosses.
    ax.text(
        100, 100, "perfect agreement",
        color=MUTED, fontsize=fs_tick, rotation=45,
        rotation_mode="anchor", ha="center", va="bottom", zorder=4,
        bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5),
    )

    # alpha dropped from 0.55 to 1.0: blended against the paper the hollow
    # points came out at 2.18:1, under the 4.5:1 floor.
    ax.scatter(interp["truth"], interp["pred"], s=s_interp, facecolors="none",
               edgecolors=MUTED, linewidths=0.8, zorder=2,
               label="interpolated label")
    ax.scatter(fix["truth"], fix["pred"], s=s_fix, color=AMBER, zorder=3,
               label="true best-track fix")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    ticks = [25, 50, 75, 100, 125, 150]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(colors=MUTED, labelsize=fs_tick, length=3, width=0.6)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.6)

    ax.set_xlabel("best-track wind  (kt)", color=MUTED, fontsize=fs_label, labelpad=10)
    ax.set_ylabel("predicted wind  (kt)", color=MUTED, fontsize=fs_label, labelpad=10)

    ax.grid(True, color=RULE, lw=0.35, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", frameon=False, fontsize=fs_legend,
              labelcolor=MUTED, handletextpad=0.5, borderpad=0)

    fig.tight_layout()
    out = OUT_DIR / (name + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out).size
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"wrote {shown}  {w}x{h}  ({note})")
    return w, h


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT_DIR = Path(sys.argv[1])
    for v in VARIANTS:
        render(*v)
