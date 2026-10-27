"""Site v2 figures from the frozen ensemble's committed H9 predictions. No model inference.

Reads results/ensemble/expanded10_h9.csv (the ten-member ensemble on the nine held-out storms, H9) and writes:
  - site/figures/heldout_scatter.png and heldout_scatter_narrow.png, in make_scatter.py's style;
  - site/frames/ian.json, with each frame's "pred" replaced by the ensemble's. index, time, truth and
    is_interpolated are kept and must match the CSV, so the frame PNGs in site/frames/ian/ stay paired.
It also prints the figures quoted on the page (reports/site_v2/sources.md cites this output).
No 2025, East Pacific or 2026 data is read.
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "ensemble" / "expanded10_h9.csv"
OUT_DIR = ROOT / "site" / "figures"
IAN_JSON = ROOT / "site" / "frames" / "ian.json"
IAN_ATCF_ID = "AL092022"

# make_scatter.py's palette and type, unchanged.
PAPER = "#EDEDE7"
MUTED = "#6B6455"
RULE = "#B9B3A4"
AMBER = "#7D5F33"

VARIANTS = [
    # name,               figsize, dpi, tick, label, legend, s_interp, s_fix, diag, note
    ("heldout_scatter",       6.2, 200,  9.0,   9.5,    9.0,        9,    13, True, "desktop, shown at 560px"),
    ("heldout_scatter_narrow", 5.0, 126, 13.0,  13.5,   12.5,      15,    20, False, "phone, shown at ~316px"),
]


def render(df, name, figsize, dpi, fs_tick, fs_label, fs_legend, s_interp, s_fix, diag, note):
    fix, interp = df[~df["is_interpolated"]], df[df["is_interpolated"]]
    plt.rcParams["font.family"] = "DejaVu Sans Mono"
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    lim = (15, 150)
    ax.plot(lim, lim, color=RULE, lw=1.0, zorder=1)
    if diag:
        ax.text(138, 144, "perfect agreement", color=MUTED, fontsize=fs_tick, ha="right", va="center", zorder=4)
    ax.scatter(interp["truth"], interp["pred"], s=s_interp, facecolors="none", edgecolors=MUTED, linewidths=0.6,
               zorder=2, label="interpolated label")
    ax.scatter(fix["truth"], fix["pred"], s=s_fix, color=AMBER, zorder=3, label="true best-track fix")
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
    ax.legend(loc="upper left", frameon=False, fontsize=fs_legend, labelcolor=MUTED, handletextpad=0.5, borderpad=0)
    fig.tight_layout()
    out = OUT_DIR / (name + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    w, h = Image.open(out).size
    print(f"wrote {out.name}  {w}x{h}  ({note})")


def ian_json(df):
    ian = df[df["atcf_id"] == IAN_ATCF_ID].copy()
    ian["t"] = pd.to_datetime(ian["sample_time"], utc=True)
    ian = ian.sort_values("t", kind="stable").reset_index(drop=True)
    payload = json.loads(IAN_JSON.read_text(encoding="utf-8"))
    frames = payload["frames"]
    assert len(frames) == len(ian), f"ian.json has {len(frames)} frames, the CSV {len(ian)}"
    for f, (_, r) in zip(frames, ian.iterrows()):
        assert pd.Timestamp(f["time"]) == r["t"], f"time mismatch at frame {f['index']}"
        assert abs(f["truth"] - r["truth"]) < 0.051, f"truth mismatch at frame {f['index']}"
        assert f["is_interpolated"] == bool(r["is_interpolated"]), f"label kind mismatch at frame {f['index']}"
        f["pred"] = round(float(r["pred"]), 1)
    IAN_JSON.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {IAN_JSON.name}: {len(frames)} frames, pred from the ensemble")
    return ian


def main():
    global OUT_DIR
    if len(sys.argv) > 1:
        OUT_DIR = Path(sys.argv[1])
    df = pd.read_csv(CSV)
    assert len(df) == 1752 and df["pred"].notna().all()
    for v in VARIANTS:
        render(df, *v)
    ian = ian_json(df)

    # Figures quoted on the page.
    print(f"H9: {len(df)} frames, {df['atcf_id'].nunique()} storms, {int((~df['is_interpolated']).sum())} fixes")
    e = ian["pred"] - ian["truth"]
    peak = ian["truth"].idxmax()
    print(f"Ian: {len(ian)} frames, {ian['t'].iloc[0]:%d %b %Y %HZ} to {ian['t'].iloc[-1]:%d %b %Y %HZ}; "
          f"MAE {e.abs().mean():.2f}, bias {e.mean():+.2f}")
    print(f"Ian peak: best track {ian['truth'].max():.0f} kt at {ian['t'][peak]:%d %b %HZ}, ensemble there "
          f"{ian['pred'][peak]:.1f}; highest ensemble reading {ian['pred'].max():.1f}")
    rise, fall = ian.index < peak, ian.index > ian["truth"][::-1].idxmax()
    print(f"Ian before the first peak frame: MAE {e[rise].abs().mean():.2f}, bias {e[rise].mean():+.2f}; "
          f"after the last peak frame: MAE {e[fall].abs().mean():.2f}, bias {e[fall].mean():+.2f}")


if __name__ == "__main__":
    main()
