#!/usr/bin/env python
"""Save one StormWindowDataset window as a PNG: the T frames side by side, intensity in each title.

    .venv\\Scripts\\python.exe scripts\\preview_windows.py --data_dir data\\storm_windows --storm_id AL142018

Frames are shown exactly as the model receives them (clipped to [tb_min, tb_max], scaled to 0-1, gray
colormap), so cold cloud tops are dark, as in the README figures. Titles give the frame time, intensity and
whether it is a best-track fix or interpolated. Default output: data/exports/previews/ (gitignored).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hurricane_intensity.config import PROJECT_ROOT  # noqa: E402
from hurricane_intensity.storm_windows import StormWindowDataset, delta_24h  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--storm_id", required=True)
    p.add_argument("--window", type=int, default=None,
                   help="window number within the storm (default: the one with the largest |dI24|)")
    p.add_argument("--T", type=int, default=9)
    p.add_argument("--step_hours", type=float, default=3)
    p.add_argument("--out", type=Path, default=None, help="PNG path (default data/exports/previews/<ID>_<start>.png)")
    args = p.parse_args(argv)

    ds = StormWindowDataset(args.data_dir, [args.storm_id], T=args.T, step_hours=args.step_hours)
    if len(ds) == 0:
        raise SystemExit(f"{args.storm_id}: no valid windows ({ds.n_skipped} skipped)")
    if args.window is None:
        reaches_24h = (24 / args.step_hours).is_integer() and 24 / args.step_hours < args.T
        idx = max(range(len(ds)), key=lambda i: abs(float(delta_24h(ds[i]["intensities"], args.step_hours)))
                  ) if reaches_24h else 0
    else:
        idx = args.window
    item = ds[idx]
    rows = ds.windows[idx][2]
    times = pd.to_datetime(ds.meta[args.storm_id]["time"][rows], utc=True)

    T = item["frames"].shape[0]
    fig, axes = plt.subplots(1, T, figsize=(2.0 * T, 2.6), squeeze=False)
    for k, ax in enumerate(axes[0]):
        ax.imshow(item["frames"][k, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        kind = "fix" if item["is_fix"][k] else "interp."
        ax.set_title(f"{times[k]:%m-%d %H:%M}Z\n{item['intensities'][k]:.0f} kt ({kind})", fontsize=9)
        ax.set_axis_off()
    head = f"{args.storm_id}, window {idx} of {len(ds)}, {args.step_hours:g} h steps"
    if (24 / args.step_hours).is_integer() and 24 / args.step_hours < T:
        head += f", ΔI24 = {float(delta_24h(item['intensities'], args.step_hours)):+.0f} kt"
    fig.suptitle(head + f"   (BT {ds.tb_min:.0f}–{ds.tb_max:.0f} K, cold = dark)", fontsize=10)
    fig.tight_layout()

    out = args.out or (PROJECT_ROOT / "data" / "exports" / "previews"
                       / f"{args.storm_id}_{times[0]:%Y%m%dT%H%MZ}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(out.resolve())


if __name__ == "__main__":
    main()
