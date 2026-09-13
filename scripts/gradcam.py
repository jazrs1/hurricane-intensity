"""Grad-CAM attribution against the trained ResNet-18 in best_model.pt.

The model has a single regression output (predicted wind speed), so there
is no class logit to pick: Grad-CAM here backpropagates the scalar
prediction itself through layer4's activations.

For a given storm id and a list of UTC timestamps, writes per frame:
  results/gradcam/<storm_id>_<timestamp>.npy   -- raw CAM, normalized to [0, 1]
  results/gradcam/<storm_id>_<timestamp>.png   -- plasma threshold-wash overlay
and appends, per frame, to:
  results/gradcam/radial_profile.csv         -- CAM mass by concentric annulus from crop center
  results/gradcam/convection_alignment.csv   -- does the CAM track the coldest brightness temps?

Usage:
    python scripts/gradcam.py AL142018 2018-10-07T12:00 2018-10-10T01:00
    python scripts/gradcam.py --survey 0 64    # distribution over all held-out frames in [0, 64) kt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms.functional import gaussian_blur
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# TCDataset reads excluded_crops.txt via a CWD-relative path (see the same
# note in scripts/export_scrubber.py), so chdir to the repo root to pick up
# the same exclusions evaluate.py saw. Every other path here is absolute
# and __file__-derived regardless of CWD.
os.chdir(REPO_ROOT)

from hurricane_intensity.dataset import TCDataset  # noqa: E402

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
MODEL_PATH = REPO_ROOT / "best_model.pt"
OUT_DIR = REPO_ROOT / "results" / "gradcam"
FRAME_SIZE = 224

# Site tokens (site/index.html), reused here only for visual consistency --
# no site file is read or written.
PAPER = (0xED, 0xED, 0xE7)
INK = (0x22, 0x20, 0x1C)
RULE = (0xB6, 0xB2, 0xA5)

# A flat mid-gray stand-in for "typical frame content", used only in the
# legend strip so a swatch shows how a given alpha actually reads -- the
# real overlay always blends against the true grayscale frame.
SWATCH_BG = 150


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _build_lut(stops):
    """(256, 3) float32 LUT, linearly interpolating `stops` (RGB 0-255
    tuples) evenly spaced across [0, 1]."""
    stops = np.array(stops, dtype=np.float32)
    xs_stops = np.linspace(0.0, 1.0, len(stops))
    xs_lut = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xs_lut, xs_stops, stops[:, c]) for c in range(3)], axis=-1)


# Default and only scheme: plasma. Hand-built from its published reference
# stops (matplotlib/viscm are not a project dependency), so this is an
# approximate reproduction, not an exact copy of matplotlib's LUT. Chosen
# over the single-hue rust/blue options and cividis after a side-by-side
# comparison.
_PLASMA_STOPS = [_hex(h) for h in (
    "0d0887", "41049d", "6a00a8", "8f0da4", "b12a90",
    "cc4778", "e16462", "f2844b", "fca636", "fcce25", "f0f921",
)]
PLASMA_LUT = _build_lut(_PLASMA_STOPS)
PLASMA_TITLE = "plasma (approx.)"

# --- Option A: discrete contour bands -----------------------------------
# Fixed CAM levels with fixed alpha per band (not a continuous ramp), plus
# a thin ink line drawn at every band boundary -- so adjacent levels read
# as visibly different steps instead of blurring into "some brown".
BAND_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BAND_ALPHAS = (0.0, 0.22, 0.42, 0.62, 0.85)  # one per band; band 0 is fully transparent
BAND_MIDS = tuple((BAND_EDGES[i] + BAND_EDGES[i + 1]) / 2 for i in range(len(BAND_ALPHAS)))

# --- Option B: threshold wash --------------------------------------------
# Below THRESHOLD the wash is off entirely (plain grayscale). Above it,
# alpha ramps continuously from ALPHA_AT_THRESHOLD to MAX_ALPHA, and hue
# (for the multi-hue schemes) follows the colormap at the CAM's own value.
THRESHOLD = 0.5
ALPHA_AT_THRESHOLD = 0.22
MAX_ALPHA = 0.85

# The raw CAM is Grad-CAM on a 7x7 layer4 feature map, bilinearly upsampled
# to 224x224 -- the hard right angles at band edges are an artifact of that
# upsampling, not signal. A Gaussian blur sized to one feature-map cell
# (224 / 7 = 32px) removes the blockiness without adding any detail finer
# than the method actually resolves (blurring can only remove high-frequency
# content, never invent it). Kernel spans 4*sigma = 32px.
CAM_BLUR_KERNEL = 33  # px, odd, ~= one feature-map cell
CAM_BLUR_SIGMA = 8.0

# Concentric annuli (px from the crop's geometric center, which is the
# storm's best-track center by construction -- see src/hurricane_intensity
# /crop.py) used to profile CAM mass by radius. 112px is the inscribed-
# circle radius; the last bin (112-158) is corners only.
RADIAL_EDGES = (0, 14, 28, 42, 56, 70, 84, 98, 112, 158)


def load_model(device):
    """Construct the model exactly as scripts/evaluate.py does."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def compute_gradcam(model, input_tensor, device):
    """Grad-CAM for a single-output regression head, against layer4.

    input_tensor: (3, 224, 224) tensor, already preprocessed by TCDataset.
    Returns a (224, 224) float32 numpy array normalized to [0, 1].
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = model.layer4.register_forward_hook(forward_hook)
    bh = model.layer4.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        x = input_tensor.unsqueeze(0).to(device)
        output = model(x)             # (1, 1) -- single regression output
        score = output.squeeze()      # backprop the prediction itself, not a class logit
        score.backward()

        A = activations["value"]      # (1, C, H, W)
        grad = gradients["value"]     # (1, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)          # spatially averaged gradient per channel
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))   # weighted sum over channels, then ReLU
        cam = F.interpolate(cam, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False)
        cam = gaussian_blur(cam, kernel_size=CAM_BLUR_KERNEL, sigma=CAM_BLUR_SIGMA)
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
    finally:
        fh.remove()
        bh.remove()

    cam -= cam.min()
    peak = cam.max()
    if peak > 0:
        cam /= peak
    return cam


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _text(draw, xy, text, anchor):
    draw.text(xy, text, fill=INK, font=_font(), anchor=anchor)


def _stack(image, strip):
    canvas = Image.new("RGB", (image.width, image.height + strip.height), PAPER)
    canvas.paste(image, (0, 0))
    canvas.paste(strip, (0, image.height))
    ImageDraw.Draw(canvas).line(
        [(0, image.height), (image.width, image.height)], fill=RULE, width=1
    )
    return canvas


SWATCH_TOP = 4
SWATCH_H = 14
TICK_LABEL_Y = SWATCH_TOP + SWATCH_H + 5
TITLE_Y = TICK_LABEL_Y + 12


def _lut_color(lut, t):
    return tuple(lut[int(np.clip(round(t * 255), 0, 255))])


def _band_strip(width, lut, title, height=46):
    strip = Image.new("RGB", (width, height), PAPER)
    n = len(BAND_ALPHAS)
    swatch_w = width // n
    for i, alpha in enumerate(BAND_ALPHAS):
        hue = _lut_color(lut, BAND_MIDS[i])
        color = tuple(SWATCH_BG * (1 - alpha) + c * alpha for c in hue)
        x0 = i * swatch_w
        x1 = width if i == n - 1 else x0 + swatch_w
        ImageDraw.Draw(strip).rectangle(
            [x0, SWATCH_TOP, x1 - 1, SWATCH_TOP + SWATCH_H], fill=tuple(int(round(c)) for c in color)
        )
    draw = ImageDraw.Draw(strip)
    for i, x in enumerate(range(0, width + 1, swatch_w)):
        if i > n:
            break
        label = f"{BAND_EDGES[i]:.1f}" if i < len(BAND_EDGES) else ""
        x_clamped = min(x, width - 1)
        draw.line([(x_clamped, SWATCH_TOP), (x_clamped, SWATCH_TOP + SWATCH_H)], fill=INK, width=1)
        _text(draw, (x_clamped, TICK_LABEL_Y), label, anchor="ma")
    _text(draw, (width / 2, TITLE_Y), f"{title} -- discrete bands", anchor="ma")
    return strip


def _threshold_strip(width, lut, title, height=46):
    strip = Image.new("RGB", (width, height), PAPER)
    bg = np.full((SWATCH_H, width), float(SWATCH_BG), dtype=np.float32)
    bg = np.stack([bg] * 3, axis=-1)
    xs = np.linspace(0.0, 1.0, width)
    alpha = np.zeros_like(xs)
    above = xs >= THRESHOLD
    span = max(1e-6, 1.0 - THRESHOLD)
    alpha[above] = ALPHA_AT_THRESHOLD + (xs[above] - THRESHOLD) / span * (MAX_ALPHA - ALPHA_AT_THRESHOLD)
    hue = lut[np.clip((xs * 255).astype(int), 0, 255)]
    row = bg * (1 - alpha[:, None]) + hue[None, :, :] * alpha[:, None]
    strip.paste(Image.fromarray(np.clip(row, 0, 255).astype(np.uint8), mode="RGB"), (0, SWATCH_TOP))

    draw = ImageDraw.Draw(strip)
    thresh_x = int(round(THRESHOLD * (width - 1)))
    for x, label, anchor in ((0, "0", "la"), (thresh_x, f"{THRESHOLD:.1f}", "ma"), (width - 1, "1.0", "ra")):
        draw.line([(x, SWATCH_TOP), (x, SWATCH_TOP + SWATCH_H)], fill=INK, width=1)
        _text(draw, (x, TICK_LABEL_Y), label, anchor=anchor)
    _text(draw, (width / 2, TITLE_Y), f"{title} -- threshold wash, off below {THRESHOLD:.1f}", anchor="ma")
    return strip


def render_bands(gray_u8, cam, lut=PLASMA_LUT, title=PLASMA_TITLE):
    """Discrete contour bands: each pixel's CAM value is bucketed into a
    fixed band with a fixed alpha (band 0 fully transparent) and a hue
    taken from the colormap at that band's midpoint. A thin ink line is
    drawn at every band boundary -- like isolines on a contour plot -- so
    levels stay visually separable instead of blending together.
    """
    base = np.stack([gray_u8] * 3, axis=-1).astype(np.float32)
    band_idx = np.digitize(cam, BAND_EDGES[1:-1])  # 0 .. len(BAND_ALPHAS)-1
    alpha = np.asarray(BAND_ALPHAS, dtype=np.float32)[band_idx]
    band_colors = lut[np.clip((np.array(BAND_MIDS) * 255).astype(int), 0, 255)]  # (5, 3)
    color = band_colors[band_idx]  # (H, W, 3)
    blended = base * (1 - alpha[..., None]) + color * alpha[..., None]

    boundary = np.zeros(cam.shape, dtype=bool)
    boundary[:, :-1] |= band_idx[:, :-1] != band_idx[:, 1:]
    boundary[:-1, :] |= band_idx[:-1, :] != band_idx[1:, :]
    blended[boundary] = INK

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    return _stack(image, _band_strip(gray_u8.shape[1], lut, title))


def render_threshold_image(gray_u8, cam, lut=PLASMA_LUT):
    """Threshold wash, image only (no legend strip): no wash at all below
    THRESHOLD, so unattended regions stay plainly gray; above it, alpha
    ramps continuously and hue follows the colormap at the CAM's own
    value. Returns a bare 224x224 PIL RGB image."""
    base = np.stack([gray_u8] * 3, axis=-1).astype(np.float32)
    alpha = np.zeros_like(cam, dtype=np.float32)
    above = cam >= THRESHOLD
    span = max(1e-6, 1.0 - THRESHOLD)
    alpha[above] = ALPHA_AT_THRESHOLD + (cam[above] - THRESHOLD) / span * (MAX_ALPHA - ALPHA_AT_THRESHOLD)
    color = lut[np.clip((cam * 255).astype(int), 0, 255)]
    blended = base * (1 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def render_threshold(gray_u8, cam, lut=PLASMA_LUT, title=PLASMA_TITLE):
    """Threshold wash with the legend strip stacked beneath it (the
    standard per-frame CLI output)."""
    image = render_threshold_image(gray_u8, cam, lut)
    return _stack(image, _threshold_strip(gray_u8.shape[1], lut, title))


def radial_profile(cam):
    """Mean CAM value and share of total CAM mass in concentric annuli
    from the crop's geometric (= storm-track) center. Distinguishes a
    hollow ring (mass peaking at a nonzero radius) from a filled disk
    (mass highest at r=0 and falling off outward)."""
    n = cam.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = (n - 1) / 2, (n - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = cam.sum()

    rows = []
    for lo, hi in zip(RADIAL_EDGES[:-1], RADIAL_EDGES[1:]):
        mask = (r >= lo) & (r < hi)
        if not mask.any():
            continue
        rows.append({
            "r_lo_px": lo,
            "r_hi_px": hi,
            "mean_cam": float(cam[mask].mean()),
            "mass_frac": float(cam[mask].sum() / total) if total > 0 else 0.0,
            "n_px": int(mask.sum()),
        })
    return rows


def convection_alignment(cam, gray_u8, threshold=THRESHOLD):
    """Does the CAM track the coldest brightness temperatures in the crop?

    gray_u8 channel 0 is the normalized BT field (higher = warmer), so
    coldness = 1 - normalized BT. Reports:
      - pearson_r: linear correlation between CAM and coldness, over all
        pixels (negative => CAM instead tracks warmth, e.g. a warm eye).
      - iou_vs_coldest: overlap between the pixels the threshold render
        would highlight (cam >= threshold) and the same number of coldest
        pixels in the frame. ~0 => no relation; ~1 => same region.
      - iou_expected_by_chance: the IoU two random same-size regions of
        this frame would show, for comparison against iou_vs_coldest.
      - peak_to_coldest_dist_px: distance between the CAM's peak pixel and
        the single coldest pixel in the frame.
    """
    bt_norm = gray_u8.astype(np.float32) / 255.0
    cold = 1.0 - bt_norm
    n_total = cam.size

    cam_flat = cam.ravel()
    cold_flat = cold.ravel()
    if cam_flat.std() == 0 or cold_flat.std() == 0:
        pearson_r = float("nan")  # degenerate frame (e.g. an all-zero CAM)
    else:
        pearson_r = float(np.corrcoef(cam_flat, cold_flat)[0, 1])

    k = int((cam_flat >= threshold).sum())
    if k == 0:
        iou = None
        iou_chance = None
    else:
        high_idx = set(np.flatnonzero(cam_flat >= threshold).tolist())
        cold_top_idx = set(np.argsort(cold_flat)[::-1][:k].tolist())
        inter = len(high_idx & cold_top_idx)
        union = len(high_idx | cold_top_idx)
        iou = inter / union if union else None
        expected_inter = k * k / n_total
        expected_union = 2 * k - expected_inter
        iou_chance = expected_inter / expected_union if expected_union else None

    cam_peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    coldest_px = np.unravel_index(int(np.argmax(cold)), cold.shape)
    peak_to_coldest_dist_px = float(np.hypot(cam_peak[0] - coldest_px[0], cam_peak[1] - coldest_px[1]))

    return {
        "pearson_r_cam_vs_cold": pearson_r,
        "high_cam_px": k,
        "iou_vs_coldest": iou,
        "iou_expected_by_chance": iou_chance,
        "cam_peak_row": cam_peak[0],
        "cam_peak_col": cam_peak[1],
        "coldest_row": coldest_px[0],
        "coldest_col": coldest_px[1],
        "peak_to_coldest_dist_px": peak_to_coldest_dist_px,
    }


def _save_csv(rows_df, name, key_col="stem"):
    csv_path = OUT_DIR / name
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[~existing[key_col].isin(rows_df[key_col].unique())]
        combined = pd.concat([existing, rows_df], ignore_index=True)
    else:
        combined = rows_df
    sort_cols = [c for c in ("storm_id", "timestamp", "r_lo_px") if c in combined.columns]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    return csv_path


def resolve_frame(ds, storm_id, timestamp):
    times = pd.to_datetime(ds.df["sample_time"], utc=True)
    target = pd.Timestamp(timestamp)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    matches = np.flatnonzero((times == target).to_numpy())
    if len(matches) == 0:
        raise SystemExit(
            f"no frame for {storm_id} at {timestamp} "
            "(check the timestamp, or it may have been dropped by excluded_crops.txt / a missing crop file)"
        )
    if len(matches) > 1:
        raise SystemExit(f"ambiguous timestamp {timestamp} for {storm_id}: {len(matches)} matches")
    return int(matches[0])


# Same three storms scripts/evaluate.py holds out -- kept as a separate
# constant here (not imported) since evaluate.py isn't a module, only a
# script; if that list changes there, change it here too. This is the
# --survey default, but --storms/--storms-file can point it at any other
# list (e.g. storms_gradcam.txt) -- those are attribution test subjects
# only, never held-out evaluation data, and must not be reported as such.
HOLDOUT_STORMS = ["AL092022", "AL172022", "AL032021"]


def load_storm_list(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.upper())
    return ids


def run_alignment_survey(model, device, min_kt, max_kt, storms=HOLDOUT_STORMS):
    """convection_alignment() over every frame of the given storms with
    min_kt <= wind_kt < max_kt, both true fixes and interpolated hours.
    No images are rendered or saved.
    """
    ds = TCDataset(str(PROCESSED_ROOT), storms)
    mask = (ds.df["wind_kt"] >= min_kt) & (ds.df["wind_kt"] < max_kt)
    idxs = np.flatnonzero(mask.to_numpy())

    rows = []
    for i in idxs:
        image, _label = ds[int(i)]
        row = ds.df.iloc[int(i)]
        cam = compute_gradcam(model, image, device)
        gray = np.clip(image[0].numpy() * 255.0, 0, 255).astype(np.uint8)
        align = convection_alignment(cam, gray)
        rows.append(dict(
            align,
            storm_id=row["atcf_id"], storm_name=row["storm_name"],
            timestamp=str(row["sample_time"]), wind_kt=float(row["wind_kt"]),
            is_interpolated=bool(row["is_interpolated"]),
        ))
    return pd.DataFrame(rows)


def _describe(series, label):
    s = series.dropna()
    dropped = len(series) - len(s)
    note = f"  ({dropped} dropped as degenerate/undefined)" if dropped else ""
    print(f"  {label}: n={len(s)}{note}")
    if len(s) == 0:
        return
    q = s.quantile([0.25, 0.5, 0.75])
    print(
        f"    mean={s.mean():+.3f}  std={s.std():.3f}  "
        f"min={s.min():+.3f}  p25={q[0.25]:+.3f}  median={q[0.5]:+.3f}  p75={q[0.75]:+.3f}  max={s.max():+.3f}"
    )


def _survey_block(df, label):
    print(f"  frames: {len(df)}")
    print(f"  true fixes: {(~df['is_interpolated']).sum()}   interpolated: {df['is_interpolated'].sum()}")

    _describe(df["pearson_r_cam_vs_cold"], "pearson(CAM, coldness)")
    frac_neg = (df["pearson_r_cam_vs_cold"] < 0).mean()
    print(f"    fraction with pearson < 0: {frac_neg:.3f}")

    _describe(df["iou_vs_coldest"], "IoU(high-CAM, coldest same-size)")
    valid = df.dropna(subset=["iou_vs_coldest", "iou_expected_by_chance"])
    if len(valid):
        below_chance = (valid["iou_vs_coldest"] < valid["iou_expected_by_chance"]).mean()
        excess = valid["iou_vs_coldest"] - valid["iou_expected_by_chance"]
        print(f"    fraction with IoU below its own chance level: {below_chance:.3f}")
        _describe(excess, "IoU minus its own chance level")

    _describe(df["peak_to_coldest_dist_px"], "peak-to-coldest-pixel distance (px)")


def print_survey_summary(df, min_kt, max_kt):
    print(f"\n=== convection alignment survey: {min_kt:g} <= wind_kt < {max_kt:g} ===")
    print(f"storms: {sorted(df['storm_name'].unique().tolist())}")

    print(f"\n--- pooled, n={len(df)} ({df['storm_name'].value_counts().to_dict()}) ---")
    _survey_block(df, "pooled")

    for storm_name, group in df.groupby("storm_name"):
        print(f"\n--- {storm_name} only ---")
        _survey_block(group, storm_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("storm_id", nargs="?", help="ATCF id, e.g. AL142018")
    parser.add_argument("timestamps", nargs="*", help="UTC timestamps, e.g. 2018-10-07T12:00")
    parser.add_argument(
        "--survey", nargs=2, type=float, metavar=("MIN_KT", "MAX_KT"),
        help="skip single-frame rendering; instead run convection_alignment over every frame of "
             "--storms-file (default: HOLDOUT_STORMS), true fixes and interpolated both, with "
             "wind_kt in [MIN_KT, MAX_KT), and report the distribution. No PNG/NPY output.",
    )
    parser.add_argument(
        "--storms-file", default=None,
        help="--survey only: text file of ATCF ids (one per line, '#' comments allowed), e.g. "
             "storms_gradcam.txt, to survey instead of HOLDOUT_STORMS.",
    )
    parser.add_argument(
        "--out-name", default=None,
        help="--survey only: override the output CSV filename stem (default: "
             "convection_alignment_survey_<min>_<max>).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if args.survey:
        min_kt, max_kt = args.survey
        storms = load_storm_list(args.storms_file) if args.storms_file else HOLDOUT_STORMS
        df = run_alignment_survey(model, device, min_kt, max_kt, storms)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = args.out_name or f"convection_alignment_survey_{min_kt:g}_{max_kt:g}"
        csv_path = OUT_DIR / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        print_survey_summary(df, min_kt, max_kt)
        print(f"\nwrote {csv_path}")
        return

    if not args.storm_id or not args.timestamps:
        parser.error("storm_id and at least one timestamp are required unless --survey is given")

    ds = TCDataset(str(PROCESSED_ROOT), [args.storm_id])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    radial_rows = []
    alignment_rows = []

    for ts in args.timestamps:
        idx = resolve_frame(ds, args.storm_id, ts)
        image, _label = ds[idx]
        row = ds.df.iloc[idx]

        cam = compute_gradcam(model, image, device)

        with torch.no_grad():
            pred_kt = model(image.unsqueeze(0).to(device)).item() * 185.0

        # Channel 0 of the post-transform tensor is the normalized BT field
        # in [0, 1] -- same source export_scrubber.py renders from.
        gray = np.clip(image[0].numpy() * 255.0, 0, 255).astype(np.uint8)

        stem = f"{args.storm_id}_{pd.Timestamp(row['sample_time']).strftime('%Y%m%dT%H%MZ')}"
        np.save(OUT_DIR / f"{stem}.npy", cam)
        render_threshold(gray, cam).save(OUT_DIR / f"{stem}.png")

        truth_kt = float(row["wind_kt"])
        common = dict(stem=stem, storm_id=args.storm_id, timestamp=str(row["sample_time"]),
                      truth_kt=truth_kt, pred_kt=pred_kt)

        profile = radial_profile(cam)
        for r in profile:
            radial_rows.append(dict(r, **common))

        align = convection_alignment(cam, gray)
        alignment_rows.append(dict(align, **common))

        print(f"{stem}: truth {truth_kt:6.1f} kt  pred {pred_kt:6.1f} kt  cam mean {cam.mean():.3f}")
        print(f"  -> {stem}.npy / {stem}.png")
        print("  radial profile (r_lo-r_hi px | mean cam | mass frac):")
        for r in profile:
            print(f"    {r['r_lo_px']:3d}-{r['r_hi_px']:3d}  {r['mean_cam']:.3f}  {r['mass_frac']:.3f}")
        iou_str = f"{align['iou_vs_coldest']:.3f}" if align['iou_vs_coldest'] is not None else "n/a"
        chance_str = f"{align['iou_expected_by_chance']:.3f}" if align['iou_expected_by_chance'] is not None else "n/a"
        print(
            f"  convection alignment: pearson(cam, coldness)={align['pearson_r_cam_vs_cold']:+.3f}  "
            f"IoU(high-cam, coldest same-size)={iou_str} (chance={chance_str})  "
            f"peak-to-coldest dist={align['peak_to_coldest_dist_px']:.1f}px"
        )

    radial_csv = _save_csv(pd.DataFrame(radial_rows), "radial_profile.csv")
    align_csv = _save_csv(pd.DataFrame(alignment_rows), "convection_alignment.csv")
    print(f"\nwrote {radial_csv}\nwrote {align_csv}")


if __name__ == "__main__":
    main()
