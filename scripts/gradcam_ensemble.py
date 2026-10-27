"""Grad-CAM for the frozen 10-member ensemble, by the method in reports/site_v2/gradcam_ensemble_method.md.

Two parts, run in order; each needs every argument explicitly:

  control   best_model.pt with today's code: the four peak frames against results/gradcam/<stem>.npy and the site
            PNGs, and the 454-frame >= 96 kt survey against the committed CSVs and +0.574891. Exits 1 on a miss.
  ensemble  only after a passing control: the ten members' renormalized mean CAM over the same 454 frames and the
            four peak frames (arrays and overlays).

    python scripts/gradcam_ensemble.py control --main-root <checkout with models and COORDINATOR.md> --processed-root <data/processed>

Every model and the processed crops are read only. Everything is written under reports/site_v2/gradcam_ensemble/.
The CAM, the coldness correlation, the IoU and the overlay are scripts/gradcam.py's own functions, unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gradcam = _load("gradcam", REPO_ROOT / "scripts" / "gradcam.py")  # also puts this checkout's src on sys.path
check = _load("check", REPO_ROOT / "scripts" / "check.py")
from hurricane_intensity.dataset import TCDataset  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "site_v2" / "gradcam_ensemble"
ORIG_DIR = REPO_ROOT / "results" / "gradcam"
SITE_FIGURES = REPO_ROOT / "site" / "figures"
MEMBERS = [f"models/storms/expanded/seed{s}.pt" for s in range(10)]
MEMBERS_SHA = "results/ensemble/members_sha256.txt"

# C1: one explicit list, never the --survey default or storms_gradcam.txt.
STORMS = ["AL092022", "AL022024", "AL132023", "AL182021"]  # Ian, Beryl, Lee, Sam
MIN_KT, MAX_KT = 96, 999
N_FRAMES = 454
ORIG_CSVS = ["convection_alignment_survey_96_999.csv", "convection_alignment_survey_96_999_gradcam_storms.csv"]
ORIG_MEAN_R = 0.5748907350256608  # mean pearson over the 454 valid rows of ORIG_CSVS
NPY_TOL = 1e-3  # L1(i)
MEAN_TOL = 5e-4

# (site figure, storm, timestamp, original stem)
PEAK_FRAMES = [
    ("ian_140kt", "AL092022", "2022-09-28T12:00", "AL092022_20220928T1200Z"),
    ("beryl_145kt", "AL022024", "2024-07-02T09:45", "AL022024_20240702T0945Z"),
    ("lee_145kt", "AL132023", "2023-09-08T06:00", "AL132023_20230908T0600Z"),
    ("sam_135kt", "AL182021", "2021-09-26T18:00", "AL182021_20210926T1800Z"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_best_model(main_root):
    expected = dict(check.load_expected(main_root / "COORDINATOR.md"))
    got = sha256(main_root / "best_model.pt")
    if got != expected["best_model.pt"]:
        raise SystemExit(f"best_model.pt is {got}, expected {expected['best_model.pt']}")
    return main_root / "best_model.pt"


def verify_members(main_root):
    """The ten members against results/ensemble/members_sha256.txt, line for line (as ensemble_eval.py --expect-sha)."""
    expected = [l for l in (REPO_ROOT / MEMBERS_SHA).read_text().splitlines() if l.strip()]
    got = [f"{sha256(main_root / m)} *{m}" for m in MEMBERS]
    if got != expected:
        raise SystemExit(f"members do not match {MEMBERS_SHA}")
    return [main_root / m for m in MEMBERS]


def load_resnet(path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def renormalize(cam):
    """gradcam.compute_gradcam's own per-frame step: shift the minimum to 0, then divide by the peak if positive."""
    cam = cam - cam.min()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam.astype(np.float32)


def ensemble_cam(models, image, device):
    """Mean of the members' normalized CAMs, renormalized (decision 1), and the members' mean prediction x 185."""
    cam = renormalize(np.mean([gradcam.compute_gradcam(m, image, device) for m in models], axis=0))
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        pred_kt = float(np.mean([m(x).item() for m in models])) * 185.0
    return cam, pred_kt


def gray_of(image):
    return np.clip(image[0].numpy() * 255.0, 0, 255).astype(np.uint8)


def original_frames():
    orig = pd.concat([pd.read_csv(ORIG_DIR / f) for f in ORIG_CSVS]).dropna(subset=["pearson_r_cam_vs_cold"])
    orig["time"] = pd.to_datetime(orig["timestamp"], utc=True)
    assert len(orig) == N_FRAMES, len(orig)
    assert abs(orig["pearson_r_cam_vs_cold"].mean() - ORIG_MEAN_R) < 1e-12
    return orig.set_index(["storm_id", "time"])["pearson_r_cam_vs_cold"]


def survey(cam_fn, processed_root):
    """convection_alignment over the >= 96 kt frames of STORMS, after asserting they are the original 454."""
    ds = TCDataset(str(processed_root), STORMS)
    idxs = np.flatnonzero(((ds.df["wind_kt"] >= MIN_KT) & (ds.df["wind_kt"] < MAX_KT)).to_numpy())
    keys = set(zip(ds.df["atcf_id"].iloc[idxs], pd.to_datetime(ds.df["sample_time"].iloc[idxs], utc=True)))
    orig = original_frames()
    assert len(idxs) == N_FRAMES, f"{len(idxs)} frames, expected {N_FRAMES}"
    assert keys == set(orig.index), "the (storm, time) set differs from the original CSVs' valid rows"
    rows = []
    for n, i in enumerate(idxs):
        image, _ = ds[int(i)]
        row = ds.df.iloc[int(i)]
        cam, pred_kt = cam_fn(image)
        rows.append(dict(gradcam.convection_alignment(cam, gray_of(image)),
                         storm_id=row["atcf_id"], storm_name=row["storm_name"], timestamp=str(row["sample_time"]),
                         wind_kt=float(row["wind_kt"]), pred_kt=pred_kt, is_interpolated=bool(row["is_interpolated"])))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(idxs)}", flush=True)
    df = pd.DataFrame(rows)
    df["orig_pearson"] = [orig[(s, pd.Timestamp(t).tz_convert("UTC"))] for s, t in zip(df["storm_id"], df["timestamp"])]
    return df


def summarize(df):
    r = df["pearson_r_cam_vs_cold"]
    iou = df["iou_vs_coldest"] - df["iou_expected_by_chance"]
    return dict(n=int(len(df)), n_valid=int(r.notna().sum()), mean_r=float(r.mean()), median_r=float(r.median()),
                frac_r_negative=float((r < 0).mean()), iou_minus_chance_mean=float(iou.mean()),
                frac_iou_below_chance=float((iou < 0).mean()),
                high_cam_share_median=float((df["high_cam_px"] / gradcam.FRAME_SIZE ** 2).median()),
                peak_to_coldest_px_median=float(df["peak_to_coldest_dist_px"].median()),
                per_storm={s: dict(n=int(len(g)), mean_r=float(g["pearson_r_cam_vs_cold"].mean()),
                                   median_r=float(g["pearson_r_cam_vs_cold"].median()))
                           for s, g in df.groupby("storm_id")})


def peak_frame(processed_root, storm_id, ts):
    ds = TCDataset(str(processed_root), [storm_id])
    idx = gradcam.resolve_frame(ds, storm_id, ts)
    image, _ = ds[idx]
    return image, ds.df.iloc[idx]


def run_control(args, device):
    model = load_resnet(verify_best_model(args.main_root), device)

    def cam_fn(image):
        cam = gradcam.compute_gradcam(model, image, device)
        with torch.no_grad():
            pred_kt = model(image.unsqueeze(0).to(device)).item() * 185.0
        return cam, pred_kt

    peaks = []
    for fig, storm_id, ts, stem in PEAK_FRAMES:
        image, _ = peak_frame(args.processed_root, storm_id, ts)
        cam, _ = cam_fn(image)
        orig = np.load(ORIG_DIR / f"{stem}.npy")
        png = np.asarray(gradcam.render_threshold_image(gray_of(image), cam))
        site = np.asarray(Image.open(SITE_FIGURES / f"{fig}.png").convert("RGB"))
        peaks.append(dict(stem=stem, npy_max_abs_diff=float(np.abs(cam - orig).max()),
                          npy_bit_identical=bool(np.array_equal(cam, orig)),
                          png_identical_to_site=bool(png.shape == site.shape and (png == site).all()),
                          png_max_abs_diff=int(np.abs(png.astype(int) - site.astype(int)).max())))
    df = survey(cam_fn, args.processed_root)
    s = summarize(df)
    s.update(orig_mean_r=ORIG_MEAN_R, mean_r_diff=s["mean_r"] - ORIG_MEAN_R,
             per_frame_r_max_abs_diff=float((df["pearson_r_cam_vs_cold"] - df["orig_pearson"]).abs().max()),
             peaks=peaks)
    s["passed"] = bool(all(p["npy_max_abs_diff"] <= NPY_TOL for p in peaks) and abs(s["mean_r_diff"]) <= MEAN_TOL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "control_best_model_survey_96_999.csv", index=False)
    (OUT_DIR / "control_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s[k] for k in ("passed", "mean_r", "mean_r_diff", "per_frame_r_max_abs_diff")}))
    for p in peaks:
        print(p)
    if not s["passed"]:
        raise SystemExit("control did not reproduce: stop, no ensemble run")


def run_ensemble(args, device):
    control = json.loads((OUT_DIR / "control_summary.json").read_text())
    if not control.get("passed"):
        raise SystemExit("no passing control in control_summary.json: run the control first")
    models = [load_resnet(p, device) for p in verify_members(args.main_root)]
    for fig, storm_id, ts, stem in PEAK_FRAMES:
        image, row = peak_frame(args.processed_root, storm_id, ts)
        cam, pred_kt = ensemble_cam(models, image, device)
        np.save(OUT_DIR / f"{stem}.npy", cam)
        gradcam.render_threshold_image(gray_of(image), cam).save(OUT_DIR / f"ensemble_{fig}.png")
        a = gradcam.convection_alignment(cam, gray_of(image))
        print(f"{stem}: truth {float(row['wind_kt']):.1f} kt  ensemble pred {pred_kt:.1f} kt  "
              f"pearson {a['pearson_r_cam_vs_cold']:+.3f}")
    df = survey(lambda image: ensemble_cam(models, image, device), args.processed_root)
    s = summarize(df)
    s.update(orig_mean_r=ORIG_MEAN_R, mean_r_minus_orig=s["mean_r"] - ORIG_MEAN_R)
    df.to_csv(OUT_DIR / "ensemble_survey_96_999.csv", index=False)
    (OUT_DIR / "ensemble_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s[k] for k in ("n", "mean_r", "median_r", "mean_r_minus_orig", "frac_r_negative")}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=["control", "ensemble"])
    p.add_argument("--main-root", type=Path, required=True, help="checkout holding best_model.pt, models/ and COORDINATOR.md")
    p.add_argument("--processed-root", type=Path, required=True, help="data/processed, read only")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    (run_control if args.part == "control" else run_ensemble)(args, device)


if __name__ == "__main__":
    main()
