"""Read-only: does run order / GPU thermal state affect the baseline's held-out MAE? CPU only."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
LOG = REPO / "logs" / "run_order"
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")

tele = pd.read_csv(LOG / "gpu_telemetry.csv", skipinitialspace=True)
tele.columns = [c.strip() for c in tele.columns]
tele["t"] = pd.to_datetime(tele["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
num = {"temp": "temperature.gpu", "sm_mhz": "clocks.current.sm [MHz]", "gr_mhz": "clocks.current.graphics [MHz]",
       "mem_mhz": "clocks.current.memory [MHz]", "power_w": "power.draw [W]", "util": "utilization.gpu [%]", "fan": "fan.speed [%]"}
for k, c in num.items():
    tele[k] = pd.to_numeric(tele[c].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
flag_cols = [c for c in tele.columns if c.startswith("clocks_event_reasons") and c != "clocks_event_reasons.active"]
print(f"telemetry: {len(tele)} samples, {tele.t.min():%H:%M:%S} to {tele.t.max():%H:%M:%S}; temp {tele.temp.min():.0f}-{tele.temp.max():.0f} C")
for c in flag_cols:
    print(f"  {c}: {dict(tele[c].astype(str).str.strip().value_counts())}")
print("  clocks_event_reasons.active values:", dict(tele["clocks_event_reasons.active"].astype(str).str.strip().value_counts()))


def counter(line, name):
    m = re.search(name + r" : (\d+) us", line)
    return int(m.group(1)) if m else np.nan


rows = []
for f in sorted(LOG.glob("*_seed?.log")):
    txt = f.read_text()
    head = re.search(r"RUN block=(\w+) seed=(\d+) order=(\d+)", txt)
    start = pd.Timestamp(re.search(r"START (\S+)", txt).group(1)).tz_localize(None)
    end = pd.Timestamp(re.search(r"END (\S+)", txt).group(1)).tz_localize(None)
    cs, ce = re.search(r"COUNTERS_START (.*)", txt).group(1), re.search(r"COUNTERS_END (.*)", txt).group(1)
    ep = [(int(e), float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt)]
    ep20 = [r for r in ep if r[0] <= 20]
    sel = min(ep20, key=lambda r: r[2])
    w = tele[(tele.t >= start) & (tele.t <= end)]
    load = w[w.util >= 20]
    r = dict(block=head.group(1), seed=int(head.group(2)), order=int(head.group(3)), start=start.strftime("%H:%M:%S"),
             dur_s=(end - start).total_seconds(), epochs=len(ep),
             temp_start=w.temp.iloc[0], temp_end=w.temp.iloc[-1], temp_peak=w.temp.max(), temp_mean=w.temp.mean(),
             sm_mhz_load_mean=load.sm_mhz.mean(), sm_mhz_load_min=load.sm_mhz.min(), power_w_load=load.power_w.mean(),
             util_mean=w.util.mean(),
             power_cap_s=(counter(ce, "SW Power Capping") - counter(cs, "SW Power Capping")) / 1e6,
             sw_thermal_s=(counter(ce, "SW Thermal Slowdown") - counter(cs, "SW Thermal Slowdown")) / 1e6,
             hw_thermal_s=(counter(ce, "HW Thermal Slowdown") - counter(cs, "HW Thermal Slowdown")) / 1e6,
             sel_epoch=sel[0], val_mae=sel[2], train_final=ep20[-1][1], failed="TRAIN_FAILED" in txt)
    ev = LOG / f"{f.stem}_eval.log"
    if ev.exists():
        m = re.search(r"overall MAE:\s+([\d.]+)", ev.read_text())
        r["heldout_mae"] = float(m.group(1)) if m else np.nan
        pred = pd.read_csv(REPO / "results" / "run_order" / f"{f.stem}.csv")
        r["heldout_bias"] = pred.signed_error.mean()
        r["bias_gt96"] = pred.signed_error[pred.truth > 96].mean()
    rows.append(r)
runs = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)

# the 60-epoch runs: what a 20-epoch budget would have selected, from the per-epoch trace
tr60 = REPO / "results" / "epoch_traces_60" / "epoch_trace.csv"
if tr60.exists():
    t = pd.read_csv(tr60)
    for s, d in t[t.epoch <= 20].groupby("seed"):
        sel = d.loc[d.val_mae.idxmin()]
        m = (runs.block == "long60") & (runs.seed == s)
        runs.loc[m, ["heldout_mae", "heldout_bias", "bias_gt96"]] = [sel.heldout_mae, sel.heldout_bias, sel.heldout_bias_gt96]

print("\n== every run of this session, in order ==")
print(runs.drop(columns=["failed"]).to_string(index=False))
print("failed runs:", int(runs.failed.sum()))
if (LOG / "idle.log").exists():
    print("\nidle periods:\n" + (LOG / "idle.log").read_text().strip())
runs.to_csv(REPO / "results" / "run_order" / "runs.csv", index=False)

sp, bb = runs[runs.block == "spaced"].set_index("seed"), runs[runs.block == "b2b"].set_index("seed")
print("\n== spaced vs back-to-back (same seeds 0-4) ==")
cmp_cols = ["heldout_mae", "val_mae", "sel_epoch", "bias_gt96", "temp_start", "temp_peak", "temp_mean", "sm_mhz_load_mean", "power_w_load", "power_cap_s", "dur_s"]
summary = pd.DataFrame({"spaced mean": sp[cmp_cols].mean(), "spaced sd": sp[cmp_cols].std(), "b2b mean": bb[cmp_cols].mean(), "b2b sd": bb[cmp_cols].std()})
summary["b2b - spaced"] = summary["b2b mean"] - summary["spaced mean"]
print(summary.to_string())
print("\nheld-out MAE by seed:")
print(pd.DataFrame({"spaced": sp.heldout_mae, "b2b": bb.heldout_mae, "b2b - spaced": bb.heldout_mae - sp.heldout_mae}).to_string())
a, b = sp.heldout_mae.to_numpy(), bb.heldout_mae.to_numpy()
obs = b.mean() - a.mean()
pool = np.concatenate([a, b])
from itertools import combinations
diffs = np.array([pool[list(c)].mean() - np.delete(pool, list(c)).mean() for c in combinations(range(10), 5)])
print(f"\nb2b - spaced = {obs:+.2f} kt; exact permutation test over all 252 splits: two-sided p = {(np.abs(diffs) >= abs(obs) - 1e-12).mean():.3f}")
print(f"pooled sd of these ten runs: {np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2):.2f} kt; a 5-vs-5 difference needs to be about {2 * np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) * np.sqrt(2 / 5):.2f} kt to stand out")

print("\n== does anything logged predict held-out MAE? the ten 20-epoch runs of this session (spearman) ==")
ten = runs[runs.block.isin(["spaced", "b2b"])]
for c in ["order", "temp_start", "temp_peak", "temp_mean", "sm_mhz_load_mean", "power_w_load", "power_cap_s", "dur_s", "val_mae", "sel_epoch"]:
    print(f"  {c:18s} {ten[c].corr(ten.heldout_mae, method='spearman'):+.2f}")

print("\n== all baseline reruns of the day in chronological order ==")
prev = pd.read_csv(REPO / "results" / "epoch_traces" / "baseline_reruns.csv")
prev = prev[prev.run != "original 9.00 run"].reset_index(drop=True)
prev["when"] = ["20:04", "20:05", "20:06", "20:08", "20:30", "20:31", "20:32", "20:36", "20:37", "20:38"]
allr = pd.concat([prev[["run", "when", "heldout_mae", "val_mae", "selected_epoch"]],
                  runs.assign(run=runs.block + " seed " + runs.seed.astype(str), when=runs.start.str[:5], selected_epoch=runs.sel_epoch)[["run", "when", "heldout_mae", "val_mae", "selected_epoch"]]],
                 ignore_index=True)
allr["chron"] = np.arange(1, len(allr) + 1)
print(allr.to_string(index=False))
ok = allr.dropna(subset=["heldout_mae"])
print(f"\n{len(ok)} runs: held-out mean {ok.heldout_mae.mean():.2f}, sd {ok.heldout_mae.std():.2f}, min {ok.heldout_mae.min():.2f}, max {ok.heldout_mae.max():.2f}")
rho = ok.chron.corr(ok.heldout_mae, method="spearman")
rng = np.random.default_rng(0)
perm = np.array([pd.Series(rng.permutation(ok.heldout_mae.values)).corr(ok.chron.reset_index(drop=True), method="spearman") for _ in range(20000)])
print(f"held-out MAE vs chronological position: spearman {rho:+.2f}, permutation two-sided p = {(np.abs(perm) >= abs(rho)).mean():.3f}")
later = ok[ok.chron > 4]
print(f"first four of the day: {ok[ok.chron <= 4].heldout_mae.mean():.2f}; all {len(later)} later runs: mean {later.heldout_mae.mean():.2f}, sd {later.heldout_mae.std():.2f}, "
      f"later runs at or above 9.94 (the best of the first four): {int((later.heldout_mae >= 9.94).sum())}")
allr.to_csv(REPO / "results" / "run_order" / "all_baseline_reruns.csv", index=False)
