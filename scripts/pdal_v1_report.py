"""Tables, figures and results.json for scripts/pdal_v1.py (run as `pdal_v1.py report`).

Evaluation set: validation windows whose anchor t is a real best-track fix. A and B are reported as mean +- std
over seeds; C is deterministic; D only on the validation storms the ensemble never saw, so A/B/C are also reported
on that subset. Spearman: embedding distance vs label distance over all pairs of evaluation windows.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from hurricane_intensity.analog import CATEGORIES, metrics_by_category  # noqa: E402

from pdal_v1 import OUT  # noqa: E402

SEEDS = (0, 1, 2)
# dataviz reference palette (light): fixed categorical order A, B, C, D; text and surface tokens; one-hue blue ramp
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
NAMES = {"A": "A: PDAL + kNN", "B": "B: MSE regression", "C": "C: frozen ImageNet + kNN", "D": "D: ResNet-18 ensemble"}
TEXT, TEXT2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
KEYS = ("all", *CATEGORIES, ">=113")


def load(name):
    z = np.load(OUT / name / "result.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def summarize(runs, mask):
    """{key: {metric: (mean, std)}} over runs (a single run gives std 0)."""
    per = [metrics_by_category(r["val_pred"][mask], r["val_kt"][mask]) for r in runs]
    out = {}
    for key in KEYS:
        out[key] = {"n": per[0][key]["n"]}
        for m in ("mae", "rmse", "bias"):
            v = np.array([p[key][m] for p in per], float)
            out[key][m] = (float(np.nanmean(v)), float(np.nanstd(v)) if len(v) > 1 else 0.0)
    return out


def spearman(emb, kt):
    i, j = np.triu_indices(len(kt), k=1)
    d_emb = np.linalg.norm(emb[i] - emb[j], axis=1)
    d_lab = np.abs(kt[i] - kt[j])  # rank-equivalent to the z-scored label distance
    return float(spearmanr(d_emb, d_lab).statistic), len(i)


def fmt(mean_std, sign=False):
    m, s = mean_std
    if np.isnan(m):
        return "-"
    f = f"{m:+.1f}" if sign else f"{m:.1f}"
    return f"{f} ± {s:.1f}" if s else f


def table(summ, arms, title):
    lines = [f"**{title}**", "", "| Arm | n | MAE (kt) | RMSE (kt) | Bias (kt) | Bias ≥113 kt (n) |", "|---|---|---|---|---|---|"]
    for a in arms:
        s = summ[a]
        lines.append(f"| {NAMES[a]} | {s['all']['n']} | {fmt(s['all']['mae'])} | {fmt(s['all']['rmse'])} | "
                     f"{fmt(s['all']['bias'], True)} | {fmt(s['>=113']['bias'], True)} ({s['>=113']['n']}) |")
    return "\n".join(lines)


def cat_table(summ, arms, metric):
    lines = ["| Arm | " + " | ".join(f"{c} (n={summ[arms[0]][c]['n']})" for c in CATEGORIES) + " |",
             "|---|" + "---|" * len(CATEGORIES)]
    for a in arms:
        lines.append(f"| {NAMES[a]} | " + " | ".join(fmt(summ[a][c][metric], metric == "bias") for c in CATEGORIES) + " |")
    return "\n".join(lines)


def main():
    A = [load(f"A_s{s}") for s in SEEDS]
    B = [load(f"B_s{s}") for s in SEEDS]
    C, D = load("C"), load("D")
    ref = A[0]
    for r in (*A[1:], *B, C, D):
        assert (r["val_storms"] == ref["val_storms"]).all() and (r["val_time"] == ref["val_time"]).all()
    fix = ref["val_fix"]
    unseen = np.isin(ref["val_storms"], D["unseen"])
    runs = {"A": A, "B": B, "C": [C]}
    summ = {a: summarize(r, fix) for a, r in runs.items()}
    summ_u = {a: summarize(r, fix & unseen) for a, r in runs.items()}
    summ_u["D"] = summarize([D], fix & unseen)
    basin = np.array([s[:2] for s in ref["val_storms"]])
    summ_basin = {b: {a: summarize(r, fix & unseen & (basin == b)) for a, r in {**runs, "D": [D]}.items()}
                  for b in ("AL", "EP")}
    rho = {"A": [spearman(r["val_emb"][fix], r["val_kt"][fix]) for r in A],
           "C": [spearman(C["val_emb"][fix], C["val_kt"][fix])]}
    ks = {"A": [int(r["k"]) for r in A], "C": int(C["k"])}
    k_maes = {"A": [json.loads(str(r["k_maes"])) for r in A], "C": json.loads(str(C["k_maes"]))}
    train_s = {"A": [float(r["train_s"]) for r in A], "B": [float(r["train_s"]) for r in B]}
    best_epoch = {"A": [int(r["best_epoch"]) for r in A], "B": [int(r["best_epoch"]) for r in B]}

    md = [table(summ, ["A", "B", "C"], f"Validation 2023, windows starting at a best-track fix "
                                         f"({int(fix.sum())} windows, {len(set(ref['val_storms'][fix]))} storms)"),
          "", "Bias by category (kt):", "", cat_table(summ, ["A", "B", "C"], "bias"),
          "", "MAE by category (kt):", "", cat_table(summ, ["A", "B", "C"], "mae"), "",
          table(summ_u, ["A", "B", "C", "D"], f"Same, storms the ResNet-18 ensemble never saw "
                                               f"({int((fix & unseen).sum())} windows, {len(D['unseen'])} storms)"), "",
          "Bias by category, unseen storms (kt):", "", cat_table(summ_u, ["A", "B", "C", "D"], "bias")]
    for b in ("AL", "EP"):
        md += ["", table(summ_basin[b], ["A", "B", "C", "D"], f"Unseen storms, basin {b}")]
    md += ["", "Spearman(embedding distance, label distance), evaluation pairs: "
           f"A {np.mean([r for r, _ in rho['A']]):.3f} ± {np.std([r for r, _ in rho['A']]):.3f}, "
           f"C {rho['C'][0][0]:.3f} ({rho['C'][0][1]} pairs)",
           f"Chosen k: A seeds {ks['A']}, C {ks['C']}. Best epochs: A {best_epoch['A']}, B {best_epoch['B']}.",
           f"Training time: A {sum(train_s['A']) / 60:.1f} min, B {sum(train_s['B']) / 60:.1f} min "
           f"(total {(sum(train_s['A']) + sum(train_s['B'])) / 3600:.2f} h)."]
    (OUT / "results_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump({"summary": summ, "summary_unseen": summ_u, "summary_unseen_by_basin": summ_basin,
               "spearman": rho, "k": ks, "k_maes": k_maes, "train_s": train_s, "best_epoch": best_epoch},
              open(OUT / "results.json", "w"), indent=1)
    print("\n".join(md))
    figures(A, B, C, D, fix, unseen, summ, summ_u)


def figures(A, B, C, D, fix, unseen, summ, summ_u):
    # 1. predicted vs true, one panel per arm (A and B: seed 0; D: unseen storms only)
    panels = [("A", A[0], fix), ("B", B[0], fix), ("C", C, fix), ("D", D, fix & unseen)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor=SURFACE)
    for ax, (a, r, m) in zip(axes, panels):
        style(ax)
        t, p = r["val_kt"][m], r["val_pred"][m]
        ax.plot([10, 170], [10, 170], color=TEXT2, linewidth=1, linestyle="--")
        ax.axvline(113, color=GRID, linewidth=1)
        ax.scatter(t, p, s=10, color=COLORS[a], alpha=0.6, edgecolors="none")
        mae = np.abs(p - t).mean()
        ax.set_title(f"{NAMES[a]}\nMAE {mae:.1f} kt, n={m.sum()}" + (" (seed 0)" if a in "AB" else "")
                     + (" (unseen storms)" if a == "D" else ""), fontsize=9, color=TEXT)
        ax.set_xlim(10, 170)
        ax.set_ylim(10, 170)
        ax.set_xlabel("Best-track intensity at t (kt)", fontsize=8, color=TEXT2)
    axes[0].set_ylabel("Predicted (kt)", fontsize=8, color=TEXT2)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_pred_vs_true.png", dpi=150)
    plt.close(fig)

    # 2. bias by category, grouped bars (arms in fixed order), seed std as error bars
    for name, s, arms in (("bias_by_category.png", summ, ["A", "B", "C"]),
                          ("bias_by_category_unseen.png", summ_u, ["A", "B", "C", "D"])):
        fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
        style(ax)
        x = np.arange(len(CATEGORIES))
        w = 0.8 / len(arms)
        for i, a in enumerate(arms):
            m = [s[a][c]["bias"][0] for c in CATEGORIES]
            e = [s[a][c]["bias"][1] for c in CATEGORIES]
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, m, w * 0.92, yerr=e, color=COLORS[a], label=NAMES[a],
                   error_kw={"ecolor": TEXT2, "elinewidth": 0.8, "capsize": 2})
        ax.axhline(0, color=TEXT2, linewidth=0.8)
        ax.set_xticks(x, [f"{c}\n(n={s[arms[0]][c]['n']})" for c in CATEGORIES])
        ax.set_ylabel("Bias, predicted − true (kt)", fontsize=8, color=TEXT2)
        ax.set_title("Bias by Saffir–Simpson category of the true intensity (validation 2023, fix windows"
                     + (", unseen storms)" if "unseen" in name else ")"), fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)

    # 3. PCA of validation embeddings, colored by intensity (one-hue sequential)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (a, r) in zip(axes, (("A", A[0]), ("C", C))):
        style(ax)
        e = r["val_emb"][fix] - r["val_emb"][fix].mean(0)
        u, sv, vt = np.linalg.svd(e, full_matrices=False)
        xy = e @ vt[:2].T
        var = sv[:2] ** 2 / (sv ** 2).sum()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=r["val_kt"][fix], cmap=BLUES, s=12, vmin=20, vmax=160,
                        edgecolors="none")
        ax.set_title(f"{NAMES[a]}" + (" (seed 0)" if a == "A" else ""), fontsize=9, color=TEXT)
        ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)", fontsize=8, color=TEXT2)
        ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=8, color=TEXT2)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label("Best-track intensity at t (kt)", fontsize=8, color=TEXT2)
    cb.ax.tick_params(colors=TEXT2, labelsize=8)
    fig.savefig(OUT / "pca_embeddings.png", dpi=150)
    plt.close(fig)
    print(f"figures in {OUT}")
