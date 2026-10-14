"""Stage-2 analysis: how does everything move as K changes?

Inputs : experiments/2026-10-12_selector_zoo/outputs/*_per_fold_stage2.csv
Outputs: results/selector_zoo_k/
           delta_by_k.csv        paired  selector-vs-random  deltas per K
           selector_curve_k.csv  marginal mean metric per (selector, K)
           model_curve_k.csv     marginal mean metric per (model, K)
           fig1_k_curve_clf.png  K vs accuracy, one line per selector
           fig2_k_curve_reg.png  K vs R2,       one line per selector
           fig3_delta_heatmap.png  selector x K  paired delta (with p-value marks)
           README.md

Pairing
-------
The noise floor (`random`) is paired with every other selector on the SAME
(model, fold) cell, so a delta is "selector minus random, holding model and
fold fixed" -- this removes the model-to-model spread and isolates the effect
of the feature set itself. Wilcoxon is one-sided (selector > random).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"
RES = ROOT / "results" / "selector_zoo_k"

METRIC = {"clf": "accuracy", "reg": "r2"}
METRIC_LABEL = {"clf": "Accuracy", "reg": "R^2"}

# order selectors by their mean rank so the legend reads sensibly
def _order(df, metric):
    return (df.groupby("selector")[metric].mean()
            .sort_values(ascending=False).index.tolist())


def paired_vs_random(df, metric, k):
    d = df[df["k"] == k]
    piv = d.pivot_table(index=["model", "fold"], columns="selector",
                        values=metric)
    if "random" not in piv.columns:
        return pd.DataFrame()
    rows = []
    for s in piv.columns:
        if s == "random":
            continue
        diff = (piv[s] - piv["random"]).dropna()
        if len(diff) < 5:
            continue
        try:
            stat, p = wilcoxon(diff, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({"k": k, "selector": s, "n_pairs": len(diff),
                     "delta_mean": float(diff.mean()),
                     "delta_median": float(diff.median()),
                     "sel_mean": float(piv[s].mean()),
                     "random_mean": float(piv["random"].mean()),
                     "p_greater": float(p)})
    return pd.DataFrame(rows)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def analyse(task):
    metric = METRIC[task]
    df = pd.read_csv(OUT / f"{task}_per_fold_stage2.csv", encoding="utf-8-sig")
    ks = sorted(df["k"].unique())

    dk = pd.concat([paired_vs_random(df, metric, k) for k in ks], ignore_index=True)
    dk["signif"] = dk["p_greater"].map(stars)

    sel_curve = (df.groupby(["selector", "k"], as_index=False)[metric]
                 .mean().rename(columns={metric: f"{metric}_mean"}))
    mod_curve = (df.groupby(["model", "k"], as_index=False)[metric]
                 .mean().rename(columns={metric: f"{metric}_mean"}))

    dk.to_csv(RES / f"delta_by_k_{task}.csv", index=False, encoding="utf-8-sig")
    sel_curve.to_csv(RES / f"selector_curve_k_{task}.csv", index=False,
                     encoding="utf-8-sig")
    mod_curve.to_csv(RES / f"model_curve_k_{task}.csv", index=False,
                     encoding="utf-8-sig")

    # ---- figure: K vs metric, one line per selector -----------------------
    order = _order(df, metric)
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=140)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(order):
        sub = sel_curve[sel_curve.selector == s].sort_values("k")
        is_ctrl = s == "random"
        ax.plot(sub["k"], sub[f"{metric}_mean"],
                marker="o", lw=3.0 if is_ctrl else 1.6,
                ls="--" if is_ctrl else "-",
                color="black" if is_ctrl else cmap(i % 20),
                zorder=5 if is_ctrl else 3,
                label=f"{s} (noise floor)" if is_ctrl else s)
    ax.set_xlabel("K = number of selected features")
    ax.set_ylabel(f"mean {METRIC_LABEL[task]} over folds x models")
    ax.set_title(f"[{task}] effect of feature-set size K  (1 seed, random folds)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(RES / f"fig{1 if task == 'clf' else 2}_k_curve_{task}.png")
    plt.close(fig)

    # ---- figure: delta heatmap -------------------------------------------
    piv = dk.pivot(index="selector", columns="k", values="delta_mean")
    piv = piv.reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(piv) + 2.0), dpi=140)
    vmax = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"K={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_title(f"[{task}] paired Δ vs random  ({METRIC_LABEL[task]})\n"
                 f"* p<.05  ** p<.01  *** p<.001  (one-sided Wilcoxon)")
    for i, s in enumerate(piv.index):
        for j, k in enumerate(piv.columns):
            v = piv.iloc[i, j]
            if not np.isfinite(v):
                continue
            p = dk[(dk.selector == s) & (dk.k == k)]["p_greater"]
            pv = float(p.iloc[0]) if len(p) else np.nan
            ax.text(j, i, f"{v:+.3f}\n{stars(pv)}", ha="center", va="center",
                    fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ")
    fig.tight_layout()
    fig.savefig(RES / f"fig3_delta_heatmap_{task}.png")
    plt.close(fig)

    return dk, sel_curve, mod_curve, order


def main():
    RES.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage-2 — K sensitivity of the selector zoo\n",
             "Protocol: full selector x model grid, **random folds**, "
             "**1 seed (42)**, K in {5, 10, 20, 30}. Paired deltas are "
             "selector-minus-`random` on the same (model, fold) cell, "
             "one-sided Wilcoxon.\n"]
    for task in ("clf", "reg"):
        dk, sel_curve, mod_curve, order = analyse(task)
        metric = METRIC[task]
        print(f"\n===== {task} : paired Δ vs random (metric={metric}) =====")
        for k in sorted(dk["k"].unique()):
            sub = dk[dk.k == k].sort_values("delta_mean", ascending=False)
            print(f"\n-- K={k} --")
            print(sub[["selector", "sel_mean", "random_mean", "delta_mean",
                       "p_greater", "signif"]].round(4).to_string(index=False))
        print(f"\n-- {task} | model marginal mean {metric} by K --")
        print(mod_curve.pivot(index="model", columns="k",
                              values=f"{metric}_mean").round(4).to_string())
        lines.append(f"\n## {task}\n")
        lines.append(f"![K curve {task}](fig{1 if task == 'clf' else 2}"
                     f"_k_curve_{task}.png)\n")
        lines.append(f"![delta heatmap {task}](fig3_delta_heatmap_{task}.png)\n")
    # NOTE: README.md is hand-written (it carries the interpretation, not just
    # the numbers). Only auto-generate it on a fresh directory so a re-run of
    # this script cannot overwrite the notes.
    readme = RES / "README.md"
    if not readme.exists():
        readme.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] wrote {readme}")
    else:
        print(f"[skip] {readme} exists -> hand-written notes preserved")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
