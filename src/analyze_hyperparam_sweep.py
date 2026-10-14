"""Analyse the hyper-parameter sweep.

Produces (in results/hyperparam_sweep/):
  * clf_tuning_table.csv / reg_tuning_table.csv
      per model: default | grid-mean | grid-median | grid-best | LOFO-honest
      plus the best parameter setting and the two gains.
  * fig1_param_curves_{clf,reg}.png
      small multiples: metric vs config_id for every model, default in red,
      best in green.
  * fig2_tuning_gain_{clf,reg}.png
      bar chart: default vs grid-mean vs LOFO-honest vs grid-best.
  * recommended_params.json
      the LOFO-honest best setting per model (the defensible recommendation).

Why LOFO (leave-one-fold-out) matters
-------------------------------------
"grid-best" is picked on the SAME 5 folds it is scored on -> winner's curse.
LOFO removes that: for each fold f, choose the best config on the other 4
folds, then score it on f. Averaging the 5 honest scores gives an unbiased
estimate of "what do I actually get if I tune on 4 folds and deploy on a 5th".
The gap  grid-best - LOFO  is exactly how much a naive best-config pick lies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "2026-10-13_hyperparam_sweep"
OUT = EXP / "outputs"
RES = ROOT / "results" / "hyperparam_sweep"
RES.mkdir(parents=True, exist_ok=True)

TASK_METRIC = {"clf": "roc_auc", "reg": "r2"}

ORDER_CLF = ["RF", "ET", "KNN", "SVM", "NB", "LDA", "SVMLin", "LR", "PLSDA", "XGB"]
ORDER_REG = ["KNN", "ET", "RF", "GPR", "XGB", "PLSR", "Ridge", "ENet", "SVR"]


def _ordered(models, task):
    pref = ORDER_CLF if task == "clf" else ORDER_REG
    return [m for m in pref if m in models] + [m for m in models if m not in pref]


def lofo_scores(per, metric):
    """Leave-one-fold-out honest tuning score, per model."""
    out = {}
    for model, g in per.groupby("model"):
        folds = sorted(g.fold.unique())
        honest = []
        picks = []
        for f in folds:
            tr = g[g.fold != f]
            cf = tr.groupby("config_id")[metric].mean().idxmax()
            picks.append(cf)
            honest.append(float(g[(g.fold == f) & (g.config_id == cf)][metric].iloc[0]))
        out[model] = {"lofo": float(np.mean(honest)), "fold_scores": honest,
                      "picks": [int(p) for p in picks],
                      "pick_stable": len(set(picks)) == 1}
    return out


def build_table(summ, per, task):
    metric = TASK_METRIC[task]
    col = f"{metric}_mean"                  # summary stores *_mean / *_std
    lofo = lofo_scores(per, metric)
    rows = []
    for model, g in summ.groupby("model"):
        g = g.sort_values(col, ascending=False).reset_index(drop=True)
        default = float(g.loc[g.config_id == 0, col].iloc[0])
        rows.append({
            "model": model,
            "default": round(default, 4),
            "grid_mean": round(float(g[col].mean()), 4),
            "grid_median": round(float(g[col].median()), 4),
            "grid_worst": round(float(g[col].min()), 4),
            "grid_best": round(float(g[col].iloc[0]), 4),
            "best_params": g.params.iloc[0],
            "gain_vs_default": round(float(g[col].iloc[0]) - default, 4),
            "lofo_honest": round(lofo[model]["lofo"], 4),
            "lofo_gain_vs_default": round(lofo[model]["lofo"] - default, 4),
            "lofo_pick_stable": lofo[model]["pick_stable"],
            "lofo_picks": str(lofo[model]["picks"]),
        })
    t = pd.DataFrame(rows)
    t["_o"] = t.model.map({m: i for i, m in enumerate(_ordered(t.model, task))})
    return t.sort_values("_o").drop(columns="_o").reset_index(drop=True), lofo


def plot_curves(summ, task, metric, path):
    models = _ordered(summ.model.unique(), task)
    ncol, nrow = 5, int(np.ceil(len(models) / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, models):
        g = summ[summ.model == model].sort_values("config_id").reset_index(drop=True)
        y = g[f"{metric}_mean"].to_numpy(float)
        err = g[f"{metric}_std"].to_numpy(float)
        x = np.arange(len(g))
        ax.errorbar(x, y, yerr=err, marker="o", ms=4, lw=1, capsize=2,
                    color="#444", ecolor="#bbb")
        if 0 in g.config_id.values:
            i0 = int(np.where(g.config_id.values == 0)[0][0])
            ax.scatter([i0], [y[i0]], s=70, facecolor="none", edgecolor="#d62728",
                       lw=1.6, zorder=5)
        ib = int(np.argmax(y))
        ax.scatter([ib], [y[ib]], s=70, facecolor="none", edgecolor="#2ca02c",
                   lw=1.6, zorder=5)
        ax.set_title(f"{model}  (best={y[ib]:.3f})", fontsize=9)
        ax.set_xlabel("config id (0 = project default)", fontsize=7)
        ax.set_ylabel(metric, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.25, lw=.5)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle(f"{task}: hyper-parameter sensitivity ({metric}, "
                 f"red ring = default, green ring = grid-best)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gain(tab, task, metric, path):
    x = np.arange(len(tab))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tab)), 4.6))
    ax.bar(x - 1.5 * w, tab.default, w, label="default (stage-1 params)", color="#c9c9c9")
    ax.bar(x - 0.5 * w, tab.grid_mean, w, label="grid mean (random config)", color="#8ea9db")
    ax.bar(x + 0.5 * w, tab.lofo_honest, w, label="LOFO-honest (tuned, unbiased)", color="#f2b134")
    ax.bar(x + 1.5 * w, tab.grid_best, w, label="grid best (NOT honest)", color="#4c9f70")
    ax.set_xticks(x)
    ax.set_xticklabels(tab.model, rotation=0, fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: tuning gain ({metric}) -- "
                 f"grid-best overstates what tuning really buys")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    summary_lines = []
    recommended = {}
    for task in ("clf", "reg"):
        metric = TASK_METRIC[task]
        summ = pd.read_csv(OUT / f"{task}_summary_sweep.csv", encoding="utf-8-sig")
        per = pd.read_csv(OUT / f"{task}_per_fold_sweep.csv", encoding="utf-8-sig")
        tab, lofo = build_table(summ, per, task)
        tab.to_csv(RES / f"{task}_tuning_table.csv", index=False, encoding="utf-8-sig")

        print(f"\n{'='*100}\n== {task} ({metric}) ==\n{'='*100}")
        print(tab[["model", "default", "grid_mean", "grid_best", "best_params",
                   "gain_vs_default", "lofo_honest", "lofo_gain_vs_default",
                   "lofo_pick_stable"]].to_string(index=False))

        # the model's most recent best setting that is at least as good as the
        # grid mean and preferred when fold picks are stable
        rec = {}
        for _, r in tab.iterrows():
            g = summ[summ.model == r.model].sort_values(f"{metric}_mean",
                                                        ascending=False)
            lofo_best = g[g.config_id.isin(lofo[r.model]["picks"])].sort_values(
                f"{metric}_mean", ascending=False)
            rec[r.model] = {"params": json.loads(lofo_best.params.iloc[0]),
                            "lofo_honest": r.lofo_honest,
                            "default": r.default}
        recommended[task] = rec

        plot_curves(summ, task, metric, RES / f"fig1_param_curves_{task}.png")
        plot_gain(tab, task, metric, RES / f"fig2_tuning_gain_{task}.png")

        summary_lines.append(
            f"[{task}] honest (LOFO) ranking:\n" +
            tab.sort_values("lofo_honest", ascending=False)
            [["model", "default", "lofo_honest", "grid_best"]]
            .to_string(index=False))

    (RES / "recommended_params.json").write_text(
        json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8")
    (RES / "_analysis_dump.txt").write_text("\n\n".join(summary_lines),
                                            encoding="utf-8")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
