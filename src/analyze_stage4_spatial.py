"""Stage-4 analysis: the same (selector x model x K) grid, but under SPATIAL
block folds instead of random folds.

Inputs
------
experiments/2026-10-12_selector_zoo/outputs/clf_per_fold_stage4.csv   (B_spatial)
experiments/2026-10-12_selector_zoo/outputs/reg_per_fold_stage4.csv   (B_spatial)
experiments/2026-10-12_selector_zoo/outputs/*_per_fold_stage2.csv     (A_random)

Both runs use **seed 42 for the model** and the same full grid, so the ONLY
thing that changes between them is the fold protocol. Every comparison below is
therefore a controlled single-variable contrast.

Outputs
-------
results/selector_zoo_spatial/
    gap_by_combo_{task}.csv       per (selector, model, k): random, spatial, gap
    gap_by_selector_{task}.csv    marginal mean over (model, k)
    gap_by_model_{task}.csv       marginal mean over (selector, k)
    gap_by_k_{task}.csv           marginal mean over (selector, model)
    leaderboard_spatial_{task}.csv  absolute ranking under spatial folds
    ranking_shift_{task}.csv      selector mean random vs spatial + rank move
    fig1_protocol_k_curve_clf.png
    fig2_protocol_k_curve_reg.png
    fig3_gap_heatmap_{task}.png
    fig4_ranking_shift_{task}.png
    README.md   (hand-written on first run only)

Definitions
-----------
gap = metric(random folds) - metric(spatial folds)   [same selector, model, K]

gap > 0  => the random-fold number was optimistic, i.e. spatial leakage inflates
            the score by that much.

IMPORTANT caveat that must travel with every gap number: the random protocol is
5-fold (train = 80% of rows) while the spatial protocol is 3-fold (train = 67%).
Part of the gap is therefore "less training data", not "spatial leakage". This
confound is unavoidable here (with 10 blocks, a 5-fold spatial split produces
degenerate 2-sample validation folds) and is reported explicitly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"
RES = ROOT / "results" / "selector_zoo_spatial"

# primary metric per task (clf: accuracy, to stay comparable with P0-2's
# gap_table; ROC-AUC reported alongside because the stage-1 notes leaned on it)
MAIN = {"clf": "accuracy", "reg": "r2"}
EXTRA = {"clf": ["roc_auc"], "reg": ["rmse"]}
LABEL = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "r2": "R^2",
         "rmse": "RMSE"}


# --------------------------------------------------------------------------- #
def load(task):
    """Per-fold rows of both protocols, side by side."""
    r = pd.read_csv(OUT / f"{task}_per_fold_stage2.csv", encoding="utf-8-sig")
    s = pd.read_csv(OUT / f"{task}_per_fold_stage4.csv", encoding="utf-8-sig")
    r = r[r["scheme"] == "A_random"].copy()
    s = s[s["scheme"].str.startswith("B_spatial")].copy()
    return r, s


def build_pair(task, metric):
    """One row per (selector, model, k) with both protocol means."""
    r, s = load(task)
    key = ["selector", "model", "k"]
    # average over folds within each protocol, then join on the grid cell
    a = (r.groupby(key, as_index=False)
         .agg(n_folds_random=("fold", "size"),
              **{f"random_{metric}": (metric, "mean")}))
    b = (s.groupby(key, as_index=False)
         .agg(n_folds_spatial=("fold", "size"),
              **{f"spatial_{metric}": (metric, "mean")}))
    m = a.merge(b, on=key, how="inner")
    m[f"gap_{metric}"] = m[f"random_{metric}"] - m[f"spatial_{metric}"]
    return m


def marginal(pair, metric, by):
    g = (pair.groupby(by, as_index=False)
         .agg(**{f"random_{metric}": (f"random_{metric}", "mean"),
                 f"spatial_{metric}": (f"spatial_{metric}", "mean"),
                 f"gap_{metric}": (f"gap_{metric}", "mean"),
                 f"gap_{metric}_std": (f"gap_{metric}", "std"),
                 "n_combos": (f"gap_{metric}", "size")}))
    return g.sort_values(f"gap_{metric}", ascending=False)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def analyse(task):
    metric = MAIN[task]
    pair = build_pair(task, metric)

    table = {"combo": pair}
    for by in (["selector"], ["model"], ["k"]):
        table["_".join(by)] = marginal(pair, metric, by)

    # absolute leaderboard under spatial folds
    lb = (pair.sort_values(f"spatial_{metric}", ascending=False)
          .loc[:, ["selector", "model", "k", f"random_{metric}",
                   f"spatial_{metric}", f"gap_{metric}"]])
    table["leaderboard"] = lb

    # ★ difference-in-differences. The `random` selector contains NO method at
    # all, yet it also loses score when we swap random folds for spatial folds
    # -- that part is purely "the spatial protocol is harder / trains on fewer
    # rows", not leakage. Subtracting the random selector's own gap leaves the
    # leakage that a selector's *use of the data* actually adds.
    sel = marginal(pair, metric, ["selector"])
    base = float(sel.loc[sel["selector"] == "random", f"gap_{metric}"].iloc[0])
    sel[f"excess_gap_{metric}"] = sel[f"gap_{metric}"] - base
    table["selector"] = sel
    table["base_gap"] = base

    # does the selector ranking survive the protocol switch?
    rs = marginal(pair, metric, ["selector"])
    rho, p_rho = spearmanr(rs[f"random_{metric}"], rs[f"spatial_{metric}"])
    rs = rs.assign(
        rank_random=rs[f"random_{metric}"].rank(ascending=False).astype(int),
        rank_spatial=rs[f"spatial_{metric}"].rank(ascending=False).astype(int))
    rs["rank_move"] = rs["rank_random"] - rs["rank_spatial"]
    table["ranking"] = rs
    table["spearman"] = (float(rho), float(p_rho))

    # descriptive test on the paired (selector, model, k) gaps (exploration only:
    # no multiplicity correction -- that is confirmation-stage work)
    g = pair[f"gap_{metric}"].dropna().to_numpy()
    try:
        p_gap = float(wilcoxon(g, alternative="greater")[1])
    except ValueError:
        p_gap = float("nan")
    table["p_gap"] = p_gap
    return metric, table


# --------------------------------------------------------------------------- #
def fig_k_curve(task, metric, pair):
    ks = sorted(pair["k"].unique())
    metrics = [metric] + EXTRA[task]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.6),
                             dpi=140, squeeze=False)
    for ax, mt in zip(axes[0], metrics):
        pm = build_pair(task, mt)
        for proto, color in (("random", "#c0392b"), ("spatial", "#2471a3")):
            mu = pm.groupby("k")[f"{proto}_{mt}"].mean().reindex(ks)
            sd = pm.groupby("k")[f"{proto}_{mt}"].std().reindex(ks)
            ax.plot(ks, mu, marker="o", lw=2.2, color=color,
                    label=f"{proto} folds")
            ax.fill_between(ks, mu - sd, mu + sd, color=color, alpha=0.12)
        ax.set_xlabel("K = number of selected features")
        ax.set_ylabel(f"mean {LABEL[mt]}")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_title(f"[{task}] {LABEL[mt]} vs K, by fold protocol")
    fig.suptitle("Stage-4: random folds vs spatial block folds "
                 "(1 seed, full selector x model grid)", y=1.02)
    fig.tight_layout()
    name = f"fig{1 if task == 'clf' else 2}_protocol_k_curve_{task}.png"
    fig.savefig(RES / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_gap_heatmap(task, metric, pair, k=10):
    d = pair[pair["k"] == k]
    piv = d.pivot(index="selector", columns="model", values=f"gap_{metric}")
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(0.72 * len(piv.columns) + 2.4,
                                    0.42 * len(piv.index) + 2.0), dpi=140)
    vmax = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=90)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_title(f"[{task}] K={k}: gap = {LABEL[metric]}(random) - "
                 f"{LABEL[metric]}(spatial)\nred = random folds optimistic "
                 f"(spatial leakage)")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="gap")
    fig.tight_layout()
    name = f"fig3_gap_heatmap_{task}.png"
    fig.savefig(RES / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_ranking_shift(task, metric, rs):
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=140)
    lo = min(rs[f"random_{metric}"].min(), rs[f"spatial_{metric}"].min())
    hi = max(rs[f"random_{metric}"].max(), rs[f"spatial_{metric}"].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=":", color="grey",
            zorder=1, label="no change (y = x)")
    ax.scatter(rs[f"random_{metric}"], rs[f"spatial_{metric}"], s=42,
               color="#2471a3", zorder=3)
    for _, row in rs.iterrows():
        ax.annotate(row["selector"],
                    (row[f"random_{metric}"], row[f"spatial_{metric}"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel(f"{LABEL[metric]} under random folds")
    ax.set_ylabel(f"{LABEL[metric]} under spatial folds")
    ax.set_title(f"[{task}] selector ranking under the two protocols")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"fig4_ranking_shift_{task}.png"
    fig.savefig(RES / name)
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
def main():
    RES.mkdir(parents=True, exist_ok=True)
    lines = []
    for task in ("clf", "reg"):
        metric, tb = analyse(task)
        pair, rs = tb["combo"], tb["ranking"]

        pair.to_csv(RES / f"gap_by_combo_{task}.csv", index=False,
                    encoding="utf-8-sig")
        tb["selector"].to_csv(RES / f"gap_by_selector_{task}.csv", index=False,
                              encoding="utf-8-sig")
        tb["model"].to_csv(RES / f"gap_by_model_{task}.csv", index=False,
                           encoding="utf-8-sig")
        tb["k"].to_csv(RES / f"gap_by_k_{task}.csv", index=False,
                       encoding="utf-8-sig")
        tb["leaderboard"].to_csv(RES / f"leaderboard_spatial_{task}.csv",
                                 index=False, encoding="utf-8-sig")
        rs.to_csv(RES / f"ranking_shift_{task}.csv", index=False,
                  encoding="utf-8-sig")

        n = len(pair)
        n_pos = int((pair[f"gap_{metric}"] > 0).sum())
        print("\n" + "=" * 78)
        print(f"===== {task} | metric={metric} | combos={n} "
              f"| gap>0 in {n_pos}/{n} ({100*n_pos/n:.1f}%) =====")
        print(f"  overall mean gap  = {pair[f'gap_{metric}'].mean():+.4f}")
        print(f"  Wilcoxon gap>0    = p={tb['p_gap']:.4f}  (descriptive only)")
        print(f"  Spearman(rank_random, rank_spatial) = "
              f"{tb['spearman'][0]:+.3f} (p={tb['spearman'][1]:.3f})")
        print(f"\n-- {task}: gap by selector (mean over model x K) --")
        print(f"   [baseline: the `random` selector's own gap = "
              f"{tb['base_gap']:+.4f}  <- pure protocol effect, no method]")
        print(tb["selector"][["selector", f"random_{metric}",
                              f"spatial_{metric}", f"gap_{metric}",
                              f"excess_gap_{metric}"]]
              .round(4).to_string(index=False))
        print(f"\n-- {task}: gap by model (mean over selector x K) --")
        print(tb["model"][["model", f"random_{metric}", f"spatial_{metric}",
                           f"gap_{metric}"]].round(4).to_string(index=False))
        print(f"\n-- {task}: gap by K --")
        print(tb["k"][["k", f"random_{metric}", f"spatial_{metric}",
                       f"gap_{metric}"]].round(4).to_string(index=False))
        print(f"\n-- {task}: top-8 combos under SPATIAL folds --")
        print(tb["leaderboard"].head(8).round(4).to_string(index=False))

        # if ROC-AUC is what people remember from stage-1, report its gap too
        for mt in EXTRA[task]:
            pm = build_pair(task, mt)
            sel = (pm.groupby("selector", as_index=False)
                   .agg(**{f"gap_{mt}": (f"gap_{mt}", "mean")}))
            sel = sel.sort_values(f"gap_{mt}", ascending=False)
            print(f"\n-- {task}: gap by selector for {mt} --")
            print(sel.round(4).to_string(index=False))

        f1 = fig_k_curve(task, metric, pair)
        f3 = fig_gap_heatmap(task, metric, pair)
        f4 = fig_ranking_shift(task, metric, rs)
        print(f"\n[fig] {f1} / {f3} / {f4}")

        lines.append(f"\n## {task} ({metric})\n")
        lines.append(f"![k curve]({f1})\n")
        lines.append(f"![gap heatmap]({f3})\n")
        lines.append(f"![ranking shift]({f4})\n")

    readme = RES / "README.md"
    if not readme.exists():
        readme.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] wrote {readme}")
    else:
        print(f"[skip] {readme} exists -> hand-written notes preserved")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
