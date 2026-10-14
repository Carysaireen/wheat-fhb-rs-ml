"""Stage-4b analysis: decompose the stage-4 "gap" into its two causes.

The problem
-----------
Stage-4 compared two protocols that differed in TWO ways at once:

    A_random   the project's 5-fold random split   train ~80% of rows
    B_spatial  3-fold spatial block split          train ~67% of rows

so its "gap = A_random - B_spatial" mixed together

    (a) random folds  ->  spatial folds      [the effect we WANT]
    (b) 5 folds       ->  3 folds            [a confound: less training data]

The fix
-------
Stage-4b adds a third protocol with the SAME fold count as the spatial one:

    A_random3  = StratifiedKFold(3, shuffle=True, seed=42)   train ~67%

and the sample counts match almost exactly, which is what makes the contrast
valid:

    clf: A_random3 trains on 88/88/88  (mean 88.0)
         B_spatial trains on 79/100/85 (mean 88.0)   <- same mean
    reg: A_random3 trains on 77/77/78  (mean 77.3)
         B_spatial trains on 85/71/76  (mean 77.3)   <- same mean

Now the original gap splits cleanly into two additive pieces:

    total_gap   = A_random5 - B_spatial
    depth_gap   = A_random5 - A_random3     <- pure training-size / fold-count
    spatial_gap = A_random3 - B_spatial     <- spatial effect at FIXED depth

    total_gap = depth_gap + spatial_gap

`spatial_gap` is the honest number: it asks "if I keep the fold count the same
and only change WHICH rows go together, how much does the score drop?"

Inputs
------
outputs/clf_per_fold_stage2.csv   (A_random   = 5-fold random)
outputs/clf_per_fold_stage4b.csv  (A_random3  = 3-fold random)   <- new
outputs/clf_per_fold_stage4.csv   (B_spatial  = 3-fold spatial)
(and the reg_* equivalents)

Outputs -> results/selector_zoo_protocol/
    protocol3_by_combo_{task}.csv      per (selector, model, k): the 3 means + 2 gaps
    protocol3_by_selector_{task}.csv   marginal over (model, k)
    protocol3_by_model_{task}.csv      marginal over (selector, k)
    protocol3_by_k_{task}.csv          marginal over (selector, model)
    decomposition_by_selector_{task}.csv  incl. excess gaps (minus `random`)
    ranking_shift_random3_vs_spatial_{task}.csv
    fig1_k_curve_{task}.png             three protocols vs K
    fig2_decomposition_{task}.png       depth vs spatial gap per selector
    fig3_rank_scatter_{task}.png        selector ranking, random3 vs spatial
    README.md                           hand-written (never overwritten)

Exploration only: one model seed, one partition seed, no multiplicity
correction. Confirmation-stage work is listed in docs/确认阶段待办清单.md.
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
RES = ROOT / "results" / "selector_zoo_protocol"

MAIN = {"clf": "accuracy", "reg": "r2"}
EXTRA = {"clf": ["roc_auc"], "reg": ["rmse"]}
LABEL = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "r2": "R^2",
         "rmse": "RMSE"}
KEY = ["selector", "model", "k"]

# protocol key -> (file tag, scheme name, short label)
PROTO = [
    ("r5", "stage2", "A_random", "random 5-fold (train ~80%)"),
    ("r3", "stage4b", "A_random3", "random 3-fold (train ~67%)"),
    ("sp", "stage4", None, "spatial 3-fold (train ~67%)"),
]
C_LABEL = {"r5": "random 5-fold", "r3": "random 3-fold", "sp": "spatial 3-fold"}


def load_proto(task, tag, scheme):
    d = pd.read_csv(OUT / f"{task}_per_fold_{tag}.csv", encoding="utf-8-sig")
    if scheme is None:
        d = d[d["scheme"].str.startswith("B_spatial")].copy()
    else:
        d = d[d["scheme"] == scheme].copy()
    return d


def build_triple(task, metric):
    """One row per (selector, model, k) with the three protocol means."""
    cols = {}
    for short, tag, scheme, _ in PROTO:
        d = load_proto(task, tag, scheme)
        g = (d.groupby(KEY, as_index=False)
             .agg(**{f"n_folds_{short}": ("fold", "size"),
                     f"{short}_{metric}": (metric, "mean")}))
        cols[short] = g
    m = cols["r5"]
    for short in ("r3", "sp"):
        m = m.merge(cols[short], on=KEY, how="inner")
    # the two additive pieces of the original stage-4 gap
    m[f"depth_gap_{metric}"] = m[f"r5_{metric}"] - m[f"r3_{metric}"]
    m[f"spatial_gap_{metric}"] = m[f"r3_{metric}"] - m[f"sp_{metric}"]
    m[f"total_gap_{metric}"] = m[f"r5_{metric}"] - m[f"sp_{metric}"]
    m[f"check_resid_{metric}"] = (m[f"total_gap_{metric}"]
                                  - m[f"depth_gap_{metric}"]
                                  - m[f"spatial_gap_{metric}"])
    return m


def marginal(tri, metric, by):
    return (tri.groupby(by, as_index=False)
            .agg(n_combos=(f"total_gap_{metric}", "size"),
                 **{f"{c}_{metric}": (f"{c}_{metric}", "mean")
                    for c in ("r5", "r3", "sp")},
                 **{f"{g}_{metric}": (f"{g}_{metric}", "mean")
                    for g in ("depth_gap", "spatial_gap", "total_gap")}))


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def paired_p(d, alternative="greater"):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(wilcoxon(d, alternative=alternative)[1])
    except ValueError:
        return float("nan")


def analyse(task, metric):
    tri = build_triple(task, metric)
    tb = {"combo": tri}
    for by in (["selector"], ["model"], ["k"]):
        tb["_".join(by)] = marginal(tri, metric, by)

    # --- difference-in-differences on BOTH pieces -------------------------- #
    # `random` contains no method at all, so any gap it shows is pure protocol.
    # Subtracting it leaves what the selector's USE OF THE DATA adds.
    sel = marginal(tri, metric, ["selector"])
    for g in ("depth_gap", "spatial_gap", "total_gap"):
        base = float(sel.loc[sel["selector"] == "random", f"{g}_{metric}"].iloc[0])
        sel[f"excess_{g}_{metric}"] = sel[f"{g}_{metric}"] - base
    tb["selector"] = sel
    tb["bases"] = {g: float(sel.loc[sel["selector"] == "random",
                                 f"{g}_{metric}"].iloc[0])
                   for g in ("depth_gap", "spatial_gap", "total_gap")}

    # --- ranking: does the selector order survive random3 -> spatial? ------- #
    rs = marginal(tri, metric, ["selector"]).copy()
    rs["rank_r3"] = rs[f"r3_{metric}"].rank(ascending=False).astype(int)
    rs["rank_sp"] = rs[f"sp_{metric}"].rank(ascending=False).astype(int)
    rs["rank_move"] = rs["rank_r3"] - rs["rank_sp"]
    rho_new, p_new = spearmanr(rs[f"r3_{metric}"], rs[f"sp_{metric}"])
    rho_old, p_old = spearmanr(rs[f"r5_{metric}"], rs[f"sp_{metric}"])
    tb["ranking"] = rs
    tb["spearman"] = {"r3_vs_sp": (float(rho_new), float(p_new)),
                      "r5_vs_sp": (float(rho_old), float(p_old))}

    # --- descriptive tests (no multiplicity correction: exploration) ------- #
    tb["p"] = {
        "depth": paired_p(tri[f"depth_gap_{metric}"]),
        "spatial": paired_p(tri[f"spatial_gap_{metric}"]),
        "total": paired_p(tri[f"total_gap_{metric}"]),
        "spatial_frac_pos": float((tri[f"spatial_gap_{metric}"] > 0).mean()),
        "depth_frac_pos": float((tri[f"depth_gap_{metric}"] > 0).mean()),
    }
    return tb


# --------------------------------------------------------------------------- #
def fig_k_curve(task, metric, tri):
    ks = sorted(tri["k"].unique())
    metrics = [metric] + EXTRA[task]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.6),
                             dpi=140, squeeze=False)
    style = {"r5": ("#c0392b", "o", "-"), "r3": ("#e67e22", "s", "--"),
             "sp": ("#2471a3", "^", "-")}
    for ax, mt in zip(axes[0], metrics):
        t = build_triple(task, mt)
        for short, (col, mk, ls) in style.items():
            mu = t.groupby("k")[f"{short}_{mt}"].mean().reindex(ks)
            sd = t.groupby("k")[f"{short}_{mt}"].std().reindex(ks)
            ax.plot(ks, mu, marker=mk, ls=ls, lw=2.1, color=col,
                    label=C_LABEL[short], markersize=6)
            ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.10)
        ax.set_xlabel("K = number of selected features")
        ax.set_ylabel(f"mean {LABEL[mt]}")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"[{task}] {LABEL[mt]} vs K, three protocols")
    fig.suptitle("Stage-4b: random 5-fold vs random 3-fold vs spatial 3-fold\n"
                 "(1 seed, full selector x model grid)", y=1.06)
    fig.tight_layout()
    name = f"fig1_k_curve_{task}.png"
    fig.savefig(RES / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_decomposition(task, metric, sel):
    d = sel[sel["selector"] != "random"].copy()
    d = d.sort_values(f"total_gap_{metric}", ascending=False)
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(0.62 * len(d) + 3.0, 4.8), dpi=140)
    ax.bar(x - w / 2, d[f"depth_gap_{metric}"], w, label="training-size effect "
           "(5-fold -> 3-fold, random)", color="#95a5a6")
    ax.bar(x + w / 2, d[f"spatial_gap_{metric}"], w,
           label="spatial effect at FIXED 3-fold depth", color="#2471a3")
    ax.plot(x, d[f"total_gap_{metric}"], "kD", ms=5,
            label="total gap (random 5-fold - spatial 3-fold)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, d["selector"], rotation=45, ha="right")
    ax.set_ylabel(f"gap in {LABEL[metric]}")
    ax.set_title(f"[{task}] stage-4 gap decomposed per selector\n"
                 f"(mean over model x K; bolted-on greys are NOT spatial leakage)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"fig2_decomposition_{task}.png"
    fig.savefig(RES / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_rank_scatter(task, metric, rs):
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=140)
    lo = min(rs[f"r3_{metric}"].min(), rs[f"sp_{metric}"].min())
    hi = max(rs[f"r3_{metric}"].max(), rs[f"sp_{metric}"].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=":", color="grey",
            zorder=1, label="no change (y = x)")
    ax.scatter(rs[f"r3_{metric}"], rs[f"sp_{metric}"], s=46, color="#e67e22",
               zorder=3)
    for _, row in rs.iterrows():
        ax.annotate(row["selector"],
                    (row[f"r3_{metric}"], row[f"sp_{metric}"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel(f"{LABEL[metric]} under random 3-fold")
    ax.set_ylabel(f"{LABEL[metric]} under spatial 3-fold")
    ax.set_title(f"[{task}] selector ranking, FIXED 3-fold depth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"fig3_rank_scatter_{task}.png"
    fig.savefig(RES / name)
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
def main():
    RES.mkdir(parents=True, exist_ok=True)
    lines = []
    for task in ("clf", "reg"):
        metric = MAIN[task]
        tb = analyse(task, metric)
        tri, sel, rs = tb["combo"], tb["selector"], tb["ranking"]

        tri.to_csv(RES / f"protocol3_by_combo_{task}.csv", index=False,
                   encoding="utf-8-sig")
        tb["k"].to_csv(RES / f"protocol3_by_k_{task}.csv", index=False,
                       encoding="utf-8-sig")
        tb["model"].to_csv(RES / f"protocol3_by_model_{task}.csv", index=False,
                           encoding="utf-8-sig")
        sel.to_csv(RES / f"decomposition_by_selector_{task}.csv", index=False,
                   encoding="utf-8-sig")
        rs.to_csv(RES / f"ranking_shift_random3_vs_spatial_{task}.csv",
                  index=False, encoding="utf-8-sig")

        p = tb["p"]
        b = tb["bases"]
        print("\n" + "=" * 80)
        print(f"===== {task} | metric={metric} | combos={len(tri)} =====")
        print(f"  mean total gap   (r5 - sp ) = {tri[f'total_gap_{metric}'].mean():+.4f}"
              f"   p={p['total']:.4f} {stars(p['total'])}")
        print(f"  mean DEPTH gap   (r5 - r3 ) = {tri[f'depth_gap_{metric}'].mean():+.4f}"
              f"   p={p['depth']:.4f} {stars(p['depth'])}   (>0 in "
              f"{100*p['depth_frac_pos']:.1f}%)   <- pure training-size effect")
        print(f"  mean SPATIAL gap (r3 - sp ) = {tri[f'spatial_gap_{metric}'].mean():+.4f}"
              f"   p={p['spatial']:.4f} {stars(p['spatial'])}   (>0 in "
              f"{100*p['spatial_frac_pos']:.1f}%)   <- honest spatial effect")
        print(f"  identity check: max|total-depth-spatial| = "
              f"{tri[f'check_resid_{metric}'].abs().max():.2e}")
        print(f"  [baseline `random` selector] depth={b['depth_gap']:+.4f} "
              f"spatial={b['spatial_gap']:+.4f} total={b['total_gap']:+.4f}")
        sp = tb["spearman"]
        print(f"  Spearman  r3 vs sp = {sp['r3_vs_sp'][0]:+.3f} "
              f"(p={sp['r3_vs_sp'][1]:.3f})   [was r5 vs sp = "
              f"{sp['r5_vs_sp'][0]:+.3f}, p={sp['r5_vs_sp'][1]:.3f}]")

        print(f"\n-- {task}: three protocols by selector (mean over model x K) --")
        print(sel[["selector", f"r5_{metric}", f"r3_{metric}", f"sp_{metric}",
                   f"depth_gap_{metric}", f"spatial_gap_{metric}",
                   f"total_gap_{metric}"]].round(4).to_string(index=False))
        print(f"\n-- {task}: EXCESS gaps (selector gap minus the `random` "
              f"baseline gap) --")
        print(sel[["selector", f"excess_depth_gap_{metric}",
                   f"excess_spatial_gap_{metric}",
                   f"excess_total_gap_{metric}"]].round(4).to_string(index=False))
        print(f"\n-- {task}: three protocols by model --")
        print(tb["model"][["model", f"r5_{metric}", f"r3_{metric}", f"sp_{metric}",
                           f"depth_gap_{metric}", f"spatial_gap_{metric}",
                           f"total_gap_{metric}"]].round(4).to_string(index=False))
        print(f"\n-- {task}: three protocols by K --")
        print(tb["k"][["k", f"r5_{metric}", f"r3_{metric}", f"sp_{metric}",
                       f"depth_gap_{metric}", f"spatial_gap_{metric}",
                       f"total_gap_{metric}"]].round(4).to_string(index=False))

        for mt in EXTRA[task]:
            t = build_triple(task, mt)
            g = (t.groupby("selector", as_index=False)
                 .agg(**{f"depth_gap_{mt}": (f"depth_gap_{mt}", "mean"),
                         f"spatial_gap_{mt}": (f"spatial_gap_{mt}", "mean"),
                         f"total_gap_{mt}": (f"total_gap_{mt}", "mean")}))
            g = g.reindex(g[f"spatial_gap_{mt}"].sort_values(ascending=False).index)
            print(f"\n-- {task}: decomposition by selector for {mt} --")
            print(g.round(4).to_string(index=False))

        f1 = fig_k_curve(task, metric, tri)
        f2 = fig_decomposition(task, metric, sel)
        f3 = fig_rank_scatter(task, metric, rs)
        print(f"\n[fig] {f1} / {f2} / {f3}")

        lines.append(f"\n## {task} ({metric})\n")
        for f in (f1, f2, f3):
            lines.append(f"![{f}]({f})\n")

    readme = RES / "README.md"
    if not readme.exists():
        readme.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] wrote {readme}")
    else:
        print(f"[skip] {readme} exists -> hand-written notes preserved")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
