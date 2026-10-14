"""
跨阶段对齐基准 · 分析
======================

读 run_baseline_alignment.py 的产出，回答两个贯穿性问题：

  Q1（回归）现在最好的回归组合，是否优于项目最早的最强基线？
  Q2（分类）现在的分类模型，是否优于「最优双特征 + LR」？

做法：
  1. 把 B_spatial3 的三个分区种子（42/7/2026）合并成一个口径（与 FS-CS 一致）。
  2. 产出「基线 vs 当前」同折同指标对照表（headline_*.csv）。
  3. 产出对比图。

用法：
    python -u src/analyze_baseline_alignment.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "2026-10-13_baseline_alignment" / "outputs"
RES = ROOT / "results" / "baseline_alignment"
RES.mkdir(parents=True, exist_ok=True)

MET = {"clf": ["accuracy", "f1", "recall", "roc_auc"],
       "reg": ["r2", "rmse", "mae"]}
MAIN = {"clf": "accuracy", "reg": "r2"}
LABEL = {"clf": "Accuracy", "reg": "R²"}
TITLE = {"clf": "Classification", "reg": "Regression"}

BASELINE_STAR = {          # 「最早基线」的代表组合
    "clf": [("current_2", "LR"), ("current_2", "SVM"),
            ("boruta_30", "RF"), ("boruta_30", "SVM"),
            ("all_117", "RF"), ("all_117", "SVM")],
    "reg": [("boruta_30", "RF"), ("boruta_30", "Ridge"),
            ("all_117", "RF"), ("current_2", "RF"),
            ("common_6", "RF")],
}


def load(task: str) -> pd.DataFrame:
    d = pd.read_csv(OUT / f"alignment_{task}_summary.csv", encoding="utf-8-sig")
    d["scheme2"] = d.scheme.str.replace(r"_s\d+$", "", regex=True)
    g = d.groupby(["set", "model", "n_features", "scheme2"], as_index=False).agg(
        n_eval=("n_eval", "sum"),
        **{m: (f"{m}_mean", "mean") for m in MET[task]})
    return g


def wide(task: str) -> pd.DataFrame:
    """一张宽表：行 = (set, model)，列 = scheme × metric。"""
    g = load(task)
    w = g.pivot_table(index=["set", "model", "n_features"],
                      columns="scheme2", values=MET[task])
    w.columns = [f"{s}__{m}" for m, s in w.columns]
    w = w.reset_index()
    return w


def headline(task: str, w: pd.DataFrame) -> pd.DataFrame:
    m = MAIN[task]
    rows = []
    for _, r in w.iterrows():
        rec = {"set": r["set"], "model": r["model"], "n_features": r["n_features"]}
        for s in ("A_random5", "B_spatial3"):
            for mm in MET[task]:
                k = f"{s}__{mm}"
                rec[f"{s}_{mm}"] = round(float(r[k]), 4) if k in w.columns and pd.notna(r[k]) else np.nan
        if f"A_random5_{m}" in rec and f"B_spatial3_{m}" in rec:
            rec["gap"] = round(rec["A_random5_" + m] - rec["B_spatial3_" + m], 4)
        rows.append(rec)
    h = pd.DataFrame(rows)
    return h.sort_values("A_random5_" + m, ascending=False).reset_index(drop=True)


def tag_group(task: str, setname: str, model: str) -> str:
    if (setname, model) in BASELINE_STAR[task]:
        return "baseline"
    if setname.startswith(("relieff_10", "fs_cs_10", "cars_10")):
        return "current"
    return "other"


def plot(task: str, h: pd.DataFrame) -> None:
    m = MAIN[task]
    tab = h.copy()
    tab["group"] = [tag_group(task, s, mo) for s, mo in zip(tab.set, tab.model)]
    pick = pd.concat([
        tab[tab.group == "baseline"],
        tab[tab.group == "current"].sort_values("A_random5_" + m, ascending=False).head(8),
    ]).drop_duplicates(["set", "model"]).sort_values("A_random5_" + m)

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(pick) + 1.4)))
    y = np.arange(len(pick))
    colA, colB = "#c0392b", "#2c7fb8"      # 红=随机折, 蓝=空间折
    ax.barh(y + 0.20, pick["A_random5_" + m], height=0.38, color=colA,
            label="A_random5 (random 5-fold)")
    ax.barh(y - 0.20, pick["B_spatial3_" + m], height=0.38, color=colB,
            label="B_spatial3 (spatial blockCV 3-fold)")
    labels = [f"{s} + {mo}" for s, mo in zip(pick.set, pick.model)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(LABEL[task], fontsize=10)
    ax.set_title(f"{TITLE[task]}: earliest baseline vs current best "
                 f"(identical folds, identical code)", fontsize=11)
    for yy, (a, b) in enumerate(zip(pick["A_random5_" + m], pick["B_spatial3_" + m])):
        ax.text(a, yy + 0.20, f" {a:.3f}", va="center", fontsize=7, color="#c0392b")
        ax.text(b, yy - 0.20, f" {b:.3f}", va="center", fontsize=7, color="#2c7fb8")
    ax.axvline(0, color="k", lw=0.6)
    ax.grid(axis="x", ls=":", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / f"fig1_{task}_baseline_vs_current.png", dpi=150)
    plt.close(fig)


def plot_gap(task: str, h: pd.DataFrame) -> tuple[float, float]:
    """gap（随机折 − 空间折）随特征集规模的变化。

    这是贯穿全项目的一条主线：特征集越大，随机折数字越"虚"（空间泄漏越多）。
    只画**默认**模型（不带 * 的），避免调优与默认混在一起。
    """
    m = MAIN[task]
    d = h.dropna(subset=["gap"]).copy()
    d = d[~d.model.str.endswith("*")]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    cmap = plt.get_cmap("tab10")
    for i, (name, g) in enumerate(d.groupby("set")):
        ax.scatter(g.n_features, g["gap"], s=50, color=cmap(i % 10), label=name,
                   edgecolor="k", linewidth=0.4, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r.model, (r.n_features, r["gap"]), fontsize=6.5,
                        xytext=(4, 2), textcoords="offset points", color="#333333")
    x = d.n_features.to_numpy(float)
    yv = d["gap"].to_numpy(float)
    b, a = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    r = float(np.corrcoef(x, yv)[0, 1])
    ax.plot(xs, a + b * xs, "--", color="k", lw=1.3,
            label=f"trend: gap = {a:+.3f} + {b:.1e}·K   (Pearson r = {r:+.2f})")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("number of features (K)", fontsize=10)
    ax.set_ylabel(f"{LABEL[task]} gap   (random 5-fold − spatial 3-fold)", fontsize=10)
    ax.set_title(f"{TITLE[task]}: spatial instability grows with feature-set size",
                 fontsize=11)
    ax.grid(ls=":", alpha=0.35)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / f"fig2_{task}_gap_vs_nfeat.png", dpi=150)
    plt.close(fig)
    return r, b


def main() -> None:
    print("=" * 78)
    print("baseline alignment :: earliest baseline vs current best")
    print("=" * 78)
    for task in ("clf", "reg"):
        w = wide(task)
        h = headline(task, w)
        h.to_csv(RES / f"headline_{task}.csv", index=False, encoding="utf-8-sig")
        plot(task, h)
        corr, slope = plot_gap(task, h)

        m = MAIN[task]
        print(f"\n########## {TITLE[task]} ({LABEL[task]}) ##########")
        print(f"-- top 12 by A_random5 --")
        cols = ["set", "model", "n_features", f"A_random5_{m}", f"B_spatial3_{m}", "gap"]
        print(h.head(12)[cols].to_string(index=False))
        print(f"\n-- earliest-baseline representatives --")
        b = h[[(s, mo) in BASELINE_STAR[task] for s, mo in zip(h.set, h.model)]]
        print(b[cols].to_string(index=False))
        print(f"\n-- best per scheme --")
        for s in ("A_random5", "B_spatial3"):
            r = h.sort_values(f"{s}_{m}", ascending=False).iloc[0]
            print(f"  {s:12s}: {r['set']} + {r['model']}  {m} = {r[f'{s}_{m}']:.4f}")
        print(f"\n-- gap vs #features (default models) --")
        print(f"  Pearson r = {corr:+.3f}   slope = {slope:+.2e} per feature")
        print()
    print(f"[done] -> {RES}")


if __name__ == "__main__":
    main()
