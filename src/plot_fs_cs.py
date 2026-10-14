"""Figures for the FS-CS selector experiment (results/fs_cs_selection/).

fig1  selector-level spatial gap (A_random - B_spatial), 3 models
fig2  FS-CS univariate feature stability (top-20 by mean rank)
fig3  random vs spatial absolute performance (dumbbell), RF model
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "2026-10-12_fs_cs" / "outputs"
OUT = ROOT / "results" / "fs_cs_selection"

C_A = "#4c78a8"      # random folds
C_B = "#e45756"      # spatial folds
C_GAP = "#f58518"
C_FSCS = "#54a24b"

SHORT = {
    "current_2": "current_2\n(2)",
    "rfmi_top3of5": "RF+MI\n(6)",
    "fs_cs_univariate_top10": "FS-CS-uni\n(10)",
    "fs_cs_greedy_top10": "FS-CS-grd\n(10)",
    "fs_cs_univariate_top20": "FS-CS-uni\n(20)",
    "fs_cs_greedy_top20": "FS-CS-grd\n(20)",
    "fs_cs_univariate_top30": "FS-CS-uni\n(30)",
    "fs_cs_greedy_top30": "FS-CS-grd\n(30)",
    "boruta_3of5": "Boruta\n(30)",
    "all_117": "all_117\n(117)",
}


def order_sets(bench):
    nf = bench.groupby("set").n_features.first().sort_values()
    return list(nf.index)


def fig1(bench):
    order = order_sets(bench)
    x = np.arange(len(order))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    for i, m in enumerate(["LR", "RF", "SVM"]):
        sub = bench[bench.model == m].set_index("set").reindex(order)
        vals = sub.accuracy_gap.to_numpy()
        ax.bar(x + (i - 1) * w, vals, w, label=m,
               color=["#4c78a8", "#54a24b", "#b279a2"][i], edgecolor="white")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(s, s) for s in order], fontsize=9)
    ax.set_ylabel("准确率 gap（A_random − B_spatial）", fontsize=10.5)
    ax.set_title("(a) 各特征选择器的空间 gap：越靠近 0 越稳健", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for m, col in zip(["LR", "RF", "SVM"], ["#4c78a8", "#54a24b", "#b279a2"]):
        sub = bench[bench.model == m]
        ax.scatter(sub.n_features, sub.accuracy_gap, s=52, color=col, label=m,
                   alpha=0.85, edgecolor="white", zorder=3)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([2, 6, 10, 20, 30, 117])
    ax.set_xticklabels(["2", "6", "10", "20", "30", "117"])
    ax.set_xlabel("特征数（对数轴）", fontsize=10.5)
    ax.set_ylabel("准确率 gap", fontsize=10.5)
    ax.set_title("(b) gap 随特征数增大而上升", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle("空间分组折下各特征选择器的乐观偏差（随机分层折 vs 空间块折）",
                 fontsize=13.5, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_selector_spatial_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig1_selector_spatial_gap.png")


def fig2(stab):
    top = stab.head(20).sort_values("mean_rank", ascending=False)
    colors = [C_FSCS if r <= 5 else "#9fc98a" for r in top.top10_folds]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(top.feature, top.mean_rank, color=colors, edgecolor="white")
    for y, (_, row) in enumerate(top.iterrows()):
        ax.text(row.mean_rank + 0.35, y, f"top10×{int(row.top10_folds)}  top20×{int(row.top20_folds)}",
                va="center", fontsize=8, color="#555")
    ax.set_xlabel("跨折平均排名（越小越重要）", fontsize=10.5)
    ax.set_title("FS-CS（单变量 J = S_B/S_W）选出的前 20 个特征\n"
                 "深色＝5 折中均进入前 10", fontsize=12)
    ax.set_xlim(0, top.mean_rank.max() * 1.35)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_fs_cs_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig2_fs_cs_stability.png")


def fig3(bench):
    sub = bench[bench.model == "RF"].sort_values("n_features")
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.plot([row.accuracy_B, row.accuracy_A], [yi, yi], color="#bbb", lw=2, zorder=1)
    ax.scatter(sub.accuracy_A, y, s=64, color=C_A, label="A_random", zorder=3,
               edgecolor="white")
    ax.scatter(sub.accuracy_B, y, s=64, color=C_B, label="B_spatial", zorder=3,
               edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(s, s).replace("\n", " ") for s in sub.set])
    ax.set_xlabel("准确率（RF 模型）", fontsize=10.5)
    ax.set_title("RF 模型：随机折 → 空间折的准确率位移\n线段越长＝对空间泄漏越敏感", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_random_vs_spatial_rf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig3_random_vs_spatial_rf.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = pd.read_csv(EXP / "selector_benchmark.csv", encoding="utf-8-sig")
    stab = pd.read_csv(EXP / "fs_cs_stability.csv", encoding="utf-8-sig")
    fig1(bench)
    fig2(stab)
    fig3(bench)


if __name__ == "__main__":
    main()
