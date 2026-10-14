"""Heatmaps of the Stage-1 (selector x model) matrix.

Two panels:
  * classification -> ROC-AUC
  * regression     -> R^2
Rows = selectors (sorted by mean performance), cols = models.
The `random` selector row is highlighted: it is the *noise floor* -- the score
obtainable by picking 10 random features. Any selector not clearly above that
row has not demonstrated value at K=10.

Input : experiments/2026-10-12_selector_zoo/outputs/{clf,reg}_summary_stage1.csv
Output: results/selector_zoo/fig1_selector_model_heatmap.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"
OUT = ROOT / "results" / "selector_zoo"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FLOOR = "random"          # the noise-floor selector


def heat(ax, df, metric, title, cmap):
    m = df.pivot_table(index="selector", columns="model", values=f"{metric}_mean")
    # order rows/cols by their own means (best first)
    m = m.loc[m.mean(axis=1).sort_values(ascending=False).index,
              m.mean(axis=0).sort_values(ascending=False).index]
    arr = m.to_numpy(float)

    vmin = np.nanmin(arr) - 0.01 * (np.nanmax(arr) - np.nanmin(arr))
    vmax = np.nanmax(arr)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index, fontsize=9)
    ax.set_title(title, fontsize=11.5, pad=8)

    # annotate each cell
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = arr[i, j]
            rel = (v - vmin) / (vmax - vmin + 1e-12)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if rel > 0.62 else "black")

    # highlight the noise-floor row
    if FLOOR in list(m.index):
        r = list(m.index).index(FLOOR)
        ax.add_patch(plt.Rectangle((-0.5, r - 0.5), m.shape[1], 1, fill=False,
                                   edgecolor="#d62728", lw=2.2, zorder=5))
        ax.text(m.shape[1] - 1.45, r - 0.75, "噪声下限 →", color="#d62728",
                fontsize=8.5, va="center", ha="left", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#d62728", alpha=0.95, lw=0.8), zorder=6)
    return im


def main():
    clf = pd.read_csv(EXP / "clf_summary_stage1.csv")
    reg = pd.read_csv(EXP / "reg_summary_stage1.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    im1 = heat(axes[0], clf, "roc_auc",
               "分类（K=10，随机折，seed=42）：ROC-AUC", "YlGnBu")
    im2 = heat(axes[1], reg, "r2",
               "回归（K=10，随机折，seed=42）：R²", "YlOrRd")
    # stretch the right edge so the red 'noise floor' label fits
    fig.subplots_adjust(right=0.86, wspace=0.35)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02)
    fig.suptitle("阶段① 选择器 × 模型 矩阵（红框 = 随机特征集的噪声下限）",
                 fontsize=13, y=1.0)
    fig.savefig(OUT / "fig1_selector_model_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig1_selector_model_heatmap.png")

    # ---- paired-vs-noise-floor table (reused for the README) ----
    for tag, df, metric in (("clf", clf, "roc_auc"), ("reg", reg, "r2")):
        perfold = pd.read_csv(EXP / f"{tag}_per_fold_stage1.csv")
        print(f"[{tag}] per-fold rows = {len(perfold)}")


if __name__ == "__main__":
    main()
