"""
空间验证章节图表生成脚本（plot_spatial_validation.py）。

输入：experiments/2026-10-11_spatial_cv/outputs/ 下的实验结果 CSV
输出：results/spatial_validation/ 下的正式图表（PNG，dpi=200）+ 说明 README

产出图表
--------
fig1_gap_classification.png   分类：随机折 vs 空间分组折 准确率对照 + gap
fig2_gap_regression.png       回归：随机折 vs 空间分组折 R² 对照 + gap
fig3_k_sensitivity.png        k=8/10/12 敏感性：gap 是否稳健
fig4_fold_design.png          折设计审计：折大小与标签构成
fig5_spatial_map.png          空间地图：块归属 + 健康/患病分布

用法：
    cd project_v2
    python src/plot_spatial_validation.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.cluster import KMeans

from spatial_group_cv import load_train, make_spatial_folds

# ----------------------------------------------------------------------------
# 全局样式（沿用项目既有风格：微软雅黑 + 负号修正 + 淡网格）
# ----------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

EXP = ROOT / "experiments" / "2026-10-11_spatial_cv" / "outputs"
OUT = ROOT / "results" / "spatial_validation"
OUT.mkdir(parents=True, exist_ok=True)

C_RANDOM = "#4c78a8"    # 蓝：随机折
C_SPATIAL = "#e45756"   # 红：空间分组折
C_HEALTHY = "#54a24b"   # 绿：健康
C_DISEASED = "#e45756"  # 红：患病
C_GAP = "#f58518"       # 橙：差值


# ----------------------------------------------------------------------------
# figure 1 / 2：gap 柱状图（分类 & 回归）
# ----------------------------------------------------------------------------
def plot_gap(gap: pd.DataFrame, task: str, metric_label: str, title: str,
             fname: str, ylim=None):
    sub = gap[gap.task == task].copy()
    models = sub.model.tolist()
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    b1 = ax.bar(x - w / 2, sub.A_random_mean, w, label="A 随机分层折（现状）",
                color=C_RANDOM, edgecolor="white")
    b2 = ax.bar(x + w / 2, sub.B_spatial_mean, w, label="B 空间分组折（本实验）",
                color=C_SPATIAL, edgecolor="white")

    # 数值标注
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=9, xytext=(0, 3 if h >= 0 else -3),
                        textcoords="offset points")

    # gap 箭头标注（贴在两根柱子中缝偏外侧，避免与柱顶数值重叠）
    for i, (a, b) in enumerate(zip(sub.A_random_mean, sub.B_spatial_mean)):
        xg = i + w * 0.52
        ax.annotate("", xy=(xg, b), xytext=(xg, a),
                    arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=1.7))
        ax.text(xg, (a + b) / 2 + (ylim[1] - ylim[0]) * 0.02 if ylim else (a + b) / 2,
                f"gap {a - b:+.3f}",
                color=C_GAP, fontsize=8, va="center", ha="left",
                fontweight="bold", rotation=90,
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec=C_GAP, alpha=0.95, lw=0.8))

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(title, fontsize=12.5, pad=12)
    ax.legend(fontsize=9.5, loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    if ylim:
        ax.set_ylim(*ylim)
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print(f"  [ok] {fname}")


# ----------------------------------------------------------------------------
# figure 3：k 敏感性
# ----------------------------------------------------------------------------
def plot_k_sensitivity():
    tables = {"k=8": "gap_table_k8.csv", "k=10 (主口径)": "gap_table_k10.csv",
              "k=12": "gap_table_k12.csv"}
    cls_rows, reg_rows = [], []
    for label, fn in tables.items():
        g = pd.read_csv(EXP / fn)
        for _, r in g[g.task == "classification"].iterrows():
            cls_rows.append({"k_config": label, "model": r.model, "gap": r.gap_A_minus_B})
        for _, r in g[g.task == "regression"].iterrows():
            reg_rows.append({"k_config": label, "model": r.model, "gap": r.gap_A_minus_B})
    cls = pd.DataFrame(cls_rows)
    reg = pd.DataFrame(reg_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    k_order = list(tables.keys())
    x = np.arange(len(k_order))
    w = 0.25
    palette = {"LR": "#4c78a8", "RF": "#54a24b", "SVM": "#b279a2",
               "Ridge": "#f58518", "SVR": "#e45756"}

    for ax, df, metric, title in [
        (axes[0], cls, "accuracy", "分类：准确率 gap（A_random − B_spatial）"),
        (axes[1], reg, "r2", "回归：R² gap（A_random − B_spatial）"),
    ]:
        models = sorted(df.model.unique())
        for i, m in enumerate(models):
            vals = [df[(df.k_config == kc) & (df.model == m)].gap.values[0]
                    for kc in k_order]
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
                          label=m, color=palette.get(m, "#888"),
                          edgecolor="white")
            for r in bars:
                h = r.get_height()
                ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=7.2,
                            xytext=(0, 1.5), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(k_order, fontsize=10)
        ax.set_ylabel("gap (越大＝空间泄漏越明显)", fontsize=10)
        ax.set_title(title, fontsize=11.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("空间块数 k 的敏感性分析：三种 k 下结论方向一致（分类 gap 恒为正）",
                 fontsize=12.5, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_k_sensitivity.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig3_k_sensitivity.png")


# ----------------------------------------------------------------------------
# figure 4：折设计审计（折大小 + 标签构成）
# ----------------------------------------------------------------------------
def plot_fold_design():
    fd = pd.read_csv(EXP / "fold_design_k10.csv")
    fd42 = fd[fd.seed == 42]
    a = fd42[fd42.scheme == "A_random"].sort_values("fold")
    b = fd42[fd42.scheme == "B_spatial"].sort_values("fold")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5),
                             gridspec_kw={"width_ratios": [1, 1]})

    # 左：A_random
    ax = axes[0]
    x = np.arange(len(a))
    ax.bar(x, a.n_healthy, 0.6, label="健康 (0)", color=C_HEALTHY, edgecolor="white")
    ax.bar(x, a.n_diseased, 0.6, bottom=a.n_healthy, label="患病 (1)",
           color=C_DISEASED, edgecolor="white")
    for i, r in enumerate(a.itertuples()):
        ax.text(i, r.n_val + 0.7, f"{r.n_val}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"折{i}" for i in a.fold], fontsize=10)
    ax.set_ylabel("验证折样本数", fontsize=10.5)
    ax.set_title("A 随机分层 5 折（现状）\n折大小均匀、标签比例一致", fontsize=11.5)
    ax.set_ylim(0, a.n_val.max() * 1.25)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # 右：B_spatial
    ax = axes[1]
    x = np.arange(len(b))
    ax.bar(x, b.n_healthy, 0.6, label="健康 (0)", color=C_HEALTHY, edgecolor="white")
    ax.bar(x, b.n_diseased, 0.6, bottom=b.n_healthy, label="患病 (1)",
           color=C_DISEASED, edgecolor="white")
    for i, r in enumerate(b.itertuples()):
        ax.text(i, r.n_val + 0.7, f"{r.n_val}\n({r.n_blocks_in_fold}块)",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"折{i}" for i in b.fold], fontsize=10)
    ax.set_ylabel("验证折样本数", fontsize=10.5)
    ax.set_title("B 空间分组 3 折（本实验）\n折大小不匀、标签比例波动＝真实空间异质性",
                fontsize=11.5)
    ax.set_ylim(0, b.n_val.max() * 1.25)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("折设计审计（seed=42, k=10）：整块留出打破了随机折的\"均匀假象\"",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_fold_design.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig4_fold_design.png")


# ----------------------------------------------------------------------------
# figure 5：空间地图（块归属 + 标签）
# ----------------------------------------------------------------------------
def plot_spatial_map():
    df, _ = load_train()
    y = df["Unnamed: 2"].astype(int).to_numpy()
    lat = df["O_Lat"].to_numpy(float)
    lng = df["O_Lng"].to_numpy(float)

    # 复现主口径：k=10, seed=42 的块归属
    groups = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(
        np.c_[lat, lng])
    fold, _ = make_spatial_folds(df, y, k=10, n_splits=3, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    # 左：块着色
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for blk in sorted(np.unique(groups)):
        m = groups == blk
        ax.scatter(lng[m], lat[m], s=52, color=cmap(blk % 10),
                   edgecolor="white", lw=0.6, label=f"块{blk} (n={m.sum()})")
        # 块质心标注
        ax.annotate(f"B{blk}", (lng[m].mean(), lat[m].mean()),
                    fontsize=8.5, fontweight="bold", ha="center", va="center",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="gray", alpha=0.75, lw=0.6))
    ax.set_xlabel("经度 O_Lng (deg)", fontsize=10.5)
    ax.set_ylabel("纬度 O_Lat (deg)", fontsize=10.5)
    ax.set_title("空间块归属（KMeans k=10, seed=42）\n每块样本整块进同一折",
                 fontsize=11.5)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left", framealpha=0.92,
              markerscale=0.9, handletextpad=0.3, columnspacing=0.8)
    ax.grid(alpha=0.2)

    # 右：标签着色
    ax = axes[1]
    mh = y == 0
    md = y == 1
    ax.scatter(lng[mh], lat[mh], s=52, color=C_HEALTHY, edgecolor="white",
               lw=0.6, label=f"健康 (n={mh.sum()})", alpha=0.9)
    ax.scatter(lng[md], lat[md], s=52, color=C_DISEASED, edgecolor="white",
               lw=0.6, label=f"患病 (n={md.sum()})", alpha=0.9, marker="^")
    ax.set_xlabel("经度 O_Lng (deg)", fontsize=10.5)
    ax.set_ylabel("纬度 O_Lat (deg)", fontsize=10.5)
    ax.set_title("标签的空间分布\n地理位置与病害状态强耦合（存在全健康块 / 全患病块）",
                 fontsize=11.5)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.suptitle("★ 空间泄漏的根源：样本高度共址 + 标签空间聚集", fontsize=13.5, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_spatial_map.png", dpi=200)
    plt.close(fig)
    print("  [ok] fig5_spatial_map.png")


# ----------------------------------------------------------------------------
# 说明文档
# ----------------------------------------------------------------------------
def write_readme(gap: pd.DataFrame):
    cls = gap[gap.task == "classification"]
    reg = gap[gap.task == "regression"]
    txt = f"""# results/spatial_validation — 空间验证章节素材

> 生成脚本：`src/plot_spatial_validation.py`
> 数据来源：`experiments/2026-10-11_spatial_cv/outputs/`（只用训练集内部 CV，未读封存测试集）
> 生成日期：2026-10-11

## 图表清单

| 文件 | 内容 | 论文用途 |
|---|---|---|
| `fig1_gap_classification.png` | 分类准确率：随机折 vs 空间分组折 + gap | 空间泄漏程度（分类） |
| `fig2_gap_regression.png` | 回归 R²：随机折 vs 空间分组折 + gap | 空间泄漏程度（回归） |
| `fig3_k_sensitivity.png` | k=8/10/12 三种块数的 gap 对照 | 结论稳健性证据 |
| `fig4_fold_design.png` | 折大小与标签构成审计 | 方法透明度 |
| `fig5_spatial_map.png` | 空间块归属 + 标签空间分布 | 现象解释（为何有泄漏） |

## 核心数字（k=10, 3 折, 3 seeds 均值）

**分类（准确率）**

| 模型 | A 随机折 | B 空间折 | gap |
|---|---|---|---|
""" + "\n".join(
        f"| {r.model} | {r.A_random_mean:.3f} | {r.B_spatial_mean:.3f} | {r.gap_A_minus_B:+.3f} |"
        for r in cls.itertuples()
    ) + """

**回归（R²）**

| 模型 | A 随机折 | B 空间折 | gap |
|---|---|---|---|
""" + "\n".join(
        f"| {r.model} | {r.A_random_mean:.3f} | {r.B_spatial_mean:.3f} | {r.gap_A_minus_B:+.3f} |"
        for r in reg.itertuples()
    ) + """

## 写作要点（诚实性）

1. **两套数字必须并报**：随机 CV 偏乐观（空间泄漏），空间分组 CV 偏保守（外推困难），真相介于两者之间。
2. **不要只强调 gap 大**：空间折下 R² 转负，部分来自观测空间过小（0.68°）导致的外推困难，
   这正是 Roberts et al. 2017 警告的保守方向。
3. **领域先例**：Morellos et al. 2025 同域对照（跨田块 R²≤0.71 vs 田块内 R²≤1.00），
   证明空间泄漏量级巨大，本结果与之同类。
4. **引用**：Roberts et al. 2017 (Ecography 40:913-929)；Morellos et al. 2025 (CEA 231:109967)。

## 方法术语与论文可引用表述

本项目内部脚本/目录名为 `spatial_group_cv`（“空间分组交叉验证”），但在遥感/生态文献中的**规范术语**是
**空间块交叉验证（spatial block cross-validation, blockCV）**，属于 **空间交叉验证（spatial cross-validation, spatial CV）** 家族。

| 术语 | 含义 | 与本项目的关系 |
|---|---|---|
| Spatial CV（空间交叉验证） | 泛指任何显式考虑空间依赖的 CV | 总称 |
| **Block CV / blocked CV（空间块交叉验证）** | 按空间块切分、整块留出 | ★ 本实验做法；论文首选术语 |
| Spatial group CV（空间分组交叉验证） | 强调分组变量由空间坐标导出 | 本项目内部命名 `spatial_group_cv` |
| Leave-one-block-out CV | 每次只留出 1 个块 | 本项目留出多块（3 折），为其变体 |
| Buffered CV（带缓冲带的块 CV） | 留出块周围再剔除一圈训练样本 | 本项目未启用（见 config.yaml → no_buffer） |

**论文 Methods 可引用表述（可直接采用）**：

> To quantify the optimism caused by spatial autocorrelation, we compared random stratified
> 5-fold cross-validation against **spatial block cross-validation**. Spatial blocks were derived
> by K-means clustering on geographic coordinates (k = 8/10/12; k = 10 as primary), and folds were
> built with StratifiedGroupKFold (3 folds) so that entire blocks were held out while class
> proportions were preserved. No buffer was applied at this stage. The difference between the two
> schemes (A − B) serves as a diagnostic of spatial leakage.

**可引用文献**：

- Roberts et al. 2017, *Ecography* 40(8):913-929, DOI 10.1111/ecog.02881 — block CV 奠基综述
- Morellos et al. 2025, *Comput. Electron. Agric.* 231:109967 — 同域空间验证实证（跨田块 R²≤0.71 vs 田块内 ≤1.00）

## 复现

```bash
cd project_v2
python src/spatial_group_cv.py          # 先跑实验（生成 outputs/）
python src/plot_spatial_validation.py   # 再生成图表
```
"""
    (OUT / "README.md").write_text(txt, encoding="utf-8")
    print("  [ok] README.md")


def main():
    print("=" * 66)
    print("空间验证章节图表生成")
    print("=" * 66)
    gap = pd.read_csv(EXP / "gap_table_k10.csv")

    plot_gap(gap, "classification", "平均准确率 (Accuracy)",
             "分类：随机分层折 vs 空间分组折",
             "fig1_gap_classification.png", ylim=(-0.05, 1.12))
    plot_gap(gap, "regression", "平均决定系数 (R²)",
             "回归：随机分层折 vs 空间分组折",
             "fig2_gap_regression.png", ylim=(-2.25, 0.62))
    plot_k_sensitivity()
    plot_fold_design()
    plot_spatial_map()
    write_readme(gap)

    print(f"\n产出目录：{OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
