"""
汇总全项目「分类 / 回归」最佳组合排行榜  v2
==============================================
v1 的问题：把擂主档案（metric=accuracy，单种子 s42）和阶段④/⑤（metric=roc_auc / r2）
直接混在一列里排序 → 两个指标不等价，混排会得出"current_2+LDA = 0.8331 全场第一"这种
**跨指标、单种子**的伪结论。v2 修正为：

    · 分类：同时给出 accuracy 列与 roc_auc 列，**分列排序**，不混用
    · 回归：只有 r2 一个口径，问题不存在
    · 所有"单种子"结果都标注 n_seeds，避免与多种子均值直接比较
    · 每种组合额外报「跨分区种子的最小/均值」以识别"运气型高分"

主口径：空间折（blockCV，KMeans(k=10)，分区种子 42）为**主裁判**，随机折为**辅裁判**。

输出
    results/stage_summary/best_combinations_clf.csv
    results/stage_summary/best_combinations_reg.csv
    results/stage_summary/model_k_curve_spatial_clf.csv
    results/stage_summary/model_k_curve_spatial_reg.csv
    results/stage_summary/_best_combo_stdout.txt
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "stage_summary"

SPATIAL_SEEDS = ("B_spatial3_s42", "B_spatial3_s7", "B_spatial3_s2026")


# ---------------------------------------------------------------------------
def load_profiles() -> pd.DataFrame:
    return pd.read_csv(RES / "champion_profile" / "champion_profile_full.csv")


def stage5(task: str) -> pd.DataFrame:
    return pd.read_csv(RES / "stage5_tuned" / f"challenge_{task}.csv")


def stage4(task: str) -> pd.DataFrame:
    p = RES / "selector_zoo_spatial" / f"leaderboard_spatial_{task}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
def build(task: str) -> pd.DataFrame:
    """
    统一成：source / selector / model / k / n_features / n_seeds /
            spatial_s42 / spatial_mean3 / spatial_min3 / random / note
    分类与回归都套同一结构（分类只能填 accuracy，另见 attach_rocauc）。
    """
    rows: list[dict] = []

    # 1) 阶段⑤（调优参数；同时有 spatial(s42) 与 random）
    s5 = stage5(task)
    for _, x in s5.iterrows():
        rows.append(
            dict(
                source="stage5_tuned(调优)",
                selector=x["selector"],
                model=x["model"],
                k=int(x["k"]),
                n_features=int(x["n_features"]),
                n_seeds=1,
                spatial_s42=float(x["spatial"]),
                random=float(x["random"]),
                note=f"Δ vs 擂主 {x['delta_vs_champ']:+.4f} ({x['verdict']})",
            )
        )

    # 2) 阶段④（默认参数；leaders 只有空间折 + 随机折）
    s4 = stage4(task)
    if not s4.empty:
        sp = [c for c in s4.columns if c.startswith("spatial_")][0]
        rd = [c for c in s4.columns if c.startswith("random_")][0]
        for _, x in s4.iterrows():
            rows.append(
                dict(
                    source="stage4_default(默认)",
                    selector=x["selector"],
                    model=x["model"],
                    k=int(x["k"]),
                    n_features=int(x["k"]),
                    n_seeds=1,
                    spatial_s42=float(x[sp]),
                    random=float(x[rd]),
                    note="默认参数",
                )
            )

    # 3) 擂主档案 current_2（唯一指定标杆；有 3 个空间分区种子 → 可报均值/最小值）
    prof = load_profiles()
    prof = prof[(prof.task == task) & (prof.set == "current_2")]
    for model, g in prof.groupby("model"):
        sp = g[g.scheme.isin(SPATIAL_SEEDS)].set_index("scheme")["mean"]
        rd = g[g.scheme == "A_random5"]["mean"]
        s42 = float(sp.get("B_spatial3_s42", np.nan))
        rows.append(
            dict(
                source="champion_current_2(擂主标杆)",
                selector="(预设)",
                model=model,
                k=2,
                n_features=2,
                n_seeds=3,
                spatial_s42=s42,
                spatial_mean3=float(sp.mean()) if len(sp) else np.nan,
                spatial_min3=float(sp.min()) if len(sp) else np.nan,
                random=float(rd.iloc[0]) if len(rd) else np.nan,
                note="2 特征预设集",
            )
        )

    df = pd.DataFrame(rows)
    for c in ("spatial_s42", "spatial_mean3", "spatial_min3", "random"):
        if c not in df.columns:
            df[c] = np.nan

    # 单种子来源没有 mean3/min3 → 用 s42 占位，并靠 n_seeds 区分
    df["spatial_mean3"] = df["spatial_mean3"].fillna(df["spatial_s42"])
    df["spatial_min3"] = df["spatial_min3"].fillna(df["spatial_s42"])

    # 按 s42 空间折降序（主口径）
    df = df.sort_values("spatial_s42", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank_s42", np.arange(1, len(df) + 1))
    return df


def attach_rocauc_from_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    分类的 roc_auc 只在 baseline_alignment（4 个特征集 × 若干模型）和阶段⑤里存在。
    这里把阶段⑤的 roc_auc 补进汇总表（阶段④只有 accuracy）。
    """
    s5 = stage5("clf")
    key = s5.set_index(["selector", "model", "k"])["spatial"]
    roc = []
    for _, x in df.iterrows():
        k = (x["selector"], x["model"], int(x["k"]))
        roc.append(float(key.get(k, np.nan)) if k in key.index else np.nan)
    df = df.copy()
    df["spatial_roc_auc_s42"] = roc
    return df


def model_k_curve(task: str) -> pd.DataFrame:
    """
    模型级「K × 空间折」：只用阶段⑤（唯一同时有 4 个 K + 空间折的矩阵）。
    对每个 (model, k) 跨全部选择器求均值/最优/最差 → 看「模型 × K」联合效应。
    """
    s5 = stage5(task)
    g = (
        s5.groupby(["model", "k"])
        .agg(
            spatial_mean=("spatial", "mean"),
            spatial_max=("spatial", "max"),
            spatial_min=("spatial", "min"),
            random_mean=("random", "mean"),
            n=("spatial", "size"),
        )
        .reset_index()
        .sort_values(["model", "k"])
    )
    best = g.loc[g.groupby("model").spatial_mean.idxmax()][["model", "k"]].rename(
        columns={"k": "best_k"}
    )
    g = g.merge(best, on="model", how="left")
    spread = (
        g.pivot(index="model", columns="k", values="spatial_mean")
        .max(axis=1)
        .sub(g.pivot(index="model", columns="k", values="spatial_mean").min(axis=1))
        .rename("k_spread")
    )
    g = g.merge(spread, on="model", how="left")
    return g


# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()

    def emit(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        buf.write(s + "\n")

    emit("=" * 100)
    emit("全项目最佳组合汇总 v2 —— 主口径：空间折（blockCV, KMeans k=10, 分区种子 42）")
    emit("★ 分类的 accuracy 与 roc_auc 是两套不同指标，本表分列呈现，不做跨指标混排")
    emit("=" * 100)

    # ---------------- 分类 ----------------
    clf = attach_rocauc_from_align(build("clf"))
    clf.to_csv(OUT / "best_combinations_clf.csv", index=False, encoding="utf-8-sig")

    emit()
    emit("-" * 100)
    emit("[分类] 全部来源汇总（按空间折 accuracy@s42 排序）")
    emit("-" * 100)
    emit(f"汇总行数：{len(clf)}")
    emit()
    emit("★ TOP 20（列 n 表示该行用了几个分区种子；n=1 表示只有 s42）")
    cols = ["rank_s42", "source", "selector", "model", "k", "n", "spatial_mean3",
            "spatial_min3", "random", "spatial_roc_auc_s42"]
    t = clf.rename(columns={"n_seeds": "n"})[cols].head(20).copy()
    for c in ["spatial_mean3", "spatial_min3", "random", "spatial_roc_auc_s42"]:
        t[c] = t[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "   -  ")
    emit(t.to_string(index=False))

    emit()
    emit("★ 各来源头名（并列出来头名、以及种子间最差的那个数，用于识别'运气型高分'）")
    for src in [
        "champion_current_2(擂主标杆)",
        "stage4_default(默认)",
        "stage5_tuned(调优)",
    ]:
        sub = clf[clf.source == src]
        if sub.empty:
            continue
        b = sub.iloc[0]
        emit(
            f"  [{src:26s}] {b['selector']:>5s}+{b['model']:<6s} K={int(b['k']):<3d} "
            f"acc(s42)={b['spatial_s42']:.4f}  acc(种子均值)={b['spatial_mean3']:.4f}  "
            f"acc(最差种子)={b['spatial_min3']:.4f}  随机折={b['random']:.4f}  n={int(b['n_seeds'])}"
        )

    # ---------------- 分类：只看 3 种子可靠的（擂主档案 2 特征） ----------------
    emit()
    emit("★ 【只看 3 个分区种子都测过的】current_2（2 特征）各模型 —— 这是最可靠的一批数字")
    c2 = clf[clf.n_seeds == 3].sort_values("spatial_mean3", ascending=False)
    emit(
        c2[["model", "spatial_mean3", "spatial_min3", "random", "note"]]
        .assign(
            spatial_mean3=lambda d: d.spatial_mean3.map("{:.4f}".format),
            spatial_min3=lambda d: d.spatial_min3.map("{:.4f}".format),
            random=lambda d: d.random.map("{:.4f}".format),
        )
        .to_string(index=False)
    )

    # ---------------- 分类：K 曲线 ----------------
    emit()
    emit("★ 模型级「K × 空间折 accuracy」曲线（阶段⑤，跨全部选择器求均值）—— 分类")
    g = model_k_curve("clf")
    g.to_csv(OUT / "model_k_curve_spatial_clf.csv", index=False, encoding="utf-8-sig")
    piv = g.pivot(index="model", columns="k", values="spatial_mean").round(4)
    piv["最优K"] = g.groupby("model").best_k.first()
    piv["K间极差"] = g.groupby("model").k_spread.first().round(4)
    piv = piv.sort_values("最优K")
    emit(piv.to_string())

    # ---------------- 回归 ----------------
    reg = build("reg")
    reg.to_csv(OUT / "best_combinations_reg.csv", index=False, encoding="utf-8-sig")

    emit()
    emit("-" * 100)
    emit("[回归] 全部来源汇总（只有 r2 一个口径，按空间折 r2@s42 排序）")
    emit("-" * 100)
    emit(f"汇总行数：{len(reg)}")
    emit()
    emit("★ TOP 20")
    cols_r = ["rank_s42", "source", "selector", "model", "k", "n", "spatial_mean3",
              "spatial_min3", "random"]
    t = reg.rename(columns={"n_seeds": "n"})[cols_r].head(20).copy()
    for c in ["spatial_mean3", "spatial_min3", "random"]:
        t[c] = t[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "   -  ")
    emit(t.to_string(index=False))

    emit()
    emit("★ 各来源头名")
    for src in [
        "champion_current_2(擂主标杆)",
        "stage4_default(默认)",
        "stage5_tuned(调优)",
    ]:
        sub = reg[reg.source == src]
        if sub.empty:
            continue
        b = sub.iloc[0]
        emit(
            f"  [{src:26s}] {b['selector']:>5s}+{b['model']:<6s} K={int(b['k']):<3d} "
            f"r2(s42)={b['spatial_s42']:.4f}  r2(种子均值)={b['spatial_mean3']:.4f}  "
            f"r2(最差种子)={b['spatial_min3']:.4f}  随机折={b['random']:.4f}  n={int(b['n_seeds'])}"
        )

    emit()
    emit("★ 【只看 3 个分区种子】current_2（2 特征）各模型回归")
    r2 = reg[reg.n_seeds == 3].sort_values("spatial_mean3", ascending=False)
    emit(
        r2[["model", "spatial_mean3", "spatial_min3", "random"]]
        .assign(
            spatial_mean3=lambda d: d.spatial_mean3.map("{:.4f}".format),
            spatial_min3=lambda d: d.spatial_min3.map("{:.4f}".format),
            random=lambda d: d.random.map("{:.4f}".format),
        )
        .to_string(index=False)
    )

    emit()
    emit("★ 模型级「K × 空间折 r2」曲线（阶段⑤，跨全部选择器求均值）—— 回归")
    g = model_k_curve("reg")
    g.to_csv(OUT / "model_k_curve_spatial_reg.csv", index=False, encoding="utf-8-sig")
    piv = g.pivot(index="model", columns="k", values="spatial_mean").round(4)
    piv["最优K"] = g.groupby("model").best_k.first()
    piv["K间极差"] = g.groupby("model").k_spread.first().round(4)
    piv = piv.sort_values("最优K")
    emit(piv.to_string())

    (OUT / "_best_combo_stdout.txt").write_text(buf.getvalue(), encoding="utf-8")
    emit()
    emit("[done] 已写出 best_combinations_{clf,reg}.csv / model_k_curve_spatial_{clf,reg}.csv")


if __name__ == "__main__":
    main()
