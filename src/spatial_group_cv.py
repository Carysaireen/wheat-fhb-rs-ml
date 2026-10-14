"""
空间分组交叉验证对照实验（spatial vs random CV）。

目的
----
在完全相同的预处理与模型下，并排评估两种交叉验证切法：
  scheme A  random_stratified  —— 现状：按行随机分层 5 折（复用封存折文件）
  scheme B  spatial_group      —— 对照：按经纬度聚类分块后整块留出 3 折
两者的差值（A − B）即现有数字中因「样本空间共址」带来的乐观水分。

纪律
----
- 只使用 data/splits/train.csv（132 样本）+ 内部交叉验证；
- 绝不读取 test_locked.csv（33 样本全程封存）；
- 预处理通过 sklearn Pipeline 绑定在模型内，只在训练折 fit，天然无泄漏；
- 分组变量只由经纬度聚类导出，不使用 O_Name。

用法
----
    cd project_v2
    python src/spatial_group_cv.py                # 默认 k=10, seeds=[42,7,2026]
    python src/spatial_group_cv.py --k 8          # 敏感性分析
    python src/spatial_group_cv.py --quick        # 只跑 seed=42（快速验证）

产出
----
所有输出文件名均带 k 后缀（如 gap_table_k10.csv），避免不同 k 互相覆盖。
主口径为 k=10，即 gap_table_k10.csv / summary_metrics_k10.csv。
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from preprocessing import build_preprocessor  # 复用项目既有预处理（折内 fit）
from spatial_folds import recorded_spatial_folds  # 折表优先（见模块说明）

DATA = ROOT / "data"
EXP = ROOT / "experiments" / "2026-10-11_spatial_cv"
OUT = EXP / "outputs"


# ----------------------------------------------------------------------------
# 1. 数据与模型定义
# ----------------------------------------------------------------------------
def load_train():
    """读取训练集，返回 (df, 特征列名)。"""
    df = pd.read_csv(DATA / "splits" / "train.csv", encoding="utf-8-sig")
    feat_cols = df.columns[9:].tolist()  # 前 9 列为元信息
    return df, feat_cols


def build_models():
    """与 traditional_baseline_cv.py 保持一致，保证对照公平。"""
    cls_models = {
        "LR": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "RF": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")),
        ]),
    }
    reg_models = {
        "Ridge": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "SVR": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", SVR(C=1.0)),
        ]),
        "RF": Pipeline([
            ("prep", build_preprocessor(clip_outliers=True)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
    }
    return cls_models, reg_models


# ----------------------------------------------------------------------------
# 2. 空间分块与折分配
# ----------------------------------------------------------------------------
def make_spatial_blocks(df, k, seed=42):
    """按经纬度 KMeans 聚类得到空间块标签（分组变量只来自坐标）。"""
    xy = df[["O_Lat", "O_Lng"]].to_numpy(float)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(xy)


def make_spatial_folds(df, y, k, n_splits=3, seed=42):
    """空间分组折：同一空间块整体归入同一折。返回 (折号, 块号) 数组。

    对外发布版的经纬度已圆整到 1 km，按公开坐标重算 KMeans 会得到不同的
    空间块。k=10 且 3 折时优先采用 data/cv/spatial_folds_k10.csv 里记录的
    块号与折号（见 spatial_folds 模块），其余参数组合按坐标现算。
    """
    rec = recorded_spatial_folds(y, seed, k=k, n_splits=n_splits)
    if rec is not None:
        return rec

    groups = make_spatial_blocks(df, k, seed=seed)
    fold = np.full(len(df), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    xy = df[["O_Lat", "O_Lng"]].to_numpy(float)
    for i, (_, va) in enumerate(sgkf.split(xy, y, groups=groups)):
        fold[va] = i
    assert (fold >= 0).all(), "存在未被分配到折的样本"
    return fold, groups


def audit_folds(df, y, fold, groups, scheme_name, seed, k=None):
    """输出折设计的审计明细（样本数、标签构成、含块数）。"""
    rows = []
    for f in sorted(np.unique(fold)):
        m = fold == f
        rows.append({
            "scheme": scheme_name,
            "seed": seed,
            "k_blocks": k if k is not None else "",
            "fold": int(f),
            "n_val": int(m.sum()),
            "n_healthy": int((y[m] == 0).sum()),
            "n_diseased": int((y[m] == 1).sum()),
            "n_blocks_in_fold": int(len(np.unique(groups[m]))) if groups is not None else "",
        })
    return rows


# ----------------------------------------------------------------------------
# 3. 评估循环
# ----------------------------------------------------------------------------
def eval_classification(df, feat_cols, y, fold, scheme, seed, out_rows):
    X = df[feat_cols].to_numpy(float)
    cls_models, _ = build_models()
    for mn, base in cls_models.items():
        for f in sorted(np.unique(fold)):
            va = fold == f
            m = clone(base)
            m.fit(X[~va], y[~va])
            pred = m.predict(X[va])
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(X[va])[:, 1]
            else:
                prob = m.decision_function(X[va])
            out_rows.append({
                "task": "classification", "scheme": scheme, "seed": seed,
                "model": mn, "fold": int(f), "n_val": int(va.sum()),
                "accuracy": accuracy_score(y[va], pred),
                "f1": f1_score(y[va], pred, zero_division=0),
                "recall": recall_score(y[va], pred, zero_division=0),
                "roc_auc": roc_auc_score(y[va], prob),
            })


def eval_regression(df, feat_cols, reg_fold, scheme, seed, out_rows):
    sick = df[df.DI > 0].copy()
    X = sick[feat_cols].to_numpy(float)
    yr = sick.DI.to_numpy(float)
    _, reg_models = build_models()
    for mn, base in reg_models.items():
        for f in sorted(np.unique(reg_fold)):
            va = reg_fold == f
            m = clone(base)
            m.fit(X[~va], yr[~va])
            pred = m.predict(X[va])
            out_rows.append({
                "task": "regression", "scheme": scheme, "seed": seed,
                "model": mn, "fold": int(f), "n_val": int(va.sum()),
                "r2": r2_score(yr[va], pred),
                "rmse": mean_squared_error(yr[va], pred) ** 0.5,
                "mae": mean_absolute_error(yr[va], pred),
            })


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="空间分组 CV 对照实验")
    ap.add_argument("--k", type=int, default=10, help="空间块数（默认 10）")
    ap.add_argument("--n-splits", type=int, default=3, help="空间分组折数（默认 3）")
    ap.add_argument("--quick", action="store_true", help="只跑 seed=42")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df, feat_cols = load_train()
    y = df["Unnamed: 2"].astype(int).to_numpy()
    seeds = [42] if args.quick else [42, 7, 2026]

    print("=" * 72)
    print(f"空间分组 CV 对照实验  |  训练集 {len(df)} 样本  |  "
          f"标签 {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"空间块数 k={args.k}  空间折数 n_splits={args.n_splits}  seeds={seeds}")
    print("=" * 72)

    metrics_rows, audit_rows, block_rows = [], [], []

    # 现状分类折（复用封存文件，不重新生成）
    cls_fold_file = pd.read_csv(DATA / "cv" / "classification_folds.csv")
    reg_fold_file = pd.read_csv(DATA / "cv" / "regression_folds.csv")
    fold_A_cls = cls_fold_file["fold"].to_numpy()

    for seed in seeds:
        print(f"\n--- seed = {seed} ---")

        # ---- scheme A：现状随机分层折（对所有 seed 都复用同一份封存折）----
        eval_classification(df, feat_cols, y, fold_A_cls, "A_random", seed, metrics_rows)
        audit_rows += audit_folds(df, y, fold_A_cls, None, "A_random", seed)

        # 回归折（现状 KFold）同样复用
        reg_fold_A = reg_fold_file["fold"].to_numpy()
        eval_regression(df, feat_cols, reg_fold_A, "A_random", seed, metrics_rows)

        # ---- scheme B：空间分组折（逐 seed 现场生成）----
        fold_B, groups = make_spatial_folds(df, y, args.k, args.n_splits, seed=seed)
        for i, g in enumerate(groups):
            block_rows.append({"seed": seed, "k": args.k, "row_index": i, "block": int(g)})
        audit_rows += audit_folds(df, y, fold_B, groups, "B_spatial", seed, k=args.k)
        eval_classification(df, feat_cols, y, fold_B, "B_spatial", seed, metrics_rows)

        # 空间折下的回归：对 DI>0 子集重新做空间分组（保证同址样本同折）
        sick_mask = (df.DI > 0).to_numpy()
        sick_df = df[sick_mask].reset_index(drop=True)
        y_sick = sick_df["Unnamed: 2"].astype(int).to_numpy()
        fold_B_sick = make_spatial_folds(sick_df, y_sick, args.k, args.n_splits, seed=seed)[0]
        eval_regression(sick_df, feat_cols, fold_B_sick, "B_spatial", seed, metrics_rows)

        # 折设计审计打印
        for r in audit_rows:
            if r["seed"] == seed and r["scheme"] == "B_spatial":
                print(f"  [B_spatial] fold{r['fold']}: {r['n_val']:2d} 样本 "
                      f"[0:{r['n_healthy']:2d}, 1:{r['n_diseased']:2d}] "
                      f"含 {r['n_blocks_in_fold']} 块")

    # ---------------- 保存明细 ----------------
    tag = f"k{args.k}"
    per_fold = pd.DataFrame(metrics_rows)
    per_fold.to_csv(OUT / f"per_fold_metrics_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(OUT / f"fold_design_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(block_rows).to_csv(OUT / f"spatial_blocks_{tag}.csv", index=False, encoding="utf-8-sig")

    # ---------------- 汇总（按 scheme × model 聚合）----------------
    agg = {}
    for task, mcols in [("classification", ["accuracy", "f1", "recall", "roc_auc"]),
                        ("regression", ["r2", "rmse", "mae"])]:
        sub = per_fold[per_fold.task == task]
        g = sub.groupby(["scheme", "model"], as_index=False)[mcols].agg(["mean", "std"])
        g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns]
        g = g.reset_index()
        g["task"] = task
        agg[task] = g
    summary = pd.concat(agg.values(), ignore_index=True)
    summary.to_csv(OUT / f"summary_metrics_{tag}.csv", index=False, encoding="utf-8-sig")

    # ---------------- ★ 核心产出：gap_table ----------------
    gap_rows = []
    for task, main_metric in [("classification", "accuracy"), ("regression", "r2")]:
        sub = summary[summary.task == task]
        for model in sorted(sub.model.unique()):
            a = sub[(sub.scheme == "A_random") & (sub.model == model)]
            b = sub[(sub.scheme == "B_spatial") & (sub.model == model)]
            if len(a) == 0 or len(b) == 0:
                continue
            a_mean = a[f"{main_metric}_mean"].values[0]
            b_mean = b[f"{main_metric}_mean"].values[0]
            gap_rows.append({
                "task": task, "model": model, "metric": main_metric,
                "A_random_mean": round(float(a_mean), 4),
                "B_spatial_mean": round(float(b_mean), 4),
                "gap_A_minus_B": round(float(a_mean - b_mean), 4),
            })
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(OUT / f"gap_table_{tag}.csv", index=False, encoding="utf-8-sig")

    # ---------------- 终端打印 ----------------
    print("\n" + "=" * 72)
    print("★ 核心结论：随机折 vs 空间分组折")
    print("=" * 72)
    print(gap.to_string(index=False))
    print("\n（gap > 0 表示随机折更乐观，差值即「空间泄漏」带来的水分）")
    print(f"\n产出已写入：{OUT}")
    print(f"  per_fold_metrics_{tag}.csv / summary_metrics_{tag}.csv / gap_table_{tag}.csv")
    print(f"  fold_design_{tag}.csv / spatial_blocks_{tag}.csv")


if __name__ == "__main__":
    main()
