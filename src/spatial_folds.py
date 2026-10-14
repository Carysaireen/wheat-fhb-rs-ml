"""空间折的记录读取。

对外发布版的经纬度圆整到 1 km，KMeans 在圆整坐标下会给出不同的空间块
（seed=42 时约 10% 样本换块），空间折因此无法按公开坐标重算。公开发布的
data/cv/spatial_folds_k10.csv 由 tools/make_release_data.py 用高精度坐标
导出，本模块按调用场景取用其中的块号与折号。

三种形态的样本集与分层变量都不同，折号不可混用：
    fold_clf_s*     132 行 + 分类标签        擂主档案、对齐基准、阶段⑤分类、阶段⑥
    fold_reg132_s*  132 行 + DI 三分位箱     选择器动物园 full、阶段⑤回归
    fold_reg116_s*  116 行（DI>0）+ 分类标签 动物园 stage4、P0-② 的回归轨

项目侧不发布该折表。文件缺失或场景不匹配时返回 None，调用点按经纬度现算，
行为与引入本模块之前相同。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SPATIAL_K = 10
SPATIAL_N_SPLITS = 3

# 样本集规模 -> 折列后缀。132 行分两种分层方式，按分层变量的取值个数区分
N_CLF, N_REG116 = 132, 116


def recorded_spatial_folds(y_strat, seed, k=SPATIAL_K, n_splits=SPATIAL_N_SPLITS):
    """返回 (fold, blocks)，无可用记录时返回 None。"""
    if k != SPATIAL_K or n_splits != SPATIAL_N_SPLITS:
        return None
    path = DATA / "cv" / f"spatial_folds_k{k}.csv"
    if not path.exists():
        return None

    rec = pd.read_csv(path)
    n, n_uniq = len(y_strat), len(np.unique(y_strat))
    if n == N_CLF and n_uniq <= 2:
        col_b, col_f = f"block132_s{seed}", f"fold_clf_s{seed}"
    elif n == N_CLF:
        col_b, col_f = f"block132_s{seed}", f"fold_reg132_s{seed}"
    elif n == N_REG116:
        col_b, col_f = f"block116_s{seed}", f"fold_reg116_s{seed}"
    else:
        return None

    if not {col_b, col_f} <= set(rec.columns):
        return None
    sub = rec[rec[col_f].notna()]
    if len(sub) != n:
        return None
    return sub[col_f].to_numpy(int), sub[col_b].to_numpy(int)
