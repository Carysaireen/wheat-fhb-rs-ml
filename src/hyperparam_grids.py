"""Hyper-parameter grids + model factories for the hyper-parameter sweep.

See ``experiments/2026-10-13_hyperparam_sweep/config.yaml`` for the design.

Design idea
-----------
ONE selector (relieff), ONE K (10), ONE protocol (random 5-fold), ONE seed (42)
-> **the model hyper-parameters are the only thing that varies**. This isolates
the question "were the stage-1/2/4 oddities (XGBoost last, PLSR collapsing with
K, SVR last) caused by un-tuned hyper-parameters, or are they properties of the
models themselves?".

Why the grids are small and hand-picked
---------------------------------------
A blind/random search over a 132-row dataset would mostly fit noise and then
reward whichever setting got lucky on the 5 validation folds (winner's curse).
Instead, for every model we sweep the ONE parameter that theory says matters
most (regularisation strength, tree depth, number of components, neighbourhood
size, ...), and we keep the grid small so the whole sweep stays interpretable.

Every grid's FIRST entry is exactly the stage-1 default, so the sweep contains
its own baseline and "best vs default" is a paired comparison on identical
folds.

No StandardScaler is baked in here: the driver fits ONE scaler per fold on the
selected training columns and feeds the SAME scaled matrix to every setting.
"""
from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from run_selector_zoo import PLSDA

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:                                   # pragma: no cover
    HAS_XGB = False

SEED = 42


# --------------------------------------------------------------------------- #
# model factories (no scaler -- the driver scales once per fold)
# --------------------------------------------------------------------------- #
def build_clf(name, **p):
    if name == "LR":
        return LogisticRegression(max_iter=2000, C=p["C"], random_state=SEED)
    # ★ probability=True 必须与全项目其它脚本（run_selector_zoo / spatial_group_cv /
    #   traditional_baseline_cv / run_baseline_alignment / evaluate_feature_sets_cv /
    #   run_fs_cs_experiment / nested_traditional_tuning）保持一致。
    #   原因：clf_scores 的 roc_auc 优先用 predict_proba；probability=False 时
    #   SVC 没有 predict_proba，会退化成 decision_function。两条路径给出的
    #   排序**不完全相同**（实测同一折同一特征集上能差 6e-3），于是
    #   "超参扫描 / 擂主档案"里的 SVM roc_auc 会与 selector-zoo 系列对不上。
    #   这是 2026-10-13 阶段⑤ 自校验抓出来的口径不一致，已统一为 True。
    if name == "SVM":                       # RBF kernel, the project's default
        return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"],
                   probability=True, random_state=SEED)
    if name == "SVMLin":                    # linear kernel
        return SVC(kernel="linear", C=p["C"], probability=True, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], class_weight="balanced",
            random_state=SEED, n_jobs=-1)
    if name == "ET":
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], class_weight="balanced",
            random_state=SEED, n_jobs=-1)
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver=p["solver"],
                                          shrinkage=p["shrinkage"])
    if name == "PLSDA":
        return PLSDA(n_components=p["n_components"])
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"],
                                    weights=p["weights"])
    if name == "NB":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if name == "XGB":
        return XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0,
            eval_metric="logloss")
    raise KeyError(name)


def build_reg(name, **p):
    if name == "Ridge":
        return Ridge(alpha=p["alpha"], random_state=SEED)
    if name == "SVR":                       # RBF kernel, the project's default
        return SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"],
                   gamma=p["gamma"])
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=300, max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=SEED, n_jobs=-1)
    if name == "ET":
        return ExtraTreesRegressor(
            n_estimators=300, max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=SEED, n_jobs=-1)
    if name == "PLSR":
        return PLSRegression(n_components=p["n_components"])
    if name == "ENet":
        return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"],
                          max_iter=5000, random_state=SEED)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=p["n_neighbors"],
                                   weights=p["weights"])
    if name == "GPR":
        return GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
            + WhiteKernel(noise_level=1.0),
            alpha=p["alpha"], normalize_y=True, random_state=SEED)
    if name == "XGB":
        return XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# grids  (first entry of every list == the stage-1 default)
# --------------------------------------------------------------------------- #
_KNN_CFG = [{"n_neighbors": n, "weights": w}
            for w in ("uniform", "distance")
            for n in (5, 1, 3, 7, 11, 15)]          # 12, first = (5, uniform)

_TREE_CFG = [{"max_depth": d, "min_samples_leaf": l}
             for d in (None, 3, 5, 10)
             for l in (1, 3)]                        # 8, first = (None, 1)

_XGB_CFG = [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for lr in (0.05, 0.3)
            for d in (3, 2, 6)
            for n in (300, 100)]                     # 12, first = (300, 3, 0.05)

CLF_GRIDS = {
    "LR":     [{"C": c} for c in (1.0, 0.01, 0.1, 10.0, 100.0)],          # 5
    "SVM":    [{"C": c, "gamma": g} for g in ("scale", 0.01, 0.1)
               for c in (1.0, 0.1, 10.0, 100.0)],                          # 12
    "SVMLin": [{"C": c} for c in (1.0, 0.01, 0.1, 10.0, 100.0)],          # 5
    "RF":     _TREE_CFG,                                                    # 8
    "ET":     _TREE_CFG,                                                    # 8
    "LDA":    [{"solver": s, "shrinkage": sh} for s, sh in
               (("lsqr", "auto"), ("svd", None), ("lsqr", 0.1),
                ("lsqr", 0.5), ("lsqr", 0.9))],                          # 5
    "PLSDA":  [{"n_components": n} for n in (5, 2, 4, 6, 8, 10)],         # 6
    "KNN":    _KNN_CFG,                                                     # 12
    "NB":     [{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3, 1e-1)],    # 4
    "XGB":    _XGB_CFG,                                                     # 12
}

REG_GRIDS = {
    "Ridge":  [{"alpha": a} for a in (1.0, 0.01, 0.1, 10.0, 100.0, 1000.0)],  # 6
    "SVR":    [{"C": c, "epsilon": e, "gamma": "scale"}
               for c in (1.0, 10.0, 100.0, 1000.0)
               for e in (0.1, 1.0, 5.0)],                                     # 12
    "RF":     _TREE_CFG,                                                      # 8
    "ET":     _TREE_CFG,                                                      # 8
    "PLSR":   [{"n_components": n} for n in (5, 1, 2, 3, 8, 10)],           # 6
    "ENet":   [{"alpha": a, "l1_ratio": r} for a in (0.1, 0.01, 1.0, 10.0)
               for r in (0.5, 0.2, 0.8)],                                     # 12
    "KNN":    _KNN_CFG,                                                       # 12
    "GPR":    [{"alpha": a} for a in (1e-10, 1e-6, 1e-2, 1.0)],             # 4
    "XGB":    _XGB_CFG,                                                       # 12
}

if not HAS_XGB:                     # drop XGBoost from the grids if unavailable
    CLF_GRIDS.pop("XGB", None)
    REG_GRIDS.pop("XGB", None)


def n_configs(grids):
    return sum(len(v) for v in grids.values())
