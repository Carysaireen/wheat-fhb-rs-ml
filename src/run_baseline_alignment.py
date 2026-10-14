"""
跨阶段对齐基准（baseline alignment）
====================================

【这个脚本在回答什么问题】
项目从 2026-08 的「最早基线」走到了现在的「选择器动物园 + 超参扫描」。要判断
"现在的成果到底比最早强了多少"，最大的坑是 **口径不一致**：

    最早基线的数字，和现在的数字，是在不同脚本、不同预处理、不同超参下算出来的。
    直接相减等于把「方法进步」和「代码差异」混在一起。

所以本脚本把两边都放进 **同一套折 + 同一套代码 + 同一套指标** 里重算：

  · 折（与项目固定折完全一致）
      A_random5   : data/cv/classification_folds.csv 的固定 5 折（随机分层）
      B_spatial3  : KMeans(k=10) 空间块 + StratifiedGroupKFold(3)，
                    分区种子 42/7/2026（与 P0-② / FS-CS 的空间折一致）
  · 代码：复用 src/run_selector_zoo.py 的模型工厂与打分函数
  · 特征集
      —— 最早基线侧（固定/全局特征集）
         all_117    全部 117 维
         common_6   项目早先的"共同 6 特征"
         boruta_30  Boruta confirmed_rate>=0.6（≈30 维，早先最强候选）
         current_2  NREDI1 + NDVIre2（早先"黄金双特征"）
      —— 当前侧（逐折选择，nested，无泄漏）
         relieff_10 ReliefF 逐折取前 10
         fs_cs_10   FS-CS(Fisher) 逐折取前 10
  · 模型：每个模型都给 **默认** 与 **调优(*)** 两版
      默认 = 与最早基线相同的定义（保证可比）
      调优 = results/hyperparam_sweep/recommended_params.json 里的参数

【自校验】
  current_2 + LR 在 A_random5 上应当复现传统基线的 accuracy ≈ 0.8046，
  在 B_spatial3 上应当复现 FS-CS 报告的 accuracy ≈ 0.8023。
  如果对不上，说明口径没对齐 —— 那么整张表都不可信。

用法：
    python -u src/run_baseline_alignment.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import selector_zoo as zoo                       # noqa: E402
import run_selector_zoo as R                     # noqa: E402  复用折构造 / 模型 / 打分

from sklearn.base import clone                   # noqa: E402
from sklearn.cross_decomposition import PLSRegression            # noqa: E402
from sklearn.ensemble import (ExtraTreesClassifier,              # noqa: E402
                              ExtraTreesRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor    # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge       # noqa: E402
from sklearn.neighbors import (KNeighborsClassifier,             # noqa: E402
                               KNeighborsRegressor)
from sklearn.pipeline import Pipeline                            # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
from sklearn.svm import SVC, SVR                                 # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:                                                 # pragma: no cover
    HAS_XGB = False

EXP = ROOT / "experiments" / "2026-10-13_baseline_alignment"
OUT = EXP / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

REC_FILE = ROOT / "results" / "hyperparam_sweep" / "recommended_params.json"

MODEL_SEED = 42
SPATIAL_SEEDS = (42, 7, 2026)      # 与 P0-② / FS-CS 的空间折一致
K_MAIN = 10                        # 逐折选择器的特征数

_LOG = OUT / "run_progress_alignment.log"


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# 特征集
# --------------------------------------------------------------------------- #
def fixed_feature_sets(names: list[str]) -> dict[str, list[str]]:
    """最早基线那一侧的**固定**特征集（全局定好后不再变）。"""
    bo = pd.read_csv(ROOT / "results" / "feature_selection" / "boruta_stability.csv")
    boruta = bo.loc[bo.confirmed_rate >= 0.6, "feature"].tolist()
    return {
        "all_117": list(names),
        "common_6": [f for f in ["B8_HOM", "B8_MEA", "NDVIre2", "NDVIre3",
                                 "NREDI1", "NREDI2"] if f in names],
        "boruta_30": [f for f in boruta if f in names],
        "current_2": [f for f in ["NREDI1", "NDVIre2"] if f in names],
    }


def selector_defs(task: str) -> dict[str, tuple]:
    """当前那一侧的**逐折选择器**（nested，只在训练折里选）。

    注意：回归轨的选择器名单与分类轨不同 —— 回归没有 fs_cs（Fisher 比值是
    为分类的类间/类内散度设计的），对应位置是 CARS（竞争自适应重加权采样）。
    这与 selector_zoo.py 里 CLF_SELECTORS / REG_SELECTORS 的差异一致。
    """
    sels = zoo.CLF_SELECTORS if task == "clf" else zoo.REG_SELECTORS
    if task == "clf":
        return {
            "relieff_10": ("fn", sels["relieff"], K_MAIN),
            "fs_cs_10": ("fn", sels["fs_cs"], K_MAIN),
        }
    return {
        "relieff_10": ("fn", sels["relieff"], K_MAIN),
        "cars_10": ("fn", sels["cars"], K_MAIN),
    }


# --------------------------------------------------------------------------- #
# 模型：默认（=最早基线定义） + 调优(*)
# --------------------------------------------------------------------------- #
def rec_params() -> dict:
    if REC_FILE.exists():
        return json.loads(REC_FILE.read_text(encoding="utf-8"))
    return {"clf": {}, "reg": {}}


def clf_model_zoo(seed: int) -> dict:
    m = {
        # ---- 默认（与 traditional_baseline_cv.py 完全一致的写法）----
        "LR": Pipeline([("sc", StandardScaler()),
                        ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("m", SVC(kernel="rbf", probability=True, random_state=seed))]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                     class_weight="balanced"),
        "ET": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                   class_weight="balanced"),
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsClassifier(n_neighbors=5))]),
        # ---- 调优（results/hyperparam_sweep/recommended_params.json）----
        "LR*": Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(C=100.0, max_iter=2000,
                                                   random_state=seed))]),
        "SVM*": Pipeline([("sc", StandardScaler()),
                          ("m", SVC(kernel="rbf", C=10.0, probability=True,
                                    random_state=seed))]),
        "RF*": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      random_state=seed, n_jobs=-1,
                                      class_weight="balanced"),
        "ET*": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=1,
                                    random_state=seed, n_jobs=-1,
                                    class_weight="balanced"),
        "KNN*": Pipeline([("sc", StandardScaler()),
                          ("m", KNeighborsClassifier(n_neighbors=11,
                                                     weights="distance"))]),
    }
    if HAS_XGB:
        m["XGB"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, n_jobs=-1, verbosity=0,
                                 eval_metric="logloss")
        m["XGB*"] = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.3,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, n_jobs=-1, verbosity=0,
                                  eval_metric="logloss")
    return m


def reg_model_zoo(seed: int) -> dict:
    m = {
        # ---- 默认（与最早基线一致）----
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(random_state=seed))]),
        "SVR": Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
        "RF": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "ET": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "PLSR": Pipeline([("sc", StandardScaler()),
                          ("m", PLSRegression(n_components=5))]),
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=5))]),
        "GPR": Pipeline([("sc", StandardScaler()),
                         ("m", GaussianProcessRegressor(
                             kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                             + WhiteKernel(noise_level=1.0),
                             normalize_y=True, random_state=seed))]),
        # ---- 调优 ----
        "Ridge*": Pipeline([("sc", StandardScaler()),
                            ("m", Ridge(alpha=100.0))]),
        "SVR*": Pipeline([("sc", StandardScaler()),
                          ("m", SVR(kernel="rbf", C=100.0, epsilon=5.0))]),
        "PLSR*": Pipeline([("sc", StandardScaler()),
                           ("m", PLSRegression(n_components=1))]),
        "RF*": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                     random_state=seed, n_jobs=-1),
        "ET*": ExtraTreesRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                   random_state=seed, n_jobs=-1),
        "KNN*": Pipeline([("sc", StandardScaler()),
                          ("m", KNeighborsRegressor(n_neighbors=15,
                                                    weights="distance"))]),
        "GPR*": Pipeline([("sc", StandardScaler()),
                          ("m", GaussianProcessRegressor(
                              kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                              + WhiteKernel(noise_level=1.0),
                              alpha=0.01, normalize_y=True, random_state=seed))]),
    }
    if HAS_XGB:
        m["XGB"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                random_state=seed, n_jobs=-1, verbosity=0)
        m["XGB*"] = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, n_jobs=-1, verbosity=0)
    return m


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
def run_task(task: str, df, names, X, y, coords) -> pd.DataFrame:
    models_fn = clf_model_zoo if task == "clf" else reg_model_zoo
    score_fn = R.clf_scores if task == "clf" else R.reg_scores
    fixed = fixed_feature_sets(names)
    sel = selector_defs(task)

    # 折集合： (scheme_name, fold_vector_full_length)
    clf_folds, reg_folds = R.load_random_folds(df)
    rand = clf_folds if task == "clf" else reg_folds
    fold_sets = [("A_random5", rand)]

    row_mask = np.ones(len(df), bool) if task == "clf" else (y > 0)
    y_strat = df["Unnamed: 2"].astype(int).to_numpy()

    for s in SPATIAL_SEEDS:
        fvec, _ = R.spatial_fold(coords, y_strat, s)
        fold_sets.append((f"B_spatial3_s{s}", fvec))

    Xt = X[row_mask]
    yt = y[row_mask]                 # ★ 回归轨 yt 长度 = DI>0 子集(116)，不是 132
    rows = []
    for scheme, fvec_full in fold_sets:
        fvec = fvec_full[row_mask]
        for fold_id in np.unique(fvec):
            if fold_id < 0:
                continue
            va = fvec == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue

            # ---- 逐折选择器（只在训练折上 fit）----
            ctx = {"task": task, "seed": MODEL_SEED, "mi_matrix": None, "_cache": {}}
            chosen: dict[str, list[int]] = {}
            for sname, spec in sel.items():
                try:
                    chosen[sname] = list(spec[1](Xt[tr], yt[tr], spec[2], ctx))
                except Exception as e:                        # pragma: no cover
                    log(f"  [warn] {task}/{sname} @ {scheme}/{fold_id}: {e}")

            # ---- 固定特征集 ----
            for fname, feats in fixed.items():
                idx = [names.index(f) for f in feats if f in names]
                chosen[fname] = idx

            log(f"[{task}] {scheme} fold={fold_id} n_train={int(tr.sum())} "
                f"n_valid={int(va.sum())} sets={len(chosen)}")

            models = models_fn(MODEL_SEED)
            for setname, idx in chosen.items():
                if len(idx) == 0:
                    continue
                Xtr, Xva = Xt[tr][:, idx], Xt[va][:, idx]
                for mname, base in models.items():
                    mdl = clone(base)
                    try:
                        mdl.fit(Xtr, yt[tr])
                        sc = score_fn(mdl, Xva, yt[va])
                    except Exception as e:                    # pragma: no cover
                        log(f"  [warn] fit {task}/{setname}/{mname}: "
                            f"{type(e).__name__}: {e}")
                        continue
                    rec = {"task": task, "scheme": scheme, "set": setname,
                           "model": mname, "n_features": len(idx),
                           "seed": MODEL_SEED, "fold": int(fold_id),
                           "n_train": int(tr.sum()), "n_valid": int(va.sum())}
                    rec.update(sc)
                    rows.append(rec)
            # 空间折的 seed 记在 scheme 名里；模型种子恒为 42。
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame, task: str) -> pd.DataFrame:
    mets = ["accuracy", "f1", "recall", "roc_auc"] if task == "clf" \
        else ["r2", "rmse", "mae"]
    g = d.groupby(["set", "model", "scheme"], as_index=False).agg(
        n_features=("n_features", "first"), n_eval=("fold", "size"),
        **{f"{m}_mean": (m, "mean") for m in mets},
        **{f"{m}_std": (m, "std") for m in mets})
    return g


def main() -> None:
    t0 = time.time()
    log("=" * 70)
    log("baseline alignment: earliest baseline vs current best, same folds")
    log(f"model seed = {MODEL_SEED} | spatial seeds = {SPATIAL_SEEDS} | "
        f"K(selector) = {K_MAIN} | HAS_XGB = {HAS_XGB}")
    log("=" * 70)

    df, names, X, y_cls, y_reg, coords = R.load_train()
    log(f"data: {len(df)} rows x {len(names)} features | "
        f"clf 0/1 = {int((y_cls==0).sum())}/{int((y_cls==1).sum())} | DI>0 = {int((y_reg>0).sum())}")

    all_summ = []
    for task, y in (("clf", y_cls), ("reg", y_reg)):
        d = run_task(task, df, names, X, y, coords)
        d.to_csv(OUT / f"alignment_{task}_per_fold.csv", index=False,
                 encoding="utf-8-sig")
        s = summarize(d, task)
        s.to_csv(OUT / f"alignment_{task}_summary.csv", index=False,
                 encoding="utf-8-sig")
        all_summ.append(s)
        log(f"[{task}] done: {len(d)} fold-rows -> alignment_{task}_*.csv")

    # ---- 自校验：必须复现最早基线的 current_2 + LR ----
    clf_s = all_summ[0]
    chk = clf_s[(clf_s.set == "current_2") & (clf_s.model == "LR")]
    log("-" * 70)
    log("SELF-CHECK  current_2 + LR  (must reproduce the earliest baseline)")
    for _, r in chk.iterrows():
        log(f"  {r.scheme:18s} accuracy = {r.accuracy_mean:.4f}  "
            f"roc_auc = {r.roc_auc_mean:.4f}   (n_eval={int(r.n_eval)})")
    log("  reference: A_random5 -> 0.8046 (traditional_baseline) | "
        "B_spatial3 avg -> 0.8023 (FS-CS)")
    log(f"[done] {time.time()-t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
