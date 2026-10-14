"""Selector-zoo experiment driver.

Evaluates a grid of  (selector x model)  under two protocols:

  * A_random  : the project's fixed folds (classification_folds.csv / regression_folds.csv)
  * B_spatial : KMeans(k=10) spatial blocks + StratifiedGroupKFold(3)  [full mode]

Two tracks:
  * clf : target = Unnamed:2 (0/1), rows = all 132
  * reg : target = DI,          rows = DI>0 subset (116)

Feature selection happens INSIDE every fold loop (training rows only) -> the
validation fold is never seen during selection (no leakage). Embedded/wrapper
selectors use their NATIVE model (design choice B); see selector_zoo.NATIVE_MODEL.

Run:
  python src/run_selector_zoo.py --smoke      # tiny grid, random folds, verify pipeline
  python src/run_selector_zoo.py --quick      # full selector/model grid, random folds only
  python src/run_selector_zoo.py              # full grid, random + spatial
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# collinear spectral features make Lasso/LassoCV emit ConvergenceWarning by the
# thousand; the selection ranking is still usable, so we silence the flood.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

import selector_zoo as zoo
from spatial_folds import recorded_spatial_folds

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA = ROOT / "data"
EXP = ROOT / "experiments" / "2026-10-12_selector_zoo"
OUT = EXP / "outputs"

K_VALUES = (5, 10, 20, 30)
SMOKE_K = (10,)
SPATIAL_K = 10
SPATIAL_N_SPLITS = 3
SPATIAL_SEEDS = (42, 7, 2026)
RANDOM_SEEDS = (42, 7, 2026)

# real-time progress log (written to a file AND stdout, with timestamps)
PROGRESS = {"file": None}


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if PROGRESS["file"]:
        with open(PROGRESS["file"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# PLS-DA classifier wrapper (sklearn-compatible)
# --------------------------------------------------------------------------- #
class PLSDA(BaseEstimator, ClassifierMixin):
    """Discriminant PLS: regress a 0/1 dummy on X, threshold at 0.5."""

    def __init__(self, n_components=5, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        nc = int(min(self.n_components, np.asarray(X).shape[1]))
        self.pls_ = PLSRegression(n_components=max(1, nc))
        self.pls_.fit(X, (y == self.classes_[-1]).astype(float))
        return self

    def predict(self, X):
        s = self.pls_.predict(X).ravel()
        return np.where(s >= self.threshold, self.classes_[-1], self.classes_[0])

    def predict_proba(self, X):
        s = self.pls_.predict(X).ravel()
        return np.c_[1.0 - s, s]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_train():
    df = pd.read_csv(DATA / "splits" / "train.csv", encoding="utf-8-sig")
    features = df.columns[9:].tolist()
    X = df[features].to_numpy(float)
    y_cls = df["Unnamed: 2"].astype(int).to_numpy()
    y_reg = df["DI"].to_numpy(float)
    coords = df[["O_Lat", "O_Lng"]].to_numpy(float)
    return df, features, X, y_cls, y_reg, coords


def load_random_folds(df):
    """Return (clf_fold, reg_fold) vectors of length n_rows (-1 = excluded)."""
    c = (pd.read_csv(DATA / "cv" / "classification_folds.csv", encoding="utf-8-sig")
         .sort_values("row_index"))
    clf = c.fold.to_numpy(int)
    r = pd.read_csv(DATA / "cv" / "regression_folds.csv", encoding="utf-8-sig")
    reg = np.full(len(df), -1, int)
    reg[r.row_index.to_numpy(int)] = r.fold.to_numpy(int)
    return clf, reg


def spatial_fold(coords, y_strat, seed):
    # 折表存在时采用其中记录的块号与折号（见 spatial_folds 模块），否则按坐标现算
    rec = recorded_spatial_folds(y_strat, seed)
    if rec is not None:
        return rec

    km = KMeans(n_clusters=SPATIAL_K, n_init=10, random_state=seed)
    groups = km.fit_predict(coords)
    fold = np.full(len(y_strat), -1, int)
    sgkf = StratifiedGroupKFold(n_splits=SPATIAL_N_SPLITS, shuffle=True,
                                random_state=seed)
    for i, (_, va) in enumerate(sgkf.split(coords, y_strat, groups=groups)):
        fold[va] = i
    return fold, groups


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def clf_models(seed):
    m = {
        "LR": Pipeline([("sc", StandardScaler()),
                        ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("m", SVC(kernel="rbf", probability=True, random_state=seed))]),
        "SVMLin": Pipeline([("sc", StandardScaler()),
                            ("m", SVC(kernel="linear", probability=True, random_state=seed))]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                     class_weight="balanced"),
        "ET": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                   class_weight="balanced"),
        "LDA": Pipeline([("sc", StandardScaler()),
                         ("m", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "PLSDA": Pipeline([("sc", StandardScaler()), ("m", PLSDA(n_components=5))]),
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsClassifier(n_neighbors=5))]),
        "NB": Pipeline([("sc", StandardScaler()), ("m", GaussianNB())]),
    }
    if HAS_XGB:
        m["XGB"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, n_jobs=-1, verbosity=0,
                                 eval_metric="logloss")
    return m


def reg_models(seed):
    m = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(random_state=seed))]),
        "SVR": Pipeline([("sc", StandardScaler()),
                         ("m", SVR(kernel="rbf"))]),
        "RF": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "ET": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "PLSR": Pipeline([("sc", StandardScaler()),
                          ("m", PLSRegression(n_components=5))]),
        "ENet": Pipeline([("sc", StandardScaler()),
                          ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                                           random_state=seed))]),
        "KNN": Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=5))]),
        "GPR": Pipeline([("sc", StandardScaler()),
                         ("m", GaussianProcessRegressor(
                             kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                             + WhiteKernel(noise_level=1.0),
                             normalize_y=True, random_state=seed))]),
    }
    if HAS_XGB:
        m["XGB"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                random_state=seed, n_jobs=-1, verbosity=0)
    return m


def clf_scores(m, X, y):
    pred = m.predict(X)
    try:
        score = m.predict_proba(X)[:, 1]
    except Exception:
        score = m.decision_function(X)
    return {"accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score)}


def reg_scores(m, X, y):
    pred = m.predict(X)
    return {"r2": r2_score(y, pred),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


# --------------------------------------------------------------------------- #
# core loop
# --------------------------------------------------------------------------- #
def run_track(task, X, y, row_mask, fold_sets, selectors, models, k_list, seeds):
    Xt, yt = X[row_mask], y[row_mask]
    rows = []
    done = 0
    need_mi = any(s in zoo.NEEDS_MI for s in selectors)
    for scheme, fvec_full in fold_sets:
        fvec = fvec_full[row_mask]
        for fold_id in np.unique(fvec):
            if fold_id < 0:
                continue
            va = fvec == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue
            done += 1
            _log(f"[{task}] {scheme} fold={fold_id} n_train={int(tr.sum())} "
                 f"n_valid={int(va.sum())} (fold #{done})")
            if need_mi:
                # MI matrix depends only on the training ROWS, not on the seed,
                # so compute it once per fold and reuse across all seeds.
                _log(f"[{task}]   computing MI redundancy matrix "
                     f"({int(tr.sum())}x{Xt.shape[1]}) ...")
                mi = zoo.compute_mi_matrix(Xt[tr])
                _log(f"[{task}]   MI matrix done")
            else:
                mi = None
            for seed in seeds:
                mmodels = models(seed)
                # one ctx (hence one _cache) per (fold, seed): CARS / RFE /
                # stability results are cached and reused across all k.
                ctx = {"task": task, "seed": seed, "mi_matrix": mi, "_cache": {}}
                for k in k_list:
                    for sname, sfn in selectors.items():
                        try:
                            sel = sfn(Xt[tr], yt[tr], k, ctx)
                        except Exception as e:
                            print(f"    [warn] {task}/{sname}/k={k}: "
                                  f"{type(e).__name__}: {e}")
                            continue
                        if len(sel) == 0:
                            continue
                        for mname, base in mmodels.items():
                            mdl = clone(base)
                            Xtr, Xva = Xt[tr][:, sel], Xt[va][:, sel]
                            try:
                                mdl.fit(Xtr, yt[tr])
                                sc = clf_scores(mdl, Xva, yt[va]) if task == "clf" \
                                    else reg_scores(mdl, Xva, yt[va])
                            except Exception as e:
                                print(f"    [warn] fit {task}/{mname}: "
                                      f"{type(e).__name__}: {e}")
                                continue
                            rec = {"task": task, "scheme": scheme, "selector": sname,
                                   "model": mname, "k": k, "seed": seed,
                                   "fold": int(fold_id), "n_features": len(sel),
                                   "n_train": int(tr.sum()), "n_valid": int(va.sum())}
                            rec.update(sc)
                            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df, task):
    mets = ["accuracy", "f1", "recall", "roc_auc"] if task == "clf" \
        else ["r2", "rmse", "mae"]
    g = df.groupby(["selector", "model", "k", "scheme"], as_index=False).agg(
        n_features=("n_features", "first"), n_eval=("fold", "size"),
        **{f"{m}_mean": (m, "mean") for m in mets},
        **{f"{m}_std": (m, "std") for m in mets})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid, random folds")
    ap.add_argument("--stage1", action="store_true",
                    help="full selector x full model, K=10 only, random folds, 1 seed")
    ap.add_argument("--stage2", action="store_true",
                    help="K sensitivity: K=5/10/20/30, random folds, 1 seed")
    ap.add_argument("--stage4", action="store_true",
                    help="spatial block folds: K=5/10/20/30, 3-fold blockCV, 1 seed")
    ap.add_argument("--stage4b", action="store_true",
                    help="RANDOM 3-fold control for stage-4: same grid/K, "
                         "3-fold random -> isolates the fold-count confound")
    ap.add_argument("--quick", action="store_true", help="full grid, random folds only")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df, features, X, y_cls, y_reg, coords = load_train()
    n = len(df)
    print(f"[data] train={n} x {X.shape[1]} | clf counts {np.bincount(y_cls).tolist()} "
          f"| DI>0 {int((y_reg > 0).sum())} | xgb={'yes' if HAS_XGB else 'NO'}")

    clf_fold, reg_fold = load_random_folds(df)

    if args.smoke:
        clf_sel = {k: zoo.CLF_SELECTORS[k] for k in
                   ("relieff", "mrmr", "lasso", "fs_cs", "stab_fscs", "random")}
        reg_sel = {k: zoo.REG_SELECTORS[k] for k in
                   ("relieff", "mrmr", "lasso", "cars", "stab_lasso", "random")}
        k_list = SMOKE_K
        seeds = (42,)
        clf_m = lambda s: {k: v for k, v in clf_models(s).items() if k in ("LR", "RF")}
        reg_m = lambda s: {k: v for k, v in reg_models(s).items() if k in ("Ridge", "RF")}
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        tag = "smoke"
    elif args.stage1:
        # Stage-1 selector sweep: the whole (selector x model) matrix at a single
        # K, random folds only, one seed. Cheap enough (~minutes) to look at the
        # big picture before spending hours on K / seeds / spatial folds.
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        k_list = (10,)
        seeds = (42,)
        clf_m, reg_m = clf_models, reg_models
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        tag = "stage1"
    elif args.stage2:
        # Stage-2 K sensitivity: hold everything else fixed (random folds, one
        # seed, full selector x model grid) and vary only the number of kept
        # features. Answers two questions: (a) does the random-feature noise
        # floor collapse at very small K? (b) are the stage-1 oddities
        # (relieff-as-only-winner, XGBoost last, PLSR low) stable across K?
        # K_VALUES includes 10, so the k=10 slice MUST reproduce stage1 exactly.
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        k_list = K_VALUES           # (5, 10, 20, 30)
        seeds = (42,)
        clf_m, reg_m = clf_models, reg_models
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        tag = "stage2"
    elif args.stage4:
        # Stage-4 spatial-block CV (blockCV). Same full grid and same K grid as
        # stage-2, but the folds are SPATIAL BLOCKS: KMeans clusters the lat/lon
        # coordinates into 10 blocks and StratifiedGroupKFold(3) holds WHOLE
        # blocks out. No block is ever split across train/validation, so a
        # validation sample cannot be "remembered" through a near-duplicate
        # training neighbour from the same field.
        #
        # Only ONE partition seed (42) is used, so the model seeds match stage-2
        # exactly and (random folds) vs (spatial folds) is a clean single-variable
        # switch. Repeating the partition over several seeds is confirmation-stage
        # work, not exploration.
        #
        # Fold construction deliberately mirrors src/spatial_group_cv.py (P0-2)
        # so that the two experiments use the SAME spatial partition and their
        # numbers can be read side by side.
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        k_list = K_VALUES
        seeds = (42,)
        clf_m, reg_m = clf_models, reg_models

        s = 42
        f_c, g_c = spatial_fold(coords, y_cls, s)
        clf_folds = [("B_spatial_s42", f_c)]

        # Regression: blocks + folds are built on the DI>0 subset itself,
        # matching spatial_group_cv.py. (The driver's default full mode instead
        # builds them on all 132 rows and then subsets, which would leave the
        # low-DI stratum empty and the regression folds unbalanced.)
        sick = y_reg > 0
        f_sick, g_sick = spatial_fold(coords[sick], y_cls[sick], s)
        f_r = np.full(n, -1, int)
        f_r[sick] = f_sick
        reg_folds = [("B_spatial_s42", f_r)]
        tag = "stage4"

        # fold-design audit -- must be inspected before trusting any number
        aud = []
        for i in np.unique(f_c):
            m = f_c == i
            aud.append({"task": "clf", "fold": int(i), "n_val": int(m.sum()),
                        "n_healthy": int((y_cls[m] == 0).sum()),
                        "n_diseased": int((y_cls[m] == 1).sum()),
                        "n_blocks": int(len(np.unique(g_c[m])))})
        for i in np.unique(f_sick):
            m = f_sick == i
            aud.append({"task": "reg", "fold": int(i), "n_val": int(m.sum()),
                        "n_healthy": int((y_cls[sick][m] == 0).sum()),
                        "n_diseased": int((y_cls[sick][m] == 1).sum()),
                        "n_blocks": int(len(np.unique(g_sick[m])))})
        pd.DataFrame(aud).to_csv(OUT / "fold_design_stage4.csv", index=False,
                                 encoding="utf-8-sig")
        pd.DataFrame([{"row_index": i, "block": int(b)}
                      for i, b in enumerate(g_c)]).to_csv(
            OUT / "spatial_blocks_clf_stage4.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"row_index": int(np.where(sick)[0][i]), "block": int(b)}
                      for i, b in enumerate(g_sick)]).to_csv(
            OUT / "spatial_blocks_reg_stage4.csv", index=False, encoding="utf-8-sig")
        print("[stage4] spatial fold design (k_blocks=10, n_splits=3, seed=42)")
        print(pd.DataFrame(aud).to_string(index=False))
    elif args.stage4b:
        # Stage-4b: same grid / same K grid as stage-4, but with RANDOM folds
        # that have the SAME fold COUNT (3).
        #
        # WHY this exists: stage-4 contrasted
        #     A_random  = the project's 5-fold random split  (train ~80%)
        #     B_spatial = 3-fold spatial block split         (train ~67%)
        # so every "gap" there mixed TWO changes at once: (a) random -> spatial,
        # and (b) 5 folds -> 3 folds, i.e. LESS training data. For regression the
        # gap (+0.61) is far too large for (b) to explain, but the classification
        # gap is only +0.017 -- the same order of magnitude as a training-size
        # effect -- so it needs a clean control before it can be believed.
        #
        # Stage-4b supplies that control. With it we can decompose:
        #     A_random3 vs B_spatial  -> spatial effect at FIXED fold depth
        #     A_random  vs A_random3  -> pure training-size effect (5f vs 3f)
        #
        # The sample counts line up almost exactly, which is what makes the
        # contrast valid:
        #   clf: random 3-fold trains on 88/88/88 (mean 88.0)
        #        spatial 3-fold trains on 79/100/85 (mean 88.0)   <- same mean
        #   reg: random 3-fold trains on 77/77/78  (mean 77.3)
        #        spatial 3-fold trains on 85/71/76  (mean 77.3)   <- same mean
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        k_list = K_VALUES
        seeds = (42,)
        clf_m, reg_m = clf_models, reg_models

        s = 42  # same seed as the stage-4 partition -> like-for-like switch

        # classification: stratified (keeps the 72/60 class ratio in every fold)
        f_c = np.full(n, -1, int)
        for i, (_, va) in enumerate(
                StratifiedKFold(n_splits=SPATIAL_N_SPLITS, shuffle=True,
                                random_state=s).split(coords, y_cls)):
            f_c[va] = i
        clf_folds = [("A_random3", f_c)]

        # regression: built on the DI>0 subset and stratified the same way as
        # the spatial folds (by the 0/1 label), so the two 3-fold designs are
        # matched on both n_train and class balance.
        sick = y_reg > 0
        f_sick = np.full(int(sick.sum()), -1, int)
        for i, (_, va) in enumerate(
                StratifiedKFold(n_splits=SPATIAL_N_SPLITS, shuffle=True,
                                random_state=s).split(coords[sick], y_cls[sick])):
            f_sick[va] = i
        f_r = np.full(n, -1, int)
        f_r[sick] = f_sick
        reg_folds = [("A_random3", f_r)]
        tag = "stage4b"

        aud = []
        for i in np.unique(f_c):
            m = f_c == i
            aud.append({"task": "clf", "fold": int(i), "n_val": int(m.sum()),
                        "n_healthy": int((y_cls[m] == 0).sum()),
                        "n_diseased": int((y_cls[m] == 1).sum())})
        for i in np.unique(f_sick):
            m = f_sick == i
            aud.append({"task": "reg", "fold": int(i), "n_val": int(m.sum()),
                        "n_healthy": int((y_cls[sick][m] == 0).sum()),
                        "n_diseased": int((y_cls[sick][m] == 1).sum())})
        pd.DataFrame(aud).to_csv(OUT / "fold_design_stage4b.csv", index=False,
                                 encoding="utf-8-sig")
        print(f"[stage4b] random {SPATIAL_N_SPLITS}-fold control "
              f"(StratifiedKFold, shuffle=True, seed={s})")
        print(pd.DataFrame(aud).to_string(index=False))
    else:
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        k_list = K_VALUES
        seeds = RANDOM_SEEDS
        clf_m, reg_m = clf_models, reg_models
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        if not args.quick:
            for s in SPATIAL_SEEDS:
                f_c, _ = spatial_fold(coords, y_cls, s)
                clf_folds.append((f"B_spatial_s{s}", f_c))
                # regression stratification: 3 quantile bins of DI (only DI>0 rows)
                bins = pd.qcut(y_reg, 3, labels=False, duplicates="drop")
                f_r, _ = spatial_fold(coords, np.where(bins >= 0, bins, 0), s)
                reg_folds.append((f"B_spatial_s{s}", f_r))
        tag = "quick" if args.quick else "full"

    PROGRESS["file"] = OUT / f"run_progress_{tag}.log"
    PROGRESS["file"].write_text("", encoding="utf-8")
    _log(f"START mode={tag} k={list(k_list)} seeds={list(seeds)} xgb={HAS_XGB} "
         f"| clf_sel={len(clf_sel)} reg_sel={len(reg_sel)}")

    t0 = time.time()
    print(f"\n=== clf track ({tag}) ===")
    clf_df = run_track("clf", X, y_cls, np.ones(n, bool), clf_folds, clf_sel,
                       clf_m, k_list, seeds)
    print(f"  rows={len(clf_df)}  ({time.time()-t0:.0f}s)")

    print(f"\n=== reg track ({tag}) ===")
    reg_mask = y_reg > 0
    reg_df = run_track("reg", X, y_reg, reg_mask, reg_folds, reg_sel,
                       reg_m, k_list, seeds)
    print(f"  rows={len(reg_df)}  ({time.time()-t0:.0f}s)")

    if args.smoke or args.stage1 or args.stage2 or args.stage4 or args.stage4b:
        clf_df.to_csv(OUT / f"clf_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
        reg_df.to_csv(OUT / f"reg_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
        cs = summarize(clf_df, "clf")
        rs = summarize(reg_df, "reg")
        cs.to_csv(OUT / f"clf_summary_{tag}.csv", index=False, encoding="utf-8-sig")
        rs.to_csv(OUT / f"reg_summary_{tag}.csv", index=False, encoding="utf-8-sig")
        if args.stage2 or args.stage4 or args.stage4b:
            # the K sweep has 4x as many rows, so print the per-K leaderboard
            # instead of one flat head() -- much easier to read while it runs.
            print("\n[clf summary | per K, top-8 by roc_auc]")
            for kk in sorted(cs["k"].unique()):
                sub = (cs[cs.k == kk].sort_values("roc_auc_mean", ascending=False)
                       .head(8))
                print(f"-- k={kk} --")
                print(sub[["selector", "model", "roc_auc_mean",
                           "accuracy_mean"]].round(4).to_string(index=False))
            print("\n[reg summary | per K, top-8 by r2]")
            for kk in sorted(rs["k"].unique()):
                sub = rs[rs.k == kk].sort_values("r2_mean", ascending=False).head(8)
                print(f"-- k={kk} --")
                print(sub[["selector", "model", "r2_mean", "rmse_mean"]]
                      .round(4).to_string(index=False))
        else:
            # stage1 has 11x10 = 110 clf rows / 10x9 = 90 reg rows -> print the head
            head = 25 if args.stage1 else None
            print("\n[clf summary]")
            print(cs.sort_values("roc_auc_mean", ascending=False).head(head).to_string(index=False))
            print("\n[reg summary]")
            print(rs.sort_values("r2_mean", ascending=False).head(head).to_string(index=False))
    else:
        clf_df.to_csv(OUT / "clf_per_fold_metrics.csv", index=False, encoding="utf-8-sig")
        reg_df.to_csv(OUT / "reg_per_fold_metrics.csv", index=False, encoding="utf-8-sig")
        summarize(clf_df, "clf").to_csv(OUT / "clf_summary.csv", index=False,
                                        encoding="utf-8-sig")
        summarize(reg_df, "reg").to_csv(OUT / "reg_summary.csv", index=False,
                                        encoding="utf-8-sig")
        print("\n[done] summaries written")

    meta = {"n_train": int(n), "n_features": int(X.shape[1]),
            "k_values": list(k_list), "mode": tag,
            "clf_selectors": list(clf_sel), "reg_selectors": list(reg_sel),
            "xgb_available": HAS_XGB, "test_set_used": False}
    (OUT / f"metadata_{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {tag} -> {OUT}  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
