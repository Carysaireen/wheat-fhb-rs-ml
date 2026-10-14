"""Hyper-parameter sweep driver.

Question it answers
-------------------
Stage 1/2/4 left three counter-intuitive results:
  * classification: XGBoost was LAST of 10 models (0.759 ROC-AUC)
  * regression:     PLSR was near-last and COLLAPSED as K grew (0.250 -> 0.094)
  * regression:     SVR was LAST (0.194 R^2)
All three models ran with untouched default hyper-parameters. This sweep asks
the only fair follow-up question: **can a sensible hyper-parameter fix them?**

Protocol (single-variable: only the model hyper-parameters change)
------------------------------------------------------------------
  selector  = relieff   (the one selector that beat the noise floor everywhere)
  K         = 10        (the middle K from stage-2)
  folds     = the project's fixed random 5-fold split (same as stage-1)
  seed      = 42
  -> identical folds, identical selected features per fold (cached), identical
     scaler per fold => only the hyper-parameters differ.

Leakage safety: feature selection AND the scaler are fit on the TRAINING fold
only; the validation fold is never touched.

Run:
  python src/run_hyperparam_sweep.py --smoke    # 2 configs/model, verify pipeline
  python src/run_hyperparam_sweep.py            # full grids
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler

import selector_zoo as zoo
import hyperparam_grids as hg
from run_selector_zoo import (clf_scores, load_random_folds, load_train,
                              reg_scores)

EXP = ROOT / "experiments" / "2026-10-13_hyperparam_sweep"
OUT = EXP / "outputs"

SELECTOR = "relieff"
K = 10
SEED = 42

PROGRESS = {"file": None}


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if PROGRESS["file"]:
        with open(PROGRESS["file"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def run_track(task, X, y, row_mask, folds, grids, build_fn, limit=None):
    """Sweep every (model, hyper-parameter config) over the fixed folds.

    `limit` keeps only the first N configs per model (used by --smoke).
    """
    Xt, yt = X[row_mask], y[row_mask]
    fvec = folds[row_mask]
    is_clf = task == "clf"
    rows = []
    for fold_id in np.unique(fvec):
        if fold_id < 0:
            continue
        va = fvec == fold_id
        tr = ~va
        if tr.sum() < 20 or va.sum() < 5:
            continue

        # --- feature selection: TRAINING fold only -------------------------
        sel = zoo.CLF_SELECTORS[SELECTOR](Xt[tr], yt[tr], K,
                                          {"task": task, "seed": SEED,
                                           "_cache": {}})
        # --- ONE scaler per fold, shared by every hyper-parameter setting ---
        sc = StandardScaler().fit(Xt[tr][:, sel])
        Xtr = sc.transform(Xt[tr][:, sel])
        Xva = sc.transform(Xt[va][:, sel])

        n_cfg = sum((len(v) if limit is None else min(len(v), limit))
                    for v in grids.values())
        _log(f"[{task}] fold={fold_id} n_train={int(tr.sum())} "
             f"n_valid={int(va.sum())} | selected {len(sel)} feats | "
             f"{n_cfg} configs to fit")

        for model, cfgs in grids.items():
            if limit is not None:
                cfgs = cfgs[:limit]
            for ci, p in enumerate(cfgs):
                mdl = build_fn(model, **p)
                try:
                    mdl.fit(Xtr, yt[tr])
                    sc_ = (clf_scores(mdl, Xva, yt[va]) if is_clf
                           else reg_scores(mdl, Xva, yt[va]))
                except Exception as e:
                    print(f"    [warn] {task}/{model}/{p}: "
                          f"{type(e).__name__}: {e}")
                    continue
                rec = {"task": task, "model": model, "config_id": ci,
                       "params": json.dumps(p, sort_keys=True),
                       "fold": int(fold_id), "n_train": int(tr.sum()),
                       "n_valid": int(va.sum())}
                rec.update(sc_)
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df, task):
    mets = ["accuracy", "f1", "recall", "roc_auc"] if task == "clf" \
        else ["r2", "rmse", "mae"]
    g = df.groupby(["model", "config_id", "params"], as_index=False).agg(
        n_eval=("fold", "size"),
        **{f"{m}_mean": (m, "mean") for m in mets},
        **{f"{m}_std": (m, "std") for m in mets})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="only the first 2 configs per model")
    args = ap.parse_args()

    # configs on the model factory are normalised to plain dicts for the log
    limit = 2 if args.smoke else None
    tag = "smoke" if args.smoke else "sweep"

    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS["file"] = OUT / f"run_progress_{tag}.log"
    PROGRESS["file"].write_text("", encoding="utf-8")

    df, features, X, y_cls, y_reg, coords = load_train()
    n = len(df)
    clf_fold, reg_fold = load_random_folds(df)

    n_clf = hg.n_configs(hg.CLF_GRIDS) if limit is None \
        else limit * len(hg.CLF_GRIDS)
    n_reg = hg.n_configs(hg.REG_GRIDS) if limit is None \
        else limit * len(hg.REG_GRIDS)

    _log(f"START mode={tag} | selector={SELECTOR} K={K} seed={SEED} | "
         f"clf: {len(hg.CLF_GRIDS)} models x configs -> {n_clf} fits/fold | "
         f"reg: {len(hg.REG_GRIDS)} models -> {n_reg} fits/fold | "
         f"xgb={'yes' if hg.HAS_XGB else 'NO'}")

    t0 = time.time()
    print(f"\n=== clf sweep ({tag}) ===")
    clf = run_track("clf", X, y_cls, np.ones(n, bool), clf_fold,
                    hg.CLF_GRIDS, hg.build_clf, limit)
    print(f"  rows={len(clf)}  ({time.time()-t0:.0f}s)")

    print(f"\n=== reg sweep ({tag}) ===")
    reg = run_track("reg", X, y_reg, y_reg > 0, reg_fold,
                    hg.REG_GRIDS, hg.build_reg, limit)
    print(f"  rows={len(reg)}  ({time.time()-t0:.0f}s)")

    clf.to_csv(OUT / f"clf_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
    reg.to_csv(OUT / f"reg_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
    cs = summarize(clf, "clf")
    rs = summarize(reg, "reg")
    cs.to_csv(OUT / f"clf_summary_{tag}.csv", index=False, encoding="utf-8-sig")
    rs.to_csv(OUT / f"reg_summary_{tag}.csv", index=False, encoding="utf-8-sig")

    print("\n[clf | per model, best config by roc_auc_mean]")
    best = (cs.sort_values("roc_auc_mean", ascending=False)
            .groupby("model", as_index=False).first()
            .sort_values("roc_auc_mean", ascending=False))
    print(best[["model", "params", "roc_auc_mean", "accuracy_mean"]]
          .round(4).to_string(index=False))
    print("\n[reg | per model, best config by r2_mean]")
    bestr = (rs.sort_values("r2_mean", ascending=False)
             .groupby("model", as_index=False).first()
             .sort_values("r2_mean", ascending=False))
    print(bestr[["model", "params", "r2_mean", "rmse_mean"]]
          .round(4).to_string(index=False))

    meta = {"n_train": int(n), "n_features": int(X.shape[1]), "selector": SELECTOR,
            "k": K, "seed": SEED, "mode": tag,
            "clf_models": list(hg.CLF_GRIDS), "reg_models": list(hg.REG_GRIDS),
            "xgb_available": hg.HAS_XGB, "test_set_used": False}
    (OUT / f"metadata_{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {tag} -> {OUT}  | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
