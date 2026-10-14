"""FS-CS selector experiment: per-fold selection, nested + global evaluation,
under both random stratified folds and spatial block folds.

Design (see experiments/2026-10-12_fs_cs/config.yaml for the full rationale):
  * selection (FS-CS) is ALWAYS fitted on training folds only -> no leakage
  * two protocols:
      - nested : re-select FS-CS inside every outer training fold (honest)
      - global : aggregate per-fold ranks -> fixed feature sets, then evaluate
                 on the same fixed folds (comparable to the existing
                 feature_set_cv numbers)
  * two schemes: A_random (fixed 5-fold) vs B_spatial (KMeans k=10 + SGKF 3-fold)
  * variants: univariate (J = S_B/S_W) and greedy (trace ratio, redundancy-aware)

Run:  python src/run_fs_cs_experiment.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fs_cs import fs_cs_score, rank_fs_cs, select_fs_cs, select_fs_cs_greedy

DATA = ROOT / "data"
EXP = ROOT / "experiments" / "2026-10-12_fs_cs"
OUT = EXP / "outputs"

K_VALUES = (10, 20, 30)
VARIANTS = ("univariate", "greedy")
SPATIAL_K = 10
SPATIAL_N_SPLITS = 3
SPATIAL_SEEDS = (42, 7, 2026)
METRICS = ("accuracy", "f1", "recall", "roc_auc")
FEAT_INDEX: dict = {}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_train():
    df = pd.read_csv(DATA / "splits" / "train.csv", encoding="utf-8-sig")
    features = df.columns[9:].tolist()
    X = df[features].to_numpy(float)
    y = df["Unnamed: 2"].astype(int).to_numpy()
    coords = df[["O_Lat", "O_Lng"]].to_numpy(float)
    return df, features, X, y, coords


def load_random_folds():
    f = pd.read_csv(DATA / "cv" / "classification_folds.csv", encoding="utf-8-sig")
    f = f.sort_values("row_index").reset_index(drop=True)
    return [("fixed5", f.fold.to_numpy(int))]


def make_spatial_folds(coords, y):
    out = []
    for seed in SPATIAL_SEEDS:
        km = KMeans(n_clusters=SPATIAL_K, n_init=10, random_state=seed)
        groups = km.fit_predict(coords)
        fold = np.full(len(y), -1, dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=SPATIAL_N_SPLITS, shuffle=True,
                                    random_state=seed)
        for i, (_, va) in enumerate(sgkf.split(coords, y, groups=groups)):
            fold[va] = i
        out.append((f"seed{seed}", fold))
    return out, groups


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def make_models():
    return {
        "LR": Pipeline([("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("model", SVC(kernel="rbf", probability=True, random_state=42))]),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                     class_weight="balanced"),
    }


def _score_model(model, Xva, yva):
    pred = model.predict(Xva)
    proba = model.predict_proba(Xva)[:, 1]
    return {
        "accuracy": accuracy_score(yva, pred),
        "f1": f1_score(yva, pred, zero_division=0),
        "recall": recall_score(yva, pred, zero_division=0),
        "roc_auc": roc_auc_score(yva, proba),
    }


def select_columns(Xtr, ytr, variant, k, greedy_cache):
    """Return column indices for the requested variant/K (greedy is cached)."""
    if variant == "univariate":
        return select_fs_cs(Xtr, ytr, k)
    if variant == "greedy":
        if greedy_cache is None:
            greedy_cache = select_fs_cs_greedy(Xtr, ytr, max(K_VALUES))
        return greedy_cache[:k]
    raise ValueError(variant)


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def nested_eval(X, y, scheme_tag, fold_vectors, models):
    rows = []
    for seed_tag, fv in fold_vectors:
        for fold_id in np.unique(fv):
            va = fv == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue
            greedy_cache = None
            for variant in VARIANTS:
                for k in K_VALUES:
                    if variant == "greedy" and greedy_cache is None:
                        greedy_cache = select_fs_cs_greedy(X[tr], y[tr], max(K_VALUES))
                    cols = select_columns(X[tr], y[tr], variant, k, greedy_cache)
                    Xtr, Xva = X[tr][:, cols], X[va][:, cols]
                    for mname, base in models.items():
                        m = clone(base)
                        m.fit(Xtr, y[tr])
                        rec = {"scheme": scheme_tag, "seed": seed_tag,
                               "fold": int(fold_id), "variant": variant, "k": k,
                               "model": mname, "n_features": len(cols),
                               "n_train": int(tr.sum()), "n_valid": int(va.sum())}
                        rec.update(_score_model(m, Xva, y[va]))
                        rows.append(rec)
    return pd.DataFrame(rows)


def derive_global_sets(X, y, fold_vector, features):
    """Aggregate per-fold rankings (over all folds) -> fixed feature sets."""
    agg = {}
    for variant in VARIANTS:
        rank_sum = np.zeros(X.shape[1])
        for fold_id in np.unique(fold_vector):
            tr = fold_vector != fold_id
            if variant == "univariate":
                order = rank_fs_cs(X[tr], y[tr])
            else:
                order = select_fs_cs_greedy(X[tr], y[tr], max(K_VALUES))
            r = np.empty(X.shape[1])
            r[order] = np.arange(1, len(order) + 1)
            rank_sum += r
        final = np.argsort(rank_sum)
        for k in K_VALUES:
            agg[(variant, k)] = final[:k]
    return agg


def load_named_sets(features):
    """Existing selector feature sets (for an apples-to-apples spatial test)."""
    sets = {
        "all_117": list(features),
        "current_2": [f for f in ("NREDI1", "NDVIre2") if f in features],
    }
    fs_dir = ROOT / "results" / "feature_selection"
    fstab = fs_dir / "feature_stability.csv"
    bstab = fs_dir / "boruta_stability.csv"
    if fstab.exists():
        s = pd.read_csv(fstab, encoding="utf-8-sig")
        sets["rfmi_top3of5"] = s.loc[s.selection_rate >= 0.6, "feature"].tolist()
    if bstab.exists():
        b = pd.read_csv(bstab, encoding="utf-8-sig")
        sets["boruta_3of5"] = b.loc[b.confirmed_rate >= 0.6, "feature"].tolist()
    return sets


def bench_eval(X, y, fold_vectors, named_cols, models, scheme_tag):
    """Evaluate a dict {set_name: [feature names]} under the given folds."""
    rows = []
    for name, cols in named_cols.items():
        idx = np.asarray([FEAT_INDEX[f] for f in cols if f in FEAT_INDEX], dtype=int)
        for seed_tag, fv in fold_vectors:
            for fold_id in np.unique(fv):
                va = fv == fold_id
                tr = ~va
                if tr.sum() < 20 or va.sum() < 5:
                    continue
                Xtr, Xva = X[tr][:, idx], X[va][:, idx]
                for mname, base in models.items():
                    m = clone(base)
                    m.fit(Xtr, y[tr])
                    rec = {"scheme": scheme_tag, "set": name, "model": mname,
                           "n_features": len(idx), "seed": seed_tag,
                           "fold": int(fold_id)}
                    rec.update(_score_model(m, Xva, y[va]))
                    rows.append(rec)
    return pd.DataFrame(rows)


def global_eval(X, y, fold_vector, columns_by_set, models):
    rows = []
    for (variant, k), cols in columns_by_set.items():
        for fold_id in np.unique(fold_vector):
            va = fold_vector == fold_id
            tr = ~va
            Xtr, Xva = X[tr][:, cols], X[va][:, cols]
            for mname, base in models.items():
                m = clone(base)
                m.fit(Xtr, y[tr])
                rec = {"variant": variant, "k": k, "set": f"fs_cs_{variant}_top{k}",
                       "model": mname, "fold": int(fold_id), "n_features": len(cols)}
                rec.update(_score_model(m, Xva, y[va]))
                rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="random scheme only")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df, features, X, y, coords = load_train()
    global FEAT_INDEX
    FEAT_INDEX = {name: i for i, name in enumerate(features)}
    print(f"[data] train={X.shape[0]} x {X.shape[1]}  | label counts "
          f"{np.bincount(y).tolist()}")

    random_folds = load_random_folds()
    models = make_models()

    # ---- 1. per-fold FS-CS scores + stability (univariate, on random folds) ---
    fv0 = random_folds[0][1]
    fold_rows = []
    for fold_id in np.unique(fv0):
        tr = fv0 != fold_id
        j = fs_cs_score(X[tr], y[tr])
        order = np.argsort(-j, kind="stable")
        rank = np.empty(len(features))
        rank[order] = np.arange(1, len(features) + 1)
        for i, name in enumerate(features):
            fold_rows.append({"fold": int(fold_id), "feature": name,
                              "j_score": float(j[i]), "rank": int(rank[i]),
                              "in_top10": rank[i] <= 10,
                              "in_top20": rank[i] <= 20,
                              "in_top30": rank[i] <= 30})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "fs_cs_fold_results.csv", index=False, encoding="utf-8-sig")

    stab = (fold_df.groupby("feature")
            .agg(mean_rank=("rank", "mean"),
                 top10_folds=("in_top10", "sum"),
                 top20_folds=("in_top20", "sum"),
                 top30_folds=("in_top30", "sum"),
                 mean_j=("j_score", "mean"))
            .reset_index()
            .sort_values("mean_rank"))
    stab.to_csv(OUT / "fs_cs_stability.csv", index=False, encoding="utf-8-sig")
    print("\n[FS-CS top-15 by mean rank across folds]")
    print(stab.head(15).to_string(index=False))

    # ---- 2. nested evaluation (A_random) --------------------------------------
    nested_a = nested_eval(X, y, "A_random", random_folds, models)
    print(f"\n[nested] A_random rows={len(nested_a)}")

    # ---- 3. nested evaluation (B_spatial) -------------------------------------
    if not args.quick:
        spatial_folds, blocks = make_spatial_folds(coords, y)
        pd.DataFrame({"row": np.arange(len(y)), "block": blocks,
                      "label": y}).to_csv(OUT / "spatial_blocks_k10.csv",
                                          index=False, encoding="utf-8-sig")
        nested_b = nested_eval(X, y, "B_spatial", spatial_folds, models)
        print(f"[nested] B_spatial rows={len(nested_b)}")
    else:
        nested_b = pd.DataFrame(columns=nested_a.columns)

    if nested_b.empty:
        nested = nested_a.copy()
    else:
        nested = pd.concat([nested_a, nested_b], ignore_index=True)
    nested.to_csv(OUT / "nested_per_fold_metrics.csv", index=False, encoding="utf-8-sig")

    nested_summary = (nested.groupby(["scheme", "variant", "k", "model"])
                      .agg(n_features=("n_features", "first"),
                           n_eval=("fold", "size"),
                           accuracy=("accuracy", "mean"),
                           accuracy_std=("accuracy", "std"),
                           f1=("f1", "mean"), recall=("recall", "mean"),
                           roc_auc=("roc_auc", "mean"),
                           roc_auc_std=("roc_auc", "std"))
                      .reset_index())
    nested_summary.to_csv(OUT / "nested_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[nested summary — A_random]")
    print(nested_summary[nested_summary.scheme == "A_random"]
          .sort_values("roc_auc", ascending=False).head(8).to_string(index=False))

    # ---- 4. gap table (A_random − B_spatial) ----------------------------------
    if not nested_b.empty:
        piv = nested_summary.pivot_table(
            index=["variant", "k", "model"],
            columns="scheme", values="roc_auc").reset_index()
        piv["gap_A_minus_B"] = piv["A_random"] - piv["B_spatial"]
        piv = piv.rename(columns={"A_random": "A_random_roc_auc",
                                  "B_spatial": "B_spatial_roc_auc"})
        piv.to_csv(OUT / "gap_table.csv", index=False, encoding="utf-8-sig")
        print("\n[gap table — ROC-AUC, A_random − B_spatial]")
        print(piv.sort_values("gap_A_minus_B", ascending=False).to_string(index=False))

    # ---- 5. global sets (compare with existing selectors) --------------------
    cols_by_set = derive_global_sets(X, y, fv0, features)
    glob = global_eval(X, y, fv0, cols_by_set, models)
    glob.to_csv(OUT / "global_per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    glob_summary = (glob.groupby(["set", "variant", "k", "model"])
                    .agg(n_features=("n_features", "first"),
                         accuracy=("accuracy", "mean"),
                         f1=("f1", "mean"), recall=("recall", "mean"),
                         roc_auc=("roc_auc", "mean"),
                         roc_auc_std=("roc_auc", "std"))
                    .reset_index()
                    .sort_values("roc_auc", ascending=False))
    glob_summary.to_csv(OUT / "global_sets_summary.csv", index=False,
                        encoding="utf-8-sig")
    print("\n[global sets summary — top 10 by ROC-AUC]")
    print(glob_summary.head(10).to_string(index=False))

    # ---- 6. selection sets (traceability) ------------------------------------
    sel_rows = []
    for (variant, k), cols in cols_by_set.items():
        sel_rows.append({"variant": variant, "k": k, "n_features": len(cols),
                         "features": ", ".join(features[i] for i in cols)})
    pd.DataFrame(sel_rows).to_csv(OUT / "selection_sets.csv", index=False,
                                  encoding="utf-8-sig")

    # ---- 7. selector benchmark: same fixed sets under both schemes -----------
    named = load_named_sets(features)
    for (variant, k), cols in cols_by_set.items():
        named[f"fs_cs_{variant}_top{k}"] = [features[i] for i in cols]
    bench_frames = [bench_eval(X, y, random_folds, named, models, "A_random")]
    if not args.quick:
        bench_frames.append(bench_eval(X, y, spatial_folds, named, models, "B_spatial"))
    bench = pd.concat(bench_frames, ignore_index=True)
    bench.to_csv(OUT / "selector_benchmark_per_fold.csv", index=False,
                 encoding="utf-8-sig")

    bs = bench.groupby(["set", "scheme", "model", "n_features"], as_index=False).agg(
        accuracy=("accuracy", "mean"), roc_auc=("roc_auc", "mean"))
    recs = []
    for (sname, mname, nf), g in bs.groupby(["set", "model", "n_features"]):
        a = g[g.scheme == "A_random"]
        b = g[g.scheme == "B_spatial"]
        rec = {"set": sname, "model": mname, "n_features": int(nf),
               "accuracy_A": float(a.accuracy.iloc[0]) if len(a) else np.nan,
               "accuracy_B": float(b.accuracy.iloc[0]) if len(b) else np.nan,
               "roc_auc_A": float(a.roc_auc.iloc[0]) if len(a) else np.nan,
               "roc_auc_B": float(b.roc_auc.iloc[0]) if len(b) else np.nan}
        rec["accuracy_gap"] = rec["accuracy_A"] - rec["accuracy_B"]
        rec["roc_auc_gap"] = rec["roc_auc_A"] - rec["roc_auc_B"]
        recs.append(rec)
    wide = pd.DataFrame(recs).sort_values(["model", "roc_auc_gap"])
    wide.to_csv(OUT / "selector_benchmark.csv", index=False, encoding="utf-8-sig")
    print("\n[selector benchmark — ROC-AUC, A_random vs B_spatial]")
    print(wide.to_string(index=False))

    meta = {
        "n_train": int(X.shape[0]), "n_features": int(X.shape[1]),
        "k_values": list(K_VALUES), "variants": list(VARIANTS),
        "spatial_k": SPATIAL_K, "spatial_n_splits": SPATIAL_N_SPLITS,
        "spatial_seeds": list(SPATIAL_SEEDS),
        "test_set_used": False,
        "reference": "Sun et al. 2025, European Journal of Agronomy "
                     "(PII: S1161030125001984)",
    }
    (OUT / "experiment_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] outputs -> {OUT}")


if __name__ == "__main__":
    main()
