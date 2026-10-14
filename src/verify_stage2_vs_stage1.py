"""Regression check: stage-2's k=10 slice must equal stage-1 exactly.

Why this exists
---------------
Stage 2 reuses the SAME data, folds, seed (42) and grid as stage 1, and only
adds three more K values. Therefore every row with k == 10 in
`*_per_fold_stage2.csv` must be bit-for-bit reproducible against
`*_per_fold_stage1.csv`.

This is also the guard for a speed-up we introduced at the same time: some
selectors (lasso / enet / perm) now cache a *k-independent* intermediate
(their score vector) and slice it with _topk(score, k). If that cache were
wrong, the k=10 numbers would drift away from stage 1 -- so a clean diff here
means the cache is faithful.

Run:
  python src/verify_stage2_vs_stage1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"

KEYS = ["selector", "model", "fold"]
METRICS = {
    "clf": ["accuracy", "f1", "recall", "roc_auc"],
    "reg": ["r2", "rmse", "mae"],
}


def compare(task):
    s1 = pd.read_csv(OUT / f"{task}_per_fold_stage1.csv", encoding="utf-8-sig")
    s2 = pd.read_csv(OUT / f"{task}_per_fold_stage2.csv", encoding="utf-8-sig")
    s2k = s2[s2["k"] == 10].copy()

    print(f"\n=== {task} : k=10 ===")
    print(f"  stage1 rows = {len(s1)}   stage2(k=10) rows = {len(s2k)}")

    chk = s1[KEYS + METRICS[task] + ["n_features"]].merge(
        s2k[KEYS + METRICS[task] + ["n_features"]],
        on=KEYS, suffixes=("_1", "_2"), how="inner")
    if len(chk) != len(s1):
        print(f"  !! merged rows {len(chk)} != stage1 rows {len(s1)}")

    same_feat = (chk["n_features_1"] == chk["n_features_2"]).all()
    print(f"  n_features identical = {bool(same_feat)}")

    main = "accuracy" if task == "clf" else "r2"
    ok = True
    for m in METRICS[task]:
        d = (chk[f"{m}_1"] - chk[f"{m}_2"]).abs()
        nbad = int((d > 1e-12).sum())
        flag = "OK " if nbad == 0 else "DIFF"
        if nbad:
            ok = False
        print(f"  {flag} {m:9s} n_diff={nbad}/{len(chk)}  max|Δ|={d.max():.3e}")

    # where do the differences live?
    chk["_d"] = (chk[f"{main}_1"] - chk[f"{main}_2"]).abs()
    g = chk.groupby("selector")["_d"].agg(
        n_diff=lambda x: int((x > 1e-12).sum()), max_diff="max")
    g = g[g["n_diff"] > 0].sort_values("max_diff", ascending=False)
    if len(g):
        print(f"\n  -- differing rows by selector ({main}) --")
        print(g.round(4).to_string())
        gm = chk.groupby("model")["_d"].agg(
            n_diff=lambda x: int((x > 1e-12).sum()), max_diff="max")
        gm = gm[gm["n_diff"] > 0].sort_values("max_diff", ascending=False)
        print(f"\n  -- differing rows by model ({main}) --")
        print(gm.round(4).to_string())
        worst = chk.sort_values("_d", ascending=False).head(8)
        print("\n  -- worst 8 rows --")
        print(worst[KEYS + [f"{main}_1", f"{main}_2", "_d"]]
              .round(4).to_string(index=False))
    return ok


def main():
    ok_c = compare("clf")
    ok_r = compare("reg")
    print("\n" + "=" * 56)
    if ok_c and ok_r:
        print("RESULT: PASS -- stage2 k=10 reproduces stage1 exactly.")
    else:
        print("RESULT: FAIL -- investigate before trusting stage2.")
    return 0 if (ok_c and ok_r) else 1


if __name__ == "__main__":
    sys.exit(main())
