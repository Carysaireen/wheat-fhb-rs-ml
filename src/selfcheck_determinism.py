"""Determinism audit for the selector zoo.

The problem this guards against
------------------------------
Early versions used sklearn's `liblinear` solver for L1 logistic regression.
liblinear solves the L1 problem with a random coordinate permutation and
exposes no seed, so the SAME data can yield numerically equivalent but
differently ORDERED coefficients. Because every selector here keeps the top-k
by |score|, a different ordering can change the selected feature SET -- and
therefore the downstream accuracy -- between two runs of the same experiment.

The failure mode that exposed it: stage-2 runs the k-loop {5,10,20,30} while
stage-1 runs only {10}. Different call ORDER -> different liblinear internal
state -> the k=10 slice of stage 2 disagreed with stage 1 on 8/550 rows.

What this script checks
-----------------------
Run every selector twice inside one process:
  * pass A : k_list = [10]           (stage-1 call order)
  * pass B : k_list = [5,10,20,30]   (stage-2 call order)
and compare the feature sets returned for k=10. Any "DIFF" means a selector
still carries hidden state / randomness and its numbers are not reproducible.

Run:
  python src/selfcheck_determinism.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# the driver silences these; do the same so the report is readable
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import selector_zoo as zoo  # noqa: E402


def load():
    df = pd.read_csv(ROOT / "data" / "splits" / "train.csv",
                     encoding="utf-8-sig")
    X = df[df.columns[9:]].to_numpy(float)
    y_cls = df["Unnamed: 2"].astype(int).to_numpy()
    y_reg = df["DI"].to_numpy(float)
    return X, y_cls, y_reg


def sweep(X, y, selectors, task, k_list, mask):
    Xs, ys = X[mask], y[mask]
    ctx = {"task": task, "seed": 42, "mi_matrix": None, "_cache": {}}
    out = {}
    for k in k_list:
        for name, fn in selectors.items():
            try:
                out[(name, k)] = tuple(np.sort(fn(Xs, ys, k, ctx)).tolist())
            except Exception as e:
                out[(name, k)] = f"ERR {type(e).__name__}"
    return out


def main():
    X, y_cls, y_reg = load()
    clf_mask = np.ones(len(X), bool)                 # all 132 rows
    reg_mask = y_reg > 0                             # 116 rows

    ok = True
    for task, y, selectors, mask in (
            ("clf", y_cls, zoo.CLF_SELECTORS, clf_mask),
            ("reg", y_reg, zoo.REG_SELECTORS, reg_mask)):
        a = sweep(X, y, selectors, task, [10], mask)
        b = sweep(X, y, selectors, task, [5, 10, 20, 30], mask)
        print(f"\n=== {task} : k=10, stage-1 order vs stage-2 order ===")
        for name in selectors:
            ka, kb = a[(name, 10)], b[(name, 10)]
            same = ka == kb
            ok &= same
            tag = "SAME" if same else "DIFF"
            extra = ""
            if not same:
                sa, sb = set(ka), set(kb)
                extra = f"  |A\\B|={len(sa - sb)} |B\\A|={len(sb - sa)}"
            print(f"  {tag}  {name:12s} top10={list(ka)[:10]}{extra}")

    print("\n" + "=" * 60)
    print("RESULT: " + ("PASS -- every selector is reproducible."
                        if ok else "FAIL -- hidden randomness remains."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
