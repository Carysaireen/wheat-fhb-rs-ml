"""
确定性审计（champion profile）
================================

【为什么必须有这个脚本】
项目已经栽过一次：sklearn 的 `liblinear` 求解器解 L1 逻辑回归时用**随机坐标
顺序**且不暴露种子，导致同一份数据两次拟合给出"数值等价、系数排序不同"的解，
而本项目所有选择器都按 |coef| 取前 k -> **选出的特征集变了** -> 结论翻转。

所以铁律是：**每跑完新阶段，必须做同一输入重复跑 + 逐位比对。**

【本脚本检查什么】
1. 把 current_2 的**全部配置**在同一个折上跑 2 遍，逐位比对指标。
2. 检查"调用次序"是否影响结果：先跑大网格再跑小子集 vs 直接跑子集，
   两者必须一致（这是 liblinear 事故的核心特征）。
3. 报告不支持的随机种子设置（SVR/PLSR/LDA 是确定性估计器）。

用法：
    python -u src/selfcheck_champion.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import run_selector_zoo as R          # noqa: E402
import hyperparam_grids as G          # noqa: E402
from run_champion_profile import eval_one, fixed_feature_sets   # noqa: E402

TOL = 1e-12


def main() -> None:
    print("=" * 70)
    print("determinism audit :: champion profile")
    print("=" * 70)

    df, names, X, y_cls, y_reg, coords = R.load_train()
    sets = fixed_feature_sets(names)
    ok_all = True

    for task, y, metric in (("clf", y_cls, "roc_auc"), ("reg", y_reg, "r2")):
        grids = G.CLF_GRIDS if task == "clf" else G.REG_GRIDS
        build = G.build_clf if task == "clf" else G.build_reg
        grids = {k: v for k, v in grids.items()
                 if all(p.get("n_components", 0) <= 2 for p in v)}  # current_2 可行集

        row_mask = np.ones(len(df), bool) if task == "clf" else (y > 0)
        clf_f, reg_f = R.load_random_folds(df)
        fvec = (clf_f if task == "clf" else reg_f)[row_mask]
        Xt, yt = X[row_mask], y[row_mask]
        idx = [names.index(f) for f in sets["current_2"]]

        def sweep(order=None):
            """跑一遍全部配置，返回 {label: score}。order 控制调用次序。"""
            out = {}
            items = [(m, cid, p) for m, g in grids.items()
                     for cid, p in enumerate(g)]
            if order is not None:
                items = order
            for fold_id in np.unique(fvec):
                if fold_id < 0:
                    continue
                va = fvec == fold_id
                tr = ~va
                if tr.sum() < 20 or va.sum() < 5:
                    continue
                Xtr, Xva = Xt[tr][:, idx], Xt[va][:, idx]
                for m, cid, p in items:
                    mdl = build(m, **p)
                    sc = eval_one(mdl, Xtr, yt[tr], Xva, yt[va], task)
                    out[(fold_id, m, cid)] = sc[metric]
            return out

        a = sweep()
        b = sweep()
        # 逐位比对
        keys = sorted(set(a) | set(b))
        diff = [abs(a.get(k, np.nan) - b.get(k, np.nan)) for k in keys]
        worst = max([d for d in diff if np.isfinite(d)], default=0.0)
        flag = "PASS" if worst <= TOL else "FAIL"
        if worst > TOL:
            ok_all = False
        print(f"[{task}] run1 vs run2 : n={len(keys)}  max|Δ| = {worst:.3e}  {flag}")

        # 次序无关性：倒序调用必须得到同一结果
        items_rev = [(m, cid, p) for m, g in reversed(list(grids.items()))
                     for cid, p in reversed(list(enumerate(g)))]
        c = sweep(order=items_rev)
        diff2 = [abs(a.get(k, np.nan) - c.get(k, np.nan)) for k in keys
                 if k in a and k in c]
        worst2 = max([d for d in diff2 if np.isfinite(d)], default=0.0)
        flag2 = "PASS" if worst2 <= TOL else "FAIL"
        if worst2 > TOL:
            ok_all = False
        print(f"[{task}] order-independence (reversed) : max|Δ| = {worst2:.3e}  {flag2}")

        # 确定性估计器提示
        det = []
        for m in grids:
            mdl = build(m, **grids[m][0])
            if "random_state" not in mdl.get_params():
                det.append(m)
        print(f"[{task}] estimators WITHOUT random_state (deterministic by design): "
              f"{det if det else 'none'}")

    print("-" * 70)
    print(f"RESULT: {'PASS' if ok_all else 'FAIL'} -- champion profile is reproducible"
          if ok_all else "RESULT: FAIL -- determinism broken, investigate before use")


if __name__ == "__main__":
    main()
