"""
阶段⑤ 自校验（stage5 self-check）
==================================

【要证明什么】
阶段⑤ 新开了一份 run_track 实现（run_stage5_tuned_matrix.run_track_tuned），
目的是能在**不改变阶段②/④ 对照产出**的前提下注入调优参数。新代码就有风险，
所以必须证明：**把模型工厂换成"出厂默认参数"时，它逐位复现阶段②的 k=10 切片**。

做法：
  1. 取阶段② clf_per_fold_stage2.csv / reg_per_fold_stage2.csv 的 k=10 子集
     （scheme = A_random，seed = 42）作为"金标准"。
  2. 用阶段⑤ 的 run_track_tuned，但把模型工厂换成 hyperparam_grids 网格的
     **第 0 号配置**（= run_selector_zoo.clf_models / reg_models 的默认值），
     在完全相同的 row_mask / folds / selectors / k 上重跑。
  3. 逐 (selector, model, fold) 比对 accuracy / roc_auc / r2 等指标，报 max|Δ|。

【为什么这一条比"再跑一遍阶段⑤"更有用】
"再跑一遍阶段⑤"只能证明阶段⑤ 自身确定（同输入同输出），无法排除
"两份代码都自洽、但语义不同"。与阶段② 比对才能证明**两份实现语义等价**。

用法：
    python -u src/selfcheck_stage5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_selector_zoo as R                        # noqa: E402
import selector_zoo as zoo                          # noqa: E402
import hyperparam_grids as G                        # noqa: E402
from run_stage5_tuned_matrix import (               # noqa: E402
    run_track_tuned, PIPELINE_MODELS)

from sklearn.pipeline import Pipeline                # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402

ZOO_OUT = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"
TOL = 1e-9


def default_factory(task: str):
    """返回 models(seed) -> 估计器，全部用网格**第 0 号配置**（= 出厂默认），
    并按 run_stage5_tuned_matrix.PIPELINE_MODELS 的同一规则决定是否包 Pipeline。

    这是"用阶段⑤ 的循环 + 阶段② 的语义"重跑，从而把'循环实现'与
    '参数来源'两个变量分离开。
    """
    grids = G.CLF_GRIDS if task == "clf" else G.REG_GRIDS
    build = G.build_clf if task == "clf" else G.build_reg
    defaults = {name: grid[0] for name, grid in grids.items()}
    need_pipe = PIPELINE_MODELS[task]

    def _factory(seed: int) -> dict:
        out = {}
        for name, p in defaults.items():
            est = build(name, **p)
            if name in need_pipe:
                inner = est.named_steps["m"] if hasattr(est, "named_steps") else est
                if "random_state" in inner.get_params():
                    inner.set_params(random_state=seed)
                out[name] = Pipeline([("sc", StandardScaler()), ("m", inner)])
            else:
                if "random_state" in est.get_params():
                    est.set_params(random_state=seed)
                out[name] = est
        return out

    return _factory


def compare(task: str) -> tuple[bool, str]:
    gold = pd.read_csv(ZOO_OUT / f"{task}_per_fold_stage2.csv", encoding="utf-8-sig")
    gold = gold[(gold.k == 10) & (gold.scheme == "A_random") & (gold.seed == 42)]
    gold = gold.drop_duplicates(["selector", "model", "fold"])

    df, names, X, y_cls, y_reg, coords = R.load_train()
    y = y_cls if task == "clf" else y_reg
    row_mask = np.ones(len(df), bool) if task == "clf" else (y > 0)
    clf_fold, reg_fold = R.load_random_folds(df)
    folds = [("A_random", clf_fold if task == "clf" else reg_fold)]

    sel = dict(zoo.CLF_SELECTORS) if task == "clf" else dict(zoo.REG_SELECTORS)
    # 阶段② 用的是同一套注册表；这里再与 gold 实际跑过的选择器取交集，
    # 避免因版本差异多跑/少跑而误判
    sel = {k: v for k, v in sel.items() if k in set(gold.selector.unique())}
    _base = default_factory(task)
    keep_models = set(gold.model.unique())

    def models(seed: int) -> dict:
        return {k: v for k, v in _base(seed).items() if k in keep_models}

    mine = run_track_tuned(task, X, y, row_mask, folds, sel, models, (10,), (42,))
    met = ["accuracy", "f1", "recall", "roc_auc"] if task == "clf" \
        else ["r2", "rmse", "mae"]

    key = ["selector", "model", "fold"]
    m = mine.set_index(key)[met]
    g = gold.set_index(key)[met]
    common = m.index.intersection(g.index)
    if len(common) == 0:
        return False, "no overlapping rows -- cannot compare"
    d = (m.loc[common] - g.loc[common]).abs()
    worst = float(d.to_numpy().max())
    coverage = f"rows mine={len(mine)} gold(k=10)={len(gold)} common={len(common)}"
    ok = worst <= TOL
    return ok, (f"{coverage} | max|delta| = {worst:.3e} "
                f"({'PASS' if ok else 'FAIL'})")


def main() -> None:
    print("=" * 72)
    print("stage5 self-check :: 默认参数下是否逐位复现阶段② 的 k=10 切片")
    print("=" * 72)
    all_ok = True
    for task in ("clf", "reg"):
        ok, msg = compare(task)
        all_ok &= ok
        print(f"[{task}] {msg}")
    print("-" * 72)
    print("RESULT:", "PASS -- stage5 loop is faithful to stage2" if all_ok
          else "FAIL -- stage5 loop differs from stage2, DO NOT TRUST stage5")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
