"""
阶段⑤ · 调优参数下的「选择器 × 模型」全矩阵
=============================================

【这个实验在回答什么问题】
阶段①/②/④ 跑的是「选择器 × 模型」矩阵，但**模型全部用的是出厂默认超参**。
而阶段⑤之前的擂主档案已经证明：在 current_2 上，RBF-SVR 的默认参数
（C=1, eps=0.1）对 0–100 尺度的 DI 是**严重欠拟合**（0.260 -> 0.4916 一提就飞），
XGBoost 的默认参数在分类上也是全场垫底。于是有一个很自然的质疑：

    「你们说'选择器几乎没用'，会不会只是因为模型根本没调好？」

阶段⑤就是来正面回答它的：**把每个模型换成在 current_2 上专属调优过的参数，
再把整张「选择器 × 模型」矩阵重跑一遍**，然后逐条拿去挑战 current_2 擂主。

【★ 单一变量控制（硬约束）】
与阶段②/④ 相比，**唯一允许变化的量 = 模型的超参来源**：
    · 阶段②/④ : 出厂默认超参（hyperparam_grids 里每个网格的第 0 号）
    · 阶段⑤   : current_2 上 LOFO 诚实口径挑出的最优超参
      （champion_best_params.json，由 run_champion_profile.py STAGE1 产出）
其余一切逐字对齐：同一份 train.csv、同一套折向量、同一份选择器实现、
同一个 K 网格 (5/10/20/30)、同一个打分函数、同一个分区种子。
所以「阶段⑤ vs 阶段②/④」的差 = **纯粹的调参效应**，可以在 analyze 阶段直接做配对比较。

⚠️ 一个必须说清的口径细节：调优参数是在 **current_2（2 特征）** 上选的，而阶段⑤
会把它用到 K=5/10/20/30 的特征集上。这**不是**"每个 K 各自调参"（那要给每个 K
单独扫一遍网格，成本 ×4，且属于超参扫描的延伸）。本阶段要回答的是
"这批参数是否让矩阵整体变强"，所以**故意**固定用 current_2 的一组参数，
把它当成"一个被调过的模型族"。这一点在报告里必须写明，否则会被误读成嵌套调参。

【用法】
    python -u src/run_stage5_tuned_matrix.py --smoke
    python -u src/run_stage5_tuned_matrix.py --random   # 仅随机折
    python -u src/run_stage5_tuned_matrix.py            # 全量（随机+空间）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_selector_zoo as R                       # noqa: E402
import selector_zoo as zoo                         # noqa: E402
import hyperparam_grids as G                       # noqa: E402

from sklearn.base import clone                     # noqa: E402
from sklearn.pipeline import Pipeline              # noqa: E402
from sklearn.preprocessing import StandardScaler   # noqa: E402

EXP = ROOT / "experiments" / "2026-10-13_stage5_tuned"
OUT = EXP / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RES = ROOT / "results" / "stage5_tuned"
RES.mkdir(parents=True, exist_ok=True)

PROFILE_OUT = (ROOT / "experiments" / "2026-10-13_champion_profile" / "outputs")

# --------------------------------------------------------------------------- #
# 实验常量（与全项目对齐）
# --------------------------------------------------------------------------- #
K_VALUES = (5, 10, 20, 30)
SMOKE_K = (10,)
SPATIAL_SEEDS = (42, 7, 2026)      # 与 P0-② / FS-CS / 擂主档案一致
RANDOM_SEEDS = (42,)               # 与阶段②/④ 保持一致的单模型种子

_PROGRESS = {"file": None}


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if _PROGRESS["file"] is not None:
        with open(_PROGRESS["file"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# 调优参数加载
# --------------------------------------------------------------------------- #
# ★ 需要"模型内部自带标准化"的模型 —— 必须与 run_selector_zoo.clf_models /
#   reg_models 的包装方式**逐一对应**。
#
# 为什么这件事是致命的：selector_zoo 的模型库里，LR/KNN/SVM/NB/LDA/PLSDA/
# SVMLin/Ridge/SVR/PLSR/ENet/GPR 都被包在 Pipeline([("sc", StandardScaler()),
# ("m", ...)]) 里，而 RF/ET/XGB 是裸估计器。也就是说"阶段②的默认参数结果"
# 里**已经含了折内标准化**。阶段⑤ 若直接喂原始特征给裸的 LR，等于偷偷改掉了
# 预处理 —— 那就不是"单一变量 = 超参来源"了。所以这里必须把同一批模型
# 包回 Pipeline，保证预处理逐字一致。
PIPELINE_MODELS = {
    "clf": {"LR", "SVM", "SVMLin", "LDA", "PLSDA", "KNN", "NB"},
    "reg": {"Ridge", "SVR", "PLSR", "ENet", "KNN", "GPR"},
}


def load_tuned_params(path: Path) -> dict:
    """读 champion_best_params.json -> {task: {model: {param: value}}}。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    return d


def model_factory(task: str, tuned: dict):
    """返回 models(seed) -> {model_name: estimator}，其中每个估计器都用
    **调优参数**构建，并把 random_state 设成当前种子（能设的才设）。

    ★ 包装方式与 run_selector_zoo 的模型库逐字对齐：
      · 在 PIPELINE_MODELS 里的 -> Pipeline([("sc", StandardScaler()), ("m", est)])
      · 其余（RF / ET / XGB）-> 裸估计器
    这样"阶段⑤ vs 阶段②"的差才真的是**超参来源**这一项。
    """
    build = G.build_clf if task == "clf" else G.build_reg
    need_pipe = PIPELINE_MODELS[task]

    def _factory(seed: int) -> dict:
        out = {}
        for mname, p in tuned.items():
            try:
                est = build(mname, **p)
            except Exception as e:                    # pragma: no cover
                _log(f"  [warn] build {task}/{mname} failed: {type(e).__name__}: {e}")
                continue
            # 先设种子（Pipeline 里 random_state 在 m 上；裸估计器在自身）
            if mname in need_pipe:
                inner = est.named_steps["m"] if hasattr(est, "named_steps") else est
                if "random_state" in inner.get_params():
                    inner.set_params(random_state=seed)
                out[mname] = Pipeline([("sc", StandardScaler()), ("m", inner)])
            else:
                if "random_state" in est.get_params():
                    est.set_params(random_state=seed)
                out[mname] = est
        return out

    return _factory


# --------------------------------------------------------------------------- #
# 核心循环（复刻 run_selector_zoo.run_track，但模型工厂可注入）
# --------------------------------------------------------------------------- #
def run_track_tuned(task, X, y, row_mask, fold_sets, selectors, models,
                    k_list, seeds):
    """与 run_selector_zoo.run_track 逐字等价，唯一差别是 models 用调优参数。

    刻意不改 run_track 本体：阶段②/④ 的产出要作为**对照组**保留原样，
    任何对照都必须"两边都没动过"。所以这里开一份新实现，并靠自校验
    （--random 与阶段② k=10 的默认参数切片比对）证明两者一致。
    """
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
                _log(f"[{task}]   computing MI redundancy matrix "
                     f"({int(tr.sum())}x{Xt.shape[1]}) ...")
                mi = zoo.compute_mi_matrix(Xt[tr])
                _log(f"[{task}]   MI matrix done")
            else:
                mi = None
            for seed in seeds:
                mmodels = models(seed)
                # 一个 (fold, seed) 一个 ctx -> CARS/RFE/stability 的缓存跨 K 复用
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
                                sc = (R.clf_scores(mdl, Xva, yt[va]) if task == "clf"
                                      else R.reg_scores(mdl, Xva, yt[va]))
                            except Exception as e:
                                print(f"    [warn] fit {task}/{mname}: "
                                      f"{type(e).__name__}: {e}")
                                continue
                            rec = {"task": task, "scheme": scheme,
                                   "selector": sname, "model": mname, "k": k,
                                   "seed": seed, "fold": int(fold_id),
                                   "n_features": len(sel),
                                   "n_train": int(tr.sum()),
                                   "n_valid": int(va.sum())}
                            rec.update(sc)
                            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df, task):
    mets = ["accuracy", "f1", "recall", "roc_auc"] if task == "clf" \
        else ["r2", "rmse", "mae"]
    return df.groupby(["selector", "model", "k", "scheme"], as_index=False).agg(
        n_features=("n_features", "first"), n_eval=("fold", "size"),
        **{f"{m}_mean": (m, "mean") for m in mets},
        **{f"{m}_std": (m, "std") for m in mets})


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小网格 + 仅随机折")
    ap.add_argument("--random", action="store_true", help="全网格但仅随机折")
    ap.add_argument("--no-spatial", action="store_true", help="等价 --random")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    tuned = load_tuned_params(PROFILE_OUT / "champion_best_params.json")
    _log("=" * 72)
    _log("stage5 :: 调优参数下的「选择器 × 模型」全矩阵")
    _log(f"tuned source = {PROFILE_OUT / 'champion_best_params.json'}")
    _log(f"clf models = {list(tuned['clf'])}")
    _log(f"reg models = {list(tuned['reg'])}")
    _log("=" * 72)

    df, features, X, y_cls, y_reg, coords = R.load_train()
    n = len(df)
    clf_fold, reg_fold = R.load_random_folds(df)
    _log(f"data: train={n} x {X.shape[1]} | clf 0/1 = "
         f"{np.bincount(y_cls).tolist()} | DI>0 = {int((y_reg > 0).sum())} | "
         f"xgb={'yes' if R.HAS_XGB else 'NO'}")

    # ---- 配置模式 ----
    if args.smoke:
        clf_sel = {k: zoo.CLF_SELECTORS[k] for k in
                   ("relieff", "mrmr", "lasso", "fs_cs", "stab_fscs", "random")}
        reg_sel = {k: zoo.REG_SELECTORS[k] for k in
                   ("relieff", "mrmr", "lasso", "cars", "stab_lasso", "random")}
        clf_m = lambda s: {k: v for k, v in model_factory("clf", tuned["clf"])(s).items()
                           if k in ("LR", "RF", "NB")}
        reg_m = lambda s: {k: v for k, v in model_factory("reg", tuned["reg"])(s).items()
                           if k in ("Ridge", "RF", "SVR")}
        k_list = SMOKE_K
        seeds = (42,)
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        tag = "smoke"
    else:
        clf_sel = dict(zoo.CLF_SELECTORS)
        reg_sel = dict(zoo.REG_SELECTORS)
        clf_m = model_factory("clf", tuned["clf"])
        reg_m = model_factory("reg", tuned["reg"])
        k_list = K_VALUES
        seeds = RANDOM_SEEDS
        clf_folds = [("A_random", clf_fold)]
        reg_folds = [("A_random", reg_fold)]
        if not (args.random or args.no_spatial):
            # ★ 空间折构造逐字对齐阶段④ / 擂主档案（含回归的 DI>0 子集重建）
            for s in SPATIAL_SEEDS:
                f_c, _ = R.spatial_fold(coords, y_cls, s)
                clf_folds.append((f"B_spatial_s{s}", f_c))
                sick = y_reg > 0
                bins = pd.qcut(y_reg, 3, labels=False, duplicates="drop")
                f_r, _ = R.spatial_fold(coords, np.where(bins >= 0, bins, 0), s)
                reg_folds.append((f"B_spatial_s{s}", f_r))
        tag = "random" if (args.random or args.no_spatial) else "full"

    _PROGRESS["file"] = OUT / f"run_progress_stage5_{tag}.log"
    _PROGRESS["file"].write_text("", encoding="utf-8")
    _log(f"START mode={tag} k={list(k_list)} seeds={list(seeds)} "
         f"| clf_sel={len(clf_sel)} reg_sel={len(reg_sel)} "
         f"| clf_m={len(clf_m(42))} reg_m={len(reg_m(42))}")

    tt = time.time()
    print(f"\n=== clf track ({tag}) ===")
    clf_df = run_track_tuned("clf", X, y_cls, np.ones(n, bool), clf_folds,
                             clf_sel, clf_m, k_list, seeds)
    print(f"  rows={len(clf_df)}  ({time.time()-tt:.0f}s)")

    print(f"\n=== reg track ({tag}) ===")
    reg_df = run_track_tuned("reg", X, y_reg, y_reg > 0, reg_folds,
                             reg_sel, reg_m, k_list, seeds)
    print(f"  rows={len(reg_df)}  ({time.time()-tt:.0f}s)")

    clf_df.to_csv(OUT / f"clf_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
    reg_df.to_csv(OUT / f"reg_per_fold_{tag}.csv", index=False, encoding="utf-8-sig")
    cs = summarize(clf_df, "clf")
    rs = summarize(reg_df, "reg")
    cs.to_csv(OUT / f"clf_summary_{tag}.csv", index=False, encoding="utf-8-sig")
    rs.to_csv(OUT / f"reg_summary_{tag}.csv", index=False, encoding="utf-8-sig")

    # 顺手把 per-K 榜首打出来，跑的时候就能看
    print("\n[clf summary | per K, top-8 by roc_auc]")
    for kk in sorted(cs["k"].unique()):
        sub = cs[cs.k == kk].sort_values("roc_auc_mean", ascending=False).head(8)
        print(f"-- k={kk} --")
        print(sub[["selector", "model", "roc_auc_mean", "accuracy_mean"]]
              .round(4).to_string(index=False))
    print("\n[reg summary | per K, top-8 by r2]")
    for kk in sorted(rs["k"].unique()):
        sub = rs[rs.k == kk].sort_values("r2_mean", ascending=False).head(8)
        print(f"-- k={kk} --")
        print(sub[["selector", "model", "r2_mean", "rmse_mean"]]
              .round(4).to_string(index=False))

    meta = {"n_train": int(n), "n_features": int(X.shape[1]),
            "k_values": list(k_list), "mode": tag,
            "clf_selectors": list(clf_sel), "reg_selectors": list(reg_sel),
            "clf_models": list(tuned["clf"]), "reg_models": list(tuned["reg"]),
            "tuned_params_file": str(PROFILE_OUT / "champion_best_params.json"),
            "xgb_available": R.HAS_XGB, "test_set_used": False}
    (OUT / f"metadata_{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[done] {tag} -> {OUT} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
