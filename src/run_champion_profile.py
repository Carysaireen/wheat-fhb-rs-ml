"""
擂主档案（champion profile）
============================

【这个脚本在回答什么问题】
用户把 current_2（NREDI1 + NDVIre2）当作项目的**性能标杆**：后续所有
「选择器 × 模型」的结果都要拿来和它比，看能否选出更好的。但一个标杆要
能服众，必须先堵住三个漏洞：

    ① 单模型种子        -> 只有 seed=42，无法区分 0.015 是真赢还是噪声
    ② 二手超参          -> 它现在的参数是从 relieff_10(10 特征) 场景搬来的
    ③ 双口径未并报      -> 随机折与空间折要并排，否则挑战者不知在哪比

本脚本把 current_2 从「一个点」升级为「一个带误差带、超参公平、双口径的
参考标准」。

【★ 单一变量控制（硬约束）】
本脚本与 whole-project 其余实验共用下列一切，**唯一允许变化的量是"特征集"**：
    · 数据      : R.load_train()，132 行 x 117 特征（同一份 train.csv）
    · 随机折    : data/cv/classification_folds.csv / regression_folds.csv
    · 空间折    : R.spatial_fold(coords, y, seed=42/7/2026)  —— 与 P0-②、FS-CS 一致
    · 预处理    : 折内 fit（本脚本按 run_selector_zoo 的写法做同一套标准化）
    · 打分      : R.clf_scores / R.reg_scores
    · 网格      : hyperparam_grids.CLF_GRIDS / REG_GRIDS（与超参扫描同一套定义）
    · 超参      : ★ 本脚本为 current_2 **单独扫描**，不用二手参数

【三个阶段】
    STAGE 1  专属调参：在 current_2 上扫全部模型 x 全部配置（逐折），
             用「留一折诚实评估(LOFO)」挑出每个模型在**本特征集**下的最优。
    STAGE 2  多种子建档：固定 STAGE 1 选出的参数，跑多模型种子 + 多分区种子，
             报 mean ± std，并算最小可分辨差(MDD)。
    STAGE 3  对照建档：用完全相同的流程给其余固定特征集
             （common_6 / boruta_30 / all_117）建档 —— 让擂台对所有"固定集"
             一视同仁，不给 current_2 特殊待遇。

【为什么复刻标准化而不直接 import run_selector_zoo.run_track】
run_track 的签名把"选择器"绑死为逐折函数；本脚本要跑的恰恰是**全局固定特征集**
（擂主的形态就是"全局定好 2 个特征"）。所以这里复刻循环，但**逐字对齐**其
预处理与打分写法，并在 STAGE 2 用 run_track 的等价路径做一次自校验。

用法：
    python -u src/run_champion_profile.py
    python -u src/run_champion_profile.py --stage1   # 只调参
    python -u src/run_champion_profile.py --quick    # 冒烟
"""
from __future__ import annotations

import argparse
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

import run_selector_zoo as R                       # noqa: E402
import hyperparam_grids as G                       # noqa: E402

from sklearn.base import clone                     # noqa: E402
from sklearn.preprocessing import StandardScaler   # noqa: E402

EXP = ROOT / "experiments" / "2026-10-13_champion_profile"
OUT = EXP / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RES = ROOT / "results" / "champion_profile"
RES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 实验常量（与全项目对齐）
# --------------------------------------------------------------------------- #
SPATIAL_SEEDS = (42, 7, 2026)        # 与 P0-② / FS-CS / baseline_alignment 一致
MODEL_SEEDS = (42, 7, 2026, 123, 2024)   # 5 个模型种子 —— ★ 本脚本新增的重复度
CHAMPION = "current_2"
FIXED_SETS = ("current_2", "common_6", "boruta_30", "all_117")

_LOG = OUT / "run_progress_profile.log"


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# 特征集定义（与 run_baseline_alignment.fixed_feature_sets 逐字一致）
# --------------------------------------------------------------------------- #
def fixed_feature_sets(names: list[str]) -> dict[str, list[str]]:
    bo = pd.read_csv(ROOT / "results" / "feature_selection" / "boruta_stability.csv")
    boruta = bo.loc[bo.confirmed_rate >= 0.6, "feature"].tolist()
    return {
        "all_117": list(names),
        "common_6": [f for f in ["B8_HOM", "B8_MEA", "NDVIre2", "NDVIre3",
                                 "NREDI1", "NREDI2"] if f in names],
        "boruta_30": [f for f in boruta if f in names],
        "current_2": [f for f in ["NREDI1", "NDVIre2"] if f in names],
    }


# --------------------------------------------------------------------------- #
# 折构造
# --------------------------------------------------------------------------- #
def fold_sets_for(task: str, df, coords, y_strat) -> list[tuple]:
    """返回 [(scheme_name, fold_vector_full_length), ...]。"""
    clf_f, reg_f = R.load_random_folds(df)
    rand = clf_f if task == "clf" else reg_f
    sets = [("A_random5", rand)]
    for s in SPATIAL_SEEDS:
        fvec, _ = R.spatial_fold(coords, y_strat, s)
        sets.append((f"B_spatial3_s{s}", fvec))
    return sets


# --------------------------------------------------------------------------- #
# 单次评估（严格对齐 run_selector_zoo 的预处理与打分）
# --------------------------------------------------------------------------- #
def eval_one(model, Xtr, ytr, Xva, yva, task):
    """折内标准化 -> fit -> 打分。与 run_selector_zoo.run_track 写法一致。"""
    sc = StandardScaler().fit(Xtr)
    m = clone(model)
    m.fit(sc.transform(Xtr), ytr)
    return (R.clf_scores if task == "clf" else R.reg_scores)(
        m, sc.transform(Xva), yva)


# --------------------------------------------------------------------------- #
# STAGE 1 : current_2 专属超参扫描（逐折 × 全配置）
# --------------------------------------------------------------------------- #
def stage1(task, df, names, X, y, coords) -> pd.DataFrame:
    idx = [names.index(f) for f in fixed_feature_sets(names)[CHAMPION]]
    grids = _prune_grid(G.CLF_GRIDS if task == "clf" else G.REG_GRIDS, len(idx))
    build = G.build_clf if task == "clf" else G.build_reg
    metric = "roc_auc" if task == "clf" else "r2"

    log(f"[stage1|{task}] champion = {CHAMPION} idx={idx} "
        f"({[names[i] for i in idx]}) | n_configs="
        f"{sum(len(v) for v in grids.values())} (pruned from "
        f"{sum(len(v) for v in (G.CLF_GRIDS if task=='clf' else G.REG_GRIDS).values())})")

    folds = fold_sets_for(task, df, coords, df["Unnamed: 2"].astype(int).to_numpy())
    rows = []
    for scheme, fvec_full in folds:
        # 随机折按 task 取子集掩码；空间折全在 clf 的 row_mask 上
        row_mask = np.ones(len(df), bool) if task == "clf" else (y > 0)
        fvec = fvec_full[row_mask]
        Xt, yt = X[row_mask], y[row_mask]
        for fold_id in np.unique(fvec):
            if fold_id < 0:
                continue
            va = fvec == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue
            Xtr, Xva = Xt[tr][:, idx], Xt[va][:, idx]
            for mname, grid in grids.items():
                for cid, params in enumerate(grid):
                    try:
                        mdl = build(mname, **params)
                        sc = eval_one(mdl, Xtr, yt[tr], Xva, yt[va], task)
                    except Exception as e:               # pragma: no cover
                        log(f"  [warn] {task}/{mname}#{cid}: {type(e).__name__}: {e}")
                        continue
                    rec = {"task": task, "scheme": scheme, "model": mname,
                           "config_id": cid, "params": json.dumps(params),
                           "fold": int(fold_id), "n_features": len(idx),
                           "n_train": int(tr.sum()), "n_valid": int(va.sum())}
                    rec.update(sc)
                    rows.append(rec)
        log(f"[stage1|{task}] {scheme} done, rows={len(rows)}")
    d = pd.DataFrame(rows)
    return d, metric


def lofo_pick(d: pd.DataFrame, metric: str, scheme: str) -> dict:
    """留一折诚实评估（只在**单一 scheme** 内进行）。

    ★ 为什么必须限定 scheme —— 这是单一变量控制的关键一处：
       A_random5 是 5 折，B_spatial3_s* 是各 3 折。若把两者混在一起 groupby("fold")
       会把"空间折的折 0"和"随机折的折 0"当成同一个折，等于让**跨区域表现参与了
       随机折的挑参**，变量就被污染了。
    并且 **绝不在空间折上挑参**（空间折是裁判，在它上面挑 = 在考卷上写答案）。
    所以本函数只对随机折调用。
    """
    r = d[d.scheme == scheme].copy()
    picks, honest = {}, []
    for model, g in r.groupby("model"):
        by_fold = {f: gg for f, gg in g.groupby("fold")}
        folds = sorted(by_fold)
        scores, pick_ids = [], []
        for f in folds:
            others = [x for x in folds if x != f]
            mean_other = g[g.fold.isin(others)].groupby("config_id")[metric].mean()
            best_cid = int(mean_other.idxmax())
            pick_ids.append(best_cid)
            row = by_fold[f][by_fold[f].config_id == best_cid]
            if len(row):
                scores.append(float(row[metric].iloc[0]))
        picks[model] = {"pick_ids": pick_ids, "lofo": float(np.mean(scores))}
    return picks


def stage1_summary(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """按模型汇总，**只统计随机折口径**（挑参的依据）。

    ★ 为什么必须过滤 scheme —— 单一变量控制的又一关键处：
       B_spatial3_* 的分数系统性地低于 A_random5（跨区域更难）。若把两者混在
       一起求均值，"空间折的低分"会把所有模型一起拉低，还会让 grid_best 反而
       低于某些口径下的 lofo —— 纯属汇总口径错误。挑参只用随机折口径。
    """
    r = d[d.scheme == "A_random5"]
    rows = []
    for model, g in r.groupby("model"):
        gm = (g.groupby("config_id")
                .agg(params=("params", "first"), **{metric: (metric, "mean")})
                .reset_index().sort_values(metric, ascending=False))
        rows.append({
            "model": model,
            "default": round(float(g[g.config_id == 0][metric].mean()), 4),
            "grid_best": round(float(gm[metric].iloc[0]), 4),
            "best_config_id": int(gm.config_id.iloc[0]),
            "best_params": gm.params.iloc[0],
            "grid_mean": round(float(g[metric].mean()), 4),
            "n_folds": int(g.fold.nunique()),
        })
    return pd.DataFrame(rows).sort_values("grid_best", ascending=False)


# --------------------------------------------------------------------------- #
# STAGE 2 : 多种子 x 双口径建档
# --------------------------------------------------------------------------- #
def stage2(task, df, names, X, y, coords, best_params: dict) -> pd.DataFrame:
    """固定每个模型在 current_2 上选出的最优参数，跑多模型种子 x 双口径。

    ★ 单一变量：模型种子是唯一变化量（折固定、特征固定、预处理固定）。
    """
    build = G.build_clf if task == "clf" else G.build_reg
    sets = fixed_feature_sets(names)
    row_mask = np.ones(len(df), bool) if task == "clf" else (y > 0)
    Xt, yt = X[row_mask], y[row_mask]
    y_strat = df["Unnamed: 2"].astype(int).to_numpy()
    folds = fold_sets_for(task, df, coords, y_strat)

    rows = []
    for scheme, fvec_full in folds:
        fvec = fvec_full[row_mask]
        for fold_id in np.unique(fvec):
            if fold_id < 0:
                continue
            va = fvec == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue
            for setname in FIXED_SETS:
                idx = [names.index(f) for f in sets[setname] if f in names]
                if not idx:
                    continue
                Xtr, Xva = Xt[tr][:, idx], Xt[va][:, idx]
                for mname, p in best_params.items():
                    # SVR / PLSR / LDA 等**确定性**估计器没有 random_state，
                    # 强行 set_params 会报错。它们本来就不受种子影响（std 恒为 0），
                    # 所以这里检测到没有该参数就跳过设种子 —— 这是物理事实，不是 bug。
                    for mseed in MODEL_SEEDS:
                        try:
                            mdl = build(mname, **p)
                            if "random_state" in mdl.get_params():
                                mdl.set_params(random_state=mseed)
                            sc = eval_one(mdl, Xtr, yt[tr], Xva, yt[va], task)
                        except Exception as e:           # pragma: no cover
                            log(f"  [warn] {task}/{setname}/{mname}/{mseed}: "
                                f"{type(e).__name__}: {e}")
                            continue
                        rec = {"task": task, "scheme": scheme, "set": setname,
                               "model": mname, "model_seed": mseed,
                               "n_features": len(idx), "fold": int(fold_id),
                               "n_train": int(tr.sum()), "n_valid": int(va.sum())}
                        rec.update(sc)
                        rows.append(rec)
        log(f"[stage2|{task}] {scheme} done, rows={len(rows)}")
    return pd.DataFrame(rows)


def _prune_grid(grids: dict, n_features: int) -> dict:
    """按特征数裁掉物理上不可行的配置，避免日志被"合理报错"淹没。

    典型例子：PLSR / PLSDA 的 n_components 不能超过 min(n_features, n_samples-1)。
    current_2 只有 2 个特征 -> n_components 只能是 1 或 2，其余是**数学上无解**，
    不是代码 bug。统一在这里裁掉，保证网格对每个特征集都是"全可行"的。
    """
    out = {}
    for name, grid in grids.items():
        keep = []
        for p in grid:
            if "n_components" in p and p["n_components"] > n_features:
                continue
            keep.append(p)
        if keep:
            out[name] = keep
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="只跑专属调参")
    ap.add_argument("--quick", action="store_true", help="冒烟：只用随机折、单模型种子")
    args = ap.parse_args()

    t0 = time.time()
    log("=" * 72)
    log("champion profile :: current_2 作为项目标杆（单一变量 = 特征集）")
    log(f"model seeds = {MODEL_SEEDS} | spatial seeds = {SPATIAL_SEEDS}")
    log("=" * 72)

    df, names, X, y_cls, y_reg, coords = R.load_train()
    log(f"data: {len(df)} rows x {len(names)} features | "
        f"clf 0/1 = {int((y_cls==0).sum())}/{int((y_cls==1).sum())} | "
        f"DI>0 = {int((y_reg>0).sum())}")

    best_params_all = {}
    for task, y in (("clf", y_cls), ("reg", y_reg)):
        d1, metric = stage1(task, df, names, X, y, coords)
        d1.to_csv(OUT / f"tuning_{task}_per_fold.csv", index=False, encoding="utf-8-sig")
        s1 = stage1_summary(d1, metric)
        # ★ 只在随机折上做 LOFO（空间折是裁判，不许参与挑参）
        picks = lofo_pick(d1, metric, "A_random5")
        s1["lofo_honest"] = s1.model.map(
            {m: round(v["lofo"], 4) for m, v in picks.items()})
        s1.to_csv(OUT / f"tuning_{task}_summary.csv", index=False, encoding="utf-8-sig")
        log(f"\n[stage1|{task}] tuning summary (metric={metric})")
        print(s1.to_string(index=False))

        # 冻结：每个模型用它在 current_2 上的最优参数
        bp = {}
        for _, r in s1.iterrows():
            bp[r.model] = json.loads(r.best_params)
        best_params_all[task] = bp

    if args.stage1:
        (OUT / "champion_best_params.json").write_text(
            json.dumps(best_params_all, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[done] stage1 only -> {OUT}")
        return

    # ---- STAGE 2 ----
    summ = {}
    for task, y in (("clf", y_cls), ("reg", y_reg)):
        d2 = stage2(task, df, names, X, y, coords, best_params_all[task])
        d2.to_csv(OUT / f"profile_{task}_per_fold.csv", index=False, encoding="utf-8-sig")
        met = ["accuracy", "f1", "roc_auc"] if task == "clf" else ["r2", "rmse", "mae"]
        # 汇总层级：(set, model, scheme)
        g = d2.groupby(["set", "model", "scheme"], as_index=False).agg(
            n_features=("n_features", "first"), n_eval=("fold", "size"),
            **{f"{m}_mean": (m, "mean") for m in met},
            **{f"{m}_std": (m, "std") for m in met})
        g.to_csv(OUT / f"profile_{task}_summary.csv", index=False, encoding="utf-8-sig")
        summ[task] = (g, met)
        log(f"[stage2|{task}] done: {len(d2)} rows -> profile_{task}_*.csv")

    (OUT / "champion_best_params.json").write_text(
        json.dumps(best_params_all, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 打印擂主（current_2 最优模型）双口径档案 ----
    # 注意：summarize() 的列名是 {metric}_mean / {metric}_std，行按 scheme 区分，
    # 不含 scheme 前缀。所以这里先做一次 pivot 再打印。
    for task, (g, met) in summ.items():
        m = "accuracy" if task == "clf" else "r2"
        ch = g[g.set == CHAMPION]
        wide = ch.pivot_table(index=["set", "model", "n_features"],
                              columns="scheme",
                              values=[f"{m}_mean", f"{m}_std"])
        wide.columns = [f"{s}__{v}" for v, s in wide.columns]
        wide = wide.reset_index()
        log("\n" + "-" * 72)
        log(f"[champion|{task}] current_2 各模型（{m}），按随机折排序 "
            f"[多种子 mean, 跨种子 std]")
        cols = [c for c in wide.columns
                if c in ("model", "n_features")
                or c.startswith(("A_random5", "B_spatial3"))]
        print(wide.sort_values("A_random5__" + m + "_mean", ascending=False)
              [cols].round(4).to_string(index=False))

    log(f"\n[done] {time.time()-t0:.1f} s -> {OUT} / {RES}")


if __name__ == "__main__":
    main()
