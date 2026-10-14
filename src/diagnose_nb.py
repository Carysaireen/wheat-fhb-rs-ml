"""
阶段⑥ · NB（朴素贝叶斯）线索专项诊断
=========================================
用户线索：「NB 在 K=10/20/30 三个档位全部拿下榜首（0.8200 / 0.8149 / 0.8045），
同时它在 current_2 里的 accuracy（0.8141）也是全场最高。极简模型 + 小特征集在
跨区域口径下反复跑到最前面。」

本实验要把「NB 反复夺冠」从**现象**推进到**机制**。四个可证伪的问题：

  Q1 【K 曲线】NB 的「K × 空间折」曲线与其他模型有何不同？它是"K 越大越好"还是"平"？
  Q2 【种子稳健性】NB 的空间折跨分区种子方差，是否系统性低于高容量模型（RF/ET/XGB）？
  Q3 【偏置-方差】用 bootstrap 分解：NB 的"稳"来自低方差还是低偏置？
  Q4 【保守性检验】NB 的高 accuracy 是否只是"预测偏向多数类"的保守策略？
      —— 若是，则它的 recall/F1/混淆矩阵应显示明显的不对称。

设计（★ 单一变量控制）
    数据/折/选择器/打分 全部复用 run_selector_zoo，不重新实现。
    - 在 K ∈ {2, 5, 10, 20, 30} 上跑 NB vs 对照模型（LR / SVM / RF / ET / XGB / KNN）
    - K=2 时直接指定 current_2；K>2 时用固定的选择器（relieff / lasso / random 三个代表）
    - 空间折用 3 个分区种子（42 / 7 / 2026），随机折用 1 个种子（42）
    - 额外记录混淆矩阵四格 + 预测正类比例，用于 Q4

输出
    results/nb_diagnosis/README.md
    results/nb_diagnosis/k_curve_with_k2.csv         Q1
    results/nb_diagnosis/seed_stability.csv          Q2
    results/nb_diagnosis/bias_variance.csv           Q3
    results/nb_diagnosis/conservatism.csv            Q4
    results/nb_diagnosis/fig_*.png
    results/nb_diagnosis/_stdout.txt

用法
    python src/diagnose_nb.py              # 全量
    python src/diagnose_nb.py --smoke      # 冒烟
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score

import run_selector_zoo as R
import selector_zoo as zoo

OUT = ROOT / "results" / "nb_diagnosis"
EXP = ROOT / "experiments" / "2026-10-13_nb_diagnosis"
CURRENT2 = ["NREDI1", "NDVIre2"]

# --- 对照模型：跨越"容量"光谱 ------------------------------------------------
FOCUS_MODELS = ["NB", "LR", "SVMLin", "SVM", "KNN", "RF", "ET", "XGB"]
# --- 容量标签（用于 Q3 的分组，先验设定，事后用结果验证）--------------------
CAPACITY = {
    "NB": "极简(生成式)",
    "LR": "低(线性)",
    "SVMLin": "低(线性)",
    "SVM": "中(核)",
    "KNN": "中(实例)",
    "RF": "高(集成)",
    "ET": "高(集成)",
    "XGB": "高(集成)",
}

K_GRID = [2, 5, 10, 20, 30]
K_SELECTORS = ["relieff", "lasso", "random"]   # K>2 时用这三个代表选择器
SPATIAL_SEEDS = (42, 7, 2026)
RANDOM_SEEDS = (42,)


def log_factory(path: Path):
    fh = path.open("w", encoding="utf-8")

    def emit(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        fh.write(s + "\n")
        fh.flush()
    return emit, fh


# ---------------------------------------------------------------------------
def build_fold_sets(coords, y_cls, seeds, schemes):
    """构造 (scheme_name, fold_vector) 列表。schemes 取 'spatial' / 'random'。"""
    out = []
    if "spatial" in schemes:
        for s in seeds:
            fv, _ = R.spatial_fold(coords, y_cls, s)
            out.append((f"B_spatial3_s{s}", fv))
    if "random" in schemes:
        df, *_ = R.load_train()
        clf_fold, _ = R.load_random_folds(df)
        out.append(("A_random5_s42", clf_fold))
    return out


def eval_one(emit, X, y, feats, k, selector_name, models, fold_sets):
    """
    在给定的 (k, selector) 上，对每个模型 × 每个折评估，返回逐折明细。
    K=2 时强制使用 current_2（selector 记为 'preset_current_2'）。
    """
    rows = []
    if k == 2:
        sel_idx = [feats.index(f) for f in CURRENT2]
        selector_used = "preset_current_2"
    else:
        selector_used = selector_name

    need_mi = selector_name in zoo.NEEDS_MI
    for scheme, fvec in fold_sets:
        for fold_id in np.unique(fvec):
            if fold_id < 0:
                continue
            va = fvec == fold_id
            tr = ~va
            if tr.sum() < 20 or va.sum() < 5:
                continue

            if k == 2:
                sel = sel_idx
            else:
                ctx = {"task": "clf", "seed": 42, "_cache": {}}
                if need_mi:
                    ctx["mi_matrix"] = zoo.compute_mi_matrix(X[tr])
                try:
                    sel = zoo.CLF_SELECTORS[selector_name](X[tr], y[tr], k, ctx)
                except Exception as e:
                    emit(f"    [warn] select {selector_name}/k={k}: {type(e).__name__}: {e}")
                    continue
                if len(sel) == 0:
                    continue

            Xtr, Xva = X[tr][:, sel], X[va][:, sel]
            ytr, yva = y[tr], y[va]
            for mname in models:
                mdl = clone(models_all(mname, 42))
                try:
                    mdl.fit(Xtr, ytr)
                    pred = mdl.predict(Xva)
                    try:
                        score = mdl.predict_proba(Xva)[:, 1]
                    except Exception:
                        score = mdl.decision_function(Xva)
                except Exception as e:
                    emit(f"    [warn] fit {mname}: {type(e).__name__}: {e}")
                    continue

                cm = confusion_matrix(yva, pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                from sklearn.metrics import roc_auc_score
                rows.append(dict(
                    scheme=scheme, fold=int(fold_id), k=k,
                    selector=selector_used, model=mname,
                    n_valid=int(va.sum()),
                    accuracy=accuracy_score(yva, pred),
                    recall_pos=recall_score(yva, pred, pos_label=1, zero_division=0),
                    recall_neg=recall_score(yva, pred, pos_label=0, zero_division=0),
                    f1=f1_score(yva, pred, zero_division=0),
                    roc_auc=roc_auc_score(yva, score) if len(np.unique(yva)) > 1 else np.nan,
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                    n_pos_pred=int(tp + fp), n_pos_true=int(tp + fn),
                    pred_pos_rate=float((pred == 1).mean()),
                    true_pos_rate=float(yva.mean()),
                ))
    return pd.DataFrame(rows)


# 模型库（缓存，避免重复构造）
_MM = {}


def models_all(name, seed):
    if seed not in _MM:
        _MM[seed] = R.clf_models(seed)
    return _MM[seed][name]


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    emit, fh = log_factory(OUT / "_stdout.txt")

    t0 = time.time()
    emit("=" * 92)
    emit("阶段⑥ · NB 线索专项诊断")
    emit("=" * 92)

    df, feats, X, y_cls, y_reg, coords = R.load_train()
    emit(f"[data] n_rows={len(df)}  n_features={X.shape[1]}  "
         f"label_counts={np.bincount(y_cls).tolist()}  "
         f"majority_baseline={max(np.bincount(y_cls))/len(y_cls):.4f}")
    emit(f"[data] current_2 = {CURRENT2} (idx={[feats.index(f) for f in CURRENT2]})")

    k_grid = [10] if args.smoke else K_GRID
    k_sels = ["relieff"] if args.smoke else K_SELECTORS
    models = ["NB", "LR", "RF"] if args.smoke else FOCUS_MODELS
    sp_seeds = (42,) if args.smoke else SPATIAL_SEEDS
    rd_seeds = (42,)

    emit(f"[plan] K={k_grid}  K>2 的选择器={k_sels}  模型={models}")
    emit(f"[plan] 空间折分区种子={sp_seeds}  随机折种子={rd_seeds}")
    emit()

    # -------- 主循环：K × selector × model × (scheme, seed) --------
    all_rows = []
    for k in k_grid:
        sels = [None] if k == 2 else k_sels   # k=2 用预设，不用选择器
        for sel in sels:
            emit(f"  running k={k} selector={sel or 'preset_current_2'} ...")
            fs = build_fold_sets(coords, y_cls, sp_seeds, {"spatial", "random"})
            part = eval_one(emit, X, y_cls, feats, k, sel or "relieff", models, fs)
            all_rows.append(part)
            if args.smoke:
                break
    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(OUT / "per_fold_raw.csv", index=False, encoding="utf-8-sig")
    emit(f"[done] per-fold rows = {len(raw)}")

    # =======================================================================
    # Q1 · K 曲线
    # =======================================================================
    emit()
    emit("-" * 92)
    emit("Q1 · 「K × 空间折 accuracy」曲线 —— NB 的曲线形状和其他模型有何不同？")
    emit("-" * 92)
    sp = raw[raw.scheme.str.startswith("B_spatial")]
    g = (sp.groupby(["model", "k"])
         .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
              auc_mean=("roc_auc", "mean"), n=("accuracy", "size"))
         .reset_index())
    g.to_csv(OUT / "k_curve_with_k2.csv", index=False, encoding="utf-8-sig")

    piv = g.pivot(index="model", columns="k", values="acc_mean").round(4)
    # 派生的汇总列必须对"实际跑过的 K"鲁棒（smoke 只有 K=10）
    k_present = sorted(int(c) for c in piv.columns if isinstance(c, (int, np.integer)))
    kmin, kmax = k_present[0], k_present[-1]
    piv[f"斜率(K{kmin}→K{kmax})"] = (piv[kmax] - piv[kmin]).round(4)
    piv["最优K"] = g.loc[g.groupby("model").acc_mean.idxmax()].set_index("model")["k"]
    piv["K间极差"] = (piv[k_present].max(axis=1) - piv[k_present].min(axis=1)).round(4)
    piv = piv.sort_values(kmax, ascending=False)
    emit(piv.to_string())
    emit()
    emit("解读要点：斜率 > 0 = 该模型吃更多特征；斜率 ≈ 0 = 对 K 不敏感（稳）。")

    # 图
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mname in ["NB", "LR", "SVMLin", "SVM", "KNN", "RF", "ET", "XGB"]:
        sub = g[g.model == mname].sort_values("k")
        if sub.empty:
            continue
        lw = 3.0 if mname == "NB" else 1.3
        ax.plot(sub.k, sub.acc_mean, marker="o", linewidth=lw,
                label=mname, alpha=1.0 if mname == "NB" else 0.65,
                zorder=5 if mname == "NB" else 1)
    ax.set_xlabel("K (# selected features)")
    ax.set_ylabel("spatial-fold accuracy (mean over selectors & spatial seeds)")
    ax.set_title("Q1 · accuracy vs K: NB vs capacity spectrum")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_q1_k_curve.png", dpi=140)
    plt.close(fig)

    # =======================================================================
    # Q2 · 种子稳健性
    # =======================================================================
    emit()
    emit("-" * 92)
    emit("Q2 · 空间折「跨分区种子」波动 —— NB 是否比高容量模型更稳？")
    emit("-" * 92)
    # 每个 (model, k, selector, 分区种子) 先求该种子下的均值，再跨种子求 std
    sp2 = sp.copy()
    sp2["seed"] = sp2.scheme.str.extract(r"_s(\d+)$").astype(int)
    per_seed = (sp2.groupby(["model", "k", "selector", "seed"])
                .agg(acc=("accuracy", "mean"), auc=("roc_auc", "mean"),
                     f1=("f1", "mean"))
                .reset_index())
    stab = (per_seed.groupby(["model", "k", "selector"])
            .agg(acc_cross_seed_std=("acc", "std"), acc_cross_seed_mean=("acc", "mean"),
                 auc_cross_seed_std=("auc", "std"))
            .reset_index())
    stab.to_csv(OUT / "seed_stability.csv", index=False, encoding="utf-8-sig")
    st = (stab.groupby("model")
          .agg(mean_acc=("acc_cross_seed_mean", "mean"),
               mean_cross_seed_std=("acc_cross_seed_std", "mean"),
               max_cross_seed_std=("acc_cross_seed_std", "max"),
               mean_auc_std=("auc_cross_seed_std", "mean"))
          .round(4).sort_values("mean_cross_seed_std"))
    emit(st.to_string())
    emit()
    emit("解读要点：mean_cross_seed_std 越小 = 换一个空间分区方式，结论越不容易变。")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = st.index.tolist()
    colors = ["#c0392b" if m == "NB" else "#7f8c8d" for m in order]
    ax.barh(range(len(order)), st.mean_cross_seed_std, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("mean cross-seed std of spatial accuracy (lower = more stable)")
    ax.set_title("Q2 · spatial-fold stability across blocking seeds")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig_q2_seed_stability.png", dpi=140)
    plt.close(fig)

    # =======================================================================
    # Q4 · 保守性检验（放在 Q3 之前，因为它可能推翻 Q3 的解释）
    # =======================================================================
    emit()
    emit("-" * 92)
    emit("Q4 · 保守性检验 —— NB 的高 accuracy 是否只是'预测偏向多数类'？")
    emit("-" * 92)
    cm = (sp.groupby(["model", "k"])
          .agg(pred_pos_rate=("pred_pos_rate", "mean"),
               true_pos_rate=("true_pos_rate", "mean"),
               recall_pos=("recall_pos", "mean"),
               recall_neg=("recall_neg", "mean"),
               acc=("accuracy", "mean"), f1=("f1", "mean"))
          .reset_index())
    cm.to_csv(OUT / "conservatism.csv", index=False, encoding="utf-8-sig")
    cs = (cm.groupby("model")
          .agg(acc=("acc", "mean"), f1=("f1", "mean"),
               pred_pos_rate=("pred_pos_rate", "mean"),
               true_pos_rate=("true_pos_rate", "mean"),
               recall_pos=("recall_pos", "mean"),
               recall_neg=("recall_neg", "mean"))
          .round(4))
    cs["pred-true 偏差"] = (cs.pred_pos_rate - cs.true_pos_rate).round(4)
    cs["recall 不对称"] = (cs.recall_pos - cs.recall_neg).round(4)
    emit(cs.sort_values("acc", ascending=False).to_string())
    emit()
    emit(f"注：训练集真实正类比例 = {np.bincount(y_cls)[1]/len(y_cls):.4f}")
    emit("解读要点：若 NB 的 pred-true 偏差接近 0，说明它不是靠'全猜多数类'刷分。")

    # =======================================================================
    # Q3 · 偏置-方差（用 K=2 与 K=30 两个端点的"训练-验证 gap"近似）
    # =======================================================================
    emit()
    emit("-" * 92)
    emit("Q3 · 容量效应 —— 同 K 下不同容量模型的「空间折 − 随机折」差距（gap = 空间泄漏暴露度）")
    emit("-" * 92)
    rd = raw[~raw.scheme.str.startswith("B_spatial")]
    gap_rows = []
    for (k, sel, mname), sub_sp in sp.groupby(["k", "selector", "model"]):
        sub_rd = rd[(rd.k == k) & (rd.selector == sel) & (rd.model == mname)]
        if len(sub_rd) == 0:
            continue
        gap_rows.append(dict(
            k=k, selector=sel, model=mname,
            spatial_acc=float(sub_sp.accuracy.mean()),
            random_acc=float(sub_rd.accuracy.mean()),
            gap=float(sub_rd.accuracy.mean() - sub_sp.accuracy.mean()),
            spatial_ci=float(sub_sp.accuracy.std()),
        ))
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(OUT / "bias_variance.csv", index=False, encoding="utf-8-sig")
    gp = (gap.groupby(["model", "k"])
          .agg(gap=("gap", "mean"), spatial=("spatial_acc", "mean"),
               random=("random_acc", "mean"))
          .round(4).reset_index())
    emit(gp.pivot(index="model", columns="k", values="gap").to_string())
    emit()
    emit("解读要点：gap 越大 = 随机折被'空间记忆'抬得越高。K 小 + 容量低 → gap 应最小。")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mname in FOCUS_MODELS:
        sub = gp[gp.model == mname].sort_values("k")
        if sub.empty:
            continue
        lw = 3.0 if mname == "NB" else 1.3
        ax.plot(sub.k, sub.gap, marker="s", linewidth=lw, label=mname,
                alpha=1.0 if mname == "NB" else 0.6)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("K (# selected features)")
    ax.set_ylabel("gap = random-fold acc − spatial-fold acc")
    ax.set_title("Q3 · spatial-leakage exposure vs K and model capacity")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_q3_capacity_gap.png", dpi=140)
    plt.close(fig)

    # 单独画 NB 的预测率对照
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ks = sorted(sp.k.unique())
    nb = cm[cm.model == "NB"].sort_values("k")
    ax.plot(nb.k, nb.pred_pos_rate, marker="o", label="NB predicted positive rate", linewidth=2)
    ax.axhline(np.bincount(y_cls)[1] / len(y_cls), color="k", linestyle="--",
               label="true positive rate (data)")
    ax.set_xlabel("K")
    ax.set_ylabel("positive-class share")
    ax.set_title("Q4 · does NB just predict the majority class?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_q4_conservatism.png", dpi=140)
    plt.close(fig)

    emit()
    emit(f"[done] 总耗时 {time.time()-t0:.0f}s")
    emit(f"[done] 输出目录 {OUT}")
    fh.close()


if __name__ == "__main__":
    main()
