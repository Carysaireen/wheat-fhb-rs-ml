"""
挑战赛分析（challenge analysis）
================================

【做什么】
读入两份产物，产出用户最终目标所需的三样东西：

    ① 擂主档案表    —— current_2 在**双口径 + 5 模型种子**下的 mean ± std，
                       以及在 current_2 这个特征集上各模型谁最好。
    ② 擂台总表      —— 已有全部结果（含 baseline_alignment 的 73/85 个组合）
                       对擂主逐条判定 赢/平/输。
    ③ 特征数-性能曲线 —— 把"特征集规模"这条全项目主线在擂台语境下画出来。

【裁判口径】
    B_spatial3（空间折）为主裁判 —— 因为它排除了空间泄漏。
    A_random5（随机折）只作辅助展示。
    胜利门槛 = 最小可分辨差 MDD（见 challenge_rules.py）。

【单一变量控制】
    挑战者与擂主共用：同一份折向量、同一套预处理、同一组打分函数。
    本脚本**只读**已有结果，不重跑任何模型。
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

import challenge_rules as CR                       # noqa: E402

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

PROF = ROOT / "experiments" / "2026-10-13_champion_profile" / "outputs"
ALIGN = ROOT / "results" / "baseline_alignment"
RES = ROOT / "results" / "champion_profile"
RES.mkdir(parents=True, exist_ok=True)

RED, GREEN, BLUE, GREY = "#c0392b", "#2e8b57", "#2c6fbb", "#7f8c8d"
CHAMPION = "current_2"      # 用户指定的标杆特征集：NREDI1 + NDVIre2


# --------------------------------------------------------------------------- #
# ① 擂主档案表：把 (set, model) 在两种口径下的 mean±std 并排
# --------------------------------------------------------------------------- #
def champion_table() -> pd.DataFrame:
    rows = []
    for task, metric in (("clf", "accuracy"), ("reg", "r2")):
        f = PROF / f"profile_{task}_per_fold.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, encoding="utf-8-sig")
        # 跨种子 std：先在折上平均，再在种子上求 std
        g = (d.groupby(["set", "model", "scheme"])
               .apply(lambda x: pd.Series({
                   "mean": x[metric].mean(),
                   "std_seed": x.groupby("model_seed")[metric].mean().std(ddof=1)
                   if x.model_seed.nunique() > 1 else np.nan,
                   "std_fold": x.groupby("fold")[metric].mean().std(ddof=1),
                   "n_valid_min": x.n_valid.min(),
                   "n_folds": x.fold.nunique(),
                   "n_features": x.n_features.iloc[0]}))
               .reset_index())
        g["task"], g["metric"] = task, metric
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pivot_protocol(g: pd.DataFrame, task: str) -> pd.DataFrame:
    """把 scheme 行转成列：A_random5_* / B_spatial3_s* 并排。"""
    t = g[g.task == task].copy()
    agg = (t.groupby(["set", "model", "scheme"])
             .agg(mean=("mean", "first"), std_seed=("std_seed", "first"),
                  n_features=("n_features", "first"))
             .reset_index())
    out = agg.pivot_table(index=["set", "model", "n_features"],
                          columns="scheme", values=["mean", "std_seed"])
    out.columns = [f"{s}__{m}" for m, s in out.columns]
    return out.reset_index()


# --------------------------------------------------------------------------- #
# ③ 特征数-性能曲线（擂台语境）
# --------------------------------------------------------------------------- #
def plot_nfeat_curve(g: pd.DataFrame, task: str, metric: str,
                     spatial_col: str, random_col: str, out: Path) -> None:
    t = g[(g.task == task) & (g.scheme == spatial_col)]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    for mname, gg in t.groupby("model"):
        gg = gg.sort_values("n_features")
        ax.plot(gg.n_features, gg["mean"], marker="o", ms=4, lw=1.3,
                label=mname, alpha=0.85)
    # 擂主高亮
    ch = t[t.set == "current_2"]
    if len(ch):
        ax.axhline(ch["mean"].max(), color=RED, ls="--", lw=1.6,
                   label=f"current_2 best = {ch['mean'].max():.3f}")
    ax.axvline(2, color=RED, ls=":", lw=1.0, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(sorted(t.n_features.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of features (log scale)")
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: {metric} under SPATIAL folds  vs  #features\n"
                 f"(champion = current_2, 2 features)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_verdict_bars(vt: pd.DataFrame, metric: str, out: Path,
                      delta_col: str = "delta_vs_champ") -> None:
    if vt.empty:
        return
    t = vt.head(25).iloc[::-1]
    colors = [RED if v == CR.WIN else (GREEN if v == CR.LOSE else GREY)
              for v in t.verdict]
    fig, ax = plt.subplots(figsize=(8.4, max(4.0, 0.32 * len(t))), dpi=140)
    lab = t.apply(lambda r: f"{r.get('set','?')}+{r.get('model','?')}", axis=1)
    ax.barh(lab, t[delta_col], color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel(f"Δ {metric} vs champion  (red=WIN, grey=TIE, green=LOSE)")
    ax.set_title("challengers vs champion @ SPATIAL folds", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("challenge analysis :: champion = current_2 (NREDI1 + NDVIre2)")
    print("=" * 74)

    g = champion_table()
    if g.empty:
        print("[!] no champion profile found -- run run_champion_profile.py first")
        return
    g.to_csv(RES / "champion_profile_full.csv", index=False, encoding="utf-8-sig")

    j = CR.Judge.from_profile(PROF)

    # ---- 擂主档案（双口径并排）----
    for task, metric in (("clf", "accuracy"), ("reg", "r2")):
        p = pivot_protocol(g, task)
        p.to_csv(RES / f"champion_{task}_wide.csv", index=False, encoding="utf-8-sig")
        print(f"\n{'='*74}\n## {task} ({metric}) —— 擂主档案：current_2 与其余固定集\n{'='*74}")
        cols = [c for c in p.columns if c.startswith(("set", "model", "n_features",
                                                      "A_random5", "B_spatial3"))]
        print(p.sort_values("A_random5__mean", ascending=False)[cols].round(4).to_string(index=False))

        # 口径内最佳擂主
        for scheme in [c for c in ["A_random5", "B_spatial3_s42"] ]:
            try:
                c = j.best_champion(scheme)
                print(f"  [best champion @ {scheme}] {c.set_name}+{c.model} "
                      f"= {c.mean:.4f} ± {c.std_across_seeds:.4f}(seed) | "
                      f"MDD = {c.mdd:.4f} | n_valid_min={c.n_valid_min}")
            except KeyError:
                pass

    # ---- 擂台总表：用 baseline_alignment 的现成结果挑战 ----
    for task, metric in (("clf", "accuracy"), ("reg", "r2")):
        f = ALIGN / f"headline_{task}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f, encoding="utf-8-sig")
        b_col = f"B_spatial3_{metric}"
        if b_col not in h.columns:
            continue

        # ★ 裁判基准 = current_2 这个特征集上、空间折口径下最好的模型。
        #   不能用"全体最强擂主"（那是 common_6+NB，会偏离用户指定标杆），
        #   也不能用 current_2 的所有模型平均（会把差模型一起算进门槛）。
        ch_rows = g[(g.task == task) & (g.set == CHAMPION)
                    & (g.scheme.str.startswith("B_spatial3"))]
        base_model = ch_rows.groupby("model")["mean"].mean().idxmax()
        base_rows = ch_rows[ch_rows.model == base_model]
        champ_mean = float(base_rows["mean"].mean())
        mdd = CR.min_detectable_difference(
            float(base_rows.std_seed.mean()),
            int(base_rows.n_valid_min.min()),
            int(base_rows.n_folds.max()))
        print(f"\n{'='*74}\n## {task} 挑战结果（裁判=空间折）")
        print(f"{'='*74}")
        print(f"  裁判基准 = {CHAMPION} + {base_model}（current_2 上的最优模型）"
              f"  spatial = {champ_mean:.4f}  MDD = {mdd:.4f}")
        print(f"  （对照：任务口径最强的固定集擂主是 "
              f"{j.best_champion('B_spatial3_s42').set_name}+"
              f"{j.best_champion('B_spatial3_s42').model}，但用户指定标杆为 current_2）")

        rows = []
        for _, r in h.iterrows():
            delta = r[b_col] - champ_mean
            verdict = CR.WIN if delta > mdd else (CR.LOSE if delta < -mdd else CR.TIE)
            rows.append({"set": r["set"], "model": r["model"],
                         "n_features": r["n_features"],
                         "spatial": round(r[b_col], 4),
                         "random": round(r[f"A_random5_{metric}"], 4),
                         "delta_vs_champ": round(delta, 4),
                         "mdd": round(mdd, 4), "verdict": verdict})
        vt = pd.DataFrame(rows).sort_values("delta_vs_champ", ascending=False)
        vt.to_csv(RES / f"challenge_{task}.csv", index=False, encoding="utf-8-sig")
        print(f"  赢 = {int((vt.verdict==CR.WIN).sum())} | "
              f"平 = {int((vt.verdict==CR.TIE).sum())} | "
              f"输 = {int((vt.verdict==CR.LOSE).sum())}  （共 {len(vt)} 个挑战者）")
        print(vt.head(12).to_string(index=False))
        print("  " + CR.multiple_comparison_note(len(vt)))
        plot_verdict_bars(vt, metric, RES / f"fig_verdict_{task}.png")

    # ---- 特征数曲线 ----
    for task, metric in (("clf", "accuracy"), ("reg", "r2")):
        plot_nfeat_curve(g, task, metric, "B_spatial3_s42", "A_random5",
                         RES / f"fig_nfeat_spatial_{task}.png")

    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
