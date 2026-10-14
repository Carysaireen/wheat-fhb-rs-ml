"""
阶段⑤ 分析 · 逐条挑战 current_2 擂主
======================================

【输入】
  · experiments/2026-10-13_stage5_tuned/outputs/{clf,reg}_summary_full.csv
    —— 阶段⑤：**调优参数**下的「选择器 × 模型 × K × 口径」全矩阵
  · experiments/2026-10-12_selector_zoo/outputs/{clf,reg}_summary_stage2.csv
    —— 阶段②：**默认参数**下的同矩阵（对照组，用来量化"调参效应"）
  · experiments/2026-10-13_champion_profile/outputs/profile_{clf,reg}_per_fold.csv
    —— 擂主档案（current_2 的多种子 mean±std，供 challenge_rules 算 MDD）

【输出】
  · results/stage5_tuned/README.md             —— 主报告
  · results/stage5_tuned/challenge_{clf,reg}.csv —— 逐条三态判定表
  · results/stage5_tuned/tuning_effect_{clf,reg}.csv —— 调参效应（阶段⑤ − 阶段②，配对）
  · results/stage5_tuned/leaderboard_{clf,reg}.csv   —— 各 K 榜首
  · results/stage5_tuned/fig_*.png

【裁判规则】复用 challenge_rules（空间折为主裁判；门槛 = MDD；三态 WIN/TIE/LOSE）。
⚠️ 擂台赛的"擂主"钉死为 **current_2** 在空间折下的最优模型，绝不用"全体最强固定集"，
   否则会变成和 common_6 比 —— 那偏离了用户指定的标杆。

用法：
    python -u src/analyze_stage5_challenge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import challenge_rules as CR                        # noqa: E402

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib import font_manager                # noqa: E402

for _f in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

S5_OUT = ROOT / "experiments" / "2026-10-13_stage5_tuned" / "outputs"
ZOO_OUT = ROOT / "experiments" / "2026-10-12_selector_zoo" / "outputs"
PROFILE_OUT = ROOT / "experiments" / "2026-10-13_champion_profile" / "outputs"
RES = ROOT / "results" / "stage5_tuned"
RES.mkdir(parents=True, exist_ok=True)

CHAMPION = "current_2"
RED, GREEN, BLUE, GREY = "#c0392b", "#2e7d32", "#1f6fb2", "#8a8a8a"

# 主指标：分类用 roc_auc（全项目选择器动物园口径），回归用 r2
METRIC = {"clf": "roc_auc", "reg": "r2"}

# ★ 擂主模型的选取必须**与被挑战的指标同口径**，否则等于用两把尺子量。
#   current_2 在空间折下：
#     · accuracy 口径 -> NB (0.8141)  ← 擂主档案的对外主数字
#     · roc_auc  口径 -> SVMLin (0.7961)
#   阶段⑤ 的矩阵是按 roc_auc 排的，所以裁判也必须在 roc_auc 下选擂主模型；
#   同时 report 里把 accuracy 口径的 NB 一并列出，避免"擂主偷偷换人"的误读。
CHAMPION_MODEL_OVERRIDE: dict[str, str] = {}   # 留空 = 按指标自动选


# --------------------------------------------------------------------------- #
# 擂主（钉死 current_2）
# --------------------------------------------------------------------------- #
def champion_ref(task: str) -> dict:
    """返回 current_2 在空间折下的**最优模型**作为擂主基准。

    与 analyze_challenge.py 的钉死逻辑一致：先在 current_2 内按模型聚合均值，
    取最高者；再取其空间折 mean/std/n_valid_min/n_folds。
    ★ 排序指标 = METRIC[task]（与挑战者同口径）。
    """
    metric = METRIC[task]
    d = pd.read_csv(PROFILE_OUT / f"profile_{task}_per_fold.csv", encoding="utf-8-sig")
    ch = d[(d.set == CHAMPION) & (d.scheme.str.startswith("B_spatial3"))]
    if ch.empty:
        raise SystemExit(f"no champion rows for {task}")
    if task in CHAMPION_MODEL_OVERRIDE:
        best_model = CHAMPION_MODEL_OVERRIDE[task]
    else:
        best_model = ch.groupby("model")[metric].mean().idxmax()
    r = ch[ch.model == best_model]
    base = r.groupby("model_seed")[metric].mean()
    std_seed = float(base.std(ddof=1)) if len(base) > 1 else np.nan
    return {
        "task": task, "metric": metric, "model": best_model,
        "mean": float(r[metric].mean()),
        "std_seed": std_seed,
        "n_valid_min": int(r.n_valid.min()),
        "n_folds": int(r.fold.nunique()),
        "scheme": "B_spatial3",
    }


def champion_alt_table(task: str) -> pd.DataFrame:
    """current_2 各模型在空间折下的 mean，用来披露"换指标就换擂主"。"""
    metric = METRIC[task]
    d = pd.read_csv(PROFILE_OUT / f"profile_{task}_per_fold.csv", encoding="utf-8-sig")
    ch = d[(d.set == CHAMPION) & (d.scheme.str.startswith("B_spatial3"))]
    cols = ["accuracy", "roc_auc"] if task == "clf" else ["r2"]
    g = ch.groupby("model").agg(**{f"{c}_mean": (c, "mean") for c in cols})
    return g.sort_values(f"{metric}_mean", ascending=False)


def mdd_of(ch: dict) -> float:
    return CR.min_detectable_difference(ch["std_seed"], ch["n_valid_min"],
                                        ch["n_folds"])


# --------------------------------------------------------------------------- #
# 口径聚合：把 B_spatial_s42/7/2026 三个分区种子合成一档
# --------------------------------------------------------------------------- #
def aggregate_schemes(s: pd.DataFrame, metric: str) -> pd.DataFrame:
    """把 per-fold 摘要按 (selector, model, k) 聚合，分成 A_random / B_spatial3。

    阶段⑤ 的 summary 已经在 (selector, model, k, scheme) 层聚合过，
    这里再跨 scheme 合并成一档：
      · A_random       -> scheme == 'A_random'
      · B_spatial3     -> scheme startswith 'B_spatial'（三种子合并求均值）
    """
    out = []
    for name, msk in (("A_random", s.scheme == "A_random"),
                      ("B_spatial3", s.scheme.str.startswith("B_spatial"))):
        g = s[msk].groupby(["selector", "model", "k"], as_index=False).agg(
            mean=(f"{metric}_mean", "mean"),
            spread=(f"{metric}_mean", "std"),
            n_features=("n_features", "first"))
        g["scheme"] = name
        out.append(g)
    return pd.concat(out, ignore_index=True)


def pivot(s: pd.DataFrame) -> pd.DataFrame:
    """把长表 (selector, model, k, n_features, scheme, mean, spread) 拉成宽表。

    输出列名形如  B_spatial3__mean / B_spatial3__spread / A_random__mean ...
    ⚠️ pandas 的 pivot_table 对多值列产生 MultiIndex 列（values × columns），
    展平顺序是 (value, scheme)，必须按 (value, scheme) 组合命名，否则会得到
    一串空列名。这里显式重建列名。
    """
    a = s.pivot_table(index=["selector", "model", "k", "n_features"],
                      columns="scheme", values=["mean", "spread"])
    a.columns = [f"{scheme}__{value}" for value, scheme in a.columns]
    return a.reset_index()


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_verdict_bars(vt: pd.DataFrame, metric: str, out: Path) -> None:
    if vt.empty:
        return
    t = vt.head(25).iloc[::-1]
    cols = [RED if v == "WIN" else (GREY if v == "TIE" else GREEN)
            for v in t.verdict]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(t) + 1.6)))
    ax.barh([f"{r.selector}\\n+{r.model} (K={r.k})" for r in t.itertuples()],
            t["delta_vs_champ"], color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(vt.mdd.iloc[0], color=RED, ls="--", lw=1)
    ax.axvline(-vt.mdd.iloc[0], color=GREEN, ls="--", lw=1)
    ax.set_xlabel(f"Δ {metric} vs current_2 champion  (red=WIN, grey=TIE, green=LOSE)")
    ax.set_title("stage5: tuned params vs champion (spatial folds, top 25)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_tuning_effect(te: pd.DataFrame, metric: str, out: Path) -> None:
    if te.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in te.groupby("model"):
        g = g.groupby("k")["delta_tuned_minus_default"].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=model, lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("K (# selected features)")
    ax.set_ylabel(f"Δ {metric}  (stage5 tuned − stage2 default)")
    ax.set_title(f"tuning effect on {metric}, by model (spatial folds)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_nfeat(lead: pd.DataFrame, metric: str, champ: dict, out: Path) -> None:
    """lead 的列是 k / selector / model / spatial / random / ...，不是 mean。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for sel, g in lead.groupby("selector"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["spatial"], marker="o", label=sel, lw=1.3)
    ax.axhline(champ["mean"], color=RED, ls="--", lw=1.4,
               label=f"champion {champ['model']} ({champ['mean']:.3f})")
    ax.set_xlabel("K (# selected features)")
    ax.set_ylabel(f"{metric} (spatial folds)")
    ax.set_title("stage5: best (selector, model) per K vs champion")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    emit("=" * 78)
    emit("阶段⑤ · 调优参数下的「选择器 × 模型」全矩阵 —— 逐条挑战 current_2")
    emit("=" * 78)

    summary = {}
    for task in ("clf", "reg"):
        s5 = pd.read_csv(S5_OUT / f"{task}_summary_full.csv", encoding="utf-8-sig")
        summary[task] = aggregate_schemes(s5, METRIC[task])
        emit(f"[load] {task}: stage5 summary rows = {len(s5)} "
             f"(selectors={s5.selector.nunique()}, models={s5.model.nunique()}, "
             f"K={sorted(s5.k.unique())})")

    # ---------------- 调参效应（配对：阶段⑤ − 阶段②/④，same selector/model/k）---- #
    # ★ 对照来源必须拼起来：
    #     · 随机折口径 -> 阶段②（*_summary_stage2.csv, scheme=A_random）
    #     · 空间折口径 -> 阶段④（*_summary_stage4.csv, scheme=B_spatial_s42）
    #   阶段⑤ 的空间折有三个分区种子，而阶段②/④ 只有 seed=42。
    #   为了严格配对（单一变量），这里**只比 s42 那一折**，不用三种子均值 ——
    #   否则"折的组成"也变了，就不是纯调参效应了。
    for task in ("clf", "reg"):
        metric = METRIC[task]
        s2 = pd.read_csv(ZOO_OUT / f"{task}_summary_stage2.csv", encoding="utf-8-sig")
        s4 = pd.read_csv(ZOO_OUT / f"{task}_summary_stage4.csv", encoding="utf-8-sig")
        ctrl_raw = pd.concat([s2, s4], ignore_index=True)
        a_ctrl = aggregate_schemes(ctrl_raw, metric)

        raw5 = pd.read_csv(S5_OUT / f"{task}_summary_full.csv", encoding="utf-8-sig")
        raw5 = raw5[raw5.scheme.isin(["A_random", "B_spatial_s42"])].copy()
        raw5["scheme"] = raw5["scheme"].replace({"B_spatial_s42": "B_spatial3"})
        a5 = aggregate_schemes(raw5, metric)

        key = ["selector", "model", "k", "scheme"]
        m = a5.merge(a_ctrl, on=key, suffixes=("_tuned", "_default"), how="inner")
        m["delta_tuned_minus_default"] = m["mean_tuned"] - m["mean_default"]
        m.to_csv(RES / f"tuning_effect_{task}.csv", index=False, encoding="utf-8-sig")

        sp = m[m.scheme == "B_spatial3"]
        emit(f"\n[调参效应|{task}] 空间折(s42)：阶段⑤(调优) − 阶段④(默认)，"
             f"n={len(sp)} 个 (选择器,模型,K) 组合")
        if len(sp):
            by_model = sp.groupby("model")["delta_tuned_minus_default"].agg(
                ["mean", "median", "min", "max"]).sort_values(
                    "mean", ascending=False)
            emit(f"-- 按模型（{metric}）--"); emit(by_model.round(4).to_string())
            by_k = sp.groupby("k")["delta_tuned_minus_default"].mean()
            emit(f"-- 按 K：{by_k.round(4).to_dict()}")
        rp = m[m.scheme == "A_random"]
        if len(rp):
            emit(f"[调参效应|{task}] 随机折：阶段⑤ − 阶段②，n={len(rp)}；"
                 f"总体均值 Δ = {rp['delta_tuned_minus_default'].mean():.4f}")

    # ---------------- 逐条挑战 ---------------- #
    report_verdicts = {}
    champion_summary = {}
    for task in ("clf", "reg"):
        metric = METRIC[task]
        champ = champion_ref(task)
        mdd = mdd_of(champ)
        champion_summary[task] = champ
        emit("\n" + "-" * 78)
        emit(f"[擂主|{task}] current_2 + {champ['model']} | 空间折 {metric} = "
             f"{champ['mean']:.4f} | std_seed={champ['std_seed']:.4f} | "
             f"MDD={mdd:.4f}")
        emit("  ⚠️ 换指标就换擂主 —— current_2 各模型在空间折下的均值：")
        alt = champion_alt_table(task).round(4)
        emit(alt.to_string())

        p = pivot(summary[task])
        p = p[p["B_spatial3__mean"].notna()]
        champ_obj = CR.Champion(
            set_name=CHAMPION, model=champ["model"], scheme="B_spatial3",
            metric=metric, mean=champ["mean"],
            std_across_folds=champ["std_seed"],
            std_across_seeds=champ["std_seed"],
            n_valid_min=champ["n_valid_min"], n_folds=champ["n_folds"])
        judge = CR.Judge(champions={("x", "y", "B_spatial3"): champ_obj})

        judges = []
        for r in p.itertuples():
            res = judge.judge(
                challenger_mean=r._asdict()["B_spatial3__mean"],
                scheme="B_spatial3",
                # challenger_std 只作诊断披露，不进门槛（见 challenge_rules 说明）
                challenger_std=r._asdict().get("B_spatial3__spread"))
            res.update({"selector": r.selector, "model": r.model, "k": r.k,
                        "n_features": r.n_features,
                        "spatial": r._asdict()["B_spatial3__mean"],
                        "random": r._asdict().get("A_random__mean")})
            judges.append(res)
        vt = pd.DataFrame(judges)
        vt = vt.rename(columns={"delta": "delta_vs_champ"})
        order = {"WIN": 0, "TIE": 1, "LOSE": 2}
        vt = vt.sort_values(["verdict", "delta_vs_champ"],
                            key=lambda s: s.map(order) if s.name == "verdict" else s)
        vt.to_csv(RES / f"challenge_{task}.csv", index=False, encoding="utf-8-sig")
        report_verdicts[task] = vt

        n = len(vt)
        counts = vt.verdict.value_counts().to_dict()
        emit(f"[挑战|{task}] 共 {n} 个挑战者 -> "
             f"WIN={counts.get('WIN',0)} / TIE={counts.get('TIE',0)} / "
             f"LOSE={counts.get('LOSE',0)}")
        emit("  最强 10 个挑战者（按 Δ 降序）：")
        top = vt.head(10)[["selector", "model", "k", "n_features",
                           "spatial", "random", "delta_vs_champ",
                           "mdd", "verdict"]]
        emit(top.round(4).to_string(index=False))
        emit("  " + CR.multiple_comparison_note(n))

        # 榜首表
        lead = vt.loc[vt.groupby("k")["spatial"].idxmax()][
            ["k", "selector", "model", "spatial", "random",
             "delta_vs_champ", "verdict"]].sort_values("k")
        lead.to_csv(RES / f"leaderboard_{task}.csv", index=False,
                    encoding="utf-8-sig")
        emit(f"[榜首|{task}] 每个 K 的最强组合：")
        emit(lead.round(4).to_string(index=False))

        plot_verdict_bars(vt, metric, RES / f"fig_verdict_stage5_{task}.png")
        te = pd.read_csv(RES / f"tuning_effect_{task}.csv", encoding="utf-8-sig")
        plot_tuning_effect(te[te.scheme == "B_spatial3"], metric,
                           RES / f"fig_tuning_effect_{task}.png")
        plot_nfeat(lead, metric, champ, RES / f"fig_nfeat_stage5_{task}.png")

    (RES / "_analysis_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
