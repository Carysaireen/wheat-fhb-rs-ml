"""挑战赛评审规则（challenge rules）
====================================

【这个模块在解决什么问题】
用户把 `current_2`（NREDI1 + NDVIre2）立为项目**擂主**，要求后续所有
「特征选择器 × 模型」的结果都来和它比。但"比"这个动作很容易被自己骗到：

    · 用随机折的数字去比空间折的数字  -> 口径不一致，不算比
    · 拿 0.8197 vs 0.8046（+0.015）就说"赢" -> 这是噪声，不是胜利
    · 在 N 个挑战者里挑最高的那个报告  -> 赢家诅咒，必然高估
    · 只在随机折上赢、空间折崩掉      -> 赢了也不能用

所以本模块把"谁赢"变成一个**可复算的判定函数**，而不是一句主观结论。

【判定规则（三条）】
1. **裁判口径**：`B_spatial3`（空间折）为主，`A_random5`（随机折）为辅。
   理由 —— 随机折有空间泄漏（P0-② / 阶段④b / 对齐基准三处独立证据），
   在它上面赢不算数。
2. **胜利门槛**：差值必须 **超过最小可分辨差（MDD）** 才算赢。
   MDD 由擂主的多模型种子 std 与折间样本刻度共同决定：
        MDD = max( 2*sqrt(2)*sigma_seed ,  1/n_valid_min )
   直觉解释见函数 docstring。差值落在 ±MDD 之内 -> 记作「打平」。
3. **三态输出**：WIN / TIE / LOSE，绝不输出"略微领先"这类模糊话。

【多重比较】
挑战者往往有几十上百个。本模块**不**替使用者做校正，但会**报告**
"一共比较了多少次"，并给出 Bonferroni-adjusted 门槛作为参考 ——
因为比较次数越多，"撞上一个赢家"的概率越高（假发现）。

用法：
    from challenge_rules import Judge
    j = Judge.from_profile(out_dir)          # 读擂主档案
    verdict = j.judge(challenger_mean, challenger_std, scheme="B_spatial3_s42")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

WIN, TIE, LOSE = "WIN", "TIE", "LOSE"


# --------------------------------------------------------------------------- #
# 最小可分辨差
# --------------------------------------------------------------------------- #
def min_detectable_difference(sigma_seed: float, n_valid_min: int,
                              n_folds: int) -> float:
    """计算"最小可分辨差"(MDD)。

    由两个独立的粗糙度来源取**较大者**，因为任何一个都足以淹没差异：

    (a) 种子噪声项  2*sqrt(2)*sigma_seed/sqrt(n_folds)
        · sigma_seed 是同一配置在不同模型种子下的标准差
        · 两个方案均值之差的标准误 = sigma*sqrt(2/n)（n = 折数）
        · 取 95% 近似 -> 乘 2
        含义：「换个随机种子，这个差异还在吗？」

    (b) 折样本刻度项  1/n_valid_min
        · 验证集最小规模 n_valid_min 时，一个样本翻转 =
          1/n_valid_min 的准确率变化
        · 含义：「这个差异，抵得上一个样本吗？」

    当 (a) 无法估计（只有单种子）时退化为 (b)，这也是为什么单种子的
    基线"不够格当裁判"。
    """
    if n_valid_min <= 0:
        raise ValueError("n_valid_min must be positive")
    resol = 1.0 / n_valid_min
    if sigma_seed is None or not np.isfinite(sigma_seed) or sigma_seed <= 0:
        return float(resol)
    seed_term = 2.0 * np.sqrt(2.0) * float(sigma_seed) / np.sqrt(max(n_folds, 1))
    return float(max(seed_term, resol))


# --------------------------------------------------------------------------- #
# 擂主档案
# --------------------------------------------------------------------------- #
@dataclass
class Champion:
    """擂主的一条记录（= 一个特征集 × 模型 × 口径）。"""
    set_name: str
    model: str
    scheme: str
    metric: str
    mean: float
    std_across_folds: float
    std_across_seeds: float
    n_valid_min: int
    n_folds: int

    @property
    def mdd(self) -> float:
        return min_detectable_difference(
            self.std_across_seeds, self.n_valid_min, self.n_folds)


@dataclass
class Judge:
    """评审员：持有擂主档案，对任意挑战者给出三态判定。"""
    champions: dict = field(default_factory=dict)   # (set, model) -> Champion

    # ---------------- 构造 ---------------- #
    @classmethod
    def from_profile(cls, out_dir: str | Path, metric_map: dict | None = None) -> "Judge":
        """从 run_champion_profile.py 的产出读入擂主。"""
        out_dir = Path(out_dir)
        metric_map = metric_map or {"clf": "accuracy", "reg": "r2"}
        champs = {}
        for task, metric in metric_map.items():
            f = out_dir / f"profile_{task}_per_fold.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, encoding="utf-8-sig")
            for (setname, model, scheme), g in d.groupby(["set", "model", "scheme"]):
                # 种子维度的 std：先把每个种子在折上平均，再跨种子求 std
                per_seed = g.groupby("model_seed")[metric].mean()
                champs[(setname, model, scheme)] = Champion(
                    set_name=setname, model=model, scheme=scheme, metric=metric,
                    mean=float(g[metric].mean()),
                    std_across_folds=float(g.groupby("fold")[metric].mean().std()),
                    std_across_seeds=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                    n_valid_min=int(g.n_valid.min()),
                    n_folds=int(g.fold.nunique()))
        return cls(champions=champs)

    # ---------------- 查询 ---------------- #
    def champion_of(self, set_name: str, scheme: str) -> Champion:
        """某个特征集在某口径下的最佳模型（按 mean 排）。"""
        cands = [c for (s, _m, sc), c in self.champions.items()
                 if s == set_name and sc == scheme]
        if not cands:
            raise KeyError(f"no champion for {set_name} @ {scheme}")
        return max(cands, key=lambda c: c.mean)

    def best_champion(self, scheme: str) -> Champion:
        """全体特征集在该口径下的最强擂主。"""
        cands = [c for (_s, _m, sc), c in self.champions.items() if sc == scheme]
        if not cands:
            raise KeyError(f"no champion @ {scheme}")
        return max(cands, key=lambda c: c.mean)

    # ---------------- 判定 ---------------- #
    def judge(self, challenger_mean: float, scheme: str,
              challenger_std: float | None = None,
              against: tuple[str, str] | None = None) -> dict:
        """给出三态判定。

        参数
        ----
        challenger_mean : 挑战者的指标均值
        scheme          : 用哪一档折口径裁判（建议 B_spatial3 聚合或 B_spatial3_s42）
        challenger_std  : 挑战者的跨种子 std —— **只用于诊断性披露，不进门槛**
        against         : (set_name, model) 指定和哪个擂主比；缺省用最强擂主

        返回
        ----
        dict(verdict, delta, mdd, champion=..., challenger_std=...)

        ★ 为什么门槛**只由擂主决定**（2026-10-13 阶段⑤ 修正）
        --------------------------------------------------------
        原实现把门槛写成 max(champ.mdd, 挑战者自身的种子噪声项)，
        结果是一个"翻车越狠反而越安全"的悖论：
        某组合空间折 R² = −2.16、跨分区种子 std 极大，门槛被抬到 2.83，
        于是 −2.28 的巨亏被判成"打平（TIE）" —— 完全反了。

        正确的语义是：**我们比较的是"这个挑战者能否取代擂主"**，
        噪声基准应当来自**擂主**（它是被比较的参照物）。挑战者自己不稳，
        是它**不能胜任**的证据，而不是给自己开脱的理由。
        所以门槛 = 擂主的 MDD；挑战者的 std 只写进结果里供人查看。
        （若挑战者均值高于擂主但极不稳定，它仍会因"没超过 MDD"而判 TIE，
          这是保守的、可接受的方向。）
        """
        champ = (self.champion_of(*against, scheme=scheme) if against
                 else self.best_champion(scheme))
        mdd = champ.mdd
        delta = float(challenger_mean) - champ.mean
        if delta > mdd:
            verdict = WIN
        elif delta < -mdd:
            verdict = LOSE
        else:
            verdict = TIE
        return {
            "verdict": verdict,
            "delta": round(delta, 4),
            "mdd": round(mdd, 4),
            "champion": f"{champ.set_name}+{champ.model}",
            "champion_mean": round(champ.mean, 4),
            "challenger_mean": round(float(challenger_mean), 4),
            "challenger_std": (None if challenger_std is None
                               or not np.isfinite(challenger_std)
                               else round(float(challenger_std), 4)),
            "scheme": scheme,
        }


# --------------------------------------------------------------------------- #
# 多重比较提示
# --------------------------------------------------------------------------- #
def multiple_comparison_note(n_comparisons: int, alpha: float = 0.05) -> str:
    """报告比较次数带来的假发现风险，并给出 Bonferroni 参考门槛。"""
    bonf = alpha / max(n_comparisons, 1)
    return (f"本次共比较 {n_comparisons} 次；不做校正时，"
            f"纯靠运气至少撞出一个'显著'的概率约 "
            f"{1 - (1 - alpha) ** max(n_comparisons, 1):.3f}；"
            f"Bonferroni 参考门槛 alpha' = {bonf:.2e}。")


# --------------------------------------------------------------------------- #
# 结果表工具
# --------------------------------------------------------------------------- #
def verdict_table(judge: Judge, challengers: pd.DataFrame, scheme: str,
                  mean_col: str, std_col: str | None = None,
                  name_cols=("set", "model")) -> pd.DataFrame:
    """把一张挑战者结果表批量判定为 赢/平/输。"""
    rows = []
    for _, r in challengers.iterrows():
        v = judge.judge(r[mean_col], scheme=scheme,
                        challenger_std=(r[std_col] if std_col and std_col in r else None))
        v.update({c: r[c] for c in name_cols if c in r})
        rows.append(v)
    out = pd.DataFrame(rows)
    order = {WIN: 0, TIE: 1, LOSE: 2}
    return out.sort_values(["verdict", "delta"],
                           key=lambda s: s.map(order) if s.name == "verdict" else s)
