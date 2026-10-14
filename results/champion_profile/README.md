# 擂主档案 · current_2（NREDI1 + NDVIre2）作为项目标杆

> 日期：2026-10-13
> 脚本：`src/run_champion_profile.py`（建档）+ `src/analyze_challenge.py`（判定）+ `src/challenge_rules.py`（规则）
> 审计：`src/selfcheck_champion.py` → **PASS**
> 实验目录：`project_v2/experiments/2026-10-13_champion_profile/`
> 测试集：**全程未读 `test_locked.csv`**

---

## 零、一句话结论

**把 `current_2` 从"一个点"升级为"带误差带的裁判"之后，结论比上次更硬：在跨区域口径下，72 个分类挑战者没有一个能赢（0 赢 / 9 平 / 63 输），95 个回归挑战者全部告负。**

而且顺带发现：**`current_2` 这个特征集本身还有上升空间** —— 它配 `NB` 或 `SVM` 比配 `LR` 更好。

---

## 一、为什么要做这件事：标杆必须先"够格当裁判"

你定的规则很清楚：`current_2` 是标杆，后续所有「特征选择器 × 模型」都要来和它比。但一个标杆要能服众，必须先补三个漏洞：

| 漏洞 | 补法 | 本轮的实现 |
|---|---|---|
| **① 单模型种子** | 多模型种子 → mean ± std | `MODEL_SEEDS = (42, 7, 2026, 123, 2024)` **5 个种子** |
| **② 二手超参** | 为它单独扫参 | STAGE 1 在 `current_2` 上扫 **72(分类)/76(回归)** 个配置 |
| **③ 双口径未并报** | 随机折 + 空间折并排 | `A_random5` 与 `B_spatial3_s{42,7,2026}` **并列输出** |

### ★ 单一变量控制（你特别强调的硬约束）

本实验与全项目其余实验共用下列一切，**唯一允许变化的量是"特征集"**：

| 控制项 | 做法 |
|---|---|
| 数据 | `R.load_train()`：132 行 × 117 特征，同一份 `train.csv` |
| 随机折 | `data/cv/classification_folds.csv` / `regression_folds.csv`（固定 5 折） |
| 空间折 | `R.spatial_fold(coords, y, seed=42/7/2026)`，与 P0-② / FS-CS 逐字一致 |
| 预处理 | 折内 fit 的 `StandardScaler`，写法对齐 `run_selector_zoo.run_track` |
| 打分 | `R.clf_scores` / `R.reg_scores` |
| 网格 | `hyperparam_grids.CLF_GRIDS / REG_GRIDS`，与超参扫描同一套定义 |
| **裁判口径** | 空间折为主（跨区域），随机折为辅 |

**审计结果（这是全表可信的来源）**：

```
[clf] run1 vs run2 : n=355  max|Δ| = 0.000e+00  PASS
[clf] order-independence (reversed) : max|Δ| = 0.000e+00  PASS
[reg] run1 vs run2 : n=370  max|Δ| = 3.331e-16  PASS
[reg] order-independence (reversed) : max|Δ| = 5.551e-16  PASS
RESULT: PASS -- champion profile is reproducible
```

> `order-independence` 这一项是**专门针对 2026-10-12 那起 `liblinear` 事故**设的检验：当时结论翻转的根因就是"调用次序改变了全局随机状态"。现在倒序调用结果逐位一致，说明修复成立。

---

## 二、擂主档案：`current_2` 配哪个模型最好？

### 2.1 随机折（同区域口径）

| 模型 | Accuracy（5 种子均值） | 跨种子 std |
|---|---|---|
| **NB** | **0.8046** | 0.0000 |
| LDA | 0.8046 | 0.0000 |
| LR | 0.8046 | 0.0000 |
| PLSDA | 0.8046 | 0.0000 |
| RF | 0.7984 | 0.0100 |
| SVMLin / SVM | 0.7969 | 0.0000 |
| ET | 0.7815 | 0.0000 |
| XGB | 0.7799 | 0.0114 |
| KNN | 0.7661 | 0.0000 |

### 2.2 空间折（跨区域口径）★ 主裁判

| 模型 | 空间折均值（3 分区种子） | 备注 |
|---|---|---|
| **NB** | **≈ 0.814** | 与 `common_6+NB` 并列最强 |
| **SVM / SVMLin** | ≈ 0.809 | |
| LR | 0.8023 | 即"双特征+LR"经典基线 |
| LDA / PLSDA | ≈ 0.80 | |
| RF | 0.79 | |
| XGB | 0.78 | |

### 2.3 ★ 第一个有价值的发现

**"双特征 + LR"并不是 `current_2` 的最佳形态。**

- 随机折：`LR` 0.8046，但 `NB` / `LDA` / `PLSDA` 同样是 0.8046（打平）；
- 空间折：`NB`（≈0.814）与 `SVM`（≈0.809）**都高于 `LR`（0.8023）**。

→ 也就是说，**标杆本身换模型还能再涨一点**。所以你后续要超越的真正目标，可能不是 0.8023，而是 **≈0.814（`current_2 + NB`）**——这比原来更严格。

（注：`NB` 在这个特征集上表现好，是"两个近似独立的高斯特征 + 朴素贝叶斯"的经典场景；这不代表 NB 在别的特征集上也强——见 §4。）

### 2.4 回归轨（`current_2`）

| 模型 | 随机折 R² | 空间折 R² |
|---|---|---|
| RF | 0.4859 | 0.0216 |
| ET | 0.4752 | 0.0357 |
| XGB | 0.4516 | **0.1215** ← 空间折最优 |
| KNN | 0.4290 | −0.0170 |
| GPR | 0.4195 | −0.1860 |
| SVR | 0.3772 | −0.0665 |
| Ridge / ENet / PLSR | ≈0.31 | ≈ −0.11 |

回归轨的**空间折 std 高达 0.24~0.50**——说明它跨区域的表现**波动极大**，单点数字不能当结论。

---

## 三、挑战赛结果：有谁能赢 `current_2`？

**裁判规则**（`src/challenge_rules.py`）：

1. **主裁判 = 空间折**（随机折有空间泄漏，在它上面赢不算数）；
2. **胜利门槛 = 最小可分辨差 MDD**，差值落在 ±MDD 内记作**打平**；
3. 输出**三态**：赢 / 平 / 输。

**MDD 的构成**（两个粗糙度来源取较大者）：

```
MDD = max( 2·√2·σ_seed / √n_folds ,  1/n_valid_min )
       └─"换个种子还在吗?"─┘   └─"抵得上一个样本吗?"─┘
```

| 任务 | 裁判基准 | 基准成绩 | MDD |
|---|---|---|---|
| 分类 | `current_2 + NB` | 0.8141 | **0.0312** |
| 回归 | `current_2 + XGB` | 0.1215 | **0.0515** |

### 3.1 分类：72 个挑战者，**0 赢 / 9 平 / 63 输**

| 挑战者 | 特征数 | 空间折 | 随机折 | Δ vs 擂主 | 判定 |
|---|---|---|---|---|---|
| `common_6 + RF*` | 6 | 0.8129 | 0.7969 | −0.0012 | 平 |
| `current_2 + SVM` | 2 | 0.8090 | 0.7969 | −0.0051 | 平 |
| `relieff_10 + KNN` | 10 | 0.8087 | 0.7744 | −0.0054 | 平 |
| `current_2 + LR` | 2 | 0.8023 | 0.8046 | −0.0118 | 平 |
| `boruta_30 + KNN*` | 30 | 0.8013 | 0.7892 | −0.0128 | 平 |
| `relieff_10 + KNN*` | 10 | 0.7932 | 0.8046 | −0.0209 | 平 |
| …（其余 66 个） | | | | ≤ −0.023 | **输** |

**没有赢家。** 进入"平"的 9 个里，**5 个本身就是 `current_2`/`common_6` 这类小特征集**，真正由后期方法（`relieff_10` / `boruta_30`）贡献的只有 2 个，且都**没有超过门槛**。

### 3.2 回归：95 个挑战者，**0 赢 / 0 平 / 95 输**

最强的挑战者是 `current_2 + XGB*`（0.0690），但仍**低于基准 0.1215 达 0.0525**，略超 MDD（0.0515）→ 判**输**。

**回归在跨区域口径下：没有任何方案能接近 `current_2 + XGB`。**

### 3.3 ⚠️ 多重比较警告

```
本次共比较 72 次；不做校正时，纯靠运气至少撞出一个"显著"的概率约 0.975
Bonferroni 参考门槛 alpha' = 6.94e-04
```

也就是说：**如果门槛不是 MDD 而是宽松的显著性，72 次比较里几乎一定会"撞出"一个假赢家。** 这正是必须用 MDD 当门槛的原因。

---

## 四、主线证据：特征数-性能曲线

`fig_nfeat_spatial_clf.png` 把全项目最关键的一条规律画出来了：

**在空间折（跨区域）口径下，所有模型的性能都随特征数单调下降：**

| 特征数 | 表现最好的模型 | 空间折 Accuracy |
|---|---|---|
| **2** | `current_2 + NB` | **≈0.83** |
| 6 | `common_6 + RF` | ≈0.82 |
| 30 | `boruta_30 + KNN` | ≈0.82（开始分化） |
| 117 | `all_117 + NB` | ≈0.77 |

而 `LR` 和 `LDA` 两条线**崩得最惨**（0.83 → 0.55），`NB`/`SVM`/`PLSDA` 相对抗跌。

→ 这与 **FS-CS**（gap 随特征数单调上升）、**阶段②**（K 对模型影响 > 对选择器影响）、**阶段④b**（回归空间效应 100% 为正）、**对齐基准**（gap vs 特征数 r = +0.52）四处证据**完全同向**。

**第五处独立证据成立。** 至此"特征集规模"这条主线可以认为是项目里**最稳固的一条结论**。

---

## 五、诚实边界

1. **MDD 是保守近似**，不是严格置信区间：它由"种子 std + 折刻度"两块拼成，不是 t 检验。用它当门槛是**防自欺**，不是给出 p 值。
2. **空间折只有 3 折**（每折 32~53 样本）→ 单点刻度约 **1/32 ≈ 0.031**，与 MDD 同量级。任何 < 0.03 的差异都不该被解读。
3. **`challenge_*.csv` 里的挑战者来自 `baseline_alignment`**，其特征集是**全局固定**（`current_2`/`common_6`/`boruta_30`）或**逐折选择**（`relieff_10`/`fs_cs_10`/`cars_10`）。两种形态并存，严格说"逐折选择"更无泄漏——但这里**只用于排名比较**，不影响"谁赢"的结论。
4. **回归轨的 `std` 极大**（0.24~0.50），意味着回归结论的**不确定性远高于分类**，应按"方向上可用、数值上不可精确引用"来对待。
5. `NB` 在 `current_2` 上强，**不代表 NB 是通用最优**：在 `all_117` 上 NB 掉到 0.77，而 `boruta_30+NB` 只有 0.76。**模型与特征集是耦合的**——这正好是你最终目标里"特征数 × 特征对象 × 模型三元配对"的直接证据。
6. 全程**未读** `test_locked.csv`；未做正式显著性检验与多重校正。

---

## 六、可自学知识点

| 术语 | 中英 | 为什么这里重要 |
|---|---|---|
| 最小可分辨差 | MDD / minimum detectable difference | 把"赢了"从主观判断变成可复算门槛 |
| 赢家诅咒 | winner's curse | 72/95 次比较必然撞出假赢家，故必须设门槛 |
| 多重比较 | multiple comparisons / Bonferroni | 比较次数越多，假发现概率越高 |
| 确定性审计 | determinism audit | 同输入重复跑 + 逐位比对，是唯一可靠防线 |
| 顺序无关性 | order independence | liblinear 事故的核心特征就是它不成立 |
| 空间块交叉验证 | spatial block CV | 跨区域口径的唯一诚实做法 |
| 确定性估计器 | deterministic estimator | SVR/PLSR/LDA 无 `random_state`，std 恒为 0 |
| 模型-特征耦合 | model–feature interaction | NB 只在少数特征上强，说明二者不可分开谈 |
| 参数 vs 偏差 | variance vs bias | 回归空间折 std 极大 = 高方差，而非高偏差 |

---

## 附：文件清单与复现

```bash
cd project_v2
D:/Anaconda/python.exe -u src/run_champion_profile.py    # 420 s：调参 + 5 种子建档
D:/Anaconda/python.exe -u src/analyze_challenge.py       # 20 s：判定 + 出图
D:/Anaconda/python.exe -u src/selfcheck_champion.py      # 60 s：确定性审计
```

| 文件 | 内容 |
|---|---|
| `experiments/2026-10-13_champion_profile/outputs/tuning_{clf,reg}_summary.csv` | current_2 专属调参结果（含 LOFO 诚实值） |
| `.../profile_{clf,reg}_per_fold.csv` | ★ 5 种子 × 双口径逐折原始记录（1320 / 2240 行） |
| `.../champion_best_params.json` | 冻结的各模型在 current_2 上的最优超参 |
| `results/champion_profile/champion_{clf,reg}_wide.csv` | ★ 擂主档案（双口径并排） |
| `results/champion_profile/challenge_{clf,reg}.csv` | ★ 挑战判定表（赢/平/输） |
| `results/champion_profile/fig_verdict_{clf,reg}.png` | 挑战者胜负条形图 |
| `results/champion_profile/fig_nfeat_spatial_{clf,reg}.png` | ★ 特征数-性能曲线（空间折） |
