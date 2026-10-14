# FS-CS 特征选择器 · 空间验证结果

> 实验目录：`project_v2/experiments/2026-10-12_fs_cs/`
> 生成脚本：`src/run_fs_cs_experiment.py`（数据）→ `src/plot_fs_cs.py`（图）
> 日期：2026-10-12 ｜ 状态：done

## 一、这个实验在问什么

把 **FS-CS**（Feature Selection with Compactness and Separability）作为**第 4 种特征选择器**，
和项目既有的 RF重要性+MI、Boruta 放在**同一套折**里比较，并同时给出两种口径：

- **A_random**：随机分层 5 折（`data/cv/classification_folds.csv`）
- **B_spatial**：空间块交叉验证（KMeans k=10 分块 + `StratifiedGroupKFold` 3 折，seeds 42/7/2026）

> 术语：见 `results/spatial_validation/README.md` —— 本实验沿用的空间块 CV 方法完全一致。
> 关键区别：本实验把**所有选择器**都放进空间折，因此得出的差异可以干净地归因于"选择器"，
> 而不是"特征数量以外的其他因素"。

## 二、方法要点

| 项目 | 设置 |
|---|---|
| 打分（主口径） | `J(j) = S_B(j) / S_W(j)`（类间散度 / 类内散度，即 Fisher 分数） |
| 冗余感知变体 | 标准化后贪心前向，最大化 `trace(S_W⁻¹·S_B)` |
| K | 10 / 20 / 30 |
| 模型 | LR / SVM(RBF) / RF（与 `feature_set_cv` 定义一致） |
| 协议 | **nested**：每个外层训练折内重新选择（无泄漏）；**global**：折聚合排名定集合 |
| 测试集 | 全程未读 `test_locked.csv` |

**性质**：`J` 是比值 → 尺度无关；无随机性 → 可复现；无超参数（除 K）。

## 三、图表

| 文件 | 内容 | 论文里放哪 |
|---|---|---|
| `fig1_selector_spatial_gap.png` | (a) 各选择器空间 gap（3 模型）(b) gap 随特征数上升趋势 | 空间验证·选择器对照 |
| `fig2_fs_cs_stability.png` | FS-CS 单变量选出的前 20 特征（跨折平均排名） | 特征选择·稳定性 |
| `fig3_random_vs_spatial_rf.png` | RF 随机折→空间折位移（哑铃图，线段越长越敏感） | 现象解释 |

## 四、核心数字（准确率，来自 `outputs/selector_benchmark.csv`）

```
set                      n_feat  model  acc_A   acc_B    gap(A-B)
--------------------------------------------------------------------
current_2                   2      LR   0.8046  0.8023   +0.0022
current_2                   2      SVM  0.7969  0.8090   -0.0121
rfmi_top3of5                6      RF   0.7664  0.7505   +0.0159
fs_cs_univariate_top10     10      LR   0.7672  0.7658   +0.0014
fs_cs_univariate_top10     10      SVM  0.7972  0.7918   +0.0054
fs_cs_univariate_top20     20      RF   0.7746  0.7111   +0.0636
boruta_3of5                30      LR   0.7818  0.6783   +0.1035
boruta_3of5                30      RF   0.7667  0.6856   +0.0810
all_117                   117      LR   0.6829  0.5794   +0.1035
all_117                   117      RF   0.7595  0.6945   +0.0651
```

## 五、结论

1. **空间 gap 随特征集规模单调上升**：2 特征 ≈ 0.00~0.03，30 特征 ≈ 0.03~0.10，117 特征 ≈ 0.06~0.10。
2. **Boruta 最"怕空间"**：随机折 ROC-AUC 最高（RF 0.853），空间折掉幅最大（RF gap +0.081）。
3. **FS-CS（单变量）在 10 特征档最稳**：LR gap +0.001、SVM +0.005，绝对空间准确率优于 Boruta / all_117。
4. **最稳健的整体解仍是原有的 `current_2`（2 特征）**：空间折准确率最高且 gap≈0。反直觉但证据一致。
5. **冗余感知的 greedy 变体未胜过单变量版** —— 诚实负结果。

## 六、可引用表述（英文，可直接放 Methods）

> Feature selection was performed with four paradigms under identical folds:
> random-forest importance intersected with mutual information, Boruta, and two
> variants of **FS-CS** (Feature Selection with Compactness and Separability),
> a univariate class-separability (Fisher) score `J = S_B / S_W` and a
> redundancy-aware greedy subset criterion `trace(S_W⁻¹ S_B)`. Selection was
> refitted **inside every training fold** so that no validation information
> leaked. All selectors were additionally evaluated under **spatial block
> cross-validation** (K-means blocks on coordinates, `StratifiedGroupKFold`)
> to isolate the optimism caused by spatial autocorrelation.

## 七、引用

- **Sun H., Mai H., Mao Y., et al. (2025).** *Multi-variety monitoring of potato
  late blight severity using UAV data with improved SMOTE-CS for small sample
  modeling and deep feature learning.* **European Journal of Agronomy**
  (PII: S1161030125001984). ← FS-CS 的文献出处
- Fisher, R. A. (1936) —— 类可分性/Fisher 判据的理论来源。
- Roberts et al. (2017), *Ecography* 40:913-929 —— 空间块 CV 奠基综述。

> ⚠️ 更正：本项目的文献检索报告此前把 FS-CS 误引为 "Zhou et al. 2023"，已于 2026-10-12 更正。

## 八、复现

```bash
cd project_v2
D:/Anaconda/python.exe src/run_fs_cs_experiment.py    # ~1.5 分钟
D:/Anaconda/python.exe src/plot_fs_cs.py
```
