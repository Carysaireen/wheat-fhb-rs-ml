# results/spatial_validation — 空间验证章节素材

> 生成脚本：`src/plot_spatial_validation.py`
> 数据来源：`experiments/2026-10-11_spatial_cv/outputs/`（只用训练集内部 CV，未读封存测试集）
> 生成日期：2026-10-11

## 图表清单

| 文件 | 内容 | 论文用途 |
|---|---|---|
| `fig1_gap_classification.png` | 分类准确率：随机折 vs 空间分组折 + gap | 空间泄漏程度（分类） |
| `fig2_gap_regression.png` | 回归 R²：随机折 vs 空间分组折 + gap | 空间泄漏程度（回归） |
| `fig3_k_sensitivity.png` | k=8/10/12 三种块数的 gap 对照 | 结论稳健性证据 |
| `fig4_fold_design.png` | 折大小与标签构成审计 | 方法透明度 |
| `fig5_spatial_map.png` | 空间块归属 + 标签空间分布 | 现象解释（为何有泄漏） |

## 核心数字（k=10, 3 折, 3 seeds 均值）

**分类（准确率）**

| 模型 | A 随机折 | B 空间折 | gap |
|---|---|---|---|
| LR | 0.713 | 0.588 | +0.124 |
| RF | 0.752 | 0.679 | +0.073 |
| SVM | 0.783 | 0.729 | +0.053 |

**回归（R²）**

| 模型 | A 随机折 | B 空间折 | gap |
|---|---|---|---|
| RF | 0.394 | -0.296 | +0.690 |
| Ridge | 0.072 | -1.481 | +1.554 |
| SVR | 0.169 | -0.737 | +0.906 |

## 写作要点（诚实性）

1. **两套数字必须并报**：随机 CV 偏乐观（空间泄漏），空间分组 CV 偏保守（外推困难），真相介于两者之间。
2. **不要只强调 gap 大**：空间折下 R² 转负，部分来自观测空间过小（0.68°）导致的外推困难，
   这正是 Roberts et al. 2017 警告的保守方向。
3. **领域先例**：Morellos et al. 2025 同域对照（跨田块 R²≤0.71 vs 田块内 R²≤1.00），
   证明空间泄漏量级巨大，本结果与之同类。
4. **引用**：Roberts et al. 2017 (Ecography 40:913-929)；Morellos et al. 2025 (CEA 231:109967)。

## 方法术语与论文可引用表述

本项目内部脚本/目录名为 `spatial_group_cv`（“空间分组交叉验证”），但在遥感/生态文献中的**规范术语**是
**空间块交叉验证（spatial block cross-validation, blockCV）**，属于 **空间交叉验证（spatial cross-validation, spatial CV）** 家族。

| 术语 | 含义 | 与本项目的关系 |
|---|---|---|
| Spatial CV（空间交叉验证） | 泛指任何显式考虑空间依赖的 CV | 总称 |
| **Block CV / blocked CV（空间块交叉验证）** | 按空间块切分、整块留出 | ★ 本实验做法；论文首选术语 |
| Spatial group CV（空间分组交叉验证） | 强调分组变量由空间坐标导出 | 本项目内部命名 `spatial_group_cv` |
| Leave-one-block-out CV | 每次只留出 1 个块 | 本项目留出多块（3 折），为其变体 |
| Buffered CV（带缓冲带的块 CV） | 留出块周围再剔除一圈训练样本 | 本项目未启用（见 config.yaml → no_buffer） |

**论文 Methods 可引用表述（可直接采用）**：

> To quantify the optimism caused by spatial autocorrelation, we compared random stratified
> 5-fold cross-validation against **spatial block cross-validation**. Spatial blocks were derived
> by K-means clustering on geographic coordinates (k = 8/10/12; k = 10 as primary), and folds were
> built with StratifiedGroupKFold (3 folds) so that entire blocks were held out while class
> proportions were preserved. No buffer was applied at this stage. The difference between the two
> schemes (A − B) serves as a diagnostic of spatial leakage.

**可引用文献**：

- Roberts et al. 2017, *Ecography* 40(8):913-929, DOI 10.1111/ecog.02881 — block CV 奠基综述
- Morellos et al. 2025, *Comput. Electron. Agric.* 231:109967 — 同域空间验证实证（跨田块 R²≤0.71 vs 田块内 ≤1.00）

## 复现

```bash
cd project_v2
python src/spatial_group_cv.py          # 先跑实验（生成 outputs/）
python src/plot_spatial_validation.py   # 再生成图表
```
