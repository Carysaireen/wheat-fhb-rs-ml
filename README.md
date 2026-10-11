<div align="center">

# 小麦赤霉病特征降维与模型构建

**Fusarium Head Blight (FHB) Severity Modeling from Multispectral Features**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-active--iterating-orange)
![Data](https://img.shields.io/badge/Samples-165-lightgrey)
![Features](https://img.shields.io/badge/Features-117-lightgrey)

</div>

> 在 **165 样本 × 117 特征**的高维小样本约束下，用尽量少、可解释的特征判断小麦是否患病并估计病情严重度——**严格遵循无泄露交叉验证**，得到一个诚实的结论，并明确锁定了下一阶段的改进方向。

---

## 目录

- [这个项目做什么](#这个项目做什么)
- [核心结果](#核心结果)
- [快速开始](#快速开始)
- [安装与环境](#安装与环境)
- [复现实验](#复现实验)
- [方法路线](#方法路线)
- [科研规范](#科研规范)
- [GNN 的边界](#gnn-的边界)
- [进行中：改进方向](#进行中改进方向)
- [目录结构](#目录结构)
- [参考文献](#参考文献)
- [许可证](#许可证)

---

## 这个项目做什么

小麦赤霉病（FHB）是世界性小麦病害。本项目针对遥感病害检测中最难的一类场景——**高维小样本（HD-SS）**：仅 165 个样本、117 维特征（植被指数 / 波段统计量 / GLCM 纹理），系统比较了「传统特征优选」与「深度表示学习（AE / GNN）」在两类任务上的表现：

1. **分类**：判断「健康 / 患病」；
2. **回归**：估计病情严重度 DI（有界连续量 [0,100]）。

**你能从这里得到什么**：一套可复现的「无泄露小样本建模」流程，以及一个关于「小样本下何时该用深度学习、何时不该用」的真实案例。

---

## 核心结果

> 所有数字均为**无泄露嵌套交叉验证**的严格结果，出处见 `results/` 下的指标 CSV。

| 方法 | 特征维度 | 分类准确率 | 回归 R²（DI） | 可解释性 |
|---|---|---|---|---|
| 传统特征 + 随机森林（Boruta 特征） | 30 | **≈0.79**（AUC 0.86） | **≈0.449** | ★★★ |
| 传统特征 + SVM / LR | 2–30 | ≈0.76–0.80 | 0.30–0.39 | ★★★ |
| AutoEncoder 2D + RF | 2 | ≈0.71 | ≈0.26 | ★☆☆ |
| 图神经网络（SageAE）2D + RF | 2 | ≈0.77 | ≈0.35 | ★★☆ |

**三个客观结论：**

1. **分类不需要深度学习** —— 少量红边指数即可达 ~0.80，且物理可解释；
2. **严重度回归是小样本硬天花板** —— 传统 RF 的 R²≈0.449 已是最优；
3. **朴素深度降维整体不如传统特征** —— 但归纳式 GNN（R²≈0.35）明显优于普通 AE（R²≈0.26），说明空间先验确有增益。

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Carysaireen/wheat-fhb-rs-ml.git
cd wheat-fhb-rs-ml

# 2. 安装依赖
pip install -r requirements.txt

# 3. 跑通完整流程（6 个脚本，按顺序执行）
cd src
python create_split.py                 # 划分并封存测试集
python traditional_baseline_cv.py      # 传统基线（交叉验证）
python nested_traditional_tuning.py    # 传统方法嵌套调优
python autoencoder_cv.py               # AE 无泄露评估（需 torch）
python gnn_inductive_cv.py             # GNN 归纳式评估（需 torch）
python nested_deep_tuning.py           # 深度方法嵌套调优（需 torch）
```

> 只想看**传统方法**的结果？前三步即可，无需安装 `torch`。

---

## 安装与环境

| 项 | 要求 |
|---|---|
| Python | ≥ 3.9 |
| 必需依赖 | `numpy` `pandas` `scikit-learn` `matplotlib` `boruta` |
| 可选依赖（仅 AE/GNN） | `torch` `torch-geometric` |

```bash
# 最小安装（仅传统方法）
pip install numpy pandas scikit-learn matplotlib boruta

# 完整安装（含深度学习）
pip install -r requirements.txt
```

**注意事项（隐性契约，务必阅读）：**

- 数据源文件为 **GBK 编码**（`data/raw/Wheat_Scab_Features_Extraction_Ordered_region.csv`），脚本已用 `encoding='gbk'` 读取，**请勿另存为 UTF-8**；
- 脚本通过 `Path(__file__).resolve().parents[1]` 定位仓库根目录，**必须在 `src/` 目录下运行**（`cd src` 后再执行）；
- 部分脚本原本引用本地 `vendor/` `vendor_torch/` 目录（本项目打包的离线依赖），现已改为**优雅降级**：目录存在则优先使用，不存在则回退到系统安装的包。**仓库不包含 vendor 目录**（见 `.gitignore`），请正常 `pip install`。

---

## 复现实验

完整实验分三步，依次产出可复现的结果：

```bash
cd src

# Step 1 — 数据划分（封存测试集，random_state=42，20% 分层）
python create_split.py

# Step 2 — 传统基线 + 嵌套调优
python traditional_baseline_cv.py
python nested_traditional_tuning.py

# Step 3 — 深度学习（可选，需 torch）
python autoencoder_cv.py
python gnn_inductive_cv.py
python nested_deep_tuning.py
```

各脚本输出的指标 CSV 会写入 `results/<阶段>/`。汇总结果见 `results/stage_summary/`。

---

## 方法路线

```
第一阶段 · 传统特征优选
  Pearson 去冗余 + ReliefF + 随机森林重要性 + Boruta 三重过滤
  → 117 维 → 16 个核心特征 → 「黄金双特征」NREDI1 + NDVIre2

第二阶段 · 严重度回归探索
  直接回归 → 发现 DI=0 干扰 → 两阶段「先分类再回归」→ 纯患病样本回归

第三阶段 · 深度学习降维
  普通 AutoEncoder（117→2）  vs  图神经网络（光谱 + 空间建图，117→2）
```

---

## 科研规范

严格遵循 **无泄露（leakage-free）** 实验设计：

- 测试集**一次性封存**，探索阶段只用训练集及内部交叉验证；
- 按行分层的 `StratifiedKFold`（分类）/ `KFold`（回归）；
- 逐训练折**独立拟合**缺失值填补、标准化、AE / GNN 编码器；
- 深度表示学习采用**训练图 + 验证节点归纳式隔离**编码。

> 正是这套流程，让结论从初版探索期的乐观估计（分类 0.82 / R² 0.5）收敛到严格估计（分类 ~0.80 / R² 0.26–0.449）。**敢于用更严的方法推翻自己的漂亮数字，是本项目的核心价值。**

---

## GNN 的边界

GNN 是本项目试错成本最高的环节，前后经历两次修正：

- **初版建图不成立**：Haversine 硬阈值（≈55 米）在麦田级样本下使节点几乎全成孤点，GCN 退化为普通 MLP，空间信息基本未生效——这一缺陷被"83% 略胜 AE"的表面数字掩盖；
- **v2 用 KNN 重建图并真正修复了孤点问题**：图审计显示 K=5 时每个节点恰有 5 个邻居、0 个孤立节点，并采用 GraphSAGE 式归纳编码 + 验证节点隔离；
- **但建图仍依赖经纬度**，而数据存在 `O_Name` 历史重复记录，「经纬度 = 独立田块」的前提存疑。

结论：归纳式 GNN（回归 R²≈0.35）确优于普通 AE（R²≈0.26），空间先验有潜在价值；但受限于样本空间独立性的可靠性，**未作为最终采纳方法**。

---

## 进行中：改进方向

本项目是一个**仍在迭代的起点**。当前结论暴露了两条硬约束——**样本量** 与 **严重度回归的定义方式**：

**短期（可直接落地）**

- 回归问题重定义：DI 改**有序回归 / 多任务**，避免"用分类型特征拟合连续梯度"的天花板；
- 分类阈值优化：仅在训练集内确定阈值，观察 Recall / F1 / 校准；
- 完全嵌套特征选择：每个外层训练折内重新运行 Boruta。

**中期（需少量新数据 / 算力）**

- 自监督 / 对比预训练（ContrastBS、LMCN）利用无标签结构；
- GNN 升级：GAT + DropEdge、光谱相似度建边；
- 补充红边 / 红光指数。

**长期（高收益、有门槛）**

- 遥感基础模型微调（SatMAE / Prithvi）；
- 深度全特征回归（Deng et al. 像素级 DI 反演），把 R² 从 0.45 推向更高。

> 判断依据与领域调研见 `复盘报告_小麦赤霉病特征降维与模型构建.md`；决策推理与踩坑见 `docs/研究过程与方法论反思.md`。

---

## 目录结构

```
.
├── src/          无泄露建模代码（13 个脚本）
├── data/raw/     原始特征数据（GBK 编码 CSV）
├── results/      关键结果图表
├── docs/         分阶段实验记录 + 研究过程与方法论反思
├── requirements.txt
├── 复盘报告_小麦赤霉病特征降维与模型构建.md
├── arxiv精读_FHB_2409.00395.md
└── README.md
```

---

## 参考文献

- Lin et al., *Self-supervised FHB Detection with HSI & Feature Mining*, arXiv:2409.00395, 2024.
- Liu et al., *Monitoring Wheat FHB with UAV Hyperspectral*, Remote Sensing 12(22):3811, 2020.
- Bhatti et al., *Dimension Reduction Fuzzy Graph Network*, Comp. Elec. Agri. 226:109431, 2024.
- Deng et al., *Pixel-level regression for UAV hyperspectral wheat stripe rust DI*, Comp. Elec. Agri., 2023.

---

## 许可证

[MIT](LICENSE)
