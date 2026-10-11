# 步骤 6：无泄露预处理流程

## 数据审计

训练集含 132 行、117 个候选特征。未发现缺失值、无穷值或常量特征；IQR 规则标记出 754 个潜在异常单元格。异常值可能包含真实病害信息，因此不自动删除样本。

## 两套候选方案

1. 标准方案：训练折中位数填补 → 标准化。
2. 稳健方案：训练折中位数填补 → IQR 截尾 → 标准化。

两套方案将在后续训练集内部交叉验证中公平比较。当前不依据测试集作决定。

## 防泄露规则

每一折都新建预处理器，只在该折训练部分执行 `fit`；验证部分只执行 `transform`。中位数、IQR 上下界、均值和标准差均不得由验证折或封存测试集贡献。

## 产出

- `src/preprocessing.py`：统一预处理组件。
- `src/audit_preprocessing.py`：逐折拟合与完整性审计。
- `results/preprocessing/fold_preprocessing_audit.csv`：分类与回归各折、两种方案的审计结果。
- `results/preprocessing/data_quality_summary.json`：训练数据质量摘要。
