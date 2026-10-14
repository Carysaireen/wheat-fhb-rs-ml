# 步骤 5：训练集内部验证机制

- 分类：对 `train.csv` 使用固定的 5 折 `StratifiedKFold`（shuffle=True，random_state=42）。所有分类候选模型必须复用 `data/cv/classification_folds.csv`。
- 回归：只保留 `DI > 0` 的训练样本。经用户确认每行独立且不重复，使用固定的 5 折 `KFold`（shuffle=True，random_state=42）。所有回归候选模型必须复用 `data/cv/regression_folds.csv`。
- 每个样本的折号已保存；模型评估代码应使用这些折号构造预定义 split，而非重新生成折分。
- 测试集 `test_locked.csv` 未参与任何折分或验证。
