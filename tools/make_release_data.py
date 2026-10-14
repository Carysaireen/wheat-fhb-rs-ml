"""生成对外发布版数据（去标识化 + 空间折分配表）。

原始数据含真实采集人姓名与高精度经纬度，不对外发布。本脚本从原始数据
生成可公开的版本，另外导出一份空间折分配表。

去标识化规则
------------
O_Name     ->  Rec_<原始行号+1>。行顺序保持不变，否则随机折无法逐位复现
O_Lat/Lng  ->  保留两位小数（约 1 km）
O_Flag     ->  Group_A..D。原值是采集人编码，与 O_Name 同属身份信息

为什么单独导出空间折分配表
--------------------------
空间折的做法是：先对经纬度做 KMeans(k=10) 得到空间块，再用
StratifiedGroupKFold(3) 把整块分到各折。坐标圆整到 1 km 之后，KMeans 在
seed=42 下约有 10% 的样本改变块归属（seed=7/2026 的划分保持不变），折分配
随之变化，因此只用公开数据无法逐位复现空间折结果。

把用高精度坐标算出的块号与折号作为结果文件一并发布，复现时直接读该文件，
既不需要公开原始坐标，也能得到与论文一致的数字。

用法
----
    python tools/make_release_data.py              # 默认从仓库上一级取原始数据
    python tools/make_release_data.py --source-root D:/path/to/project
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]
RAW_NAME = "Wheat_Scab_Features_Extraction_Ordered_region.csv"

# 采集人编码 -> 匿名组标签。顺序打乱，避免与原始编号形成单调对应
FLAG_MAP = {np.int64(52): "Group_C", np.int64(12): "Group_A",
            np.int64(64): "Group_B", np.int64(16): "Group_D"}
# 空间折的固定配置，与 spatial_group_cv.py 保持一致
SPATIAL_K = 10
SPATIAL_N_SPLITS = 3
SPATIAL_SEEDS = (42, 7, 2026)


def deidentify(df: pd.DataFrame, rec_ids: list[str]) -> pd.DataFrame:
    """去标识化：姓名换匿名号、坐标降精度、采集人编码换组标签。

    rec_ids 按 df 的行序给出，长度需与 df 一致。
    """
    assert len(rec_ids) == len(df), "匿名号数量与行数不符"
    out = df.copy()
    out["O_Name"] = list(rec_ids)
    out["O_Lat"] = out["O_Lat"].round(2)
    out["O_Lng"] = out["O_Lng"].round(2)
    out["O_Flag"] = out["O_Flag"].map(FLAG_MAP)
    return out


def _sgkf(blocks: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """按空间块整块留出，返回长度与 y 相同的折号数组。"""
    fold = np.full(len(y), -1, dtype=int)
    sp = StratifiedGroupKFold(n_splits=SPATIAL_N_SPLITS, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(sp.split(np.zeros((len(y), 1)), y, groups=blocks)):
        fold[va] = f
    assert (fold >= 0).all(), "存在未被分配到折的样本"
    return fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=REPO.parent,
                    help="项目根目录（含 data/raw 与 project_v2/data）")
    ap.add_argument("--out", type=Path, default=REPO,
                    help="发布版输出目录")
    args = ap.parse_args()

    src = args.source_root
    out = args.out
    (out / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (out / "data" / "splits").mkdir(parents=True, exist_ok=True)
    (out / "data" / "cv").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(src / "data" / "raw" / RAW_NAME, encoding="gbk")
    rec_of_row = {i: f"Rec_{i + 1:03d}" for i in raw.index}

    # 1) 原始数据 -> 发布版
    release_raw = deidentify(raw, [rec_of_row[i] for i in raw.index])
    release_raw.to_csv(out / "data" / "raw" / RAW_NAME, index=False, encoding="gbk")

    # 2) 训练/测试划分：沿用原始行序，随机折才可复现
    split_dir = src / "project_v2" / "data" / "splits"
    train_idx = pd.read_csv(split_dir / "train_indices.csv")["row_index"].to_numpy()
    test_idx = pd.read_csv(split_dir / "test_indices_locked.csv")["row_index"].to_numpy()

    for tag, idx in (("train", train_idx), ("test_locked", test_idx)):
        sub = raw.loc[idx].copy()
        sub.index = np.arange(len(sub))
        deidentify(sub, [rec_of_row[i] for i in idx]).to_csv(
            out / "data" / "splits" / f"{tag}.csv", index=False, encoding="utf-8-sig")

    # 3) 随机折：只依赖行序与标签，可直接搬运
    for fn in ("classification_folds.csv", "regression_folds.csv", "cv_metadata.json"):
        p = src / "project_v2" / "data" / "cv" / fn
        if p.exists():
            (out / "data" / "cv" / fn).write_bytes(p.read_bytes())

    # 4) 空间折分配表：用高精度坐标计算，只发布结果
    #
    # 项目里的空间折有三种形态，样本集与分层变量都不同，折号不可混用：
    #   fold_clf_s*     132 行 + 分类标签      —— 擂主档案、对齐基准、阶段⑤分类、阶段⑥
    #   fold_reg132_s*  132 行 + DI 三分位箱   —— 选择器动物园 full、阶段⑤回归
    #   fold_reg116_s*  116 行（DI>0）+ 分类标签 —— 动物园 stage4、P0-② 的回归轨
    train_hi = raw.loc[train_idx].copy()
    train_hi.index = np.arange(len(train_hi))
    y_cls = train_hi["Unnamed: 2"].astype(int).to_numpy()
    y_reg = train_hi["DI"].to_numpy(float)
    xy = train_hi[["O_Lat", "O_Lng"]].to_numpy(float)
    sick = y_reg > 0
    xy_sick = xy[sick]

    table = pd.DataFrame({"Rec_id": [rec_of_row[i] for i in train_idx],
                          "label": y_cls})
    for seed in SPATIAL_SEEDS:
        blk = KMeans(n_clusters=SPATIAL_K, n_init=10, random_state=seed).fit_predict(xy)
        blk_sick = KMeans(n_clusters=SPATIAL_K, n_init=10,
                          random_state=seed).fit_predict(xy_sick)
        bins = pd.qcut(y_reg, 3, labels=False, duplicates="drop")

        table[f"block132_s{seed}"] = blk
        table[f"fold_clf_s{seed}"] = _sgkf(blk, y_cls, seed)
        table[f"fold_reg132_s{seed}"] = _sgkf(blk, np.where(bins >= 0, bins, 0), seed)

        col_b = np.full(len(train_hi), np.nan)
        col_f = np.full(len(train_hi), np.nan)
        col_b[sick], col_f[sick] = blk_sick, _sgkf(blk_sick, y_cls[sick], seed)
        table[f"block116_s{seed}"] = col_b
        table[f"fold_reg116_s{seed}"] = col_f

        print(f"seed={seed:<5d} clf132 每折 {np.bincount(table[f'fold_clf_s{seed}']).tolist()}"
              f"  reg132 每折 {np.bincount(table[f'fold_reg132_s{seed}']).tolist()}"
              f"  reg116 每折 {np.bincount(col_f[sick].astype(int)).tolist()}")
    table.to_csv(out / "data" / "cv" / f"spatial_folds_k{SPATIAL_K}.csv", index=False)

    # 5) 自检：发布版与原始的特征列、标签列必须逐位一致
    check = pd.read_csv(out / "data" / "raw" / RAW_NAME, encoding="gbk")
    assert check.shape == raw.shape, "行列数改变"
    assert (check.iloc[:, 2].to_numpy() == raw.iloc[:, 2].to_numpy()).all(), "标签列被改动"
    assert (check.iloc[:, 3].to_numpy() == raw.iloc[:, 3].to_numpy()).all(), "DI 列被改动"
    assert (check.iloc[:, 9:].to_numpy() == raw.iloc[:, 9:].to_numpy()).all(), "特征列被改动"
    assert check["O_Name"].nunique() == 165 and check["O_Flag"].nunique() == 4
    assert abs(check["O_Lat"] - raw["O_Lat"]).max() < 0.0051, "坐标圆整越界"

    print(f"\n写入 {out}")
    print(f"  data/raw/{RAW_NAME}                {check.shape}")
    print(f"  data/splits/train.csv              {len(train_idx)} 行")
    print(f"  data/splits/test_locked.csv        {len(test_idx)} 行")
    print(f"  data/cv/spatial_folds_k{SPATIAL_K}.csv      {table.shape[0]} 行 x {table.shape[1]} 列")
    print("自检通过：标签与全部 117 维特征逐位一致，仅身份字段与坐标精度被处理")


if __name__ == "__main__":
    main()
