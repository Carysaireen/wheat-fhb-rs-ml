from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
# 原始数据已集中到项目根 data/raw/（全项目唯一副本），故从 project_v2 上溯一层
RAW = ROOT.parent / 'data' / 'raw' / 'Wheat_Scab_Features_Extraction_Ordered_region.csv'
OUT = ROOT / 'data' / 'splits'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, encoding='gbk')
    all_idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(all_idx, test_size=0.20, random_state=42, stratify=df['Unnamed: 2'])
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    train.to_csv(OUT / 'train.csv', index=False, encoding='utf-8-sig')
    test.to_csv(OUT / 'test_locked.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'row_index': train_idx, 'split': 'train'}).to_csv(OUT / 'train_indices.csv', index=False)
    pd.DataFrame({'row_index': test_idx, 'split': 'test_locked'}).to_csv(OUT / 'test_indices_locked.csv', index=False)
    meta = {'random_state': 42, 'test_size': 0.20, 'split_method': 'train_test_split(stratify=Unnamed: 2)', 'independence_assumption': 'each row is an independent unique object', 'n_total': len(df), 'n_train': len(train), 'n_test': len(test)}
    (OUT / 'split_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ['n_total','n_train','n_test','split_method']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
