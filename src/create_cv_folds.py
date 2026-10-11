from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / 'data' / 'splits'
OUT_DIR = ROOT / 'data' / 'cv'

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SPLIT_DIR / 'train.csv')
    y_cls = train['Unnamed: 2'].astype(int).to_numpy()

    # Classification: same fixed stratified folds for every classifier.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cls = pd.DataFrame({'row_index': train.index, 'fold': -1})
    for fold, (_, valid_idx) in enumerate(skf.split(train, y_cls)):
        cls.loc[valid_idx, 'fold'] = fold
    cls['label'] = y_cls
    cls.to_csv(OUT_DIR / 'classification_folds.csv', index=False)

    # Regression: rows are independent, so use ordinary fixed KFold.
    sick = train[train['DI'] > 0].copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg = pd.DataFrame({'row_index': sick.index, 'fold': -1})
    for fold, (_, valid_idx) in enumerate(kf.split(sick)):
        reg.loc[valid_idx, 'fold'] = fold
    reg['DI'] = sick['DI'].to_numpy()
    reg.to_csv(OUT_DIR / 'regression_folds.csv', index=False)

    meta = {
        'random_state': 42,
        'n_splits': 5,
        'classification': {'method': 'StratifiedKFold', 'shuffle': True, 'fold_file': 'classification_folds.csv'},
        'regression': {'method': 'KFold', 'shuffle': True, 'fold_file': 'regression_folds.csv', 'filtered_condition': 'DI > 0'},
        'all_candidates_must_reuse_these_assignments': True,
    }
    (OUT_DIR / 'cv_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'classification rows={len(cls)}, folds={sorted(cls.fold.unique())}')
    print(f'regression rows={len(reg)}, folds={sorted(reg.fold.unique())}')

if __name__ == '__main__':
    main()
