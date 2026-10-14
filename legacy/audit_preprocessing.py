from pathlib import Path
import json
import numpy as np
import pandas as pd
from preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'preprocessing'


def audit(task, data, folds, row_indices):
    features = data.columns[9:].tolist()
    rows = []
    for variant in ['standard', 'iqr_clip']:
        for fold in sorted(folds.fold.unique()):
            valid_rows = set(folds.loc[folds.fold == fold, 'row_index'].astype(int))
            valid_mask = np.array([idx in valid_rows for idx in row_indices])
            X_train = data.loc[~valid_mask, features].to_numpy(float)
            X_valid = data.loc[valid_mask, features].to_numpy(float)
            pipe = build_preprocessor(clip_outliers=(variant == 'iqr_clip'))
            Xt = pipe.fit_transform(X_train)
            Xv = pipe.transform(X_valid)
            rows.append({
                'task': task, 'variant': variant, 'fold': int(fold),
                'n_train': len(X_train), 'n_valid': len(X_valid),
                'train_missing_after': int(np.isnan(Xt).sum()),
                'valid_missing_after': int(np.isnan(Xv).sum()),
                'train_scaled_mean_abs_max': float(np.abs(Xt.mean(axis=0)).max()),
                'train_scaled_std_error_max': float(np.abs(Xt.std(axis=0) - 1).max()),
                'valid_abs_max': float(np.abs(Xv).max()),
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / 'splits' / 'train.csv')
    cls_folds = pd.read_csv(DATA / 'cv' / 'classification_folds.csv')
    rows = audit('classification', train, cls_folds, train.index.to_numpy())

    sick = train[train['DI'] > 0].copy()
    reg_folds = pd.read_csv(DATA / 'cv' / 'regression_folds.csv')
    rows += audit('regression', sick, reg_folds, sick.index.to_numpy())
    report = pd.DataFrame(rows)
    report.to_csv(OUT / 'fold_preprocessing_audit.csv', index=False, encoding='utf-8-sig')

    X = train.iloc[:, 9:].apply(pd.to_numeric, errors='coerce')
    q1, q3 = X.quantile(.25), X.quantile(.75)
    mask = X.lt(q1 - 1.5 * (q3-q1)) | X.gt(q3 + 1.5 * (q3-q1))
    summary = {
        'n_training_rows': len(train), 'n_features': X.shape[1],
        'missing_cells': int(X.isna().sum().sum()),
        'infinite_cells': int(np.isinf(X.to_numpy()).sum()),
        'constant_features': int((X.nunique() <= 1).sum()),
        'potential_iqr_outlier_cells': int(mask.sum().sum()),
        'test_set_read': False,
        'policy': 'compare standard vs training-fold-only IQR clipping in later CV; never delete rows automatically'
    }
    (OUT / 'data_quality_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
