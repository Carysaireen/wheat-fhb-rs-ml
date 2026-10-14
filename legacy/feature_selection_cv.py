from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
_v = ROOT / 'vendor'; sys.path.insert(0, str(_v)) if _v.exists() else None
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy

DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'feature_selection'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / 'splits' / 'train.csv')
    folds = pd.read_csv(DATA / 'cv' / 'classification_folds.csv')
    names = train.columns[9:].tolist()
    X = train[names].to_numpy(float)
    y = train['Unnamed: 2'].astype(int).to_numpy()
    records = []
    importance_rows = []
    boruta_rows = []
    top_k = 20
    for fold in sorted(folds.fold.unique()):
        valid = folds.fold.to_numpy() == fold
        Xtr, ytr = X[~valid], y[~valid]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        rf = RandomForestClassifier(n_estimators=300, random_state=42 + int(fold), n_jobs=-1, class_weight='balanced')
        rf.fit(Xtr_s, ytr)
        boruta_rf = RandomForestClassifier(n_estimators=200, random_state=100 + int(fold), n_jobs=-1, class_weight='balanced', max_depth=6)
        boruta = BorutaPy(boruta_rf, n_estimators='auto', max_iter=30, perc=80, random_state=100 + int(fold), verbose=0)
        boruta.fit(Xtr_s, ytr)
        mi = mutual_info_classif(Xtr_s, ytr, random_state=42 + int(fold))
        rf_rank = np.argsort(rf.feature_importances_)[::-1]
        mi_rank = np.argsort(mi)[::-1]
        selected = sorted(set(rf_rank[:top_k]).intersection(mi_rank[:top_k]))
        if len(selected) < 5:
            selected = list(rf_rank[:top_k])
        for rank, idx in enumerate(rf_rank, 1):
            importance_rows.append({'fold': int(fold), 'feature': names[idx], 'rf_importance': rf.feature_importances_[idx], 'mi_score': mi[idx], 'rf_rank': rank, 'selected': names[idx] in [names[i] for i in selected]})
        for idx in selected:
            records.append({'fold': int(fold), 'feature': names[idx], 'selection_rule': 'RF_top20_intersect_MI_top20'})
        for idx, name in enumerate(names):
            boruta_rows.append({'fold': int(fold), 'feature': name, 'boruta_confirmed': bool(boruta.support_[idx]), 'boruta_tentative': bool(boruta.support_weak_[idx]), 'boruta_rank': int(boruta.ranking_[idx])})
    selections = pd.DataFrame(records)
    importance = pd.DataFrame(importance_rows)
    boruta_df = pd.DataFrame(boruta_rows)
    selections.to_csv(OUT / 'fold_selected_features.csv', index=False, encoding='utf-8-sig')
    importance.to_csv(OUT / 'fold_feature_importance.csv', index=False, encoding='utf-8-sig')
    boruta_df.to_csv(OUT / 'boruta_fold_results.csv', index=False, encoding='utf-8-sig')
    boruta_stability = boruta_df.groupby('feature').agg(confirmed_folds=('boruta_confirmed','sum'), tentative_folds=('boruta_tentative','sum')).reset_index()
    boruta_stability['confirmed_rate'] = boruta_stability.confirmed_folds / folds.fold.nunique()
    boruta_stability['tentative_or_confirmed_rate'] = (boruta_stability.confirmed_folds + boruta_stability.tentative_folds) / folds.fold.nunique()
    boruta_stability.sort_values(['confirmed_folds','tentative_folds','feature'], ascending=[False,False,True]).to_csv(OUT / 'boruta_stability.csv', index=False, encoding='utf-8-sig')
    stability = selections.groupby('feature').agg(folds_selected=('fold','nunique')).reset_index()
    stability['selection_rate'] = stability.folds_selected / folds.fold.nunique()
    stability = stability.sort_values(['folds_selected','feature'], ascending=[False,True])
    stability.to_csv(OUT / 'feature_stability.csv', index=False, encoding='utf-8-sig')
    summary = {'method': 'training-fold-only RF+mutual-information and Boruta', 'n_folds': int(folds.fold.nunique()), 'top_k_each_method': top_k, 'boruta_max_iter': 30, 'boruta_perc': 80, 'test_set_used': False, 'note': 'Exploratory selectors; final selector will be compared by nested CV.'}
    (OUT / 'selection_metadata.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.figure(figsize=(10, 6))
    top = stability.head(20).sort_values('selection_rate')
    plt.barh(top.feature, top.selection_rate, color='#2ca02c')
    plt.xlabel('Selection rate across 5 training folds')
    plt.ylabel('Feature')
    plt.title('Feature stability (RF + mutual information)')
    plt.xlim(0, 1.05); plt.grid(axis='x', alpha=.3); plt.tight_layout()
    plt.savefig(OUT / 'feature_stability.png', dpi=180); plt.close()
    counts = selections.groupby('fold').size()
    plt.figure(figsize=(7,4)); plt.bar(counts.index.astype(str), counts.values, color='#1f77b4')
    plt.xlabel('Validation fold'); plt.ylabel('Selected feature count'); plt.title('Selected features per fold'); plt.tight_layout()
    plt.savefig(OUT / 'selected_count_by_fold.png', dpi=180); plt.close()
    print(stability.head(10).to_string(index=False)); print('counts', counts.to_dict()); print('Boruta confirmed stability:'); print(boruta_stability.sort_values(['confirmed_folds','tentative_folds'], ascending=False).head(10).to_string(index=False))

if __name__ == '__main__': main()
