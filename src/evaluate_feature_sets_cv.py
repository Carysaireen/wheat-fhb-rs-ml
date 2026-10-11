from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'; FS = ROOT / 'results' / 'feature_selection'; OUT = ROOT / 'results' / 'feature_set_cv'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / 'splits' / 'train.csv')
    folds = pd.read_csv(DATA / 'cv' / 'classification_folds.csv')
    all_features = df.columns[9:].tolist(); y = df['Unnamed: 2'].astype(int).to_numpy()
    stability = pd.read_csv(FS / 'feature_stability.csv')
    boruta = pd.read_csv(FS / 'boruta_stability.csv')
    sets = {
        'all_117': all_features,
        'current_2': [f for f in ['NREDI1','NDVIre2'] if f in all_features],
        'stable_rf_mi_3of5': stability.loc[stability.selection_rate >= .6, 'feature'].tolist(),
        'stable_rf_mi_4of5': stability.loc[stability.selection_rate >= .8, 'feature'].tolist(),
        'stable_rf_mi_5of5': stability.loc[stability.selection_rate >= 1.0, 'feature'].tolist(),
        'boruta_confirmed_3of5': boruta.loc[boruta.confirmed_rate >= .6, 'feature'].tolist(),
        'boruta_confirmed_4of5': boruta.loc[boruta.confirmed_rate >= .8, 'feature'].tolist(),
        'boruta_confirmed_5of5': boruta.loc[boruta.confirmed_rate >= 1.0, 'feature'].tolist(),
    }
    models = {
        'logistic_regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=2000, random_state=42))]),
        'svm_rbf': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', probability=True, random_state=42))]),
        'random_forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'),
    }
    rows=[]
    for set_name, features in sets.items():
        X = df[features].to_numpy(float)
        for model_name, base in models.items():
            for fold in sorted(folds.fold.unique()):
                valid = folds.fold.to_numpy() == fold
                model = clone(base); model.fit(X[~valid], y[~valid])
                pred = model.predict(X[valid])
                score = model.predict_proba(X[valid])[:,1] if hasattr(model,'predict_proba') else model.decision_function(X[valid])
                rows.append({'feature_set':set_name,'model':model_name,'fold':int(fold),'n_features':len(features),'accuracy':accuracy_score(y[valid],pred),'f1':f1_score(y[valid],pred,zero_division=0),'recall':recall_score(y[valid],pred,zero_division=0),'roc_auc':roc_auc_score(y[valid],score)})
    result=pd.DataFrame(rows); result.to_csv(OUT/'per_fold_metrics.csv',index=False,encoding='utf-8-sig')
    summary=result.groupby(['feature_set','model'],as_index=False).agg(n_features=('n_features','first'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),f1_mean=('f1','mean'),recall_mean=('recall','mean'),roc_auc_mean=('roc_auc','mean')).sort_values('roc_auc_mean',ascending=False)
    summary.to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
    plt.figure(figsize=(11,6));
    for model in summary.model.unique():
        s=summary[summary.model==model]; plt.plot(s.roc_auc_mean, s.feature_set, 'o-', label=model)
    plt.xlabel('Mean ROC-AUC across fixed folds'); plt.ylabel('Feature set'); plt.title('Feature-set comparison under identical CV folds'); plt.grid(axis='x',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'feature_set_comparison.png',dpi=180); plt.close()
    print(summary.head(10).to_string(index=False))

if __name__=='__main__': main()
