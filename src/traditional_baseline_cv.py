from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; FS=ROOT/'results'/'feature_selection'; OUT=ROOT/'results'/'traditional_baseline'; OUT.mkdir(parents=True,exist_ok=True)

def feature_sets(df):
    names=df.columns[9:].tolist(); rf=pd.read_csv(FS/'feature_stability.csv'); bo=pd.read_csv(FS/'boruta_stability.csv')
    return {'all_117':names,'current_2':[f for f in ['NREDI1','NDVIre2'] if f in names],'common_6':['B8_HOM','B8_MEA','NDVIre2','NDVIre3','NREDI1','NREDI2'],'boruta_3of5':bo.loc[bo.confirmed_rate>=.6,'feature'].tolist()}

def main():
    df=pd.read_csv(DATA/'splits'/'train.csv'); sets=feature_sets(df); cls=pd.read_csv(DATA/'cv'/'classification_folds.csv'); y=df['Unnamed: 2'].astype(int).to_numpy(); cr=[]
    models={'LR':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=42))]),'SVM':Pipeline([('scaler',StandardScaler()),('model',SVC(kernel='rbf',probability=True,random_state=42))]),'RF':RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight='balanced')}
    for sn,fs in sets.items():
        X=df[fs].to_numpy(float)
        for mn,base in models.items():
            for fold in sorted(cls.fold.unique()):
                v=cls.fold.to_numpy()==fold; m=clone(base); m.fit(X[~v],y[~v]); pred=m.predict(X[v]); prob=m.predict_proba(X[v])[:,1] if hasattr(m,'predict_proba') else m.decision_function(X[v]); cr.append({'task':'classification','feature_set':sn,'model':mn,'fold':int(fold),'n_features':len(fs),'accuracy':accuracy_score(y[v],pred),'f1':f1_score(y[v],pred,zero_division=0),'recall':recall_score(y[v],pred,zero_division=0),'roc_auc':roc_auc_score(y[v],prob)})
    sick=df[df.DI>0].copy(); rg=pd.read_csv(DATA/'cv'/'regression_folds.csv'); rr=[]; yr=sick.DI.to_numpy(float)
    rmodels={'Ridge':Pipeline([('scaler',StandardScaler()),('model',Ridge(alpha=1.0))]),'SVR':Pipeline([('scaler',StandardScaler()),('model',SVR(C=1.0))]),'RF':RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)}
    for sn,fs in sets.items():
        X=sick[fs].to_numpy(float)
        for mn,base in rmodels.items():
            for fold in sorted(rg.fold.unique()):
                v=rg.fold.to_numpy()==fold; m=clone(base); m.fit(X[~v],yr[~v]); pred=m.predict(X[v]); rr.append({'task':'regression','feature_set':sn,'model':mn,'fold':int(fold),'n_features':len(fs),'r2':r2_score(yr[v],pred),'rmse':mean_squared_error(yr[v],pred)**.5,'mae':mean_absolute_error(yr[v],pred)})
    allr=pd.DataFrame(cr+rr); allr.to_csv(OUT/'per_fold_metrics.csv',index=False,encoding='utf-8-sig'); summary=allr.groupby(['task','feature_set','model'],as_index=False).agg(n_features=('n_features','first'),metric_1_mean=('accuracy','mean'),metric_1_std=('accuracy','std'),f1_mean=('f1','mean'),recall_mean=('recall','mean'),roc_auc_mean=('roc_auc','mean'),r2_mean=('r2','mean'),r2_std=('r2','std'),rmse_mean=('rmse','mean'),mae_mean=('mae','mean')); summary.to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
    for task,metric,title,path in [('classification','roc_auc_mean','Classification ROC-AUC','classification_baseline.png'),('regression','r2_mean','Regression R2','regression_baseline.png')]:
        s=summary[summary.task==task]; plt.figure(figsize=(10,6)); labels=s.feature_set+' | '+s.model; plt.barh(labels,s[metric]); plt.xlabel(metric); plt.title(title+' (training CV)'); plt.tight_layout(); plt.savefig(OUT/path,dpi=180); plt.close()
    print(summary.to_string(index=False))
if __name__=='__main__': main()
