from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from preprocessing import build_preprocessor

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; RES=ROOT/'results'; OUT=RES/'nested_tuning'; OUT.mkdir(parents=True,exist_ok=True)

def sets(df):
    bo=pd.read_csv(RES/'feature_selection'/'boruta_stability.csv')
    return {'golden_2':['NREDI1','NDVIre2'],'common_6':['B8_HOM','B8_MEA','NDVIre2','NDVIre3','NREDI1','NREDI2'],'boruta_3of5':bo.loc[bo.confirmed_rate>=.6,'feature'].tolist()}

def candidates(task):
    if task=='classification': return {
      'LR':(LogisticRegression(max_iter=3000,random_state=42),{'model__C':[.01,.1,1,10]}),
      'SVM':(SVC(probability=True,random_state=42),[{'model__kernel':['rbf'],'model__C':[.1,1,10],'model__gamma':[.01,.1,'scale']},{'model__kernel':['linear'],'model__C':[.1,1,10]}]),
      'RF':(RandomForestClassifier(random_state=42,n_jobs=-1,class_weight='balanced'),{'model__n_estimators':[200,400],'model__max_depth':[None,6],'model__min_samples_leaf':[1,3],'model__max_features':['sqrt',.5]})}
    return {
      'Ridge':(Ridge(),{'model__alpha':[.1,1,10,100]}),
      'SVR':(SVR(),{'model__C':[.1,1,10],'model__gamma':[.01,.1,'scale'],'model__epsilon':[.1,2]}),
      'RF':(RandomForestRegressor(random_state=42,n_jobs=-1),{'model__n_estimators':[200,400],'model__max_depth':[None,6],'model__min_samples_leaf':[1,3],'model__max_features':['sqrt',.5]})}

def run_task(task,df,folds,feature_sets):
    y=(df['Unnamed: 2'].astype(int) if task=='classification' else df.DI).to_numpy(); rows=[]
    for outer in sorted(folds.fold.unique()):
      start=time.time(); valid=folds.fold.to_numpy()==outer; best_by_model={}
      for model_name,(est,grid) in candidates(task).items():
        trials=[]
        for fs_name,fs in feature_sets.items():
          X=df[fs].to_numpy(float)
          for clip in [False,True]:
            pipe=Pipeline([('prep',build_preprocessor(clip)),('model',clone(est))])
            inner=StratifiedKFold(3,shuffle=True,random_state=100+int(outer)) if task=='classification' else KFold(3,shuffle=True,random_state=100+int(outer))
            search=GridSearchCV(pipe,grid,cv=inner,scoring='roc_auc' if task=='classification' else 'neg_root_mean_squared_error',n_jobs=-1)
            search.fit(X[~valid],y[~valid]); trials.append((search.best_score_,fs_name,fs,clip,search.best_estimator_,search.best_params_))
        score,fs_name,fs,clip,best,params=max(trials,key=lambda z:z[0]); X=df[fs].to_numpy(float); pred=best.predict(X[valid]); row={'task':task,'outer_fold':int(outer),'model':model_name,'feature_set':fs_name,'n_features':len(fs),'iqr_clip':clip,'inner_best_score':score,'best_params':json.dumps(params,ensure_ascii=False),'seconds':time.time()-start}
        if task=='classification':
          prob=best.predict_proba(X[valid])[:,1]; row.update(accuracy=accuracy_score(y[valid],pred),f1=f1_score(y[valid],pred,zero_division=0),recall=recall_score(y[valid],pred,zero_division=0),roc_auc=roc_auc_score(y[valid],prob))
        else: row.update(r2=r2_score(y[valid],pred),rmse=mean_squared_error(y[valid],pred)**.5,mae=mean_absolute_error(y[valid],pred))
        rows.append(row); print(task,outer,model_name,fs_name,'clip',clip)
    return rows

def main():
    train=pd.read_csv(DATA/'splits'/'train.csv'); fs=sets(train)
    rows=run_task('classification',train,pd.read_csv(DATA/'cv'/'classification_folds.csv'),fs)
    sick=train[train.DI>0].copy(); rows+=run_task('regression',sick,pd.read_csv(DATA/'cv'/'regression_folds.csv'),fs)
    out=pd.DataFrame(rows); out.to_csv(OUT/'outer_fold_results.csv',index=False,encoding='utf-8-sig')
    summary=out.groupby(['task','model'],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),f1_mean=('f1','mean'),recall_mean=('recall','mean'),roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),r2_mean=('r2','mean'),r2_std=('r2','std'),rmse_mean=('rmse','mean'),mae_mean=('mae','mean'),seconds_total=('seconds','sum')); summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig'); print(summary.to_string(index=False))
if __name__=='__main__': main()
