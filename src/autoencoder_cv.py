from pathlib import Path
import sys, random
ROOT=Path(__file__).resolve().parents[1]; _v=ROOT/'vendor_torch'; sys.path.insert(0,str(_v)) if _v.exists() else None
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score,r2_score,mean_squared_error,mean_absolute_error

DATA=ROOT/'data'; OUT=ROOT/'results'/'autoencoder_cv'; OUT.mkdir(parents=True,exist_ok=True)
class AE(nn.Module):
    def __init__(self,p):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(p,64),nn.ReLU(),nn.Linear(64,16),nn.ReLU(),nn.Linear(16,2)); self.dec=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,64),nn.ReLU(),nn.Linear(64,p))
    def forward(self,x): z=self.enc(x); return z,self.dec(z)
def fit_encode(a,b,seed):
    seed=int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); sc=StandardScaler().fit(a); ta=torch.tensor(sc.transform(a),dtype=torch.float32); tb=torch.tensor(sc.transform(b),dtype=torch.float32); m=AE(a.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=.005); lossf=nn.MSELoss(); hist=[]
    for _ in range(120): opt.zero_grad(); _,r=m(ta); loss=lossf(r,ta); loss.backward(); opt.step(); hist.append(float(loss.detach()))
    m.eval();
    with torch.no_grad(): za=m.enc(ta).numpy(); zb=m.enc(tb).numpy()
    return za,zb,hist
def main():
    d=pd.read_csv(DATA/'splits'/'train.csv'); fs=d.columns[9:].tolist(); cf=pd.read_csv(DATA/'cv'/'classification_folds.csv'); y=d['Unnamed: 2'].astype(int).to_numpy(); rows=[]; losses=[]
    for fold in sorted(cf.fold.unique()):
        v=cf.fold.to_numpy()==fold; za,zv,h=fit_encode(d.loc[~v,fs].to_numpy(float),d.loc[v,fs].to_numpy(float),42+fold); losses += [{'task':'classification','fold':fold,'epoch':i+1,'loss':x} for i,x in enumerate(h)]
        for name,m in [('LR',LogisticRegression(max_iter=2000,random_state=42)),('RF',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))]:
            m.fit(za,y[~v]); p=m.predict(zv); s=m.predict_proba(zv)[:,1]; rows.append({'task':'classification','representation':'autoencoder_2d','model':name,'fold':fold,'accuracy':accuracy_score(y[v],p),'f1':f1_score(y[v],p),'recall':recall_score(y[v],p),'roc_auc':roc_auc_score(y[v],s)})
    sick=d[d.DI>0].copy(); rf=pd.read_csv(DATA/'cv'/'regression_folds.csv'); yr=sick.DI.to_numpy(float)
    for fold in sorted(rf.fold.unique()):
        v=rf.fold.to_numpy()==fold; za,zv,h=fit_encode(sick.loc[~v,fs].to_numpy(float),sick.loc[v,fs].to_numpy(float),142+fold); losses += [{'task':'regression','fold':fold,'epoch':i+1,'loss':x} for i,x in enumerate(h)]; m=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1); m.fit(za,yr[~v]); p=m.predict(zv); rows.append({'task':'regression','representation':'autoencoder_2d','model':'RF','fold':fold,'r2':r2_score(yr[v],p),'rmse':mean_squared_error(yr[v],p)**.5,'mae':mean_absolute_error(yr[v],p)})
    res=pd.DataFrame(rows); res.to_csv(OUT/'per_fold_metrics.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(losses).to_csv(OUT/'training_losses.csv',index=False,encoding='utf-8-sig'); print(res.groupby(['task','model']).mean(numeric_only=True).to_string())
if __name__=='__main__': main()
