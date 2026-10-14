from pathlib import Path
import sys, random
ROOT=Path(__file__).resolve().parents[1]; _v=ROOT/'vendor_torch'; sys.path.insert(0,str(_v)) if _v.exists() else None
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score,r2_score,mean_squared_error,mean_absolute_error

DATA=ROOT/'data'; OUT=ROOT/'results'/'gnn_inductive_cv'; OUT.mkdir(parents=True,exist_ok=True); K=5
class SageAE(nn.Module):
    def __init__(self,p):
        super().__init__(); self.s1=nn.Linear(p,32); self.n1=nn.Linear(p,32); self.s2=nn.Linear(32,2); self.n2=nn.Linear(32,2); self.dec=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,p))
    def encode(self,x,a): h=torch.relu(self.s1(x)+self.n1(a@x)); return self.s2(h)+self.n2(a@h),h
    def forward(self,x,a): z,h=self.encode(x,a); return z,self.dec(z),h
def adjacency(coords,k=K):
    n=len(coords); kk=min(k+1,n); ind=NearestNeighbors(n_neighbors=kk).fit(coords).kneighbors(return_distance=False); a=np.zeros((n,n),np.float32)
    for i,js in enumerate(ind):
        js=[j for j in js if j!=i][:k]; a[i,js]=1/len(js) if js else 0
    return a
def fit_encode(xtr,xv,ctr,cv,seed):
    seed=int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); sc=StandardScaler().fit(xtr); xt=torch.tensor(sc.transform(xtr),dtype=torch.float32); xv_t=torch.tensor(sc.transform(xv),dtype=torch.float32); A=torch.tensor(adjacency(ctr),dtype=torch.float32); m=SageAE(xtr.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=.005,weight_decay=5e-4); lossf=nn.MSELoss(); hist=[]
    for _ in range(150): opt.zero_grad(); z,r,h=m(xt,A); loss=lossf(r,xt); loss.backward(); opt.step(); hist.append(float(loss.detach()))
    m.eval()
    with torch.no_grad():
        ztr,htr=m.encode(xt,A); nbr=NearestNeighbors(n_neighbors=min(K,len(ctr))).fit(ctr); ids=nbr.kneighbors(cv,return_distance=False); rawagg=torch.stack([xt[j].mean(0) for j in ids]); hv=torch.relu(m.s1(xv_t)+m.n1(rawagg)); hagg=torch.stack([htr[j].mean(0) for j in ids]); zv=m.s2(hv)+m.n2(hagg)
    return ztr.numpy(),zv.numpy(),hist,A.numpy()
def task_run(task,d,folds,seed0):
    fs=d.columns[9:].tolist(); y=(d['Unnamed: 2'].astype(int) if task=='classification' else d.DI).to_numpy(); coords=d[['O_Lat','O_Lng']].to_numpy(float); rows=[]; graphs=[]; losses=[]
    for fold in sorted(folds.fold.unique()):
        v=folds.fold.to_numpy()==fold; zt,zv,h,A=fit_encode(d.loc[~v,fs].to_numpy(float),d.loc[v,fs].to_numpy(float),coords[~v],coords[v],seed0+fold); graphs.append({'task':task,'fold':fold,'train_nodes':int((~v).sum()),'valid_nodes':int(v.sum()),'k':K,'train_edges':int((A>0).sum()),'mean_neighbors':float((A>0).sum(1).mean()),'isolated_nodes':int(((A>0).sum(1)==0).sum())}); losses += [{'task':task,'fold':fold,'epoch':i+1,'loss':x} for i,x in enumerate(h)]
        models=[('LR',LogisticRegression(max_iter=2000,random_state=42)),('RF',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))] if task=='classification' else [('RF',RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1))]
        for name,m in models:
            m.fit(zt,y[~v]); p=m.predict(zv); row={'task':task,'model':name,'fold':fold}
            if task=='classification': s=m.predict_proba(zv)[:,1]; row.update(accuracy=accuracy_score(y[v],p),f1=f1_score(y[v],p,zero_division=0),recall=recall_score(y[v],p,zero_division=0),roc_auc=roc_auc_score(y[v],s))
            else: row.update(r2=r2_score(y[v],p),rmse=mean_squared_error(y[v],p)**.5,mae=mean_absolute_error(y[v],p))
            rows.append(row)
    return rows,graphs,losses
def main():
    tr=pd.read_csv(DATA/'splits'/'train.csv'); rows,g,l=task_run('classification',tr,pd.read_csv(DATA/'cv'/'classification_folds.csv'),242); sick=tr[tr.DI>0].copy(); a,b,c=task_run('regression',sick,pd.read_csv(DATA/'cv'/'regression_folds.csv'),342); rows+=a;g+=b;l+=c
    r=pd.DataFrame(rows); r.to_csv(OUT/'per_fold_metrics.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(g).to_csv(OUT/'graph_audit.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(l).to_csv(OUT/'training_losses.csv',index=False,encoding='utf-8-sig'); print(r.groupby(['task','model']).mean(numeric_only=True).to_string()); print(pd.DataFrame(g).groupby('task').mean(numeric_only=True).to_string())
if __name__=='__main__': main()
