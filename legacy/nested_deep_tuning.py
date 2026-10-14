from pathlib import Path
import sys,time,json,random
ROOT=Path(__file__).resolve().parents[1]; _v=ROOT/'vendor_torch'; sys.path.insert(0,str(_v)) if _v.exists() else None
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,r2_score,mean_squared_error,mean_absolute_error

DATA=ROOT/'data'; OUT=ROOT/'results'/'nested_deep_tuning'; OUT.mkdir(parents=True,exist_ok=True)
class Net(nn.Module):
 def __init__(self,p,z,graph=False):
  super().__init__(); self.graph=graph; self.a=nn.Linear(p,32); self.b=nn.Linear(p,32) if graph else None; self.z=nn.Linear(32,z); self.nz=nn.Linear(32,z) if graph else None; self.dec=nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,p))
 def encode(self,x,A=None):
  h=torch.relu(self.a(x)+(self.b(A@x) if self.graph else 0)); return self.z(h)+(self.nz(A@h) if self.graph else 0),h
 def forward(self,x,A=None): z,h=self.encode(x,A); return z,self.dec(z),h
def adj(c,k):
 ids=NearestNeighbors(n_neighbors=min(k+1,len(c))).fit(c).kneighbors(return_distance=False); A=np.zeros((len(c),len(c)),np.float32)
 for i,js in enumerate(ids):
  js=[j for j in js if j!=i][:k]; A[i,js]=1/len(js) if js else 0
 return A
def encode(xa,xb,ca,cb,kind,zdim,k,epochs,seed):
 seed=int(seed); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);sc=StandardScaler().fit(xa);a=torch.tensor(sc.transform(xa),dtype=torch.float32);b=torch.tensor(sc.transform(xb),dtype=torch.float32); graph=kind=='gnn'; A=torch.tensor(adj(ca,k),dtype=torch.float32) if graph else None;m=Net(xa.shape[1],zdim,graph);o=torch.optim.Adam(m.parameters(),lr=.005,weight_decay=5e-4);lf=nn.MSELoss()
 for _ in range(epochs): o.zero_grad();_,r,_=m(a,A);loss=lf(r,a);loss.backward();o.step()
 m.eval()
 with torch.no_grad():
  za,ha=m.encode(a,A)
  if not graph: zb,_=m.encode(b)
  else:
   ids=NearestNeighbors(n_neighbors=min(k,len(ca))).fit(ca).kneighbors(cb,return_distance=False); raw=torch.stack([a[j].mean(0) for j in ids]);hb=torch.relu(m.a(b)+m.b(raw));agg=torch.stack([ha[j].mean(0) for j in ids]);zb=m.z(hb)+m.nz(agg)
 return za.numpy(),zb.numpy()
def score(task,zt,zv,yt,yv):
 if task=='classification': m=LogisticRegression(max_iter=2000,random_state=42);m.fit(zt,yt);return roc_auc_score(yv,m.predict_proba(zv)[:,1])
 m=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1);m.fit(zt,yt);return -mean_squared_error(yv,m.predict(zv))**.5
def run(task,d,folds):
 fs=d.columns[9:].tolist();X=d[fs].to_numpy(float);c=d[['O_Lat','O_Lng']].to_numpy(float);y=(d['Unnamed: 2'].astype(int) if task=='classification' else d.DI).to_numpy();rows=[]; configs=[('ae',z,0,e) for z in [2,4,8] for e in [80,150]]+[('gnn',z,k,120) for z in [2,4,8] for k in [3,5,8]]
 for outer in sorted(folds.fold.unique()):
  st=time.time();ov=folds.fold.to_numpy()==outer;Xi,yi,ci=X[~ov],y[~ov],c[~ov]; inner=StratifiedKFold(3,shuffle=True,random_state=500+int(outer)) if task=='classification' else KFold(3,shuffle=True,random_state=500+int(outer)); trials=[]
  for kind,z,k,e in configs:
   ss=[]
   for j,(a,b) in enumerate(inner.split(Xi,yi if task=='classification' else None)): za,zb=encode(Xi[a],Xi[b],ci[a],ci[b],kind,z,k,e,700+outer*100+j);ss.append(score(task,za,zb,yi[a],yi[b]))
   trials.append((np.mean(ss),kind,z,k,e))
  ins,kind,z,k,e=max(trials);zt,zv=encode(X[~ov],X[ov],c[~ov],c[ov],kind,z,k,e,900+outer); row={'task':task,'outer_fold':int(outer),'kind':kind,'encoding_dim':z,'k':k,'epochs':e,'inner_score':ins,'seconds':time.time()-st}
  if task=='classification':m=LogisticRegression(max_iter=2000,random_state=42);m.fit(zt,y[~ov]);p=m.predict(zv);s=m.predict_proba(zv)[:,1];row.update(accuracy=accuracy_score(y[ov],p),f1=f1_score(y[ov],p,zero_division=0),recall=recall_score(y[ov],p,zero_division=0),roc_auc=roc_auc_score(y[ov],s))
  else:m=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1);m.fit(zt,y[~ov]);p=m.predict(zv);row.update(r2=r2_score(y[ov],p),rmse=mean_squared_error(y[ov],p)**.5,mae=mean_absolute_error(y[ov],p))
  rows.append(row);print(row)
 return rows
def main():
 tr=pd.read_csv(DATA/'splits'/'train.csv');rows=run('classification',tr,pd.read_csv(DATA/'cv'/'classification_folds.csv'));s=tr[tr.DI>0].copy();rows+=run('regression',s,pd.read_csv(DATA/'cv'/'regression_folds.csv'));r=pd.DataFrame(rows);r.to_csv(OUT/'outer_fold_results.csv',index=False,encoding='utf-8-sig');sm=r.groupby('task',as_index=False).agg(accuracy_mean=('accuracy','mean'),f1_mean=('f1','mean'),recall_mean=('recall','mean'),roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),r2_mean=('r2','mean'),r2_std=('r2','std'),rmse_mean=('rmse','mean'),mae_mean=('mae','mean'),seconds=('seconds','sum'));sm.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig');print(sm.to_string(index=False))
if __name__=='__main__':main()
