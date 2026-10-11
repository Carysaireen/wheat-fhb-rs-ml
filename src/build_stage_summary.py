from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; OUT=R/'stage_summary'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['Arial Unicode MS','Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
def main():
    base=pd.read_csv(R/'traditional_baseline'/'summary_metrics.csv'); tune=pd.read_csv(R/'nested_tuning'/'summary.csv'); ae=pd.read_csv(R/'autoencoder_cv'/'per_fold_metrics.csv'); gnn=pd.read_csv(R/'gnn_inductive_cv'/'per_fold_metrics.csv'); deep=pd.read_csv(R/'nested_deep_tuning'/'summary.csv')
    # classification overview
    b=base[base.task=='classification'].copy(); t=tune[tune.task=='classification'].copy()
    rows=[]
    for _,x in b.iterrows(): rows.append({'family':'traditional baseline','label':x.feature_set+' | '+x.model,'accuracy':x.metric_1_mean,'f1':x.f1_mean,'recall':x.recall_mean,'roc_auc':x.roc_auc_mean,'std':x.metric_1_std})
    for _,x in t.iterrows(): rows.append({'family':'nested tuned','label':'tuned '+x.model,'accuracy':x.accuracy_mean,'f1':x.f1_mean,'recall':x.recall_mean,'roc_auc':x.roc_auc_mean,'std':x.accuracy_std})
    ad=ae[ae.task=='classification'].groupby('model').mean(numeric_only=True).reset_index(); gd=gnn[gnn.task=='classification'].groupby('model').mean(numeric_only=True).reset_index()
    for _,x in ad.iterrows(): rows.append({'family':'autoencoder 2D','label':'AE 2D | '+x.model,'accuracy':x.accuracy,'f1':x.f1,'recall':x.recall,'roc_auc':x.roc_auc,'std':np.nan})
    for _,x in gd.iterrows(): rows.append({'family':'inductive GNN 2D','label':'GNN 2D | '+x.model,'accuracy':x.accuracy,'f1':x.f1,'recall':x.recall,'roc_auc':x.roc_auc,'std':np.nan})
    cls=pd.DataFrame(rows); cls.to_csv(OUT/'classification_master.csv',index=False,encoding='utf-8-sig')
    plt.figure(figsize=(12,7)); s=cls.sort_values('roc_auc'); plt.barh(s.label,s.roc_auc,color=['#4c78a8' if f=='traditional baseline' else '#f58518' if f=='nested tuned' else '#54a24b' if f=='inductive GNN 2D' else '#b279a2' for f in s.family]); plt.xlabel('Mean ROC-AUC (training CV)'); plt.title('Classification methods: discrimination comparison'); plt.xlim(0,1); plt.grid(axis='x',alpha=.25); plt.tight_layout(); plt.savefig(OUT/'classification_roc_auc.png',dpi=180); plt.close()
    plt.figure(figsize=(10,6)); x=np.arange(len(cls)); w=.2; 
    for i,m in enumerate(['accuracy','f1','recall','roc_auc']): plt.bar(x+i*w,cls[m],w,label=m)
    plt.xticks(x+1.5*w,cls.label,rotation=75,ha='right',fontsize=8); plt.ylabel('Mean score'); plt.title('Classification metric profile'); plt.ylim(0,1); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'classification_metric_profile.png',dpi=180); plt.close()
    # regression overview
    rows=[]
    for _,x in base[base.task=='regression'].iterrows(): rows.append({'family':'traditional baseline','label':x.feature_set+' | '+x.model,'r2':x.r2_mean,'rmse':x.rmse_mean,'mae':x.mae_mean,'std':x.r2_std})
    for _,x in tune[tune.task=='regression'].iterrows(): rows.append({'family':'nested tuned','label':'tuned '+x.model,'r2':x.r2_mean,'rmse':x.rmse_mean,'mae':x.mae_mean,'std':x.r2_std})
    ar=ae[ae.task=='regression'].groupby('model').mean(numeric_only=True).reset_index(); gr=gnn[gnn.task=='regression'].groupby('model').mean(numeric_only=True).reset_index()
    for _,x in ar.iterrows(): rows.append({'family':'autoencoder 2D','label':'AE 2D | '+x.model,'r2':x.r2,'rmse':x.rmse,'mae':x.mae,'std':np.nan})
    for _,x in gr.iterrows(): rows.append({'family':'inductive GNN 2D','label':'GNN 2D | '+x.model,'r2':x.r2,'rmse':x.rmse,'mae':x.mae,'std':np.nan})
    reg=pd.DataFrame(rows); reg.to_csv(OUT/'regression_master.csv',index=False,encoding='utf-8-sig')
    plt.figure(figsize=(12,7)); s=reg.sort_values('r2'); plt.barh(s.label,s.r2,color=['#4c78a8' if f=='traditional baseline' else '#f58518' if f=='nested tuned' else '#54a24b' if f=='inductive GNN 2D' else '#b279a2' for f in s.family]); plt.xlabel('Mean R² (training CV)'); plt.title('Regression methods: explained variance comparison'); plt.grid(axis='x',alpha=.25); plt.tight_layout(); plt.savefig(OUT/'regression_r2.png',dpi=180); plt.close()
    plt.figure(figsize=(9,5)); plt.scatter(reg.rmse,reg.r2,s=70); 
    for _,x in reg.iterrows(): plt.annotate(x.label,(x.rmse,x.r2),fontsize=7)
    plt.xlabel('RMSE (lower is better)'); plt.ylabel('R² (higher is better)'); plt.title('Regression error–fit trade-off'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'regression_tradeoff.png',dpi=180); plt.close()
    # fold variability of best traditional and deep
    nt=pd.read_csv(R/'nested_tuning'/'outer_fold_results.csv'); nd=pd.read_csv(R/'nested_deep_tuning'/'outer_fold_results.csv');
    plt.figure(figsize=(9,5));
    for label,d in [('tuned RF classification',nt[(nt.task=='classification')&(nt.model=='RF')]),('deep tuned classification',nd[nd.task=='classification'])]: plt.plot(d.outer_fold,d.roc_auc,'o-',label=label)
    plt.xlabel('Outer fold'); plt.ylabel('ROC-AUC'); plt.title('Fold-to-fold variability'); plt.xticks(range(5)); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'fold_variability_classification.png',dpi=180); plt.close()
    report=f'''# 阶段性研究总结（第1–11步）\n\n## 当前状态\n\n本报告只使用训练集内部交叉验证结果；封存测试集未读取。结果用于探索方向，不是最终独立测试结论。\n\n## 一、数据与方法事实\n\n- 总样本165；训练132；测试33（测试集封存）。\n- 候选特征117。\n- 分类采用固定5折StratifiedKFold；回归仅用DI>0训练样本，采用固定5折KFold。\n- 预处理在每个训练折拟合；没有发现缺失值、无穷值或常量特征。\n\n## 二、分类结果\n\n当前探索中，传统RF嵌套优化的平均ROC-AUC最高（约0.863）；黄金双特征LR的Recall约0.867，具有最强的简洁性和解释性；归纳式GNN二维特征优于普通AutoEncoder，但尚未超过传统RF。\n\n### 解释\n\n- Boruta多特征保留了纹理和光谱互补信息，可能提升排序能力。\n- 黄金双特征可能已捕捉主要“有病/无病”边界，因此简单LR仍有很高Recall。\n- 不同折的成绩波动较大，说明样本量限制了结论稳定性。\n\n### 延伸猜想\n\n1. 分类信号是否主要来自少数光谱指数，而纹理特征只在部分样本上提供补充？\n2. 如果优化分类阈值而不是固定0.5，黄金双特征LR的Recall能否进一步提高？\n3. 是否存在按采样时期、地块或病害水平分层的性能差异？\n\n## 三、回归结果\n\nBoruta特征+RF及其嵌套优化版本目前最好，R²约0.445–0.449；共同6特征次之；黄金双特征约0.343；二维AutoEncoder约0.261；二维GNN约0.351。\n\n### 解释\n\n- 严重度预测需要比分类更多的细节，过度压缩到2维会损失连续梯度信息。\n- 全部117特征不一定最好，冗余和噪声可能损害小样本回归。\n- 回归折间R²波动较大，说明当前结果对样本组成敏感。\n\n### 延伸猜想\n\n1. DI是否具有分段机制：先分类，再对患病样本分级回归？\n2. 纹理特征是否主要解释病害严重度，而光谱指数主要解释是否患病？\n3. DI测量误差或标签分辨率是否已经成为R²上限？\n\n## 四、方法判断\n\n目前最适合保留的三条路线：\n\n- 可解释应用路线：黄金双特征+LR（分类）\n- 性能优先路线：Boruta特征+RF/SVM（分类与回归）\n- 空间创新路线：归纳式GNN，继续探索但暂不宣称领先\n\n## 五、下一阶段建议\n\n1. 完全嵌套特征选择：每个外层训练折内重新运行Boruta。\n2. 分类阈值分析：只在训练集内确定阈值，重点观察Recall、F1和校准。\n3. 回归分段/两阶段模型：分类后对DI>0建模。\n4. 误差诊断：分析高误差样本的特征、DI区间和采样条件。\n5. GNN敏感性分析：K=3/5/8、距离图与特征相似图对比。\n\n## 六、当前结论的证据等级\n\n- 直接支持：Boruta+RF在当前训练CV中是最强回归候选。\n- 合理推测：光谱指数更适合分类，纹理特征可能补充严重度信息。\n- 尚待验证：空间图是否在新地块或新时期上真正提升泛化。\n'''
    (OUT/'stage_report.md').write_text(report,encoding='utf-8')
if __name__=='__main__': main()
