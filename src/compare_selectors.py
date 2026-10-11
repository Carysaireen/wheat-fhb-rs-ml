from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / 'results' / 'feature_selection'; OUT = ROOT / 'results' / 'feature_selection'

def main():
    rf = pd.read_csv(FS / 'feature_stability.csv')
    bo = pd.read_csv(FS / 'boruta_stability.csv')
    sets = {
        'RF+MI ≥3/5': set(rf.loc[rf.selection_rate >= .6, 'feature']),
        'RF+MI ≥4/5': set(rf.loc[rf.selection_rate >= .8, 'feature']),
        'RF+MI =5/5': set(rf.loc[rf.selection_rate >= 1.0, 'feature']),
        'Boruta ≥3/5': set(bo.loc[bo.confirmed_rate >= .6, 'feature']),
        'Boruta ≥4/5': set(bo.loc[bo.confirmed_rate >= .8, 'feature']),
        'Boruta =5/5': set(bo.loc[bo.confirmed_rate >= 1.0, 'feature']),
    }
    rows=[]
    for name,s in sets.items(): rows.append({'set':name,'n_features':len(s)})
    pd.DataFrame(rows).to_csv(OUT/'selector_set_sizes.csv',index=False,encoding='utf-8-sig')
    common = sorted(sets['RF+MI ≥3/5'] & sets['Boruta ≥3/5'])
    only_rf = sorted(sets['RF+MI ≥3/5'] - sets['Boruta ≥3/5'])
    only_bo = sorted(sets['Boruta ≥3/5'] - sets['RF+MI ≥3/5'])
    pd.DataFrame({'category':['common_3of5','only_rf_mi_3of5','only_boruta_3of5'],'features':[', '.join(common),', '.join(only_rf),', '.join(only_bo)]}).to_csv(OUT/'selector_overlap.csv',index=False,encoding='utf-8-sig')
    overlap_counts = [len(common), len(only_rf), len(only_bo)]
    plt.figure(figsize=(7,5)); bars=plt.bar(['共同','仅 RF+MI','仅 Boruta'], overlap_counts, color=['#2ca02c','#1f77b4','#d62728']); plt.ylabel('特征数量'); plt.title('3/5 稳定性阈值下的选择器重叠'); plt.grid(axis='y',alpha=.3); [plt.text(b.get_x()+b.get_width()/2,b.get_height()+.2,str(int(b.get_height())),ha='center') for b in bars]; plt.tight_layout(); plt.savefig(OUT/'selector_overlap_counts.png',dpi=180); plt.close()
    plt.figure(figsize=(9,5)); names=[r['set'] for r in rows]; vals=[r['n_features'] for r in rows]; bars=plt.bar(names,vals,color=['#1f77b4']*3+['#d62728']*3); plt.ylabel('Number of selected features'); plt.title('Selected feature count by method and stability threshold'); plt.xticks(rotation=25,ha='right'); plt.grid(axis='y',alpha=.3); plt.tight_layout(); plt.savefig(OUT/'selector_set_sizes.png',dpi=180); plt.close()
    print('common',len(common),common); print('only_rf',len(only_rf),only_rf); print('only_boruta',len(only_bo),only_bo)

if __name__=='__main__': main()
