import json,time,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import shap,matplotlib.pyplot as plt
from scis_experiment import prep_citylearn,macro_metrics,ENG_FEATURES,ROOT,read_schema_assets,run_strategy
OUT=Path('catboost_fulltest_results'); OUT.mkdir(exist_ok=True)
SEED=42

def main():
    df=prep_citylearn(); tr=df[df.split=='train']; va=df[df.split=='val']; te=df[df.split=='test'].copy()
    model=CatBoostRegressor(iterations=600,depth=8,learning_rate=.03,l2_leaf_reg=3,loss_function='RMSE',verbose=False,random_seed=SEED,thread_count=-1)
    t0=time.perf_counter(); model.fit(tr[ENG_FEATURES].values,tr.target.values); train_time=(time.perf_counter()-t0)/17
    te['pred_CatBoost']=model.predict(te[ENG_FEATURES].values)
    met=macro_metrics(te,'pred_CatBoost')
    val_rmse=float(mean_squared_error(va.target,model.predict(va[ENG_FEATURES].values))**.5)
    sample=te.sample(min(2000,len(te)),random_state=SEED); X=sample[ENG_FEATURES]
    ex=shap.TreeExplainer(model); sv=np.asarray(ex.shap_values(X)); imp=np.mean(np.abs(sv),axis=0); order=np.argsort(imp)[::-1]
    ranking=[]
    for i in order:
        corr=float(np.corrcoef(X.iloc[:,i],sv[:,i])[0,1]) if np.std(X.iloc[:,i])>0 and np.std(sv[:,i])>0 else 0.0
        ranking.append({'feature':ENG_FEATURES[i],'mean_abs_shap':float(imp[i]),'corr_value_shap':corr})
    g=te[te.building==1].sort_values('t').iloc[:168]
    plt.figure(figsize=(7.2,3.8)); plt.plot(np.arange(len(g)),g.target,label='Actual',lw=1.4); plt.plot(np.arange(len(g)),g.pred_CatBoost,label='CatBoost prediction',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Non-shiftable load (kWh)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure3_catboost_fulltest.png',dpi=400,bbox_inches='tight'); plt.close()
    plt.figure(figsize=(7.2,4.8)); shap.summary_plot(sv,X,show=False,max_display=10); plt.tight_layout(); plt.savefig(OUT/'figure4_catboost_fulltest.png',dpi=400,bbox_inches='tight'); plt.close()
    price=pd.read_csv(ROOT/'pricing.csv')['electricity_pricing'].values; carbon=pd.read_csv(ROOT/'carbon_intensity.csv')['carbon_intensity'].values; assets=read_schema_assets()
    agg={k:{'energy':0,'peak':0,'cost':0,'co2':0,'time':[]} for k in ['baseline','rbc','optimizer']}; rep=None
    for b,gg in te.groupby('building'):
        gg=gg.sort_values('t'); ix=gg.t.astype(int).values; a=assets[int(b)]; pv=gg.solar_profile.values*a['pv_power']/1000.; actual=gg.target.values; pred=gg.pred_CatBoost.values; pr=price[ix]; ci=carbon[ix]
        runs={m:run_strategy(actual,pred,pv,pr,ci,a,m) for m in ['baseline','rbc','optimizer']}
        for m,r in runs.items():
            agg[m]['energy']+=r['energy']; agg[m]['cost']+=r['cost']; agg[m]['co2']+=r['co2']; agg[m]['peak']=max(agg[m]['peak'],r['peak'])
            if r['time']>0: agg[m]['time'].append(r['time'])
        if int(b)==1: rep=pd.DataFrame({'baseline':runs['baseline']['grid'],'optimized':runs['optimizer']['grid']})
    for m in agg: agg[m]['time']=float(np.mean(agg[m]['time'])) if agg[m]['time'] else 0.0
    q=rep.iloc[:168]; plt.figure(figsize=(7.2,3.8)); plt.plot(np.arange(len(q)),q.baseline,label='No-control baseline',lw=1.4); plt.plot(np.arange(len(q)),q.optimized,label='CatBoost-informed optimizer',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Grid demand (kWh/h)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure5_catboost_fulltest.png',dpi=400,bbox_inches='tight'); plt.close()
    base=agg['baseline']; rbc=agg['rbc']; opt=agg['optimizer']
    out={'selected_model':'CatBoost','n_test_rows':int(len(te)),'validation_rmse':val_rmse,'citylearn':{'MAE':met[0],'RMSE':met[1],'MAPE':met[2],'R2':met[3],'training_time_s_per_building':train_time},'shap_ranking':ranking,'optimization':agg,'reduction_vs_baseline_pct':{k:(base[k]-opt[k])/base[k]*100 for k in ['energy','peak','cost','co2']},'reduction_vs_rbc_pct':{k:(rbc[k]-opt[k])/rbc[k]*100 for k in ['energy','peak','cost','co2']}}
    with open(OUT/'catboost_fulltest_results.json','w') as f: json.dump(out,f,indent=2)
    te[['building','t','target','pred_CatBoost']].to_csv(OUT/'catboost_fulltest_predictions.csv',index=False)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
