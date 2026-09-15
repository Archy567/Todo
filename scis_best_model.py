import json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import shap, matplotlib.pyplot as plt
from scis_experiment import prep_citylearn, macro_metrics, ENG_FEATURES, ROOT, EPS_MAPE, read_schema_assets, run_strategy

OUT=Path('best_results'); OUT.mkdir(exist_ok=True)
SEED=42

def main():
    df=prep_citylearn(); tr=df[df.split=='train']; va=df[df.split=='val']; te=df[df.split=='test'].copy()
    model=CatBoostRegressor(iterations=600,depth=8,learning_rate=.03,l2_leaf_reg=3,loss_function='RMSE',verbose=False,random_seed=SEED,thread_count=-1)
    t0=time.perf_counter(); model.fit(tr[ENG_FEATURES],tr.target); train_time=(time.perf_counter()-t0)/17
    te['pred_CatBoost']=model.predict(te[ENG_FEATURES])
    # align exactly with the common interval used in the seven-model comparison
    te=te.groupby('building',group_keys=False).apply(lambda g:g.sort_values('t').iloc[23:]).reset_index(drop=True)
    met=macro_metrics(te,'pred_CatBoost')
    val_rmse=float(mean_squared_error(va.target,model.predict(va[ENG_FEATURES]))**.5)

    # CatBoost-specific SHAP analysis
    sample=te.sample(min(2000,len(te)),random_state=SEED); X=sample[ENG_FEATURES]
    ex=shap.TreeExplainer(model); sv=np.asarray(ex.shap_values(X)); imp=np.mean(np.abs(sv),axis=0); order=np.argsort(imp)[::-1]
    ranking=[]
    for i in order:
        corr=float(np.corrcoef(X.iloc[:,i],sv[:,i])[0,1]) if np.std(X.iloc[:,i])>0 and np.std(sv[:,i])>0 else 0.0
        ranking.append({'feature':ENG_FEATURES[i],'mean_abs_shap':float(imp[i]),'corr_value_shap':corr})
    plt.figure(figsize=(7.2,4.8)); shap.summary_plot(sv,X,show=False,max_display=10); plt.tight_layout(); plt.savefig(OUT/'figure4_catboost_shap.png',dpi=400,bbox_inches='tight'); plt.close()

    # Figure 3 with selected CatBoost forecaster
    g=te[te.building==1].sort_values('t').iloc[:168]
    plt.figure(figsize=(7.2,3.8)); plt.plot(np.arange(len(g)),g.target,label='Actual',lw=1.4); plt.plot(np.arange(len(g)),g.pred_CatBoost,label='CatBoost prediction',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Non-shiftable load (kWh)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure3_catboost_forecast.png',dpi=400,bbox_inches='tight'); plt.close()

    # Forecast-informed optimization driven by selected CatBoost prediction.
    price=pd.read_csv(ROOT/'pricing.csv')['electricity_pricing'].values
    carbon=pd.read_csv(ROOT/'carbon_intensity.csv')['carbon_intensity'].values
    assets=read_schema_assets(); agg={k:{'energy':0,'peak':0,'cost':0,'co2':0,'time':[]} for k in ['baseline','rbc','optimizer']}; rep=None
    for b,gg in te.groupby('building'):
        gg=gg.sort_values('t'); ix=gg.t.astype(int).values; a=assets[int(b)]
        pv=gg.solar_profile.values*a['pv_power']/1000.; actual=gg.target.values; pred=gg.pred_CatBoost.values; pr=price[ix]; ci=carbon[ix]
        runs={m:run_strategy(actual,pred,pv,pr,ci,a,m) for m in ['baseline','rbc','optimizer']}
        for m,r in runs.items():
            agg[m]['energy']+=r['energy']; agg[m]['cost']+=r['cost']; agg[m]['co2']+=r['co2']; agg[m]['peak']=max(agg[m]['peak'],r['peak'])
            if r['time']>0: agg[m]['time'].append(r['time'])
        if int(b)==1: rep=pd.DataFrame({'baseline':runs['baseline']['grid'],'optimized':runs['optimizer']['grid']})
    for m in agg: agg[m]['time']=float(np.mean(agg[m]['time'])) if agg[m]['time'] else 0.0
    q=rep.iloc[:168]; plt.figure(figsize=(7.2,3.8)); plt.plot(np.arange(len(q)),q.baseline,label='No-control baseline',lw=1.4); plt.plot(np.arange(len(q)),q.optimized,label='CatBoost-informed optimizer',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Grid demand (kWh/h)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure5_catboost_optimization.png',dpi=400,bbox_inches='tight'); plt.close()

    # CatBoost-only external replication on the two BDG2 buildings.
    bdg={}; path=Path('bdg2/data/meters/cleaned/electricity_cleaned.csv')
    if path.exists():
        d=pd.read_csv(path,usecols=lambda c:c in ['timestamp','Moose_education_Ricardo','Cockatoo_education_Erik']); d.timestamp=pd.to_datetime(d.timestamp)
        for name in ['Moose_education_Ricardo','Cockatoo_education_Erik']:
            z=d[['timestamp',name]].dropna().rename(columns={name:'target'}).reset_index(drop=True); y=z.target.astype(float)
            z['hour_sin']=np.sin(2*np.pi*z.timestamp.dt.hour/24); z['hour_cos']=np.cos(2*np.pi*z.timestamp.dt.hour/24); z['dow_sin']=np.sin(2*np.pi*z.timestamp.dt.dayofweek/7); z['dow_cos']=np.cos(2*np.pi*z.timestamp.dt.dayofweek/7); z['weekend']=(z.timestamp.dt.dayofweek>=5).astype(int); z['month']=z.timestamp.dt.month
            z['lag_1']=y.shift(1); z['lag_24']=y.shift(24); z['lag_168']=y.shift(168); s=y.shift(1); z['roll_mean_24']=s.rolling(24).mean(); z['roll_max_24']=s.rolling(24).max(); z=z.dropna().reset_index(drop=True)
            f=['month','hour_sin','hour_cos','dow_sin','dow_cos','weekend','lag_1','lag_24','lag_168','roll_mean_24','roll_max_24']; n=len(z); a=int(.8*n); c=int(.9*n); tr2=z.iloc[:a]; te2=z.iloc[c:]
            m=CatBoostRegressor(iterations=600,depth=8,learning_rate=.03,l2_leaf_reg=3,loss_function='RMSE',verbose=False,random_seed=SEED,thread_count=-1); m.fit(tr2[f],tr2.target); p=m.predict(te2[f])
            bdg[name]={'n_raw_nonmissing':int(len(d[['timestamp',name]].dropna())),'n_after_lag':int(len(z)),'mae':float(mean_absolute_error(te2.target,p)),'rmse':float(mean_squared_error(te2.target,p)**.5),'mape':float(np.mean(np.abs(te2.target-p)/np.maximum(np.abs(te2.target),1.))*100),'r2':float(r2_score(te2.target,p))}
    out={'selected_model':'CatBoost','validation_rmse':val_rmse,'citylearn':{'MAE':met[0],'RMSE':met[1],'MAPE':met[2],'R2':met[3],'training_time_s_per_building':train_time},'shap_ranking':ranking,'optimization':agg,'bdg2_catboost':bdg}
    base=agg['baseline']; rbc=agg['rbc']; opt=agg['optimizer']
    out['reduction_vs_baseline_pct']={k:(base[k]-opt[k])/base[k]*100 for k in ['energy','peak','cost','co2']}
    out['reduction_vs_rbc_pct']={k:(rbc[k]-opt[k])/rbc[k]*100 for k in ['energy','peak','cost','co2']}
    with open(OUT/'best_model_results.json','w') as f: json.dump(out,f,indent=2)
    te[['building','t','target','pred_CatBoost']].to_csv(OUT/'catboost_test_predictions.csv',index=False)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
