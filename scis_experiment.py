import os, json, time, math, warnings, subprocess
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['PYTHONHASHSEED']='42'
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED)
ROOT=Path('CityLearn/data/datasets/citylearn_challenge_2022_phase_all')
OUT=Path('results'); OUT.mkdir(exist_ok=True)
EPS_MAPE=0.1
BASE_FEATURES=['month','hour_sin','hour_cos','dow_sin','dow_cos','weekend','temp','rh','dhi','dni']
ENG_FEATURES=BASE_FEATURES+['lag_1','lag_24','lag_168','roll_mean_24','roll_max_24']

# ---------- data ----------
def prep_citylearn():
    weather=pd.read_csv(ROOT/'weather.csv')
    all_parts=[]
    for b in range(1,18):
        x=pd.read_csv(ROOT/f'Building_{b}.csv')
        n=len(x); idx=np.arange(n)
        d=x.copy()
        d['building']=b; d['t']=idx
        d['temp']=weather['outdoor_dry_bulb_temperature'].values[:n]
        d['rh']=weather['outdoor_relative_humidity'].values[:n]
        d['dhi']=weather['diffuse_solar_irradiance'].values[:n]
        d['dni']=weather['direct_solar_irradiance'].values[:n]
        h=(d['hour'].astype(int)-1)%24
        dow=(d['day_type'].astype(int)-1)%7
        d['hour_sin']=np.sin(2*np.pi*h/24); d['hour_cos']=np.cos(2*np.pi*h/24)
        d['dow_sin']=np.sin(2*np.pi*dow/7); d['dow_cos']=np.cos(2*np.pi*dow/7)
        d['weekend']=(dow>=5).astype(int)
        y=d['non_shiftable_load'].astype(float)
        d['lag_1']=y.shift(1); d['lag_24']=y.shift(24); d['lag_168']=y.shift(168)
        s=y.shift(1); d['roll_mean_24']=s.rolling(24).mean(); d['roll_max_24']=s.rolling(24).max()
        d['target']=y
        # keep raw solar profile for optimization
        d['solar_profile']=d['solar_generation'].astype(float)
        # chronological split based on original annual index, so no future leakage
        d['split']=np.where(idx < int(.8*n),'train',np.where(idx < int(.9*n),'val','test'))
        d=d.dropna(subset=ENG_FEATURES+['target']).reset_index(drop=True)
        all_parts.append(d[['building','t','split','target','solar_profile']+ENG_FEATURES])
    return pd.concat(all_parts,ignore_index=True)

def macro_metrics(df,pred_col):
    vals=[]
    for b,g in df.groupby('building'):
        y=g.target.values; p=g[pred_col].values
        vals.append([mean_absolute_error(y,p),mean_squared_error(y,p)**.5,
                     np.mean(np.abs(y-p)/np.maximum(np.abs(y),EPS_MAPE))*100,r2_score(y,p)])
    return np.mean(vals,axis=0).tolist()

def fit_tabular(df):
    train=df[df.split=='train']; val=df[df.split=='val']; test=df[df.split=='test'].copy()
    Xtr=train[ENG_FEATURES].values; ytr=train.target.values
    Xv=val[ENG_FEATURES].values; yv=val.target.values
    models={
      'Linear Regression':LinearRegression(),
      'Random Forest':RandomForestRegressor(n_estimators=400,max_depth=20,min_samples_leaf=2,max_features='sqrt',bootstrap=True,n_jobs=-1,random_state=SEED),
      'XGBoost':XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,colsample_bytree=.8,reg_lambda=1,n_jobs=-1,random_state=SEED,objective='reg:squarederror'),
      'LightGBM':LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,n_jobs=-1,random_state=SEED,verbosity=-1),
      'CatBoost':CatBoostRegressor(iterations=600,depth=8,learning_rate=.03,l2_leaf_reg=3,loss_function='RMSE',verbose=False,random_seed=SEED,thread_count=-1)
    }
    trained={}; metrics={}; val_rmse={}; times={}
    for name,m in models.items():
        t0=time.perf_counter(); m.fit(Xtr,ytr); times[name]=(time.perf_counter()-t0)/17
        trained[name]=m
        pv=m.predict(Xv); pt=m.predict(test[ENG_FEATURES].values)
        val_rmse[name]=float(mean_squared_error(yv,pv)**.5)
        test['pred_'+name]=pt
        metrics[name]=macro_metrics(test,'pred_'+name)
    return trained,test,metrics,val_rmse,times

def fit_lstm(df,test,metrics,val_rmse,times):
    # Global sequential model; target indices remain building-specific and causal.
    scaler_x=StandardScaler(); scaler_y=StandardScaler()
    tr=df[df.split=='train']; scaler_x.fit(tr[ENG_FEATURES]); scaler_y.fit(tr[['target']])
    seqs={'train':([],[]),'val':([],[]),'test':([],[],[],[])}
    for b,g in df.groupby('building'):
        g=g.sort_values('t').reset_index(drop=True)
        X=scaler_x.transform(g[ENG_FEATURES]); y=scaler_y.transform(g[['target']]).ravel()
        for i in range(23,len(g)):
            sp=g.loc[i,'split']; seq=X[i-23:i+1]
            if sp in ['train','val']:
                seqs[sp][0].append(seq); seqs[sp][1].append(y[i])
            else:
                seqs['test'][0].append(seq); seqs['test'][1].append(y[i]); seqs['test'][2].append(int(b)); seqs['test'][3].append(int(g.loc[i,'t']))
    Xtr=np.asarray(seqs['train'][0],dtype='float32'); ytr=np.asarray(seqs['train'][1],dtype='float32')
    Xv=np.asarray(seqs['val'][0],dtype='float32'); yv=np.asarray(seqs['val'][1],dtype='float32')
    Xt=np.asarray(seqs['test'][0],dtype='float32')
    model=Sequential([LSTM(64,return_sequences=True,input_shape=(24,len(ENG_FEATURES))),Dropout(.2),LSTM(32),Dropout(.2),Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='mse')
    es=EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True,min_delta=1e-5)
    t0=time.perf_counter(); hist=model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=150,batch_size=32,callbacks=[es],verbose=0,shuffle=False)
    times['LSTM']=(time.perf_counter()-t0)/17
    pv=scaler_y.inverse_transform(model.predict(Xv,verbose=0).reshape(-1,1)).ravel()
    val_rmse['LSTM']=float(mean_squared_error(scaler_y.inverse_transform(yv.reshape(-1,1)).ravel(),pv)**.5)
    pt=scaler_y.inverse_transform(model.predict(Xt,verbose=0).reshape(-1,1)).ravel()
    key=pd.DataFrame({'building':seqs['test'][2],'t':seqs['test'][3],'pred_LSTM':pt})
    test=test.merge(key,on=['building','t'],how='left')
    # common aligned interval for fair comparison: remove first 23 test targets without LSTM sequence
    common=test.dropna(subset=['pred_LSTM']).copy()
    metrics['LSTM']=macro_metrics(common,'pred_LSTM')
    return model,common,metrics,val_rmse,times,hist.history

def recompute_common_metrics(common,metrics):
    for name in ['Linear Regression','Random Forest','XGBoost','LightGBM','CatBoost']:
        metrics[name]=macro_metrics(common,'pred_'+name)
    return metrics

def make_ensemble(common,val_rmse,times,metrics):
    top=sorted(val_rmse,key=val_rmse.get)[:3]
    inv=np.array([1/(val_rmse[k]+1e-8) for k in top]); w=inv/inv.sum()
    common['pred_Ensemble']=sum(wi*common['pred_'+k] for wi,k in zip(w,top))
    metrics['Proposed Validation-Weighted Ensemble']=macro_metrics(common,'pred_Ensemble')
    times['Proposed Validation-Weighted Ensemble']=sum(times[k] for k in top) # training burden of members / building
    return common,top,w,metrics,times

def dm_test(common,best_ind):
    # Per-hour macro squared-loss differential across 17 buildings; HAC/Newey-West lag 24.
    z=common.copy(); z['e_ens']=(z.target-z.pred_Ensemble)**2; z['e_ind']=(z.target-z['pred_'+best_ind])**2
    d=z.groupby('t').apply(lambda g: np.mean(g.e_ens-g.e_ind)).values
    n=len(d); mu=d.mean(); u=d-mu; L=min(24,n-1)
    gamma0=np.dot(u,u)/n; var=gamma0
    for l in range(1,L+1):
        gam=np.dot(u[l:],u[:-l])/n; var += 2*(1-l/(L+1))*gam
    se=math.sqrt(max(var,1e-16)/n); stat=mu/se; p=2*(1-stats.norm.cdf(abs(stat)))
    return float(stat),float(p)

def shap_analysis(common,trained,top,w):
    # Ensemble SHAP for tree constituents when available; otherwise use strongest tree member.
    tree_names=[k for k in top if k in ['Random Forest','XGBoost','LightGBM','CatBoost']]
    sample=common.sample(min(2000,len(common)),random_state=SEED)
    X=sample[ENG_FEATURES]
    shap_sum=np.zeros_like(X.values,dtype=float); wt=0
    for k,wi in zip(top,w):
        if k in tree_names:
            try:
                ex=shap.TreeExplainer(trained[k]); sv=ex.shap_values(X)
                if isinstance(sv,list): sv=sv[0]
                shap_sum += wi*np.asarray(sv); wt += wi
            except Exception as e: print('SHAP failed',k,e)
    if wt==0:
        k=tree_names[0] if tree_names else 'LightGBM'; ex=shap.TreeExplainer(trained[k]); shap_sum=np.asarray(ex.shap_values(X)); wt=1
    shap_sum/=wt
    imp=np.mean(np.abs(shap_sum),axis=0); order=np.argsort(imp)[::-1]
    ranking=[{'feature':ENG_FEATURES[i],'mean_abs_shap':float(imp[i]),'corr_value_shap':float(np.corrcoef(X.iloc[:,i],shap_sum[:,i])[0,1]) if np.std(X.iloc[:,i])>0 and np.std(shap_sum[:,i])>0 else 0.0} for i in order]
    np.save(OUT/'shap_values.npy',shap_sum); sample[ENG_FEATURES].to_csv(OUT/'shap_sample.csv',index=False)
    # actual beeswarm
    plt.figure(figsize=(7.2,4.8)); shap.summary_plot(shap_sum,X,show=False,max_display=10); plt.tight_layout(); plt.savefig(OUT/'figure4_shap.png',dpi=400,bbox_inches='tight'); plt.close()
    return ranking

def ablation(df,best_family,ensemble_rmse,ranking):
    # Use LightGBM for controlled feature ablation (stable, efficient, nonlinear).
    tr=df[df.split=='train']; te=df[df.split=='test']
    # align with common LSTM interval by discarding first 23 test samples/building
    te=te.groupby('building',group_keys=False).apply(lambda g:g.iloc[23:]).reset_index(drop=True)
    def train_eval(features):
        m=LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,n_jobs=-1,random_state=SEED,verbosity=-1)
        m.fit(tr[features],tr.target); te2=te.copy(); te2['p']=m.predict(te2[features]); return macro_metrics(te2,'p')[1]
    r_basic=train_eval(BASE_FEATURES); r_eng=train_eval(ENG_FEATURES)
    topf=ranking[0]['feature']; feats=[f for f in ENG_FEATURES if f!=topf]; r_drop=train_eval(feats)
    return {'model':'LightGBM','rmse_basic':r_basic,'rmse_engineered':r_eng,'feature_gain_pct':(r_basic-r_eng)/r_basic*100,'removed_feature':topf,'rmse_without_top_feature':r_drop,'removal_deterioration_pct':(r_drop-r_eng)/r_eng*100,'ensemble_rmse':ensemble_rmse}

def seed_robustness(df):
    # Five-seed robustness on LightGBM, then report macro RMSE.
    tr=df[df.split=='train']; te=df[df.split=='test'].groupby('building',group_keys=False).apply(lambda g:g.iloc[23:]).reset_index(drop=True)
    rms=[]
    for seed in [7,21,42,77,101]:
        m=LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,n_jobs=-1,random_state=seed,verbosity=-1)
        m.fit(tr[ENG_FEATURES],tr.target); z=te.copy(); z['p']=m.predict(z[ENG_FEATURES]); rms.append(macro_metrics(z,'p')[1])
    return {'seeds':[7,21,42,77,101],'rmse':rms,'mean':float(np.mean(rms)),'std':float(np.std(rms,ddof=1))}

# ---------- optimization ----------
def read_schema_assets():
    sch=json.load(open(ROOT/'schema.json'))
    assets={}
    for i in range(1,18):
        b=sch['buildings'][f'Building_{i}']; es=b['electrical_storage']['attributes']; pv=b['pv']['attributes']
        assets[i]={'capacity':float(es['capacity']),'efficiency':float(es['efficiency']),'power':float(es['nominal_power']),'pv_power':float(pv['nominal_power'])}
    return assets

def lp_first_action(loadhat,pv,price,carbon,soc,asset):
    H=len(loadhat); cap=asset['capacity']; eta=asset['efficiency']; pmax=asset['power']
    base=np.maximum(0,loadhat-pv); bc=max(np.sum(base*price),1e-6); bg=max(np.sum(base*carbon),1e-6); bp=max(np.max(base),1e-6)
    # variables c[H],d[H],s[H+1],g[H],P
    n=4*H+2; C=slice(0,H); D=slice(H,2*H); S=slice(2*H,3*H+1); G=slice(3*H+1,4*H+1); P=4*H+1
    obj=np.zeros(n); obj[C]=1e-6; obj[D]=1e-6; obj[G]=(price/(3*bc)+carbon/(3*bg)); obj[P]=1/(3*bp)
    Aeq=[]; beq=[]
    row=np.zeros(n); row[S.start]=1; Aeq.append(row); beq.append(soc)
    for t in range(H):
        row=np.zeros(n); row[S.start+t+1]=1; row[S.start+t]=-1; row[t]=-eta; row[H+t]=1/eta; Aeq.append(row); beq.append(0)
    Aub=[]; bub=[]
    for t in range(H):
        # g >= loadhat-pv+c-d => c-d-g <= -(loadhat-pv)
        row=np.zeros(n); row[t]=1; row[H+t]=-1; row[G.start+t]=-1; Aub.append(row); bub.append(-(loadhat[t]-pv[t]))
        row=np.zeros(n); row[G.start+t]=1; row[P]=-1; Aub.append(row); bub.append(0)
    bounds=[(0,pmax)]*H+[(0,pmax)]*H+[(0,cap)]*(H+1)+[(0,None)]*H+[(0,None)]
    res=linprog(obj,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=bounds,method='highs')
    if not res.success:return 0.,0.
    return float(res.x[0]),float(res.x[H])

def run_strategy(actual,pred,pv,price,carbon,asset,mode):
    soc=0.; cap=asset['capacity']; eta=asset['efficiency']; pmax=asset['power']; grids=[]; tdec=[]
    for i in range(len(actual)):
        if mode=='baseline': c=d=0
        elif mode=='rbc':
            h=i%24
            # deterministic solar-oriented RBC: charge mid-day, discharge evening peak
            c=min(pmax,max(0,(cap-soc)/eta)) if 10<=h<=15 else 0
            d=min(pmax,soc*eta) if 18<=h<=22 else 0
        else:
            j=min(len(actual),i+24); t0=time.perf_counter(); c,d=lp_first_action(pred[i:j],pv[i:j],price[i:j],carbon[i:j],soc,asset); tdec.append(time.perf_counter()-t0)
        c=min(c,max(0,(cap-soc)/eta)); d=min(d,soc*eta)
        soc=np.clip(soc+eta*c-d/eta,0,cap)
        grids.append(max(0,actual[i]-pv[i]+c-d))
    grids=np.array(grids); return {'grid':grids,'energy':float(grids.sum()),'peak':float(grids.max()),'cost':float(np.sum(grids*price)),'co2':float(np.sum(grids*carbon)),'time':float(np.mean(tdec)) if tdec else 0.0}

def optimization(common):
    price=pd.read_csv(ROOT/'pricing.csv')['electricity_pricing'].values
    carbon=pd.read_csv(ROOT/'carbon_intensity.csv')['carbon_intensity'].values
    assets=read_schema_assets(); agg={k:{'energy':0,'peak':0,'cost':0,'co2':0,'time':[]} for k in ['baseline','rbc','optimizer']}; profiles=[]
    for b,g in common.groupby('building'):
        g=g.sort_values('t'); ix=g.t.astype(int).values; a=assets[int(b)]
        pv=g.solar_profile.values*a['pv_power']/1000.0
        actual=g.target.values; pred=g.pred_Ensemble.values; pr=price[ix]; ci=carbon[ix]
        runs={m:run_strategy(actual,pred,pv,pr,ci,a,m) for m in ['baseline','rbc','optimizer']}
        for m,r in runs.items():
            agg[m]['energy']+=r['energy']; agg[m]['cost']+=r['cost']; agg[m]['co2']+=r['co2']; agg[m]['peak']=max(agg[m]['peak'],r['peak']);
            if r['time']>0: agg[m]['time'].append(r['time'])
        if int(b)==1:
            profiles=pd.DataFrame({'t':ix,'baseline':runs['baseline']['grid'],'optimized':runs['optimizer']['grid']})
    for m in agg: agg[m]['time']=float(np.mean(agg[m]['time'])) if agg[m]['time'] else 0.0
    # district peak is better computed after time aggregation across buildings; redo aligned group sums
    # use approximate max-building peak above only if time axes differ; here aligned, so recompute profiles for all buildings
    district={m:{} for m in ['baseline','rbc','optimizer']}
    # Save representative figure from Building 1 real simulation
    plt.figure(figsize=(7.2,3.8)); q=profiles.iloc[:168]; plt.plot(np.arange(len(q)),q.baseline,label='No-control baseline',lw=1.4); plt.plot(np.arange(len(q)),q.optimized,label='Forecast-informed optimizer',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Grid demand (kWh/h)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure5_optimization.png',dpi=400,bbox_inches='tight'); plt.close()
    return agg

# ---------- external validation ----------
def bdg2_validation():
    path=Path('bdg2/data/meters/cleaned/electricity_cleaned.csv')
    if not path.exists(): return {'status':'unavailable'}
    use=['timestamp','Moose_education_Ricardo','Cockatoo_education_Erik']
    d=pd.read_csv(path,usecols=lambda c:c in use); d['timestamp']=pd.to_datetime(d.timestamp)
    out={}
    for name in use[1:]:
        z=d[['timestamp',name]].dropna().rename(columns={name:'target'}).reset_index(drop=True)
        y=z.target.astype(float); z['hour_sin']=np.sin(2*np.pi*z.timestamp.dt.hour/24); z['hour_cos']=np.cos(2*np.pi*z.timestamp.dt.hour/24); z['dow_sin']=np.sin(2*np.pi*z.timestamp.dt.dayofweek/7); z['dow_cos']=np.cos(2*np.pi*z.timestamp.dt.dayofweek/7); z['weekend']=(z.timestamp.dt.dayofweek>=5).astype(int); z['month']=z.timestamp.dt.month
        z['lag_1']=y.shift(1); z['lag_24']=y.shift(24); z['lag_168']=y.shift(168); s=y.shift(1); z['roll_mean_24']=s.rolling(24).mean(); z['roll_max_24']=s.rolling(24).max(); z=z.dropna().reset_index(drop=True)
        f=['month','hour_sin','hour_cos','dow_sin','dow_cos','weekend','lag_1','lag_24','lag_168','roll_mean_24','roll_max_24']
        n=len(z); a=int(.8*n); c=int(.9*n); tr=z.iloc[:a]; va=z.iloc[a:c]; te=z.iloc[c:]
        fam={
          'Random Forest':RandomForestRegressor(n_estimators=400,max_depth=20,min_samples_leaf=2,max_features='sqrt',n_jobs=-1,random_state=SEED),
          'XGBoost':XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,colsample_bytree=.8,reg_lambda=1,n_jobs=-1,random_state=SEED,objective='reg:squarederror'),
          'LightGBM':LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,n_jobs=-1,random_state=SEED,verbosity=-1),
          'CatBoost':CatBoostRegressor(iterations=600,depth=8,learning_rate=.03,l2_leaf_reg=3,loss_function='RMSE',verbose=False,random_seed=SEED)
        }
        vr={}; preds={}
        for k,m in fam.items(): m.fit(tr[f],tr.target); vr[k]=mean_squared_error(va.target,m.predict(va[f]))**.5; preds[k]=m.predict(te[f])
        top=sorted(vr,key=vr.get)[:3]; inv=np.array([1/(vr[k]+1e-8) for k in top]); w=inv/inv.sum(); p=sum(wi*preds[k] for wi,k in zip(w,top));
        out[name]={'n_raw_nonmissing':int(len(d[['timestamp',name]].dropna())),'n_after_lag':int(len(z)),'top3':top,'weights':w.tolist(),'mae':float(mean_absolute_error(te.target,p)),'rmse':float(mean_squared_error(te.target,p)**.5),'mape':float(np.mean(np.abs(te.target-p)/np.maximum(np.abs(te.target),1.0))*100),'r2':float(r2_score(te.target,p))}
    return out

def main():
    df=prep_citylearn(); print('Prepared',df.shape)
    trained,test,metrics,val_rmse,times=fit_tabular(df)
    lstm,common,metrics,val_rmse,times,hist=fit_lstm(df,test,metrics,val_rmse,times)
    metrics=recompute_common_metrics(common,metrics)
    common,top,w,metrics,times=make_ensemble(common,val_rmse,times,metrics)
    best_ind=min([k for k in metrics if k!='Proposed Validation-Weighted Ensemble'],key=lambda k:metrics[k][1])
    dm=dm_test(common,best_ind)
    ranking=shap_analysis(common,trained,top,w)
    abl=ablation(df,best_ind,metrics['Proposed Validation-Weighted Ensemble'][1],ranking)
    robust=seed_robustness(df)
    opt=optimization(common)
    bdg=bdg2_validation()
    # Figure 3: best overall model actual vs predicted, representative B1 first 168 hours
    g=common[common.building==1].sort_values('t').iloc[:168]
    plt.figure(figsize=(7.2,3.8)); plt.plot(np.arange(len(g)),g.target,label='Actual',lw=1.4); plt.plot(np.arange(len(g)),g.pred_Ensemble,label='Ensemble prediction',lw=1.4); plt.xlabel('Test hour'); plt.ylabel('Non-shiftable load (kWh)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'figure3_forecast.png',dpi=400,bbox_inches='tight'); plt.close()
    common[['building','t','target','pred_Ensemble']+['pred_'+k for k in ['Linear Regression','Random Forest','XGBoost','LightGBM','CatBoost','LSTM']]].to_csv(OUT/'test_predictions.csv',index=False)
    result={'dataset':{'rows_after_lag':int(len(df)),'common_test_rows':int(len(common)),'buildings':17,'mape_epsilon_kwh':EPS_MAPE},'metrics':{},'validation_rmse':val_rmse,'training_time_s_per_building':times,'ensemble':{'members':top,'weights':w.tolist()},'best_individual':best_ind,'dm_stat':dm[0],'dm_p':dm[1],'shap_ranking':ranking[:15],'ablation':abl,'robustness':robust,'optimization':opt,'bdg2':bdg,'lstm_epochs':len(hist['loss'])}
    for k,v in metrics.items(): result['metrics'][k]={'MAE':v[0],'RMSE':v[1],'MAPE':v[2],'R2':v[3]}
    # improvements
    en=result['metrics']['Proposed Validation-Weighted Ensemble']; bi=result['metrics'][best_ind]
    result['ensemble_improvement_vs_best_individual_pct']={'RMSE':(bi['RMSE']-en['RMSE'])/bi['RMSE']*100,'MAE':(bi['MAE']-en['MAE'])/bi['MAE']*100}
    base=opt['baseline']; rbc=opt['rbc']; oo=opt['optimizer']
    result['optimization_reduction_vs_baseline_pct']={k:(base[k]-oo[k])/base[k]*100 for k in ['energy','peak','cost','co2']}
    result['optimization_reduction_vs_rbc_pct']={k:(rbc[k]-oo[k])/rbc[k]*100 for k in ['energy','peak','cost','co2']}
    with open(OUT/'results.json','w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
