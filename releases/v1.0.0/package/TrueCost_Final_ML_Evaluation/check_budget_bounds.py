"""Nested training-only budget experiment; genuine calibration set remains unused."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold,StratifiedShuffleSplit
from improve_predictions import load,pipe,CAT

t,nums,_,_=load();y=t.total_tokens.to_numpy();groups=np.array(sorted(t.group_id.unique()));gf=t.groupby('group_id').family.first()
pred=t[['run_id','group_id','family','model','total_tokens']].copy();notes=[]
for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=440).split(groups),1):
 develop=groups[a];v=groups[b];i,j=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=1200+fold).split(develop,gf.loc[develop]))
 train_groups=develop[i];cal_groups=develop[j]
 assert not set(train_groups)&set(cal_groups) and not set(develop)&set(v)
 fit=np.flatnonzero(t.group_id.isin(train_groups));cal=np.flatnonzero(t.group_id.isin(cal_groups));val=np.flatnonzero(t.group_id.isin(v))
 m=pipe(HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440),nums,False)
 m.fit(t.iloc[fit][CAT+nums],y[fit]);qc=np.maximum(1,m.predict(t.iloc[cal][CAT+nums]));qv=np.maximum(1,m.predict(t.iloc[val][CAT+nums]))
 residual=pd.DataFrame({'group_id':t.iloc[cal].group_id.to_numpy(),'score':(y[cal]-qc)/qc}).groupby('group_id').score.max().to_numpy()
 # Order statistic at ceil((n+1)*(1-alpha)); clip correction to >=0 to avoid shrinking budgets.
 k=int(np.ceil((len(residual)+1)*.9));assert k<=len(residual)
 correction=max(0,float(np.sort(residual)[k-1]))
 pred.loc[val,'fold']=fold;pred.loc[val,'q90_uncalibrated']=qv;pred.loc[val,'group_calibrated_budget']=qv*(1+correction)
 notes.append(dict(fold=fold,fit_groups=len(train_groups),internal_calibration_groups=len(cal_groups),validation_groups=len(v),correction=correction))
 print(f'budget fold {fold} complete correction={correction:.3f}',flush=True)
rows=[]
for col in ['q90_uncalibrated','group_calibrated_budget']:
 for label,g in [('all',pred)]+list(pred.groupby('family')):
  yv=g.total_tokens.to_numpy();p=g[col].to_numpy();e=yv-p;covered=g.total_tokens<=g[col]
  rows.append(dict(predictor=col,family=label,n=len(g),row_coverage=covered.mean(),group_coverage=covered.groupby(g.group_id).all().mean(),mean_budget=p.mean(),q90_pinball=np.mean(np.maximum(.9*e,-.1*e)),mean_exceedance=np.maximum(e,0).mean()))
out=Path(__file__).parent/'budget_development';out.mkdir(exist_ok=True);pred.to_csv(out/'predictions.csv',index=False);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(notes).to_csv(out/'fold_protocol.csv',index=False)
print(pd.DataFrame(rows).query("family=='all'").to_string(index=False),flush=True)
