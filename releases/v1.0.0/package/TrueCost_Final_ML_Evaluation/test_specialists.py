"""Predeclared follow-up: test separate model/family regressors; training only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from improve_predictions import load,pipe,CAT,metrics

t,nums,_,_=load();y=t.total_tokens.to_numpy();groups=np.array(sorted(t.group_id.unique()))
for seed in [440,917]:
 out=Path(__file__).parent/f'specialists_seed{seed}';out.mkdir(exist_ok=True)
 pred=t[['run_id','group_id','family','model','total_tokens']].copy();pred['tail_threshold']=np.nan
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fit=np.flatnonzero(t.group_id.isin(groups[a]));val=np.flatnonzero(t.group_id.isin(groups[b]));pred.loc[val,'tail_threshold']=np.quantile(y[fit],.9)
  for name,cols,loss in [('by_model',['model'],'absolute_error'),('by_family',['family'],'absolute_error'),('by_model_family',['model','family'],'absolute_error'),('rich_boost_squared',[],'squared_error')]:
   model=pipe(HistGradientBoostingRegressor(loss=loss,max_iter=300,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=440),nums,False)
   if not cols:
    p=clone(model).fit(t.iloc[fit][CAT+nums],y[fit]).predict(t.iloc[val][CAT+nums]);pred.loc[val,name]=np.maximum(0,p)
   else:
    for key,g in t.iloc[val].groupby(cols):
     keys=key if isinstance(key,tuple) else (key,);mask=np.ones(len(fit),dtype=bool)
     for c,v in zip(cols,keys):mask&=t.iloc[fit][c].eq(v).to_numpy()
     fi=fit[mask];assert len(fi)>=100
     p=clone(model).fit(t.iloc[fi][CAT+nums],y[fi]).predict(g[CAT+nums]);pred.loc[g.index,name]=np.maximum(0,p)
  print(f'{seed} specialist fold {fold} complete',flush=True)
 tail=y>=pred.tail_threshold.to_numpy();names=['by_model','by_family','by_model_family','rich_boost_squared'];s=pd.DataFrame([dict(model=n,**metrics(y,pred[n].to_numpy(),tail)) for n in names]).sort_values('mae');s.to_csv(out/'metrics.csv',index=False);pred.to_csv(out/'oof_predictions.csv',index=False);print(s.to_string(index=False),flush=True)
