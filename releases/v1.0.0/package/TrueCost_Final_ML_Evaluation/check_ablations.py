import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from improve_predictions import load,pipe,CAT,BASE,metrics

t,nums,_,_=load();y=t.total_tokens.to_numpy();groups=np.array(sorted(t.group_id.unique()));out=Path(__file__).parent/'ablations';out.mkdir(exist_ok=True)
rows=[]
for seed in [440,917]:
 pred=t[['run_id','group_id','total_tokens']].copy();pred['threshold']=np.nan
 specs={'absolute_basic':(CAT,BASE),'absolute_no_model':([c for c in CAT if c!='model'],nums)}
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fi=np.flatnonzero(t.group_id.isin(groups[a]));va=np.flatnonzero(t.group_id.isin(groups[b]));pred.loc[va,'threshold']=np.quantile(y[fi],.9)
  for name,(cats,ns) in specs.items():
   pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('num','passthrough',ns)])
   m=make_pipeline(pre,HistGradientBoostingRegressor(loss='absolute_error',max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440))
   pred.loc[va,name]=np.maximum(0,m.fit(t.iloc[fi][cats+ns],y[fi]).predict(t.iloc[va][cats+ns]))
 for n in specs:rows.append(dict(seed=seed,model=n,**metrics(y,pred[n].to_numpy(),y>=pred.threshold.to_numpy())))
 pred.to_csv(out/f'predictions_{seed}.csv',index=False)
s=pd.DataFrame(rows);s.to_csv(out/'metrics.csv',index=False);print(s.to_string(index=False))
