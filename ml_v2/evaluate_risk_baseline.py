"""Compare auxiliary request-limit forecasts with smoothed family/model rates."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss
HERE=Path(__file__).parent
rows=[]
for seed in [440,917]:
 p=pd.read_csv(HERE/f'risk_results/oof_predictions_{seed}.csv');p['family_model_baseline']=np.nan
 for fold in sorted(p.fold.unique()):
  fit=p[p.fold.ne(fold)].drop_duplicates(['group_id','model']);val=p[p.fold.eq(fold)]
  assert not set(fit.group_id)&set(val.group_id)
  counts=fit.groupby(['family','model']).request_cap.agg(['sum','count']);rate=(counts['sum']+1)/(counts['count']+2)
  p.loc[val.index,'family_model_baseline']=[rate.loc[(r.family,r.model)] for r in val.itertuples()]
 y=p.request_cap.astype(int)
 rows.append(dict(seed=seed,model='smoothed family/model rate',average_precision=average_precision_score(y,p.family_model_baseline),brier=brier_score_loss(y,p.family_model_baseline),log_loss=log_loss(y,p.family_model_baseline)))
pd.DataFrame(rows).to_csv(HERE/'risk_results/family_model_baseline.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
