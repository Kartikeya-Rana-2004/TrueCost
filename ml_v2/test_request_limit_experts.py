"""Route token predictions using an OOF pre-run request-limit risk forecast.
Realized request-limit labels supervise experts only on the fit fold.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from train_feature_ablations import HERE,V1,STRUCTURE,REQUEST,enhanced,estimator,CAT,metrics
from improve_predictions import task_features
starter=Path('/Users/kay/Downloads/ml_starter');f=json.loads((V1/'selected_models/development_freeze.json').read_text());data=starter/'data/20261020T021459Z'
for n,h in f['data_hashes'].items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h
rows=pd.read_csv(data/'audited_visible_runs.csv');rows=rows[rows.partition.eq('training')].reset_index(drop=True)
tasks_path=starter/'data/ml_feature_provenance/training_task_inputs.json';assert hashlib.sha256(tasks_path.read_bytes()).hexdigest()==f['task_inputs_sha256'];tasks=json.loads(tasks_path.read_text())
t=rows.merge(task_features(tasks),on='task_id',validate='many_to_one').merge(enhanced(tasks),on='task_id',validate='many_to_one');assert len(t)==8396
labels=pd.read_csv(HERE/'trace_analysis/training_diagnostics.csv')[['run_id','request_cap']];t=t.merge(labels,on='run_id',validate='one_to_one')
nums=[c for c in f['features'] if c not in CAT]+STRUCTURE+REQUEST;y=t.total_tokens.to_numpy();cap=t.request_cap.to_numpy(dtype=bool);groups=np.array(sorted(t.group_id.unique()));scores=[];foldscores=[]
for seed in [440,917]:
 risks=pd.read_csv(HERE/f'risk_results/oof_predictions_{seed}.csv').set_index('run_id');pred=t[['run_id','group_id','family','model','total_tokens']].copy();pred['threshold']=np.nan;pred['fold']=-1
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fit=np.flatnonzero(t.group_id.isin(groups[a]));val=np.flatnonzero(t.group_id.isin(groups[b]));assert not set(t.iloc[fit].group_id)&set(t.iloc[val].group_id)
  assert risks.loc[t.iloc[val].run_id,'fold'].eq(fold).all()
  prob=risks.loc[t.iloc[val].run_id,'request_cap_prob'].to_numpy();lofit=fit[~cap[fit]];hifit=fit[cap[fit]]
  assert len(lofit)>100 and len(hifit)>100
  lo=estimator(nums).fit(t.iloc[lofit][CAT+nums],y[lofit]).predict(t.iloc[val][CAT+nums]);hi=estimator(nums).fit(t.iloc[hifit][CAT+nums],y[hifit]).predict(t.iloc[val][CAT+nums])
  threshold=np.quantile(y[fit],.9);tail=y[val]>=threshold;pred.loc[val,'threshold']=threshold;pred.loc[val,'fold']=fold
  for name,p in [('limit_risk_soft',(1-prob)*lo+prob*hi),('limit_risk_hard',np.where(prob>=.5,hi,lo))]:
   p=np.maximum(0,p);pred.loc[val,name]=p;foldscores.append(dict(seed=seed,fold=fold,model=name,**metrics(y[val],p,tail)))
  print(f'limit experts seed={seed} fold={fold} complete',flush=True)
 for name in ['limit_risk_soft','limit_risk_hard']:scores.append(dict(seed=seed,model=name,**metrics(y,pred[name].to_numpy(),y>=pred.threshold.to_numpy())))
 out=HERE/'request_limit_experts';out.mkdir(exist_ok=True);pred.to_csv(out/f'oof_predictions_{seed}.csv',index=False)
pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False);pd.DataFrame(foldscores).to_csv(out/'fold_metrics.csv',index=False);print(pd.DataFrame(scores).to_string(index=False),flush=True)
(out/'protocol.json').write_text(json.dumps(dict(training_only=True,models_saved=False,calibration_outcomes_used=False,confirmation_outcomes_used=False,expert_fit_labels='Realized six-request limit in training fold only',routing_input='Predicted probability from group-disjoint outer-fold classifier',hard_threshold=.5,soft_rule='Probability-weighted conditional median predictions; exploratory heuristic',candidate_rationale='Chosen after training-trace/risk diagnostics; all reported scores remain development evidence, not a final evaluation',fresh_evaluation_required=True),indent=2))
