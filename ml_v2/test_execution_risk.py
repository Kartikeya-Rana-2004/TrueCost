"""Pre-execution risk forecasts; execution outcomes are labels only, not features."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from sklearn.model_selection import KFold
from train_feature_ablations import HERE,V1,STRUCTURE,REQUEST,enhanced,estimator,CAT
from improve_predictions import task_features
starter=Path('/Users/kay/Downloads/ml_starter');frozen=json.loads((V1/'selected_models/development_freeze.json').read_text());data=starter/'data/20261020T021459Z'
for n,h in frozen['data_hashes'].items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h
rows=pd.read_csv(data/'audited_visible_runs.csv');rows=rows[rows.partition.eq('training')].reset_index(drop=True)
tasks_path=starter/'data/ml_feature_provenance/training_task_inputs.json';assert hashlib.sha256(tasks_path.read_bytes()).hexdigest()==frozen['task_inputs_sha256'];tasks=json.loads(tasks_path.read_text())
t=rows.merge(task_features(tasks),on='task_id',validate='many_to_one').merge(enhanced(tasks),on='task_id',validate='many_to_one')
diag=pd.read_csv(HERE/'trace_analysis/training_diagnostics.csv')[['run_id','request_cap','hit_output_cap']];t=t.merge(diag,on='run_id',validate='one_to_one')
assert len(t)==8396 and t.group_id.eq(t.group_id_check).all()
nums=[c for c in frozen['features'] if c not in CAT]+STRUCTURE+REQUEST
assert not {'requests','termination','input_tokens','output_tokens','total_tokens','request_cap','hit_output_cap','run_id','seed','group_id','task_id'}&set(CAT+nums)
groups=np.array(sorted(t.group_id.unique()));summary=[]
for seed in [440,917]:
 pred=t[['run_id','group_id','family','model','request_cap','hit_output_cap']].copy()
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fi=np.flatnonzero(t.group_id.isin(groups[a]));va=np.flatnonzero(t.group_id.isin(groups[b]));pred.loc[va,'fold']=fold
  for target in ['request_cap','hit_output_cap']:
   y=t[target].astype(int).to_numpy();m=estimator(nums,'classifier').fit(t.iloc[fi][CAT+nums],y[fi]);prob=m.predict_proba(t.iloc[va][CAT+nums])[:,1]
   pred.loc[va,target+'_prob']=prob;pred.loc[va,target+'_base_prob']=y[fi].mean()
  print(f'risk seed={seed} fold={fold} complete',flush=True)
 for target in ['request_cap','hit_output_cap']:
  y=pred[target].astype(int).to_numpy();p=pred[target+'_prob'].to_numpy();b=pred[target+'_base_prob'].to_numpy();assert np.isfinite(p).all()
  top=pred.nlargest(int(np.ceil(len(pred)*.1)),target+'_prob');summary.append(dict(seed=seed,target=target,prevalence=y.mean(),average_precision=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),brier=brier_score_loss(y,p),constant_brier=brier_score_loss(y,b),log_loss=log_loss(y,p),constant_log_loss=log_loss(y,b),top_ten_percent_precision=top[target].mean(),top_ten_percent_recall=top[target].sum()/y.sum()))
 out=HERE/'risk_results';out.mkdir(exist_ok=True);pred.to_csv(out/f'oof_predictions_{seed}.csv',index=False)
pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
(out/'protocol.json').write_text(json.dumps(dict(training_only=True,labels=['reached six-request limit','any request output equaled 1024-token ceiling'],labels_used_as_features=False,features=CAT+nums,probabilities_calibrated=False,models_saved=False,interpretation='Exploratory auxiliary forecasting only; not a calibrated warning or final risk model. Different from predicting task correctness. No outcome labels were corrected.'),indent=2))
print(pd.DataFrame(summary).to_string(index=False),flush=True)
