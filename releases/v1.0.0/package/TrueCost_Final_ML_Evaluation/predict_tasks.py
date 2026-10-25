"""Frozen point predictions and optional calibrated upper budgets for pre-run tasks."""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from improve_predictions import task_features

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 here=Path(__file__).parent
 p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path,default=here/'selected_models');p.add_argument('--calibration',type=Path,default=here/'official_calibration/calibration_parameters.json');a=p.parse_args()
 tasks=json.loads(a.tasks.read_text());assert tasks and len({t['task_id'] for t in tasks})==len(tasks)
 f=task_features(tasks);meta=pd.DataFrame([{k:t[k] for k in ['task_id','family','source_id','size_band']} for t in tasks]);f=f.merge(meta,on='task_id',validate='one_to_one')
 correction=None
 if a.calibration.exists():
  params=json.loads(a.calibration.read_text())
  assert params['point_sha256']==sha(a.models/'point.joblib')
  assert params['budget_sha256']==sha(a.models/'budget_uncalibrated.joblib')
  correction=params['normalized_correction'];assert correction>=0
 point_model=joblib.load(a.models/'point.joblib');q_model=joblib.load(a.models/'budget_uncalibrated.joblib');rows=[]
 for model in ['qwen2.5-coder:7b','qwen2.5-coder:14b']:
  x=f.assign(model=model);point=np.maximum(0,point_model.predict(x));q=np.maximum(1,q_model.predict(x))
  row=pd.DataFrame({'task_id':f.task_id,'group_id':f.group_id_check,'model':model,'predicted_total_tokens':point,'uncalibrated_q90_tokens':q})
  if correction is not None:row['calibrated_upper_budget']=np.ceil(q*(1+correction)).astype('int64')
  rows.append(row)
 with a.out.open('x') as out:pd.concat(rows,ignore_index=True).to_csv(out,index=False)
 print('Saved frozen predictions with '+('calibrated group upper budgets.' if correction is not None else 'uncalibrated q90 estimates.'))
if __name__=='__main__':main()
