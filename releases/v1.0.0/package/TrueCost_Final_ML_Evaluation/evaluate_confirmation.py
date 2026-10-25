"""One frozen confirmation evaluation. Requires originally audited input exports."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from improve_predictions import task_features,ROOT

HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tasks',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--provenance',type=Path,required=True);ap.add_argument('--out',type=Path,default=HERE/'confirmation_evaluation');a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Use the original one-time receipt; do not rerun final evaluation.')
 ready=json.loads((HERE/'official_calibration/ready_for_confirmation.json').read_text())
 for name,h in ready['locked_files'].items():assert sha(HERE/name)==h,name
 # Export provenance is a receipt from the original strict collection audit, not a replacement audit.
 prov=json.loads(a.provenance.read_text())
 assert prov['manifest_sha256']=='ee380785a4222add8b4177f267a758a26265851138c1ca8c89ab35cf3977a828'
 assert prov['partition']=='confirmation' and prov['strict_audit_passed'] is True
 assert sha(a.tasks)==prov['tasks_sha256'] and sha(a.runs)==prov['runs_sha256']
 assert prov.get('audit_evidence_reference'),'Original strict audit reference required'
 params=json.loads((HERE/'official_calibration/calibration_parameters.json').read_text())
 freeze=json.loads((HERE/'selected_models/development_freeze.json').read_text())
 disp_path=ROOT/'data/20261020T021459Z/assignment_dispositions.csv'
 assert sha(disp_path)==freeze['data_hashes']['assignment_dispositions.csv']
 disp=pd.read_csv(disp_path);expected=disp[disp.partition.eq('confirmation')]
 tasks=json.loads(a.tasks.read_text());meta=pd.DataFrame([{k:t[k] for k in ['task_id','group_id','family','source_id','size_band']} for t in tasks])
 assert len(meta)==450 and meta.task_id.is_unique and meta.group_id.is_unique
 # Outcome read begins only after all frozen artifacts and input provenance have passed.
 d=pd.read_csv(a.runs)
 assert len(d)==1800 and d.run_id.is_unique and set(d.run_id)==set(expected.run_id)
 assert d.partition.eq('confirmation').all() and d.telemetry_complete.eq(True).all()
 assert d.total_tokens.notna().all() and (d.total_tokens>=0).all()
 assert (d.input_tokens+d.output_tokens==d.total_tokens).all()
 for key in ['task_id','group_id','model','seed']:
  assert d.set_index('run_id')[key].sort_index().equals(expected.set_index('run_id')[key].sort_index()),key
 assert d.groupby('group_id').size().eq(4).all()
 d=d.merge(meta,on='task_id',validate='many_to_one',suffixes=('','_task'));assert len(d)==1800
 for c in ['group_id','family','source_id','size_band']:assert d[c].eq(d[c+'_task']).all(),c
 d=d.merge(task_features(tasks),on='task_id',validate='many_to_one');assert d.group_id.eq(d.group_id_check).all()
 x=d[freeze['features']];assert not x.isna().any().any()
 p=np.maximum(0,joblib.load(HERE/'selected_models/point.joblib').predict(x));q=np.maximum(1,joblib.load(HERE/'selected_models/budget_uncalibrated.joblib').predict(x));upper=np.ceil(q*params['budget_multiplier']).astype('int64')
 assert np.isfinite(p).all() and np.isfinite(q).all()
 threshold=json.loads((HERE/'official_calibration/point_diagnostics.json').read_text())['tail_threshold_from_training']
 pred=d[['run_id','task_id','group_id','family','model','seed','total_tokens']].copy();pred['point_prediction']=p;pred['base_q90_budget']=q;pred['upper_budget']=upper
 rows=[]
 for col in ['all','family','model']:
  for label,g in ([('all',pred)] if col=='all' else pred.groupby(col)):
   y=g.total_tokens.to_numpy();e=g.point_prediction.to_numpy()-y;tail=y>=threshold;covered=g.total_tokens<=g.upper_budget;excess=y-g.upper_budget.to_numpy();den=np.sum((y-y.mean())**2)
   rows.append(dict(stratum=col,label=label,n=len(g),groups=g.group_id.nunique(),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/den) if den else None,bias=float(e.mean()),p90_absolute_error=float(np.quantile(abs(e),.9)),tail_threshold=threshold,tail_n=int(tail.sum()),tail_mae=float(abs(e[tail]).mean()) if tail.any() else None,tail_bias=float(e[tail].mean()) if tail.any() else None,row_budget_coverage=float(covered.mean()),joint_group_budget_coverage=float(covered.groupby(g.group_id).all().mean()),mean_budget=float(g.upper_budget.mean()),mean_exceedance=float(np.maximum(excess,0).mean()),q90_pinball=float(np.maximum(.9*excess,-.1*excess).mean())))
 a.out.mkdir(parents=True,exist_ok=False)
 pred.to_csv(a.out/'confirmation_predictions.csv',index=False);pd.DataFrame(rows).to_csv(a.out/'confirmation_metrics.csv',index=False)
 receipt=dict(ready_for_confirmation_sha256=sha(HERE/'official_calibration/ready_for_confirmation.json'),input_provenance=prov,confirmation_read=True,complete_rows=1800,task_groups=450,no_refit=True,no_tuning=True,warning='Any changes informed by these outcomes require new independent evaluation data.')
 (a.out/'evaluation_receipt.json').write_text(json.dumps(receipt,indent=2));print(pd.DataFrame(rows).query("stratum=='all'").to_string(index=False))
if __name__=='__main__':main()
