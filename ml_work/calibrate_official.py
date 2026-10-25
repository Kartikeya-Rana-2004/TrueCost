"""Apply frozen group-level one-sided calibration; never select or refit models."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from improve_predictions import task_features,ROOT

HERE=Path(__file__).parent
OUT=HERE/'official_calibration'
MODEL_DIR=HERE/'selected_models'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def correction(scores,alpha=.1):
 scores=np.asarray(scores,dtype=float)
 assert len(scores)>0 and np.isfinite(scores).all()
 rank=math.ceil((len(scores)+1)*(1-alpha))
 if rank>len(scores):raise ValueError('Not enough calibration groups')
 return max(0.,float(np.sort(scores)[rank-1])),rank

def main():
 OUT.mkdir(exist_ok=True)
 frozen=json.loads((MODEL_DIR/'development_freeze.json').read_text())
 prior_hashes=json.loads((HERE/'deliverable_hashes.json').read_text())
 for name in ['selected_models/point.joblib','selected_models/budget_uncalibrated.joblib','improve_predictions.py','selected_models/development_freeze.json']:
  assert sha(HERE/name)==prior_hashes[name],name
 prov=json.loads((OUT/'input/calibration_task_inputs.provenance.json').read_text())
 task_path=OUT/'input/calibration_task_inputs.json'
 assert sha(task_path)==prov['export_sha256'] and prov['partition']=='calibration'
 assert prov['manifest_sha256']=='ee380785a4222add8b4177f267a758a26265851138c1ca8c89ab35cf3977a828'
 assert not prov['confirmation_exported'] and not prov['outcomes_exported']
 protocol=dict(schema_version=1,locked_at_utc=datetime.now(timezone.utc).isoformat(),alpha=.1,unit='task group; all four declared model/seed assignments jointly',base_budget='max(1, frozen q90 predictor)',score='max_over_group((actual_total_tokens-base_budget)/base_budget)',order_statistic='ceil((n_groups+1)*0.9), one-indexed',correction='max(0, selected_order_statistic)',deployed_budget='ceil(base_budget*(1+correction))',point_predictor='frozen absolute-error histogram gradient boosting; clamp predictions to zero',no_refitting=True,no_model_selection=True,no_family_specific_corrections=True,confirmation_accessed=False,inputs_sha256=sha(task_path),point_sha256=sha(MODEL_DIR/'point.joblib'),budget_sha256=sha(MODEL_DIR/'budget_uncalibrated.joblib'),feature_code_sha256=sha(HERE/'improve_predictions.py'),script_sha256=sha(Path(__file__)),data_hashes=frozen['data_hashes'],confirmation_plan=dict(primary='MAE over all complete confirmation attempts',secondary=['RMSE','R2','mean signed error','90th percentile absolute error','tail MAE and signed bias using training 90th percentile threshold','row upper-budget coverage','all-runs group upper-budget coverage','mean upper budget','mean budget exceedance','q90 pinball loss','family/model slices'],predictors='Evaluate frozen selected point model and calibrated budget once; no subsequent tuning labeled as final-test evaluation',missingness='Report dispositions separately; no target imputation; group joint coverage requires all four attempts'),limitation='Empirical calibration; nominal marginal coverage requires exchangeability assumptions not established by this purposive deterministic sample')
 protocol_path=OUT/'locked_analysis_protocol.json'
 if protocol_path.exists():
  existing=json.loads(protocol_path.read_text());protocol['locked_at_utc']=existing['locked_at_utc']
  assert protocol==existing,'Protocol differs from prior lock'
 else:
  with protocol_path.open('x') as f:json.dump(protocol,f,indent=2)
 # Only after the specification is written do we inspect official calibration outcomes.
 data=ROOT/'data/20261020T021459Z'
 for name,h in frozen['data_hashes'].items():assert sha(data/name)==h,name
 visible=pd.read_csv(data/'audited_visible_runs.csv')
 assert set(visible.partition)=={'training','calibration'}
 cal=visible.loc[visible.partition.eq('calibration')].copy().reset_index(drop=True)
 train=visible.loc[visible.partition.eq('training'),['group_id','total_tokens']].copy()
 del visible
 assert len(cal)==1800 and cal.run_id.is_unique and cal.group_id.nunique()==450
 assert cal.groupby('group_id').size().eq(4).all()
 assert not set(cal.group_id)&set(train.group_id)
 assert cal.telemetry_complete.eq(True).all()
 assert (cal.input_tokens+cal.output_tokens==cal.total_tokens).all()
 assert cal[['group_id','model','seed']].duplicated().sum()==0
 for _,g in cal.groupby('group_id'):
  assert set(zip(g.model,g.seed))=={(m,s) for m in ['qwen2.5-coder:7b','qwen2.5-coder:14b'] for s in [440,441]}
 tasks=json.loads(task_path.read_text());assert len(tasks)==450
 desc=task_features(tasks)
 meta=pd.DataFrame([{k:t[k] for k in ['task_id','group_id','family','source_id','size_band']} for t in tasks])
 assert meta.task_id.is_unique and meta.group_id.is_unique
 joined=cal.merge(meta,on='task_id',suffixes=('','_input'),validate='many_to_one')
 assert len(joined)==len(cal) and set(joined.task_id)==set(meta.task_id)
 for col in ['group_id','family','source_id','size_band']:assert joined[col].eq(joined[col+'_input']).all(),col
 joined=joined.merge(desc,on='task_id',validate='many_to_one');assert joined.group_id.eq(joined.group_id_check).all()
 x=joined[frozen['features']];assert not x.isna().any().any()
 point=joblib.load(MODEL_DIR/'point.joblib');quantile=joblib.load(MODEL_DIR/'budget_uncalibrated.joblib')
 p=np.maximum(0,point.predict(x));q=np.maximum(1,quantile.predict(x));y=joined.total_tokens.to_numpy()
 assert np.isfinite(p).all() and np.isfinite(q).all()
 pred=joined[['run_id','task_id','group_id','family','model','seed','total_tokens']].copy()
 pred['point_prediction']=p;pred['base_q90_budget']=q;pred['normalized_residual']=(y-q)/q
 scores=pred.groupby('group_id').normalized_residual.max()
 c,rank=correction(scores);pred['calibrated_upper_budget']=np.ceil(q*(1+c)).astype('int64')
 summaries=[]
 for name in ['base_q90_budget','calibrated_upper_budget']:
  for col in ['all','family','model']:
   for label,g in ([('all',pred)] if col=='all' else pred.groupby(col)):
    e=g.total_tokens-g[name];covered=e<=0
    summaries.append(dict(budget=name,stratum=col,label=label,n=len(g),groups=g.group_id.nunique(),row_coverage=float(covered.mean()),group_joint_coverage=float(covered.groupby(g.group_id).all().mean()),mean_budget=float(g[name].mean()),mean_exceedance=float(e.clip(lower=0).mean()),q90_pinball=float(np.maximum(.9*e,-.1*e).mean())))
 error=p-y;tail_threshold=float(train.total_tokens.quantile(.9));tail=y>=tail_threshold
 point_summary=dict(mae=float(abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),p90_absolute_error=float(np.quantile(abs(error),.9)),tail_threshold_from_training=tail_threshold,tail_n=int(tail.sum()),tail_mae=float(abs(error[tail]).mean()),tail_bias=float(error[tail].mean()),interpretation='Calibration diagnostics only; not used to select or refit predictors and not final-test evidence')
 parameters=dict(alpha=.1,calibration_groups=len(scores),order_statistic_rank=rank,normalized_correction=c,budget_multiplier=1+c,point_sha256=sha(MODEL_DIR/'point.joblib'),budget_sha256=sha(MODEL_DIR/'budget_uncalibrated.joblib'),protocol_sha256=sha(protocol_path),calibration_inputs_sha256=sha(task_path),calibration_rows=len(cal),confirmation_outcomes_used=False,scope='Complete telemetry under frozen controller; joint upper bound over the four model/seed assignments of a task group; exchangeability caveat applies')
 # Known-score tests: check rank and one-sided clipping, plus joint score/bound equivalence.
 assert correction(np.arange(9))[0]==8 and correction(-np.arange(9)-1)[0]==0
 assert np.array_equal((pred.normalized_residual<=c).groupby(pred.group_id).all().to_numpy(),scores.le(c).to_numpy())
 assert pred.calibrated_upper_budget.ge(pred.base_q90_budget).all()
 pred.to_csv(OUT/'calibration_predictions.csv',index=False);scores.rename('group_score').to_csv(OUT/'calibration_group_scores.csv');pd.DataFrame(summaries).to_csv(OUT/'budget_diagnostics.csv',index=False)
 (OUT/'calibration_parameters.json').write_text(json.dumps(parameters,indent=2));(OUT/'point_diagnostics.json').write_text(json.dumps(point_summary,indent=2))
 assert parameters['point_sha256']==sha(MODEL_DIR/'point.joblib') and parameters['budget_sha256']==sha(MODEL_DIR/'budget_uncalibrated.joblib')
 print(json.dumps(parameters,indent=2));print(json.dumps(point_summary,indent=2));print(pd.DataFrame(summaries).query("stratum=='all'").to_string(index=False))
if __name__=='__main__':main()
