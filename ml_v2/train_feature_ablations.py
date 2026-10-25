"""V2 training-only feature ablation and high-usage mixture experiments.
All candidates use the same grouped folds. No V1 holdout outcomes are loaded.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,ast,hashlib,json,re,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
HERE=Path(__file__).parent
V1=HERE.parent/'releases/v1.0.0/package/TrueCost_Final_ML_Evaluation'
sys.path.insert(0,str(V1))
from improve_predictions import task_features,CAT
from prior_train_advanced_cv import metrics

STRUCTURE=['paragraphs','longest_line','mean_line_chars','nonempty_lines','list_items','numeric_list_items','table_rows','max_table_cells','code_fences','inline_code_spans','examples_markers','instruction_questions','function_markers','assert_markers','branch_markers','loop_markers']
REQUEST=['requested_quantity','has_requested_quantity','request_list','request_explain','request_compare','request_steps','request_brief','request_detail','request_json','request_code']

def enhanced(tasks):
 rows=[]
 for r in tasks:
  ins=r['instruction'];ctx=r['context'];text=ins+'\n'+ctx;lines=[s for s in text.splitlines() if s.strip()]
  quantities=[int(x.replace(',','')) for x in re.findall(r'\b(\d[\d,]*)\s*(?:words?|sentences?|paragraphs?|bullet points?|items?|examples?|lines?)\b',ins,re.I)]
  row=dict(task_id=r['task_id'],paragraphs=len(re.findall(r'\n\s*\n',text))+1,longest_line=max(map(len,lines),default=0),mean_line_chars=float(np.mean(list(map(len,lines)))) if lines else 0,nonempty_lines=len(lines),list_items=len(re.findall(r'^\s*[-*•]\s+',text,re.M)),numeric_list_items=len(re.findall(r'^\s*\d+[.)]\s+',text,re.M)),table_rows=sum(('|' in line or '\t' in line or '#' in line) for line in ctx.splitlines()),max_table_cells=max([max(line.count('|'),line.count('\t'),line.count('#'))+1 for line in ctx.splitlines()],default=0),code_fences=text.count('```'),inline_code_spans=len(re.findall(r'(?<!`)`[^`\n]+`(?!`)',text)),examples_markers=len(re.findall(r'\bexamples?\b|>>>',ins,re.I)),instruction_questions=ins.count('?'),function_markers=len(re.findall(r'\bdef\s+\w+\s*\(',ins)),assert_markers=len(re.findall(r'\bassert\b',ins)),branch_markers=len(re.findall(r'\b(?:if|elif|else)\b',ins)),loop_markers=len(re.findall(r'\b(?:for|while)\b',ins)),requested_quantity=min(max(quantities,default=0),100000),has_requested_quantity=int(bool(quantities)))
  for name,pat in [('list',r'\b(?:list|enumerate|name|give me)\b'),('explain',r'\b(?:explain|describe|why|how)\b'),('compare',r'\b(?:compare|contrast|difference|versus)\b'),('steps',r'\b(?:steps?|step-by-step)\b'),('brief',r'\b(?:brief|briefly|short|concise|concisely)\b'),('detail',r'\b(?:detail|detailed|thorough|comprehensive)\b'),('json',r'\bjson\b'),('code',r'\b(?:function|python|code|program|implement)\b')]:row['request_'+name]=int(bool(re.search(pat,ins,re.I)))
  rows.append(row)
 return pd.DataFrame(rows)

def estimator(nums,kind='point'):
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT),('num','passthrough',nums)])
 params=dict(max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440)
 model=HistGradientBoostingClassifier(**params) if kind=='classifier' else HistGradientBoostingRegressor(loss='absolute_error',**params)
 return make_pipeline(prep,model)

def run(starter,seed):
 freeze=json.loads((V1/'selected_models/development_freeze.json').read_text());data=starter/'data/20261020T021459Z'
 for n,h in freeze['data_hashes'].items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h
 # Exclude calibration immediately; its outcomes do not inform these candidates.
 t=pd.read_csv(data/'audited_visible_runs.csv');t=t.loc[t.partition.eq('training')].reset_index(drop=True)
 tasks_path=starter/'data/ml_feature_provenance/training_task_inputs.json';assert hashlib.sha256(tasks_path.read_bytes()).hexdigest()==freeze['task_inputs_sha256']
 tasks=json.loads(tasks_path.read_text());base=task_features(tasks);features=enhanced(tasks)
 t=t.merge(base,on='task_id',validate='many_to_one').merge(features,on='task_id',validate='many_to_one')
 assert len(t)==8396 and t.group_id.nunique()==2100 and t.group_id.eq(t.group_id_check).all()
 old=[c for c in freeze['features'] if c not in CAT];nums=old+STRUCTURE+REQUEST
 assert not t[CAT+nums].isna().any().any()
 prohibited={'input_tokens','output_tokens','total_tokens','requests','termination','quality_score','wall_seconds','run_id','task_id','group_id','seed'};assert not prohibited&set(CAT+nums)
 y=t.total_tokens.to_numpy();groups=np.array(sorted(t.group_id.unique()));pred=t[['run_id','group_id','family','model','total_tokens']].copy();pred['fold']=-1;pred['threshold']=np.nan
 specs={'v1_retrained':old,'structure_only_addition':old+STRUCTURE,'request_only_addition':old+REQUEST,'all_new_features':nums}
 fold_scores=[]
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fit=np.flatnonzero(t.group_id.isin(groups[a]));val=np.flatnonzero(t.group_id.isin(groups[b]));assert not set(t.iloc[fit].group_id)&set(t.iloc[val].group_id)
  threshold=float(np.quantile(y[fit],.9));tail=y[val]>=threshold;pred.loc[val,'fold']=fold;pred.loc[val,'threshold']=threshold
  for name,cs in specs.items():
   m=estimator(cs);p=np.maximum(0,m.fit(t.iloc[fit][CAT+cs],y[fit]).predict(t.iloc[val][CAT+cs]));pred.loc[val,name]=p;fold_scores.append(dict(seed=seed,fold=fold,model=name,**metrics(y[val],p,tail)))
  high=y[fit]>=threshold;assert high.sum()>100 and (~high).sum()>100
  clf=estimator(nums,'classifier');clf.fit(t.iloc[fit][CAT+nums],high.astype(int));prob=clf.predict_proba(t.iloc[val][CAT+nums])[:,1]
  low_model=estimator(nums).fit(t.iloc[fit[~high]][CAT+nums],y[fit[~high]])
  high_model=estimator(nums).fit(t.iloc[fit[high]][CAT+nums],y[fit[high]])
  lo=np.maximum(0,low_model.predict(t.iloc[val][CAT+nums]));hi=np.maximum(0,high_model.predict(t.iloc[val][CAT+nums]))
  pred.loc[val,'high_usage_probability']=prob
  for name,p in [('mixture_soft',(1-prob)*lo+prob*hi),('mixture_hard',np.where(prob>=.5,hi,lo))]:
   pred.loc[val,name]=p;fold_scores.append(dict(seed=seed,fold=fold,model=name,**metrics(y[val],p,tail)))
  print(f'seed={seed} fold={fold} complete',flush=True)
 tail=y>=pred.threshold.to_numpy();names=list(specs)+['mixture_soft','mixture_hard'];assert pred.fold.gt(0).all() and pred[names].notna().all().all()
 scores=pd.DataFrame([dict(seed=seed,model=n,**metrics(y,pred[n].to_numpy(),tail)) for n in names]).sort_values('mae');out=HERE/f'feature_results_seed{seed}';out.mkdir(exist_ok=True)
 pred.to_csv(out/'oof_predictions.csv',index=False);scores.to_csv(out/'metrics.csv',index=False);pd.DataFrame(fold_scores).to_csv(out/'fold_metrics.csv',index=False)
 strata=[]
 for n in names:
  for fam,g in pred.groupby('family'):strata.append(dict(model=n,family=fam,n=len(g),mae=float(abs(g[n]-g.total_tokens).mean()),bias=float((g[n]-g.total_tokens).mean())))
 pd.DataFrame(strata).to_csv(out/'family_metrics.csv',index=False)
 desc=features.drop(columns='task_id');pd.DataFrame({'feature':desc.columns,'nonzero_tasks':[(desc[c]!=0).sum() for c in desc],'total_tasks':len(desc)}).to_csv(out/'feature_availability.csv',index=False)
 (out/'protocol.json').write_text(json.dumps(dict(training_only=True,training_rows=len(t),groups=len(groups),split_seed=seed,baseline_features=old,structure_features=STRUCTURE,request_features=REQUEST,candidates=list(specs)+['mixture_soft','mixture_hard'],calibration_outcomes_used=False,confirmation_outcomes_used=False,tail_threshold='fit-fold training 90th percentile',mixture_note='Conditional-median mixture is an exploratory heuristic, not the exact marginal median.',new_independent_test_required=True,feature_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print(scores.to_string(index=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--starter',type=Path,default=Path('/Users/kay/Downloads/ml_starter'));a=p.parse_args()
 for seed in [440,917]:run(a.starter,seed)
