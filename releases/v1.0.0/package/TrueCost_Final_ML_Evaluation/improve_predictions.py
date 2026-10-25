"""Training-only grouped experiments. Never fit/select on calibration or confirmation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from prior_train_advanced_cv import metrics

ROOT=Path(os.environ.get('TRUECOST_ML_ROOT','/Users/kay/Downloads/ml_starter'))
BASE=['instruction_chars','context_chars','instruction_words','context_words','context_lines','has_public_tests']
CAT=['family','source_id','size_band','model']

def task_features(tasks):
 rows=[]
 for r in tasks:
  ins=r['instruction']; ctx=r['context']; tests=json.dumps(r.get('public_tests') or '',sort_keys=True)
  text=ins+'\n'+ctx; words=text.split(); n=max(1,len(text))
  f=dict(task_id=r['task_id'],group_id_check=r['group_id'],instruction_chars=len(ins),context_chars=len(ctx),instruction_words=len(ins.split()),context_words=len(ctx.split()),context_lines=len(ctx.splitlines()),has_public_tests=int(bool(r.get('public_tests'))),tests_chars=len(tests),text=text,total_chars=len(text),total_words=len(words),unique_word_ratio=len(set(words))/max(1,len(words)),digit_fraction=sum(c.isdigit() for c in text)/n,punctuation_fraction=sum(not c.isalnum() and not c.isspace() for c in text)/n,newline_count=text.count('\n'),question_count=text.count('?'),avg_word_chars=sum(map(len,words))/max(1,len(words)),instruction_context_ratio=len(ins)/max(1,len(ctx)))
  for name,pattern in [('numbers',r'\b\d+(?:\.\d+)?\b'),('sentences',r'[.!?](?:\s|$)'),('table_separators',r'\|'),('code_markers',r'\b(?:def|class|return|assert|for|while|import)\b')]: f[name]=len(re.findall(pattern,text))
  for name in ['instruction_chars','context_chars','total_chars','total_words','tests_chars']:
   f['log_'+name]=np.log1p(f[name])
  rows.append(f)
 return pd.DataFrame(rows)

def load():
 data=ROOT/'data/20261020T021459Z'
 hashes=json.loads((data/'export_hashes.json').read_text())
 for name,h in hashes.items():
  assert hashlib.sha256((data/name).read_bytes()).hexdigest()==h,name
 all_rows=pd.read_csv(data/'audited_visible_runs.csv')
 # Filter immediately; held-out outcomes never enter features or models.
 t=all_rows.loc[all_rows.partition.eq('training')].copy().reset_index(drop=True)
 del all_rows
 assert len(t)==8396 and t.run_id.is_unique
 assert (t.input_tokens+t.output_tokens==t.total_tokens).all()
 tasks_path=ROOT/'data/ml_feature_provenance/training_task_inputs.json'
 tasks=json.loads(tasks_path.read_text()); assert len(tasks)==2100
 desc=task_features(tasks)
 t=t.merge(desc,on='task_id',validate='many_to_one')
 assert t.group_id.eq(t.group_id_check).all()
 nums=[c for c in desc if c not in ['task_id','group_id_check','text']]
 assert not t[CAT+nums].isna().any().any()
 return t,nums,hashes,hashlib.sha256(tasks_path.read_bytes()).hexdigest()

def pipe(est,nums,log=True):
 pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT),('num','passthrough',nums)])
 if log: est=TransformedTargetRegressor(regressor=est,func=np.log1p,inverse_func=np.expm1)
 return make_pipeline(pre,est)

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);t,nums,hashes,taskhash=load();y=t.total_tokens.to_numpy();groups=np.array(sorted(t.group_id.unique()))
 rf=lambda leaf: RandomForestRegressor(n_estimators=240,min_samples_leaf=leaf,max_depth=None,n_jobs=2,random_state=440)
 specs={
 'baseline':pipe(RandomForestRegressor(n_estimators=200,min_samples_leaf=12,max_depth=16,n_jobs=2,random_state=440),BASE),
 'rich_rf_leaf5':pipe(rf(5),nums),
 'rich_rf_leaf2':pipe(rf(2),nums),
 'rich_extra_leaf3':pipe(ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,n_jobs=2,random_state=440),nums),
 'rich_boost_log':pipe(HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440),nums),
 'rich_boost_absolute':pipe(HistGradientBoostingRegressor(loss='absolute_error',max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440),nums,False),
 }
 pred=t[['run_id','group_id','family','model','seed','total_tokens']].copy();pred['fold']=-1;pred['tail_threshold']=np.nan
 fold_scores=[]
 for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1):
  fit=np.flatnonzero(t.group_id.isin(groups[a]));val=np.flatnonzero(t.group_id.isin(groups[b]));assert not set(t.iloc[fit].group_id)&set(t.iloc[val].group_id)
  threshold=np.quantile(y[fit],.9);tail=y[val]>=threshold
  pred.loc[val,'fold']=fold;pred.loc[val,'tail_threshold']=threshold
  for name,model in specs.items():
   start=time.time();p=clone(model).fit(t.iloc[fit][CAT+nums],y[fit]).predict(t.iloc[val][CAT+nums]);p=np.maximum(p,0);pred.loc[val,name]=p
   fold_scores.append(dict(model=name,fold=fold,**metrics(y[val],p,tail)));print(f'seed={seed} fold={fold} {name} MAE={np.mean(abs(p-y[val])):.1f} {time.time()-start:.1f}s',flush=True)
  # Every vocabulary and scaling step is learned exclusively inside this fold.
  pre=ColumnTransformer([('text',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=18000,sublinear_tf=True),'text'),('cat',OneHotEncoder(handle_unknown='ignore'),CAT),('num',StandardScaler(),nums)])
  textmodel=make_pipeline(pre,Ridge(alpha=10,solver='lsqr'))
  p=np.maximum(0,np.expm1(textmodel.fit(t.iloc[fit][CAT+nums+['text']],np.log1p(y[fit])).predict(t.iloc[val][CAT+nums+['text']])));pred.loc[val,'text_ridge_log']=p
  print(f'seed={seed} fold={fold} text_ridge_log MAE={np.mean(abs(p-y[val])):.1f}',flush=True)
  pred.to_csv(out/'oof_predictions_partial.csv',index=False)
 for name,weight in [('blend_trees',0),('blend_text20',.2)]:
  tree=(pred.rich_rf_leaf5+pred.rich_extra_leaf3)/2
  pred[name]=(1-weight)*tree+weight*pred.text_ridge_log
 names=list(specs)+['text_ridge_log','blend_trees','blend_text20'];tail=y>=pred.tail_threshold.to_numpy()
 summary=pd.DataFrame([dict(model=n,**metrics(y,pred[n].to_numpy(),tail)) for n in names]).sort_values('mae')
 summary.to_csv(out/'metrics.csv',index=False);pred.to_csv(out/'oof_predictions.csv',index=False);pd.DataFrame(fold_scores).to_csv(out/'fold_metrics.csv',index=False)
 slices=[]
 for name in names:
  for col in ['family','model','fold']:
   for label,g in pred.groupby(col):
    e=g[name]-g.total_tokens;slices.append(dict(predictor=name,stratum=col,label=label,n=len(g),mae=float(abs(e).mean()),bias=float(e.mean())))
 pd.DataFrame(slices).to_csv(out/'strata.csv',index=False)
 (out/'protocol.json').write_text(json.dumps(dict(seed=seed,training_rows=len(t),groups=len(groups),data_hashes=hashes,task_inputs_sha256=taskhash,features=CAT+nums,calibration_used=False,confirmation_used=False,missing_targets_imputed=False,interpretation='Exploratory comparison; repeated folds are sensitivity checks on the same tasks, not independent final test evidence.'),indent=2))
 print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--seed',type=int,default=440);p.add_argument('--out',type=Path,required=True);a=p.parse_args();ROOT=a.root;run(a.seed,a.out)
