"""Training-only five-fold grouped comparison; development evidence, not final test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def metrics(y, p, tail):
    e = p-y
    return dict(mae=float(mean_absolute_error(y,p)), rmse=float(np.sqrt(mean_squared_error(y,p))),
                r2=float(r2_score(y,p)), bias=float(e.mean()),
                p90_absolute_error=float(np.quantile(abs(e),.9)),
                tail_n=int(tail.sum()), tail_mae=float(abs(e[tail]).mean()),
                tail_bias=float(e[tail].mean()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--tasks',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('results_cv'))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    hashes=json.loads((a.data/'export_hashes.json').read_text())
    for name,h in hashes.items():
        assert hashlib.sha256((a.data/name).read_bytes()).hexdigest()==h, name
    d=pd.read_csv(a.data/'audited_visible_runs.csv')
    dispositions=pd.read_csv(a.data/'assignment_dispositions.csv')
    assert dispositions.groupby('group_id').partition.nunique().max()==1
    assert not d.run_id.duplicated().any()
    assert (d.input_tokens+d.output_tokens==d.total_tokens).all()
    assert d.telemetry_complete.eq(True).all()
    t=d.loc[d.partition.eq('training')].reset_index(drop=True)
    assert len(t)==8396
    tasks=json.loads(a.tasks.read_text())
    assert len(tasks)==2100 and len({r['task_id'] for r in tasks})==2100
    rows=[]
    for r in tasks:
        instruction=r['instruction']; context=r['context']
        rows.append(dict(task_id=r['task_id'], group_id=r['group_id'],
            family=r['family'],source_id=r['source_id'],size_band=r['size_band'],
            instruction_chars=len(instruction),context_chars=len(context),
            instruction_words=len(instruction.split()),context_words=len(context.split()),
            context_lines=len(context.splitlines()),has_public_tests=int(bool(r.get('public_tests')))))
    descriptors=pd.DataFrame(rows)
    check=t.merge(descriptors,on='task_id',suffixes=('','_task'),validate='many_to_one')
    assert len(check)==len(t)
    for column in ['group_id','family','source_id','size_band']:
        assert check[column].eq(check[column+'_task']).all(), column
    numeric=['instruction_chars','context_chars','instruction_words','context_words','context_lines','has_public_tests']
    t=t.merge(descriptors[['task_id']+numeric],on='task_id',validate='many_to_one')
    categorical=['family','source_id','size_band','model']
    features=categorical+numeric
    assert not t[features].isna().any().any()
    x=t[features]; y=t.total_tokens.to_numpy()
    groups=np.array(sorted(t.group_id.unique()))
    models={'median':DummyRegressor(strategy='median'),'ridge_raw':Ridge(alpha=10),
            'ridge_log':TransformedTargetRegressor(regressor=Ridge(alpha=10),func=np.log1p,inverse_func=np.expm1)}
    forest=RandomForestRegressor(n_estimators=200,min_samples_leaf=12,max_depth=16,n_jobs=2,random_state=440)
    models['forest_raw']=forest
    models['forest_log']=TransformedTargetRegressor(regressor=clone(forest),func=np.log1p,inverse_func=np.expm1)
    boost=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440)
    specs=[('length_forest_log',models['forest_log'],features),
           ('boost_raw',boost,features),
           ('boost_log',TransformedTargetRegressor(regressor=clone(boost),func=np.log1p,inverse_func=np.expm1),features),
           ('budget_q90',HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_iter=200,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440),features)]
    pred=t[['run_id','group_id','family','model','seed','total_tokens']].copy()
    pred['fold']=-1; pred['tail_threshold']=np.nan
    for name,_,_ in specs: pred[name]=np.nan
    scores=[]
    for fold,(gi,gv) in enumerate(KFold(5,shuffle=True,random_state=440).split(groups),1):
        fit=np.flatnonzero(t.group_id.isin(groups[gi])); val=np.flatnonzero(t.group_id.isin(groups[gv]))
        assert not set(t.iloc[fit].group_id)&set(t.iloc[val].group_id)
        threshold=float(np.quantile(y[fit],.9)); tail=y[val]>=threshold
        pred.loc[val,'fold']=fold; pred.loc[val,'tail_threshold']=threshold
        for name,model,cols in specs:
            prep=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[c for c in cols if c in categorical])],remainder='passthrough')
            pipe=make_pipeline(prep,clone(model))
            pipe.fit(x.iloc[fit][cols],y[fit]); p=np.maximum(0,pipe.predict(x.iloc[val][cols]))
            pred.loc[val,name]=p
            scores.append(dict(model=name,fold=fold,**metrics(y[val],p,tail)))
        print(f'Fold {fold}/5 complete',flush=True)
    assert pred['fold'].gt(0).all() and not pred[[s[0] for s in specs]].isna().any().any()
    pooled=[]; slices=[]
    tail=y>=pred.tail_threshold.to_numpy()
    for name,_,_ in specs:
        pooled.append(dict(model=name,**metrics(y,pred[name].to_numpy(),tail)))
        for column in ['family','model']:
            for label,g in pred.groupby(column):
                e=g[name]-g.total_tokens
                slices.append(dict(predictor=name,stratum=column,label=label,n=len(g),mae=float(abs(e).mean()),bias=float(e.mean())))
    pd.DataFrame(scores).to_csv(a.out/'fold_metrics.csv',index=False)
    pd.DataFrame(pooled).to_csv(a.out/'oof_metrics.csv',index=False)
    pd.DataFrame(slices).to_csv(a.out/'stratum_metrics.csv',index=False)
    pred.to_csv(a.out/'oof_predictions.csv',index=False)
    errors=pred.copy()
    errors['forest_absolute_error']=abs(errors.length_forest_log-errors.total_tokens)
    errors['forest_underprediction']=errors.total_tokens-errors.length_forest_log
    errors.nlargest(50,'forest_absolute_error').to_csv(a.out/'largest_forest_errors.csv',index=False)
    budgets=[]
    for name,_,_ in specs:
        for fold,g in [('all',pred)]+list(pred.groupby('fold')):
            covered=g.total_tokens<=g[name]
            pin=np.where(g.total_tokens>=g[name],.9*(g.total_tokens-g[name]),.1*(g[name]-g.total_tokens))
            group_covered=covered.groupby(g.group_id).all()
            budgets.append(dict(predictor=name,fold=fold,row_coverage=float(covered.mean()),
                all_replicates_group_coverage=float(group_covered.mean()),mean_budget=float(g[name].mean()),
                pinball_q90=float(np.mean(pin)),mean_exceedance=float(np.maximum(g.total_tokens-g[name],0).mean())))
    pd.DataFrame(budgets).to_csv(a.out/'budget_diagnostics.csv',index=False)
    (a.out/'analysis_notes.json').write_text(json.dumps(dict(training_rows=len(t),groups=len(groups),folds=5,random_state=440,
        calibration_outcomes_used=False,confirmation_outcomes_used=False,missing_totals_imputed=False,
        features=features,feature_provenance_verified=True,task_inputs_sha256=hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
        interpretation='Exploratory training-only model comparison. Descriptors derive only from frozen pre-run task inputs. Constant complexity removed. Candidate token estimates excluded. Tail means actual usage >= fit-fold 90th percentile. Scores condition on complete telemetry. Four fixed candidates; no hyperparameter search. Quantile budgets are uncalibrated. No deployment model saved.',
        data_hashes=hashes),indent=2))
    print(pd.DataFrame(pooled).sort_values('mae').to_string(index=False))
    print(pd.DataFrame(budgets).query("fold == 'all'").to_string(index=False))

if __name__=='__main__': main()
