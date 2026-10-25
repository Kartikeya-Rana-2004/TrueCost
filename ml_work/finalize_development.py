"""Save selected development models and task-cluster comparison diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from improve_predictions import load,task_features,pipe,CAT,ROOT

root=Path(__file__).parent;out=root/'selected_models';out.mkdir(exist_ok=True)
t,nums,hashes,taskhash=load();y=t.total_tokens.to_numpy()
point_params=dict(loss='absolute_error',max_iter=300,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=440)
budget_params={**point_params,'loss':'quantile','quantile':.9}
for name,params in [('point',point_params),('budget_uncalibrated',budget_params)]:
 m=pipe(HistGradientBoostingRegressor(**params),nums,False);m.fit(t[CAT+nums],y)
 assert np.isfinite(m.predict(t[CAT+nums])).all()
 path=out/f'{name}.joblib';joblib.dump(m,path)
 assert np.allclose(joblib.load(path).predict(t.iloc[:20][CAT+nums]),m.predict(t.iloc[:20][CAT+nums]))
comparisons=[];rng=np.random.default_rng(440)
for seed in [440,917]:
 p=pd.read_csv(root/f'results_seed{seed}/oof_predictions.csv')
 p['improvement']=abs(p.baseline-p.total_tokens)-abs(p.rich_boost_absolute-p.total_tokens)
 group=p.groupby('group_id').improvement.mean().to_numpy()
 boot=np.array([rng.choice(group,size=len(group),replace=True).mean() for _ in range(2000)])
 base=abs(p.baseline-p.total_tokens).mean();new=abs(p.rich_boost_absolute-p.total_tokens).mean()
 comparisons.append(dict(split_seed=seed,baseline_mae=base,new_mae=new,relative_mae_reduction_pct=100*(base-new)/base,task_weighted_mean_reduction=group.mean(),task_resampling_interval_95=np.quantile(boot,[.025,.975]).tolist()))
pairs=t.pivot(index=['group_id','model'],columns='seed',values='total_tokens').dropna();gap=abs(pairs.iloc[:,0]-pairs.iloc[:,1]);assert (gap==0).all()
# Check that post-run columns cannot change extracted predictors.
raw=json.loads((ROOT/'data/ml_feature_provenance/training_task_inputs.json').read_text());a=task_features(raw[:2]);aug=[{**r,'requests':999,'total_tokens':999999,'termination':'spoof','quality_score':-999} for r in raw[:2]];pd.testing.assert_frame_equal(a,task_features(aug))
freeze=dict(status='Development configuration selected; official calibration pending; confirmation sealed',target='Cumulative total processed tokens under fixed six-request bounded controller',point_parameters=point_params,budget_parameters=budget_params,features=CAT+nums,feature_code_sha256=hashlib.sha256((root/'improve_predictions.py').read_bytes()).hexdigest(),training_rows=len(t),task_groups=t.group_id.nunique(),complete_seed_pairs=len(pairs),identical_seed_pair_token_totals=int((gap==0).sum()),task_inputs_sha256=taskhash,data_hashes=hashes,comparisons=comparisons,calibration_outcomes_used=False,confirmation_outcomes_used=False,missing_totals_imputed=False,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,checks=['Data hashes verified','All fit rows are training','Group folds disjoint','Post-run columns ignored by feature extraction','Saved model predictions round-trip unchanged'])
(out/'development_freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(comparisons,indent=2))
