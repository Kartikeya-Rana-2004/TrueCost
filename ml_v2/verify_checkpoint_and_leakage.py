"""Verify V1 snapshot integrity, feature isolation, and matched-fold comparisons."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from train_feature_ablations import enhanced,HERE,V1
root=HERE.parent/'releases/v1.0.0';release=json.loads((root/'release.json').read_text())
for name,h in release['artifacts'].items():
 p=root/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
 assert p.stat().st_mode&0o222==0,name
samples=[dict(task_id='synthetic-guard',instruction='List 10 examples in a concise paragraph.',context='a | b\n1 | 2',group_id='guard',public_tests=False)]
augmented=[{**samples[0],'total_tokens':999999,'requests':6,'termination':'request_limit','quality_score':0,'output_tokens':1024,'request_cap':True,'hit_output_cap':True}]
pd.testing.assert_frame_equal(enhanced(samples),enhanced(augmented))
assert enhanced(samples).requested_quantity.iloc[0]==10
for seed in [440,917]:
 old=pd.read_csv(V1/f'results_seed{seed}/oof_predictions.csv').set_index('run_id')
 new=pd.read_csv(HERE/f'feature_results_seed{seed}/oof_predictions.csv').set_index('run_id')
 assert set(old.index)==set(new.index)
 assert new.fold.equals(old.loc[new.index,'fold'])
 assert np.allclose(new.v1_retrained,old.loc[new.index,'rich_boost_absolute'])
 assert (new.groupby('group_id').fold.nunique()==1).all()
result=dict(v1_release_hashes_verified=len(release['artifacts']),v1_snapshot_files_read_only=True,post_run_field_perturbation_leaves_features_unchanged=True,baseline_predictions_match_original_v1_cv=True,fold_assignments_match_v1=True,all_group_replicates_together=True,training_only_v2=True,promoted_candidate=None,fresh_independent_evaluation_required=True)
(HERE/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
