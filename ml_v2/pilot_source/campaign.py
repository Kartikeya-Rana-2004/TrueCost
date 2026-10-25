#!/usr/bin/env python3
"""Operator CLI: preparation, bounded collection and sealed auditing. No ML."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from prepare.manifest import read,write,freeze_campaign,validate_manifest,digest
ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'frozen'

def reviewed_selection():
    from prepare.grouping import group_candidates,select_tasks
    tasks=read(ROOT/'data/candidates.json');reviews=read(ROOT/'data/reviews.json') if (ROOT/'data/reviews.json').exists() else {}
    relations=read(ROOT/'data/relations.json') if (ROOT/'data/relations.json').exists() else []
    if not (ROOT/'data/group_review.json').exists():raise ValueError('Group/near-duplicate review missing')
    group_review=read(ROOT/'data/group_review.json')
    if not group_review.get('reviewer') or not group_review.get('rationale') or group_review.get('candidates_sha256')!=digest((ROOT/'data/candidates.json').read_bytes()) or group_review.get('relations_sha256')!=digest(json.dumps(relations,sort_keys=True).encode()):raise ValueError('Group review is missing or stale')
    for task in tasks:
        review=reviews.get(task['task_id'],{})
        h=digest(json.dumps(task,sort_keys=True).encode())
        if review.get('decision')=='approved' and review.get('task_sha256')==h and review.get('reviewer') and review.get('rationale'):
            task['review_status']='approved';task['complexity']=review.get('complexity','unrated');task['complexity_rationale']=review.get('complexity_rationale','')
        if not task['candidate_admitted']:task['review_status']='rejected'
    # Smoke groups are development only and cannot enter the research split.
    smoke_task_ids={t['task_id'] for t in read(ROOT/'smoke/tasks.json')}
    for task in tasks:
        if task['task_id'] in smoke_task_ids:task['excluded_prior']=True
    tasks=group_candidates(tasks,relations)
    return select_tasks(tasks,read(ROOT/'config.json')['quotas'],set())

def verify_sources():
    from prepare.sources import verify_snapshot
    for entry in read(ROOT/'sources/registry.json').values():verify_snapshot(ROOT/entry['path'],entry)
    tables=read(ROOT/'sources/snapshots/tables_manifest.json')
    for name,entry in tables.items():
        if 'sha256' in entry:verify_snapshot(ROOT/'sources/snapshots/tables'/name,entry)

def prepare():
    from runtime.smoke import prepare_smoke,audit_smoke
    verify_sources();smoke=prepare_smoke()
    if not (ROOT/'data/reviews.json').exists():return dict(status='smoke ready; source/group reviews pending',smoke_runs=len(smoke['runs']),research_frozen=False)
    tasks=reviewed_selection();config=read(ROOT/'config.json');audit=audit_smoke()
    from prepare.token_counts import admit_task
    from prepare.release import release_gate
    for t in tasks:t['admissions']={m:admit_task(t,m,{'context':16384,'output':1024}) for m in config['models']}
    config['tokenizers_verified']=True;config['runtime_binary_sha256']=audit['runtime_binary_sha256'];config['smoke_audit_sha256']=digest(json.dumps(audit,sort_keys=True).encode())
    release_gate(tasks,config,audit)
    manifest=freeze_campaign(tasks,config,FROZEN)
    return dict(research_frozen=True,runs=len(manifest['runs']),shards=len(manifest['shards']))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit-smoke','run-smoke','audit-smoke','submit-research','collect-shard','progress','export-visible']);p.add_argument('--account');p.add_argument('--index',type=int);p.add_argument('--shards',nargs='+');args=p.parse_args()
    os.chdir(ROOT)
    if args.action=='prepare':result=prepare()
    elif args.action=='submit-smoke':
        from runtime.smoke import prepare_smoke
        prepare_smoke();cmd=['sbatch']+(['--account='+args.account] if args.account else [])+['scripts/smoke.sbatch'];result={'submission':subprocess.check_output(cmd,text=True).strip()}
    elif args.action=='run-smoke':
        from runtime.smoke import run_smoke
        result=run_smoke()
    elif args.action=='audit-smoke':
        from runtime.smoke import audit_smoke
        result=audit_smoke();reg=read(ROOT/'data/tokenizer_registry.json')
        for value in reg.values():value['verified']=True
        (ROOT/'data/tokenizer_registry.json').write_text(json.dumps(reg,indent=2))
        (ROOT/'smoke/audit.json').write_text(json.dumps(result,indent=2))
    elif args.action in ('submit-research','collect-shard'):
        from runtime.smoke import audit_smoke
        from prepare.release import release_gate
        m=validate_manifest(FROZEN);config=read(FROZEN/'config.json');audit=audit_smoke()
        if config['smoke_audit_sha256']!=digest(json.dumps(audit,sort_keys=True).encode()):raise ValueError('Smoke audit changed')
        release_gate(read(FROZEN/'tasks.json'),config,audit)
        if args.action=='submit-research':
            from scripts.submit import submission_command
            cmd=submission_command(m,args.shards or sorted(m['shards']))
            if args.account:cmd.insert(1,'--account='+args.account)
            result={'submission':subprocess.check_output(cmd,text=True).strip()}
        else:
            from scripts.submit import collect_shard
            ids=sorted(m['shards'])
            if args.index is None or not 0<=args.index<len(ids):raise ValueError('Invalid shard index')
            end=int(os.environ['SLURM_JOB_END_TIME']) if os.environ.get('SLURM_JOB_END_TIME','').isdigit() else None
            result=collect_shard(FROZEN,ids[args.index],end)
    else:
        from runtime.audit import aggregate_visible,audit_confirmation_sealed
        result={'visible':aggregate_visible(FROZEN),'confirmation':audit_confirmation_sealed(FROZEN)}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
