"""Immutable assignments and exclusive, persistent shard claims."""
import hashlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

MODELS=['qwen2.5-coder:7b','qwen2.5-coder:14b']
SEEDS=[440,441]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2)

def assign_runs(tasks):
    if len(tasks)!=3000 or len({t['group_id'] for t in tasks})!=3000:raise ValueError('Require 3000 unique groups')
    if len({t['task_id'] for t in tasks})!=3000:raise ValueError('Duplicate tasks')
    if any(t['review_status']!='approved' or t.get('relation_review_pending') for t in tasks):raise ValueError('Reviews incomplete')
    families=defaultdict(list)
    for t in tasks:families[t['family']].append(t)
    parts={p:[] for p in ('training','calibration','confirmation')}
    # Proportional family split; fill rounding residuals deterministically.
    residual=[]
    for family,rows in sorted(families.items()):
        rows=sorted(rows,key=lambda t:digest(('partition-440:'+t['group_id']).encode()))
        a,b=int(len(rows)*.7),int(len(rows)*.15)
        parts['training']+=rows[:a];parts['calibration']+=rows[a:a+b];residual+=rows[a+b:]
    targets={'training':2100,'calibration':450,'confirmation':450}
    for p in parts:
        n=targets[p]-len(parts[p]);parts[p]+=residual[:n];residual=residual[n:]
    runs=[];shards={}
    for p,rows in parts.items():
        rows=sorted(rows,key=lambda t:digest(('order:'+t['group_id']).encode()))
        width=420 if p=='training' else 450
        for mi,model in enumerate(MODELS):
            for seed in SEEDS:
                for start in range(0,len(rows),width):
                    sid=f'{p}-m{mi}-s{seed}-{start//width:02d}'
                    shards[sid]=dict(shard_id=sid,partition=p,model=model,seed=seed)
                    for t in rows[start:start+width]:
                        rid=digest((t['task_id']+'|'+model+'|'+str(seed)).encode())[:24]
                        runs.append(dict(run_id=rid,task_id=t['task_id'],group_id=t['group_id'],shard_id=sid,partition=p,model=model,seed=seed,family=t.get('family','unknown'),source_id=t.get('source_id','unknown'),complexity=t.get('complexity','unrated'),size_band=t.get('size_band','unknown')))
    return dict(schema=1,runs=runs,shards=shards)

def collector_code(project,include_shell=False):
    """Hash executable project source; attempt outputs/backups are evidence."""
    project=Path(project)
    paths=list(project.glob('*.py'))
    for name in ('prepare','runtime','scripts','tests'):
        paths.extend((project/name).rglob('*.py'))
    if include_shell:paths.extend((project/'scripts').glob('*.sbatch'))
    return {str(p.relative_to(project)):digest(p.read_bytes()) for p in sorted(set(paths)) if p.is_file()}

def input_fingerprint(smoke=False):
    project=Path(__file__).resolve().parents[1]
    names=['sources/registry.json','sources/snapshots/tables_manifest.json','data/exclusions.json','data/candidates.json','data/tokenizer_registry.json','config.json']
    if not smoke:names+=['data/reviews.json','data/relations.json','data/group_review.json','smoke/fingerprint_fix_receipt.json','maintenance/fingerprint_fix_v1/original/prepare/manifest.py','maintenance/fingerprint_fix_v1/original/runtime/smoke.py']
    result={}
    for name in names:
        p=project/name
        if not p.exists():continue
        raw=p.read_bytes()
        if smoke and name=='data/tokenizer_registry.json':
            data=json.loads(raw)
            for entry in data.values():entry.pop('verified',None)
            raw=json.dumps(data,sort_keys=True).encode()
        result[name]=digest(raw)
    registry=read(project/'data/tokenizer_registry.json')
    for entry in registry.values():
        path=project/entry['path'];resolved=path.resolve()
        if not resolved.is_relative_to(project):raise ValueError('Tokenizer path outside project')
        result[str(resolved.relative_to(project))]=digest(resolved.read_bytes())
    # Registry byte hashes plus raw snapshots/table bytes pin source provenance.
    for entry in read(project/'sources/registry.json').values():
        path=(project/entry['path']).resolve()
        if not path.is_relative_to(project):raise ValueError('Source path outside project')
        result[str(path.relative_to(project))]=digest(path.read_bytes())
    table_manifest=project/'sources/snapshots/tables_manifest.json'
    if table_manifest.exists():
        for name,entry in read(table_manifest).items():
            if 'sha256' in entry:
                p=project/'sources/snapshots/tables'/name;result[str(p.relative_to(project))]=digest(p.read_bytes())
    return result

def freeze_campaign(tasks,config,destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    manifest=assign_runs(tasks)
    write(root/'tasks.json',tasks);write(root/'config.json',config)
    manifest['files']={n:digest((root/n).read_bytes()) for n in ('tasks.json','config.json')}
    project=Path(__file__).resolve().parents[1]
    manifest['inputs']=input_fingerprint()
    manifest['code']=collector_code(project,include_shell=True)
    write(root/'manifest.json',manifest)
    write(root/'freeze.json',{'manifest_sha256':digest((root/'manifest.json').read_bytes())})
    return manifest

def validate_manifest(root):
    root=Path(root);m=read(root/'manifest.json')
    if digest((root/'manifest.json').read_bytes())!=read(root/'freeze.json')['manifest_sha256']:raise ValueError('Manifest changed')
    for n,h in m['files'].items():
        if digest((root/n).read_bytes())!=h:raise ValueError('Frozen file changed: '+n)
    project=Path(__file__).resolve().parents[1]
    if m.get('inputs')!=input_fingerprint():raise ValueError('Preparation/tokenizer inputs changed')
    if m['code']!=collector_code(project,include_shell=True):raise ValueError('Code changed')
    expected=assign_runs(read(root/'tasks.json'))
    if m['runs']!=expected['runs'] or m['shards']!=expected['shards']:raise ValueError('Assignments changed')
    return m

def shard_path(root,shard):
    side='sealed_confirmation' if shard['partition']=='confirmation' else 'visible'
    return Path(root)/'outputs'/side/shard['shard_id']

@contextmanager
def claim_shard(root,shard_id):
    if not shard_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in shard_id):raise ValueError('Invalid shard ID')
    path=Path(root)/'locks'/shard_id;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump({'pid':os.getpid(),'job':os.environ.get('SLURM_JOB_ID')},f)
    try:yield
    finally:path.unlink()
