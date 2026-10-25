"""V2 pilot-only freeze: 32 groups, 128 development attempts; no V1 writes."""
import hashlib,json,os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=['qwen2.5-coder:7b','qwen2.5-coder:14b'];SEEDS=[440,441]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2)
def assign_runs(tasks):
 if len(tasks)!=32 or len({t['group_id'] for t in tasks})!=32 or len({t['task_id'] for t in tasks})!=32:raise ValueError('Pilot requires 32 distinct task groups')
 families={'classification','coding','extraction','grounded_qa','ideation','math','summarization','tabular'}
 if Counter(t['family'] for t in tasks)!=Counter({f:4 for f in families}):raise ValueError('Four groups per family required')
 if any(t['review_status']!='approved' or t.get('relation_review_pending') or t.get('excluded_prior') for t in tasks):raise ValueError('Unreviewed or excluded task')
 tasks=sorted(tasks,key=lambda t:digest(('v2-pilot-order:'+t['group_id']).encode()));runs=[];shards={}
 for mi,model in enumerate(MODELS):
  for seed in SEEDS:
   sid=f'development-m{mi}-s{seed}-00';shards[sid]=dict(shard_id=sid,partition='development',model=model,seed=seed)
   for t in tasks:
    rid=digest(('v2-development-pilot|'+t['task_id']+'|'+model+'|'+str(seed)).encode())[:24]
    runs.append(dict(run_id=rid,task_id=t['task_id'],group_id=t['group_id'],shard_id=sid,partition='development',model=model,seed=seed,family=t['family'],source_id=t['source_id'],complexity=t.get('complexity','unrated'),size_band=t['size_band']))
 return dict(schema=1,study='truecost-v2-development-pilot',runs=runs,shards=shards)
def collector_code(project,include_shell=True):
 p=Path(project);files=list(p.glob('*.py'))
 for name in ('prepare','runtime','scripts'):files.extend((p/name).rglob('*.py'))
 files.extend((p/'scripts').glob('*.sbatch'))
 return {str(f.relative_to(p)):digest(f.read_bytes()) for f in sorted(set(files)) if f.is_file()}
def input_fingerprint():
 registry=read(ROOT/'data/tokenizer_registry.json');names=['data/tokenizer_registry.json','provenance/selection_receipt.json','provenance/source_receipt.json','provenance/development_exclusions.json']
 for model in MODELS:
  entry=registry[model]
  if not entry.get('verified'):raise ValueError('Unverified source tokenizer')
  path=(ROOT/entry['path']).resolve()
  if not path.is_relative_to(ROOT):raise ValueError('Copied tokenizer outside pilot')
  if digest(path.read_bytes())!=entry['sha256']:raise ValueError('Tokenizer bytes changed')
  names.append(str(path.relative_to(ROOT)))
 return {n:digest((ROOT/n).read_bytes()) for n in sorted(set(names))}
def freeze_pilot(tasks,config):
 root=ROOT/'frozen';root.mkdir(exist_ok=False);m=assign_runs(tasks)
 write(root/'tasks.json',tasks);write(root/'config.json',config)
 m['files']={n:digest((root/n).read_bytes()) for n in ['tasks.json','config.json']};m['inputs']=input_fingerprint();m['code']=collector_code(ROOT)
 write(root/'manifest.json',m);write(root/'freeze.json',{'manifest_sha256':digest((root/'manifest.json').read_bytes())});return m
def validate_manifest(root):
 root=Path(root).resolve()
 if root!=ROOT/'frozen':raise ValueError('Pilot validator cannot operate on another frozen study')
 m=read(root/'manifest.json')
 if digest((root/'manifest.json').read_bytes())!=read(root/'freeze.json')['manifest_sha256']:raise ValueError('Pilot manifest changed')
 for n,h in m['files'].items():
  if digest((root/n).read_bytes())!=h:raise ValueError('Frozen pilot file changed: '+n)
 if m['inputs']!=input_fingerprint() or m['code']!=collector_code(ROOT):raise ValueError('Pilot inputs or code changed')
 expected=assign_runs(read(root/'tasks.json'))
 if any(m[k]!=expected[k] for k in ['study','runs','shards']):raise ValueError('Pilot assignments changed')
 return m
def shard_path(root,shard):return Path(root)/'outputs'/'visible'/shard['shard_id']
@contextmanager
def claim_shard(root,shard_id):
 if not shard_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in shard_id):raise ValueError('Invalid shard ID')
 path=Path(root)/'locks'/shard_id;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump({'pid':os.getpid(),'job':os.environ.get('SLURM_JOB_ID')},f)
 try:yield
 finally:path.unlink()
