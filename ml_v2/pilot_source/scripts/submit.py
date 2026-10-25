"""Validated array commands and clean continuation; no shell interpolation."""
import os
import time
from pathlib import Path
from prepare.manifest import validate_manifest,read,claim_shard,shard_path,write,digest

def submission_command(manifest,shard_ids,max_parallel=2):
    if not shard_ids or len(shard_ids)!=len(set(shard_ids)) or not 1<=max_parallel<=2:raise ValueError('Invalid array selection/throttle')
    if set(shard_ids)-set(manifest['shards']):raise ValueError('Unknown shard')
    # Indices refer to the complete sorted shard list, including subset resubmissions.
    all_ids=sorted(manifest['shards']);indices=sorted(all_ids.index(s) for s in shard_ids)
    expression=f'0-{len(indices)-1}' if indices==list(range(len(indices))) and len(indices)>1 else ','.join(map(str,indices))
    return ['sbatch',f'--array={expression}%{max_parallel}','scripts/collect.sbatch']

def should_continue(now,job_end):return job_end-now>=2100

def resumable_runs(runs,root):
    todo=[]
    for run in runs:
        p=Path(root)/run['run_id']
        if p.exists() and not (p/'report.json').is_file():raise ValueError('Incomplete attempt requires audit: '+run['run_id'])
        if not p.exists():todo.append(run)
    return todo

def check_residency(value,model):
    rows=[r for r in value.get('models',[]) if r.get('model',r.get('name'))==model]
    if len(rows)!=1 or rows[0].get('size_vram',0)<rows[0].get('size',1) or rows[0].get('context_length')!=16384:raise ValueError('Full GPU residency/context required')
    return rows[0]

def collect_shard(root,shard_id,job_end_epoch=None):
    import subprocess
    from runtime.server import owned_server,api
    from runtime.worker import run_attempt
    from runtime.transport import invoke_local
    from runtime.audit import audit_shard
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Collection requires allocated Slurm GPU node')
    host=os.uname().nodename
    if host.startswith('submit'):raise ValueError('Inference on login nodes forbidden')
    root=Path(root).resolve();m=validate_manifest(root);shard=m['shards'][shard_id];out=shard_path(root,shard);out.mkdir(parents=True,exist_ok=True)
    config=read(root/'config.json');tasks={t['task_id']:t for t in read(root/'tasks.json')}
    config['image']=os.environ['ROAR_TOOL_IMAGE']
    if digest(Path(os.environ['ROAR_OLLAMA_BIN']).read_bytes())!=config['runtime_binary_sha256']:raise ValueError('Runtime binary changed')
    if digest(Path(config['image']).read_bytes())!=config['image_sha256']:raise ValueError('Container hash changed')
    audit_shard(m,out)
    runs=[r for r in m['runs'] if r['shard_id']==shard_id]
    job_end=job_end_epoch or time.time()+4*3600
    with claim_shard(root,shard_id):
        todo=resumable_runs(runs,out/'runs')
        if not todo:return {'status':'complete','completed':len(runs)}
        session=out/('session-'+os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()));session.mkdir()
        with owned_server(os.environ['ROAR_OLLAMA_BIN'],os.environ['OLLAMA_MODELS'],session/'runtime.log') as endpoint:
            tags=api(endpoint,'/api/tags')['models']
            found=[x for x in tags if x['name']==shard['model']]
            if len(found)!=1 or found[0]['digest']!=config['model_digests'][shard['model']]:raise ValueError('Model digest mismatch')
            warm=session/'warmup';warm.mkdir()
            payload=dict(model=shard['model'],messages=[{'role':'user','content':'Reply OK.'}],stream=False,keep_alive='30m',options={'num_ctx':16384,'num_predict':1,'seed':shard['seed'],'temperature':0})
            invoke_local(payload,warm,480,endpoint);check_residency(api(endpoint,'/api/ps'),shard['model'])
            config['invoke']=lambda p,e,t:invoke_local(p,e,t,endpoint)
            config['residency']=lambda:check_residency(api(endpoint,'/api/ps'),shard['model'])
            done=0
            for run in todo:
                if (root/'PAUSE_REQUESTED').exists() or not should_continue(time.time(),job_end):return {'status':'paused','completed_this_job':done}
                run_attempt(dict(run,task=tasks[run['task_id']]),config,out/'runs'/run['run_id'])
                check_residency(api(endpoint,'/api/ps'),shard['model']);done+=1
                # Confirmation never logs per-run metrics or answers.
                print(__import__('json').dumps({'shard':shard_id,'completed_this_job':done}),flush=True)
    return {'status':'complete','completed':len(runs)}
