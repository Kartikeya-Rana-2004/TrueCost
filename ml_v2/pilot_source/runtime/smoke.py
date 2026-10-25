"""Separate source/band/model integration gate. No research group is consumed."""
import json
import os
import time
from pathlib import Path
from prepare.manifest import read,write,digest,input_fingerprint,claim_shard,collector_code
from prepare.token_counts import candidate_count
from runtime.worker import run_attempt
from runtime.server import owned_server,api
from runtime.transport import invoke_local
from scripts.submit import check_residency
ROOT=Path(__file__).resolve().parents[1]

def code_fingerprint():
    return collector_code(ROOT)

def verify_smoke_code(manifest):
    current=code_fingerprint()
    if current==manifest['code']:return
    receipt_path=ROOT/'smoke/fingerprint_fix_receipt.json'
    if not receipt_path.is_file():raise ValueError('Smoke code changed; verified fingerprint migration required')
    receipt=read(receipt_path)
    if (receipt.get('schema')!=1 or receipt.get('original_manifest_sha256')!=digest((ROOT/'smoke/manifest.json').read_bytes())
        or receipt.get('before_code')!=manifest['code'] or receipt.get('after_code')!=current):
        raise ValueError('Fingerprint migration/code mismatch')
    changed={p for p in set(current)|set(manifest['code']) if current.get(p)!=manifest['code'].get(p)}
    allowed={'prepare/manifest.py','runtime/smoke.py','tests/test_fingerprint.py'}
    if changed!=allowed:raise ValueError('Migration contains unrelated source changes')
    for name in ('prepare/manifest.py','runtime/smoke.py'):
        backup=ROOT/'maintenance/fingerprint_fix_v1/original'/name
        if not backup.is_file() or digest(backup.read_bytes())!=manifest['code'][name]:raise ValueError('Original experiment code backup differs')

def prepare_smoke():
    rows=read(ROOT/'data/candidates.json');eligible=[t for t in rows if t['candidate_admitted'] and not t['excluded_prior']]
    # Cover adapter x populated source-bearing band, plus every family.
    selected={}
    for t in sorted(eligible,key=lambda t:digest(('smoke:'+t['task_id']).encode())):
        key=(t['source_id'],t['family'],t['size_band'])
        if key not in selected:selected[key]=t
    if not any(t['size_band']=='4097-8192' for t in selected.values()):raise ValueError('No genuinely long smoke source')
    tasks=list(selected.values());runs=[]
    for i,t in enumerate(tasks):
        for model in read(ROOT/'config.json')['models']:
            runs.append(dict(run_id=digest(('smoke:'+t['task_id']+model).encode())[:24],task_id=t['task_id'],group_id=t['group_id'],model=model,seed=440,partition='development',shard_id='smoke',source_id=t['source_id'],family=t['family'],size_band=t['size_band']))
    manifest=dict(schema=1,inputs=input_fingerprint(smoke=True),runs=runs,shards={'smoke':{'shard_id':'smoke','partition':'development'}},code=code_fingerprint(),tasks_sha256=digest(json.dumps(tasks,sort_keys=True).encode()))
    path=ROOT/'smoke/manifest.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='code'}!={k:v for k,v in manifest.items() if k!='code'}:raise ValueError('Smoke plan changed; retain prior attempt and use a new package version')
        verify_smoke_code(old)
        return old
    write(ROOT/'smoke/tasks.json',tasks);write(path,manifest)
    return manifest

def may_start_smoke_attempt(now,job_end,paused):
    return not paused and job_end-now>=2100

def run_smoke():
    with claim_shard(ROOT/'smoke','smoke'):
        return _run_smoke()

def _run_smoke():
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('submit'):raise ValueError('Run smoke only in Slurm GPU allocation')
    manifest=read(ROOT/'smoke/manifest.json');tasks=read(ROOT/'smoke/tasks.json')
    if manifest.get('inputs')!=input_fingerprint(smoke=True):raise ValueError('Smoke preparation/tokenizer inputs changed')
    verify_smoke_code(manifest)
    if manifest['tasks_sha256']!=digest(json.dumps(tasks,sort_keys=True).encode()):raise ValueError('Smoke task change')
    config=read(ROOT/'config.json');config['image']=os.environ['ROAR_TOOL_IMAGE']
    if digest(Path(config['image']).read_bytes())!=config['image_sha256']:raise ValueError('Image hash differs')
    out=ROOT/'smoke/evidence/smoke';out.mkdir(parents=True,exist_ok=True)
    from runtime.audit import audit_shard
    from scripts.submit import resumable_runs
    audit_shard(manifest,out);todo=resumable_runs(manifest['runs'],out/'runs')
    if not todo:return {'status':'complete','completed':len(manifest['runs'])}
    job_start=time.monotonic()
    job_end=int(os.environ['SLURM_JOB_END_TIME']) if os.environ.get('SLURM_JOB_END_TIME','').isdigit() else None
    session=out/('session-'+os.environ['SLURM_JOB_ID']+'-'+str(time.time_ns()));session.mkdir()
    with owned_server(os.environ['ROAR_OLLAMA_BIN'],os.environ['OLLAMA_MODELS'],session/'runtime.log') as endpoint:
        tags={r['name']:r['digest'] for r in api(endpoint,'/api/tags')['models']}
        if any(tags.get(m)!=d for m,d in config['model_digests'].items()):raise ValueError('Model digest differs')
        metadata=dict(binary_sha256=digest(Path(os.environ['ROAR_OLLAMA_BIN']).read_bytes()),image_sha256=config['image_sha256'],models={m:api(endpoint,'/api/show',{'model':m}) for m in config['models']},code=code_fingerprint())
        write(session/'runtime_metadata.json',metadata)
        by_id={t['task_id']:t for t in tasks}
        for model in config['models']:
            warm=session/('warmup-'+model.split(':')[-1]);warm.mkdir()
            invoke_local(dict(model=model,messages=[dict(role='user',content='Reply OK.')],stream=False,options={'num_ctx':16384,'num_predict':1,'temperature':0},keep_alive='30m'),warm,480,endpoint)
            check_residency(api(endpoint,'/api/ps'),model)
            for r in [r for r in todo if r['model']==model]:
                remaining=job_end-time.time() if job_end else 4*3600-(time.monotonic()-job_start)
                if not may_start_smoke_attempt(0,remaining,(ROOT/'smoke/PAUSE_REQUESTED').exists()):return {'status':'paused','remaining_runs':len([x for x in todo if not (out/'runs'/x['run_id']/'report.json').exists()])}
                cfg=dict(config,counter=lambda messages,m=model:candidate_count(messages,m),invoke=lambda p,e,t:invoke_local(p,e,t,endpoint),residency=lambda m=model:check_residency(api(endpoint,'/api/ps'),m))
                run_attempt(dict(r,task=by_id[r['task_id']]),cfg,out/'runs'/r['run_id'])
                print(json.dumps({'smoke_completed_run':r['run_id']}),flush=True)
    return {'status':'complete','planned':len(manifest['runs'])}

def audit_smoke():
    from runtime.audit import audit_shard
    m=read(ROOT/'smoke/manifest.json');tasks=read(ROOT/'smoke/tasks.json');out=ROOT/'smoke/evidence/smoke'
    if m.get('inputs')!=input_fingerprint(smoke=True):raise ValueError('Smoke preparation/tokenizer inputs changed')
    verify_smoke_code(m)
    if m['tasks_sha256']!=digest(json.dumps(tasks,sort_keys=True).encode()):raise ValueError('Smoke task fingerprint changed')
    a=audit_shard(m,out)
    if a['pending']:raise ValueError('Smoke incomplete')
    metadata=[read(p) for p in out.glob('session-*/runtime_metadata.json')]
    if not metadata:raise ValueError('Missing runtime provenance')
    if any(x['code']!=m['code'] for x in metadata):raise ValueError('Smoke runtime code changed')
    binary={x['binary_sha256'] for x in metadata}
    if len(binary)!=1:raise ValueError('Mixed runtime binaries')
    for report in a['reports']:
        path=out/'runs'/report['run_id']
        ledger=[json.loads(x) for x in (path/'ledger.jsonl').read_text().splitlines()]
        if any(r['predicted_input_tokens']!=r['input_tokens'] for r in ledger):raise ValueError('Tokenizer/runtime mismatch; inspect retained evidence before freeze')
        if 'runtime_residency' not in report:raise ValueError('Missing full GPU residency evidence')
        check_residency({'models':[report['runtime_residency']]},report['model'])
        if report['family']=='coding':
            tools=[read(p) for p in path.glob('tool-*/result.json')]
            if not tools or not any(t.get('started') for t in tools):raise ValueError('Coding harness was not exercised')
        if report['size_band']=='4097-8192' and report['max_request_input_tokens']<4097:raise ValueError('Long source was not exercised')
    return dict(passed=True,completed=a['complete'],code=m['code'],audit_code=code_fingerprint(),runtime_binary_sha256=binary.pop(),manifest_sha256=digest((ROOT/'smoke/manifest.json').read_bytes()),scope='integration, not research outcomes')
