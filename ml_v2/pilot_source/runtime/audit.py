"""Manifest-driven evidence reconciliation; holdout metrics stay sealed."""
import csv
import json
from pathlib import Path
from prepare.manifest import read,digest,validate_manifest,shard_path
from runtime.transport import validate_response

def audit_shard(manifest,shard_root):
    root=Path(shard_root);sid=root.name
    if sid not in manifest['shards']:raise ValueError('Unknown shard path')
    expected={r['run_id']:r for r in manifest['runs'] if r['shard_id']==sid}
    found={p.name:p for p in (root/'runs').iterdir()} if (root/'runs').exists() else {}
    if set(found)-set(expected):raise ValueError('Unexpected run IDs')
    reports=[]
    for rid,path in found.items():
        if not path.is_dir() or not (path/'report.json').is_file() or (path/'failure.json').exists():raise ValueError('Incomplete attempt: '+rid)
        report=read(path/'report.json');assignment=read(path/'assignment.json')
        for k,v in expected[rid].items():
            if report.get(k)!=v or assignment.get(k)!=v:raise ValueError('Run metadata mismatch: '+k)
        ledger=[json.loads(line) for line in (path/'ledger.jsonl').read_text().splitlines()]
        if not 1<=len(ledger)<=6 or [r['index'] for r in ledger]!=list(range(1,len(ledger)+1)):raise ValueError('Ledger request sequence mismatch')
        if sorted(p.name for p in path.glob('request-*'))!=[f'request-{i:02d}' for i in range(1,len(ledger)+1)]:raise ValueError('Unexpected request evidence')
        for row in ledger:
            q=path/f"request-{row['index']:02d}"
            for name,key in [('raw.json','raw_sha256'),('request.json','request_sha256')]:
                if digest((q/name).read_bytes())!=row[key]:raise ValueError('Raw/request evidence changed')
            raw=validate_response(read(q/'raw.json'),expected[rid]['model']);payload=read(q/'request.json')
            if payload['model']!=expected[rid]['model']:raise ValueError('Payload model mismatch')
            opts=payload['options']
            for k,v in dict(seed=expected[rid]['seed'],num_ctx=16384,num_predict=1024,temperature=0).items():
                if opts.get(k)!=v:raise ValueError('Payload configuration mismatch')
            if (raw['prompt_eval_count'],raw['eval_count'])!=(row['input_tokens'],row['output_tokens']):raise ValueError('Raw counters differ from ledger')
        sent=sum(r['input_tokens'] for r in ledger);received=sum(r['output_tokens'] for r in ledger)
        if (report.get('input_tokens'),report.get('output_tokens'),report.get('total_tokens'),report.get('requests'),report.get('telemetry_complete'))!=(sent,received,sent+received,len(ledger),True):raise ValueError('Report/ledger mismatch')
        for tool in path.glob('tool-*'):
            if not (tool/'result.json').exists():raise ValueError('Incomplete tool evidence')
            result=read(tool/'result.json');command=read(tool/'command.json');stdout=(tool/'stdout.bin').read_bytes()
            if digest(stdout)!=result['stdout_sha256'] or digest((tool/'command.json').read_bytes())!=result['command_sha256']:raise ValueError('Tool evidence changed')
            if result['started']!=((result['startup_marker']+'\n').encode() in stdout):raise ValueError('Tool startup proof mismatch')
        reports.append(report)
    return dict(planned=len(expected),complete=len(reports),pending=len(expected)-len(reports),reports=reports)

def visible_records(rows):
    result=[r for r in rows if r['partition'] in ('training','calibration')]
    if len(result)!=len({r['run_id'] for r in result}):raise ValueError('Duplicate visible run IDs')
    return result

def aggregate_visible(root):
    root=Path(root);m=validate_manifest(root);rows=[];complete=planned=0
    for shard in m['shards'].values():
        if shard['partition']=='confirmation':continue
        a=audit_shard(m,shard_path(root,shard));rows+=a['reports'];complete+=a['complete'];planned+=a['planned']
    rows=visible_records(rows);target=root/'outputs/visible/audited_runs.csv';target.parent.mkdir(parents=True,exist_ok=True)
    fields=['family','source_id','complexity','size_band','source_bearing_input_tokens','run_id','task_id','group_id','partition','model','seed','termination','requests','input_tokens','output_tokens','total_tokens','initial_input_tokens','max_request_input_tokens','quality_metric','quality_check_passed','quality_score','wall_seconds']
    with target.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');out.writeheader();out.writerows(rows)
    return dict(planned=planned,complete=complete,pending=planned-complete,csv=str(target))

def audit_confirmation_sealed(root):
    root=Path(root);m=validate_manifest(root);planned=complete=0
    for shard in m['shards'].values():
        if shard['partition']!='confirmation':continue
        a=audit_shard(m,shard_path(root,shard));planned+=a['planned'];complete+=a['complete']
    return dict(planned=planned,complete=complete,pending=planned-complete,scope='integrity/completeness only')
