#!/usr/bin/env python3
"""Isolated V2 development pilot orchestration. Never execute inference on login nodes."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,hashlib,io,json,os,re,shutil,subprocess,time,zipfile
from datetime import datetime
from pathlib import Path
from prepare.manifest import ROOT,read,write,digest,validate_manifest,shard_path
from runtime.audit import audit_shard

def hash_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def manifest():return validate_manifest(ROOT/'frozen')

def discover_assets(config):
 receipt=read(ROOT/'provenance/source_receipt.json');source=Path(receipt['source_project']);parent=source.parent
 binaries=[];images=[];models=[]
 default_models=Path.home()/'.ollama/models'
 if default_models.is_dir():models.append(default_models)
 candidates=[parent,Path.home()/'.local/bin',Path.home()/'.ollama']
 seen=set()
 for top in candidates:
  if not top.is_dir():continue
  for base,dirs,files in os.walk(top,followlinks=False):
   path=Path(base)
   if (path/'blobs').is_dir() and (path/'manifests').is_dir():models.append(path)
   depth=len(path.relative_to(top).parts)
   dirs[:]=[d for d in dirs if depth<5 and d not in {'frozen','outputs','sources','data','logs','__pycache__','.git','blobs','manifests','node_modules','site-packages'} and not d.startswith(('.venv','venv','truecost_analysis_venv'))]
   for name in files:
    p=path/name
    if p in seen:continue
    seen.add(p)
    if (name=='ollama' or name.startswith('ollama-')) and os.access(p,os.X_OK):
     if hash_file(p)==config['runtime_binary_sha256']:binaries.append(p.resolve())
    elif name.endswith('.sif') and hash_file(p)==config['image_sha256']:images.append(p.resolve())
 def matches_models(folder):
  for model,wanted in config['model_digests'].items():
   name,tag=model.rsplit(':',1);p=folder/'manifests/registry.ollama.ai/library'/name/tag
   if not p.is_file() or digest(p.read_bytes())!=wanted:return False
  return True
 models=[p.resolve() for p in models if matches_models(p)]
 found={}
 for key,paths in [('ROAR_OLLAMA_BIN',binaries),('ROAR_TOOL_IMAGE',images),('OLLAMA_MODELS',models)]:
  paths=sorted(set(paths))
  if len(paths)==1:found[key]=str(paths[0])
 return found

def runtime_paths(config):
 saved=read(ROOT/'runtime_paths.json') if (ROOT/'runtime_paths.json').exists() else {}
 values={k:os.environ.get(k) or saved.get(k) for k in ['ROAR_OLLAMA_BIN','ROAR_TOOL_IMAGE','OLLAMA_MODELS']}
 if not all(values.values()):
  print('Locating hash-matching runtime assets in the existing Roar workspace...',flush=True)
  found=discover_assets(config)
  values={k:v or found.get(k) for k,v in values.items()}
 missing=[k for k,v in values.items() if not v]
 if missing:raise ValueError('Could not uniquely locate '+', '.join(missing)+'. Set these environment variables to the staged original assets, then rerun preflight.')
 values={k:str(Path(v).expanduser().resolve()) for k,v in values.items()}
 if hash_file(values['ROAR_OLLAMA_BIN'])!=config['runtime_binary_sha256']:raise ValueError('Ollama binary hash mismatch')
 if not os.access(values['ROAR_OLLAMA_BIN'],os.X_OK):raise ValueError('Ollama binary is not executable')
 if hash_file(values['ROAR_TOOL_IMAGE'])!=config['image_sha256']:raise ValueError('Tool image hash mismatch')
 if not Path(values['OLLAMA_MODELS']).is_dir():raise ValueError('Model cache directory missing')
 for model,wanted in config['model_digests'].items():
  name,tag=model.rsplit(':',1);mp=Path(values['OLLAMA_MODELS'])/'manifests/registry.ollama.ai/library'/name/tag
  if not mp.is_file() or hash_file(mp)!=wanted:raise ValueError('Pinned model cache manifest missing or changed: '+model)
 if not (ROOT/'runtime_paths.json').exists():write(ROOT/'runtime_paths.json',values)
 return values

def preflight(check_admission=True):
 m=manifest();cfg=read(ROOT/'frozen/config.json')
 if sys.version_info<(3,9):raise ValueError('Python >=3.9 required')
 try:import tokenizers
 except ImportError as e:raise ValueError('This Python environment lacks tokenizers. Activate the original collection environment before running preflight.') from e
 assets=runtime_paths(cfg)
 if not shutil.which('apptainer'):raise ValueError('apptainer is not on PATH; load the original container module/environment')
 if check_admission:
  from prepare.token_counts import admit_task
  counts=[]
  for task in read(ROOT/'frozen/tasks.json'):
   for model in cfg['models']:
    entry=admit_task(task,model,dict(context=16384,output=1024))
    if not entry['admitted'] or entry['size_band']!=task['size_band']:raise ValueError('Pilot context admission/band mismatch: '+task['task_id'])
    counts.append(dict(task_id=task['task_id'],model=model,**entry))
  admission=dict(manifest_sha256=digest((ROOT/'frozen/manifest.json').read_bytes()),all_admitted=True,checks=counts)
  path=ROOT/'preflight_admissions.json'
  if path.exists():
   if read(path)!=admission:raise ValueError('Admission result changed')
  else:write(path,admission)
 os.environ.update(assets)
 return m,assets

def audit_all(m):
 audits=[];reports=[]
 for sid,shard in sorted(m['shards'].items()):
  a=audit_shard(m,shard_path(ROOT/'frozen',shard));reports.extend(a['reports']);audits.append(dict(shard=sid,planned=a['planned'],complete=a['complete'],pending=a['pending']))
 return audits,reports

def actual_job_end():
 value=os.environ.get('SLURM_JOB_END_TIME','')
 if value.isdigit():return int(value)
 text=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
 match=re.search(r'\bEndTime=(\S+)',text)
 if not match or match.group(1) in ('Unknown','None','N/A'):raise ValueError('Cannot establish actual Slurm deadline; stopped before inference')
 return datetime.fromisoformat(match.group(1)).timestamp()

def submit(resume=False):
 m,assets=preflight();audit_all(m)
 if (ROOT/'frozen/PAUSE_REQUESTED').exists():raise ValueError('Pause marker exists; inspect and remove it explicitly before resuming')
 receipt=ROOT/'submission.json'
 if receipt.exists():
  if not resume:raise ValueError('Already submitted; use status, or submit --resume after the old array has ended')
  prior=read(receipt);query=subprocess.run(['squeue','-h','-j',prior['job_id']],text=True,capture_output=True)
  if query.returncode and 'invalid job id' not in query.stderr.lower():raise RuntimeError('Could not establish previous job state: '+query.stderr)
  if query.returncode==0 and query.stdout.strip():raise ValueError('Previous array is still queued/running; no duplicate submission')
 audits,_=audit_all(m);pending={a['shard'] for a in audits if a['pending']}
 if not pending:print('All 128 assignments already complete; nothing submitted.');return
 ids=sorted(m['shards']);indices=[str(i) for i,sid in enumerate(ids) if sid in pending]
 command=['sbatch','--parsable','--account=open','--array='+','.join(indices)+'%2','scripts/pilot.sbatch']
 env=dict(os.environ,**assets,TRUECOST_PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1')
 # Keep history before writing another successful receipt. Existing evidence is never removed.
 text=subprocess.check_output(command,cwd=ROOT,env=env,text=True).strip();job_id=text.split(';')[0]
 if not job_id.isdigit():raise ValueError('Unexpected sbatch response: '+text)
 entry=dict(job_id=job_id,command=command,python=sys.executable,submitted_at=time.time(),manifest_sha256=digest((ROOT/'frozen/manifest.json').read_bytes()))
 history=ROOT/'submission_history';history.mkdir(exist_ok=True);write(history/(job_id+'.json'),entry)
 receipt.write_text(json.dumps(entry,indent=2))
 print('Submitted V2 development pilot job '+job_id+'; at most two V100 tasks concurrently.')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','submit','collect','status','export']);p.add_argument('--index',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
 os.chdir(ROOT)
 if a.action=='preflight':
  m,paths=preflight();print(json.dumps(dict(status='ready',groups=32,attempts=len(m['runs']),development_only=True,runtime_paths=paths),indent=2))
 elif a.action=='submit':submit(a.resume)
 elif a.action=='collect':
  if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('submit'):raise ValueError('Inference only on allocated Slurm compute nodes')
  m,_=preflight();ids=sorted(m['shards'])
  if a.index is None or not 0<=a.index<len(ids):raise ValueError('Invalid array index')
  from scripts.submit import collect_shard
  print(json.dumps(collect_shard(ROOT/'frozen',ids[a.index],actual_job_end()),indent=2))
 elif a.action=='status':
  m=manifest();rows=[]
  for sid,s in sorted(m['shards'].items()):
   try:
    audit=audit_shard(m,shard_path(ROOT/'frozen',s));rows.append(dict(shard=sid,planned=audit['planned'],complete=audit['complete'],pending=audit['pending'],audit='passed'))
   except (ValueError,OSError,KeyError) as e:rows.append(dict(shard=sid,audit='blocked',reason=str(e)))
  print(json.dumps(rows,indent=2));print('Blocked attempts must be inspected; do not delete or retry their evidence.')
 else:
  m=manifest();audits,reports=audit_all(m)
  if len(reports)!=128 or any(a['pending'] for a in audits):raise ValueError('Export requires all 128 complete audited assignments; inspect status/logs for partial or failed work')
  export=ROOT/'TrueCost_V2_Pilot_Results.zip'
  if export.exists():raise FileExistsError('Results archive already exists; reuse it')
  fields=['run_id','task_id','group_id','shard_id','partition','model','seed','family','source_id','size_band','termination','requests','input_tokens','output_tokens','total_tokens','initial_input_tokens','max_request_input_tokens','source_bearing_input_tokens','telemetry_complete','wall_seconds','quality_metric','quality_check_passed','quality_score']
  stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(reports)
  raw=stream.getvalue().encode();summary=dict(groups=32,planned=128,complete=128,pending=0,partition='development',strict_audit_passed=True,manifest_sha256=digest((ROOT/'frozen/manifest.json').read_bytes()),csv_sha256=digest(raw),shards=audits,excluded_from_future_final_evaluation=True)
  with zipfile.ZipFile(export,'x',zipfile.ZIP_DEFLATED) as z:
   z.writestr('audited_pilot_runs.csv',raw);z.writestr('audit_summary.json',json.dumps(summary,indent=2))
   files=[ROOT/'frozen/manifest.json',ROOT/'frozen/tasks.json',ROOT/'frozen/config.json',ROOT/'frozen/freeze.json',ROOT/'provenance/selection_receipt.json',ROOT/'provenance/source_receipt.json',ROOT/'provenance/development_exclusions.json',ROOT/'data/tokenizer_registry.json']
   files+=list((ROOT/'frozen/outputs').rglob('*'))
   hashes={}
   for path in files:
    if not path.is_file():continue
    if not path.resolve().is_relative_to(ROOT):raise ValueError('Unexpected external evidence link')
    n=str(path.relative_to(ROOT));hashes[n]=digest(path.read_bytes());z.write(path,n)
   z.writestr('evidence_hashes.json',json.dumps(hashes,indent=2))
  print('EXPORTED: '+str(export))
if __name__=='__main__':main()
