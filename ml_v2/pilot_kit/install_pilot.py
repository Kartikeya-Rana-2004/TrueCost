#!/usr/bin/env python3
"""Install into a NEW sibling workspace; source project is read-only input."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
from select_pilot import select,digest,read,EXPECTED
KIT=Path(__file__).resolve().parent
COPY_SOURCE=['prepare/adapters.py','prepare/token_counts.py','runtime/worker.py','runtime/tools.py','runtime/transport.py','runtime/server.py','runtime/audit.py','scripts/submit.py']
def main():
 if sys.version_info<(3,9):raise SystemExit('Activate truecost_analysis_venv; Python >=3.9 required.')
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path.home()/'truecost_roar/TrueCost_LargeScale_12000_Reviewed_v2');p.add_argument('--destination',type=Path,default=Path.home()/'truecost_roar/TrueCost_V2_Development_Pilot');a=p.parse_args();source=a.source.resolve();dest=a.destination.resolve()
 if dest.exists():raise FileExistsError('Pilot destination already exists; inspect it rather than overwrite: '+str(dest))
 if dest==source or dest.is_relative_to(source) or source.is_relative_to(dest):raise ValueError('Pilot must be a separate workspace')
 lock=read(KIT/'kit_hashes.json')
 for n,h in lock.items():
  if digest((KIT/n).read_bytes())!=h:raise ValueError('Pilot kit changed: '+n)
 tasks,selection=select(source)
 if tasks!=read(KIT/'payload/tasks.json') or selection!=read(KIT/'payload/selection_receipt.json'):raise ValueError('Selection differs from prepared kit')
 m=read(source/'frozen/manifest.json');original_config=read(source/'frozen/config.json')
 copied={}
 for n in COPY_SOURCE:
  h=digest((source/n).read_bytes())
  if h!=m['code'][n] or h!=digest((KIT/'template'/n).read_bytes()):raise ValueError('Original execution code changed: '+n)
  copied[n]=h
 regpath=source/'data/tokenizer_registry.json';raw=regpath.read_bytes()
 if digest(raw)!=m['inputs']['data/tokenizer_registry.json']:raise ValueError('Original tokenizer registry changed')
 registry=json.loads(raw);tokenizers={};newregistry={}
 for model in original_config['models']:
  entry=registry[model]
  if not entry.get('verified'):raise ValueError('Original tokenizer is unverified')
  path=(source/entry['path']).resolve()
  if not path.is_relative_to(source):raise ValueError('Original tokenizer resolves outside source project')
  blob=path.read_bytes();h=digest(blob)
  if h!=entry['sha256']:raise ValueError('Original tokenizer file hash mismatch')
  name='assets/tokenizers/'+h+'.json';tokenizers[name]=blob;newregistry[model]=dict(entry,path=name)
 # Stage atomically outside the source; failed installation leaves no usable partial workspace.
 dest.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.truecost-v2-install-',dir=dest.parent) as temp:
  project=Path(temp)/'pilot';shutil.copytree(KIT/'template',project,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for n,blob in tokenizers.items():q=project/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(blob)
  (project/'data').mkdir();(project/'provenance').mkdir()
  (project/'data/tokenizer_registry.json').write_text(json.dumps(newregistry,sort_keys=True,indent=2))
  (project/'provenance/selection_receipt.json').write_text(json.dumps(selection,sort_keys=True,indent=2))
  source_receipt=dict(source_project=str(source),v1_manifest_sha256=EXPECTED,source_frozen_files=m['files'],copied_execution_code=copied,source_tokenizer_registry_sha256=digest(raw),tokenizer_relocation='Identical verified bytes copied into pilot; only registry paths changed',scope='32 new groups, 128 development-only attempts; no future calibration or confirmation use',kit_sha256=digest((KIT/'kit_hashes.json').read_bytes()))
  (project/'provenance/source_receipt.json').write_text(json.dumps(source_receipt,sort_keys=True,indent=2))
  development_exclusions=dict(study='truecost-v2-development-pilot',future_calibration_or_final_test_eligible=False,task_ids=sorted(t['task_id'] for t in tasks),group_ids=sorted(t['group_id'] for t in tasks),source_group_ids=sorted({t['source_group_id'] for t in tasks}),normalized_texts=sorted({' '.join((t['instruction']+'\n'+t['context']).split()).casefold() for t in tasks}),normalized_contexts=sorted({' '.join(t['context'].split()).casefold() for t in tasks if t['context']}),rule='Future selection must propagate these exclusions through its complete source/text/context/relation grouping graph.')
  (project/'provenance/development_exclusions.json').write_text(json.dumps(development_exclusions,ensure_ascii=False,sort_keys=True,indent=2))
  cfg=dict(original_config,target_groups=32,target_runs=128,quotas={f:4 for f in selection['by_family']},study='truecost-v2-development-pilot',attempt_seconds=1800)
  for k in ['gpu','tool_probe','slurm_job_id']:cfg.pop(k,None)
  (project/'tasks_install.json').write_text(json.dumps(tasks,ensure_ascii=False))
  (project/'config_install.json').write_text(json.dumps(cfg))
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
  subprocess.run([sys.executable,'-c',"from prepare.manifest import freeze_pilot,read; freeze_pilot(read('tasks_install.json'),read('config_install.json'))"],cwd=project,env=env,check=True)
  (project/'tasks_install.json').unlink();(project/'config_install.json').unlink()
  subprocess.run([sys.executable,'-c',"from prepare.manifest import validate_manifest; m=validate_manifest('frozen'); assert len(m['runs'])==128; print('Pilot freeze validated: 32 groups / 128 attempts')"],cwd=project,env=env,check=True)
  # Rename only after validation; never write to the original source.
  project.rename(dest)
 print('INSTALLED:',dest)
 print('NEXT: cd '+str(dest)+' && python pilot.py preflight')
 print('No jobs submitted. Runtime asset paths must pass preflight before submission.')
if __name__=='__main__':main()
