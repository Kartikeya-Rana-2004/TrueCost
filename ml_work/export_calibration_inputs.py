"""Run on Roar from the collection project; exports inputs only, never outcomes."""
import argparse,hashlib,json
from pathlib import Path
EXPECTED='ee380785a4222add8b4177f267a758a26265851138c1ca8c89ab35cf3977a828'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,default=Path('frozen'));p.add_argument('--out',type=Path,default=Path('calibration_task_inputs.json'));a=p.parse_args()
 if digest(a.frozen/'manifest.json')!=EXPECTED:raise ValueError('Unexpected research manifest; do not export')
 m=json.loads((a.frozen/'manifest.json').read_text())
 if digest(a.frozen/'tasks.json')!=m['files']['tasks.json']:raise ValueError('Frozen tasks hash mismatch')
 runs=m['runs'];ids={r['task_id'] for r in runs if r['partition']=='calibration'}
 assert len(ids)==450
 assert not ids&{r['task_id'] for r in runs if r['partition']!='calibration'}
 tasks=json.loads((a.frozen/'tasks.json').read_text())
 fields=['task_id','group_id','family','source_id','size_band','instruction','context','public_tests']
 selected=[{k:t[k] for k in fields if k in t} for t in tasks if t['task_id'] in ids]
 assert len(selected)==450 and all('instruction' in r and 'context' in r for r in selected)
 # Exclusive creates: never overwrite an existing collection or export file.
 with a.out.open('x') as f:json.dump(selected,f,ensure_ascii=False,indent=2)
 prov=dict(manifest_sha256=EXPECTED,frozen_tasks_sha256=m['files']['tasks.json'],export_sha256=digest(a.out),partition='calibration',task_groups=450,outcomes_exported=False,confirmation_exported=False,fields=fields)
 with a.out.with_suffix('.provenance.json').open('x') as f:json.dump(prov,f,indent=2)
 print(f'Exported {len(selected)} calibration inputs to {a.out}; no outcome files read.')
if __name__=='__main__':main()
