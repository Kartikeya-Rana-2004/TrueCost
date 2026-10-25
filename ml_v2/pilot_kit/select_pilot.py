"""Select development-only tasks from the pinned original candidate/review snapshot."""
import hashlib,json,importlib.util
from collections import Counter
from pathlib import Path
EXPECTED='ee380785a4222add8b4177f267a758a26265851138c1ca8c89ab35cf3977a828'
FAMILIES=['classification','coding','extraction','grounded_qa','ideation','math','summarization','tabular']
BANDS=['<=512','513-2048','2049-4096','4097-8192']
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def select(source):
 source=Path(source).resolve();m=read(source/'frozen/manifest.json');assert digest((source/'frozen/manifest.json').read_bytes())==EXPECTED,'Different V1 campaign'
 for name in ['data/candidates.json','data/reviews.json','data/relations.json','data/group_review.json','data/exclusions.json']:
  assert digest((source/name).read_bytes())==m['inputs'][name],name
 for name in ['tasks.json','config.json']:assert digest((source/'frozen'/name).read_bytes())==m['files'][name],name
 assert digest((source/'prepare/grouping.py').read_bytes())==m['code']['prepare/grouping.py']
 spec=importlib.util.spec_from_file_location('v1_grouping',source/'prepare/grouping.py');grouping=importlib.util.module_from_spec(spec);spec.loader.exec_module(grouping)
 tasks=read(source/'data/candidates.json');reviews=read(source/'data/reviews.json');relations=read(source/'data/relations.json');exclusions=read(source/'data/exclusions.json');review=read(source/'data/group_review.json')
 assert review.get('reviewer') and review.get('rationale')
 assert review['candidates_sha256']==digest((source/'data/candidates.json').read_bytes())
 assert review['relations_sha256']==digest(json.dumps(relations,sort_keys=True).encode())
 old_ids=set(exclusions['task_ids']);old_text=set(exclusions['normalized_texts']);reserved=set(exclusions['reserved_group_metadata']);sr=exclusions.get('source_rows',{})
 smoke_ids=exclusions.get('prior_large12000_smoke',{}).get('task_ids',[]);assert isinstance(smoke_ids,list) and all(isinstance(s,str) for s in smoke_ids);old_ids.update(smoke_ids)
 original=read(source/'frozen/tasks.json');old_ids.update(t['task_id'] for t in original)
 assert (source/'smoke/tasks.json').is_file();old_ids.update(t['task_id'] for t in read(source/'smoke/tasks.json'))
 checked=[]
 for t in tasks:
  source_id=t['source_id'];row=t['source_row_id'];r=reviews.get(t['task_id'],{})
  valid=r.get('decision')=='approved' and r.get('task_sha256')==digest(json.dumps(t,sort_keys=True).encode()) and bool(r.get('reviewer')) and bool(r.get('rationale'))
  blocked=bool(t.get('excluded_prior') or t['task_id'] in old_ids or t.get('group_id') in reserved or grouping.normalized(t['instruction']+'\n'+t['context']) in old_text or row in sr.get(source_id,[]) or (source_id=='mbpp' and 'mbpp-local-'+str(row) in old_ids) or (source_id=='gsm8k' and digest(t['instruction'].encode()) in sr.get('gsm8k_question_hash',[])))
  checked.append(dict(t,_valid=bool(valid),_blocked=blocked))
 grouped=grouping.group_candidates(checked,relations);blocked_groups=reserved|{t['group_id'] for t in grouped if t['_blocked']}
 eligible=[t for t in grouped if t['group_id'] not in blocked_groups and t['_valid'] and t.get('candidate_admitted',False) and not t.get('relation_review_pending') and t.get('size_band') in BANDS]
 # Bipartite slot matching: 4 distinct groups per family, preferring different natural bands.
 # Augmenting paths prevent a flexible family consuming another family's only group.
 by_family={f:[t for t in eligible if t['family']==f] for f in FAMILIES}
 slots=[(f,i) for f in sorted(FAMILIES,key=lambda f:(len({t['group_id'] for t in by_family[f]}),f)) for i in range(4)]
 options={};representatives={}
 for slot in slots:
  f,i=slot;band_order=BANDS[i:]+BANDS[:i]
  ranked=sorted(by_family[f],key=lambda t:(band_order.index(t['size_band']),digest(('truecost-v2-pilot-440:'+t['task_id']).encode())))
  opts=[]
  for t in ranked:
   gid=t['group_id']
   if gid not in opts:opts.append(gid);representatives[(slot,gid)]=t
  options[slot]=opts
 owner={};allocation={}
 def match(slot,seen):
  for gid in options[slot]:
   if gid in seen:continue
   seen.add(gid)
   if gid not in owner or match(owner[gid],seen):owner[gid]=slot;allocation[slot]=gid;return True
  return False
 for slot in slots:
  if not match(slot,set()):raise ValueError('Cannot allocate four distinct groups to every family')
 selected=[]
 for slot in sorted(slots):
  t=representatives[(slot,allocation[slot])];record={k:v for k,v in t.items() if not k.startswith('_')};r=reviews[t['task_id']]
  record.update(review_status='approved',complexity=r.get('complexity','unrated'),complexity_rationale=r.get('complexity_rationale',''),excluded_prior=False)
  selected.append(record)
 assert len(selected)==len({t['group_id'] for t in selected})==len({t['task_id'] for t in selected})==32
 assert Counter(t['family'] for t in selected)==Counter({f:4 for f in FAMILIES})
 assert not {t['task_id'] for t in selected}&old_ids and not {t['group_id'] for t in selected}&blocked_groups
 sourcefiles=['frozen/manifest.json','frozen/tasks.json','frozen/config.json','smoke/tasks.json','data/candidates.json','data/reviews.json','data/relations.json','data/group_review.json','data/exclusions.json','prepare/grouping.py']
 receipt=dict(v1_manifest_sha256=EXPECTED,source_files={n:digest((source/n).read_bytes()) for n in sourcefiles},eligible_records=len(eligible),eligible_groups=len({t['group_id'] for t in eligible}),selection_rule='4 slots per family; deterministic augmenting-path group matching; slot preferences cycle short-to-long natural size bands; task tie-break SHA256 truecost-v2-pilot-440 prefix',groups=32,attempts=128,partition='development',future_calibration_or_final_test_eligible=False,by_family=dict(Counter(t['family'] for t in selected)),by_size=dict(Counter(t['size_band'] for t in selected)),all_original_pilot_smoke_and_reserved_exclusions_replayed=True,limitations='Existing reviewed relation decisions reduce known overlap, not all possible semantic overlap or model pretraining exposure.')
 return selected,receipt
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();tasks,receipt=select(a.source);a.out.mkdir(parents=True,exist_ok=True)
 for name,value in [('tasks.json',tasks),('selection_receipt.json',receipt)]:
  with (a.out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2)
 print(json.dumps(receipt,indent=2))
