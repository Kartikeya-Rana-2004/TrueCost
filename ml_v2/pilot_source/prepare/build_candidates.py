"""Local-only materialization. Candidate eligibility is not manual review."""
import gzip
import hashlib
import json
from pathlib import Path
from prepare.adapters import adapt_record,FAMILIES
from prepare.grouping import group_candidates,normalized
from prepare.coverage import coverage

ROOT=Path(__file__).resolve().parents[1]
def load_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]

def tabfact_claim(table_id,raw):
    claims,labels,caption=raw
    wanted=int(hashlib.sha256(table_id.encode()).hexdigest(),16)%2
    choices=[i for i,label in enumerate(labels) if label==wanted] or list(range(len(claims)))
    index=choices[int(hashlib.sha256(('claim:'+table_id).encode()).hexdigest(),16)%len(choices)]
    return dict(id=table_id+':'+str(index),table_id=table_id,claim=claims[index],label=labels[index],claim_index=index)

def mbpp_records(snap):
    """Preserve sanitized rows; stage only additional IDs supported by this wrapper."""
    snap=Path(snap)
    legacy=json.loads((snap/'sanitized-mbpp.json').read_text())
    rows=[(r,'sanitized-mbpp.json') for r in legacy]
    seen={str(r['task_id']) for r in legacy}
    full=snap/'mbpp-full.jsonl'
    if full.exists():
        for r in load_jsonl(full):
            key=str(r['task_id'])
            if key in seen:continue
            seen.add(key)
            # Existing adapter has no setup-code execution protocol. Never drop it.
            if r.get('test_setup_code','').strip():continue
            rows.append((r,'mbpp-full.jsonl'))
    return rows

def build_candidates():
    snap=ROOT/'sources/snapshots';rows=[]
    for raw in [json.loads(x) for x in gzip.decompress((snap/'HumanEval.jsonl.gz').read_bytes()).decode().splitlines()]:rows.append(adapt_record('humaneval',raw))
    mbpp_rows=mbpp_records(snap)
    mbpp_sources={'mbpp:'+str(raw['task_id']):name for raw,name in mbpp_rows}
    for raw,name in mbpp_rows:rows.append(adapt_record('mbpp',raw))
    for i,raw in enumerate(load_jsonl(snap/'gsm8k_test.jsonl')):rows.append(adapt_record('gsm8k',dict(raw,id=i)))
    for i,raw in enumerate(load_jsonl(snap/'dolly.jsonl')):
        if raw['category'] in FAMILIES:rows.append(adapt_record('dolly',dict(raw,id=i)))
    import pyarrow.parquet as pq
    for i,raw in enumerate(pq.read_table(snap/'govreport_validation.parquet').to_pylist()):rows.append(adapt_record('govreport',dict(raw,id=i)))
    tab=json.loads((snap/'tabfact_train.json').read_text())
    for path in sorted((snap/'tables').glob('*.csv')):
        claim=tabfact_claim(path.name,tab[path.name])
        rows.append(adapt_record('tabfact',dict(claim,table=path.read_text())))
    exclusions=json.loads((ROOT/'data/exclusions.json').read_text())
    old_ids=set(exclusions['task_ids']);old_text=set(exclusions['normalized_texts'])
    tokenizer_file=snap/'tokenizer.json'
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_file))
    from runtime.worker import initial_messages
    def count(m):
        text=''.join('<|im_start|>'+x['role']+'\n'+x['content']+'<|im_end|>\n' for x in m)+'<|im_start|>assistant\n'
        return len(tok.encode(text,add_special_tokens=False).ids)
    from prepare.token_counts import size_band
    for t in rows:
        text=normalized(t['instruction']+'\n'+t['context'])
        legacy='mbpp-local-'+t['source_row_id']
        source_exclusions=exclusions.get('source_rows',{})
        old_source_row=t['source_row_id'] in source_exclusions.get(t['source_id'],[])
        old_math=t['source_id']=='gsm8k' and hashlib.sha256(t['instruction'].encode()).hexdigest() in source_exclusions.get('gsm8k_question_hash',[])
        t['excluded_prior']=old_source_row or old_math or t['task_id'] in old_ids or (t['source_id']=='mbpp' and legacy in old_ids) or text in old_text
        m=initial_messages(t)
        if t['context']:m=m+[dict(role='assistant',content='{"action":"read_context"}'),dict(role='user',content='Tool observation:\n'+t['context'])]
        n=count(m);t['candidate_source_bearing_tokens']=n;t['size_band']=size_band(n)
        t['candidate_admitted']=n+6*1024+5*256+512<=16384 and n<=8192
        t['review_status']='pending'
        t['provenance']={'snapshot_file':{'mbpp':mbpp_sources.get(t['task_id'],'sanitized-mbpp.json'),'humaneval':'HumanEval.jsonl.gz','gsm8k':'gsm8k_test.jsonl','dolly':'dolly.jsonl','tabfact':'tabfact_train.json','govreport':'govreport_validation.parquet'}[t['source_id']],
                         'wrapper':'prepare/adapters.py','token_counts':'candidate ChatML; requires actual runtime smoke'}
    rows=group_candidates(rows,[])
    (ROOT/'data/candidates.json').write_text(json.dumps(rows,ensure_ascii=False))
    eligible=[t for t in rows if not t['excluded_prior'] and t['candidate_admitted']]
    from collections import Counter
    report=dict(tabfact_reference_labels=dict(Counter(t['reference'] for t in rows if t['source_id']=='tabfact')),all_candidates=coverage(rows),eligible_candidates=coverage(eligible),candidate_count=len(rows),eligible_count=len(eligible),reviewed_count=0,warning='Pending row/group review; token lengths provisional until Roar smoke')
    (ROOT/'data/coverage.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':print(json.dumps(build_candidates(),indent=2))
