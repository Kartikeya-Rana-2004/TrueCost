"""Conservative grouping and explicit review gates, independent of outcomes."""
import hashlib
import re
from difflib import SequenceMatcher

def normalized(s):return re.sub(r'\s+',' ',s).strip().casefold()

def group_candidates(tasks,relations):
    ids=[t['task_id'] for t in tasks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task ID')
    parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    seen={}
    for t in tasks:
        keys=[('source_group',t['source_group_id']),('text',normalized(t['instruction']+'\n'+t['context']))]
        if t['context']:keys.append(('context',normalized(t['context'])))
        for k in keys:
            if k in seen:join(t['task_id'],seen[k])
            seen[k]=t['task_id']
    pending=set()
    for rel in relations:
        if rel['left'] not in parent or rel['right'] not in parent:raise ValueError('Relation references unknown task')
        if rel['decision']=='related':join(rel['left'],rel['right'])
        elif rel['decision']=='pending':pending.update((rel['left'],rel['right']))
        elif rel['decision']!='unrelated':raise ValueError('Invalid relation review')
    roots={find(i) for i in pending}
    return [dict(t,group_id=hashlib.sha256(find(t['task_id']).encode()).hexdigest(),
                 relation_review_pending=find(t['task_id']) in roots) for t in tasks]

def near_duplicate_candidates(tasks,threshold=.85):
    # Screening only, decisions must be reviewed. No independence claim.
    pairs=[]
    for i,a in enumerate(tasks):
        x=normalized(a['instruction']+'\n'+a['context'])
        for b in tasks[i+1:]:
            y=normalized(b['instruction']+'\n'+b['context'])
            if min(len(x),len(y))/max(1,len(x),len(y))<threshold:continue
            ratio=SequenceMatcher(None,x,y,autojunk=False).ratio()
            if ratio>=threshold:pairs.append(dict(left=a['task_id'],right=b['task_id'],similarity=ratio,decision='pending'))
    return pairs

def select_tasks(tasks,quotas,excluded_groups):
    selected=[];used=set(excluded_groups)|{t['group_id'] for t in tasks if t.get('excluded_prior')};counts={f:0 for f in quotas}
    for t in sorted(tasks,key=lambda x:hashlib.sha256(('selection-440:'+x['task_id']).encode()).hexdigest()):
        f=t['family'];g=t['group_id']
        if f not in quotas or counts[f]>=quotas[f] or g in used:continue
        if t['review_status']!='approved' or t.get('relation_review_pending'):continue
        if t.get('excluded_prior'):continue
        selected.append(t);used.add(g);counts[f]+=1
    missing={f:quotas[f]-counts[f] for f in quotas if counts[f]!=quotas[f]}
    if missing:raise ValueError('Quota shortfall: '+str(missing))
    return selected
