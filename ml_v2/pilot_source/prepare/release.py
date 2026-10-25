"""Research release is derived from audited evidence, never an optimistic marker."""
def release_gate(tasks,config,smoke):
    if any(t.get('review_status')!='approved' or t.get('relation_review_pending',True) for t in tasks):raise ValueError('Source/group reviews incomplete')
    if not config.get('tokenizers_verified'):raise ValueError('Tokenizer verification incomplete')
    if not smoke or not smoke.get('passed'):raise ValueError('Audited smoke required')
    if len(tasks)!=3000 or len({t['group_id'] for t in tasks})!=3000:raise ValueError('Require 3000 distinct reviewed groups')
    models=set(config.get('models',[]))
    for task in tasks:
        admissions=task.get('admissions',{})
        if not models or set(admissions)!=models or any(not a['admitted'] for a in admissions.values()):raise ValueError('Context admission incomplete')
    return True
