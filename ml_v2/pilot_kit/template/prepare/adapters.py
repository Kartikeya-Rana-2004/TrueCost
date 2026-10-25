"""Lossless source wrappers; answers never enter public task objects."""
import ast
import hashlib
import json

FAMILIES = {'closed_qa':'grounded_qa', 'information_extraction':'extraction',
            'classification':'classification', 'summarization':'summarization',
            'brainstorming':'ideation'}

def adapt_record(source_id, raw):
    row = str(raw.get('task_id', raw.get('id', hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest())))
    t = dict(task_id=f'{source_id}:{row}', source_id=source_id, source_row_id=row,
             source_group_id=f'{source_id}:{row}', instruction='', context='',
             reference=None, tests='', entry_point=None, family='', provenance={},
             review_status='pending', complexity='unrated', complexity_rationale='',
             public_tests=False)
    if source_id == 'humaneval':
        t.update(family='coding', instruction='Complete this Python function, preserving its signature:\n'+raw['prompt'],
                 entry_point=raw['entry_point'], tests='from solution import '+raw['entry_point']+'\n'+raw['test']+'\ncheck('+raw['entry_point']+')\n',
                 reference=raw.get('canonical_solution'), public_tests=False)
    elif source_id == 'mbpp':
        checks=raw['test_list']; names=set()
        for check in checks:
            tree=ast.parse(check)
            names.update(n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name))
        names-=set(dir(__builtins__)) if not isinstance(__builtins__,dict) else set(__builtins__)
        imports='\n'.join(raw.get('test_imports',[]))
        helpers=set()
        for node in ast.walk(ast.parse(imports)):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                helpers.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        names-=helpers
        # Explicit imports preserve private function names excluded by import *.
        test=imports+'\nfrom solution import '+', '.join(sorted(names))+'\n'+'\n'.join(checks)+'\n'
        t.update(family='coding', instruction=raw.get('prompt',raw.get('text','')), tests=test,
                 reference=raw.get('code'),public_tests=True)
    elif source_id == 'gsm8k':
        t.update(family='math', instruction=raw['question'],reference=raw['answer'].split('####')[-1].strip())
    elif source_id == 'dolly':
        t.update(family=FAMILIES[raw['category']],instruction=raw['instruction'],context=raw['context'],reference=raw['response'])
    elif source_id == 'tabfact':
        t.update(family='tabular', instruction='Determine whether the claim is entailed or refuted by the table. Answer entailed or refuted.\nClaim: '+raw['claim'],
                 context=raw['table'],reference={0:'refuted',1:'entailed'}[raw['label']],source_group_id='tabfact:table:'+raw['table_id'])
        t['source_claim_index']=raw.get('claim_index',0)
    elif source_id == 'govreport':
        t.update(family='summarization',instruction='Summarize the report accurately, preserving its main findings.',
                 context=raw.get('report',raw.get('document','')),reference=raw['summary'])
    else: raise ValueError('Unsupported source: '+source_id)
    if not t['instruction'].strip(): raise ValueError('Empty instruction')
    return t

def public_task(task):
    from runtime.tools import tools_for
    return dict(instruction=task['instruction'],context_available=bool(task['context']),
                context_characters=len(task['context']),tools=tools_for(task),
                public_tests=task['tests'] if task.get('public_tests') else '')
