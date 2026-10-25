"""Bounded typed-action controller, with all usage retained on task failure."""
import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from prepare.adapters import public_task
from prepare.manifest import write,digest
from prepare.token_counts import ContextBudget,count_messages
from runtime.tools import tools_for,calculate,code_command,bounded_observation,execute_code
from runtime.transport import invoke_local,validate_response

SYSTEM='Return one JSON object with action, answer, code and expression string fields. Use only listed actions. Read available context before finishing. For coding, write the complete solution.py source, run_tests, observe results, then finish. Preserve supplied tests. Never use networks. Answer accurately.'

def initial_messages(task):return [dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(public_task(task),ensure_ascii=False))]

def quality(task,answer,code_passed):
    f=task['family'];ref=str(task.get('reference',''))
    norm=lambda s:' '.join(str(s).strip().casefold().split())
    if f=='coding':return dict(quality_metric='supplied_checks',quality_check_passed=bool(code_passed))
    if f=='math':
        def number(s):
            try:return float(s.replace(',','').strip())
            except ValueError:return None
        a,b=number(answer),number(ref)
        return dict(quality_metric='numeric_exact',quality_check_passed=a is not None and b is not None and a==b)
    if f=='tabular':return dict(quality_metric='canonical_label',quality_check_passed=norm(answer)==norm(ref),format_compliant=norm(answer) in ('entailed','refuted'))
    tokenize=lambda s:re.findall(r'\w+',s.casefold())
    a,b=Counter(tokenize(answer)),Counter(tokenize(ref));common=sum((a&b).values())
    score=2*common/(sum(a.values())+sum(b.values())) if a and b else 0
    return dict(quality_metric='reference_token_f1_diagnostic',quality_score=score,quality_check_passed=None)

def run_attempt(run,config,evidence):
    folder=Path(evidence);folder.mkdir(parents=True,exist_ok=False)
    task=run['task'];model=run['model'];started=time.monotonic();deadline=started+config.get('attempt_seconds',1800)
    write(folder/'assignment.json',{k:v for k,v in run.items() if k!='task'})
    messages=initial_messages(task);ledger=[];termination='request_limit';answer='';context_read=not task['context'];tested=False;passed=False;written=False
    counter=config.get('counter',lambda m:count_messages(m,model))
    invoke=config.get('invoke',invoke_local)
    policy=tools_for(task);case=folder/'case';case.mkdir();(case/'test_solution.py').touch()
    source_bearing_input_tokens=None
    try:
        for index in range(1,7):
            if time.monotonic()>=deadline:raise TimeoutError('Attempt deadline exceeded')
            predicted=ContextBudget(counter,16384,1024).check(messages)
            p=dict(model=model,messages=messages,stream=False,format='json',keep_alive='30m',options=dict(num_ctx=16384,num_predict=1024,temperature=0,seed=run['seed'],top_p=.9,top_k=40,repeat_penalty=1.0))
            req=folder/f'request-{index:02d}';req.mkdir()
            value=validate_response(invoke(p,req,min(480,deadline-time.monotonic())),model)
            # Injected transports in offline tests may return objects without files.
            if not (req/'request.json').exists():write(req/'request.json',p)
            if not (req/'raw.json').exists():write(req/'raw.json',value)
            row=dict(index=index,input_tokens=value['prompt_eval_count'],output_tokens=value['eval_count'],
                     predicted_input_tokens=predicted,raw_sha256=digest((req/'raw.json').read_bytes()),request_sha256=digest((req/'request.json').read_bytes()))
            if config.get('residency'): row['runtime_residency']=config['residency']()
            ledger.append(row)
            with (folder/'ledger.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            if source_bearing_input_tokens is None and context_read:source_bearing_input_tokens=value['prompt_eval_count']
            messages.append(dict(role='assistant',content=value['message']['content']))
            try:action=json.loads(value['message']['content'])
            except (json.JSONDecodeError,TypeError):action={}
            if not isinstance(action,dict):action={}
            name=action.get('action');observation='Unsupported action; use listed actions.'
            if name=='finish' and context_read and (task['family']!='coding' or tested):
                answer=str(action.get('answer',''));termination='completed';break
            elif name=='finish':observation='Read context and run required tests before finish.'
            elif name=='read_context' and name in policy:
                if context_read:observation='Context already supplied; use it from history.'
                else:observation=task['context'];context_read=True
            elif name=='calculate' and name in policy:
                try:observation=calculate(str(action.get('expression','')))
                except ValueError as error:observation=str(error)
            elif name=='write_solution' and name in policy:
                code=action.get('code','')
                if not isinstance(code,str) or len(code.encode())>16384:observation='Code size exceeds declared cap.'
                else:(case/'solution.py').write_text(code);written=True;tested=False;passed=False;observation='Solution written. Run tests.'
            elif name=='run_tests' and name in policy:
                if not written:observation='Write solution first.'
                else:
                    with tempfile.TemporaryDirectory(prefix='truecost-tests-') as td:
                        tests=Path(td)/'test_solution.py';tests.write_text(task['tests']);before=digest(tests.read_bytes())
                        tool=folder/f'tool-{index:02d}';tool.mkdir()
                        result=execute_code(case.resolve(),tests,config['image'],tool,min(30,deadline-time.monotonic()))
                        if not result['started']:raise RuntimeError('Coding harness did not start; inspect retained tool evidence')
                        passed=result['passed']
                        if digest(tests.read_bytes())!=before:raise ValueError('Tests changed')
                    tested=True;observation='Tests passed.' if passed else 'Tests failed or timed out.'
            # Full source is never truncated. Other observations reserve at most 256 bytes.
            if name!='read_context' or observation!=task['context']:bounded_observation(observation,256)
            messages.append(dict(role='user',content='Tool observation:\n'+observation))
        if not context_read:answer=''
        report={k:v for k,v in run.items() if k!='task'}
        report.update(termination=termination,requests=len(ledger),input_tokens=sum(x['input_tokens'] for x in ledger),output_tokens=sum(x['output_tokens'] for x in ledger),telemetry_complete=True,wall_seconds=time.monotonic()-started)
        report['total_tokens']=report['input_tokens']+report['output_tokens']
        report.update({k:task.get(k,'unknown') for k in ('family','source_id','complexity','size_band')})
        report['source_bearing_input_tokens']=source_bearing_input_tokens
        report.update(quality(task,answer,passed));report['max_request_input_tokens']=max(x['input_tokens'] for x in ledger)
        report['initial_input_tokens']=ledger[0]['input_tokens'];report['answer']=answer
        if 'runtime_residency' in ledger[-1]:report['runtime_residency']=ledger[-1]['runtime_residency']
        write(folder/'report.json',report)
        return report
    except BaseException as error:
        write(folder/'failure.json',dict(error=type(error).__name__+': '+str(error),telemetry_complete=False,requests_recorded=len(ledger)))
        raise
