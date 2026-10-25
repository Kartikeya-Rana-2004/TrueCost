"""Pinned ChatML tokenizer, admitted only after comparison to staged runtime."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def candidate_count(messages,model):
    registry=json.loads((ROOT/'data/tokenizer_registry.json').read_text())
    entry=registry.get(model)
    if not entry:raise ValueError('Tokenizer/template not verified for '+model)
    path=ROOT/entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Tokenizer hash changed')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(path))
    text=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
    return len(tokenizer.encode(text,add_special_tokens=False).ids)

def count_messages(messages,model):
    registry=json.loads((ROOT/'data/tokenizer_registry.json').read_text())
    if not registry.get(model,{}).get('verified'):raise ValueError('Tokenizer/template not verified for '+model)
    return candidate_count(messages,model)

class ContextBudget:
    def __init__(self,counter,context=16384,output=1024):self.counter=counter;self.context=context;self.output=output
    def check(self,messages):
        n=self.counter(messages)
        if n+self.output>self.context:raise ValueError('Reserved context capacity exceeded')
        return n

def size_band(n):
    for boundary,label in [(512,'<=512'),(2048,'513-2048'),(4096,'2049-4096'),(8192,'4097-8192')]:
        if n<=boundary:return label
    return 'rejected'

def admit_task(task,model,limits):
    counter=limits.get('counter',count_messages)
    from runtime.worker import initial_messages
    initial=initial_messages(task)
    source=initial+[{'role':'assistant','content':'{"action":"read_context"}'},{'role':'user','content':'Tool observation:\n'+task['context']}] if task['context'] else initial
    a,b=counter(initial,model),counter(source,model)
    # Five previous outputs, five bounded observations, plus the next output.
    reserved=6*limits['output']+5*256+512
    okay=b+reserved<=limits['context'] and b<=8192
    return dict(initial_tokens=a,source_bearing_tokens=b,size_band=size_band(b),admitted=okay,
                reserved_history_tokens=reserved,rejection_reason=None if okay else 'Context/history reservation exceeded')
