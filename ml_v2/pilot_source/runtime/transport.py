"""One-shot loopback requests with original-byte evidence; no retries."""
import json
import urllib.request
import urllib.error
from pathlib import Path
from prepare.manifest import write

def validate_response(value,model):
    if value.get('done') is not True or value.get('model')!=model:raise ValueError('Incomplete or unexpected runtime response')
    for field in ('prompt_eval_count','eval_count'):
        if type(value.get(field)) is not int or value[field]<0:raise ValueError('Missing runtime counters')
    if not value['prompt_eval_count']:raise ValueError('Empty input counter')
    if not isinstance(value.get('message',{}).get('content'),str):raise ValueError('Missing response content')
    return value

def invoke_local(payload,evidence,timeout_seconds,endpoint=None):
    import os
    endpoint=endpoint or os.environ['TRUECOST_ENDPOINT']
    from urllib.parse import urlparse
    u=urlparse(endpoint)
    if u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.path not in ('','/'):raise ValueError('Only loopback endpoint allowed')
    evidence=Path(evidence);write(evidence/'request.json',payload)
    request=urllib.request.Request(endpoint.rstrip('/')+'/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request,timeout=timeout_seconds) as response:raw=response.read()
    except urllib.error.HTTPError as error:
        (evidence/'http_error.bin').write_bytes(error.read());raise
    with (evidence/'raw.json').open('xb') as f:f.write(raw)
    return validate_response(json.loads(raw),payload['model'])
