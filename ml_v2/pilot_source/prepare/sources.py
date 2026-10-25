"""Preparation-only snapshot retrieval. Collection never downloads sources."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REGISTRY=Path(__file__).resolve().parents[1]/'sources/registry.json'

def verify_snapshot(path,entry):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Snapshot hash mismatch: '+str(path))
    return raw

def fetch_snapshot(source_id,destination):
    aliases={'mbpp':'sanitized-mbpp.json','humaneval':'HumanEval.jsonl.gz','gsm8k':'gsm8k_test.jsonl','dolly':'dolly.jsonl','tabfact':'tabfact_train.json','govreport':'govreport_validation.parquet'}
    entry=json.loads(REGISTRY.read_text())[aliases.get(source_id,source_id)]
    p=Path(destination)
    if not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True)
        raw=urllib.request.urlopen(entry['url'],timeout=120).read()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Downloaded snapshot hash mismatch')
        with p.open('xb') as out:out.write(raw)
    verify_snapshot(p,entry)
    return dict(entry,retrieved_utc=datetime.now(timezone.utc).isoformat(),size=p.stat().st_size)
