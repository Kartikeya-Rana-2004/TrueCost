"""Own one loopback runtime; never reuse a pre-existing listener."""
import json
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager

def port_available(port):
    with socket.socket() as s:
        try:s.bind(('127.0.0.1',port));return True
        except OSError:return False

def api(endpoint,route,payload=None):
    req=urllib.request.Request(endpoint+route,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=10) as r:return json.load(r)

@contextmanager
def owned_server(binary,models,logfile):
    # Runtime must bind its own port; liveness + exclusive listener + clean startup log checked before use.
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    if not port_available(port):raise RuntimeError('Selected port was claimed')
    endpoint=f'http://127.0.0.1:{port}'
    env=dict(os.environ,OLLAMA_HOST=f'127.0.0.1:{port}',OLLAMA_MODELS=models,OLLAMA_NO_CLOUD='1',OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1')
    with open(logfile,'x') as log:
        process=subprocess.Popen([binary,'serve'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            ready=False
            for _ in range(180):
                if process.poll() is not None:raise RuntimeError('Owned runtime exited during startup')
                if ('Listening on 127.0.0.1:'+str(port)) not in __import__('pathlib').Path(logfile).read_text():
                    time.sleep(1);continue
                try:
                    result=api(endpoint,'/api/version')
                    if result.get('version')=='0.35.1':ready=True;break
                except (OSError,ValueError):pass
                time.sleep(1)
            if not ready:raise TimeoutError('Owned runtime readiness timeout')
            time.sleep(.2)
            if process.poll() is not None:raise RuntimeError('Runtime lost listener race')
            if 'address already in use' in __import__('pathlib').Path(logfile).read_text().casefold():raise RuntimeError('Listener conflict')
            yield endpoint
        finally:
            import signal
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
