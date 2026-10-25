"""Family-specific menus. Model-generated code never executes on the host."""
def tools_for(task):
    tools=['finish']
    if task.get('context'): tools.insert(0,'read_context')
    if task['family'] in ('math','tabular'): tools.insert(0,'calculate')
    if task['family']=='coding': tools=['write_solution','run_tests','finish']
    return tools

import ast
import math
import operator
from pathlib import Path

def calculate(expression):
    if len(expression)>256:raise ValueError('Expression too long')
    try:tree=ast.parse(expression,mode='eval')
    except SyntaxError as e:raise ValueError('Invalid arithmetic') from e
    if sum(1 for _ in ast.walk(tree))>64:raise ValueError('Expression too complex')
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):value=n.value
        elif isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UAdd,ast.USub):value=visit(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        elif isinstance(n,ast.BinOp) and type(n.op) in ops:
            try:value=ops[type(n.op)](visit(n.left),visit(n.right))
            except ArithmeticError as e:raise ValueError('Arithmetic error') from e
        else:raise ValueError('Only bounded arithmetic is allowed')
        if not math.isfinite(value) or abs(value)>1e15:raise ValueError('Numeric range exceeded')
        return value
    return str(visit(tree.body))

def bounded_observation(value,limit):
    if len(value.encode())>limit:raise ValueError('Tool observation exceeds declared cap')
    return value

def code_command(case,tests,image):
    return ['apptainer','exec','--containall','--cleanenv','--no-home','--no-mount','hostfs,bind-paths,cwd',
            '--net','--network','none','--bind',str(case)+':/case:ro','--bind',str(tests)+':/case/test_solution.py:ro',
            '--pwd','/case',str(image),'python','-B','test_solution.py']

def execute_code(case,tests,image,evidence,timeout):
    """Retain startup proof separately from assertion outcomes and tool timing."""
    import hashlib,json,os,secrets,signal,subprocess,time
    from prepare.manifest import write
    evidence=Path(evidence);marker='TRUECOST_HARNESS_STARTED_'+secrets.token_hex(16)
    test_hash=hashlib.sha256(tests.read_bytes()).hexdigest()
    launcher="import runpy,pathlib,hashlib,socket; p=pathlib.Path('/case/test_solution.py'); assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(test_hash)+"; assert {name for _,name in socket.if_nameindex()} <= {'lo'}; print("+repr(marker)+",flush=True); runpy.run_path(str(p),run_name='__main__')"
    command=code_command(case,tests,image)[:-1]+['-c',launcher]
    write(evidence/'command.json',{'argv':command,'test_sha256':hashlib.sha256(tests.read_bytes()).hexdigest()})
    env={k:v for k,v in os.environ.items() if k not in ('APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH')}
    started=time.monotonic();timed_out=False;output_cap=False
    output=evidence/'stdout.bin'
    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024*1024,2*1024*1024))
    with output.open('xb') as out:
        process=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=limits)
        try:
            while process.poll() is None:
                timed_out=time.monotonic()-started>=timeout;output_cap=output.stat().st_size>=1024*1024
                if timed_out or output_cap:os.killpg(process.pid,signal.SIGKILL);break
                time.sleep(.05)
            process.wait()
        finally:
            # Remove descendants even if the direct process has already exited.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    raw=output.read_bytes();launched=(marker+'\n').encode() in raw
    result=dict(started=launched,passed=launched and process.returncode==0 and not timed_out and not output_cap,
                returncode=process.returncode,timed_out=timed_out,output_cap=output_cap,wall_seconds=time.monotonic()-started,
                stdout_sha256=hashlib.sha256(raw).hexdigest(),startup_marker=marker,command_sha256=hashlib.sha256((evidence/'command.json').read_bytes()).hexdigest())
    write(evidence/'result.json',result)
    return result
