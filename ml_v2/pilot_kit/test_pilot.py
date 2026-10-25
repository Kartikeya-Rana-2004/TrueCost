"""Offline tests only: no GPU, Ollama, Slurm, or generated-code execution."""
import sys
sys.dont_write_bytecode=True
import copy,hashlib,importlib,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
KIT=Path(__file__).resolve().parent

class PilotTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'pilot';shutil.copytree(KIT/'template',cls.root)
  (cls.root/'data').mkdir();(cls.root/'provenance').mkdir();(cls.root/'assets').mkdir()
  (cls.root/'assets/test-tokenizer.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
  reg={m:dict(path='assets/test-tokenizer.json',sha256=h,verified=True) for m in ['qwen2.5-coder:7b','qwen2.5-coder:14b']}
  (cls.root/'data/tokenizer_registry.json').write_text(json.dumps(reg))
  for n in ['selection_receipt.json','source_receipt.json','development_exclusions.json']:(cls.root/'provenance'/n).write_text('{}')
  sys.path.insert(0,str(cls.root));cls.man=importlib.import_module('prepare.manifest');cls.worker=importlib.import_module('runtime.worker');cls.audit=importlib.import_module('runtime.audit');cls.submit=importlib.import_module('scripts.submit')
  cls.tasks=json.loads((KIT/'payload/tasks.json').read_text());cls.cfg=json.loads((KIT/'payload/reference_config.json').read_text());cls.m=cls.man.freeze_pilot(cls.tasks,cls.cfg);cls.taskmap={t['task_id']:t for t in cls.tasks}
 @classmethod
 def tearDownClass(cls):
  sys.path.remove(str(cls.root));cls.temp.cleanup()
 def run_mock(self,run,folder,cap=False,fail=False):
  task=self.taskmap[run['task_id']];steps=[]
  if task['family']=='coding':steps=[dict(action='write_solution',code='# offline fixture; never executed'),dict(action='run_tests'),dict(action='finish',answer='offline fixture')]
  elif task['context']:steps=[dict(action='read_context'),dict(action='finish',answer='offline fixture')]
  else:steps=[dict(action='finish',answer='offline fixture')]
  counter=[0]
  def invoke(payload,evidence,timeout):
   counter[0]+=1
   if fail:raise RuntimeError('Injected offline transport failure')
   action={} if cap else steps[min(counter[0]-1,len(steps)-1)]
   return dict(done=True,model=run['model'],prompt_eval_count=100,eval_count=20,message=dict(content=json.dumps(action)))
  def fake_tool(case,tests,image,evidence,timeout):
   # Only emits a synthetic tool-evidence fixture. Never executes supplied/generated code.
   marker='OFFLINE_FIXTURE_STARTED';raw=(marker+'\n').encode();self.man.write(evidence/'command.json',{'argv':['OFFLINE-MOCK-ONLY']});(evidence/'stdout.bin').write_bytes(raw)
   result=dict(started=True,passed=True,stdout_sha256=self.man.digest(raw),startup_marker=marker,command_sha256=self.man.digest((evidence/'command.json').read_bytes()))
   self.man.write(evidence/'result.json',result);return result
  cfg=dict(self.cfg,counter=lambda messages:100,invoke=invoke,image='OFFLINE-NOT-A-CONTAINER')
  with patch.object(self.worker,'execute_code',fake_tool):return self.worker.run_attempt(dict(run,task=task),cfg,folder)
 def test_01_all_128_mock_attempts_pass_original_audit(self):
  for run in self.m['runs']:
   folder=self.man.shard_path(self.root/'frozen',self.m['shards'][run['shard_id']])/'runs'/run['run_id'];report=self.run_mock(run,folder)
   self.assertEqual(report['partition'],'development');self.assertTrue(report['telemetry_complete']);self.assertLessEqual(report['requests'],6)
  reports=[]
  for shard in self.m['shards'].values():
   result=self.audit.audit_shard(self.m,self.man.shard_path(self.root/'frozen',shard));self.assertEqual(result['complete'],32);self.assertEqual(result['pending'],0);reports.extend(result['reports'])
  self.assertEqual(len(reports),128)
 def test_02_duplicate_group_and_quota_rejected(self):
  broken=copy.deepcopy(self.tasks);broken[1]['group_id']=broken[0]['group_id']
  with self.assertRaises(ValueError):self.man.assign_runs(broken)
  broken=copy.deepcopy(self.tasks);broken[0]['review_status']='pending'
  with self.assertRaises(ValueError):self.man.assign_runs(broken)
 def test_03_frozen_payload_tamper_detected(self):
  p=self.root/'frozen/config.json';before=p.read_bytes()
  try:
   p.write_bytes(before+b' ')
   with self.assertRaises(ValueError):self.man.validate_manifest(self.root/'frozen')
  finally:p.write_bytes(before)
  self.man.validate_manifest(self.root/'frozen')
 def test_04_foreign_v1_root_rejected(self):
  with self.assertRaises(ValueError):self.man.validate_manifest(self.root/'different-v1-frozen')
 def test_05_request_cap_retains_all_counters(self):
  run=next(r for r in self.m['runs'] if not self.taskmap[r['task_id']]['context'] and r['family']!='coding')
  with tempfile.TemporaryDirectory() as td:
   folder=Path(td)/run['shard_id']/'runs'/run['run_id'];report=self.run_mock(run,folder,cap=True)
   self.assertEqual(report['termination'],'request_limit');self.assertEqual(report['requests'],6);self.assertEqual(report['total_tokens'],720)
   self.assertEqual(self.audit.audit_shard(self.m,folder.parents[1])['complete'],1)
 def test_06_incomplete_attempt_blocks_audit_and_resume(self):
  run=self.m['runs'][0]
  with tempfile.TemporaryDirectory() as td:
   folder=Path(td)/run['shard_id']/'runs'/run['run_id']
   with self.assertRaises(RuntimeError):self.run_mock(run,folder,fail=True)
   self.assertTrue((folder/'failure.json').is_file());self.assertFalse((folder/'report.json').exists())
   with self.assertRaises(ValueError):self.audit.audit_shard(self.m,folder.parents[1])
   with self.assertRaises(ValueError):self.submit.resumable_runs([run],folder.parent)
 def test_07_altered_raw_response_rejected(self):
  run=self.m['runs'][0]
  with tempfile.TemporaryDirectory() as td:
   folder=Path(td)/run['shard_id']/'runs'/run['run_id'];self.run_mock(run,folder);raw=folder/'request-01/raw.json';raw.write_bytes(raw.read_bytes()+b' ')
   with self.assertRaises(ValueError):self.audit.audit_shard(self.m,folder.parents[1])
 def test_08_lock_and_walltime_guards(self):
  with self.man.claim_shard(self.root/'frozen','unit-test-lock'):
   with self.assertRaises(FileExistsError):
    with self.man.claim_shard(self.root/'frozen','unit-test-lock'):pass
  self.assertFalse(self.submit.should_continue(100,2199));self.assertTrue(self.submit.should_continue(100,2200))
 def test_10_atomic_installer_and_no_overwrite(self):
  import install_pilot
  with tempfile.TemporaryDirectory() as td:
   source=Path(td)/'source';dest=Path(td)/'new-pilot';(source/'frozen').mkdir(parents=True);(source/'data').mkdir();(source/'assets').mkdir()
   original=json.loads((KIT/'payload/reference_config.json').read_text());(source/'frozen/config.json').write_text(json.dumps(original));(source/'assets/tokenizer.json').write_text('{}')
   reg={m:dict(path='assets/tokenizer.json',sha256=hashlib.sha256(b'{}').hexdigest(),verified=True) for m in original['models']};(source/'data/tokenizer_registry.json').write_text(json.dumps(reg))
   m=dict(code={},inputs={'data/tokenizer_registry.json':hashlib.sha256((source/'data/tokenizer_registry.json').read_bytes()).hexdigest()},files={})
   for name in install_pilot.COPY_SOURCE:
    p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(KIT/'template'/name,p);m['code'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
   (source/'frozen/manifest.json').write_text(json.dumps(m));before={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
   selection=json.loads((KIT/'payload/selection_receipt.json').read_text())
   with patch.object(install_pilot,'select',return_value=(self.tasks,selection)),patch.object(sys,'argv',['install_pilot.py','--source',str(source),'--destination',str(dest)]):
    install_pilot.main()
    with self.assertRaises(FileExistsError):install_pilot.main()
   self.assertTrue((dest/'frozen/manifest.json').is_file())
   after={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()};self.assertEqual(before,after)
 def test_09_reference_not_public(self):
  from prepare.adapters import public_task
  for t in self.tasks:self.assertNotIn('reference',public_task(t));self.assertNotIn('answer',public_task(t))
if __name__=='__main__':unittest.main(verbosity=2)
