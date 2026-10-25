import unittest
import sqlite3
from feedback import Ledger

class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.l = Ledger(':memory:')
    def p(self, **kw):
        p = dict(task_id='t1',group_id='g1',model='qwen2.5-coder:7b',model_version='sha256:example',controller_version='six-request-v1',predicted_total_tokens=100,upper_budget=150)
        p.update(kw)
        return self.l.predict(p)
    def c(self, **kw):
        c = dict(status='completed',telemetry_complete=True,input_tokens=60,output_tokens=60,total_tokens=120,task_success=True)
        c.update(kw)
        return c
    def test_link_and_metrics(self):
        pid=self.p(); self.l.complete(pid,self.c())
        r=self.l.report()
        self.assertEqual(r['overall']['mae'],20)
        self.assertEqual(r['overall']['bias'],-20)
        self.assertEqual(r['overall']['budget_coverage'],1)
        self.assertEqual(len(self.l.export()['records']),1)
    def test_unknown_and_duplicate_completion(self):
        with self.assertRaises(ValueError): self.l.complete('unknown',self.c())
        pid=self.p();self.l.complete(pid,self.c())
        with self.assertRaises(sqlite3.IntegrityError):self.l.complete(pid,self.c())
    def test_missing_not_zero(self):
        pid=self.p();self.l.complete(pid,self.c(telemetry_complete=False,total_tokens=None))
        self.assertIsNone(self.l.report()['overall']['mae'])
        self.assertEqual(self.l.export()['records'],[])
    def test_validation(self):
        for value in (-1,float('nan'),True):
            with self.assertRaises(ValueError):self.p(predicted_total_tokens=value)
        pid=self.p()
        with self.assertRaises(ValueError):self.l.complete(pid,self.c(total_tokens=121))
        with self.assertRaises(ValueError):self.l.complete(pid,self.c(input_tokens=1.5))
    def test_protected_group_and_failed_runs(self):
        pid=self.p();self.l.complete(pid,self.c(status='failed',task_success=False))
        self.assertEqual(len(self.l.export()['records']),1)
        self.p(partition='confirmation')
        self.assertEqual(self.l.export()['records'],[])
    def test_zero_and_budget_exceedance(self):
        pid=self.p();self.l.complete(pid,self.c(input_tokens=0,output_tokens=0,total_tokens=0))
        self.assertIsNone(self.l.rows()[0]['percentage_error'])
        pid=self.p(group_id='g2');self.l.complete(pid,self.c(input_tokens=100,output_tokens=100,total_tokens=200))
        self.assertEqual(self.l.report()['overall']['budget_coverage'],.5)
    def test_pending_and_interrupted(self):
        self.p();pid=self.p(group_id='g2');self.l.complete(pid,self.c(status='interrupted'))
        self.assertEqual(self.l.report()['pending'],1)
        self.assertEqual(self.l.export()['records'],[])

if __name__=='__main__': unittest.main()
