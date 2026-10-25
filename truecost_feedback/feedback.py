"""TrueCost prediction-to-completion feedback ledger. Python 3.10+, stdlib only."""
import argparse
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def number(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    if integer and int(value) != value:
        raise ValueError(f'{name} must be an integer')
    return value


class Ledger:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS predictions (
          prediction_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, group_id TEXT NOT NULL,
          model TEXT NOT NULL, model_version TEXT NOT NULL, controller_version TEXT NOT NULL,
          partition TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS completions (
          prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
          recorded_at TEXT NOT NULL, payload TEXT NOT NULL);
        ''')

    def predict(self, p):
        for key in ('task_id', 'group_id', 'model', 'model_version', 'controller_version'):
            if not isinstance(p.get(key), str) or not p[key].strip():
                raise ValueError(f'{key} is required')
        number(p.get('predicted_total_tokens'), 'predicted_total_tokens')
        if p.get('upper_budget') is not None:
            number(p['upper_budget'], 'upper_budget')
        if p.get('partition', 'feedback') not in ('feedback', 'training', 'calibration', 'confirmation', 'development'):
            raise ValueError('Unknown partition')
        p = dict(p)
        p['prediction_id'] = p.get('prediction_id') or str(uuid.uuid4())
        p['partition'] = p.get('partition', 'feedback')
        with self.db:
            self.db.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)',
                tuple(p[k] for k in ('prediction_id', 'task_id', 'group_id', 'model', 'model_version', 'controller_version', 'partition')) + (now(), json.dumps(p, allow_nan=False)))
        return p['prediction_id']

    def complete(self, pid, c):
        if not self.db.execute('SELECT 1 FROM predictions WHERE prediction_id=?', (pid,)).fetchone():
            raise ValueError('Unknown prediction_id')
        if c.get('status') not in ('completed', 'failed', 'step_limit', 'interrupted'):
            raise ValueError('Invalid status')
        if not isinstance(c.get('telemetry_complete'), bool):
            raise ValueError('telemetry_complete must be boolean')
        if c.get('task_success') is not None and not isinstance(c['task_success'], bool):
            raise ValueError('task_success must be boolean or null')
        for key in ('input_tokens', 'output_tokens', 'total_tokens', 'requests', 'retries', 'tool_calls'):
            if c.get(key) is not None:
                number(c[key], key, integer=True)
        if c.get('elapsed_seconds') is not None:
            number(c['elapsed_seconds'], 'elapsed_seconds')
        if c['telemetry_complete']:
            for key in ('input_tokens', 'output_tokens', 'total_tokens'):
                number(c.get(key), key, integer=True)
            if c['total_tokens'] != c['input_tokens'] + c['output_tokens']:
                raise ValueError('Token arithmetic mismatch')
        elif all(c.get(k) is not None for k in ('input_tokens', 'output_tokens', 'total_tokens')):
            if c['total_tokens'] != c['input_tokens'] + c['output_tokens']:
                raise ValueError('Token arithmetic mismatch')
        with self.db:
            self.db.execute('INSERT INTO completions VALUES (?,?,?)', (pid, now(), json.dumps(c, allow_nan=False)))

    def rows(self):
        rows = []
        for r in self.db.execute('SELECT p.payload, c.payload AS completion FROM predictions p LEFT JOIN completions c USING(prediction_id) ORDER BY p.created_at'):
            p = json.loads(r['payload'])
            c = json.loads(r['completion']) if r['completion'] else None
            row = dict(p, completion=c)
            if c and c['telemetry_complete']:
                actual = c['total_tokens']
                error = p['predicted_total_tokens'] - actual
                row.update(actual_total_tokens=actual, signed_error=error, absolute_error=abs(error),
                           percentage_error=None if actual == 0 else 100 * error / actual,
                           budget_exceeded=None if p.get('upper_budget') is None else actual > p['upper_budget'])
            rows.append(row)
        return rows

    def report(self):
        rows = self.rows()
        measured = [r for r in rows if 'absolute_error' in r]
        def metrics(items):
            budgeted = [r for r in items if r['budget_exceeded'] is not None]
            n = len(items)
            return dict(n=n, mae=sum(r['absolute_error'] for r in items)/n if n else None,
                        bias=sum(r['signed_error'] for r in items)/n if n else None,
                        rmse=math.sqrt(sum(r['signed_error']**2 for r in items)/n) if n else None,
                        budget_n=len(budgeted), budget_coverage=sum(not r['budget_exceeded'] for r in budgeted)/len(budgeted) if budgeted else None)
        strata = {}
        for r in measured:
            key = json.dumps([r['model'], r['model_version'], r['controller_version'], r['partition']])
            strata.setdefault(key, []).append(r)
        return dict(predictions=len(rows), pending=sum(r['completion'] is None for r in rows),
                    incomplete_telemetry=sum(r['completion'] is not None and not r['completion']['telemetry_complete'] for r in rows),
                    overall=metrics(measured), strata={k: metrics(v) for k, v in strata.items()}, runs=rows)

    def export(self):
        rows = self.rows()
        # Once a group is marked protected, all of its variants are excluded.
        protected = {r['group_id'] for r in rows if r['partition'] != 'feedback'}
        eligible = [r for r in rows if 'absolute_error' in r and r['group_id'] not in protected
                    and r['completion']['status'] != 'interrupted']
        return dict(schema_version=1, exported_at=now(), purpose='candidate training only; requires group-disjoint evaluation before promotion',
                    records=eligible, excluded=len(rows)-len(eligible))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='feedback.sqlite')
    sub = ap.add_subparsers(dest='command', required=True)
    for command in ('predict', 'complete'):
        sp = sub.add_parser(command)
        sp.add_argument('json_file', type=Path)
        if command == 'complete':
            sp.add_argument('--prediction-id', required=True)
    sub.add_parser('report')
    sub.add_parser('export')
    a = ap.parse_args()
    ledger = Ledger(a.db)
    if a.command == 'predict':
        result = {'prediction_id': ledger.predict(json.loads(a.json_file.read_text()))}
    elif a.command == 'complete':
        ledger.complete(a.prediction_id, json.loads(a.json_file.read_text()))
        result = {'recorded': True}
    elif a.command == 'export':
        result = ledger.export()
    else:
        result = ledger.report()
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
