"""Run the existing frozen predictor and register predictions before task execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from feedback import Ledger


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ml-dir',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--db',required=True)
    p.add_argument('--controller-version',required=True)
    a=p.parse_args()
    ml=a.ml_dir.resolve()
    version=hashlib.sha256((ml/'selected_models/point.joblib').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'predictions.csv'
        subprocess.run([sys.executable,str(ml/'predict_tasks.py'),'--tasks',str(a.tasks.resolve()),'--out',str(out)],check=True,stdout=sys.stderr)
        rows=list(csv.DictReader(out.open()))
    ledger=Ledger(a.db)
    ids=[]
    for row in rows:
        pid=ledger.predict(dict(task_id=row['task_id'],group_id=row['group_id'],model=row['model'],
            model_version='sha256:'+version,controller_version=a.controller_version,partition='feedback',
            predicted_total_tokens=float(row['predicted_total_tokens']),
            upper_budget=float(row['calibrated_upper_budget']) if row.get('calibrated_upper_budget') else None))
        ids.append(dict(prediction_id=pid,task_id=row['task_id'],model=row['model']))
    print(json.dumps(ids,indent=2))

if __name__=='__main__':main()
