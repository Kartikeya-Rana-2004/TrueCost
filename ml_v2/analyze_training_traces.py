"""V2 diagnostics: training-only OOF errors joined to retained trace counters.
Post-run fields generated here are diagnostic, never prediction features.
"""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,default=Path('/Users/kay/Downloads/truecost_raw_visible_evidence.tar.gz'));ap.add_argument('--starter',type=Path,default=Path('/Users/kay/Downloads/ml_starter'));a=ap.parse_args()
 v1=HERE.parent/'releases/v1.0.0/package/TrueCost_Final_ML_Evaluation'
 preds=pd.read_csv(v1/'results_seed440/oof_predictions.csv');assert len(preds)==8396
 rows=pd.read_csv(a.starter/'data/20261020T021459Z/audited_visible_runs.csv');rows=rows[rows.partition.eq('training')].copy()
 assert set(preds.run_id)==set(rows.run_id)
 preds['absolute_error']=abs(preds.rich_boost_absolute-preds.total_tokens)
 selected_groups=set(preds.groupby('group_id').absolute_error.max().nlargest(30).index)
 selected_ids=set(preds.loc[preds.group_id.isin(selected_groups),'run_id']);ids=set(rows.run_id)
 ledgers={};reports={};actions={};names={};hashes={}
 # Streaming read; never extract paths or load calibration trace contents.
 with tarfile.open(a.archive,'r|gz') as tf:
  for member in tf:
   parts=Path(member.name).parts
   if 'runs' not in parts or not member.isfile():continue
   at=parts.index('runs');rid=parts[at+1]
   if rid not in ids:continue
   base=parts[-1];isledger=base=='ledger.jsonl';isreport=base=='report.json';israw=rid in selected_ids and base=='raw.json' and parts[-2].startswith('request-')
   if not (isledger or isreport or israw):continue
   data=tf.extractfile(member).read();key=(rid,parts[-2] if israw else base)
   digest=hashlib.sha256(data).hexdigest()
   if key in hashes:
    assert hashes[key]==digest,'Conflicting retained evidence';continue
   hashes[key]=digest
   if isledger:ledgers[rid]=[json.loads(line) for line in data.splitlines() if line.strip()]
   elif isreport:reports[rid]=json.loads(data)
   else:
    raw=json.loads(data);content=raw.get('message',{}).get('content','')
    try: action=json.loads(content).get('action','unknown')
    except (ValueError,AttributeError):action='invalid_json'
    actions.setdefault(rid,[]).append(str(action))
   if len(reports) and len(reports)%2000==0 and isreport:print('Training reports read',len(reports),flush=True)
 assert set(ledgers)==set(reports)==ids,(len(ledgers),len(reports))
 records=[]
 for r in rows.to_dict('records'):
  rid=r['run_id'];ledger=ledgers[rid];report=reports[rid]
  assert [x['index'] for x in ledger]==list(range(1,len(ledger)+1))
  assert report['partition']=='training'
  for c in ['input_tokens','output_tokens','total_tokens','requests']:
   assert report[c]==r[c],(rid,c)
  inputs=np.array([x['input_tokens'] for x in ledger]);outputs=np.array([x['output_tokens'] for x in ledger]);assert inputs.sum()==r['input_tokens'] and outputs.sum()==r['output_tokens']
  records.append(dict(run_id=rid,requests=len(ledger),termination=report['termination'],output_cap_requests=int((outputs==1024).sum()),hit_output_cap=bool((outputs==1024).any()),first_request_input=int(inputs[0]),input_after_first_request=int(inputs[1:].sum()),input_after_first_share=float(inputs[1:].sum()/max(1,inputs.sum())),max_request_input=int(inputs.max()),last_request_input=int(inputs[-1]),diagnostic_actions=' | '.join(actions.get(rid,[]))))
 trace=pd.DataFrame(records);d=preds.merge(trace,on='run_id',validate='one_to_one');d['request_cap']=d.requests.eq(6);d['underprediction']=d.total_tokens-d.rich_boost_absolute
 out=HERE/'trace_analysis';out.mkdir(exist_ok=True);d.to_csv(out/'training_diagnostics.csv',index=False)
 summaries=[]
 for col in ['family','model','requests','request_cap','hit_output_cap','termination']:
  for label,g in d.groupby(col):summaries.append(dict(stratum=col,label=label,n=len(g),groups=g.group_id.nunique(),mae=g.absolute_error.mean(),bias=-g.underprediction.mean(),mean_tokens=g.total_tokens.mean(),total_absolute_error=g.absolute_error.sum(),request_cap_rate=g.request_cap.mean(),output_cap_rate=g.hit_output_cap.mean(),mean_input_after_first_share=g.input_after_first_share.mean()))
 pd.DataFrame(summaries).to_csv(out/'stratum_diagnostics.csv',index=False)
 d[d.group_id.isin(selected_groups)].sort_values('absolute_error',ascending=False).to_csv(out/'largest_error_groups.csv',index=False)
 top=d.nlargest(int(np.ceil(len(d)*.01)),'absolute_error');cap=d[d.request_cap]
 note=dict(training_runs=len(d),task_groups=d.group_id.nunique(),raw_archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),request_cap_runs=len(cap),request_cap_error_share=float(cap.absolute_error.sum()/d.absolute_error.sum()),top_one_percent_runs=len(top),top_one_percent_error_share=float(top.absolute_error.sum()/d.absolute_error.sum()),output_cap_runs=int(d.hit_output_cap.sum()),diagnostic_only_fields=list(trace.columns[1:]),calibration_or_confirmation_traces_used=False,limitations=['Input after first request includes newly retrieved context and history; it is not a direct measure of waste.','Output at 1024 tokens is a ceiling indicator, not proof that a larger limit would improve quality.','Associations with post-run behavior diagnose errors but are not causal estimates.','Export/ledger consistency checks do not replace original strict raw-evidence audit.'])
 (out/'summary.json').write_text(json.dumps(note,indent=2));print(json.dumps(note,indent=2))
if __name__=='__main__':main()
