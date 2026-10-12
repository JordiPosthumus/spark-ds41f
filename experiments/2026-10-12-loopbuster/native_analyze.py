"""Exact run+full request ID audit. Native test injections do not validate detector quality."""
import argparse,json
from pathlib import Path
def analyze(log,result):
 rid=result['request_id'];run=result['run'];mode=result['mode'];events=[]
 for line in log.splitlines():
  marker='ds41f_loopbuster_canary '
  if marker not in line:continue
  event=json.loads(line.split(marker,1)[1])
  if event.get('request_id')==rid and event.get('run')==run and event.get('mode')==mode:events.append(event)
 if sum(e['event']=='canary_selected' for e in events)!=1:raise ValueError('missing or duplicated lifecycle')
 if any(e['event']=='canary_skipped' for e in events):raise ValueError('canary skipped')
 starts=[e for e in events if e['event']=='native_attempt_start'];injected=[e for e in events if e['event']=='canary_injected']
 expected=3 if rid.endswith('-01') else 2
 if len(starts)!=expected or len(injected)!=expected-1:raise ValueError('injected/native retry counts do not match')
 keys=('exact_original_prompt','multimodal_identity_preserved','kwargs_preserved','budget_correct','sampler_preserved','fresh_retry_seed','previous_abort_and_close_complete')
 if any(not all(s.get(k) is True for k in keys) for s in starts):raise ValueError('native invariant failed')
 if any(not s.get('temporary_words_active') or s.get('temporary_expiry')!=48 or s.get('temporary_stop_ids_valid') is not True for s in starts[1:]):raise ValueError('temporary retry word mask not configured')
 if rid.endswith('-01') and any(s.get('image_placeholders')!=6 for s in starts):raise ValueError('six-image placeholders missing')
 if [s['attempt'] for s in starts]!=list(range(expected)):raise ValueError('duplicated or missing attempt')
 native_ids={s['native_request_id'] for s in starts}
 if len(native_ids)!=expected or any(not(n==rid or n.startswith(rid+'.loop-retry-')) for n in native_ids):raise ValueError('foreign native lifecycle')
 closed={e['native_request_id'] for e in events if e['event']=='native_iterator_closed'}
 if not native_ids<=closed:raise ValueError('missing native iterator close')
 if not result['fresh_idle']:raise ValueError('fresh idle check failed')
 if rid.endswith('-02'):
  last=starts[-1]['native_request_id'];aborted={e['native_request_id'] for e in events if e['event']=='native_abort_completed'}
  if not result['http_disconnected'] or result['done'] or last not in aborted:raise ValueError('HTTP disconnect not proven after retry')
 return {'request_id':rid,'run':run,'native_attempts':len(starts),'native_abort_restart_mechanics_passed':True,'detector_quality_proven':False,'fresh_idle':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.log.read_text(),json.loads(a.result.read_text())),indent=2))
