"""One explicit synthetic HTTP case; plan-only unless --execute. No lifecycle operations."""
import argparse,hashlib,json,math,re,time,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METRICS=('vllm:num_requests_running','vllm:num_requests_waiting','vllm:generation_tokens_total')
def metrics(opener,url):
 with opener.open(url+'/metrics',timeout=10) as response:text=response.read(4000000).decode()
 values={}
 for key in METRICS:
  found=re.findall(r'^'+re.escape(key)+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)(?:\s|$)',text,re.M)
  if not found:raise ValueError('missing metric '+key)
  value=sum(map(float,found))
  if not math.isfinite(value) or value<0:raise ValueError('invalid metric '+key)
  values[key]=value
 return {'monotonic':time.monotonic(),**values}
def run(case,origin,out,disconnect_seconds):
 u=urllib.parse.urlsplit(origin)
 if u.scheme!='http' or u.hostname not in {'127.0.0.1','localhost','::1'} or not u.port or u.path not in {'','/'} or u.username or u.password or u.query or u.fragment:raise ValueError('explicit loopback origin only')
 if out.exists():raise ValueError('new result path required')
 body=json.loads((ROOT/'requests'/f'{case}.json').read_text());payload=json.dumps(body).encode()
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));before=metrics(opener,origin)
 if before[METRICS[0]] or before[METRICS[1]]:raise RuntimeError('engine not idle')
 start=time.monotonic();first=None;done=False;disconnected=False;rows=[];content=[];reasoning=[];tools={};usage=None
 request=urllib.request.Request(origin+'/v1/chat/completions',data=payload,headers={'Content-Type':'application/json','X-Request-Id':body['request_id']})
 with opener.open(request,timeout=90) as response:
  for raw in response:
   line=raw.decode().strip()
   if not line.startswith('data: '):continue
   if line=='data: [DONE]':done=True;break
   obj=json.loads(line[6:]);rows.append(obj)
   if obj.get('error'):raise RuntimeError('server SSE error')
   if obj.get('usage'):usage=obj['usage']
   for choice in obj.get('choices',[]):
    delta=choice.get('delta',{})
    if delta.get('content'):content.append(delta['content'])
    if delta.get('reasoning') or delta.get('reasoning_content'):reasoning.append(delta.get('reasoning') or delta.get('reasoning_content'))
    for tool in delta.get('tool_calls',[]):
     dest=tools.setdefault(tool['index'],{'name':'','arguments':''})
     for key,value in tool.get('function',{}).items():
      if key in dest:dest[key]+=value
    if first is None and any(delta.get(k) for k in ('content','reasoning','reasoning_content','tool_calls')):first=time.monotonic()
   if case=='mechanism-02' and time.monotonic()-start>=disconnect_seconds:
    disconnected=True;break
 ended=time.monotonic();after=[]
 for _ in range(32):
  after.append(metrics(opener,origin))
  if len(after)>=3:
   tail=after[-3:]
   if tail[-1]['monotonic']-tail[0]['monotonic']>=.45 and all(x[METRICS[0]]==x[METRICS[1]]==0 for x in tail) and len({x[METRICS[2]] for x in tail})==1:break
  time.sleep(.25)
 idle=len(after)>=3 and all(x[METRICS[0]]==x[METRICS[1]]==0 for x in after[-3:]) and len({x[METRICS[2]] for x in after[-3:]})==1 and after[-1]['monotonic']-after[-3]['monotonic']>=.45
 tool_json_valid=all(isinstance(json.loads(v['arguments']),dict) for v in tools.values()) if tools else None
 result={'case':case,'request_id':'chatcmpl-'+body['request_id'],'run':'a71012c0ffee','mode':'injected-auto-recover' if case.startswith('mechanism') else 'auto-recover','body_sha256':hashlib.sha256(payload).hexdigest(),'start_monotonic':start,'end_monotonic':ended,'ttft_seconds':None if first is None else first-start,'elapsed_seconds':ended-start,'done':done,'http_disconnected':disconnected,'valid_sse_rows':len(rows),'content_chars':sum(map(len,content)),'reasoning_chars':sum(map(len,reasoning)),'tool_count':len(tools),'tool_json_valid':tool_json_valid,'usage':usage,'metrics_before':before,'fresh_metrics_after_cleanup':after,'fresh_idle':idle}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if not idle or (case!='mechanism-02' and not done) or (case=='tool' and not tool_json_valid):raise SystemExit(2)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['mechanism-00','mechanism-01','mechanism-02','normal','tool'],required=True);p.add_argument('--origin',default='http://127.0.0.1:8000');p.add_argument('--out',type=Path);p.add_argument('--disconnect-seconds',type=float,default=6);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps({'case':a.case,'origin':a.origin,'execute':False}))
 else:
  if a.out is None:p.error('--out required')
  run(a.case,a.origin.rstrip('/'),a.out,a.disconnect_seconds)
