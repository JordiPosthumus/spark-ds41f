"""Isolated native bad_words enforcement/lifetime tests; no model or HTTP."""
import argparse,importlib.util,json,os,sys,time,types
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--tables',required=True);p.add_argument('--output',required=True);p.add_argument('--cuda-graph',action='store_true');args=p.parse_args()
os.environ['SPARK_DS41F_TEMPORARY_PHRASES']='1';os.environ['SPARK_DS41F_PHRASE_TABLES']=args.tables
import vllm.v1.worker.gpu.sample
R=Path(__file__).parent;S=R/'vllm/v1/worker/gpu/sample';pkg='vllm.v1.worker.gpu.sample.'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
rt=load(pkg+'loop_protection_tables',S/'loop_protection_tables.py')
nat=load(pkg+'loop_protection',S/'loop_protection.py')
bw=load(pkg+'bad_words_candidate',S/'bad_words.py')
stock=load(pkg+'stock_bad_words',R/'stock_bad_words.py')
fx=json.loads((R/'fixtures.json').read_text());policy,seqs=rt.get_policy();tables=rt.get_tables();vocab=129280;dev=torch.device('cuda:0')
wrap=lambda t:types.SimpleNamespace(gpu=t)
req=types.SimpleNamespace(max_num_reqs=4,vocab_size=vocab,device=dev,
 all_token_ids=wrap(torch.zeros((4,16384),device=dev,dtype=torch.int32)),
 total_len=wrap(torch.zeros(4,device=dev,dtype=torch.int32)),
 prompt_len=wrap(torch.zeros(4,device=dev,dtype=torch.int32)))
timer=nat.LoopProtectionState(req,24);block=bw.BadWordsState(req);reference=stock.BadWordsState(req)
slots={};report={'tests':[],'note':'Synthetic GPU enforcement evidence, not proof of model recovery.'}
def add(slot,output=(),seed=(),protected=True,client=False):
 lex,utf=rt.advance(tables,(0,0),seed);prompt=[77,88,99]+list(seed)
 meta=(2,lex,utf,min(len(seed),max(map(len,seqs))-1)) if protected else None
 ids=seqs if protected or client else []
 param=types.SimpleNamespace(_ds41f_loop_protection=meta,bad_words=policy if ids else [],bad_words_token_ids=ids)
 timer.add_request(slot,len(prompt),param);block.add_request(slot,param);reference.add_request(slot,param)
 history=prompt+list(output);req.all_token_ids.gpu[slot].zero_();req.all_token_ids.gpu[slot,:len(history)]=torch.tensor(history,device=dev,dtype=torch.int32)
 req.prompt_len.gpu[slot]=len(prompt);req.total_len.gpu[slot]=len(history)
 slots[slot]=(list(seed),list(output),protected,ids,prompt)
def flush():
 timer.apply_staged_writes();block.apply_staged_writes();reference.apply_staged_writes()
def cpu_active(seed,history):
 lex,utf=rt.advance(tables,(0,0),seed);words=0
 for t in history:
  if t in (1,128822,128825) or t<0 or t>=vocab:return False
  lex,utf,c,_=rt.feed_token(tables,lex,utf,t);words+=c
  if words>=2000:return False
 return True
def check(name,batch,compare_stock=False):
 flush();expanded=[];pos=[];inputs=[];indices=[];expect=[]
 for slot,drafts in batch:
  seed,out,protected,ids,prompt=slots[slot];indices.append(slot)
  for j in range(len(drafts)+1):
   expanded.append(slot);pos.append(j)
   hist=(seed[-(max(map(len,seqs))-1):] if protected else [])+out+list(drafts[:j])
   active=not protected or cpu_active(seed,out+list(drafts[:j]))
   masked={s[-1] for s in ids if active and (len(s)==1 or (len(hist)>=len(s)-1 and hist[-(len(s)-1):]==s[:-1]))}
   expect.append(masked)
  inputs.extend([out[-1] if out else prompt[-1]]+list(drafts))
 gpu=lambda x:torch.tensor(x,dtype=torch.int32,device=dev)
 em,im,inp,lp=gpu(expanded),gpu(indices),gpu(inputs),gpu(pos);npidx=np.asarray(indices,dtype=np.int32)
 logits=torch.full((len(pos),vocab),1.25,device=dev)
 flags,floor=timer.rows(len(pos),em,im,npidx,inp,lp)
 block.apply_bad_words(logits,em,npidx,inp,lp,flags,floor)
 actual=logits.cpu().numpy()
 for i,want in enumerate(expect):
  got=set(np.flatnonzero(np.isneginf(actual[i])).tolist());assert got==want,(name,i,got^want)
  assert np.all(actual[i,~np.isneginf(actual[i])]==1.25)
 if compare_stock:
  original=torch.full_like(logits,1.25);reference.apply_bad_words(original,em,npidx,inp,lp)
  assert torch.equal(original,logits),(name,'stock differs')
 report['tests'].append({'name':name,'rows':len(pos),'passed':True})
 return logits,(em,im,npidx,inp,lp)
try:
 for slot in range(4):add(slot,protected=False)
 for phrase in policy:
  ids=fx[phrase]
  for i in range(len(ids)):
   add(0,output=ids[:i]);logits,_=check('phrase '+phrase+' prefix '+str(i),[(0,[])],True)
   if i==len(ids)-1:assert torch.isneginf(logits[0,ids[i]])
 for text in ['Let mego.','Let meprint.','Let me看.','Hmmh.','OKwait.']:
  ids=fx[text];caught=False
  for i in range(len(ids)):
   add(0,output=ids[:i]);logits,_=check('derivative '+text+' prefix '+str(i),[(0,[])],True)
   caught=caught or bool(torch.isneginf(logits[0,ids[i]]))
  assert caught,text
 for text in ['actually','ACTUALLY','Let memory work.','Safe reasoning. ']:
  ids=fx[text]
  for i in range(len(ids)):
   add(0,output=ids[:i]);logits,_=check('legitimate '+text+' prefix '+str(i),[(0,[])],True)
   assert not torch.isneginf(logits[0,ids[i]]),(text,i)
 add(0,seed=fx['Let']);check('activation across retained prefix',[(0,[])])
 add(0);add(1,protected=False);add(2,protected=False,client=True)
 check('mixed requests speculative all positions',[(2,fx['Let mego.']),(1,fx['Hmmh.']),(0,fx['Let meprint.'])],True)
 add(0);check('unreachable draft placeholders',[(0,[-1,42,6002])])
 for control in (1,128822,128825):
  add(0);check('speculative boundary '+str(control),[(0,[control,42,6002])])
  add(0,output=[control]);check('committed boundary '+str(control),[(0,[42,6002])])
 add(0,output=fx['safe '*1999]);check('word 2000 inside speculative rows',[(0,fx['safe ']+fx['Hmm'])])
 add(0,output=fx['safe '*2000]);check('committed 2000 word expiry',[(0,fx['Hmm'])])
 timer.remove_request(0);add(0,protected=False);check('expired slot reused unprotected',[(0,fx['Hmm'])],True)
 add(0);check('active slot reused',[(0,fx['Let me'])],True)
 add(0,output=fx['Let']);check('accepted history only',[(0,[])],True)
 check('same accepted history twice',[(0,[])],True)
 add(0);logits,data=check('graph preparation',[(0,fx['Let me'])],True)
 em,im,npidx,inp,lp=data
 def apply():
  flags,floor=timer.rows(logits.shape[0],em,im,npidx,inp,lp)
  block.apply_bad_words(logits,em,npidx,inp,lp,flags,floor)
 for _ in range(5):apply()
 torch.cuda.synchronize()
 if args.cuda_graph:
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):apply()
  graph.replay();torch.cuda.synchronize();report['tests'].append({'name':'CUDA graph capture/replay','passed':True})
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
 for _ in range(100):apply()
 end.record();end.synchronize();report['synthetic_ms_per_apply']=start.elapsed_time(end)/100
 report['passed']=True
finally:
 report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report.get('passed',False),'tests':len(report['tests']),'synthetic_ms_per_apply':report.get('synthetic_ms_per_apply')}))
