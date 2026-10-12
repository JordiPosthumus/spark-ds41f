"""Explicit tiny native-kernel test. Plan-only default. No model/inference/lifecycle work."""
import argparse,hashlib,json
from pathlib import Path

def run(expected):
 import numpy as np
 import torch
 from types import SimpleNamespace as NS
 from vllm.sampling_params import SamplingParams
 from vllm.v1.worker.gpu.sample import bad_words as native
 actual=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError('loaded bad_words source hash mismatch')
 T=lambda v:torch.tensor(v,dtype=torch.int32,device='cuda')
 checks=[]
 def check(name,output_len,positions,until,expected_mask,history=None,draft=None,mapping=None,expiry=True):
  mapping=mapping or [0]*len(positions);nreq=len(output_len)
  logits=torch.zeros((len(positions),16),dtype=torch.float32,device='cuda')
  allids=T(history if history is not None else [[0]*300 for _ in range(nreq)])
  kwargs={} if not expiry else {'bad_words_until':T(until),'bad_words_stop0':T([128822]*nreq),'bad_words_stop1':T([128825]*nreq)}
  native.apply_bad_words(logits,T(mapping),T([[3]]*nreq),T([[0,1]]*nreq),T([1]*nreq),allids,T([0]*nreq),T(output_len),T(draft or [0]*(len(positions)+2)),T(positions),1,**kwargs)
  torch.cuda.synchronize();observed=torch.isneginf(logits[:,3]).tolist()
  if observed!=expected_mask:raise AssertionError((name,observed,expected_mask))
  checks.append({'test':name,'passed':True,'masked':observed})
 check('expiry_speculative_bonus_rows',[47],[0,1,2],[48],[True,False,False])
 check('mixed_temporary_and_permanent_requests',[47,49],[0,1,2,0,1,2],[48,-1],[True,False,False,True,True,True],mapping=[0,0,0,1,1,1])
 check('original_apply_api_without_expiry',[50],[0,1],[-1],[True,True],expiry=False)
 h=[[0]*300];h[0][3]=128822
 check('committed_reasoning_end_releases_words',[4],[0,1],[48],[False,False],history=h)
 check('speculative_dsml_marker_releases_next_row',[0],[0,1,2],[48],[True,False,False],draft=[0,128825,5,0,0])
 holder=lambda value:NS(gpu=T(value))
 req=NS(max_num_reqs=2,device=torch.device('cuda'),all_token_ids=holder([[0]*300]*2),prompt_len=holder([0,0]),total_len=holder([50,50]))
 state=native.BadWordsState(req)
 def sp(words,temporary):
  p=SamplingParams();p._bad_words_token_ids=words;p.extra_args={'ds41f_loopbuster_bad_words_until':48,'ds41f_loopbuster_stop_ids':[128822,128825]} if temporary else None;return p
 state.add_request(0,sp([[3]],True));state.add_request(1,sp(None,False));state.apply_staged_writes()
 state.add_request(0,sp(None,False));state.apply_staged_writes()
 assert state.num_bad_words.np[0]==0 and state.bad_words_until.np[0]==-1 and state.bad_words_stop0.np[0]==-1
 logits=torch.zeros((1,16),device='cuda');state.apply_bad_words(logits,T([0]),np.array([0],dtype=np.int32),T([0]),T([0]));torch.cuda.synchronize();assert torch.isfinite(logits).all().item()
 state.add_request(0,sp([[3]],False));state.apply_staged_writes()
 logits.zero_();state.apply_bad_words(logits,T([0]),np.array([0],dtype=np.int32),T([0]),T([0]));torch.cuda.synchronize();assert torch.isneginf(logits[0,3]).item()
 checks.append({'test':'slot_reuse_empty_then_permanent','passed':True})
 return {'source_sha256':actual,'native_gpu_kernel_executed':True,'model_loaded':False,'checks':checks}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--expected-source-sha',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
 if not a.execute:print(json.dumps({'execute':False,'test':'native TTL expiry, speculative boundaries, mixed requests, slot reuse'}))
 else:
  if a.out is None or a.out.exists():p.error('--out must be a new result path')
  result=run(a.expected_source_sha);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
