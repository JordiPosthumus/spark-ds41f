import asyncio, copy, importlib.util, json, os, sys, unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
E=Path(__file__).resolve().parents[1]
T=Path(os.environ['SPARK_TEST_TOKENIZER'])
os.environ['SPARK_DS41F_PHRASE_TABLES']=str(E/'tables')
sys.path.insert(0,str(E/'overlay'))
name='vllm.v1.worker.gpu.sample.loop_protection_tables'
spec=importlib.util.spec_from_file_location(name,E/'native/vllm/v1/worker/gpu/sample/loop_protection_tables.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[name]=module
from tokenizers import Tokenizer
from ds41f_phrase_recovery.recovery import generate_with_recovery, RecoveryConfig
from ds41f_phrase_recovery.phrase_detector import PhraseDensity
from ds41f_phrase_recovery.collapse import Detection
TOK=Tokenizer.from_file(str(T));HINT=[]
def ids(s):return TOK.encode(s,add_special_tokens=False).ids
class Kind(IntEnum):DELTA=1
def params(**updates):
 p=NS(n=1,max_tokens=10000,output_kind=Kind.DELTA,seed=42,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,min_tokens=0,stop=[],stop_token_ids=[],bad_words=[],detokenize=True,extra_args=None)
 p.__dict__.update(updates);return p
def row(text,final=False):return (text,ids(text),final)
def output(text,tokens,final=False):return NS(request_id='test',prompt='prompt',prompt_token_ids=[11,12],prompt_logprobs=None,encoder_prompt=None,encoder_prompt_token_ids=None,num_cached_tokens=0,num_cache_creation_tokens=0,metrics=None,finished=final,outputs=[NS(index=0,text=text,token_ids=tokens,finish_reason='stop' if final else None,stop_reason=None,spec_decode_metrics=None)])
class Engine:
 def __init__(self,scripts):self.scripts=scripts;self.calls=[];self.aborts=[];self.closed=[]
 async def generate(self,prompt,p,rid,**kw):
  i=len(self.calls);self.calls.append((prompt,copy.deepcopy(p),rid,kw))
  try:
   for text,tokens,final in self.scripts[i]:
    out=output(text,tokens,final);out.prompt_token_ids=prompt['prompt_token_ids'];yield out
  finally:self.closed.append(rid)
 async def abort(self,rid):self.aborts.append(rid)
def initial():return [row('Useful concrete prefix.\n\n')]+[row(x) for x in ['Let one. ','let two. ','LET three. ','Let four. ']]
class Tests(unittest.IsolatedAsyncioTestCase):
 async def run_case(self,scripts,detector=PhraseDensity,client=None,**kwargs):
  e=Engine(scripts);events=[];p=params(**kwargs);prompt={'type':'multimodal','prompt_token_ids':[11,12],'mm_kwargs':object()}
  out=[x async for x in generate_with_recovery(e,prompt,p,'test',config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True),detector_factory=detector,dynamic_client=client,on_event=events.append,reasoning_ended=False)]
  return e,events,out,p,prompt
 async def test_rewind_no_hint_notice_sampling_budget_and_mm(self):
  script=initial();e,ev,out,p,prompt=await self.run_case([script,[row('Recovered. </think>Done.',True)]])
  self.assertEqual(len(e.calls),2);self.assertEqual(len(e.aborts),1)
  resume,sp,_,_=e.calls[1]
  self.assertEqual(resume['prompt_token_ids'],[11,12]+script[0][1]+HINT)
  self.assertIs(resume['mm_kwargs'],prompt['mm_kwargs'])
  self.assertEqual(sp.bad_words,module.get_policy()[0]);self.assertEqual(sp._bad_words_token_ids,module.get_policy()[1]);self.assertEqual(p.bad_words,[])
  self.assertEqual(sp.temperature,1.);self.assertNotEqual(sp.seed,42)
  self.assertEqual(sp.max_tokens,p.max_tokens-sum(len(x[1]) for x in script)-len(HINT))
  notes=[x for x in out if getattr(x,'ds41f_loop_notice',None)]
  self.assertEqual(len(notes),2);self.assertEqual(notes[0].outputs,[])
  self.assertIn('LOOP PROTECTION MODE ACTIVATED',notes[0].ds41f_loop_notice);self.assertIn('LOOP PROTECTION MODE OFF',notes[1].ds41f_loop_notice);self.assertNotIn('I must not get stuck',notes[0].ds41f_loop_notice)
  visible=''.join(x.text for r in out for x in r.outputs)
  self.assertIn('Let one.',visible);self.assertNotIn('I must not get stuck',visible)
  self.assertEqual(ev[-1]['server_injected_guidance_tokens'],0);self.assertTrue(ev[-1]['completed'])
  self.assertEqual(ev[-1]['observed_generated_tokens'],sum(len(x[1]) for x in script)+len(ids('Recovered. </think>Done.')))
  self.assertEqual(ev[-1]['discarded_generated_tokens'],sum(len(x[1]) for x in script[1:]))
 async def test_split_spaced_hm_rewind(self):
  script=[row('Safe prefix.\n')]+[row(x) for x in ['H',' ','m','m','. ']*4]
  e,ev,out,_,_=await self.run_case([script,[row('Recovered.',True)]])
  self.assertEqual(len(e.calls),2);self.assertEqual(e.calls[1][0]['prompt_token_ids'],[11,12]+script[0][1]+HINT)
  a=next(x for x in ev if x['event']=='temporary_phrases_activated');self.assertEqual(a['family'],'hm');self.assertEqual(a['hits'],4)
 async def test_boundary_and_final_never_rewind(self):
  for ending in ['</think>Answer','<｜DSML｜calls>']:
   e,ev,_,_,_=await self.run_case([[row('Let go. '*4+ending),row('Done',True)]])
   self.assertEqual(len(e.calls),1)
  e,ev,_,_,_=await self.run_case([[row('Let go. '*4,True)]])
  self.assertEqual(len(e.calls),1)
 async def test_tiny_remaining_budget_no_injection(self):
  script=initial();budget=sum(len(x[1]) for x in script)+5
  e,ev,out,_,_=await self.run_case([script,[row('Done',True)]],max_tokens=budget)
  self.assertEqual(e.calls[1][1].max_tokens,5);self.assertEqual(e.calls[1][0]['prompt_token_ids'],[11,12]+script[0][1])
  self.assertFalse(any('reminder' in x['event'] for x in ev))
 async def test_severe_retry_no_hint(self):
  class Severe(PhraseDensity):
   def feed(self,text,start,end):
    if 'SEVERE' in text:return Detection(start,{'kind':'offline_test'})
    return super().feed(text,start,end)
  e,ev,_,_,_=await self.run_case([initial(),[row('SEVERE ')],[row('Recovered. </think>Done.',True)]],detector=Severe)
  self.assertEqual(len(e.calls),3);self.assertEqual(len(e.aborts),2)
  self.assertEqual(e.calls[2][0]['prompt_token_ids'],[11,12]+initial()[0][1]+HINT)
  self.assertEqual(ev[-1]['server_injected_guidance_tokens'],0);self.assertEqual(ev[-1]['retries'],1)
 async def test_expiry_no_extra_generation(self):
  e,ev,_,_,_=await self.run_case([initial(),[row(' '.join('word'+str(i) for i in range(2001))+' '),row('</think>Done',True)]])
  self.assertEqual(len(e.calls),2)
  release=next(x for x in ev if x['event']=='temporary_phrases_released');self.assertEqual(release['reason'],'word_limit');self.assertEqual(release['protected_words'],2000)
 async def test_dynamic_update_notices_without_extra_retry(self):
  class Client:
   encoded=[]
   def __init__(self):self.calls=[]
   async def update(self,rid,words):
    self.calls.append((rid,words));self.encoded=[[100]]
    return {'status':'applied','ranks':3,'sequences':1,'update_ms':1}
  client=Client()
  text=' '.join('Hh topic'+str(i)+'.' for i in range(13))+' '
  e,ev,out,p,prompt=await self.run_case([initial(),[row(text),row('</think>Done',True)]],client=client)
  self.assertEqual(len(e.calls),2);self.assertEqual(len(client.calls),1)
  self.assertEqual(client.calls[0][1],['Hh'])
  notes=[x.ds41f_loop_notice for x in out if hasattr(x,'ds41f_loop_notice')]
  self.assertEqual(len(notes),3);self.assertIn('ADDED: Hh CURRENT: Let, me',notes[1])
  self.assertIn('MODE OFF',notes[2]);self.assertEqual(ev[-1]['server_injected_guidance_tokens'],0)
  self.assertEqual(e.calls[1][1].temperature,1.)
  self.assertTrue(out[-1].finished)
 async def test_cancellation_at_notice_aborts_no_new_attempt(self):
  e=Engine([initial()]);ev=[]
  it=generate_with_recovery(e,{'type':'token','prompt_token_ids':[11,12]},params(),'cancel',config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True),detector_factory=PhraseDensity,on_event=ev.append,reasoning_ended=False)
  async for o in it:
   if hasattr(o,'ds41f_loop_notice'):break
  await it.aclose()
  self.assertEqual(len(e.calls),1);self.assertEqual(len(e.aborts),1);self.assertEqual(len(e.closed),1)
  self.assertFalse(ev[-1]['completed'])
if __name__=='__main__':unittest.main(verbosity=2)
