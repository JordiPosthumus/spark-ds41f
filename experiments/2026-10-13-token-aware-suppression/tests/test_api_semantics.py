"""Independent API integration checks using actual tokenizer and compiler tables.
Engine transport is a double; no GPU/model calls. This cannot prove mask enforcement.
"""
import asyncio,importlib.util,json,os,sys,unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent
EXPERIMENT=Path(os.environ.get('SPARK_TEST_EXPERIMENT',R.parent))
COMPILER=EXPERIMENT/'compiler'
OVERLAY=EXPERIMENT/'overlay'
TOKENIZER=Path(os.environ.get('SPARK_TEST_TOKENIZER',EXPERIMENT/'tokenizer.json'))
if not TOKENIZER.is_file():
 raise SystemExit('Set SPARK_TEST_TOKENIZER to the model tokenizer.json path.')
sys.path.insert(0,str(OVERLAY))
name='vllm.v1.worker.gpu.sample.phrase_suppression_tables'
helper=EXPERIMENT/'native/vllm/v1/worker/gpu/sample/phrase_suppression_tables.py'
# API tests can run before native packaging, using the same CPU table runtime.
if not helper.is_file():helper=COMPILER/'runtime.py'
spec=importlib.util.spec_from_file_location(name,helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
os.environ['SPARK_DS41F_PHRASE_TABLES']=str(EXPERIMENT/'tables')
if not hasattr(module,'get_tables'):
 from functools import lru_cache
 module.get_tables=lru_cache(maxsize=1)(lambda:module.load_tables(EXPERIMENT/'tables'))
sys.modules[name]=module
from tokenizers import Tokenizer
from ds41f_phrase_recovery.token_guard import TokenGuard
from ds41f_phrase_recovery.temporary_phrases import TemporaryPhraseTrigger
from ds41f_phrase_recovery.phrase_detector import PhraseDensity
from ds41f_phrase_recovery.collapse import Detection
from ds41f_phrase_recovery.recovery import generate_with_recovery,RecoveryConfig
TOK=Tokenizer.from_file(str(TOKENIZER))
def ids(s):return TOK.encode(s,add_special_tokens=False).ids
class Kind(IntEnum):DELTA=1

def params():return NS(n=1,max_tokens=10000,output_kind=Kind.DELTA,seed=42,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,min_tokens=0,stop=[],stop_token_ids=[],bad_words=[],detokenize=True,extra_args=None)
def output(text,tokenids,final=False):return NS(request_id='offline',prompt='prompt',prompt_token_ids=[11,12],prompt_logprobs=None,encoder_prompt=None,encoder_prompt_token_ids=None,num_cached_tokens=0,num_cache_creation_tokens=0,metrics=None,finished=final,outputs=[NS(index=0,text=text,token_ids=tokenids,finish_reason='stop' if final else None,stop_reason=None,spec_decode_metrics=None)])
class Trigger(TemporaryPhraseTrigger):
 def feed(self,text,start,end):
  self.triggered=True;self.evidence={'words_seen':10,'hmm':8,'commands':0,'window_words':200}
class Engine:
 def __init__(self,scripts):self.scripts=scripts;self.calls=[];self.aborts=[];self.closed=[]
 async def generate(self,prompt,p,rid,**kwargs):
  index=len(self.calls);self.calls.append((prompt,p,rid,kwargs))
  try:
   for text,tokenids,final in self.scripts[index]:
    row=output(text,tokenids,final);row.request_id=rid;row.prompt_token_ids=prompt['prompt_token_ids'];yield row
  finally:self.closed.append(rid)
 async def abort(self,rid):self.aborts.append(rid)

def row(text,final=False):return (text,ids(text),final)
class GuardTests(unittest.TestCase):
 def test_partialword_seed_violation_and_control_exception(self):
  guard=TokenGuard();guard.feed(ids('Initial Hmm'));meta=guard.begin()
  self.assertEqual(meta,(1,*guard.state));start=guard.offset
  guard.feed(ids('.\n\n'));self.assertEqual(guard.violations,1);self.assertEqual(guard.words,1)
  self.assertEqual(guard.start,start)
  guard=TokenGuard();guard.feed(ids('Initial Hmm'));guard.begin();guard.feed([128822])
  self.assertEqual(guard.violations,0);self.assertEqual(guard.finished,'reasoning_boundary');self.assertFalse(guard.active)
 def test_whole_word_negatives_and_unicode_prefix(self):
  for prefix,suffix in [('Hmm','foo '),('let me run','ning '),('hmm','é '),('hmm','你 ')]:
   guard=TokenGuard();guard.feed(ids(prefix));guard.begin();guard.feed(ids(suffix));self.assertEqual(guard.violations,0,(prefix,suffix))
  # Force UTF8 split at native byte tokens, not canonical encoding.
  sys.path.insert(0,str(COMPILER));from compiler import byte_inverse
  x=json.loads(TOKENIZER.read_bytes());inv=byte_inverse()
  by={bytes(inv[c] for c in s):i for s,i in x['model']['vocab'].items() if all(c in inv for c in s)}
  guard=TokenGuard();guard.feed(ids('hmm')+[by[b'\xc3']]);guard.begin();guard.feed([by[b'\xa9']]+ids(' '));self.assertEqual(guard.violations,0)
 def test_exact_sixty_words_with_chunk_overshoot(self):
  guard=TokenGuard();guard.begin();guard.feed(ids('safe '*59+'Hmm. Hmm.'))
  self.assertEqual(guard.violations,1);self.assertEqual(guard.words,60);self.assertEqual(guard.finished,'word_limit')
  self.assertEqual(guard.evidence()['protected_words'],60)
  self.assertGreater(guard.offset,guard.end)
 def test_rewind_then_reactivate_clears_prefix(self):
  guard=TokenGuard();guard.feed(ids('old Hmm'));guard.begin();guard.feed(ids('. more '))
  self.assertEqual(guard.total_violations,1)
  retained=ids('New meaningful ');guard.rewind(retained)
  self.assertEqual(guard.offset,len(retained));self.assertFalse(guard.active)
  expected=module.advance(module.get_tables(),module.initial(),retained)
  self.assertEqual(guard.begin(),(1,*expected));self.assertEqual(guard.violations,0);self.assertEqual(guard.total_violations,1)
 def test_two_requests_isolate_state(self):
  a=TokenGuard();b=TokenGuard();a.feed(ids('Hmm'));a.begin();b.begin();a.feed(ids('.'))
  self.assertEqual(a.violations,1);self.assertEqual(b.words,0);self.assertEqual(b.state,(0,0))

class TriggerTests(unittest.TestCase):
 def test_whitespace_only_between_command_words(self):
  for gap,wanted in [(' ',True),('\t',True),('\n',True),('\u00a0',True),('.',False),(' , ',False),('—',False)]:
   text=(('Let'+gap+'me'+gap+'run. meaningful concrete progress. ')*60)
   for width in [1,7,len(text)]:
    detector=TemporaryPhraseTrigger()
    for i in range(0,len(text),width):detector.feed(text[i:i+width],i,i+width)
    self.assertEqual(detector.triggered,wanted,(repr(gap),width,detector.evidence))
 def test_fences_and_substrings_dont_count(self):
  for text in [('hmmfoo let me running. normal meaningful work. '*80),'```text\n'+('Hmm. Let me go. '*100)+'\n```\n']:
   detector=TemporaryPhraseTrigger();detector.feed(text,0,len(text));self.assertFalse(detector.triggered)

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
 async def case(self,scripts,detector=PhraseDensity):
  engine=Engine(scripts);events=[];visible=[];original=params();mm=object();prompt={'type':'multimodal','prompt_token_ids':[11,12],'mm_kwargs':mm}
  async for out in generate_with_recovery(engine,prompt,original,'test',config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True,max_retries=1),detector_factory=detector,temporary_trigger_factory=Trigger,on_event=events.append,reasoning_ended=False):visible.append(out)
  return engine,events,visible,prompt,original
 async def test_activation_prefix_and_no_restart_at_expiry(self):
  e,events,out,p,original=await self.case([[row('Initial Hmm')],[row('foo '+('meaningful '*60)+'Hmm. '),row('</think>Done.',True)]])
  self.assertEqual(len(e.calls),2);self.assertEqual(len(e.aborts),1)
  expected=module.advance(module.get_tables(),module.initial(),ids('Initial Hmm'))
  self.assertEqual(e.calls[1][1]._ds41f_phrase_suppression,(1,*expected))
  self.assertIs(e.calls[1][0]['mm_kwargs'],p['mm_kwargs']);self.assertEqual(e.calls[1][0]['prompt_token_ids'],[11,12]+ids('Initial Hmm'))
  self.assertEqual(original.bad_words,[]);self.assertEqual(e.calls[1][1].bad_words,[])
  release=next(x for x in events if x['event']=='temporary_phrases_released')
  self.assertEqual(release['protected_words'],60);self.assertEqual(release['mask_violation_tokens'],0)
  self.assertTrue(events[-1]['completed'])
 async def test_natural_finish_retains_span_evidence(self):
  e,events,out,_,_=await self.case([[row('Initial normal ')],[row('Useful short conclusion.',True)]])
  self.assertEqual(len(e.calls),2)
  release=next(x for x in events if x['event']=='temporary_phrases_released')
  self.assertEqual(release['release_reason'],'request_finished');self.assertEqual(release['mask_violation_tokens'],0)
 async def test_severe_rewind_removes_active_metadata_and_keeps_accounting(self):
  class Severe(PhraseDensity):
   def feed(self,text,start,end):
    if 'SEVERE' in text:return Detection(0,{'kind':'offline_test'})
    return None
  e,events,out,p,original=await self.case([[row('Initial Hmm')],[row('foo SEVERE ')],[row('Recovered. </think>Done.',True)]],detector=Severe)
  self.assertEqual(len(e.calls),3);self.assertEqual(len(e.aborts),2)
  self.assertEqual(e.calls[2][0]['prompt_token_ids'],p['prompt_token_ids'])
  self.assertIsNone(getattr(e.calls[2][1],'_ds41f_phrase_suppression',None))
  self.assertEqual(next(x for x in events if x['event']=='temporary_phrases_released')['release_reason'],'loop_rewind')
  self.assertEqual(events[-1]['retries'],1);self.assertTrue(events[-1]['completed'])
 async def test_cancel_active_span_closes_abort_and_reports_evidence(self):
  class WaitEngine(Engine):
   def __init__(self):super().__init__([[row('Initial normal ')]]);self.waiting=asyncio.Event()
   async def generate(self,prompt,p,rid,**kw):
    if not self.calls:
     async for item in super().generate(prompt,p,rid,**kw):yield item
    else:
     self.calls.append((prompt,p,rid,kw));self.waiting.set()
     try:await asyncio.Event().wait()
     finally:self.closed.append(rid)
  e=WaitEngine();events=[]
  async def consume():
   async for out in generate_with_recovery(e,{'type':'token','prompt_token_ids':[11,12]},params(),'cancel',config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True,max_retries=1),detector_factory=PhraseDensity,temporary_trigger_factory=Trigger,on_event=events.append,reasoning_ended=False):pass
  task=asyncio.create_task(consume());await asyncio.wait_for(e.waiting.wait(),2);task.cancel()
  with self.assertRaises(asyncio.CancelledError):await task
  self.assertEqual(len(e.aborts),2);self.assertEqual(len(e.closed),2)
  release=next(x for x in events if x['event']=='temporary_phrases_released')
  self.assertEqual(release['release_reason'],'request_closed');self.assertEqual(release['protected_words'],0)
  self.assertFalse(events[-1]['completed'])
 async def test_control_boundary_ends_inplace(self):
  e,events,out,_,_=await self.case([[row('Initial normal ')],[row('Useful. </think>Hmm. Let me run.'),row('Final.',True)]])
  self.assertEqual(len(e.calls),2);self.assertEqual(len(e.aborts),1)
  release=next(x for x in events if x['event']=='temporary_phrases_released')
  self.assertEqual(release['release_reason'],'reasoning_boundary');self.assertEqual(release['mask_violation_tokens'],0)
  self.assertEqual(''.join(r.outputs[0].text for r in out).count('</think>'),1)

if __name__=='__main__':unittest.main(verbosity=2)
