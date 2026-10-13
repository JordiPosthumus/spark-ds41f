"""Transport-double regression tests; native sampling requires a separate GPU check."""
import asyncio, unittest, sys
from pathlib import Path
from types import SimpleNamespace as NS
from enum import IntEnum
sys.path.insert(0, str(Path(__file__).resolve().parent/'overlay'))
from ds41f_phrase_recovery.recovery import generate_with_recovery, RecoveryConfig
from ds41f_phrase_recovery.phrase_detector import PhraseDensity
from ds41f_phrase_recovery.temporary_phrases import TemporaryPhraseTrigger, WordCounter, PHRASES
class Kind(IntEnum):
 DELTA=1
 FINAL_ONLY=2
def params(**updates):
 return NS(**(dict(n=1,max_tokens=262144,output_kind=Kind.DELTA,seed=42,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,min_tokens=0,stop=[],stop_token_ids=[],bad_words=[],detokenize=True,extra_args=None)|updates))
def tokenizer():return None
def output(text,ids,final=False):
 return NS(request_id='test',prompt='prompt',prompt_token_ids=[11,12],prompt_logprobs=None,encoder_prompt=None,encoder_prompt_token_ids=None,num_cached_tokens=0,num_cache_creation_tokens=0,metrics=None,finished=final,outputs=[NS(index=0,text=text,token_ids=ids,finish_reason='stop' if final else None,stop_reason=None,spec_decode_metrics=None)])
class Engine:
 def __init__(self,tok,scripts,width):
  self.scripts,self.width=scripts,width
  self.calls=[];self.aborts=[];self.closed=[]
 async def generate(self,prompt,sampling,rid,**kwargs):
  attempt=len(self.calls);self.calls.append((prompt,sampling,rid,kwargs))
  try:
   text=self.scripts[min(attempt,len(self.scripts)-1)]
   for start in range(0,len(text),self.width):
    piece=text[start:start+self.width];row=output(piece,list(map(ord,piece)))
    row.request_id=rid;row.prompt_token_ids=prompt['prompt_token_ids'];yield row
   row=output('',[],True);row.prompt_token_ids=prompt['prompt_token_ids'];yield row
  finally:self.closed.append(rid)
 async def abort(self,rid):self.aborts.append(rid)
LOOP = 'Let me go. Let me write. Hmm. Actually, let me go.\n' * 400

class DetectorTests(unittest.TestCase):
    def test_case_and_chunk_boundaries(self):
        for width in (1,7,128):
            d=PhraseDensity(); text=LOOP.swapcase().replace(' ', '\t')
            for n in range(0,len(text),width):
                found=d.feed(text[n:n+width],n,n+width)
                if found:break
            self.assertIsNotNone(found)
    def test_hmm_variants(self):
        for word in ('Hmm,', 'HMMM.', 'hmmmm…', 'Actually,'):
            d=PhraseDensity();self.assertIsNotNone(d.feed((word+' ')*1000,0,1000))
    def test_isolated_and_fenced_usage(self):
        for text in ('Let me check. Actually, that is correct. Hmm, okay.', '```text\n'+LOOP+'```\n'):
            self.assertIsNone(PhraseDensity().feed(text,0,1000))

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,scripts,width=7,budget=10000,**kwargs):
        tok=tokenizer();engine=Engine(tok,scripts,width);events=[];visible=[]
        original=params(max_tokens=budget)
        mm=object();prompt={'type':'multimodal','prompt_token_ids':[11,12],'mm_kwargs':mm}
        async for out in generate_with_recovery(engine,prompt,original,'test',
                config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0),
                detector_factory=PhraseDensity,on_event=events.append,reasoning_ended=False,**kwargs):
            visible.append(out)
        return engine,events,visible,prompt,original
    async def test_recovery_stream_prefix_matches_retry_prompt(self):
        engine,events,visible,prompt,original=await self.run_case([LOOP,'Recovered. </think>42.'])
        self.assertEqual(len(engine.calls),2)
        retry=engine.calls[1]
        retained=retry[0]['prompt_token_ids'][2:]
        published=[t for o in visible for t in o.outputs[0].token_ids]
        self.assertEqual(published[:len(retained)],retained)
        self.assertIs(retry[0]['mm_kwargs'],prompt['mm_kwargs'])
        self.assertNotEqual(retry[1].seed,original.seed)
        self.assertEqual(retry[1].temperature,original.temperature)
        self.assertLess(retry[1].max_tokens,original.max_tokens)
        self.assertEqual(retry[1].bad_words,[])
        account=events[-1]
        self.assertGreater(account['discarded_generated_tokens'],0)
        self.assertLess(sum(len(o.outputs[0].text) for o in visible),len(LOOP))
    async def test_model_rewind_removes_already_streamed_loop(self):
        engine,events,visible,prompt,original=await self.run_case([LOOP,'Recovered. </think>42.'])
        self.assertEqual(engine.calls[1][0]['prompt_token_ids'],prompt['prompt_token_ids'])
        account=events[-1]
        self.assertGreater(account['visible_completion_tokens'],0)
        self.assertGreater(account['discarded_generated_tokens'],account['unpublished_discarded_tokens'])
        self.assertEqual(account['visible_completion_tokens']+account['unpublished_discarded_tokens'],account['observed_generated_tokens'])
        self.assertEqual(visible[-1].ds41f_discarded_completion_tokens,account['unpublished_discarded_tokens'])

    async def test_normal_stream_and_tool_boundary_unchanged(self):
        for text in ('A useful thought. </think>42.', '</think>'+LOOP,
                     '<｜DSML｜ calls>'+LOOP):
            engine,events,visible,_,_=await self.run_case([text])
            self.assertEqual(len(engine.calls),1)
            self.assertEqual(''.join(o.outputs[0].text for o in visible),text)
    async def test_ten_retries_bounded_and_distinct(self):
        engine,events,visible,_,_=await self.run_case([LOOP],budget=100000)
        self.assertEqual(len(engine.calls),11)
        self.assertEqual(len(set(c[1].seed for c in engine.calls)),11)
        self.assertEqual(len(engine.closed),11)
        self.assertEqual(visible[-1].outputs[0].stop_reason,'ds41f_reasoning_loop')
        budgets=[c[1].max_tokens for c in engine.calls]
        self.assertTrue(all(a>b for a,b in zip(budgets,budgets[1:])))

    async def test_cancellation_waits_for_owned_abort(self):
        class Waiting(Engine):
            def __init__(self):
                super().__init__(tokenizer(),[LOOP],7)
                self.waiting=asyncio.Event();self.aborting=asyncio.Event();self.release=asyncio.Event()
            async def generate(self,prompt,sampling,rid,**kwargs):
                if not self.calls:
                    async for row in super().generate(prompt,sampling,rid,**kwargs):yield row
                else:
                    self.calls.append((prompt,sampling,rid,kwargs))
                    self.waiting.set()
                    await asyncio.Event().wait()
            async def abort(self,rid):
                self.aborts.append(rid)
                if len(self.calls)>1:
                    self.aborting.set();await self.release.wait()
        engine=Waiting()
        async def consume():
            async for row in generate_with_recovery(engine,{'type':'token','prompt_token_ids':[11,12]},params(),'cancel',config=RecoveryConfig(mode='auto-recover'),detector_factory=PhraseDensity,reasoning_ended=False):pass
        task=asyncio.create_task(consume())
        await asyncio.wait_for(engine.waiting.wait(),5)
        task.cancel();await asyncio.wait_for(engine.aborting.wait(),1)
        task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done())
        engine.release.set()
        with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,1)
        self.assertEqual(len(engine.aborts),2)



class TriggerOnce(TemporaryPhraseTrigger):
 def feed(self,text,start,end):
  super().feed(text,start,end)
  if self.words>=8:
   self.triggered=True;self.evidence={'words_seen':self.words,'hmm':2,'commands':0,'window_words':self.words}
class TemporaryTests(unittest.IsolatedAsyncioTestCase):
 async def run_case(self,scripts,config=None,engine=None):
  engine=engine or Engine(tokenizer(),scripts,1);events=[];visible=[];original=params(max_tokens=3000)
  mm=object();prompt={'type':'multimodal','prompt_token_ids':[11,12],'mm_kwargs':mm}
  async for o in generate_with_recovery(engine,prompt,original,'test',config=config or RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True,max_retries=1),detector_factory=PhraseDensity,temporary_trigger_factory=TriggerOnce,on_event=events.append,reasoning_ended=False):visible.append(o)
  return engine,events,visible,prompt,original
 async def test_activation_expiry_metadata_budget(self):
  e,ev,vis,p,original=await self.run_case(['Hmm. One two three four five six seven eight nine. '*50,'Useful concrete progress. '*100,'</think>42'])
  self.assertEqual(len(e.calls),3)
  self.assertEqual(e.calls[1][1].bad_words,list(PHRASES));self.assertEqual(e.calls[2][1].bad_words,[])
  self.assertEqual(original.bad_words,[])
  for i,c in enumerate(e.calls):
   self.assertIs(c[0]['mm_kwargs'],p['mm_kwargs']);self.assertEqual(c[1].temperature,original.temperature)
   self.assertEqual(c[1].max_tokens,3000-(len(c[0]['prompt_token_ids'])-2))
  self.assertEqual(len(e.aborts),2);self.assertEqual(len(e.closed),3)
  self.assertEqual(next(x for x in ev if x['event']=='temporary_phrases_released')['words'],60)
  self.assertEqual(ev[-1]['observed_generated_tokens'],ev[-1]['visible_completion_tokens'])
  self.assertTrue(all(o.prompt_token_ids==[11,12] for o in vis))
  self.assertTrue(ev[-1]['completed'])
 async def test_boundary_removes_ban_keeps_parser_text(self):
  e,ev,vis,p,original=await self.run_case(['Hmm. One two three four five six seven eight nine. '*50,'Some work. </think><｜DSML｜calls>tool argument','finish'])
  self.assertEqual(len(e.calls),3)
  self.assertEqual(e.calls[2][1].bad_words,[])
  self.assertTrue(e.calls[2][3]['reasoning_ended'])
  self.assertEqual(''.join(o.outputs[0].text for o in vis).count('</think>'),1)
  self.assertEqual(next(x for x in ev if x['event']=='temporary_phrases_released')['reason'],'reasoning_boundary')
 async def test_natural_finish_does_not_restart(self):
  e,ev,vis,p,original=await self.run_case(['Hmm. One two three four five six seven eight nine. '*50,'Finished.'])
  self.assertEqual(len(e.calls),2);self.assertTrue(vis[-1].finished)
 def test_counter_chunking(self):
  c=WordCounter()
  for t in ['o','ne tw','o.\n','th','ree ']:c.feed(t)
  self.assertEqual(c.words,3)
 def test_trigger_case_fences(self):
  text='Hmm, Let me run. Let me go. distinct concrete measurement. '*100
  for width in [1,7,128]:
   d=TemporaryPhraseTrigger()
   for i in range(0,len(text),width):d.feed(text[i:i+width].swapcase(),i,i+width)
   self.assertTrue(d.triggered)
  d=TemporaryPhraseTrigger();d.feed('```\n'+text+'\n```\n',0,1000);self.assertFalse(d.triggered)
  d=TemporaryPhraseTrigger();d.feed('Hmm. Let me check the numbers. '+('Useful varied reasoning. '*300),0,1000);self.assertFalse(d.triggered)
if __name__=='__main__':unittest.main()
