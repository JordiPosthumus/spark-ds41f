"""Captured-failure shapes, normal prose, exact prefix masking, and real-wrapper retry.
The engine here is a transport double; native/model evidence is separate.
"""
from test_api_semantics import (unittest, TokenGuard, ids, PhraseDensity, TemporaryPhraseTrigger, Engine, row, params, generate_with_recovery, RecoveryConfig)

class RegressionTests(unittest.IsolatedAsyncioTestCase):
 def test_all_let_me_continuations_and_ok_mask(self):
  for text in ['Let me go.','Let me goo.','Let me write.','Let me do it.','LET ME think.','let\tme\nreconsider.','OK.','ok,','hMm.']:
   g=TokenGuard();g.begin();g.feed(ids(text));self.assertGreater(g.violations,0,text)
  for text in ['Okay.','token.','Let memory work.','Let method.','hmmfoo.']:
   g=TokenGuard();g.begin();g.feed(ids(text));self.assertEqual(g.violations,0,text)
 def test_detect_actual_repetition_without_keyword(self):
  for phrase in ['OK.\n\n','Let me goo.\n\n','Writing now.\n\n']:
   for width in [1,7,1000]:
    d=PhraseDensity();text=phrase*9
    for i in range(0,len(text),width):d.feed(text[i:i+width],i,i+width)
    self.assertIsNotNone(d.detected,(phrase,width))
 def test_fenced_repetitions_and_normal_prose_not_loops(self):
  for text in ['```text\n'+'OK.\n'*50+'```\n','Let me check the file. The first entry contains a valid name. Actually, the second entry needs a date. Hmm, the third is complete.']:
   d=PhraseDensity()
   for i,c in enumerate(text):d.feed(c,i,i+1)
   self.assertIsNone(d.detected)
 def test_let_me_trigger_covers_different_verbs(self):
  d=TemporaryPhraseTrigger();text='Let me write. Each separate calculation has a useful new result. '*40
  for i,c in enumerate(text):d.feed(c,i,i+1)
  self.assertTrue(d.triggered)
 def test_exact_2000_word_expiry(self):
  g=TokenGuard();g.begin();g.feed(ids('safe '*1999));self.assertTrue(g.active)
  g.feed(ids('OK.'));self.assertGreater(g.violations,0);self.assertFalse(g.active)
  violations=g.violations;g.feed(ids('Let me go. OK.'));self.assertEqual(g.violations,violations)
 async def test_actual_detector_rolls_back_then_arms_retry(self):
  prefix='The required value is 42. '
  scripts=[[row(prefix)]+[row('OK.\n\n') for _ in range(12)], [row('The calculation gives forty-two. </think>42',True)]]
  e=Engine(scripts);events=[];original=params();prompt={'type':'token','prompt_token_ids':[11,12]}
  results=[x async for x in generate_with_recovery(e,prompt,original,'regression',config=RecoveryConfig(mode='auto-recover',normal_holdback_tokens=0,temporary_phrases=True),detector_factory=PhraseDensity,on_event=events.append,reasoning_ended=False)]
  self.assertEqual(len(e.calls),2)
  self.assertEqual(e.calls[1][0]['prompt_token_ids'],prompt['prompt_token_ids']+ids(prefix))
  self.assertIsNotNone(e.calls[1][1]._ds41f_phrase_suppression)
  self.assertNotEqual(e.calls[1][1].seed,original.seed)
  self.assertEqual(events[-1]['retries'],1);self.assertTrue(events[-1]['completed'])
  self.assertTrue(any(x.get('cause')=='loop_retry' and x.get('word_limit')==2000 for x in events))
  self.assertEqual(len(e.aborts),1)

if __name__=='__main__':unittest.main()
