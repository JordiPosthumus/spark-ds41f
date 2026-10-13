import importlib.util, json, sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/"overlay"))
from ds41f_phrase_recovery.dynamic_words import DynamicWords, spellings
P=json.loads((E/"tables/word_policy.json").read_text())["dynamic"]
spec=importlib.util.spec_from_file_location("native_dynamic",E/"native/vllm/v1/worker/gpu/sample/dynamic_bad_words.py")
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
class Words(unittest.TestCase):
 def test_exact_threshold_and_chunks(self):
  for chunk in (1,2,9,1000):
   d=DynamicWords(P);text='Hh. '*12;out=[]
   for i in range(0,len(text),chunk):out+=d.feed(text[i:i+chunk])
   self.assertEqual(out,[])
   self.assertEqual(d.feed('Hh.'),['Hh']);self.assertEqual(d.feed('Hh.'),[])
 def test_window_expiry(self):
  d=DynamicWords(P);d.feed('Hh. '*12);d.feed(' '*3001)
  self.assertEqual(d.feed('Hh.'),[]);self.assertEqual(d.counts['hh'],1)
 def test_grammar_and_base(self):
  for word in P['grammar_exclusions']:
   self.assertEqual(DynamicWords(P).feed((word+' ')*15),[],word)
  self.assertEqual(DynamicWords(P,['Let','me']).feed('Let me. '*20),[])
  for word in ['let','me','okay','actually','hm']:
   self.assertNotIn(word,P['grammar_exclusions'])
 def test_code_excluded_across_chunks(self):
  for chunk in (1,7,3000):
   for text in ['```python\n'+'widget '*20+'\n```\n','~~~js\n'+'widget '*20+'\n~~~\n','`'+'widget '*20+'`\n','    '+'widget '*20+'\n']:
    d=DynamicWords(P);out=[]
    for i in range(0,len(text),chunk):out+=d.feed(text[i:i+chunk])
    self.assertEqual(out,[],text)
    self.assertEqual(d.feed('widget '*13),['widget'])
 def test_case_nfkc_and_punctuation(self):
  d=DynamicWords(P);self.assertEqual(d.feed('Hh, HH. hh! '*4),[])
  self.assertEqual(d.feed('Ｈｈ.'),['Ｈｈ'])
  self.assertEqual(spellings('actually'),['Actually'])
 def test_long_words_and_memory(self):
  d=DynamicWords(P);d.feed('x'*50000+' ')
  self.assertEqual(d.blocked,{})
  d.feed('word '*10000)
  self.assertLessEqual(len(d.rows),601);self.assertLessEqual(len(d.word),3000)
 def test_numbers_not_blocked(self):
  self.assertEqual(DynamicWords(P).feed('123 '*20),[])
 def test_new_words_remain_after_window(self):
  d=DynamicWords(P);self.assertEqual(d.feed('Hh '*13),['Hh']);d.feed(' '*4000)
  self.assertIn('hh',d.blocked);self.assertEqual(d.feed('Hh '*13),[])
class State:
 def __init__(self):self.rows={};self.writes=0
 def add_request(self,idx,p):self.rows[idx]=p.bad_words_token_ids
 def apply_staged_writes(self):self.writes+=1
class Native(unittest.TestCase):
 def runner(self):
  return NS(req_states=NS(req_id_to_index={'a':0,'b':1}),ds41f_dynamic_records={'a':(0,[]),'b':(0,[])},sampler=NS(dynamic_bad_words_state=State(),sampling_states=NS(vocab_size=129280),loop_protection_state=NS(use_protection=[True,True])))
 def test_request_isolation_and_version(self):
  r=self.runner();self.assertEqual(N.update(r,'a',[[4,5]],1)['status'],'applied')
  self.assertEqual(r.sampler.dynamic_bad_words_state.rows,{0:[[4,5]]})
  N.update(r,'a',[[8]],1);self.assertEqual(r.sampler.dynamic_bad_words_state.rows,{0:[[4,5]]})
  self.assertEqual(N.update(r,'gone',[[8]],2)['status'],'finished')
 def test_preemption_restore(self):
  r=self.runner();del r.req_states.req_id_to_index['a']
  self.assertEqual(N.update(r,'a',[[9]],1)['status'],'deferred')
  r.req_states.req_id_to_index['a']=2;N.restore(r,'a',2)
  self.assertEqual(r.sampler.dynamic_bad_words_state.rows[2],[[9]])
 def test_invalid_update_does_not_mutate(self):
  r=self.runner()
  for ids in ([[129280]],[[]],[[True]],[[1]]*2049):
   with self.assertRaises(ValueError):N.update(r,'a',ids,1)
   self.assertEqual(r.sampler.dynamic_bad_words_state.rows,{})
if __name__=='__main__':unittest.main(verbosity=2)
