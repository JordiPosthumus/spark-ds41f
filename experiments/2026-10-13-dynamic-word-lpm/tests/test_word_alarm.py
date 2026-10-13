from pathlib import Path
import os,sys,types,json,importlib,unittest,shutil
exp=Path(__file__).resolve().parents[1];os.environ['SPARK_DS41F_PHRASE_TABLES']=str(exp/'tables')
pkg=types.ModuleType('alarmcheck');pkg.__path__=[str(exp/'overlay/ds41f_phrase_recovery')];sys.modules['alarmcheck']=pkg
m=importlib.import_module('alarmcheck.temporary_phrases')
def detect(text,chunk=1):
 d=m.TemporaryPhraseTrigger()
 for start in range(0,len(text),chunk):d.feed(text[start:start+chunk],start,start+chunk)
 d.feed(' ',len(text),len(text)+1)
 return d
class Checks(unittest.TestCase):
 def test_families_and_chunk_splits(self):
  samples={'let':['Let','let','LET','Lеt'],'me':['me','Me','ME','меня'],'ok':['OK','okay','Ok','OKAY'],'actually':['Actually','actually','ACTUALLY','Actually'],'hm':['Hm','Hmm','H mm','H m m']}
  for family,words in samples.items():
   for size in (1,2,7,99):
    with self.subTest(family=family,size=size):
     d=detect('. '.join(words)+'.',size);self.assertTrue(d.triggered);self.assertEqual(d.evidence['family'],family);self.assertEqual(d.evidence['hits'],4)
 def test_no_double_count(self):
  for t in ('Let me. '*3,'H m m. '*3,'Hm. Hmm. H mm.'):
   self.assertFalse(detect(t).triggered,t)
 def test_threshold_window_edges(self):
  self.assertTrue(detect('let '+'safe '*46+'let let let.').triggered)
  self.assertFalse(detect('let '+'safe '*47+'let let let.').triggered)
 def test_code_and_nonwords(self):
  for size in (1,7,999):
   for t in ('```text\n'+'let me Actually H mm '*20+'\n```\n','letter memory okayish actuallyness H,m. '*20):
    self.assertFalse(detect(t,size).triggered)
 def test_code_ages_hits(self):
  self.assertFalse(detect('let let let.\n```\n'+'x '*50+'\n```\nlet.').triggered)
 def test_hm_spacing_not_punctuation(self):
  self.assertTrue(detect('H\tmm. H\nmm. h mm. H MM.').triggered)
  self.assertFalse(detect('H,mm. H-mm. H.mm. H/mm.').triggered)
 def test_legacy_long_hmm(self):
  self.assertTrue(detect(('H'+'m'*16+'. ')*4).triggered)
 def test_policy_flatten(self):
  p=json.loads((exp/'tables/word_policy.json').read_text());flat=list(dict.fromkeys(v for x in p['families'].values() for v in x['block']))
  self.assertEqual(flat,json.loads((exp/'tables/bad_words.json').read_text()));self.assertIn('Actually',flat);self.assertNotIn('actually',flat)
if __name__ == '__main__': unittest.main()
