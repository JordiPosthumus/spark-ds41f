"""Portable CPU acceptance test; SPARK_TEST_TOKENIZER points to model tokenizer.json."""
import codecs,hashlib,json,os,random,re,sys,time,unicodedata
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
EXPERIMENT=Path(os.environ.get('SPARK_TEST_EXPERIMENT',ROOT.parent))
COMPILER=EXPERIMENT/'compiler'
sys.path.insert(0,str(COMPILER))
from compiler import category,byte_inverse,utf_feed,LEX,unpack,IDS,NO_HIT,set_unicode_policy
from runtime import load_tables,feed_token,advance,initial
set_unicode_policy(COMPILER/'target-unicode.json')
TOKENIZER=Path(os.environ.get('SPARK_TEST_TOKENIZER',EXPERIMENT/'tokenizer.json'))
if not TOKENIZER.is_file():
    raise SystemExit('Set SPARK_TEST_TOKENIZER to the model tokenizer.json path.')
TABLES=EXPERIMENT/'tables'
x=json.loads(TOKENIZER.read_bytes());added={i['id']:i for i in x['added_tokens']};inv=byte_inverse()
raws={i:bytes(inv[c] for c in s) for s,i in x['model']['vocab'].items() if i not in added}
raws.update({i:a['content'].encode() for i,a in added.items()})
t=load_tables(TABLES);tokenizer=Tokenizer.from_file(str(TOKENIZER));rng=random.Random(703)
byteid={r[0]:i for i,r in raws.items() if len(r)==1}
assert len(byteid)==256
for token_id,raw in raws.items():
    assert tokenizer.decode([token_id],skip_special_tokens=False)==raw.decode("utf8","replace"),token_id


def run_ids(ids):
    lex=utf=words=0;hit=255
    for i in ids:
        lex,utf,n,h=feed_token(t,lex,utf,i)
        if hit==255 and h!=255:hit=words+h
        words+=n
    return lex,utf,words,hit


def expected(text):
    # Independent complete-word regex oracle; unfinished trailing word is not
    # counted until another scalar establishes its boundary.
    words=list(re.finditer(r'\w+',text));complete=[w for w in words if w.end()<len(text)]
    hits=list(re.finditer(r'(?<!\w)(?:hm{2,16}|let\s+me|ok)(?!\w)',text,re.I))
    hits=[h for h in hits if h.end()<len(text)]
    first=min((sum(w.end()<=h.end() for w in complete) for h in hits),default=255)
    return len(complete),first

cases=[
    'OK.','ok.','Okay.','smoke.','Let me goo.','Let me write.','let memory.', 'Hmm.','hMm —','HMMMMMMMMMMMMMMMM,','hmmmmmmmmmmmmmmmmm.','hmmfoo ',
    'ahmm.','hmm_variable ', 'hmmé ', 'hmm你 ', 'hmm\u0301 ', 'hmm\u00a0tail ',
    'Let\tme\nRUN!','let me go.','Let me running.','let me gone.','let me\u00a0run.',
    'let. me run.', 'let me gogogo.', 'let me let me go.','hmm let me run. Hmm.',
    'Hmm — let me run — let me go. hmmmm.','go run let me normalwords.','\u2028Hmm\u2029',
    'hmm\x00','hmm9 ', 'hmm\U0001D7CE ', 'hmm\U0001F600 ', '嗯Hmm.', ' hmm',
]
for text in cases:
    want=expected(text)
    paths=[tokenizer.encode(text,add_special_tokens=False).ids,[byteid[b] for b in text.encode()]]
    for ids in paths:
        got=run_ids(ids)
        assert got[2:]==want,(text,got,want,ids)
# Alternate token paths that fuse punctuation/newlines (old escape).
for text in ['Hmm.\n\n','HMM,\n','Let me run.\n','hMm—actually ']:
    assert run_ids(tokenizer.encode(text,add_special_tokens=False).ids)[2:]==expected(text)
# Every supported m count and all mixed-case patterns up through m=6.
for n in range(2,17):
    word='h'+'m'*n
    for _ in range(40):
        text=''.join(c.upper() if rng.randrange(2) else c for c in word)+rng.choice(['.','—','\n','\t',',',' '])
        assert run_ids(tokenizer.encode(text,add_special_tokens=False).ids)[2:]==(1,1),text
# Activation in middle of prefix or multibyte word; scalar state walk equivalence.
for text in ['Let me run.','hMm,','hmmé ','let me\u00a0go.','hmm\U0001F600 ']:
    ids=[byteid[b] for b in text.encode()]
    for k in range(len(ids)+1):
        state=advance(t,initial(),ids[:k]);state=advance(t,state,ids[k:])
        assert state==run_ids(ids)[:2]
# Cross-UTF8-edge bridge is independently compared to uncompressed byte walker
# across every reachable pending UTF8 state and every lexical state subset.
checks=0
for utf in range(t['unicode_table'].shape[0]):
    choices=rng.sample(list(raws),60)+[byteid[b] for b in [0,32,46,72,104,128,159,160,191,192,224,237,240,244,255]]
    for lex in [0,1,IDS['hmm'],IDS['let_ws'],IDS['me_ws']]:
        for tid in choices:
            u=utf;s=lex;words=0;hit=255
            for b in raws[tid]:
                u,events=utf_feed(t['unicode_table'],u,b)
                for cl in events:
                    s,n,h=unpack(LEX[s,cl])
                    if hit==255 and h!=255:hit=words+h
                    words+=n
            got=feed_token(t,lex,utf,tid)
            assert got==(s,u,words,hit),(utf,lex,tid,got,(s,u,words,hit),raws[tid])
            checks+=1
# Random malformed byte streams match Python's incremental UTF8 replacement
# behavior, and independent word/phrase regex above.
for _ in range(300):
    data=b'Hmm'+bytes(rng.randrange(256) for _ in range(20))+b' let me run. '+bytes(rng.randrange(256) for _ in range(8))
    text=codecs.getincrementaldecoder('utf8')('replace').decode(data,final=False)
    got=run_ids([byteid[b] for b in data])
    assert got[2:]==expected(text),(data,text,got,expected(text))
# Exhaustively verify scalar classification (including Unicode edge characters)
# under split-byte paths; bridge/lexer intermediate state can retain fragments.
for cp in range(0x110000):
    if 0xD800<=cp<=0xDFFF:continue
    c=chr(cp);u=0;events=[]
    for b in c.encode():
        u,delta=utf_feed(t['unicode_table'],u,b);events+=delta
    assert u==0 and events==[category(c)],(cp,u,events)
report={'status':'pass','text_cases':len(cases),'mixedcase_cases':600,'bridge_comparisons':checks,'malformed_streams':300,'unicode_scalar_values':1112064,'vocab_size':len(raws),'tokenizer_decode_equivalence':len(raws),'notes':'No GPU/model/runtime calls; exact emitted-output effectiveness still requires native acceptance tests.'}
(ROOT/'test-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
