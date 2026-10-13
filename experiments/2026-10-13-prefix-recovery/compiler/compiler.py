"""Isolated prototype: whole-word temporary suppression table compiler.

No production runtime/config edits. ByteLevel tokenizer only; special tokens
handled by caller. Pin Unicode database version with compiled artifact.
"""
from __future__ import annotations
import argparse, hashlib, json, time, unicodedata
from pathlib import Path
import numpy as np

D,S,W=0,1,2
# Stable explicit IDs shared with sampler.
LETTER_IDS={c:i+3 for i,c in enumerate('hmletkungo')}
NAMES=['boundary','otherword']+['h'+'m'*i for i in range(17)]+['l','le','let','let_ws','m','me','me_ws','o','ok','unused1','unused2','unused3']
IDS={s:i for i,s in enumerate(NAMES)}
NO_HIT=255
UNICODE_POLICY=None
UNICODE_CLASSES=None

def set_unicode_policy(path):
    global UNICODE_POLICY,UNICODE_CLASSES
    raw=Path(path).read_bytes();policy=json.loads(raw)
    assert policy["classes"]=={"boundary":0,"space":1,"word":2}
    classes=bytearray(0x110000);previous=-1
    for start,end,cl in policy["ranges"]:
        assert 0<=start<=end<0x110000 and start>previous and cl in (1,2)
        classes[start:end+1]=bytes([cl])*(end-start+1);previous=end
    UNICODE_CLASSES=classes
    UNICODE_POLICY={"unicode_version":policy["unicode_version"],"sha256":hashlib.sha256(raw).hexdigest()}
    return UNICODE_POLICY

def pack(state,words=0,hit=NO_HIT):
    assert state<32 and 0<=words<256 and (hit==255 or 1<=hit<255)
    return state | words<<5 | hit<<13

def unpack(x):return int(x)&31,(int(x)>>5)&255,(int(x)>>13)&255

def is_wordstate(state):return NAMES[state] not in ('boundary','let_ws','me_ws')

def terminal(state):
    name=NAMES[state]
    return name in ('me','ok') or (name.startswith('h') and 3<=len(name)<=17)

def lex_step(state,cl):
    name=NAMES[state]; word=is_wordstate(state)
    if cl in (D,S):
        new='boundary'
        if cl==S:
            if name in ('let','let_ws'):new='let_ws'
            elif name in ('me','me_ws'):new='me_ws'
        return pack(IDS[new],int(word),1 if word and terminal(state) else NO_HIT)
    letter=next((k for k,v in LETTER_IDS.items() if v==cl),None)
    if name in ('boundary','let_ws','me_ws'):
        if name=='let_ws' and letter=='m':new='m'
        elif letter in ('h','l','o'):new=letter
        else:new='otherword'
    elif name.startswith('h'):
        new=name+'m' if letter=='m' and len(name)<17 else 'otherword'
    else:
        edges={'l':('e','le'),'le':('t','let'),'m':('e','me'),'o':('k','ok')}
        new=edges[name][1] if name in edges and letter==edges[name][0] else 'otherword'
    return pack(IDS[new])

LEX=np.array([[lex_step(s,c) for c in range(13)] for s in range(len(NAMES))],dtype=np.int32)

def category(c):
    if c in LETTER_IDS:return LETTER_IDS[c]
    if 'A'<=c<='Z' and c.lower() in LETTER_IDS:return LETTER_IDS[c.lower()]
    if UNICODE_CLASSES is not None:return UNICODE_CLASSES[ord(c)]
    if c=='_' or unicodedata.category(c)[0] in 'LN':return W
    if c.isspace():return S
    return D

def compile_unicode():
    root={}
    for cp in range(128,0x110000):
        if 0xD800<=cp<=0xDFFF:continue
        c=chr(cp); seq=c.encode(); d=root
        for b in seq[:-1]:d=d.setdefault(b,{})
        d[seq[-1]]=category(c)
    intern={}; nodes=[]
    def minim(d):
        key=tuple((b,minim(v) if isinstance(v,dict) else -1-v) for b,v in sorted(d.items()))
        if key not in intern:intern[key]=len(nodes);nodes.append(key)
        return intern[key]
    rootid=minim(root)
    # Root is zero for caller; reindex all node references accordingly.
    reorder=[rootid]+[i for i in range(len(nodes)) if i!=rootid]
    lookup={old:new for new,old in enumerate(reorder)}
    table=np.full((len(nodes),256),-99,dtype=np.int32)
    for new,old in enumerate(reorder):
        for b,q in nodes[old]:table[new,b]=lookup[q] if q>=0 else q
    for b in range(128):table[0,b]=-1-category(chr(b))
    return table

def utf_feed(table,state,byte):
    out=[]; q=int(table[state,byte])
    if q==-99:
        if state:out.append(D)
        state=0;q=int(table[0,byte])
        if q==-99:return 0,out+[D]
    if q<0:return 0,out+[-1-q]
    return q,out

def decode_classes(table,raw):
    state=0;out=[]
    for b in raw:
        state,events=utf_feed(table,state,b);out.extend(events)
    return tuple(out),state

def byte_inverse():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    return {chr(c):b for b,c in zip(bs,cs)}

def transition(classes,state):
    words=0;hit=NO_HIT
    for cl in classes:
        state,n,h=unpack(LEX[state,cl])
        if hit==NO_HIT and h!=NO_HIT:hit=words+h
        words+=n
    return pack(state,words,hit)

def build(tokenizer_path,out,unicode_policy=None):
    set_unicode_policy(unicode_policy or Path(__file__).with_name("target-unicode.json"))
    t=time.monotonic();source=Path(tokenizer_path).read_bytes();tok=json.loads(source)
    assert tok['decoder']['type']=='ByteLevel'
    added={x['id']:x for x in tok['added_tokens']}
    vocab=tok['model']['vocab'];V=1+max([*vocab.values(),*added])
    inverse=byte_inverse();utf=compile_unicode()
    prefixes={};bodies={};body_keys=[];prefix_keys=[]
    def body_id(key):
        if key not in bodies:bodies[key]=len(body_keys);body_keys.append(key)
        return bodies[key]
    def prefix_id(key):
        if key not in prefixes:prefixes[key]=len(prefix_keys);prefix_keys.append(key)
        return prefixes[key]
    empty=body_id(((),0,False));baseprefix=prefix_id(((),False))
    token_prefix=np.full(V,baseprefix,dtype=np.int32);token_body=np.full(V,empty,dtype=np.int32)
    special=np.zeros(V,dtype=np.uint8);raw_by_id={}
    all_raw={i:bytes(inverse[c] for c in s) for s,i in vocab.items() if i not in added}
    all_raw.update({i:item['content'].encode('utf8') for i,item in added.items()})
    for i,raw in sorted(all_raw.items()):
        raw_by_id[i]=raw;n=0
        while n<len(raw) and 128<=raw[n]<192:n+=1
        prefix,body=raw[:n],raw[n:]
        classes,tail=decode_classes(utf,body)
        token_prefix[i]=prefix_id((tuple(prefix),bool(body)))
        token_body[i]=body_id((classes,tail,bool(body)))
    for i in added:special[i]=1
    assert len(utf)<512 and max(len(p[0]) for p in prefix_keys)<=3
    bridge=np.zeros((len(utf),len(prefixes)),dtype=np.int32)
    for u in range(len(utf)):
        for p,(raw,has_body) in enumerate(prefix_keys):
            state=u;events=[]
            for byte in raw:
                state,delta=utf_feed(utf,state,byte);events.extend(delta)
            if has_body and state:events.append(D);state=0
            assert len(events)<=3 and all(c in (D,S,W) for c in events),(u,p,events)
            packed=state | len(events)<<9
            for j,c in enumerate(events):packed|=c<<(11+4*j)
            bridge[u,p]=packed
    body_table=np.array([[transition(k[0],s) for k in body_keys] for s in range(len(NAMES))],dtype=np.int32)
    arrays={'lex_table':LEX,'unicode_table':utf,'bridge_table':bridge,'body_table':body_table,
            'token_prefix_id':token_prefix,'token_body_id':token_body,
            'body_nonempty':np.array([k[2] for k in body_keys],dtype=np.uint8),
            'body_tail':np.array([k[1] for k in body_keys],dtype=np.int32),'special':special}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'tables.npz',**arrays)
    archive_sha=hashlib.sha256((out/'tables.npz').read_bytes()).hexdigest()
    meta={'archive_sha256':archive_sha,'format_version':1,'tokenizer_sha256':hashlib.sha256(source).hexdigest(),'unicode_version':UNICODE_POLICY['unicode_version'],'unicode_policy_sha256':UNICODE_POLICY['sha256'],
          'lex_states':NAMES,'class_ids':{'boundary':D,'space':S,'otherword':W,**LETTER_IDS},
          'vocab_size':V,'body_count':len(bodies),'prefix_count':len(prefixes),'unicode_states':len(utf),
          'tables':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.nbytes} for k,v in arrays.items()},
          'bytes_total':sum(v.nbytes for v in arrays.values()),'compile_seconds':time.monotonic()-t,
          'special_tokens':{str(i):x['content'] for i,x in added.items()}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('special_tokens','lex_states','class_ids')},indent=2))
    return arrays,raw_by_id

class Machine:
    def __init__(self,tables,state=0,utf=0):self.t=tables;self.state=state;self.utf=utf
    def token(self,token_id):
        words=0;hit=NO_HIT
        b=int(self.t['bridge_table'][self.utf,self.t['token_prefix_id'][token_id]])
        self.utf=b&511
        for j in range((b>>9)&3):
            self.state,n,h=unpack(self.t['lex_table'][self.state,(b>>(11+4*j))&15])
            if hit==NO_HIT and h!=NO_HIT:hit=words+h
            words+=n
        bid=int(self.t['token_body_id'][token_id])
        self.state,n,h=unpack(self.t['body_table'][self.state,bid])
        if hit==NO_HIT and h!=NO_HIT:hit=words+h
        words+=n
        if self.t['body_nonempty'][bid]:self.utf=int(self.t['body_tail'][bid])
        return self.state,self.utf,words,hit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tokenizer');p.add_argument('out');p.add_argument('--unicode-policy');a=p.parse_args();build(a.tokenizer,a.out,a.unicode_policy)
