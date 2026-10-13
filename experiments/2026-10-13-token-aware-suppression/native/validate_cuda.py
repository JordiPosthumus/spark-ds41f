#!/usr/bin/env python3
"""Isolated synthetic-tensor CUDA validation. Never loads a model or sends HTTP.

Run ONLY in an owner-approved idle GPU window. Example:
  python validate_cuda.py --tables /tmp/candidate/tables --output result.json
All candidate modules are loaded by explicit path into this process only.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import types

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--tables',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--iterations',type=int,default=100)
parser.add_argument('--cuda-graph',action='store_true')
args=parser.parse_args()
assert 1<=args.iterations<=1000
os.environ['SPARK_DS41F_TEMPORARY_PHRASES']='1'
os.environ['SPARK_DS41F_PHRASE_TABLES']=str(args.tables.resolve())
import sys
# Import native package first, then replace only the new candidate module names.
import vllm.v1.worker.gpu.sample

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod
pkg='vllm.v1.worker.gpu.sample.'
source=ROOT/'vllm/v1/worker/gpu/sample'
runtime=load(pkg+'phrase_suppression_tables',source/'phrase_suppression_tables.py')
native=load(pkg+'phrase_suppression_candidate',source/'phrase_suppression.py')
tables=runtime.get_tables()
fixture=json.loads((ROOT/'cuda-fixtures.json').read_text())
text=fixture['text'];byte=fixture['bytes'];controls=set(fixture['controls'])
vocab=len(tables['token_prefix_id']);device=torch.device('cuda:0')
assert torch.cuda.is_available()
req=types.SimpleNamespace(
    max_num_reqs=4,vocab_size=vocab,device=device,
    all_token_ids=types.SimpleNamespace(gpu=torch.zeros((4,1024),device=device,dtype=torch.int32)),
    total_len=types.SimpleNamespace(gpu=torch.zeros(4,device=device,dtype=torch.int32)),
)
start=time.perf_counter();holder=native.PhraseSuppressionState(req);torch.cuda.synchronize()
report={'initialization_seconds':time.perf_counter()-start,'cuda_device':torch.cuda.get_device_name(),
        'tests':[], 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        'table_archive_sha256':hashlib.sha256((args.tables/'tables.npz').read_bytes()).hexdigest(),
        'note':'Synthetic kernel validation only; not production stability or end-to-end rescue evidence.'}
ref_states=np.zeros((4,8),dtype=np.int32)
prompts={};outputs={};guarded={}

def step_cpu(state,token):
    lex,utf,words,active=state
    if not active:return state
    if token in controls or token < 0 or token >= vocab:return lex,utf,words,False
    lex,utf,n,_=runtime.feed_token(tables,lex,utf,int(token))
    return lex,utf,min(60,words+n),words+n<60

@lru_cache(maxsize=256)
def expected_mask(lex,utf,words,active):
    mask=np.zeros(vocab,dtype=bool)
    if active:
        for tok in range(vocab):
            if tok not in controls:
                _,_,_,bad=runtime.feed_token(tables,lex,utf,tok)
                mask[tok]=bad!=255 and words+bad<=60
    return mask

def set_output(slot,ids):
    outputs[slot]=list(ids)
    ids=prompts[slot]+list(ids)
    req.all_token_ids.gpu[slot].zero_()
    req.all_token_ids.gpu[slot,:len(ids)].copy_(torch.tensor(ids,device=device,dtype=torch.int32))
    req.total_len.gpu[slot]=len(ids)

def add(slot,seed_ids=(),enabled=True,initial_output=()):
    lex,utf=runtime.advance(tables,runtime.initial(),seed_ids)
    prompts[slot]=[77,88,99]
    guarded[slot]=enabled
    metadata=(1,lex,utf) if enabled else None
    holder.add_request(slot,len(prompts[slot]),types.SimpleNamespace(_ds41f_phrase_suppression=metadata))
    if enabled:ref_states[slot]=[lex,utf,0,3,1,lex,utf,3]
    else:ref_states[slot]=[0,0,0,3,0,0,0,3]
    set_output(slot,initial_output)

def remove(slot):
    holder.remove_request(slot);ref_states[slot]=0;guarded[slot]=False

def update_expected(slot):
    state=ref_states[slot]
    if not state[4]:return
    end=len(prompts[slot])+len(outputs[slot]);scan=int(state[3])
    lex,utf,words,active=int(state[0]),int(state[1]),int(state[2]),bool(state[4])
    if scan>end:lex,utf,scan,words=int(state[5]),int(state[6]),int(state[7]),0
    assert scan>=len(prompts[slot])
    while scan<end and active:
        lex,utf,words,active=step_cpu((lex,utf,words,active),outputs[slot][scan-len(prompts[slot])]);scan+=1
    state[:5]=[lex,utf,words,end,int(active)]

def inputs(batch):
    slots=[];expanded=[];positions=[];ids=[]
    for slot,drafts in batch:
        slots.append(slot)
        expanded.extend([slot]*(1+len(drafts)))
        positions.extend(range(1+len(drafts)))
        ids.extend([outputs[slot][-1] if outputs[slot] else prompts[slot][-1]]+list(drafts))
    npidx=np.asarray(slots,dtype=np.int32)
    gpu=lambda x:torch.tensor(x,device=device,dtype=torch.int32)
    return gpu(expanded),gpu(slots),npidx,gpu(ids),gpu(positions)

def check(name,batch):
    holder.apply_staged_writes()
    for slot,_ in batch:update_expected(slot)
    data=inputs(batch)
    logits=torch.full((len(data[-1]),vocab),1.25,device=device,dtype=torch.float32)
    holder.apply(logits,*data)
    torch.cuda.synchronize()
    actual=logits.cpu().numpy();actual_states=holder.state.gpu.cpu().numpy()
    assert np.array_equal(actual_states,ref_states),(name,'state',actual_states.tolist(),ref_states.tolist())
    row=0;masked=0
    for slot,drafts in batch:
        st=ref_states[slot]
        initial=(int(st[0]),int(st[1]),int(st[2]),bool(st[4]))
        for pos in range(len(drafts)+1):
            current=initial
            for tok in drafts[:pos]:current=step_cpu(current,int(tok))
            expected=expected_mask(*current)
            got=np.isneginf(actual[row])
            assert np.array_equal(got,expected),(name,'mask row',row,'unexpected',np.flatnonzero(got!=expected)[:10].tolist())
            assert np.all(actual[row,~expected]==1.25),(name,'unmasked logits modified')
            masked+=int(got.sum());row+=1
    report['tests'].append({'name':name,'rows':row,'masked_entries':masked,'passed':True})
    return data,logits

try:
    from validate_serialization import validate as validate_serialization
    report['serialization']=validate_serialization()
    for slot in range(4):add(slot,enabled=False)
    add(0,text['hmm']);add(1,enabled=False)
    check('canonical newline leak and mixed active/inactive requests',[(0,[]),(1,text['draft_hmm'])])
    add(0)
    check('all speculative prefixes and bonus rows',[(0,text['draft_hmm'])])
    check('unreachable negative draft placeholders',[(0,[-1]+text['draft_hmm'])])
    # Only the accepted prefix enters committed history. Rejected draft suffix
    # from the previous mask computation must not survive into this step.
    set_output(0,text['draft_hmm'][:1])
    check('accepted prefix only; rejected suffix forgotten',[(0,[])])
    add(0,text['let_me'])
    check('multiword whitespace phrase across drafts',[(0,text['draft_go'])])
    for control in sorted(controls):
        add(0,text['hmm'])
        check(f'control {control} inside speculative batch',[(0,[control]+text['draft_hmm'])])
    add(0,text['hmm']+[byte['e2']])
    check('split UTF8 punctuation continuation',[(0,[byte['80'],byte['94']])])
    add(0,text['hmm']+[byte['c3']])
    check('split UTF8 letter is not word boundary',[(0,[byte['a9']])])
    add(0,initial_output=text['fifty_nine'])
    check('word60 remains protected and word61 free',[(0,text['draft_hmm']+text['space']+text['draft_hmm'])])
    add(0,initial_output=text['sixty'])
    check('TTL expired in committed output',[(0,text['draft_hmm'])])
    # Cancellation and immediate slot reuse before staged writes are flushed.
    remove(0);add(0,enabled=False)
    check('cancel and immediate inactive reuse in same batch',[(0,text['draft_hmm'])])
    remove(0);add(0,text['hmm']);remove(0);add(0,text['let_me_go'])
    check('multiple removes and active reuse before staged flush',[(0,[])])
    # Re-admission after preemption gets the original seed and current retained
    # generated output, exactly as RequestState.add_request supplies it.
    remove(0);add(0,text['let_me'],initial_output=text['ordinary'])
    check('preemption and readmission with retained output',[(0,text['draft_hmm'])])
    # Batch slot indices and row order must be independent.
    add(2,text['hmm_mixed']);add(3,enabled=False)
    check('noncontiguous reordered request slots',[(3,text['draft_hmm']),(2,[]),(0,[])])
    # GPU state updates are incremental, not duplicate token consumption.
    check('same committed state applied twice',[(2,[]),(0,[])])
    # Validating bad metadata cannot result in a silently unprotected request.
    for bad in [(0,0,0),(1,-1,0),(1,31,0),(1,0,99999),(1,True,0),(1,0)]:
        try:holder.add_request(3,3,types.SimpleNamespace(_ds41f_phrase_suppression=bad))
        except ValueError:pass
        else:raise AssertionError(('accepted invalid internal metadata',bad))
    report['tests'].append({'name':'invalid internal metadata rejected','passed':True})
    # Synthetic sampling overhead: no model, no reset/memcpy in timed region.
    def timing(batch,label):
        data=inputs(batch);logits=torch.ones((len(data[-1]),vocab),device=device)
        holder.apply_staged_writes()
        for _ in range(5):holder.apply(logits,*data)
        torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record()
        for _ in range(args.iterations):holder.apply(logits,*data)
        end.record();end.synchronize()
        return {'label':label,'iterations':args.iterations,'rows':len(data[-1]),
                'gpu_ms_per_apply':start.elapsed_time(end)/args.iterations,
                'wall_ms_per_apply':(time.perf_counter()-wall)*1000/args.iterations}
    add(0,text['hmm']);add(1,enabled=False)
    report['timing']=[timing([(1,[])],'inactive one request'),
                      timing([(0,[])],'active one request'),
                      timing([(0,text['draft_hmm'])],'active speculative rows')]
    if args.cuda_graph:
        # Optional sampler-only graph test. Production model graphs have a
        # separate scope; this does not assert full engine graph integration.
        data=inputs([(0,[])]);logits=torch.ones((1,vocab),device=device)
        for _ in range(3):holder.apply(logits,*data)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):holder.apply(logits,*data)
        graph.replay();torch.cuda.synchronize()
        assert torch.isneginf(logits[0,339]).item()
        report['tests'].append({'name':'sampler-only CUDA graph replay','passed':True})
    report['passed']=True
finally:
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'test_cases':len(report['tests']),'timing':report['timing'],'output':str(args.output)},indent=2))
