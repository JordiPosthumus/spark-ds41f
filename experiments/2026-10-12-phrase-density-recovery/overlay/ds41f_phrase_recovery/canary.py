"""TEST ONLY: exact synthetic IDs; native abort/restart mechanism, NOT detector quality.

This file is absent from the production overlay. Do not use in normal service.
"""
import hashlib,json,logging,time
from .collapse import Detection
from .recovery import RecoveryConfig,PhaseGuard,eligibility,generate_with_recovery
from .integration import generate_for_ds41f as production_generate
RUN='density-20261012t233616z'
IDS={f'chatcmpl-ds41f-loopbuster-{RUN}-{n:02d}':(2 if n==1 else 1) for n in range(4)}
SEEN=set()
LOG=logging.getLogger(__name__)

def digest(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()

def generate_for_ds41f(server,prompt,params,request_id,**kwargs):
    if request_id not in IDS:return production_generate(server,prompt,params,request_id,**kwargs)
    def emit(event):
        LOG.warning('ds41f_loopbuster_canary %s',json.dumps({**event,'request_id':request_id,
            'run':RUN,'mode':'injected-auto-recover','monotonic':time.monotonic()},sort_keys=True))
    reason=eligibility(prompt,params,kwargs)
    if request_id in SEEN or params.max_tokens!=1024 or reason:
        emit({'event':'canary_skipped','reason':reason or 'duplicate_or_budget_mismatch'})
        return server.engine_client.generate(prompt,params,request_id,**kwargs)
    SEEN.add(request_id)
    emit({'event':'canary_selected','detector_decision':'injected'})
    retry_count=IDS[request_id];detector_number=0
    class Injected:
        def __init__(self,index):self.index=index;self.calls=0
        def feed(self,text,start,end):
            self.calls+=1
            if self.index<retry_count and self.calls>=2 and start>=8:
                emit({'event':'canary_injected','attempt':self.index,'checkpoint':0,
                      'published_native_prefix_tokens':start,'current_end':end})
                return Detection(0,{'kind':'INJECTED_MECHANISM_ONLY','native_end_tokens':end})
            return None
    def factory():
        nonlocal detector_number
        obj=Injected(detector_number);detector_number+=1;return obj
    class Proxy:
        def __init__(self):self.attempt=0;self.observed=0;self.aborted=set();self.closed=set();self.last_id=None;self.seeds=set()
        async def generate(self,new_prompt,new_params,rid,**new_kwargs):
            attempt=self.attempt;self.attempt+=1
            exact_original=(new_prompt['prompt_token_ids']==prompt['prompt_token_ids'])
            mm_ok=all(new_prompt.get(k) is prompt.get(k) for k in ('mm_kwargs','mm_placeholders','mm_hashes'))
            expected_budget=params.max_tokens-self.observed
            previous_complete=self.last_id is None or (self.last_id in self.aborted and self.last_id in self.closed)
            fresh=attempt==0 or new_params.seed not in self.seeds
            sampler_preserved=all(getattr(new_params,k,None)==getattr(params,k,None) for k in
                                  ('temperature','top_p','top_k','min_p','stop','stop_token_ids',
                                   'presence_penalty','frequency_penalty','repetition_penalty'))
            emit({'event':'native_attempt_start','attempt':attempt,'native_request_id':rid,
                  'prompt_tokens':len(new_prompt['prompt_token_ids']),'prompt_sha256':digest(new_prompt['prompt_token_ids']),
                  'exact_original_prompt':exact_original,'multimodal_identity_preserved':mm_ok,
                  'kwargs_preserved':new_kwargs==kwargs,'max_tokens':new_params.max_tokens,
                  'budget_correct':new_params.max_tokens==expected_budget,'sampler_preserved':sampler_preserved,
                  'fresh_retry_seed':fresh,'previous_abort_and_close_complete':previous_complete,
                  'temporary_words_active':bool(new_params.bad_words),
                  'temporary_expiry':(new_params.extra_args or {}).get('ds41f_loopbuster_bad_words_until'),
                  'temporary_stop_ids_valid':(new_params.extra_args or {}).get('ds41f_loopbuster_stop_ids')==[128822,128825],
                  'image_placeholders':len((new_prompt.get('mm_placeholders') or {}).get('image',[]))})
            if not all((exact_original,mm_ok,new_kwargs==kwargs,new_params.max_tokens==expected_budget,fresh,sampler_preserved,previous_complete)):
                raise RuntimeError('Loopbuster synthetic native invariant failed')
            self.seeds.add(new_params.seed);self.last_id=rid
            iterator=server.engine_client.generate(new_prompt,new_params,rid,**new_kwargs)
            try:
                async for row in iterator:
                    tokens=list(row.outputs[0].token_ids);self.observed+=len(tokens)
                    emit({'event':'native_delta','attempt':attempt,'native_request_id':rid,
                          'tokens':len(tokens),'token_sha256':digest(tokens),'finished':row.finished})
                    yield row
            finally:
                await iterator.aclose();self.closed.add(rid)
                emit({'event':'native_iterator_closed','attempt':attempt,'native_request_id':rid})
        async def abort(self,rid):
            emit({'event':'native_abort_started','native_request_id':rid})
            await server.engine_client.abort(rid);self.aborted.add(rid)
            emit({'event':'native_abort_completed','native_request_id':rid})
    boundaries=[tuple(server.renderer.tokenizer.encode(m,add_special_tokens=False)) for m in PhaseGuard.MARKERS]
    return generate_with_recovery(Proxy(),prompt,params,request_id,config=RecoveryConfig(mode='auto-recover'),
           detector_factory=factory,boundary_token_sequences=boundaries,on_event=emit,
           **kwargs)
