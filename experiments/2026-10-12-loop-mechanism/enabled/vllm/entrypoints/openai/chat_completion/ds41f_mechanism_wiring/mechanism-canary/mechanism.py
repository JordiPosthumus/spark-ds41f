"""Explicit test-only detector injection through the frozen callback seam.

No installation, environment activation, endpoint, startup, or model import.
The caller must deliberately use make_callback with exact synthetic identities.
"""
import asyncio
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parent
MODE = 'mechanism-recover-injected'
PACKAGE = '_frozen_ds41f_mechanism_dependency'
lock=json.loads((ROOT/'source-lock.json').read_text())
for dependency in ['collapse.py','suspicion.py','recovery.py']:
    relative='adaptive-prototype/'+dependency
    if hashlib.sha256((ROOT.parent/relative).read_bytes()).hexdigest()!=lock['dependencies'][relative]:
        raise RuntimeError('frozen callback dependency hash mismatch: '+relative)
package = ModuleType(PACKAGE)
package.__path__=[str(ROOT.parent/'adaptive-prototype')]
sys.modules.setdefault(PACKAGE, package)
from importlib import import_module
native_wrapper = import_module(PACKAGE+'.recovery')
Detection = import_module(PACKAGE+'.collapse').Detection


@dataclass(frozen=True)
class MechanismSpec:
    expected_id: str
    injected_retries: int = 1
    holdback_tokens: int = 128
    discard_tokens: int = 16
    maximum_observed_tokens: int = 1024
    minimum_published_tokens: int = 1

    def __post_init__(self):
        if not re.fullmatch(r'chatcmpl-ds41f-mechanism-[0-9a-f]{12}-0[0-2]', self.expected_id):
            raise ValueError('only exact bounded synthetic mechanism IDs are accepted')
        for name, lower, upper in [('injected_retries',1,2),('holdback_tokens',32,256),
                ('discard_tokens',1,64),('maximum_observed_tokens',32,2048),('minimum_published_tokens',0,128)]:
            value=getattr(self,name)
            if type(value) is not int or not lower<=value<=upper:
                raise ValueError(name)
        if self.discard_tokens>self.holdback_tokens//2:
            raise ValueError('discard target must fit conservatively inside holdback')


def fingerprint(ids):
    return hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()


class Evidence:
    def __init__(self, spec, emit):
        self.spec, self.emit = spec, emit
        self.published=[]
        self.attempts=0
        self.injected=0
        self.observed=0
        self.completed_aborts=0
        self.attempts_with_output=set()
        self.retry_output=asyncio.Event()
        self.events=[]

    def event(self, kind, **fields):
        row={**fields,'event':kind,'test_only':True,'detector_decision':'injected',
             'request_id':self.spec.expected_id,'run':self.spec.expected_id.split('-')[-2],
             'mode':MODE}
        # No raw text, token IDs, seeds or image data.
        if len(self.events)<4096:
            self.events.append(row)
            try:
                self.emit(row)
            except Exception:
                pass

    def factory(self):
        owner=self
        class InjectedDetector:
            def __init__(self):
                self.checkpoint=None
                self.used=False
            def feed(self,text,start,end):
                if self.used or owner.injected>=owner.spec.injected_retries:
                    return None
                if owner.observed>owner.spec.maximum_observed_tokens:
                    return None
                published=len(owner.published)
                if self.checkpoint is not None and self.checkpoint<published:
                    owner.event('injection_checkpoint_expired', checkpoint=self.checkpoint, published=published)
                    self.checkpoint=None
                # Select ONLY an observed native chunk start. A spec may require
                # an already-published prefix, or explicitly test the first
                # unpublished chunk. Never invent a token/text cut inside it.
                if self.checkpoint is None and published>=owner.spec.minimum_published_tokens:
                    if 0<end-start<=owner.spec.holdback_tokens and start>=published:
                        self.checkpoint=start
                if self.checkpoint is None:
                    return None
                retained=end-self.checkpoint
                if retained>owner.spec.holdback_tokens:
                    self.checkpoint=None
                    return None
                if retained>=owner.spec.discard_tokens:
                    self.used=True
                    owner.injected+=1
                    owner.event('injected_confirmation', ordinal=owner.injected,
                        checkpoint=self.checkpoint, current_token_end=end,
                        published_tokens=published, unpublished_suffix_tokens=retained)
                    return Detection(self.checkpoint, {'kind':'synthetic_mechanism_injection',
                        'injected':True,'ordinal':owner.injected})
                return None
        return InjectedDetector()


class EngineView:
    def __init__(self, engine, original_generate, evidence, original_prompt, original_params):
        self.engine, self.original_generate, self.evidence = engine, original_generate, evidence
        self.prompt, self.params = original_prompt, original_params
        self.base=[];self.ids=[];self.seeds=[]
        fields=getattr(original_params,'__struct_fields__',None)
        if fields is None:
            fields=tuple(vars(original_params))
        self.settings={k:copy.deepcopy(getattr(original_params,k)) for k in fields
                       if k not in {'seed','max_tokens'}}

    def __getattr__(self,name):
        return getattr(self.engine,name)

    async def generate(self,prompt,params,request_id,**kwargs):
        state=self.evidence
        state.attempts+=1
        original=self.prompt['prompt_token_ids']
        retained=prompt['prompt_token_ids'][len(original):]
        previous=self.base+self.ids
        seed=getattr(params,'seed',None)
        settings_equal=all(getattr(params,k)==v for k,v in self.settings.items())
        mm=['mm_kwargs','mm_hashes','mm_placeholders']
        state.event('native_attempt_start',attempt=state.attempts,
            original_prompt_prefix_equal=prompt['prompt_token_ids'][:len(original)]==original,
            retained_prefix_equal=state.attempts==1 or retained==previous[:len(retained)],
            published_prefix_equal=retained[:len(state.published)]==state.published,
            retained_tokens=len(retained),retained_token_sha256=fingerprint(retained),
            unchanged_sampler_fields_equal=settings_equal,
            initial_seed_equal=(seed==getattr(self.params,'seed',None)) if state.attempts==1 else None,
            retry_seed_fresh=(seed not in self.seeds) if state.attempts>1 else None,
            remaining_budget_equal=params.max_tokens==self.params.max_tokens-len(retained),
            max_tokens=params.max_tokens,
            mm_object_identity_equal=all(prompt.get(k) is self.prompt.get(k) for k in mm),
            image_placeholders=len((prompt.get('mm_placeholders') or {}).get('image',[])))
        self.seeds.append(seed)
        self.base,self.ids=list(retained),[]
        iterator=self.original_generate(prompt,params,request_id,**kwargs)
        try:
            async for result in iterator:
                for output in result.outputs:
                    start=len(self.base)+len(self.ids)
                    self.ids.extend(output.token_ids)
                    state.observed+=len(output.token_ids)
                    if output.token_ids:
                        state.attempts_with_output.add(state.attempts)
                    state.event('native_chunk',attempt=state.attempts,token_start=start,
                        token_end=len(self.base)+len(self.ids),token_sha256=fingerprint(output.token_ids),
                        text_sha256=hashlib.sha256(output.text.encode()).hexdigest(),finished=result.finished)
                if state.attempts>1 and any(o.token_ids for o in result.outputs):
                    state.retry_output.set()
                yield result
        finally:
            await iterator.aclose()

    async def abort(self,request_id,**kwargs):
        self.evidence.event('native_abort_enter')
        await self.engine.abort(request_id,**kwargs)
        self.evidence.completed_aborts+=1
        self.evidence.event('native_abort_completed', api_unfinished_requests=self.engine.get_num_unfinished_requests())


def make_callback(engine,specs=(),emit=lambda event:None,*,boundary_token_sequences=None):
    """Return an uninstalled callback and evidence map.

    Empty/default specs are exact native passthrough. No monkey-patching occurs.
    A future isolated caller substitutes this callback at its generate seam.
    """
    specs=tuple(specs)
    if len(specs)>3 or len({s.expected_id for s in specs})!=len(specs):
        raise ValueError('at most three distinct synthetic mechanism cases')
    selected={s.expected_id:s for s in specs}
    original_generate=engine.generate
    evidence={}
    def callback(prompt,params,request_id,**kwargs):
        spec=selected.get(request_id)
        if spec is None:
            return original_generate(prompt,params,request_id,**kwargs)
        reason=('unverified_boundary_tokens' if not boundary_token_sequences or any(not row for row in boundary_token_sequences)
                else native_wrapper.eligibility(prompt,params,kwargs))
        if reason is None and params.max_tokens>2048:
            reason='mechanism_fixture_output_budget_exceeds_2048'
        if request_id in evidence:
            reason='duplicate_mechanism_request_identity'
        if reason:
            emit({'event':'mechanism_skipped','reason':reason,'test_only':True,
                  'request_id':request_id,'run':request_id.split('-')[-2],'mode':MODE})
            return original_generate(prompt,params,request_id,**kwargs)
        state=Evidence(spec,emit);evidence[request_id]=state
        return run(state,prompt,params,request_id,kwargs)
    async def run(state,prompt,params,request_id,kwargs):
        state.event('mechanism_start',requested_injections=state.spec.injected_retries)
        proxy=EngineView(engine,original_generate,state,prompt,params)
        iterator=native_wrapper.generate_with_recovery(proxy,prompt,params,request_id,
            config=native_wrapper.RecoveryConfig(mode='recover',max_retries=state.spec.injected_retries,
                    holdback_tokens=state.spec.holdback_tokens,adaptive=False),
            detector_factory=state.factory,on_event=lambda e:state.event('wrapper_'+e['event'],**{k:v for k,v in e.items() if k!='event'}),
            boundary_token_sequences=boundary_token_sequences,
            **kwargs)
        try:
            async for result in iterator:
                for output in result.outputs:
                    state.published.extend(output.token_ids)
                yield result
        finally:
            await iterator.aclose()
            state.event('mechanism_summary',native_attempts=state.attempts,injected_decisions=state.injected,
                observed_tokens=state.observed,published_tokens=len(state.published),
                published_token_sha256=fingerprint(state.published),
                completed_native_aborts=state.completed_aborts,
                native_retry_attempts_with_output=len(state.attempts_with_output-{1}),
                requested_injections_exercised=(state.injected==state.spec.injected_retries
                    and state.completed_aborts>=state.spec.injected_retries
                    and len(state.attempts_with_output-{1})==state.spec.injected_retries),
                detector_quality_proven=False,model_quality_improvement_proven=False,
                scheduler_idle_proven=False)
    return callback,evidence


def make_server_callback(server,specs=(),emit=lambda event:None):
    """Future native-server helper, with the frozen integration exclusions."""
    import importlib.metadata
    engine=server.engine_client
    reason=None
    config=getattr(engine,'vllm_config',None)
    if getattr(server.parser_cls,'__name__',None)!='DeepSeekV41Parser':
        reason='unverified_model_parser'
    elif config is None:
        reason='unknown_engine_config'
    elif getattr(config,'watermark_config',None) is not None:
        reason='engine_watermark_not_qualified_for_mechanism_test'
    elif getattr(config.model_config,'enable_return_routed_experts',False):
        reason='unmapped_routed_expert_metadata'
    elif getattr(config.model_config,'logits_processors',None) or importlib.metadata.entry_points(group='vllm.logits_processors'):
        reason='unverified_logits_processor'
    if reason:
        emit({'event':'mechanism_skipped','reason':reason,'test_only':True})
        return engine.generate,{}
    try:
        boundaries=[tuple(server.renderer.tokenizer.encode(marker,add_special_tokens=False))
                    for marker in native_wrapper.PhaseGuard.MARKERS]
    except Exception:
        boundaries=None
    return make_callback(engine,specs,emit,boundary_token_sequences=boundaries)
