"""Reject ambiguous lifecycle evidence; no case-number-only correlation."""
import re

MODE='mechanism-recover-injected'


def analyze(events,result,request_id,mode=MODE):
    if not re.fullmatch(r'chatcmpl-ds41f-mechanism-[0-9a-f]{12}-0[0-2]',request_id):
        raise ValueError('exact synthetic request identity required')
    run=request_id.split('-')[-2]
    def matching(row):
        return row.get('request_id')==request_id and row.get('mode')==mode and row.get('run')==run
    if not matching(result):
        raise ValueError('result identity/mode does not match requested lifecycle')
    rows=[(i,e) for i,e in enumerate(events) if matching(e)]
    starts=[i for i,e in rows if e['event']=='mechanism_start']
    ends=[i for i,e in rows if e['event']=='mechanism_summary']
    if len(starts)!=1 or len(ends)!=1 or starts[0]>=ends[0]:
        raise ValueError('missing, duplicate or unordered lifecycle boundaries')
    start,end=starts[0],ends[0]
    if any(not matching(e) for e in events[start:end+1]):
        raise ValueError('foreign identity or mode interleaved with target lifecycle')
    if any(i<start or i>end for i,e in rows):
        raise ValueError('target events outside its lifecycle; possible duplicate/reuse')
    rows=[e for _,e in rows]
    attempts=[e for e in rows if e['event']=='native_attempt_start']
    injections=[e for e in rows if e['event']=='injected_confirmation']
    aborts=[e for e in rows if e['event']=='native_abort_completed']
    summary=rows[-1]
    invariant_keys=['original_prompt_prefix_equal','retained_prefix_equal','published_prefix_equal',
                    'unchanged_sampler_fields_equal','remaining_budget_equal','mm_object_identity_equal']
    fields_ok=bool(attempts) and all(all(e[k] for k in invariant_keys) for e in attempts)
    six_images_ok=None
    if request_id.endswith('-01'):
        six_images_ok=bool(attempts) and all(e.get('image_placeholders')==6 for e in attempts)
    seeds_ok=bool(attempts) and attempts[0]['initial_seed_equal'] is True and all(e['retry_seed_fresh'] is True for e in attempts[1:])
    boundaries_ok=bool(injections) and all(e['published_tokens']<=e['checkpoint']<e['current_token_end'] for e in injections)
    cancellation_ok=None
    if result.get('probe')=='repeated_cancel_after_native_retry_output':
        cancellation_ok=(result.get('caller_cancelled') is True and result.get('cleanup_task_pending') is False
                         and result.get('api_unfinished_requests')==0 and result.get('scheduler_idle_proven') is True)
    return {'request_id':request_id,'run':run,'mode':mode,'unambiguous_lifecycle':True,
            'attempts':len(attempts),'injections':len(injections),'completed_native_aborts':len(aborts),
            'exact_prefix_mm_sampler_budget_checks':fields_ok,'seed_checks':seeds_ok,
            'six_image_placeholders_preserved':six_images_ok,
            'injected_checkpoints_unpublished':boundaries_ok,
            'requested_injections_exercised':summary['requested_injections_exercised'],
            'native_retry_requests_started':len(attempts)-1,
            'repeated_cancel_and_idle_checks':cancellation_ok,
            'scheduler_idle_proven':result.get('scheduler_idle_proven') is True,
            'detector_quality_proven':False,'model_quality_improvement_proven':False}
