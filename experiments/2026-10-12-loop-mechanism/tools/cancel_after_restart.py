"""Future actual-engine cancellation check; no transport or scheduler stand-ins."""
import asyncio
import math
import time

PENDING = set()


def valid_idle_samples(samples,*,not_before,not_after):
    """Require finite zero-work telemetry acquired in this fresh probe window.

    Timestamps must use this process's time.monotonic() clock. These numerical
    checks cannot authenticate the source; the probe must actually fetch fresh
    scheduler metrics rather than re-label cached values with new timestamps.
    """
    def finite_nonnegative(value):
        try:
            return type(value) in (int,float) and value>=0 and math.isfinite(value)
        except OverflowError:
            return False
    if not (finite_nonnegative(not_before) and finite_nonnegative(not_after)
            and not_before<=not_after and isinstance(samples,list) and len(samples)>=6):
        return False
    keys=('running','waiting','generation_tokens','monotonic_seconds')
    if not all(isinstance(s,dict) and all(finite_nonnegative(s.get(k)) for k in keys)
               and s['running']==0 and s['waiting']==0 for s in samples):
        return False
    return (len({s['generation_tokens'] for s in samples})==1
            and not_before<=samples[0]['monotonic_seconds']
            and samples[-1]['monotonic_seconds']<=not_after
            and samples[-1]['monotonic_seconds']-samples[0]['monotonic_seconds']>=10
            and all(a['monotonic_seconds']<b['monotonic_seconds'] for a,b in zip(samples,samples[1:])))


async def exercise(iterator,evidence,engine,*,idle_probe=None,timeout=30):
    """Cancel twice after the restarted native request produces a token.

    idle_probe, when provided by the operator, must fetch fresh scheduler-gauge
    and generation-counter samples AFTER this cleanup attempt. Timestamp each
    fresh observation with this process's time.monotonic(), not a remote clock
    or cached historical timestamps. No endpoint is chosen by this helper.
    """
    if engine.get_num_unfinished_requests()!=0:
        raise RuntimeError('mechanism canary requires an idle engine')
    async def consume():
        try:
            async for result in iterator:
                pass
        finally:
            await iterator.aclose()
    task=asyncio.create_task(consume());PENDING.add(task)
    task.add_done_callback(PENDING.discard)
    ready=asyncio.create_task(evidence.retry_output.wait())
    identity={'request_id':evidence.spec.expected_id,'run':evidence.spec.expected_id.split('-')[-2],
              'mode':'mechanism-recover-injected','test_only':True,
              'probe':'repeated_cancel_after_native_retry_output'}
    try:
        done,_=await asyncio.wait({task,ready},timeout=timeout,return_when=asyncio.FIRST_COMPLETED)
        if task in done or ready not in done:
            if task.done() and not task.cancelled():
                task.result()
            return {**identity,'inconclusive':True,'reason':'no_restarted_native_output_before_finish_or_timeout',
                    'scheduler_idle_proven':False}
        task.cancel();await asyncio.sleep(0);task.cancel()
        done,_=await asyncio.wait({task},timeout=timeout)
        if task in done and not task.cancelled():
            task.result()
        probe_started=time.monotonic()
        samples=await idle_probe() if idle_probe is not None else None
        probe_finished=time.monotonic()
        valid=valid_idle_samples(samples,not_before=probe_started,not_after=probe_finished)
        return {**identity,'inconclusive':False,'native_restart_output_seen':True,
                'caller_cancelled':task in done and task.cancelled(),
                'cleanup_task_pending':task not in done,
                'api_unfinished_requests':engine.get_num_unfinished_requests(),
                'idle_probe_started_monotonic':probe_started,
                'idle_probe_finished_monotonic':probe_finished,
                'scheduler_idle_samples':samples,'scheduler_idle_proven':valid}
    finally:
        ready.cancel();await asyncio.gather(ready,return_exceptions=True)
        if not task.done():
            # The owned wrapper cleanup remains strongly held; timeout is not
            # success, and this helper never kills a worker/server to hide it.
            task.cancel()
