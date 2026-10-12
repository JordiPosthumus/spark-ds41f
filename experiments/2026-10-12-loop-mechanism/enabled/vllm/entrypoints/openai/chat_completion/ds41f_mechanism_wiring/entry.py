"""Temporary synthetic-only native seam. Missing/disabled/bad plan = native.

No environment knobs, endpoint, request parsing, model imports or monkey patch.
The real serving call constructs one gate from its embedded plan digest.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import weakref

from .transform import restore

ROOT = Path(__file__).resolve().parent
VLLM_ROOT = Path('/opt/spark3/candidate/vllm/vllm')
SERVING = 'entrypoints/openai/chat_completion/serving.py'
MODE = 'mechanism-recover-injected'
AUDIT_PREFIX = 'DS41F_MECHANISM_AUDIT '


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(row):
    # Only bounded synthetic identities and aggregate/hash fields reach here.
    try:
        sys.stderr.write(AUDIT_PREFIX + json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        sys.stderr.flush()
    except (OSError, ValueError):
        # A lost audit makes the canary inconclusive, not an engine failure.
        pass


def validate_plan(plan):
    if plan.get('schema') != 1 or type(plan.get('enabled')) is not bool or plan.get('mode') != MODE:
        raise ValueError('unsupported plan schema/mode')
    run = plan.get('run')
    if not isinstance(run, str) or not re.fullmatch('[0-9a-f]{12}', run):
        raise ValueError('invalid exact run identity')
    cases = plan.get('cases')
    expected = [f'chatcmpl-ds41f-mechanism-{run}-{i:02}' for i in range(3)]
    if not isinstance(cases, list) or [c.get('request_id') for c in cases] != expected:
        raise ValueError('plan must have exactly three ordered full identities')
    for case in cases:
        if case.get('spec', {}).get('expected_id') != case['request_id'] or case.get('expected_max_tokens') != 1024:
            raise ValueError('invalid request-only specification')
    for key in ('native_source_sha256', 'payload_sha256'):
        if not isinstance(plan.get(key), dict) or not plan[key]:
            raise ValueError('missing source locks')
        for relative, sha in plan[key].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('invalid source lock')
    if SERVING not in plan['native_source_sha256'] or len(plan.get('baseline_overlays', {})) != 7:
        raise ValueError('missing serving or seven-overlay baseline lock')
    if any(plan['native_source_sha256'].get(k) != v for k, v in plan['baseline_overlays'].items()):
        raise ValueError('baseline overlay/source disagreement')
    return plan


def load_plan(root, expected_sha256):
    path = root / 'plan.json'
    if digest(path) != expected_sha256:
        raise ValueError('immutable plan digest mismatch')
    return validate_plan(json.loads(path.read_text()))


def verify_sources(root, installed_root, plan, plan_sha256):
    """Read bytes, not imports. Runs freshly before each allowlisted request."""
    if digest(root / 'plan.json') != plan_sha256:
        raise ValueError('immutable plan changed after import')
    for relative, expected in plan['payload_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError('helper source mismatch: ' + relative)
    for relative, expected in plan['native_source_sha256'].items():
        path = installed_root / relative
        if relative == SERVING:
            actual = hashlib.sha256(restore(path.read_text(), plan_sha256).encode()).hexdigest()
        else:
            actual = digest(path)
        if actual != expected:
            raise ValueError('native source mismatch: ' + relative)


def load_mechanism(root):
    path = root / 'mechanism-canary' / 'mechanism.py'
    name = '_ds41f_temporary_mechanism_' + digest(path)[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def make_plan_gate(expected_sha256):
    return Gate(ROOT, VLLM_ROOT, expected_sha256)


class Gate:
    """Synchronous selection preserves native iterator identity outside canary."""
    def __init__(self, root, installed_root, plan_sha256):
        self.root, self.installed_root, self.plan_sha256 = root, installed_root, plan_sha256
        self.callbacks = weakref.WeakKeyDictionary()
        self.plan = None
        self.plan_error = None
        try:
            self.plan = load_plan(root, plan_sha256)
        except Exception as error:
            self.plan_error = type(error).__name__
            emit({'event': 'wiring_disabled', 'mode': 'mechanism-wiring',
                  'reason': 'invalid_or_missing_plan', 'error_type': self.plan_error,
                  'plan_sha256': plan_sha256})

    def audit(self, request_id, event, **fields):
        emit({**fields, 'event': event, 'request_id': request_id,
              'run': self.plan['run'], 'mode': 'mechanism-wiring',
              'plan_sha256': self.plan_sha256, 'test_only': True,
              'monotonic_seconds': time.monotonic(), 'process_id': os.getpid()})

    def __call__(self, server, prompt, params, request_id, **kwargs):
        native = server.engine_client.generate
        if self.plan is None:
            return native(prompt, params, request_id, **kwargs)
        selected = {c['request_id']: c for c in self.plan['cases']}
        case = selected.get(request_id)
        if case is None:
            return native(prompt, params, request_id, **kwargs)
        if not self.plan['enabled']:
            self.audit(request_id, 'wiring_native', reason='plan_disabled')
            return native(prompt, params, request_id, **kwargs)
        if getattr(params, 'max_tokens', None) != case['expected_max_tokens']:
            self.audit(request_id, 'wiring_skipped', reason='synthetic_budget_mismatch_no_override')
            return native(prompt, params, request_id, **kwargs)
        try:
            verify_sources(self.root, self.installed_root, self.plan, self.plan_sha256)
            if server not in self.callbacks:
                module = load_mechanism(self.root)
                specs = [module.MechanismSpec(**c['spec']) for c in self.plan['cases']]
                def bound_emit(row):
                    # Native server eligibility failures occur while building
                    # the callback and have no identity; bind to this selection.
                    rid = row.get('request_id', request_id)
                    if rid not in selected:
                        raise ValueError('foreign mechanism audit identity')
                    emit({**row, 'request_id': rid, 'run': self.plan['run'],
                          'mode': MODE, 'plan_sha256': self.plan_sha256,
                          'monotonic_seconds': time.monotonic(), 'process_id': os.getpid()})
                self.callbacks[server] = module.make_server_callback(server, specs, bound_emit)
            callback, _ = self.callbacks[server]
        except Exception as error:
            # A changed build or incompatible server continues its established
            # request; it never gets a reduced budget or a fallback sampler.
            self.audit(request_id, 'wiring_skipped', reason='source_or_callback_gate_failed',
                       error_type=type(error).__name__)
            return native(prompt, params, request_id, **kwargs)
        self.audit(request_id, 'wiring_selected', source_hash_gate_passed=True)
        # Do not catch an exception here and silently submit the request again.
        return callback(prompt, params, request_id, **kwargs)

    def evidence_for(self, server, request_id):
        """In-process canary harness only; no HTTP control/inspection endpoint."""
        if self.plan is None or request_id not in {c['request_id'] for c in self.plan['cases']}:
            return None
        entry = self.callbacks.get(server)
        return entry[1].get(request_id) if entry else None
