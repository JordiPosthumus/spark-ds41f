"""Exactly reversible transform of the captured native generation call."""
import ast
import re

IMPORT = 'from vllm.sampling_params import BeamSearchParams, SamplingParams\n'
CALL = '                generator = self.engine_client.generate(\n'
NEW_CALL = '                generator = _ds41f_mechanism_generate(\n                    self,\n'


def addition(plan_sha256):
    if not re.fullmatch('[0-9a-f]{64}', plan_sha256):
        raise ValueError('invalid immutable plan digest')
    return ('try:\n'
            '    from vllm.entrypoints.openai.chat_completion.ds41f_mechanism_wiring.entry import make_plan_gate as _ds41f_make_plan_gate\n'
            f'    _ds41f_mechanism_generate = _ds41f_make_plan_gate("{plan_sha256}")\n'
            'except Exception:\n'
            '    init_logger(__name__).warning("DS41F mechanism helper unavailable; all requests remain native")\n'
            '    def _ds41f_mechanism_generate(server, prompt, params, request_id, **kwargs):\n'
            '        return server.engine_client.generate(prompt, params, request_id, **kwargs)\n')


def transform(source, plan_sha256):
    if source.count(IMPORT) != 1 or source.count(CALL) != 1:
        raise ValueError('captured native seam changed')
    output = source.replace(IMPORT, IMPORT + addition(plan_sha256)).replace(CALL, NEW_CALL)
    ast.parse(output)
    if restore(output, plan_sha256) != source:
        raise ValueError('non-reversible native seam transform')
    return output


def restore(source, plan_sha256):
    extra = addition(plan_sha256)
    if source.count(extra) != 1 or source.count(NEW_CALL) != 1:
        raise ValueError('mounted native seam changed')
    return source.replace(extra, '').replace(NEW_CALL, CALL)
