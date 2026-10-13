#!/usr/bin/env python3
"""CPU-only candidate SamplingParams clone/msgpack check; no model or CUDA use."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import msgspec

ROOT=Path(__file__).resolve().parent

def validate():
    name='candidate_phrase_sampling_params'
    spec=importlib.util.spec_from_file_location(name,ROOT/'vllm/sampling_params.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    cls=module.SamplingParams
    rows=[]
    for skip_clone in (False,True):
        original=cls(temperature=1.0,seed=72,max_tokens=64,skip_clone=skip_clone,
                     _ds41f_phrase_suppression=(1,12,7))
        objects=[copy.deepcopy(original),original.clone(),msgspec.msgpack.decode(msgspec.msgpack.encode(original),type=cls)]
        for obj in objects:
            assert obj._ds41f_phrase_suppression==(1,12,7)
            assert obj.temperature==1 and obj.seed==72 and obj.max_tokens==64
            assert obj is not original
        default=cls(max_tokens=64)
        assert default._ds41f_phrase_suppression is None
        assert msgspec.msgpack.decode(msgspec.msgpack.encode(default),type=cls)._ds41f_phrase_suppression is None
        rows.append({'skip_clone':skip_clone,'deepcopy':True,'clone':True,'msgpack':True,'default_none':True})
    return {'passed':True,'cases':rows,'msgspec_version':msgspec.__version__}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
