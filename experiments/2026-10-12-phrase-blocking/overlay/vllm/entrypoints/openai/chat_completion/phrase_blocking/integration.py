"""Opt-in native exact-phrase blocklist for DeepSeek V4.1 chat completions."""
import copy
import logging
import os
from .phrases import PHRASES

LOG=logging.getLogger(__name__)

class PhraseSkip(ValueError):pass

def merge_phrases(params,tokenizer):
    for name in ('trace_decode_token_ids','structured_outputs','allowed_token_ids'):
        if getattr(params,name,None) is not None:raise PhraseSkip('caller_'+name)
    result=copy.deepcopy(params)
    result.bad_words=list(dict.fromkeys([*(params.bad_words or []),*PHRASES]))
    result._bad_words_token_ids=None
    result.update_from_tokenizer(tokenizer)
    from vllm import envs
    if sum(map(len,result.bad_words_token_ids or [])) > envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS:
        raise PhraseSkip('native_total_bad_word_token_limit')
    return result

def generate_for_ds41f(server,prompt,params,request_id,**kwargs):
    if os.environ.get('SPARK_DS41F_PHRASE_BLOCKLIST','0')!='1':
        return server.engine_client.generate(prompt,params,request_id,**kwargs)
    model=getattr(server,'model_config',None)
    if getattr(getattr(model,'hf_config',None),'model_type',None)!='deepseek_v41':
        LOG.info('ds41f_phrase_blocklist skipped: model_scope')
        return server.engine_client.generate(prompt,params,request_id,**kwargs)
    try:
        filtered=merge_phrases(params,server.renderer.tokenizer)
    except PhraseSkip as exc:
        LOG.info('ds41f_phrase_blocklist skipped: %s',str(exc))
        return server.engine_client.generate(prompt,params,request_id,**kwargs)
    except Exception as exc:
        LOG.warning('ds41f_phrase_blocklist skipped: native_validation_%s',type(exc).__name__)
        return server.engine_client.generate(prompt,params,request_id,**kwargs)
    LOG.info('ds41f_phrase_blocklist applied: phrases=%d token_sequences=%d',len(PHRASES),len(filtered.bad_words_token_ids or []))
    return server.engine_client.generate(prompt,filtered,request_id,**kwargs)
