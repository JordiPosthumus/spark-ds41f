"""Bounded in-memory ordinary-word candidates; never log text or token IDs."""
from collections import Counter,deque
import copy
import re

EXPIRY_KEY='ds41f_loopbuster_bad_words_until'
EXPIRY_TOKENS=48
class RetryWords:
    def __init__(self):self.words=deque(maxlen=600);self.partial='';self.overlong=False
    def feed(self,text):
        for ch in text:
            if ch.isascii() and ch.isalpha():
                if len(self.partial)<24:self.partial+=ch.lower()
                else:self.overlong=True
            else:self.flush()
    def flush(self):
        if len(self.partial)>=2 and not self.overlong:self.words.append(self.partial)
        self.partial='';self.overlong=False
    def candidates(self):
        words=list(self.words);grams=Counter(tuple(words[i:i+3]) for i in range(len(words)-2))
        repeated={w for gram,n in grams.items() if n>=2 for w in gram}
        counts=Counter(words)
        return [w for w,n in counts.most_common() if w in repeated and n>=4][:16]

def apply_retry_words(params,words,tokenizer):
    if getattr(params,'allowed_token_ids',None) is not None:
        return {'applied':False,'reason':'caller_allowed_token_ids','word_count':0}
    variants=[];token_count=0;special=set(getattr(tokenizer,'all_special_ids',[]) or [])
    for word in words:
        for form in (word,word.capitalize(),word.upper()):
            if form in variants:continue
            sequences=[list(tokenizer.encode(prefix+form,add_special_tokens=False)) for prefix in ('',' ')]
            if not all(sequences) or any(special.intersection(seq) for seq in sequences):continue
            # Native tokenizer adds at most two forms. Stay below native limits
            # without changing the established global limit configuration.
            count=sum(map(len,sequences))
            if len(variants)>=32 or token_count+count>256:break
            variants.append(form);token_count+=count
    if not variants:return {'applied':False,'reason':'no_safe_repeated_words','word_count':0}
    end_ids=list(tokenizer.encode('</think>',add_special_tokens=False))
    dsml=list(tokenizer.encode('<｜DSML｜ calls>',add_special_tokens=False))
    if end_ids!=[128822] or dsml!=[30,128825,10699,32]:
        return {'applied':False,'reason':'unverified_control_token_ids','word_count':0}
    candidate=copy.deepcopy(params)
    candidate.bad_words=variants
    candidate._bad_words_token_ids=None
    candidate.extra_args={EXPIRY_KEY:EXPIRY_TOKENS,'ds41f_loopbuster_stop_ids':[128822,128825]}
    # Native request preprocessing calls this too; precompute here so the
    # invariant and bounds are verified before another engine request starts.
    try:
        candidate.update_from_tokenizer(tokenizer)
    except Exception as exc:
        return {'applied':False,'reason':'native_bad_words_validation','exception_type':type(exc).__name__,'word_count':0}
    params.bad_words=candidate.bad_words
    params._bad_words_token_ids=candidate._bad_words_token_ids
    params.extra_args=candidate.extra_args
    return {'applied':True,'word_count':len(words),'word_variants':len(variants),
            'token_sequences':len(params.bad_words_token_ids or []),'expiry_generated_tokens':EXPIRY_TOKENS}
