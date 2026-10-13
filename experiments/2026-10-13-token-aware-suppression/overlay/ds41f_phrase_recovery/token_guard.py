"""Track trusted generated-prefix state and audit native temporary-mask spans.

Retains bounded numeric state only. Token tables are shared with the native
sampler; independent text-oracle tests validate their semantics separately.
"""


class TokenGuard:
    def __init__(self, tables=None):
        from vllm.v1.worker.gpu.sample.phrase_suppression_tables import (
            BOUNDARY_IDS, NO_HIT, feed_token, get_tables, initial,
        )
        self.tables = get_tables() if tables is None else tables
        self.feed_token = feed_token
        self.boundaries = BOUNDARY_IDS
        self.no_hit = NO_HIT
        self.state = initial()
        self.offset = 0
        self.active = False
        self.words = 0
        self.violations = 0
        self.total_violations = 0
        self.finished = None
        self.start = None
        self.end = None

    def begin(self):
        if self.active:
            raise RuntimeError("temporary guard already active")
        self.active = True
        self.words = 0
        self.violations = 0
        self.finished = None
        self.start = self.offset
        self.end = None
        return (1, *self.state)

    def feed(self, token_ids):
        for token in token_ids:
            token = int(token)
            if self.active and token in self.boundaries:
                self.active = False
                self.finished = "reasoning_boundary"
                self.end = self.offset
            lex, utf, words, hit = self.feed_token(self.tables, *self.state, token)
            self.state = (lex, utf)
            self.offset += 1
            if self.active:
                if hit != self.no_hit and hit <= 60 - self.words:
                    self.violations += 1
                    self.total_violations += 1
                self.words += words
                if self.words >= 60:
                    self.active = False
                    self.finished = "word_limit"
                    self.end = self.offset

    def finish(self, reason):
        if self.active:
            self.active = False
            self.finished = reason
            self.end = self.offset

    def rewind(self, retained_ids):
        # Rebuild only on the existing rare severe-loop rewind. Do not carry
        # lexical state belonging to discarded reasoning into the new prefix.
        self.state = (0, 0)
        self.offset = 0
        self.active = False
        self.finished = None
        self.start = self.end = None
        self.feed(retained_ids)

    def evidence(self):
        return {"protected_start_token": self.start,
                "protected_end_token": self.end,
                "completed_words_including_last_token": self.words,
                "protected_words": min(self.words, 60),
                "mask_violation_tokens": self.violations,
                "release_reason": self.finished}
