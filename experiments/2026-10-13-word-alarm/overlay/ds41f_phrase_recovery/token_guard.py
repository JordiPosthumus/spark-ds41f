"""Word-window accounting and exact native token-sequence audit, not a blocker."""
from collections import deque

class TokenGuard:
    def __init__(self, tables=None):
        from vllm.v1.worker.gpu.sample.loop_protection_tables import (
            BOUNDARY_IDS, feed_token, get_tables, get_policy)
        self.tables = get_tables() if tables is None else tables
        self.phrases, self.bad_ids = get_policy()
        self.feed_token = feed_token
        self.boundaries = BOUNDARY_IDS
        self.max_phrase_tokens = max(map(len, self.bad_ids))
        self.history = deque(maxlen=self.max_phrase_tokens)
        self.state = (0, 0)
        self.offset = 0
        self.active = False
        self.words = self.violations = self.total_violations = 0
        self.finished = self.start = self.end = None
        self.pending_control = False
        self.control_prefix_resolution = None

    def begin(self):
        if self.active:
            raise RuntimeError("temporary guard already active")
        self.active = True
        self.words = self.violations = 0
        self.finished = self.end = None
        self.start = self.offset
        return (2, *self.state, min(self.offset, self.max_phrase_tokens - 1))

    def feed(self, token_ids):
        for token in token_ids:
            token = int(token)
            if self.active and token in self.boundaries:
                self.finish("reasoning_boundary")
            self.history.append(token)
            history = list(self.history)
            if self.active and any(len(seq) <= len(history) and history[-len(seq):] == seq
                                   for seq in self.bad_ids):
                self.violations += 1
                self.total_violations += 1
            lex, utf, words, _ = self.feed_token(self.tables, *self.state, token)
            self.state = (lex, utf)
            self.offset += 1
            if self.active:
                self.words += words
                if self.words >= 2000:
                    self.finish("word_limit")

    def finish(self, reason):
        if self.active:
            self.active = False
            self.finished = reason
            self.end = self.offset

    def rewind(self, retained_ids):
        self.state = (0, 0)
        self.offset = 0
        self.active = False
        self.finished = self.start = self.end = None
        self.history.clear()
        self.feed(retained_ids)

    def evidence(self):
        return {"protected_start_token": self.start,
                "protected_end_token": self.end,
                "completed_words_including_last_token": self.words,
                "protected_words": min(self.words, 2000),
                "mask_violation_tokens": self.violations,
                "violation_semantics": "native_bad_words_token_sequences",
                "release_reason": self.finished,
                "incomplete_control_prefix": False,
                "control_prefix_resolution": None}
