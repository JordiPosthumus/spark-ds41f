"""Experimental weak warning: buffering signal only, never loop confirmation.

Uses the separately researched 300-word sensitivity. These thresholds were
explored on two selected sessions, not independently calibrated.
"""
from dataclasses import dataclass

try:
    from .collapse import CollapseConfig, CollapseDetector
except ImportError:
    from collapse import CollapseConfig, CollapseDetector


@dataclass(frozen=True)
class WarningUpdate:
    matched: bool
    chars_seen: int
    words_seen: int


class WeakSuspicion(CollapseDetector):
    def __init__(self, notify):
        super().__init__(CollapseConfig(window=300, unique=.65, coverage=.20,
                         prose_tokens=64, prose_unique=.70, prose_coverage=.25))
        self.notify = notify

    def _flush(self):
        previous = self.next_evaluation
        super()._flush()
        if self.next_evaluation != previous and self.normalized_chars >= self.config.min_chars:
            self.notify(WarningUpdate(self.hits > 0, self.chars, self.words))
        # Never latch or return a collapse decision from the weak path.
        self.detected = None


class HoldbackPolicy:
    """Token-count bounds; publication is handled only by the native chunk queue."""
    def __init__(self, normal_tokens, maximum_tokens, release_misses, on_event):
        if not (1 <= normal_tokens <= maximum_tokens <= 8192):
            raise ValueError("invalid adaptive token bounds")
        if type(release_misses) is not int or not 2 <= release_misses <= 8:
            raise ValueError("release_misses")
        self.normal_tokens = normal_tokens
        self.maximum_tokens = maximum_tokens
        self.release_misses = release_misses
        self.on_event = on_event
        self.armed = False
        self.misses = 0
        self.expansions = self.releases = 0

    @property
    def limit_tokens(self):
        return self.maximum_tokens if self.armed else self.normal_tokens

    def update(self, evidence):
        if evidence.matched:
            self.misses = 0
            if not self.armed:
                self.armed = True
                self.expansions += 1
                self.on_event("reasoning_buffer_expanded", chars_seen=evidence.chars_seen,
                              words_seen=evidence.words_seen, maximum_tokens=self.maximum_tokens)
        elif self.armed:
            self.misses += 1
            if self.misses >= self.release_misses:
                self.armed = False
                self.releases += 1
                self.on_event("reasoning_buffer_released", chars_seen=evidence.chars_seen,
                              words_seen=evidence.words_seen, normal_tokens=self.normal_tokens)

    def reset(self):
        self.armed = False
        self.misses = 0
