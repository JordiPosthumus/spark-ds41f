"""Bounded, hash-only *experimental* recurrence detector.

This is a Python port of the owner's offline gateway candidate, not a calibrated
classifier. It returns a conservative chunk-aligned rewind checkpoint: the
beginning of the first suspicious window, NOT a proven semantic loop onset.
No text is retained after feed(); an unfinished word is a streaming hash.
"""
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import unicodedata


@dataclass(frozen=True)
class Detection:
    start_token: int
    evidence: dict


@dataclass(frozen=True)
class CollapseConfig:
    window: int = 600
    min_chars: int = 1200
    every: int = 64
    consecutive: int = 2
    gram: int = 3
    occurrences: int = 2
    unique: float = .35
    coverage: float = .5
    prose_tokens: int = 96
    prose_unique: float = .4
    prose_coverage: float = .6

    def __post_init__(self):
        for field, low, high in (("window", 200, 4096), ("min_chars", 1200, 65536),
                                 ("every", 16, 512), ("consecutive", 2, 8),
                                 ("gram", 3, 8), ("occurrences", 2, 8),
                                 ("prose_tokens", 64, 2048)):
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(field)
        if self.prose_tokens > self.window:
            raise ValueError("prose_tokens > window")
        for field in ("unique", "coverage", "prose_unique", "prose_coverage"):
            if not 0 < getattr(self, field) < 1:
                raise ValueError(field)


def _metrics(rows, gram, occurrences, prose=False):
    positions = [i for i, row in enumerate(rows) if not prose or row[1]]
    grams = [tuple(row[0] for row in rows[i:i+gram])
             if not prose or all(row[1] for row in rows[i:i+gram]) else None
             for i in range(len(rows)-gram+1)]
    counts = Counter(grams)
    covered = set()
    for i, key in enumerate(grams):
        if key is not None and counts[key] >= occurrences:
            covered.update(range(i, i+gram))
    n = len(positions)
    return (n, len({rows[i][0] for i in positions})/n if n else 1.,
            sum(i in covered for i in positions)/n if n else 0.)


class CollapseDetector:
    def __init__(self, config=None, fence_state=None):
        self.config = config or CollapseConfig()
        self.rows = deque(maxlen=self.config.window)
        self.word = None
        self.word_chars = 0
        self.normalized_chars = self.chars = self.words = self.lines = 0
        self.hits = 0
        self.suspect_start = None
        self.next_evaluation = self.config.window
        self.detected = None
        self.in_fence = False
        self.fence_char = None
        self.fence_length = 0
        self._line_start()
        if fence_state is not None:
            self.restore_fence(fence_state)

    def _line_start(self):
        self.mode = "start"
        self.marker = None
        self.marker_count = 0
        self.fence_candidate = self.fence_rest = False

    def fence_state(self):
        return (self.in_fence, self.fence_char, self.fence_length, self.mode,
                self.marker, self.marker_count, self.fence_candidate, self.fence_rest)

    def restore_fence(self, state):
        (self.in_fence, self.fence_char, self.fence_length, self.mode,
         self.marker, self.marker_count, self.fence_candidate, self.fence_rest) = state

    def feed(self, text, start_token, end_token):
        if self.detected:
            return self.detected
        for char in text:
            # Match JS UTF-16 char counts for cross-language evidence comparison.
            self.chars += 2 if ord(char) > 0xffff else 1
            if char == "\n":
                self._flush()
                self.lines += 1
                if self.fence_candidate:
                    if not self.in_fence:
                        self.in_fence = True
                        self.fence_char, self.fence_length = self.marker, self.marker_count
                    elif (self.marker == self.fence_char and
                          self.marker_count >= self.fence_length and not self.fence_rest):
                        self.in_fence = False
                        self.fence_char, self.fence_length = None, 0
                self._line_start()
            else:
                if self.mode == "start":
                    if char.isspace():
                        continue
                    if char in "`~":
                        self.mode, self.marker, self.marker_count = "marker", char, 1
                        continue
                    self.mode = "body"
                elif self.mode == "marker":
                    if char == self.marker:
                        self.marker_count += 1
                        self.fence_candidate = self.marker_count >= 3
                        continue
                    self.fence_candidate = self.marker_count >= 3
                    self.mode = "body"
                if self.fence_candidate and not char.isspace():
                    self.fence_rest = True
                if char == "_" or unicodedata.category(char)[0] in "LN":
                    if self.word is None:
                        self.word = hashlib.sha256()
                        self.word_chars = 0
                        self.word_start = start_token
                        self.word_prose = not self.in_fence and not self.fence_candidate
                    self.word.update(char.lower().encode())
                    self.word_chars += 2 if ord(char) > 0xffff else 1
                else:
                    self._flush()
            if self.detected:
                break
        return self.detected

    def _flush(self):
        if self.word is None:
            return
        self.rows.append((self.word.digest()[:12], self.word_prose, self.word_start))
        self.normalized_chars += self.word_chars
        self.words += 1
        self.word = None
        self.word_chars = 0
        if self.words < self.next_evaluation:
            return
        self.next_evaluation = self.words + self.config.every
        c = self.config
        if len(self.rows) < c.window or self.normalized_chars < c.min_chars:
            self.hits, self.suspect_start = 0, None
            return
        rows = list(self.rows)
        _, unique, coverage = _metrics(rows, c.gram, c.occurrences)
        prose, prose_unique, prose_coverage = _metrics(rows, c.gram, c.occurrences, True)
        match = (unique <= c.unique and coverage >= c.coverage and
                 prose >= c.prose_tokens and prose_unique <= c.prose_unique and
                 prose_coverage >= c.prose_coverage)
        if not match:
            self.hits, self.suspect_start = 0, None
            return
        if not self.hits:
            self.suspect_start = rows[0][2]
        self.hits += 1
        if self.hits >= c.consecutive:
            self.detected = Detection(self.suspect_start, {
                "kind": "collapse", "window_words": len(rows), "unique_ratio": unique,
                "kgram_coverage": coverage, "prose_words": prose,
                "prose_unique_ratio": prose_unique, "prose_kgram_coverage": prose_coverage,
                "chars_seen": self.chars, "words_seen": self.words,
                "consecutive": self.hits,
            })
