"""Trigger for owner-authorized bounded native phrase suppression."""
from collections import Counter
import unicodedata
from .collapse import CollapseDetector, CollapseConfig
from .phrase_detector import hashed, LET, ME, HMM


class TemporaryPhraseTrigger(CollapseDetector):
    """Two 200-word windows, 32 words apart, each with >=8 targeted hits.

    Fenced code is excluded. Detection is case-insensitive and counts Hmm
    variants and every whole whitespace-separated let-me prefix.
    This is an experimental trigger, not proof of semantic failure.
    """
    def __init__(self):
        super().__init__(CollapseConfig(window=200, every=32, consecutive=2))
        self.triggered = False
        self.evidence = None
        self._separator_seen = False
        self._separator_whitespace = True
        self._word_separator = False

    def feed(self, text, start_token, end_token):
        # Keep only separator flags, never reasoning text. The base detector
        # owns word hashing and fenced-code state across arbitrary chunk splits.
        for char in text:
            is_word = char == "_" or unicodedata.category(char)[0] in "LN"
            if is_word and self.word is None:
                self._word_separator = self._separator_seen and self._separator_whitespace
                self._separator_seen = False
                self._separator_whitespace = True
            super().feed(char, start_token, end_token)
            if not is_word:
                self._separator_seen = True
                self._separator_whitespace &= char.isspace()
        return self.detected

    def _flush(self):
        if self.word is None:
            return
        self.rows.append((self.word.digest()[:12], self.word_prose, self.word_start, self._word_separator))
        self.words += 1
        self.word = None
        self.word_chars = 0
        if self.words < self.next_evaluation:
            return
        self.next_evaluation = self.words + 32
        rows = list(self.rows)
        prose = [r[0] for r in rows if r[1]]
        counts = Counter(prose)
        hmm = sum(counts[k] for k in HMM)
        commands = sum(a[1] and b[1] and b[3] and a[0] == LET and b[0] == ME
                       for a, b in zip(rows, rows[1:]))
        total = hmm + commands
        match = len(prose) >= 160 and total >= 8
        self.hits = self.hits + 1 if match else 0
        if self.hits >= 2:
            self.triggered = True
            self.evidence = {"hmm": hmm, "commands": commands,
                             "window_words": len(prose), "words_seen": self.words}

class WordCounter:
    """Count completed Unicode letter/number words across arbitrary deltas."""
    def __init__(self):
        self.words = 0
        self.in_word = False

    def feed(self, text):
        for char in text:
            is_word = char == "_" or unicodedata.category(char)[0] in "LN"
            if self.in_word and not is_word:
                self.words += 1
            self.in_word = is_word
        return self.words
