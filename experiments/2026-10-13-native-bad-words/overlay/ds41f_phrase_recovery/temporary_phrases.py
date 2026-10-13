"""Trigger for owner-authorized bounded native phrase suppression."""
from collections import deque
import unicodedata
from .collapse import CollapseDetector, CollapseConfig
from .phrase_detector import LET, ME, HMM


class TemporaryPhraseTrigger(CollapseDetector):
    """Check every completed word: 4 Let me/50 or 6 combined hits/100.

    No warm-up or second confirmation. Fenced code never supplies hits, but
    still occupies the word window. Whole whitespace-separated prefixes and
    Hmm variants are case-insensitive. This heuristic can flag valid prose.
    Only bounded hashes/positions are retained, never reasoning text.
    """
    def __init__(self):
        super().__init__(CollapseConfig(window=200, every=32, consecutive=2))
        self._commands50 = deque()
        self._commands100 = deque()
        self._hmm100 = deque()
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
        if self.triggered:
            return
        # Store the phrase's FIRST word position: both words must fit inside
        # the window. Code and unrelated words age hits out normally.
        current = self.rows[-1]
        if current[1] and current[0] in HMM:
            self._hmm100.append(self.words)
        if len(self.rows) >= 2:
            previous = self.rows[-2]
            if (previous[1] and current[1] and current[3]
                    and previous[0] == LET and current[0] == ME):
                self._commands50.append(self.words - 1)
                self._commands100.append(self.words - 1)
        for positions, width in ((self._commands50, 50),
                                 (self._commands100, 100), (self._hmm100, 100)):
            while positions and positions[0] < self.words - width + 1:
                positions.popleft()
        short = len(self._commands50) >= 4
        combined = len(self._commands100) + len(self._hmm100) >= 6
        if short or combined:
            width = 50 if short else 100
            rows = list(self.rows)[-width:]
            self.triggered = True
            self.evidence = {
                "rule": "let_me_4_in_50" if short else "combined_6_in_100",
                "hmm": sum(pos >= self.words - width + 1 for pos in self._hmm100),
                "commands": len(self._commands50 if short else self._commands100),
                "window_words": len(rows), "prose_words": sum(r[1] for r in rows),
                "window_limit": width, "words_seen": self.words,
            }

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
