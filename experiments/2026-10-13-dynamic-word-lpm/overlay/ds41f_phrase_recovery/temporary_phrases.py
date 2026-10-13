"""Bounded same-word-family trigger for temporary native bad_words protection."""
from collections import deque
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import unicodedata
from .collapse import CollapseDetector, CollapseConfig


@lru_cache(maxsize=1)
def word_policy():
    root = Path(os.environ["SPARK_DS41F_PHRASE_TABLES"])
    policy = json.loads((root / "word_policy.json").read_text())
    flat = list(dict.fromkeys(value for family in policy["families"].values()
                             for value in family["block"]))
    if flat != json.loads((root / "bad_words.json").read_text()):
        raise ValueError("Word-family policy and native blocking entries differ")
    width, threshold = policy["window_words"], policy["threshold"]
    if width != 50 or threshold != 4:
        raise ValueError("Unreviewed word-family thresholds")
    by_last = {}
    for family, settings in policy["families"].items():
        forms = set(settings["block"])
        # Preserve recognition of the previously supported long Hmm forms.
        # Concrete blocking spellings remain explicit; no Unicode/regex claim.
        if family == "hm":
            forms.update("h" + "m" * count for count in range(
                1, policy["detection_only_legacy_hm_max_m"] + 1))
        patterns = {tuple(hashlib.sha256(word.lower().encode()).digest()[:12]
                          for word in value.split()) for value in forms}
        for pattern in patterns:
            if not 1 <= len(pattern) <= 3:
                raise ValueError("Word-family spelling too long")
            by_last.setdefault(pattern[-1], []).append((family, pattern))
    return policy, by_last


class TemporaryPhraseTrigger(CollapseDetector):
    """Four hits of any one family in 50 words, checked at word boundaries.

    Counts families separately; Let me contributes one let and one me, never
    two hits to a single counter. Recognized spaced interjections count once.
    Code words age the window but cannot supply hits. Retains hashes/positions.
    """
    def __init__(self):
        super().__init__(CollapseConfig(window=200, every=32, consecutive=2))
        self.policy, self._patterns = word_policy()
        self._family_hits = {name: deque() for name in self.policy["families"]}
        self.triggered = False
        self.evidence = None
        self._separator_seen = False
        self._separator_whitespace = True
        self._word_separator = False

    def feed(self, text, start_token, end_token):
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
        self.words += 1
        self.rows.append((self.word.digest()[:12], self.word_prose,
                          self.word_start, self._word_separator, self.words))
        self.word = None
        self.word_chars = 0
        if self.triggered:
            return
        width = self.policy["window_words"]
        lower = self.words - width + 1
        for hits in self._family_hits.values():
            while hits and hits[0] < lower:
                hits.popleft()
        current = self.rows[-1]
        if current[1]:
            rows = list(self.rows)
            for family, pattern in self._patterns.get(current[0], ()):
                tail = rows[-len(pattern):]
                if (len(tail) == len(pattern) and tail[0][4] >= lower
                        and all(row[1] for row in tail)
                        and all(row[3] for row in tail[1:])
                        and tuple(row[0] for row in tail) == pattern):
                    hits = self._family_hits[family]
                    start = tail[0][4]
                    # H m and its extension H m m are one interjection.
                    if not hits or hits[-1] != start:
                        hits.append(start)
        for family, hits in self._family_hits.items():
            if len(hits) >= self.policy["threshold"]:
                recent = list(self.rows)[-width:]
                self.triggered = True
                self.evidence = {
                    "rule": "word_family_4_in_50", "family": family,
                    "hits": len(hits),
                    "first_hit_word": hits[0],
                    "rewind_start_token": next(row[2] for row in recent if row[4] == hits[0]),
                    "family_counts": {name: len(values) for name, values in self._family_hits.items()},
                    "window_words": len(recent), "prose_words": sum(row[1] for row in recent),
                    "window_limit": width, "words_seen": self.words,
                }
                break


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
