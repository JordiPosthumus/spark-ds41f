"""Owner-authorized bounded phrase intervention; no GPU/sampler modifications."""
from collections import Counter
import unicodedata
from .collapse import CollapseDetector, CollapseConfig
from .phrase_detector import hashed, LET, ME, HMM

RUN, GO = hashed("run"), hashed("go")

# Exact punctuated phrases, ordinary/title/upper case. Native tokenizer also
# supplies leading-space forms. Individual ordinary words are never banned.
PHRASES = tuple(dict.fromkeys(
    variant + punctuation
    for phrase in ("hmm", "hmmm", "hmmmm", "let me run", "let me go")
    for variant in (phrase, phrase.capitalize(), phrase.upper())
    for punctuation in ((",", ".") if phrase.startswith("h") else (".",))
))

class TemporaryPhraseTrigger(CollapseDetector):
    """Two 200-word windows, 32 words apart, each with >=8 targeted hits.

    Fenced code is excluded. Detection is case-insensitive and counts Hmm
    variants and the whole let-me-run/go sequence, not every 'let me'.
    This is an experimental trigger, not proof of semantic failure.
    """
    def __init__(self):
        super().__init__(CollapseConfig(window=200, every=32, consecutive=2))
        self.triggered = False
        self.evidence = None

    def _flush(self):
        if self.word is None:
            return
        self.rows.append((self.word.digest()[:12], self.word_prose, self.word_start))
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
        commands = sum(all(r[1] for r in rows[i:i+3]) and
                       tuple(r[0] for r in rows[i:i+3]) in ((LET, ME, RUN), (LET, ME, GO))
                       for i in range(len(rows)-2))
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
