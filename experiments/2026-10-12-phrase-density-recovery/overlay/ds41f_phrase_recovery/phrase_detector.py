"""Experimental bounded phrase-density detector; no sampling restrictions.

Reuses streaming word boundaries and fenced-code exclusion. Stores word hashes,
not private reasoning. Policy needs representative production validation.
"""
from collections import Counter
import hashlib
from collapse import CollapseDetector, CollapseConfig, Detection


def hashed(word):
    return hashlib.sha256(word.encode()).digest()[:12]

LET, ME, ACTUALLY = map(hashed, ('let', 'me', 'actually'))
HMM = {hashed('h' + 'm' * n) for n in range(2, 17)}


class PhraseDensity(CollapseDetector):
    def __init__(self, consecutive=8, density=.24):
        # Base config owns the bounded deque and streaming fence state only.
        super().__init__(CollapseConfig(window=200, every=32, consecutive=2))
        self.next_evaluation = 200
        self.required_hits = consecutive
        self.density = density

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
        let_me = sum(a[1] and b[1] and a[0] == LET and b[0] == ME
                     for a, b in zip(rows, rows[1:]))
        counts = Counter(prose)
        hmm = sum(counts[k] for k in HMM)
        actually = counts[ACTUALLY]
        total = let_me + hmm + actually
        unique = len(counts) / len(prose) if prose else 1
        # All three are evidence, never banned words. Require repeated bursts
        # plus low diversity; ordinary self-correction alone is insufficient.
        match = (len(prose) >= 160 and total >= 16 and total / len(prose) >= self.density
                 and unique <= .55)
        if not match:
            self.hits, self.suspect_start = 0, None
            return
        if not self.hits:
            self.suspect_start = rows[0][2]
        self.hits += 1
        if self.hits >= self.required_hits:
            self.detected = Detection(self.suspect_start, {
                'kind': 'phrase_density', 'chars_seen': self.chars,
                'words_seen': self.words, 'window_words': len(rows),
                'let_me': let_me, 'actually': actually, 'hmm': hmm,
                'phrase_density': total / len(prose), 'unique_ratio': unique,
                'consecutive': self.hits})
