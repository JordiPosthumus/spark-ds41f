"""Bounded rolling prose-word counts. No generated text is written to logs."""
from collections import Counter, deque
import unicodedata


def canonical(word):
    return unicodedata.normalize("NFKC", word).casefold()


def word_char(char):
    return char == "_" or unicodedata.category(char)[0] in "LNM"


class DynamicWords:
    def __init__(self, policy, base_phrases=()):
        self.width = policy["window_characters"]
        self.threshold = policy["more_than"]
        self.excluded = set(policy["grammar_exclusions"])
        self.base = {canonical(p) for p in base_phrases}
        self.blocked = {}  # Canonical word -> spelling used in Pi notice.
        self.rows = deque()
        self.counts = Counter()
        self.position = 0
        self.word = ""
        self.word_start = 0
        self.word_prose = True
        self.fence_char = None
        self.fence_width = 0
        self.line_start = True
        self.indent = 0
        self.ticks = ""
        self.inline_width = 0
        self.pending = []

    def _flush(self):
        if not self.word:
            return
        text, start, prose = self.word, self.word_start, self.word_prose
        self.word = ""
        key = canonical(text)
        if prose and start >= self.position - self.width:
            self.rows.append((start, key))
            self.counts[key] += 1
            if (self.counts[key] > self.threshold and key not in self.excluded
                    and key not in self.base and key not in self.blocked
                    and any(c.isalpha() for c in text)):
                self.blocked[key] = text
                self.pending.append(text)

    def _resolve_ticks(self):
        if not self.ticks:
            return
        ticks = self.ticks
        self.ticks = ""
        if self.line_start and len(ticks) >= 3:
            if self.fence_char is None:
                self.fence_char, self.fence_width = ticks[0], len(ticks)
            elif ticks[0] == self.fence_char and len(ticks) >= self.fence_width:
                self.fence_char, self.fence_width = None, 0
        elif ticks[0] == "`" and self.fence_char is None:
            if not self.inline_width:
                self.inline_width = len(ticks)
            elif len(ticks) == self.inline_width:
                self.inline_width = 0
        self.line_start = False

    def feed(self, text):
        self.pending = []
        for char in text:
            self.position += 1
            while self.rows and self.rows[0][0] < self.position - self.width:
                _, key = self.rows.popleft()
                self.counts[key] -= 1
                if not self.counts[key]:
                    del self.counts[key]
            if char in "`~" and (not self.ticks or char == self.ticks[0]):
                self._flush()
                self.ticks += char
                continue
            self._resolve_ticks()
            if word_char(char):
                if not self.word:
                    self.word_start = self.position - 1
                    self.word_prose = self.fence_char is None and not self.inline_width and self.indent < 4
                self.word += char
                # A word longer than the entire window cannot qualify.
                if len(self.word) > self.width:
                    self.word = self.word[-self.width:]
                    self.word_prose = False
            else:
                self._flush()
            if char == "\n":
                self.line_start, self.indent = True, 0
            elif self.line_start and char in " \t":
                self.indent += 4 if char == "\t" else 1
            elif not char.isspace():
                self.line_start = False
        return list(self.pending)


def spellings(word):
    # Preserve the owner's explicit capital-A-only Actually exception.
    if canonical(word) == "actually":
        return ["Actually"]
    normalized = unicodedata.normalize("NFKC", word)
    return list(dict.fromkeys((word, normalized, normalized.lower(), normalized.title(), normalized.upper())))


def encode_words(tokenizer, words):
    # Use the native bad_words tokenizer exactly, including its leading-space
    # variants. This is token-sequence blocking, not a regex/substring guarantee.
    from vllm.sampling_params import SamplingParams
    phrases = list(dict.fromkeys(s for w in words for s in spellings(w)))
    if not phrases:
        return []
    ids = []
    # Tokenize each literal through native vLLM so its public per-request
    # 128-sequence limit does not incorrectly cap our separate dynamic table.
    for phrase in phrases:
        params = SamplingParams(bad_words=[phrase])
        params.update_from_tokenizer(tokenizer)
        ids.extend(tuple(x) for x in params.bad_words_token_ids)
    return [list(x) for x in dict.fromkeys(ids)]
