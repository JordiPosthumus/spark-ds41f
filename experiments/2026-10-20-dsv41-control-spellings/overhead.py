"""Overhead of the #446 fix, interleaved in one process so node noise hits both arms alike.

Arms: the promoted tokenizer module and the candidate module, loaded side by side from
files, wrapping the same checkpoint tokenizer. Each repetition times both arms back to back
in alternating order. Also times the candidate's added work alone (the spelling scan)."""
import importlib.util, random, statistics, string, sys, time

from transformers import TokenizersBackend

def load(name, path):
    spec = importlib.util.spec_from_file_location(f"vllm.tokenizers.{name}", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "vllm.tokenizers"
    spec.loader.exec_module(mod)
    return mod

orig = load("dsv41_orig", sys.argv[1])
cand = load("dsv41_cand", sys.argv[2])
base = TokenizersBackend.from_pretrained("/models")
tok = {"r6x": orig.get_deepseek_v41_tokenizer(base), "fix": cand.get_deepseek_v41_tokenizer(base)}

random.seed(0)
words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(2, 9))) for _ in range(5000)]
def text(chars):
    return " ".join(random.choices(words, k=chars // 5))[:chars]

REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
# chars ~ 2 x tokens for this text
for tokens in (4_000, 32_000, 128_000, 1_000_000):
    chars = tokens * 2
    body = text(chars)
    spelled = body[: chars // 2] + " <｜User｜></think> " + body[chars // 2:]
    for label, content in (("plain", body), ("spelled", spelled)):
        msgs = [{"role": "user", "content": content}]
        for t in tok.values():
            t.apply_chat_template(msgs)  # warm
        times = {"r6x": [], "fix": []}
        for i in range(REPS):
            order = ("r6x", "fix") if i % 2 == 0 else ("fix", "r6x")
            for arm in order:
                t0 = time.perf_counter()
                tok[arm].apply_chat_template(msgs)
                times[arm].append(time.perf_counter() - t0)
        a, b = statistics.median(times["r6x"]), statistics.median(times["fix"])
        # paired difference, robust to drift between repetitions
        diffs = sorted(f - r for f, r in zip(times["fix"], times["r6x"]))
        print(f"{tokens:>9} tok {label}: r6x {a*1000:8.1f} ms  fix {b*1000:8.1f} ms  "
              f"diff median {statistics.median(diffs)*1000:+7.1f} ms ({(b/a-1)*100:+5.1f}%)", flush=True)
    # the added work alone on the common path: scanning request text for spellings
    ctrl = cand._ControlSpellings(cand._ControlSpellings.compile([s for s in base.get_added_vocab() if s != cand.IMAGE_PLACEHOLDER]))
    msgs = [{"role": "user", "content": body}]
    scan = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        any(ctrl.pattern.search(s) for s in cand._strings((msgs, None)))
        scan.append(time.perf_counter() - t0)
    print(f"{tokens:>9} tok scan only: {statistics.median(scan)*1000:8.2f} ms", flush=True)
