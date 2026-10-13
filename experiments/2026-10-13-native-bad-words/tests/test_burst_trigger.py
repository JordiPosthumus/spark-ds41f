"""Short-window trigger boundaries and streaming/no-rewind integration."""
from test_api_semantics import (unittest, TemporaryPhraseTrigger, Engine, row,
    params, ids, generate_with_recovery, RecoveryConfig, PhraseDensity)


def detect(text, width=1):
    d = TemporaryPhraseTrigger()
    for i in range(0, len(text), width):
        d.feed(text[i:i+width], i, i+width)
    return d


class BurstTests(unittest.TestCase):
    def test_four_prefixes_without_warmup(self):
        text = 'Let me run. Let me go. Let me write. Let me capture. '
        for width in (1, 2, 7, len(text)):
            d = detect(text, width)
            self.assertEqual(d.evidence['rule'], 'let_me_4_in_50')
            self.assertEqual(d.evidence['words_seen'], 11)
            self.assertEqual(d.evidence['commands'], 4)
        self.assertFalse(detect('Let me run. '*3).triggered)

    def test_combined_and_case_punctuation(self):
        for text in ('Hmm. hMm, HMMM! hmm; Hmm — hMm. ',
                     'Hmm. Let me check. HMM. LET\tME think. Hmmm. let\nme act. '):
            for width in (1, 7, len(text)):
                d = detect(text, width)
                self.assertEqual(d.evidence['rule'], 'combined_6_in_100')
                self.assertEqual(d.evidence['hmm'] + d.evidence['commands'], 6)
        self.assertFalse(detect('Hmm. '*5).triggered)

    def test_exact_window_edges_include_both_prefix_words(self):
        # First let at word 1, final me at 50 (inclusive) or 51 (expired).
        for padding, wanted in ((42, True), (43, False)):
            text = 'Let me ' + 'safe '*padding + 'Let me '*3
            self.assertEqual(detect(text).triggered, wanted)
        for padding, wanted in ((94, True), (95, False)):
            self.assertEqual(detect('Hmm '+('safe '*padding)+'Hmm '*5).triggered, wanted)

    def test_code_ages_hits_without_supplying_them(self):
        text = 'Let me run. '*3 + '\n```text\n' + 'safe '*100 + '\n```\nLet me act. '
        self.assertFalse(detect(text).triggered)
        for width in (1, 7, 10000):
            self.assertFalse(detect('```text\n'+'Hmm. Let me run. '*50+'\n```\n', width).triggered)

    def test_normal_reasoning_and_substrings(self):
        samples = [
            'Let me inspect the file. Its first record has a date. Hmm, the second record is missing one. The solution is to validate the date before saving. Let me check the test result. It passed.',
            'Let memory work. hmmfoo. Let, me run. Let—me go. '*30,
            ('Let me check. '+'useful '*50)*10,
            'Actually, the value differs. '*30,
        ]
        for text in samples:
            self.assertFalse(detect(text).triggered)

    def test_incomplete_word_waits_and_requests_are_isolated(self):
        d = detect('Let me run. '*3+'Let m')
        self.assertFalse(d.triggered)
        d.feed('e', 99, 100)
        self.assertFalse(d.triggered)
        d.feed(' ', 100, 101)
        self.assertTrue(d.triggered)
        self.assertFalse(detect('Let me go. ').triggered)

    def test_dense_but_legitimate_repetition_is_documented_tradeoff(self):
        d = detect('Let me check the first result. Let me check the second result. Let me check the third result. Let me check the fourth result. ')
        self.assertTrue(d.triggered)  # Concentration is not a semantic classifier.


class BurstIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_trigger_keeps_all_prefix_and_arms_2000_words(self):
        pieces = ['Let me run. ', 'Let me go. ', 'Let me write. ', 'Let me capture. ']
        engine = Engine([[row(x) for x in pieces], [row('The result is verified. </think>42', True)]])
        events = []; original = params(); prompt = {'type':'token','prompt_token_ids':[11,12]}
        out = [x async for x in generate_with_recovery(engine, prompt, original, 'burst',
            config=RecoveryConfig(mode='auto-recover', normal_holdback_tokens=0, temporary_phrases=True),
            detector_factory=PhraseDensity, on_event=events.append, reasoning_ended=False)]
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[1][0]['prompt_token_ids'], [11,12]+sum((ids(x) for x in pieces), []))
        self.assertNotEqual(engine.calls[1][1].seed, original.seed)
        self.assertEqual(engine.calls[1][1].temperature, original.temperature)
        self.assertIsNotNone(engine.calls[1][1]._ds41f_loop_protection)
        activated = next(x for x in events if x['event']=='temporary_phrases_activated')
        self.assertEqual(activated['word_limit'], 2000)
        self.assertEqual(activated['rule'], 'let_me_4_in_50')
        self.assertEqual(events[-1]['retries'], 0)
        self.assertEqual(events[-1]['discarded_generated_tokens'], 0)
        self.assertTrue(events[-1]['completed'])
        self.assertFalse(any(x['event']=='loop_detected' for x in events))
