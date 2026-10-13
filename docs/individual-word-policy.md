# Individual words during loop protection

The owner requested individual Let/let/LET, Me/me/ME, Cyrillic меня/Меня/МЕНЯ
and explicitly listed Latin/Cyrillic variants, replacing compound Let-me phrases.
The current exact list and codepoints are under
`experiments/2026-10-13-individual-word-policy/`.

All other behavior is unchanged, including the native bad_words implementation,
2000-word lifetime, temperature and existing recovery mechanics. No loop-blocking
evaluation is claimed: the owner will evaluate behavior. This finite list cannot
guarantee coverage of all alternative spellings or token segmentations.
