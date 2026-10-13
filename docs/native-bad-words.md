# Temporary native bad_words deployment

The approved 16-entry list is in
`experiments/2026-10-13-native-bad-words/tables/bad_words.json`.
The existing detector activates vLLM bad_words for 2000 completed words or until
an earlier reasoning/tool boundary. Temperature is unchanged. Existing early
continuation and severe-loop retry mechanics are preserved.

The owner requested immediate staged deployment and will evaluate loop behavior.
API/detector regression checks pass; startup/readiness alone do not demonstrate
successful model recovery. See the experiment README for exact implementation,
limitations, validation tooling, and rollback instructions.
