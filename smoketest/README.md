# smoketest

    smoketest/run.sh http://127.0.0.1:8002

Cases are shell functions in [run.sh](run.sh); checks are jq expressions. Needs
curl and jq. Run against the head's API or through mentat-serve; both pass
(8/8 through the router, 2026-09-23). Exit status is the number of failed cases.

`page-table.png` is the same synthetic page the dots-ocr and qwen36-a3b suites
use: one question that can only be answered by reading it.

Speed, recall and determinism are not here: they take minutes, and belong in a
benchmark on a quiet box (profile.py, dev/repro/needle.py, dev/repro/greedy_nondet.py).
