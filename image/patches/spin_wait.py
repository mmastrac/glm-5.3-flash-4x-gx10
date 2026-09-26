#!/usr/bin/env python3
"""Cut the shm_broadcast reader's busy-wait from 1 s to 2 ms.

vLLM's MessageQueue readers spin for `busy_loop_s` after each read before they
start sleeping. At the default of 1 s every TP rank spins almost continuously,
and on GB10 that spin is not free: CPU and GPU share one thermal and power
budget, so it cost ~20 C and measurable decode. 0.002 keeps the fast path for
back-to-back steps; always-blocking (0) measured WORSE. After
https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/.

Replaces the whole-file mount of shm_broadcast.py the old deployment used.
Same contract as the other patchers: one anchor, exactly once, or the build
fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py")
OLD = "        busy_loop_s: float = 1,\n"
NEW = "        busy_loop_s: float = 0.002,\n"

text = PATH.read_text()
if text.count(NEW) == 1 and text.count(OLD) == 0:
    print("[spin-wait] already 0.002")
else:
    assert text.count(OLD) == 1, f"[spin-wait] anchor matched {text.count(OLD)} times; stock tree changed"
    PATH.write_text(text.replace(OLD, NEW))
    print("[spin-wait] busy_loop_s default 1 -> 0.002")
