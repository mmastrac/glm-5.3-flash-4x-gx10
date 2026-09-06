#!/usr/bin/env python3
"""Prove the model actually works before the node reports itself as serving.

This exists because of a specific failure mode: a TP/shard/quantisation mismatch
loads CLEANLY, binds the API, reports healthy, and then serves fluent nonsense.
Nothing upstream of a real generation catches it. So the head asks a handful of
questions with unambiguous answers and refuses to advertise itself until they
come back right.

The MTP acceptance rate is the second signal and the more sensitive one: a
correctly loaded model runs ~85-97%, while corrupted weights collapse to ~50%
because the drafter and the target no longer agree. It moves long before the
answers become visibly wrong, so it catches partial corruption that a
capital-city question would sail straight through.

Exit 0 = healthy. Exit 1 = the model is up but wrong.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# Answers must be unambiguous, single-token-ish, and not sensitive to phrasing.
# Each entry is (prompt, accepted substrings) -- matched case-insensitively
# against the FINAL answer only, never the reasoning trace, which will happily
# mention the right answer while the answer itself is wrong.
CHECKS = [
    ("What is the capital of France? Answer with one word only.", ["paris"]),
    ("What is 12 + 7? Answer with just the number.", ["19"]),
    ("Complete with one word: the opposite of hot is", ["cold"]),
]

# Health is judged on ACCEPTANCE LENGTH, not the per-token rate.
#
# With k draft tokens per step, length ranges 1..(k+1) and per-token rate falls
# naturally as k grows -- later positions are harder to predict. Measured on
# Qwen3.8: k=1 gives rate 92% / length 2.0, while k=7 gives rate 57% / length
# 5.0. The second is 2.5x better despite the far worse-looking rate, and a
# rate-based threshold flagged it as broken. Length near 1.0 is the real
# corruption signal: it means nothing is being accepted at all.
MIN_ACCEPTANCE_LENGTH_FRACTION = 0.25   # of the theoretical max (k+1)


# Probes are retried only for timeouts, which on a busy box mean queueing
# rather than breakage. Three tries at the default 300s bounds the gate at
# ~15 minutes worst case, which still beats reporting a healthy model broken.
RETRIES = 3
RETRY_SLEEP_S = 10


def _is_timeout(e: BaseException) -> bool:
    """A read timeout arrives bare; a connect timeout arrives wrapped."""
    return isinstance(e, TimeoutError) or (
        isinstance(e, urllib.error.URLError)
        and isinstance(getattr(e, "reason", None), (TimeoutError, OSError))
        and "timed out" in str(e.reason).lower()
    )


def post(base, model, prompt, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Generous: this is a reasoning model, and a budget too small yields an
        # empty `content` with everything spent in `reasoning_content` -- which
        # looks exactly like a failure but is not one.
        "max_tokens": 800,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def acceptance(base):
    """(per-token rate, acceptance length, k) or (None, None, None) if unused.

    Acceptance length is the mean tokens emitted per verification step, which is
    what actually drives throughput. k is inferred from the per-position buckets.
    """
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return None
    vals = {}
    for line in text.splitlines():
        for key in ("spec_decode_num_draft_tokens_total",
                    "spec_decode_num_accepted_tokens_total"):
            if line.startswith(f"vllm:{key}{{"):
                try:
                    vals[key] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    d = vals.get("spec_decode_num_draft_tokens_total", 0.0)
    a = vals.get("spec_decode_num_accepted_tokens_total", 0.0)
    if d <= 0:
        return None, None, None
    # Positions are 0..k-1, so the bucket count gives k directly.
    k = len(re.findall(r'spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="', text))
    steps = d / k if k else d
    length = 1 + (a / steps) if steps else 1.0
    return a / d, length, k


def wait_for_api(base, deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="ds4-flash")
    ap.add_argument("--wait", type=int, default=2400,
                    help="seconds to wait for the API to bind")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-request timeout")
    a = ap.parse_args()

    if not wait_for_api(a.base, time.time() + a.wait):
        print(f"self-test: API never bound within {a.wait}s", flush=True)
        return 1

    failures = []
    for prompt, expect in CHECKS:
        # A timeout is not a wrong answer. This gate exists to catch a model
        # that loads cleanly and serves fluent nonsense, and a request that
        # never came back says nothing about that. On a box already carrying
        # traffic a probe can simply queue behind someone else's long prefill
        # -- observed 2026-08-25, where two probes answered correctly, MTP
        # measured 88.7%, and the third timed out purely from contention.
        # Retry timeouts; let every other error fail on the first try.
        answer = None
        for attempt in range(1, RETRIES + 1):
            try:
                d = post(a.base, a.model, prompt, a.timeout)
                msg = d["choices"][0]["message"]
                answer = (msg.get("content") or "").strip()
                break
            except Exception as e:
                if not _is_timeout(e):
                    failures.append(
                        f"{prompt!r} -> request failed: {type(e).__name__}: {e}")
                    break
                if attempt < RETRIES:
                    print(f"self-test: retry {prompt[:44]!r} after timeout "
                          f"({attempt}/{RETRIES})", flush=True)
                    time.sleep(RETRY_SLEEP_S)
                    continue
                failures.append(
                    f"{prompt!r} -> timed out {RETRIES}x at {a.timeout}s each; "
                    f"the server is up but too busy to answer, which is not the "
                    f"same as wrong")
        if answer is None:
            continue
        if any(e in answer.lower() for e in expect):
            print(f"self-test: ok    {prompt[:44]!r} -> {answer[:40]!r}", flush=True)
        else:
            failures.append(f"{prompt!r} -> got {answer[:80]!r}, expected one of {expect}")
            print(f"self-test: FAIL  {prompt[:44]!r} -> {answer[:60]!r}", flush=True)

    acc, length, k = acceptance(a.base)
    if acc is None:
        print("self-test: speculative decoding not in use (no drafts produced)",
              flush=True)
    else:
        floor = (k + 1) * MIN_ACCEPTANCE_LENGTH_FRACTION if k else 0
        print(f"self-test: acceptance length {length:.2f} of max {k + 1} "
              f"(per-token rate {acc:.1%}, k={k})", flush=True)
        if k and length < floor:
            failures.append(
                f"acceptance length {length:.2f} is below {floor:.2f} "
                f"({MIN_ACCEPTANCE_LENGTH_FRACTION:.0%} of the max {k + 1}) -- "
                "near 1.0 means drafts are never accepted, the signature of "
                "corrupted or mismatched weights")

    if failures:
        print("\nself-test FAILED:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    print("self-test: PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
