#!/usr/bin/env python3
"""Identical greedy requests over a long prompt return different completions.

Standard library only. Point it at a running vLLM OpenAI server:

    python greedy_nondet.py --url http://127.0.0.1:8000 --model glm53

Sends the same ~42k-token prompt N times at temperature 0 with a fixed seed and
groups the completions. More than one group means the forward pass is not
reproducible.
"""
import argparse
import collections
import hashlib
import json
import queue
import threading
import urllib.request

# Repeated verbatim to reach the target length. Content does not matter: any
# filler of the same size behaves the same.
PARAGRAPH = """The job runner reads a JSON manifest, then runs each job in order.
Every job carries a name, a command, a retry count and an environment overlay.
The overlay is applied on top of the inherited environment on each attempt, so a
key set to null must remove the inherited value rather than set it to the empty
string. Failures are retried up to the configured count, and the runner returns
the worst exit code it saw. The exporter reads the same manifest and publishes
one gauge per job plus a counter for retries.
"""

QUESTION = "\nIn one sentence: what does the runner return when every job fails?"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build_prompt(base, model, target):
    """Repeat PARAGRAPH until the tokenizer reports at least `target` tokens."""
    low, high = 1, 16
    while True:
        text = PARAGRAPH * high + QUESTION
        if count_tokens(base, model, text) >= target:
            break
        low, high = high, high * 2
    while low < high:
        mid = (low + high) // 2
        if count_tokens(base, model, PARAGRAPH * mid + QUESTION) >= target:
            high = mid
        else:
            low = mid + 1
    text = PARAGRAPH * low + QUESTION
    return text, count_tokens(base, model, text)


def count_tokens(base, model, text):
    return post(base + "/tokenize", {"model": model, "prompt": text}, timeout=300)["count"]


def complete(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 1234,
        "max_tokens": max_tokens,
    }
    choice = post(base + "/v1/chat/completions", body)["choices"][0]
    message = choice["message"]
    # Compare the reasoning too, under either spelling: `reasoning_content` from
    # SGLang, `reasoning` from vLLM. A thinking model whose trace does not close
    # inside max_tokens returns empty content, and hashing content alone then
    # groups every run together and reports perfect agreement -- the divergence
    # this script exists to find happens in the trace as readily as in the answer.
    text = "".join(
        message.get(key) or ""
        for key in ("reasoning_content", "reasoning", "content")
    )
    if not text:
        raise SystemExit(
            f"empty completion and empty reasoning ({choice.get('finish_reason')}, "
            f"{max_tokens} max_tokens): nothing to compare. Raise --max-tokens so "
            "the trace can close, or send reasoning_effort=none."
        )
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=42000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    base = a.url.rstrip("/")

    prompt, n_tokens = build_prompt(base, a.model, a.prompt_tokens)
    print(f"prompt: {n_tokens} tokens, {a.runs} runs at temperature 0, seed 1234")

    results, lock = [], threading.Lock()
    work = queue.Queue()
    for i in range(a.runs):
        work.put(i)

    def worker():
        while True:
            try:
                work.get_nowait()
            except queue.Empty:
                return
            try:
                out = complete(base, a.model, prompt, a.max_tokens)
            except Exception as exc:
                out = f"<error {exc!r}>"
            with lock:
                results.append(out)
                print(f"  {len(results):>3}/{a.runs} {hashlib.sha1(out.encode()).hexdigest()[:8]}",
                      flush=True)

    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    groups = collections.Counter(results)
    print(f"\n{len(groups)} distinct completion(s) from {a.runs} identical requests")
    for text, n in groups.most_common():
        print(f"\n--- {n} run(s) ---\n{text}")
    raise SystemExit(0 if len(groups) == 1 else 1)


if __name__ == "__main__":
    main()
