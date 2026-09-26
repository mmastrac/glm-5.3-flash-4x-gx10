"""Decode rate over N short-prompt runs, plus DFlash2 draft acceptance.

Acceptance length, not rate, is what predicts throughput for a k-token drafter.
"""
import json, re, statistics, sys, time, urllib.request
BASE, MODEL, N = sys.argv[1], sys.argv[2], int(sys.argv[3])

def metrics():
    raw = urllib.request.urlopen(BASE.replace("/v1", "") + "/metrics", timeout=30).read().decode()
    out = {}
    for key in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"):
        m = re.search(rf"^vllm:{key}\S*\s+([0-9.e+]+)$", raw, re.M)
        if m:
            out[key] = float(m.group(1))
    return out

def once(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                if (delta.get("content") or delta.get("reasoning")) and ttft is None:
                    ttft = time.time() - t0
    total = time.time() - t0
    completion = (usage or {}).get("completion_tokens", 0)
    return (completion - 1) / (total - ttft) if ttft and total > ttft else 0

before = metrics()
rates = [once("Explain unified memory on GB10 in two sentences.", 200) for _ in range(N)]
after = metrics()
acc = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
draft = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
print(f"  decode tok/s over {N} runs: " + " ".join(f"{r:.1f}" for r in sorted(rates)))
print(f"  median {statistics.median(rates):.1f}  mean {statistics.mean(rates):.1f}  "
      f"spread {max(rates)-min(rates):.1f}")
if draft:
    print(f"  draft: accepted {acc:.0f} of {draft:.0f} = {acc/draft:.3f}; "
          f"accepted per step {acc/(draft/7):.2f} of 7")
