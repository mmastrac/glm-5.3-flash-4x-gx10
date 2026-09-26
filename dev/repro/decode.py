# Per-workload decode with DFlash2 acceptance: thinking off, temperature 0,
# 512 tokens, median of 3 after one discarded warm-up. Same prompts as the
# 2026-09-06..26 measurements. usage: decode.py [base-url]
import json, re, statistics, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"  # no /v1
W = [("structured", "Count from 1 to 200, comma separated. No commentary."),
     ("code", "Write a red-black tree in Python with insert, delete and rebalancing. Code only."),
     ("prose", "Explain how a hash map works, in flowing prose. No code, no lists.")]
def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}\S*\s+([0-9.e+]+)$", raw, re.M))
    return get("spec_decode_num_accepted_tokens_total"), get("spec_decode_num_draft_tokens_total")
def run(prompt, n=512):
    body = json.dumps({"model": "glm53", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"thinking": False}}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.time(); d = json.loads(urllib.request.urlopen(r, timeout=600).read()); e = time.time() - t
    return d["usage"]["completion_tokens"] / e
print("%-12s %10s %8s %s" % ("workload", "tok/s", "accept", "runs"))
for name, p in W:
    run(p, 16)
    a0, d0 = metrics()
    rates = [run(p) for _ in range(3)]
    a1, d1 = metrics()
    acc = (a1 - a0) / (d1 - d0) if d1 > d0 else float("nan")
    print("%-12s %10.1f %7.1f%% %s" % (name, statistics.median(rates), acc * 100, " ".join("%.1f" % r for r in rates)), flush=True)
