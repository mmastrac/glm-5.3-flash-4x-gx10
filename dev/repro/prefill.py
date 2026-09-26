"""Cold-cache prefill throughput: unique filler each time so nothing is cached."""
import json, sys, time, urllib.request, uuid
BASE, MODEL = sys.argv[1], sys.argv[2]
def once(target):
    nonce = uuid.uuid4().hex
    unit = f"Reading {nonce} note {{n}}: the pump was serviced and the filter replaced.\n"
    text = "".join(unit.format(n=i) for i in range(target // 16))
    body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\nReply with OK."}],
            "max_tokens": 4, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
    n = (usage or {}).get("prompt_tokens", 0)
    if ttft:
        print(f"  prompt={n:>7} ttft={ttft:6.1f}s  prefill={n/ttft:8.0f} tok/s", flush=True)
    else:
        print(f"  prompt={n:>7} no first token seen (usage={usage})", flush=True)
for t in ([int(a) for a in sys.argv[3:]] or [8000, 42000, 128000]):
    once(t)
