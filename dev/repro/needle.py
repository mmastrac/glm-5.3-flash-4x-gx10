#!/usr/bin/env python3
"""Long-context retrieval: can the model find one fact buried in a long prompt?

Standard library only.

    python needle.py --url http://127.0.0.1:8000 --model glm53

Places a distinctive sentence at several depths in filler of several lengths and
asks for it back. Reports one line per (length, depth) with the exact-match
verdict, so a partial credit answer still reads as a miss.
"""
import argparse
import json
import time
import urllib.request

NEEDLE = "The access code for the Vermilion vault is 7741-KESTREL."
ANSWER = "7741-KESTREL"
QUESTION = ("\n\nQuestion: what is the access code for the Vermilion vault? "
            "Reply with the code alone.")

# Numbered so the filler is not a degenerate repeat of one block.
FILLER = ("Maintenance note {n}: the west pump was serviced and the intake "
          "filter replaced; readings held within tolerance for the shift.\n")


def post(url, payload, timeout=3600):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def count(base, model, text):
    return post(base + "/tokenize", {"model": model, "prompt": text}, timeout=600)["count"]


def build(base, model, target, depth):
    """Filler of ~target tokens with the needle at `depth` through it."""
    per = count(base, model, FILLER.format(n=1))
    lines = max(1, target // max(per, 1))
    body = [FILLER.format(n=i) for i in range(lines)]
    at = min(len(body), max(0, int(len(body) * depth)))
    body.insert(at, NEEDLE + "\n")
    text = "".join(body)
    return text + QUESTION


def ask(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    started = time.time()
    result = post(base + "/v1/chat/completions", body)
    message = result["choices"][0]["message"]
    text = (message.get("content") or "") + " " + (message.get("reasoning") or "")
    return text.strip(), result.get("usage", {}), time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[32000, 128000, 256000])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    base = a.url.rstrip("/")

    passed = total = 0
    for length in a.lengths:
        for depth in a.depths:
            prompt = build(base, a.model, length, depth)
            try:
                text, usage, secs = ask(base, a.model, prompt, a.max_tokens)
            except Exception as exc:
                print(f"  {length:>7} tok  depth {depth:>4.0%}  ERROR {exc!r}", flush=True)
                total += 1
                continue
            hit = ANSWER in text
            passed += hit
            total += 1
            finish = usage.get("completion_tokens")
            capped = finish is not None and finish >= a.max_tokens
            note = " (hit max_tokens)" if capped and not hit else ""
            print(f"  {usage.get('prompt_tokens', '?'):>7} tok  depth {depth:>4.0%}  "
                  f"{'FOUND' if hit else 'MISS ':5}  {secs:6.1f}s{note}  {text[:60]!r}",
                  flush=True)
    print(f"\n  retrieval: {passed}/{total}")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
