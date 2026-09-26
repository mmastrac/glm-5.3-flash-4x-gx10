#!/usr/bin/env python3
"""Offline report over glm53_step_tap output.

    python3 glm53_tap_report.py DIR [--req SUBSTR] [--steps A-B] [--all-pairs]
    python3 glm53_tap_report.py --compare DECODE.pt PREFILL.pt --pos N [--rel 0.05]

--compare diffs the row at absolute position N of two vector dumps (one from
a decode step, one from a prefill replay of the same ids) checkpoint by
checkpoint, in float, in the decode file's execution order, and flags the
first checkpoint whose relative difference exceeds --rel. KDA state rows are
absent from prefill dumps (one state per request), so the first divergence
there is read from kda.qkv/g1/beta (layer input) vs kda.out (kernel output).

DIR holds tap-rank<r>.jsonl for one or more ranks (copy each box's file into
the same directory). Steps are aligned by their step number: every rank runs
the same scheduler output sequence, so step N is the same batch on every rank.

Sections:
  pairs        per step, per pair: first checkpoint where the two rows differ
               (execution order), and the step a pair first split
  events       host/device num_computed mismatch, SM90 planned!=valid,
               KDA state changed between steps, MLA cache row changed
               between steps, sampler argmax!=sampled
  ranks        cross-rank disagreement on fed tokens, input ids, positions,
               slots, and every replicated checkpoint hash
  tokens       per request: the token fed at each step on rank 0 (find the
               step where the wrong token appears, then look up the sections
               above at that step and the one before it)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

SHARDED = ("kda.", "mla.q", "mla.out", "mla.kvidx", "mla.attn_out", "attn_mla.out", "attn_kda.out")


def load(d: str):
    ranks = {}
    for p in sorted(glob.glob(os.path.join(d, "tap-rank*.jsonl"))):
        r = int(os.path.basename(p)[len("tap-rank") : -len(".jsonl")])
        recs = []
        hdr = None
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("header"):
                    hdr = o
                else:
                    recs.append(o)
        ranks[r] = (hdr, {o["step"]: o for o in recs})
    return ranks


def replicated(name: str) -> bool:
    return not any(s in name for s in SHARDED)


def compare(a_path: str, b_path: str, pos: int, rel_thr: float) -> int:
    import torch

    A = torch.load(a_path, map_location="cpu")
    B = torch.load(b_path, map_location="cpu")

    def row_of(blob, name):
        cands = [i for i, r in enumerate(blob["rows"]) if r["pos"] == pos]
        if not cands:
            print(f"{name}: no row at pos {pos}; positions {sorted({r['pos'] for r in blob['rows']})[:12]}...")
            return None
        return cands[-1]

    ia, ib = row_of(A, a_path), row_of(B, b_path)
    if ia is None or ib is None:
        return 1
    print(f"row A {A['rows'][ia]}\nrow B {B['rows'][ib]}")
    first = None
    for name, ta in A["cps"].items():
        tb = B["cps"].get(name)
        if tb is None:
            continue
        va, vb = ta[ia].float().reshape(-1), tb[ib].float().reshape(-1)
        if va.numel() != vb.numel():
            print(f"{name:32s} shape mismatch {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        if ta.dtype in (torch.int32, torch.int64, torch.uint8, torch.int8):
            neq = int((va != vb).sum())
            mark = "  <-- differs" if neq else ""
            print(f"{name:32s} int  {neq} of {va.numel()} entries differ{mark}")
            if neq and first is None:
                first = name
            continue
        d = (va - vb).abs()
        rel = float(d.norm() / (vb.norm() + 1e-12))
        mark = ""
        if rel > rel_thr and first is None:
            first = name
            mark = "  <-- FIRST above threshold"
        print(f"{name:32s} maxabs {float(d.max()):.4g} rel {rel:.4g} |B|max {float(vb.abs().max()):.4g}{mark}")
    print("\nfirst checkpoint above threshold:", first)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("A.pt", "B.pt"))
    ap.add_argument("--pos", type=int)
    ap.add_argument("--rel", type=float, default=0.05)
    ap.add_argument("--req", default="")
    ap.add_argument("--steps", default="")
    ap.add_argument("--all-pairs", action="store_true", help="print every pair every step, not only splits")
    args = ap.parse_args()
    if args.compare:
        if args.pos is None:
            print("--compare needs --pos N (absolute position of the row)")
            return 2
        return compare(args.compare[0], args.compare[1], args.pos, args.rel)
    if not args.dir:
        ap.print_help()
        return 2
    ranks = load(args.dir)
    if not ranks:
        print("no tap-rank*.jsonl in", args.dir)
        return 1
    lo, hi = 0, 1 << 40
    if args.steps:
        a, _, b = args.steps.partition("-")
        lo, hi = int(a or 0), int(b or hi)

    def want(o):
        return lo <= o["step"] <= hi and (not args.req or any(args.req in x["req"] for x in o["rows"]))

    for r, (hdr, _) in sorted(ranks.items()):
        print(f"rank {r}: {json.dumps(hdr) if hdr else 'no header'}")
    r0 = min(ranks)
    steps0 = ranks[r0][1]

    print("\n== pairs ==")
    seen_split = set()
    for s in sorted(steps0):
        o = steps0[s]
        if not want(o):
            continue
        for p in o.get("pairs", []):
            key = (p["a"], p["b"])
            split = p.get("first_diff")
            if split and key not in seen_split:
                seen_split.add(key)
                print(f"step {s} pos {p['pos']} pair {p['a']} / {p['b']}: FIRST SPLIT at {split} "
                      f"({p.get('ndiff')} of {p.get('ncp')} checkpoints differ; {p.get('first_diff_nbytes_ne')} bytes differ there)")
                print("   differing:", ", ".join(p.get("diffs", [])[:40]))
            elif args.all_pairs or (split and args.req):
                print(f"step {s} pos {p['pos']} pair {p['a']} / {p['b']}: {split or 'identical'} ({p.get('ndiff')}/{p.get('ncp')})")
    if not seen_split:
        print("(no pair split in the selected steps)")

    print("\n== events ==")
    n_ev = 0
    for r, (_, steps) in sorted(ranks.items()):
        for s in sorted(steps):
            o = steps[s]
            if not want(o):
                continue
            for k in ("host_dev_mismatch", "plan_mismatch", "state_changed", "kv_changed", "err"):
                v = o.get(k)
                if v:
                    n_ev += 1
                    print(f"rank {r} step {s} {k}: {json.dumps(v)[:600]}")
            for sm in o.get("sample", []):
                if sm.get("sampled") is not None and sm["sampled"] != sm["argmax"]:
                    n_ev += 1
                    print(f"rank {r} step {s} sampled!=argmax {sm}")
    if not n_ev:
        print("(none)")

    print("\n== ranks ==")
    if len(ranks) < 2:
        print("(single rank file; nothing to compare)")
    else:
        n_bad = 0
        for s in sorted(steps0):
            o0 = steps0[s]
            if not want(o0):
                continue
            others = [(r, ranks[r][1].get(s)) for r in sorted(ranks) if r != r0]
            for r, o in others:
                if o is None:
                    print(f"step {s}: rank {r} has no record")
                    n_bad += 1
                    continue
                bad = []
                for f in ("fed", "nc_after", "sampled0"):
                    if o0.get(f) != o.get(f):
                        bad.append(f"{f} r{r0}={o0.get(f)} r{r}={o.get(f)}")
                rows0 = [(x["req"], x["pos"], x["tok"], x["seq_dev"], x["nc_dev"]) for x in o0["rows"]]
                rows1 = [(x["req"], x["pos"], x["tok"], x["seq_dev"], x["nc_dev"]) for x in o["rows"]]
                if rows0 != rows1:
                    bad.append(f"rows r{r0}={rows0} r{r}={rows1}")
                for n, h in o0.get("cps", {}).items():
                    if not replicated(n):
                        continue
                    h1 = o.get("cps", {}).get(n)
                    if h1 is not None and h1 != h:
                        bad.append(f"cp {n}")
                for n, v in o0.get("ints", {}).items():
                    if n.endswith((".mla.slot", ".idx.slot", ".idx.tail_slot", ".kda.state_idx", ".mla.planned", ".mla.valid")):
                        v1 = o.get("ints", {}).get(n)
                        if v1 is not None and v1 != v:
                            bad.append(f"int {n} r{r0}={v} r{r}={v1}")
                if bad:
                    n_bad += 1
                    print(f"step {s} rank {r} vs {r0}: " + "; ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))
        if not n_bad:
            print("(all ranks agree on fed tokens, rows, placement and replicated checkpoints)")

    print("\n== tokens (rank %d) ==" % r0)
    per_req = defaultdict(list)
    for s in sorted(steps0):
        o = steps0[s]
        if not want(o):
            continue
        for req, tok in (o.get("fed") or {}).items():
            if not args.req or args.req in req:
                pos = next((x["pos"] for x in o["rows"] if x["req"] == req), None)
                per_req[req].append((s, pos, tok))
    for req, seq in per_req.items():
        print(f"{req}: " + " ".join(f"{s}:{p}>{t}" for s, p, t in seq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
