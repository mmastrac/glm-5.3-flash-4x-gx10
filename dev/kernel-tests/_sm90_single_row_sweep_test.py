#!/usr/bin/env python3
"""T16: the SM90 sparse-MLA wrapper at ONE query row, every kv_len 1..MAXL.

T5 only ran the wrapper at 8 query rows. Every decode step is a single row,
and FlashInfer's MLA plan splits a lone row's KV range across CTAs and merges
partial results, so the single-row case has its own code path. This sweeps
kv_len 1..MAXL at R=1 (and R=2 pairs, R=93 prefill-like control) for H=16
(TP=4) and H=32 (TP=2), fp8 and bf16 caches, with the plan sized both like the
old tests (64 rows) and like production (max_num_batched_tokens rows: the
padded rows keep the full-width lens), against an fp32 torch reference on the
dequantised cache. Flags any row above REL_THR.

    python3 _sm90_single_row_sweep_test.py [MAXL] [max_tokens_prod]
"""
import sys

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import _SM90State

dev = torch.device("cuda")
torch.manual_seed(0)
MAXL = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PROD_MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
d = 512
sm_scale = 1.0 / (256**0.5)
topk_width = 2176
bs = 64
REL_THR = 2e-2


def make_cache(N, fp8):
    num_blocks = (N + 200) // bs + 2
    kv_c = torch.randn(N, d, device=dev, dtype=torch.bfloat16)
    k_pe = torch.empty(N, 0, device=dev, dtype=torch.bfloat16)
    kv_cache = torch.zeros(num_blocks, bs, d, device=dev, dtype=torch.uint8 if fp8 else torch.bfloat16)
    slots = (torch.arange(N, device=dev) + 37).to(torch.int64)
    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_e4m3" if fp8 else "auto", torch.tensor(1.0, device=dev))
    deq = (kv_cache.view(torch.float8_e4m3fn) if fp8 else kv_cache).reshape(-1, d).float()
    return kv_cache, slots, deq


def run(state, kv_cache, slots, deq, lens, heads, fp8, q_nope=None):
    R = len(lens)
    if q_nope is None:
        q_nope = torch.randn(R, heads, d, device=dev, dtype=torch.bfloat16)
    q_pe = torch.empty(R, heads, 0, device=dev, dtype=torch.bfloat16)
    state.kv_indices.fill_(0)
    for r, L in enumerate(lens):
        state.kv_indices[r * topk_width : r * topk_width + L] = slots[:L].to(torch.int32)
    state.plan(R, torch.tensor(lens, dtype=torch.int32))
    flat = (kv_cache.view(torch.float8_e4m3fn) if fp8 else kv_cache).reshape(-1, 1, d)
    kw = {"ckv_scale": 1.0, "kpe_scale": 1.0} if fp8 else {}
    out = state.wrapper.run(q_nope, q_pe, flat[..., :d], flat[..., d:], **kw)
    torch.cuda.synchronize()
    rels = []
    for r, L in enumerate(lens):
        K = deq[slots[:L]]
        p = torch.softmax((q_nope[r].float() @ K.T) * sm_scale, dim=-1)
        ref = p @ K
        o = out[r].float()
        rels.append(float((o - ref).abs().max() / (ref.abs().max() + 1e-6)))
    return rels


worst_all = 0.0
bad_total = 0
for heads in (16, 32):
    for fp8 in (True, False):
        kv_cache, slots, deq = make_cache(MAXL + 100, fp8)
        for max_tokens in (64, PROD_MAX_TOKENS):
            state = _SM90State(dev, heads, torch.float8_e4m3fn if fp8 else torch.bfloat16, max_tokens, topk_width,
                               kv_lora_rank=d, qk_rope_head_dim=0, sm_scale=sm_scale)
            bad = []
            worst = 0.0
            for L in range(1, MAXL + 1):
                r = run(state, kv_cache, slots, deq, [L], heads, fp8)[0]
                worst = max(worst, r)
                if r > REL_THR:
                    bad.append((L, round(r, 4)))
            # pairs (S2) and a prefill-like 93-row batch
            pair_worst = 0.0
            for L in (93, 176, 274, 302):
                pair_worst = max(pair_worst, max(run(state, kv_cache, slots, deq, [L, L], heads, fp8)))
            pre_rows = min(93, max_tokens)
            pre = max(run(state, kv_cache, slots, deq, list(range(1, pre_rows + 1)), heads, fp8))
            tag = f"H={heads} kv={'fp8' if fp8 else 'bf16'} max_tokens={max_tokens}"
            print(f"{tag:36s} R=1 worst rel {worst:.4g} over L=1..{MAXL}, {len(bad)} rows > {REL_THR}: {bad[:12]}{' ...' if len(bad) > 12 else ''}")
            print(f"{'':36s} R=2 pairs worst {pair_worst:.4g}; R={pre_rows} prefill-like worst {pre:.4g}")
            worst_all = max(worst_all, worst, pair_worst, pre)
            bad_total += len(bad)
print(f"\nT16 worst rel {worst_all:.4g}; {bad_total} single-row cases above {REL_THR}")
print("T16: PASS" if bad_total == 0 and worst_all < REL_THR else "T16: FAIL")
