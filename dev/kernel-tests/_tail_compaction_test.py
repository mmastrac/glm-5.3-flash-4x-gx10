#!/usr/bin/env python3
"""T8: does the SM90 path see the tail tokens?

Builds indexer rows exactly as the GLM kpool decode path leaves them in
topk_indices_buffer (width 2176): pool tokens at columns [0, 4*pool_len),
-1 up to column 2048, the 0-3 tail tokens at columns 2048.., -1 after.
Runs triton_convert_req_index_to_global_index(return_valid_counts=True) as
FlashInferMLASparseSM90Impl.forward_mqa does and checks that the first `ctx`
compacted entries (what _SM90State plans for) contain every causal slot,
in particular the tail. Also runs it inside a CUDA graph.
"""
import sys

import torch

from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

dev = torch.device("cuda")
WIDTH = 2176
TOPK = 2048
KPOOL = 4
BLOCK = 2304


def build_rows(positions, block_table_row):
    R = len(positions)
    rows = torch.full((R, WIDTH), -1, dtype=torch.int32, device=dev)
    for r, pos in enumerate(positions):
        seq = pos + 1
        pool_len = seq // KPOOL
        n_hist = min(pool_len, TOPK // KPOOL) * KPOOL
        # pools are selected in an arbitrary order by top-k; use a permutation
        pools = torch.randperm(pool_len, device=dev)[: TOPK // KPOOL]
        toks = (pools[:, None] * KPOOL + torch.arange(KPOOL, device=dev)[None, :]).reshape(-1)
        rows[r, : toks.numel()] = toks.to(torch.int32)
        tail = torch.arange(pool_len * KPOOL, seq, device=dev, dtype=torch.int32)
        rows[r, TOPK : TOPK + tail.numel()] = tail
    return rows


bad = 0
for positions in ([99, 100, 101, 102, 103, 104, 105, 106], [600, 601, 602, 603], [2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054], [5000, 5001, 5002, 5003]):
    R = len(positions)
    req_id = torch.zeros(R, dtype=torch.int32, device=dev)
    block_table = torch.tensor([[7, 3, 11] + [0] * 5], dtype=torch.int32, device=dev)
    rows = build_rows(positions, block_table[0])
    for graph in (False, True):
        out = torch.empty_like(rows)
        counts = torch.empty(R, dtype=torch.int32, device=dev)
        def run():
            triton_convert_req_index_to_global_index(
                req_id, block_table, rows, BLOCK_SIZE=BLOCK, NUM_TOPK_TOKENS=WIDTH,
                return_valid_counts=True, out=out, valid_counts_out=counts)
        if graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(); run()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            rows2 = build_rows(positions, block_table[0]); rows.copy_(rows2)
            g.replay()
        else:
            run()
        torch.cuda.synchronize()
        for r, pos in enumerate(positions):
            ctx = pos + 1
            lens = ctx if ctx <= TOPK else TOPK + ctx % KPOOL
            got = out[r, :lens].tolist()
            # expected slots: block_table[tok // BLOCK]*BLOCK + tok % BLOCK for all causal toks
            toks = torch.arange(ctx, device=dev)
            if ctx > TOPK:
                exp_toks = set((toks[-(ctx % KPOOL):] if ctx % KPOOL else toks[:0]).tolist())
                need_tail_only = True
            else:
                exp_toks = set(toks.tolist()); need_tail_only = False
            exp_slots = {int(block_table[0, t // BLOCK]) * BLOCK + t % BLOCK for t in exp_toks}
            gotset = set(got)
            missing = exp_slots - gotset
            tail_slots = {int(block_table[0, t // BLOCK]) * BLOCK + t % BLOCK for t in range((ctx // KPOOL) * KPOOL, ctx)}
            tail_missing = tail_slots - gotset
            neg = sum(1 for v in got if v < 0)
            ok = (not tail_missing) and (need_tail_only or not missing) and int(counts[r]) == lens and neg == 0
            if not ok:
                bad += 1
            print(f"  graph={graph} pos={pos} ctx={ctx} planned_lens={lens} valid_count={int(counts[r])} "
                  f"neg_in_prefix={neg} tail_missing={sorted(tail_missing)} missing={len(missing)} -> {'ok' if ok else 'BAD'}")
print("T8:", "PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(0 if bad == 0 else 1)
