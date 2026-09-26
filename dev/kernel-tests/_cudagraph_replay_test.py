#!/usr/bin/env python3
"""T6: FULL-cudagraph replay checks for the two decode kernels whose plan is
built host-side: the SM90 FlashInfer MLA wrapper (_SM90State) and the
DeepGEMM paged MQA logits + per_row top-k of the sparse indexer.

vLLM captures decode graphs with a dummy batch whose seq_len equals the query
length (InputBatch.make_dummy: seq_lens = ceil(num_tokens/num_reqs), positions
zero), then replans every step and replays. This mirrors that: capture with
8 rows of lens 1..8, replay with real lens, compare against a reference.
"""
import sys
import traceback

import torch

torch.manual_seed(0)
dev = torch.device("cuda")
results = {}


def section(name):
    def deco(fn):
        def run():
            print(f"\n===== {name} =====", flush=True)
            try:
                fn()
                results[name] = "PASS"
            except Exception as e:  # noqa: BLE001
                results[name] = f"FAIL: {e}"
                traceback.print_exc()
        return run
    return deco


@section("T6a SM90 MLA wrapper: capture at dummy lens, replay at real lens")
def t6a():
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import _SM90State

    heads, d = 16, 512
    sm_scale = 1.0 / 16.0
    topk_width = 2176
    max_tokens = 256
    R = 8
    N = 2300
    bs = 64
    num_blocks = N // bs + 3
    kv_c = torch.randn(N, d, device=dev, dtype=torch.bfloat16)
    k_pe = torch.empty(N, 0, device=dev, dtype=torch.bfloat16)
    kv_cache = torch.zeros(num_blocks, bs, d, device=dev, dtype=torch.uint8)
    slots = (torch.arange(N, device=dev) + 37).to(torch.int64)
    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_e4m3", torch.tensor(1.0, device=dev))
    deq = kv_cache.view(torch.float8_e4m3fn).reshape(-1, d).float()
    flat = kv_cache.view(torch.float8_e4m3fn).reshape(-1, 1, d)
    ckv, kpe = flat[..., :d], flat[..., d:]

    state = _SM90State(dev, heads, torch.float8_e4m3fn, max_tokens, topk_width,
                       kv_lora_rank=d, qk_rope_head_dim=0, sm_scale=sm_scale)
    # static input buffers (what the captured graph reads)
    q_nope = torch.randn(R, heads, d, device=dev, dtype=torch.bfloat16)
    q_pe = torch.empty(R, heads, 0, device=dev, dtype=torch.bfloat16)
    out_buf = torch.empty(R, heads, d, device=dev, dtype=torch.bfloat16)

    def set_rows(lens):
        state.kv_indices.fill_(0)
        for r, L in enumerate(lens):
            state.kv_indices[r * topk_width: r * topk_width + L] = slots[:L].to(torch.int32)
        state.plan(R, torch.tensor(lens, dtype=torch.int32))

    def reference(lens):
        ref = torch.empty(R, heads, d, device=dev)
        for r, L in enumerate(lens):
            K = deq[slots[:L]]
            s = (q_nope[r].float() @ K.T) * sm_scale
            ref[r] = torch.softmax(s, dim=-1) @ K
        return ref

    def run_once():
        out = state.wrapper.run(q_nope, q_pe, ckv, kpe, ckv_scale=1.0, kpe_scale=1.0)
        out_buf.copy_(out)

    # --- capture with the dummy lens vLLM uses ---
    cap_lens = list(range(1, R + 1))
    set_rows(cap_lens)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run_once()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()
    torch.cuda.synchronize()

    worst = 0.0
    for lens in (
        cap_lens,
        list(range(100, 108)),
        list(range(693, 701)),
        list(range(400, 408)),
        [1, 5, 63, 64, 65, 2047, 2048, 2051],
        list(range(2044, 2052)),
    ):
        q_nope.copy_(torch.randn_like(q_nope))
        set_rows(lens)
        graph.replay()
        torch.cuda.synchronize()
        ref = reference(lens)
        err = (out_buf.float() - ref).abs()
        rel = err.max().item() / max(ref.abs().max().item(), 1e-6)
        row_err = [round(v, 4) for v in err.flatten(1).max(dim=1).values.tolist()]
        nan = torch.isnan(out_buf).any().item()
        print(f"  replay lens={lens[0]}..{lens[-1]}: rel err {rel:.4g} nan={nan} per-row {row_err}")
        worst = max(worst, rel)
        assert not nan
    # eager run at the same lens for comparison
    set_rows(list(range(693, 701)))
    run_once()
    torch.cuda.synchronize()
    ref = reference(list(range(693, 701)))
    print(f"  eager lens=693..700: rel err {((out_buf.float() - ref).abs().max() / ref.abs().max()).item():.4g}")
    print(f"T6a worst replay rel err {worst:.4g}")
    assert worst < 0.1


@section("T6b indexer paged MQA logits + per_row topk: capture short, replay long")
def t6b():
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    from vllm import _custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    # Indexer cache: [num_blocks, pool_block_size, 1, head_dim+4] uint8 (fp8 + fp32 scale)
    head_dim = 128
    kpool = 4
    index_heads = 32  # index_n_heads
    pool_bs = 64  # production indexer kernel page: 64 pools = 256 tokens
    num_blocks = 80
    max_pool_len = 524288 // kpool
    fp8 = torch.float8_e4m3fn
    k_pool = torch.randn(num_blocks * pool_bs, head_dim, device=dev).clamp(-4, 4)
    kv_cache = torch.zeros(num_blocks, pool_bs, 1, head_dim + 4, device=dev, dtype=torch.uint8)
    kv_cache[..., :head_dim] = k_pool.to(fp8).view(torch.uint8).reshape(num_blocks, pool_bs, 1, head_dim)
    kv_cache[..., head_dim:] = torch.ones(num_blocks * pool_bs, 1, 1, device=dev).view(torch.uint8).reshape(num_blocks, pool_bs, 1, 4)
    num_sms = num_compute_units(dev.index if dev.index is not None else 0)

    R = 8
    # per-token expanded block table (flattened path): every row -> request 0's blocks [1..num_blocks-1]
    block_table = torch.zeros(R, num_blocks, device=dev, dtype=torch.int32)
    block_table[:, : num_blocks - 1] = torch.arange(1, num_blocks, device=dev, dtype=torch.int32)
    q = torch.randn(R, 1, index_heads, head_dim, device=dev).to(fp8)
    weights = torch.rand(R, index_heads, device=dev, dtype=torch.float32)
    seq_lens = torch.zeros(R, 1, device=dev, dtype=torch.int32)  # pool-granular, refreshed
    sched_buf = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=dev)
    select_k = 2048 // kpool
    pool_topk = torch.full((R, select_k), -1, dtype=torch.int32, device=dev)
    logits_holder = {}

    def set_lens(pool_lens):
        seq_lens[:, 0] = torch.tensor(pool_lens, device=dev, dtype=torch.int32)
        meta = get_paged_mqa_logits_metadata(seq_lens, pool_bs, num_sms)
        sched = sched_buf[: meta.shape[0]]
        sched[:] = meta
        return sched

    def run_once(sched):
        logits = fp8_fp4_paged_mqa_logits((q, None), kv_cache, weights, seq_lens, block_table,
                                          sched, max_model_len=max_pool_len, clean_logits=False)
        logits_holder["l"] = logits
        pool_topk.fill_(-1)
        ops.top_k_per_row_decode(logits, 1, seq_lens, pool_topk, R, logits.stride(0), logits.stride(1), select_k)

    def reference(pool_lens):
        # logits[r, p] = sum_h w[r,h] * relu(q[r,h] . k[p]) (DeepSeek indexer semantics)
        out = []
        for r, L in enumerate(pool_lens):
            blocks = block_table[r, : (L + pool_bs - 1) // pool_bs]
            K = k_pool.view(num_blocks, pool_bs, head_dim)[blocks.long()].reshape(-1, head_dim)[:L]
            Kq = K.to(fp8).float()
            s = torch.relu(q[r, 0].float() @ Kq.T)  # [heads, L]
            out.append((weights[r][:, None] * s).sum(0))
        return out

    cap_lens = [max(1, (i + 1) // kpool) for i in range(1, R + 1)]  # dummy: seq_len = query_len
    sched = set_lens(cap_lens)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run_once(sched)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once(sched)
    torch.cuda.synchronize()

    bad = 0
    for tok_lens in (list(range(100, 108)), list(range(693, 701)), list(range(1500, 1508)), list(range(2600, 2608)), list(range(4600, 4608))):
        pool_lens = [t // kpool for t in tok_lens]
        # the schedule buffer is refreshed in place; its slot count must be stable
        meta = get_paged_mqa_logits_metadata(seq_lens.new_tensor([[p] for p in pool_lens]), pool_bs, num_sms)
        assert meta.shape[0] == sched.shape[0], f"schedule slot count changed {meta.shape} vs {sched.shape}"
        set_lens(pool_lens)
        graph.replay()
        torch.cuda.synchronize()
        logits = logits_holder["l"]
        refs = reference(pool_lens)
        for r, L in enumerate(pool_lens):
            got = logits[r, :L].float()
            ref = refs[r]
            nan = torch.isnan(got).any().item()
            err = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            sel = pool_topk[r][pool_topk[r] >= 0].tolist()
            if L <= select_k:
                ok_sel = set(sel) == set(range(L)) and len(sel) == L
            else:
                ok_sel = set(sel) == set(ref.topk(select_k).indices.tolist())
            if nan or err > 0.05 or not ok_sel:
                bad += 1
                miss = sorted(set(range(min(L, select_k))) - set(sel))[:8] if L <= select_k else []
                print(f"  BAD replay tok_len={tok_lens[r]} pools={L}: nan={nan} rel_err={err:.3g} "
                      f"sel_ok={ok_sel} nsel={len(sel)} missing_pools={miss}")
        print(f"  replay tok_lens {tok_lens[0]}..{tok_lens[-1]}: checked")
    # eager comparison
    pool_lens = [t // kpool for t in range(693, 701)]
    sched = set_lens(pool_lens)
    run_once(sched)
    torch.cuda.synchronize()
    refs = reference(pool_lens)
    e = max(((logits_holder["l"][r, :L].float() - refs[r]).abs().max() / refs[r].abs().max()).item() for r, L in enumerate(pool_lens))
    print(f"  eager 693..700 rel err {e:.3g}")
    print(f"T6b bad rows: {bad}")
    assert bad == 0


for fn in (t6a, t6b):
    fn()
print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"{k}: {v}")
sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)
