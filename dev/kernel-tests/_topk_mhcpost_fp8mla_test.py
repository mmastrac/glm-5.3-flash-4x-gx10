#!/usr/bin/env python3
"""Kernel-level checks for the GLM-5.3-Flash decode path on GB10 (spark-glm53:v4).

T1: sparse-indexer decode top-k (per_row) + expand_pools_and_append_tail on
    short rows: must return exactly the causal set for every row.
T3: mhc_post_tilelang must not write into any of its inputs, and must match
    the torch reference.
T5: FlashInfer SM90 MLA wrapper with an fp8 KV cache (planned as
    float8_e4m3fn, as patches/sm90_fp8_kv_dtype.py does) against a bf16
    reference, through vLLM's own _SM90State and concat_and_cache_mla.
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


@section("T1 per_row topk + expand (short rows)")
def t1():
    from vllm import _custom_ops as ops
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    max_pool_len = 524288 // 4
    kpool = 4
    select_k = 2048 // kpool
    bad = 0
    for garbage in ("neg", "pos", "nan", "inf"):
        for trial in range(3):
            rows = 8
            base = int(torch.randint(20, 3000, (1,)))
            lens_tok = torch.arange(base, base + rows, dtype=torch.int32)
            pool_lens = lens_tok // kpool
            fill = {"neg": -1e4, "pos": 1e4, "nan": float("nan"), "inf": float("inf")}[garbage]
            logits = torch.full((rows, max_pool_len), fill, device=dev, dtype=torch.float32)
            for r in range(rows):
                n = int(pool_lens[r])
                logits[r, :n] = torch.randn(n, device=dev)
            seq_lens = pool_lens.to(torch.int32).to(dev).view(rows, 1)
            out = torch.full((rows, select_k), -1, dtype=torch.int32, device=dev)
            ops.top_k_per_row_decode(
                logits, 1, seq_lens, out, rows, logits.stride(0), logits.stride(1), select_k
            )
            torch.cuda.synchronize()
            # reference
            for r in range(rows):
                n = int(pool_lens[r])
                got = out[r][out[r] >= 0].tolist()
                if n <= select_k:
                    exp = set(range(n))
                else:
                    exp = set(logits[r, :n].topk(select_k).indices.tolist())
                if set(got) != exp or len(got) != len(set(got)):
                    bad += 1
                    print(f"  topk MISMATCH garbage={garbage} row={r} len_tok={int(lens_tok[r])} pools={n}: "
                          f"got {len(got)} ({len(set(got))} uniq), missing={sorted(exp - set(got))[:10]}, "
                          f"extra={sorted(set(got) - exp)[:10]}")
            expanded = kpool_ops.expand_pools_and_append_tail(out.to(torch.int64), lens_tok.to(dev), kpool)
            torch.cuda.synchronize()
            for r in range(rows):
                L = int(lens_tok[r])
                got = expanded[r][expanded[r] >= 0].tolist()
                if L <= 2048:
                    exp = set(range(L))
                    if set(got) != exp or len(got) != L:
                        bad += 1
                        print(f"  expand MISMATCH garbage={garbage} row={r} L={L}: got {len(got)} "
                              f"missing={sorted(exp - set(got))[:10]} extra={sorted(set(got) - exp)[:10]}")
                else:
                    if len(got) != 2048 + (L % kpool) or max(got) >= L or len(set(got)) != len(got):
                        bad += 1
                        print(f"  expand MISMATCH(long) row={r} L={L}: n={len(got)} max={max(got)}")
    # Also drive the module path production uses.
    mod = SparseIndexerTopk("per_row")
    rows = 16
    lens_tok = torch.randint(5, 900, (rows,), dtype=torch.int32)
    pool_lens = (lens_tok // kpool)
    logits = torch.full((rows, max_pool_len), 1e4, device=dev)
    for r in range(rows):
        logits[r, : int(pool_lens[r])] = torch.randn(int(pool_lens[r]), device=dev)
    out = torch.full((rows, select_k), -1, dtype=torch.int32, device=dev)
    mod(logits, pool_lens.to(dev).view(rows, 1), 1, out, select_k, int(lens_tok.max()))
    torch.cuda.synchronize()
    for r in range(rows):
        n = int(pool_lens[r])
        got = out[r][out[r] >= 0].tolist()
        if set(got) != set(range(n)) or len(got) != n:
            bad += 1
            print(f"  module MISMATCH row={r} pools={n}: got {len(got)}")
    print(f"T1 mismatches: {bad}")
    assert bad == 0


@section("T3 mhc_post_tilelang purity + reference")
def t3():
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    T, n, H = 24, 4, 4096
    worst = 0.0
    for pdtype in (torch.float32, torch.bfloat16):
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        residual = torch.randn(T, n, H, device=dev, dtype=torch.bfloat16)
        post = torch.randn(T, n, 1, device=dev, dtype=pdtype)
        comb = torch.randn(T, n, n, device=dev, dtype=pdtype)
        keep = [t.clone() for t in (x, residual, post, comb)]
        try:
            out = torch.ops.vllm.mhc_post_tilelang(x, residual, post, comb)
        except Exception as e:  # noqa: BLE001
            print(f"  post/comb dtype {pdtype}: op rejected ({type(e).__name__}: {str(e)[:120]})")
            continue
        torch.cuda.synchronize()
        for name, a, b in zip(("x", "residual", "post", "comb"), (x, residual, post, comb), keep):
            assert torch.equal(a, b), f"mhc_post_tilelang MUTATED input {name} (post dtype {pdtype})"
        ref = mhc_post_torch(x, residual, post, comb)
        err = (out.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        worst = max(worst, err / max(scale, 1e-6))
        print(f"  post/comb dtype {pdtype}: out {tuple(out.shape)} {out.dtype}, max abs err {err:.4g} (ref max {scale:.4g})")
        assert not out.data_ptr() in {t.data_ptr() for t in (x, residual, post, comb)}
    print(f"T3 worst relative error {worst:.3g}")
    assert worst < 0.05


@section("T5 SM90 fp8 MLA wrapper vs reference")
def t5():
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import _SM90State

    heads = 16  # 64 heads / TP4
    d = 512
    sm_scale = 1.0 / (256 ** 0.5)
    topk_width = 2176  # (2048 + 3) rounded up to 128
    bs = 64
    max_tokens = 64

    def run_case(kv_dtype_name, N, lens, q_mag=1.0, kv_mag=1.0, label=""):
        fp8 = kv_dtype_name == "fp8_e4m3"
        num_blocks = (N + 100) // bs + 2
        kv_c = torch.randn(N, d, device=dev, dtype=torch.bfloat16) * kv_mag
        k_pe = torch.empty(N, 0, device=dev, dtype=torch.bfloat16)
        if fp8:
            kv_cache = torch.zeros(num_blocks, bs, d, device=dev, dtype=torch.uint8)
        else:
            kv_cache = torch.zeros(num_blocks, bs, d, device=dev, dtype=torch.bfloat16)
        slots = (torch.arange(N, device=dev) + 37).to(torch.int64)
        scale = torch.tensor(1.0, device=dev, dtype=torch.float32)
        ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, kv_dtype_name if fp8 else "auto", scale)
        torch.cuda.synchronize()
        if fp8:
            deq = kv_cache.view(torch.float8_e4m3fn).reshape(-1, d).float()
        else:
            deq = kv_cache.reshape(-1, d).float()
        # cache write check
        werr = (deq[slots] - kv_c.float()).abs().max().item()
        R = len(lens)
        q_nope = torch.randn(R, heads, d, device=dev, dtype=torch.bfloat16) * q_mag
        q_pe = torch.empty(R, heads, 0, device=dev, dtype=torch.bfloat16)
        state = _SM90State(dev, heads, torch.float8_e4m3fn if fp8 else torch.bfloat16,
                           max_tokens, topk_width, kv_lora_rank=d, qk_rope_head_dim=0, sm_scale=sm_scale)
        state.kv_indices.fill_(0)
        for r, L in enumerate(lens):
            state.kv_indices[r * topk_width: r * topk_width + L] = slots[:L].to(torch.int32)
        state.plan(R, torch.tensor(lens, dtype=torch.int32))
        flat = (kv_cache.view(torch.float8_e4m3fn) if fp8 else kv_cache).reshape(-1, 1, d)
        ckv, kpe = flat[..., :d], flat[..., d:]
        kw = {"ckv_scale": 1.0, "kpe_scale": 1.0} if fp8 else {}
        out = state.wrapper.run(q_nope, q_pe, ckv, kpe, **kw)
        torch.cuda.synchronize()
        # reference in fp32 on the dequantised cache
        ref = torch.empty(R, heads, d, device=dev, dtype=torch.float32)
        for r, L in enumerate(lens):
            K = deq[slots[:L]]  # [L, d]
            s = (q_nope[r].float() @ K.T) * sm_scale  # [heads, L]
            p = torch.softmax(s, dim=-1)
            ref[r] = p @ K
        err = (out.float() - ref).abs()
        rel = err.max().item() / max(ref.abs().max().item(), 1e-6)
        # per-row error
        row_err = err.flatten(1).max(dim=1).values.tolist()
        print(f"  [{label}] kv={kv_dtype_name} N={N} lens={lens[:3]}..{lens[-1]} write_err={werr:.3g} "
              f"max_abs_err={err.max().item():.4g} rel={rel:.4g} out_shape={tuple(out.shape)} {out.dtype}")
        print(f"     per-row max err: {[round(v, 4) for v in row_err]}")
        return rel

    worst = 0.0
    # 8-row spec-verify step at short context (the count task regime)
    for N, base in ((110, 103), (300, 293), (700, 693)):
        lens = list(range(base, base + 8))
        worst = max(worst, run_case("fp8_e4m3", N, lens, label="short8"))
        run_case("auto", N, lens, label="short8-bf16")
    # single row, mixed lengths, and lengths crossing 2048 (topk + tail)
    N = 2100
    worst = max(worst, run_case("fp8_e4m3", N, [1, 5, 63, 64, 65, 2047, 2048, 2051], label="mixed"))
    # larger magnitudes (fp8 saturation regime)
    worst = max(worst, run_case("fp8_e4m3", 300, list(range(293, 301)), q_mag=4.0, kv_mag=8.0, label="big"))
    print(f"T5 worst relative error (fp8): {worst:.4g}")
    assert worst < 0.1


for fn in (t1, t3, t5):
    fn()

print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"{k}: {v}")
sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)
