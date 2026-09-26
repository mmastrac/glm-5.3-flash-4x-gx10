#!/usr/bin/env python3
"""T7: FlashInfer CUTLASS NVFP4 fused MoE on GB10 -- row consistency vs M.

Mirrors FlashInferExperts.apply (vllm/model_executor/layers/fused_moe/experts/
flashinfer_cutlass_moe.py) for the nvfp4 path at GLM-5.3-Flash's per-rank
shape (E=288, K=4096, N=512, top-k 8, swiglu_limit 10): pre-quantized
activation via ops.scaled_fp4_quant (swizzled sf), [w3;w1] packed weights,
swizzle_blockscale'd block scales viewed as int32, per-expert global scales,
output= preallocated.

For each M, every batched row is compared with the same row run alone (M=1,
same routing). A row that disagrees is checked against every other row's
solo output to detect row mixing.
"""
import sys

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_type,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w1w3_to_w3w1,
)
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

torch.manual_seed(0)
dev = torch.device("cuda")
E, K, N, TOPK = 288, 4096, 512, 8
fp8 = torch.float8_e4m3fn

# ---- synthetic weights in checkpoint layout, then kernel layout ----
g = torch.Generator(device=dev).manual_seed(7)
w13 = torch.randint(0, 256, (E, 2 * N, K // 2), device=dev, dtype=torch.uint8, generator=g)
w2 = torch.randint(0, 256, (E, K, N // 2), device=dev, dtype=torch.uint8, generator=g)
w13_scale = (torch.rand(E, 2 * N, K // 16, device=dev, generator=g) * 1.5 + 0.5).to(fp8)
w2_scale = (torch.rand(E, K, N // 16, device=dev, generator=g) * 1.5 + 0.5).to(fp8)
w13_scale_2 = torch.full((E,), 1.0 / 96, device=dev)  # weight global scale
w2_scale_2 = torch.full((E,), 1.0 / 96, device=dev)
w13_input_scale = torch.full((E,), 8.0 / (448 * 6), device=dev)  # amax 8 -> ModelOpt convention
w2_input_scale = torch.full((E,), 8.0 / (448 * 6), device=dev)

w13_k, w13_scale_k = reorder_w1w3_to_w3w1(w13.clone(), w13_scale.clone())
w13_scale_k = swizzle_blockscale(w13_scale_k)
w2_scale_k = swizzle_blockscale(w2_scale.clone())
g1_alphas = (w13_scale_2 * w13_input_scale).contiguous()
g2_alphas = (w2_scale_2 * w2_input_scale).contiguous()
a1_gscale = (1.0 / w13_input_scale).contiguous()
a2_gscale = (1.0 / w2_input_scale).contiguous()
swiglu_limit = torch.full((E,), 10.0, device=dev, dtype=torch.float32)
act = activation_to_flashinfer_type(MoEActivation.SILU)
quant_scales = [a1_gscale, w13_scale_k.view(torch.int32), g1_alphas,
                a2_gscale, w2_scale_k.view(torch.int32), g2_alphas]
fc1 = w13_k.view(torch.long)
fc2 = w2.view(torch.long)


def moe(x_bf16, topk_ids, topk_w, prequant=True, tp_size=1, tp_rank=0):
    M = x_bf16.shape[0]
    out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    if prequant:
        xq, xsf = ops.scaled_fp4_quant(x_bf16, a1_gscale[:1].reshape(()).clone(), is_sf_swizzled_layout=True)
        inp, sf = xq, xsf
    else:
        inp, sf = x_bf16, None
    flashinfer_cutlass_fused_moe(
        input=inp,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_w,
        fc1_expert_weights=fc1,
        fc2_expert_weights=fc2,
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=swiglu_limit,
        output=out,
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
        input_sf=sf,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        activation_type=act,
        use_deepseek_fp8_block_scale=False,
        use_mxfp8_act_scaling=False,
        use_w4_group_scaling=False,
    )
    return out


def routing(M, gen):
    ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)])
    w = torch.rand(M, TOPK, device=dev, generator=gen) + 0.1
    w = (w / w.sum(-1, keepdim=True)) * 2.5  # routed_scaling_factor
    return ids.to(torch.int32), w.to(torch.float32)


def check(prequant, tp_size=1, Ms=(1, 2, 4, 8, 16, 32, 64), trials=12, tag=""):
    gen = torch.Generator(device=dev).manual_seed(11)
    total_bad = 0
    for M in Ms:
        worst = 0.0
        bad_rows = 0
        mixes = []
        nondet = 0.0
        for t in range(trials):
            x = torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16) * 1.5
            ids, w = routing(M, gen)
            outB = moe(x, ids, w, prequant, tp_size)
            outB2 = moe(x, ids, w, prequant, tp_size)
            nondet = max(nondet, (outB.float() - outB2.float()).abs().max().item())
            solo = torch.stack([moe(x[r : r + 1], ids[r : r + 1], w[r : r + 1], prequant, tp_size)[0] for r in range(M)])
            torch.cuda.synchronize()
            for r in range(M):
                scale = solo[r].float().abs().max().item() + 1e-6
                err = (outB[r].float() - solo[r].float()).abs().max().item() / scale
                worst = max(worst, err)
                if err > 0.05:
                    bad_rows += 1
                    # does the batched row match some other row's solo output?
                    d = (solo.float() - outB[r].float()[None]).abs().amax(dim=1) / scale
                    j = int(d.argmin())
                    mixes.append((t, r, round(err, 3), j, round(float(d[j]), 3)))
        total_bad += bad_rows
        print(f"  [{tag}] M={M:3d}: worst row err vs solo {worst:.4f}, bad rows {bad_rows}/{M * trials}, "
              f"repeat-call nondeterminism max|d| {nondet:.4g}" + (f", mixes(trial,row,err,best_row,best_err)={mixes[:6]}" if mixes else ""))
    return total_bad


bad = 0
print("== production path: pre-quantized nvfp4 input + swizzled input_sf, tp_size=1 ==")
bad += check(True, 1, tag="prequant")
print("== bf16 input (kernel-internal quant), tp_size=1 ==")
bad += check(False, 1, tag="bf16in")
print("== production tp flags: prequant, tp_size=4 tp_rank=0 ==")
bad += check(True, 4, Ms=(1, 4, 8, 16), trials=8, tag="prequant-tp4")
print("T7:", "PASS" if bad == 0 else f"FAIL ({bad} bad rows)")
sys.exit(0 if bad == 0 else 1)
