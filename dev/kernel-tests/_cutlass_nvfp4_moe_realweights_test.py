#!/usr/bin/env python3
"""T10: FlashInfer CUTLASS NVFP4 fused MoE VALUES on real GLM-5.3-Flash experts.

Loads 16 experts of layer 5 (extracted read-only from a checkpoint into
/kv/moe-l5-<name>.safetensors), builds the kernel-format weights exactly as
vLLM's FLASHINFER_CUTLASS path does (reorder [w1;w3]->[w3;w1], swizzled block
scales, g1_alphas = weight_scale_2 * input_scale, a1_gscale = 1/input_scale),
routes random tokens among those experts, and compares the kernel against:

  ref_a4 : dequantised weights (vLLM's dequantize_to_dtype, linear scales),
           activations quantised by vLLM's own scaled_fp4_quant with the
           checkpoint's global input scales and dequantised -- what the
           kernel is supposed to compute (w4a4)
  ref_a16: dequantised weights, bf16 activations -- the w4a16 quantity
           (what the marlin backend computes)

Both checkpoints are supported: ModelOpt naming (weight/weight_scale/
weight_scale_2/input_scale) and compressed-tensors naming (weight_packed/
weight_scale/weight_global_scale/input_global_scale, reciprocal convention).
usage: python3 this.py [nvidia|rh]
"""
import sys

import torch
from safetensors.torch import load_file

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_type,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w1w3_to_w3w1,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

name = sys.argv[1] if len(sys.argv) > 1 else "nvidia"
dev = torch.device("cuda")
torch.manual_seed(0)
E_REAL, E, K, N, TOPK = 16, 288, 4096, 2048, 8
LIMIT = 10.0
fp8 = torch.float8_e4m3fn
sd = load_file(f"/kv/moe-l5-{name}.safetensors", device="cuda")
pre = next(k for k in sd if ".experts.0." in k).split(".experts.0.")[0] + ".experts."


def get(e, proj, kind):
    """kind in weight, scale, gscale (weight_scale_2 convention), iscale (input_scale convention)."""
    base = f"{pre}{e}.{proj}."
    if kind == "weight":
        return sd.get(base + "weight", sd.get(base + "weight_packed"))
    if kind == "scale":
        return sd[base + "weight_scale"]
    if kind == "gscale":
        return sd[base + "weight_scale_2"].reshape(()) if base + "weight_scale_2" in sd else 1.0 / sd[base + "weight_global_scale"].reshape(())
    if kind == "iscale":
        return sd[base + "input_scale"].reshape(()) if base + "input_scale" in sd else 1.0 / sd[base + "input_global_scale"].reshape(())
    raise KeyError(kind)


# checkpoint-layout tensors, padded to E experts (zeros beyond the real ones)
w13 = torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8, device=dev)
w13_scale = torch.ones(E, 2 * N, K // 16, device=dev).to(fp8)
w2 = torch.zeros(E, K, N // 2, dtype=torch.uint8, device=dev)
w2_scale = torch.ones(E, K, N // 16, device=dev).to(fp8)
w13_gs = torch.ones(E, device=dev)
w2_gs = torch.ones(E, device=dev)
a13 = torch.zeros(E_REAL, device=dev)
a2 = torch.zeros(E_REAL, device=dev)
for e in range(E_REAL):
    w13[e, :N] = get(e, "gate_proj", "weight"); w13[e, N:] = get(e, "up_proj", "weight")
    w13_scale[e, :N] = get(e, "gate_proj", "scale"); w13_scale[e, N:] = get(e, "up_proj", "scale")
    g1, g3 = float(get(e, "gate_proj", "gscale")), float(get(e, "up_proj", "gscale"))
    assert abs(g1 - g3) <= 1e-9 * max(abs(g1), 1e-30), f"gate/up weight_scale_2 differ for expert {e}: {g1} vs {g3}"
    w13_gs[e] = g1
    w2[e] = get(e, "down_proj", "weight"); w2_scale[e] = get(e, "down_proj", "scale"); w2_gs[e] = float(get(e, "down_proj", "gscale"))
    a13[e] = max(float(get(e, "gate_proj", "iscale")), float(get(e, "up_proj", "iscale")))
    a2[e] = float(get(e, "down_proj", "iscale"))
print(f"[{name}] input_scale gate/up per expert: min {a13.min():.6g} max {a13.max():.6g}; down: min {a2.min():.6g} max {a2.max():.6g}")
print(f"[{name}] weight_scale_2 w13: {w13_gs[:E_REAL].min():.4g}..{w13_gs[:E_REAL].max():.4g}; w2: {w2_gs[:E_REAL].min():.4g}..{w2_gs[:E_REAL].max():.4g}")
# FLASHINFER_CUTLASS uses one global activation scale = max over experts (prepare_nvfp4_moe_layer_for_fi_or_cutlass)
a13_g = a13.max().item(); a2_g = a2.max().item()
w13_input_scale = torch.full((E,), a13_g, device=dev); w2_input_scale = torch.full((E,), a2_g, device=dev)

# kernel format (prepare_nvfp4_moe_layer_for_fi_or_cutlass + FlashInferExperts.process_weights_after_loading)
w13_k, w13_scale_k = reorder_w1w3_to_w3w1(w13.clone(), w13_scale.clone())
w13_scale_k = swizzle_blockscale(w13_scale_k); w2_scale_k = swizzle_blockscale(w2_scale.clone())
g1_alphas = (w13_gs * w13_input_scale).contiguous(); g2_alphas = (w2_gs * w2_input_scale).contiguous()
a1_gscale = (1.0 / w13_input_scale).contiguous(); a2_gscale = (1.0 / w2_input_scale).contiguous()
quant_scales = [a1_gscale, w13_scale_k.view(torch.int32), g1_alphas, a2_gscale, w2_scale_k.view(torch.int32), g2_alphas]
swiglu_limit = torch.full((E,), LIMIT, device=dev)
act = activation_to_flashinfer_type(MoEActivation.SILU)


def kernel(x, ids, w):
    M = x.shape[0]
    out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    xq, xsf = ops.scaled_fp4_quant(x, a1_gscale[:1].reshape(()).clone(), is_sf_swizzled_layout=True)
    flashinfer_cutlass_fused_moe(
        input=xq, token_selected_experts=ids.to(torch.int), token_final_scales=w,
        fc1_expert_weights=w13_k.view(torch.long), fc2_expert_weights=w2.view(torch.long),
        fc1_expert_biases=None, fc2_expert_biases=None, swiglu_alpha=None, swiglu_beta=None,
        swiglu_limit=swiglu_limit, output=out, output_dtype=torch.bfloat16, quant_scales=quant_scales,
        input_sf=xsf, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, activation_type=act,
        use_deepseek_fp8_block_scale=False, use_mxfp8_act_scaling=False, use_w4_group_scaling=False)
    return out, xq, xsf


# dequantised weights for the real experts
W13 = dequantize_to_dtype(w13[:E_REAL], w13_scale[:E_REAL], w13_gs[:E_REAL], torch.float32, 16, swizzle=False)  # [E, 2N, K]
W2 = dequantize_to_dtype(w2[:E_REAL], w2_scale[:E_REAL], w2_gs[:E_REAL], torch.float32, 16, swizzle=False)  # [E, K, N]


def act_q(h, gscale_inv):
    """quantise like the kernel's a2 path: scaled_fp4_quant with global scale 1/input_scale, then dequantise."""
    q, sf = ops.scaled_fp4_quant(h.to(torch.bfloat16), torch.tensor(1.0 / gscale_inv, device=dev), is_sf_swizzled_layout=True)
    return dequantize_to_dtype(q, sf, torch.tensor(gscale_inv, device=dev), torch.float32, 16, swizzle=True)


def reference(x, ids, w, xq, xsf, quant_act):
    M = x.shape[0]
    if quant_act:
        xd = dequantize_to_dtype(xq, xsf, torch.tensor(a13_g, device=dev), torch.float32, 16, swizzle=True)
    else:
        xd = x.float()
    out = torch.zeros(M, K, device=dev)
    for r in range(M):
        for j in range(TOPK):
            e = int(ids[r, j])
            gate = xd[r] @ W13[e, :N].T
            up = xd[r] @ W13[e, N:].T
            h = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
            hd = act_q(h[None], a2_g)[0] if quant_act else h
            out[r] += w[r, j] * (hd @ W2[e].T)
    return out


gen = torch.Generator(device=dev).manual_seed(1)
worst_a4 = worst_a16 = 0.0
for M, amp in ((1, 0.5), (4, 0.5), (8, 0.5), (8, 1.0), (16, 0.5), (1, 2.0)):
    x = (torch.randn(M, K, device=dev, generator=gen) * amp).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E_REAL, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOPK, device=dev, generator=gen) + 0.1
    w = (w / w.sum(-1, keepdim=True)) * 2.5
    out, xq, xsf = kernel(x, ids, w)
    torch.cuda.synchronize()
    ref4 = reference(x, ids, w, xq, xsf, True)
    ref16 = reference(x, ids, w, xq, xsf, False)
    e4 = (out.float() - ref4).abs().amax(1) / (ref4.abs().amax(1) + 1e-6)
    e16 = (out.float() - ref16).abs().amax(1) / (ref16.abs().amax(1) + 1e-6)
    q_loss = (ref4 - ref16).abs().amax(1) / (ref16.abs().amax(1) + 1e-6)
    cos = torch.nn.functional.cosine_similarity(out.float(), ref16, dim=1)
    print(f"[{name}] M={M} amp={amp}: kernel vs ref_a4 rel err per row {[round(v, 3) for v in e4.tolist()]}; "
          f"kernel vs ref_a16 {[round(v, 3) for v in e16.tolist()]}; a4-vs-a16 quant loss {[round(v, 3) for v in q_loss.tolist()]}; "
          f"cos(kernel, ref_a16) min {cos.min():.4f}; |out| mean {out.float().abs().mean():.3g} nan={torch.isnan(out).any().item()}")
    worst_a4 = max(worst_a4, e4.max().item()); worst_a16 = max(worst_a16, e16.max().item())
print(f"[{name}] T10 worst: kernel vs w4a4 reference {worst_a4:.3f}, kernel vs w4a16 reference {worst_a16:.3f}")
print("T10:", "PASS" if worst_a4 < 0.1 else "FAIL")
