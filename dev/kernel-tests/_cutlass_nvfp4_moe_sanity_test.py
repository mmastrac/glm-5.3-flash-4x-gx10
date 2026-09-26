#!/usr/bin/env python3
"""T7b: sanity for T7 -- are the CUTLASS NVFP4 MoE outputs real (finite,
non-zero, input-dependent) and do they agree with a torch dequantised
reference? Same synthetic layer as kt7.py."""
import sys

import torch

sys.argv = [sys.argv[0]]
exec(open("/kv/kt7.py").read().split("def routing(")[0])  # reuse weights + moe()

FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)


def deq_fp4(packed: torch.Tensor, scale_fp8: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., K/2], scale [..., K/16] fp8, gscale scalar -> float [..., K]."""
    lo = packed & 0xF
    hi = packed >> 4
    nib = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    mag = FP4_LUT[(nib & 7).long()]
    sign = torch.where((nib & 8) != 0, -1.0, 1.0)
    vals = mag * sign
    sc = scale_fp8.float().repeat_interleave(16, dim=-1)
    return vals * sc * gscale


def quant_deq_act(x: torch.Tensor, gscale_inv: float) -> torch.Tensor:
    """Emulate nvfp4 activation quant: per-16 block fp8 scale, global scale."""
    M, Kd = x.shape
    xb = x.float().view(M, Kd // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True)
    # block scale (fp8) = amax / 6 * global; ModelOpt: sf = amax/6 / input_scale
    sf = (amax / 6.0 * gscale_inv).to(fp8).float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = (xb / (sf / gscale_inv)).clamp(-6, 6)
    # round to e2m1 grid
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    idx = (q.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    qv = grid[idx] * torch.sign(q)
    return (qv * (sf / gscale_inv)).view(M, Kd)


gen = torch.Generator(device=dev).manual_seed(3)
M = 8
x = torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16) * 1.5
ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
w = torch.rand(M, TOPK, device=dev, generator=gen) + 0.1
w = (w / w.sum(-1, keepdim=True)) * 2.5
out = moe(x, ids, w, True, 1)
torch.cuda.synchronize()
print("out finite:", torch.isfinite(out).all().item(), "abs mean:", out.float().abs().mean().item(),
      "abs max:", out.float().abs().max().item(), "zero rows:", int((out.float().abs().amax(1) == 0).sum()))
out_b = moe(x, ids, w, False, 1)
print("bf16-input variant vs prequant: max |d| =", (out.float() - out_b.float()).abs().max().item())
x2 = x.clone(); x2[0] = torch.randn(K, device=dev, generator=gen).to(torch.bfloat16)
out2 = moe(x2, ids, w, True, 1)
print("changing row 0 input changes row 0 output:", (out2[0].float() - out[0].float()).abs().max().item() > 1e-3,
      "and leaves other rows unchanged:", (out2[1:].float() - out[1:].float()).abs().max().item())

# torch reference for a few rows
a1_inv = float(1.0 / a1_gscale[0])  # = w13_input_scale
xd = quant_deq_act(x, a1_inv)
ref = torch.zeros(M, K, device=dev)
for r in range(M):
    for j in range(TOPK):
        e = int(ids[r, j])
        W13 = deq_fp4(w13[e], w13_scale[e], float(w13_scale_2[e]))  # [2N, K] rows: [w1(gate); w3(up)]
        W2 = deq_fp4(w2[e], w2_scale[e], float(w2_scale_2[e]))  # [K, N]
        gate = xd[r] @ W13[:N].T
        up = xd[r] @ W13[N:].T
        gate = gate.clamp(max=10.0)
        up = up.clamp(-10.0, 10.0)
        h = torch.nn.functional.silu(gate) * up
        hd = quant_deq_act(h[None], float(w2_input_scale[e]))[0]
        ref[r] += w[r, j] * (hd @ W2.T)
err = (out.float() - ref).abs().amax(1) / (ref.abs().amax(1) + 1e-6)
print("per-row rel err vs torch reference:", [round(v, 3) for v in err.tolist()])
print("ref abs mean:", ref.abs().mean().item())
