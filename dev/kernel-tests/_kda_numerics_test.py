#!/usr/bin/env python3
"""T9: GLM-5.3-Flash KDA numerics against an fp32 torch reference.

Covers the three kernels the nightly uses for the recurrent state (per rank:
16 heads x 128, safe gate, lower_bound -5): FlashKDA chunked prefill (the
nightly's new default on SM12x), the Triton chunk_kda_with_fused_gate prefill
(what the old image ran), and fused_recurrent_kda plain decode, with the
decode state seeded from the prefill's final state. The reference replicates
the recurrent kernel's math: l2norm(q,k), q*=K^-0.5, g = lb/(1+exp(-exp(A)(g1+b))),
h*=exp(g), v-=h@k, v*=sigmoid(beta), h+=v k^T, o=h@q.
"""
import sys

import torch

import vllm._flashkda_C  # noqa: F401
from vllm.models.glm5next.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)

dev = torch.device("cuda")
H, K, V = 16, 128, 128
LB = -5.0
torch.manual_seed(5)


def ref_run(q, k, v, g1, beta, a_log, dt_bias, h0):
    """q,k,v,g1: [T,H,K] bf16; beta [T,H]; h0 [H,V,K] fp32 -> (o [T,H,V] fp32, h)"""
    T = q.shape[0]
    h = h0.clone().float()
    o = torch.zeros(T, H, V, device=dev)
    A = torch.exp(a_log.view(H).float())
    for t in range(T):
        qf, kf, vf = q[t].float(), k[t].float(), v[t].float()
        qn = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * (K ** -0.5)
        kn = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        gate = LB / (1.0 + torch.exp(-(A[:, None] * (g1[t].float() + dt_bias.float()))))  # [H,K]
        h = h * torch.exp(gate)[:, None, :]
        vp = vf - torch.einsum("hvk,hk->hv", h, kn)
        vp = vp * torch.sigmoid(beta[t].float())[:, None]
        h = h + vp[:, :, None] * kn[:, None, :]
        o[t] = torch.einsum("hvk,hk->hv", h, qn)
    return o, h


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)).item()


a_log = torch.log(torch.rand(1, 1, H, 1, device=dev) * 8 + 1).float().contiguous()
dt_bias = (torch.randn(H * K, device=dev) * 0.5).float().contiguous()
ok = True
for L in (20, 300, 2500):
    T = L + 40
    q = torch.randn(T, H, K, device=dev).to(torch.bfloat16)
    k = torch.randn(T, H, K, device=dev).to(torch.bfloat16)
    v = torch.randn(T, H, V, device=dev).to(torch.bfloat16)
    g1 = torch.randn(T, H, K, device=dev).to(torch.bfloat16)
    beta = torch.randn(T, H, device=dev).to(torch.bfloat16)
    h0 = torch.zeros(H, V, K, device=dev)
    o_ref, h_ref_pref = ref_run(q[:L], k[:L], v[:L], g1[:L], beta[:L], a_log, dt_bias.view(H, K), h0)

    # --- FlashKDA prefill (nightly default) ---
    ws = torch.ops._flashkda_C.get_workspace_size(T, H, 4)
    workspace = torch.empty(ws, dtype=torch.uint8, device=dev)
    out = torch.empty(1, L, H, V, device=dev, dtype=torch.bfloat16)
    init = torch.zeros(1, H, V, K, device=dev)
    final = torch.zeros(1, H, V, K, device=dev)
    cu = torch.tensor([0, L], device=dev, dtype=torch.int32)
    torch.ops._flashkda_C.fwd(q[:L][None].contiguous(), k[:L][None].contiguous(), v[:L][None].contiguous(),
                              g1[:L][None].contiguous(), beta[:L][None].contiguous(), K ** -0.5, out, workspace,
                              a_log.view(-1), dt_bias.view(-1, K), LB, init.contiguous(), final, cu, None, None)
    torch.cuda.synchronize()
    e_o = rel(out[0], o_ref); e_h = rel(final[0], h_ref_pref)
    print(f"L={L} flashkda prefill: out rel err {e_o:.4g}, final state rel err {e_h:.4g}, "
          f"nan={torch.isnan(out).any().item() or torch.isnan(final).any().item()}")
    ok &= e_o < 0.05 and e_h < 0.05

    # --- Triton chunk prefill (old image path) ---
    init_t = torch.zeros(1, H, V, K, device=dev)
    o_t, h_t = chunk_kda_with_fused_gate(
        q=q[:L][None], k=k[:L][None], v=v[:L][None], raw_g=g1[:L][None],
        beta=torch.sigmoid(beta[:L].float())[None], A_log=a_log, g_bias=dt_bias,
        initial_state=init_t, output_final_state=True, use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu, safe_gate=True, lower_bound=LB)
    torch.cuda.synchronize()
    e_ot = rel(o_t[0], o_ref); e_ht = rel(h_t[0], h_ref_pref)
    print(f"L={L} triton chunk prefill: out rel err {e_ot:.4g}, final state rel err {e_ht:.4g}")
    ok &= e_ot < 0.05 and e_ht < 0.05

    # --- plain decode from the flashkda state ---
    pool = torch.zeros(4, H, V, K, device=dev)
    pool[1] = final[0]
    o_dec_ref, h_dec_ref = ref_run(q[L:], k[L:], v[L:], g1[L:], beta[L:], a_log, dt_bias.view(H, K), h_ref_pref)
    worst = 0.0
    for i in range(L, T):
        o, _ = fused_recurrent_kda(q=q[i:i+1][None], k=k[i:i+1][None], v=v[i:i+1][None], g=g1[i:i+1][None],
                                   beta=beta[i:i+1][None], initial_state=pool, use_qk_l2norm_in_kernel=True,
                                   cu_seqlens=torch.tensor([0, 1], device=dev, dtype=torch.int32),
                                   ssm_state_indices=torch.tensor([1], device=dev, dtype=torch.int32),
                                   sigmoid_beta=True, a_log=a_log, g_bias=dt_bias, compute_gate=True, lower_bound=LB)
        torch.cuda.synchronize()
        worst = max(worst, rel(o[0, 0], o_dec_ref[i - L]))
    e_hd = rel(pool[1], h_dec_ref)
    print(f"L={L} plain decode x40: worst out rel err {worst:.4g}, state after decode rel err {e_hd:.4g}, "
          f"|state| {pool[1].abs().mean().item():.4g}")
    ok &= worst < 0.05 and e_hd < 0.05
print("T9:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
