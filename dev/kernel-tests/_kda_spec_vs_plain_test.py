#!/usr/bin/env python3
"""T2: GLM-5.3-Flash KDA spec-decode path vs the plain one-token-per-step path.

Drives the exact kernels kda.py uses (causal_conv1d_update with
num_accepted_tokens/query_start_loc, fused_recurrent_kda with 2D
ssm_state_indices + num_accepted_tokens) through a simulated speculative
decode with random acceptance, and compares every ACCEPTED row against the
same token processed one at a time by the non-spec decode kernels (what a
spec-off boot runs). GLM-5.3-Flash dims at TP=1: 64 heads x 128, conv width 4.
"""
import sys

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

torch.manual_seed(1234)
dev = torch.device("cuda")
H, D = 64, 128
PROJ = H * D
CONV_DIM = 3 * PROJ
WIDTH = 4
LOWER_BOUND = -5.0
LINES = 64


def torch_conv_ref(xs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """xs [T, dim] bf16, w [dim, WIDTH] fp32 -> silu(causal conv) [T, dim] fp32."""
    T = xs.shape[0]
    x = xs.float()
    pad = torch.zeros(WIDTH - 1, x.shape[1], device=dev)
    xp = torch.cat([pad, x], 0)
    out = torch.zeros_like(x)
    for j in range(WIDTH):
        out += w[:, j][None, :] * xp[j : j + T]
    return torch.nn.functional.silu(out)


def run(k: int, T: int = 260, warm: int = 20, seed: int = 0, verbose: bool = True):
    g = torch.Generator(device=dev).manual_seed(seed)
    state_len = WIDTH - 1 + k
    conv_w = (torch.randn(CONV_DIM, WIDTH, device=dev, generator=g) * 0.3).float()
    a_log = torch.log(torch.rand(1, 1, H, 1, device=dev, generator=g) * 8 + 1).float().contiguous()
    dt_bias = (torch.randn(PROJ, device=dev, generator=g) * 0.5).float().contiguous()

    xs = torch.randn(T, CONV_DIM, device=dev, generator=g).to(torch.bfloat16)
    g1s = torch.randn(T, H, D, device=dev, generator=g).to(torch.bfloat16)
    betas = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    garbage = torch.randn(T, CONV_DIM, device=dev, generator=g).to(torch.bfloat16)
    ggarb = torch.randn(T, H, D, device=dev, generator=g).to(torch.bfloat16)
    bgarb = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)

    conv_pool = torch.zeros(LINES, CONV_DIM, state_len, device=dev, dtype=torch.bfloat16)
    rec_pool = torch.zeros(LINES, H, D, D, device=dev, dtype=torch.float32)

    ref_line = 1
    spec_cols = torch.tensor([[2 + i for i in range(k + 1)]], device=dev, dtype=torch.int32)

    def plain_step(line: int, t: int):
        x = xs[t : t + 1].clone()
        y = causal_conv1d_update(
            x, conv_pool, conv_w, None, activation="silu",
            conv_state_indices=torch.tensor([line], device=dev, dtype=torch.int32),
        )
        q, kk, v = y.split(PROJ, dim=-1)
        o, _ = fused_recurrent_kda(
            q=q.reshape(1, -1, H, D), k=kk.reshape(1, -1, H, D), v=v.reshape(1, -1, H, D),
            g=g1s[t : t + 1].reshape(1, -1, H, D), beta=betas[t : t + 1].reshape(1, -1, H),
            initial_state=rec_pool, use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, 1], device=dev, dtype=torch.int32),
            ssm_state_indices=torch.tensor([line], device=dev, dtype=torch.int32),
            sigmoid_beta=True, a_log=a_log, g_bias=dt_bias, compute_gate=True,
            lower_bound=LOWER_BOUND,
        )
        return y[0].float(), o[0, 0].float()

    # Warm both from the same state using the plain path on the ref line, then copy.
    for t in range(warm):
        plain_step(ref_line, t)
    conv_pool[int(spec_cols[0, 0])] = conv_pool[ref_line]
    rec_pool[int(spec_cols[0, 0])] = rec_pool[ref_line]

    # Reference outputs for every remaining token via the plain path.
    o_ref = torch.zeros(T, H, D, device=dev)
    y_ref = torch.zeros(T, CONV_DIM, device=dev)
    for t in range(warm, T):
        y_ref[t], o_ref[t] = plain_step(ref_line, t)
    y_torch = torch_conv_ref(xs, conv_w)

    # Spec path.
    p = warm
    na_prev = 1
    step = 0
    worst_o = worst_y = worst_yt = 0.0
    hist = []
    while p + k + 1 <= T:
        na = int(torch.randint(1, k + 2, (1,), generator=g, device=dev))
        xq = torch.cat([xs[p : p + na], garbage[p + na : p + k + 1]], 0).clone()
        gq = torch.cat([g1s[p : p + na], ggarb[p + na : p + k + 1]], 0)
        bq = torch.cat([betas[p : p + na], bgarb[p + na : p + k + 1]], 0)
        assert xq.shape[0] == k + 1
        nacc = torch.tensor([na_prev], device=dev, dtype=torch.int32)
        qsl = torch.tensor([0, k + 1], device=dev, dtype=torch.int32)
        y = causal_conv1d_update(
            xq, conv_pool, conv_w, None, activation="silu",
            conv_state_indices=spec_cols[:, 0], num_accepted_tokens=nacc,
            query_start_loc=qsl, max_query_len=k + 1,
        )
        q, kk, v = y.split(PROJ, dim=-1)
        o, _ = fused_recurrent_kda(
            q=q.reshape(1, -1, H, D), k=kk.reshape(1, -1, H, D), v=v.reshape(1, -1, H, D),
            g=gq.reshape(1, -1, H, D), beta=bq.reshape(1, -1, H),
            initial_state=rec_pool, use_qk_l2norm_in_kernel=True,
            cu_seqlens=qsl, ssm_state_indices=spec_cols, num_accepted_tokens=nacc,
            sigmoid_beta=True, a_log=a_log, g_bias=dt_bias, compute_gate=True,
            lower_bound=LOWER_BOUND,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(o[0, :na].float()).all(), f"spec recurrent output has NaN/Inf at step {step}"
        assert torch.isfinite(o_ref[p : p + na]).all(), f"plain recurrent output has NaN/Inf at p={p}"
        assert o_ref[p : p + na].abs().max().item() > 1e-3, f"plain recurrent output is ~zero at p={p}"
        ey = (y[:na].float() - y_ref[p : p + na]).abs().max().item()
        eyt = (y[:na].float() - y_torch[p : p + na]).abs().max().item()
        eo = (o[0, :na].float() - o_ref[p : p + na]).abs().max().item()
        oscale = o_ref[p : p + na].abs().max().item()
        hist.append((step, p, na_prev, na, ey, eo, oscale))
        worst_y = max(worst_y, ey)
        worst_yt = max(worst_yt, eyt)
        worst_o = max(worst_o, eo / max(oscale, 1e-6))
        if verbose and (eo / max(oscale, 1e-6) > 0.02 or ey > 0.05):
            per_row = [(o[0, i].float() - o_ref[p + i]).abs().max().item() for i in range(na)]
            print(f"  k={k} step={step} p={p} na_prev={na_prev} na={na}: conv err {ey:.3g} "
                  f"rec err {eo:.3g} (scale {oscale:.3g}) per-row {[round(e, 3) for e in per_row]}")
        na_prev = na
        p += na
        step += 1
    print(f"k={k}: recurrent out |mean| {o_ref[warm:].abs().mean().item():.4g} max {o_ref[warm:].abs().max().item():.4g}; "
          f"conv out |mean| {y_ref[warm:].abs().mean().item():.4g}")
    print(f"k={k}: {step} spec steps, {p - warm} accepted tokens; worst conv err vs plain {worst_y:.3g}, "
          f"vs torch {worst_yt:.3g}; worst recurrent rel err {worst_o:.3g}")
    return worst_y, worst_o


ok = True
for k in (3, 7):
    for seed in (0, 1):
        wy, wo = run(k, seed=seed)
        ok &= wy < 0.05 and wo < 0.02
print("T2:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
