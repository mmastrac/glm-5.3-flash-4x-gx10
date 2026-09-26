#!/usr/bin/env python3
"""fused_marlin_moe returns different bits for the same inputs at some M.

No checkpoint needed: random NVFP4 weights, one CUDA device. Runs the same
call repeatedly and compares the outputs bit for bit.

    python marlin_moe_nondet.py
"""
import argparse

import torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,
)
from vllm.scalar_type import scalar_types

DEV = "cuda"


def quant_stack(experts, n, k, group):
    q, s, g = [], [], []
    for _ in range(experts):
        w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.05
        _, qw, sw, gw = rand_marlin_weight_nvfp4_like(w, group)
        q.append(qw), s.append(sw), g.append(gw)
    return torch.stack(q), torch.stack(s), torch.stack(g)


def trial(w1, w2, m, reps, experts, hidden, topk):
    gen = torch.Generator(device=DEV).manual_seed(1)
    x = (torch.randn(m, hidden, device=DEV, generator=gen) * 0.5).to(torch.bfloat16)
    weights, ids = torch.topk(
        torch.softmax(torch.randn(m, experts, device=DEV, generator=gen), -1), topk, dim=-1
    )
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    ids = ids.to(torch.int32)

    outs = []
    for _ in range(reps):
        out = fused_marlin_moe(
            x, w1[0], w2[0], None, None, w1[1], w2[1], weights, ids,
            scalar_types.float4_e2m1f.id, global_num_experts=experts,
            global_scale1=w1[2], global_scale2=w2[2],
        )
        torch.cuda.synchronize()
        outs.append(out.clone())

    differing = sum(1 for o in outs[1:] if not torch.equal(o, outs[0]))
    worst = max(float((o.float() - outs[0].float()).abs().max()) for o in outs[1:])
    return differing, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 1024, 2048, 2816, 3072, 3104, 3136])
    a = ap.parse_args()

    torch.manual_seed(0)
    w1 = quant_stack(a.experts, 2 * a.inter, a.hidden, a.group)  # gate and up
    w2 = quant_stack(a.experts, a.hidden, a.inter, a.group)      # down
    print(f"{torch.cuda.get_device_name(0)}  experts={a.experts} topk={a.topk} "
          f"hidden={a.hidden} inter={a.inter} reps={a.reps}")
    for m in a.m:
        differing, worst = trial(w1, w2, m, a.reps, a.experts, a.hidden, a.topk)
        print(f"M={m:<6} runs differing from the first: {differing}/{a.reps - 1}   "
              f"max abs diff {worst:.3e}")


if __name__ == "__main__":
    main()
