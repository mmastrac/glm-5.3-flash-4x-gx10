#!/usr/bin/env python3
"""Offline MLP/MoE reference for rows dumped by glm53_step_tap (DEEP mode).

For each `file:pos` spec the tool takes the row's `L<L>.mlp.in0` (the MLP
input, replicated) and `L<L>.mlp.out0` (the MLP output after the TP
all-reduce) from the vector dump, recomputes the layer from the nvidia NVFP4
checkpoint two ways, and reports how far the dumped output is from each:

  ref16  dequantised fp32 weights, fp32 activations, no activation quant
         (what a W4A16 path such as marlin approximates)
  ref4   same, but the activations are quantised to NVFP4 with the layer's
         static input_scale before each GEMM, exactly like the W4A4 cutlass
         path (scaled_fp4_quant + dequant)

Layers < first_k_dense_replace (0..2) are dense Glm5NextMLP; the others are
MoE: sigmoid scores, noaux_tc top-8 on scores + e_score_correction_bias,
weights renormalised and scaled by routed_scaling_factor, plus the BF16
shared expert. Between every pair of specs it also prints the input
difference, the dumped output difference, and what ref16/ref4 make of the
same two inputs, which separates legitimate sensitivity from a wrong path.

    python3 mlp_ref.py --ckpt /ckpt --layer 0 /hit/vec-rank0-step3257.pt:92 /hit/vec-rank0-step3283.pt:92
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

dev = torch.device("cuda")
PREFIX = "model.language_model.layers."


class Ckpt:
    def __init__(self, d: str):
        self.d = d
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        f = self._open.get(shard)
        if f is None:
            f = safe_open(os.path.join(self.d, shard), "pt", device="cuda")
            self._open[shard] = f
        return f.get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.map

    def fp4_linear(self, base: str):
        """Dequantised fp32 weight [out, in], and (weight_scale_2, input_scale)."""
        w = self.get(base + ".weight")
        s = self.get(base + ".weight_scale")
        g = self.get(base + ".weight_scale_2").reshape(()).float()
        a = self.get(base + ".input_scale").reshape(()).float()
        W = dequantize_to_dtype(w[None], s[None], g[None], torch.float32, 16, swizzle=False)[0]
        return W, float(g), float(a)


def quant_act(x: torch.Tensor, input_scale: float) -> torch.Tensor:
    """NVFP4-quantise a [M, K] activation with a static global scale, then dequantise (fp32)."""
    q, sf = ops.scaled_fp4_quant(
        x.to(torch.bfloat16), torch.tensor(1.0 / input_scale, device=dev), is_sf_swizzled_layout=True
    )
    return dequantize_to_dtype(q, sf, torch.tensor(input_scale, device=dev), torch.float32, 16, swizzle=True)


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def mlp3(x: torch.Tensor, Wg, Wu, Wd, a_gu: float | None, a_d: float | None, limit: float) -> torch.Tensor:
    """gate/up/down with optional activation quant (a_* = input_scale, None = no quant)."""
    xq = quant_act(x, a_gu) if a_gu else x
    h = swiglu(xq @ Wg.T, xq @ Wu.T, limit)
    hq = quant_act(h, a_d) if a_d else h
    return hq @ Wd.T


def dense_refs(ck: Ckpt, L: int, X: torch.Tensor, limit: float):
    base = f"{PREFIX}{L}.mlp."
    Wg, gg, ag = ck.fp4_linear(base + "gate_proj")
    Wu, gu, au = ck.fp4_linear(base + "up_proj")
    Wd, gd, ad = ck.fp4_linear(base + "down_proj")
    a_gu = max(ag, au)  # vLLM's merged gate_up takes the max shard input_scale
    print(f"dense L{L}: input_scale gate {ag:.5g} up {au:.5g} (merged {a_gu:.5g}) down {ad:.5g}; "
          f"weight_scale_2 {gg:.4g}/{gu:.4g}/{gd:.4g}")
    r16 = mlp3(X, Wg, Wu, Wd, None, None, limit)
    r4 = mlp3(X, Wg, Wu, Wd, a_gu, ad, limit)
    return r16, r4, {}


def moe_refs(ck: Ckpt, L: int, X: torch.Tensor, limit: float):
    cfg = ck.cfg
    E, K8, rsf = int(cfg["n_routed_experts"]), int(cfg["num_experts_per_tok"]), float(cfg["routed_scaling_factor"])
    base = f"{PREFIX}{L}.mlp."
    gw = ck.get(base + "gate.weight").float()  # [E, hidden]
    bias = ck.get(base + "gate.e_score_correction_bias").float() if ck.has(base + "gate.e_score_correction_bias") else torch.zeros(E, device=dev)
    logits = X @ gw.T
    scores = torch.sigmoid(logits)
    choice = scores + bias
    top = torch.topk(choice, K8 + 1, dim=-1)
    ids = top.indices[:, :K8]
    margin = (top.values[:, K8 - 1] - top.values[:, K8]).tolist()
    w = torch.gather(scores, 1, ids)
    if cfg.get("norm_topk_prob", True):
        w = w / w.sum(-1, keepdim=True)
    w = w * rsf
    # global activation scales as the cutlass path uses them: max over experts
    a13 = max(float(ck.get(f"{base}experts.{e}.{p}.input_scale").reshape(())) for e in range(E) for p in ("gate_proj", "up_proj"))
    a2 = max(float(ck.get(f"{base}experts.{e}.down_proj.input_scale").reshape(())) for e in range(E))
    print(f"moe L{L}: global input_scale w13 {a13:.5g} w2 {a2:.5g}; routed_scaling {rsf}; renorm {cfg.get('norm_topk_prob')}")
    # shared expert: BF16 (in the quantisation ignore list)
    sg = ck.get(base + "shared_experts.gate_proj.weight").float()
    su = ck.get(base + "shared_experts.up_proj.weight").float()
    sd = ck.get(base + "shared_experts.down_proj.weight").float()
    shared = mlp3(X, sg, su, sd, None, None, limit)
    r16 = shared.clone()
    r4 = shared.clone()
    cache = {}
    for r in range(X.shape[0]):
        for j in range(K8):
            e = int(ids[r, j])
            if e not in cache:
                Wg, _, _ = ck.fp4_linear(f"{base}experts.{e}.gate_proj")
                Wu, _, _ = ck.fp4_linear(f"{base}experts.{e}.up_proj")
                Wd, _, _ = ck.fp4_linear(f"{base}experts.{e}.down_proj")
                cache[e] = (Wg, Wu, Wd)
            Wg, Wu, Wd = cache[e]
            xr = X[r : r + 1]
            r16[r] += w[r, j] * mlp3(xr, Wg, Wu, Wd, None, None, limit)[0]
            r4[r] += w[r, j] * mlp3(xr, Wg, Wu, Wd, a13, a2, limit)[0]
    info = {"ids": ids.tolist(), "w": [[round(v, 4) for v in row] for row in w.tolist()], "margin_8_vs_9": margin}
    return r16, r4, info


def rel(a: torch.Tensor, b: torch.Tensor) -> str:
    l2 = float((a - b).norm() / (b.norm() + 1e-12))
    mx = float((a - b).abs().max() / (b.abs().max() + 1e-12))
    return f"l2 {l2:.4f} max {mx:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("specs", nargs="+", help="vec.pt:pos")
    args = ap.parse_args()
    ck = Ckpt(args.ckpt)
    L = args.layer
    limit = float(ck.cfg.get("swiglu_limit") or 1e30)
    dense = L < int(ck.cfg.get("first_k_dense_replace", 0))
    names, X, Y = [], [], []
    for spec in args.specs:
        path, _, pos = spec.rpartition(":")
        b = torch.load(path, map_location="cpu")
        rows = [i for i, r in enumerate(b["rows"]) if r["pos"] == int(pos)]
        if not rows:
            print(f"{spec}: no row at pos {pos}")
            return 1
        i = rows[-1]
        names.append(f"{os.path.basename(path)}:{pos}")
        X.append(b["cps"][f"L{L}.mlp.in0"][i].float())
        Y.append(b["cps"][f"L{L}.mlp.out0"][i].float())
    X = torch.stack(X).to(dev)
    Y = torch.stack(Y).to(dev)
    with torch.no_grad():
        r16, r4, info = (dense_refs if dense else moe_refs)(ck, L, X, limit)
    print(f"\nlayer L{L} ({'dense' if dense else 'moe'}), swiglu_limit {limit}")
    for i, n in enumerate(names):
        extra = ""
        if info:
            extra = f"  experts {info['ids'][i]} w {info['w'][i]} margin8/9 {info['margin_8_vs_9'][i]:.4f}"
        print(f"{n:34s} |in|max {float(X[i].abs().max()):.3f} |out|max {float(Y[i].abs().max()):.3f}{extra}")
        print(f"   dumped vs ref16: {rel(Y[i], r16[i])}   dumped vs ref4: {rel(Y[i], r4[i])}   ref4 vs ref16 (quant loss): {rel(r4[i], r16[i])}")
    print("\npairs (a vs b):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = ""
            if info and info["ids"][i] != info["ids"][j]:
                same = f"  EXPERT SETS DIFFER: {sorted(set(info['ids'][i]) ^ set(info['ids'][j]))}"
            print(f"{names[i]} vs {names[j]}{same}")
            print(f"   input {rel(X[i], X[j])} | dumped out {rel(Y[i], Y[j])} | ref16 out {rel(r16[i], r16[j])} | ref4 out {rel(r4[i], r4[j])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
