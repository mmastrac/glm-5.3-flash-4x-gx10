#!/usr/bin/env python3
"""Offline reference for the mHC ops in a DEEP glm53_step_tap dump.

The dump holds every mHC op call's inputs (by keyword: x, residual,
post_layer_mix, comb_res_mix) and outputs (residual, post, comb, layer input)
per layer: `hc_pre` (layer 0), `hc_fused` (attn pre, fused with the previous
layer's post), `hc_fused@1` (ffn pre) and `hc_post` (last layer). This replays
each call with vLLM's own torch implementation (mhc_pre_torch / mhc_post_torch)
on the checkpoint's hc_* parameters and reports the relative difference of
every output. The CUDA op fuses the RMSNorm into the layer input, which the
torch reference does not, so the norm is applied here before comparing out3.

    python3 hc_ref.py --ckpt /ckpt /hit/vec-rank0-step3257.pt:92 /hit/vec-rank0-step3283.pt:92
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

import vllm.model_executor.kernels.mhc as mk

dev = torch.device("cuda")
PREFIX = "model.language_model.layers."


class Ckpt:
    def __init__(self, d):
        self.d = d
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def get(self, name):
        f = self._open.get(self.map[name])
        if f is None:
            f = safe_open(os.path.join(self.d, self.map[name]), "pt", device="cuda")
            self._open[self.map[name]] = f
        return f.get_tensor(name)


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / (b.norm() + 1e-12))


def rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, default=1e-2)
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()
    ck = Ckpt(args.ckpt)
    cfg = ck.cfg
    n = int(cfg.get("mhc_num_residual_streams") or 4)
    hid = int(cfg["hidden_size"])
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    hc_eps = float(cfg.get("hc_eps") or 1e-6)
    post_mult = float(cfg.get("mhc_post_mult_value") or 2.0)
    sink = int(cfg.get("mhc_sinkhorn_iterations") or 20)
    for spec in args.specs:
        path, _, pos = spec.rpartition(":")
        b = torch.load(path, map_location="cpu")
        rows = [i for i, r in enumerate(b["rows"]) if r["pos"] == int(pos)]
        if not rows:
            print(f"{spec}: no row at pos {pos}")
            continue
        i = rows[-1]
        cps = b["cps"]
        g = lambda k: cps[k][i].to(dev)
        layers = sorted({int(k[1:].split(".")[0]) for k in cps if ".hc_" in k})
        print(f"\n== {os.path.basename(path)} pos {pos} (rows in dump: {len(b['rows'])}) ==")
        print(f"{'layer':6s} {'call':10s} {'residual':>10s} {'post':>10s} {'comb':>10s} {'x(normed)':>10s}")
        worst = 0.0
        for L in layers:
            P = f"{PREFIX}{L}."
            params = {
                "attn": (ck.get(P + "hc_attn_fn").float(), ck.get(P + "hc_attn_scale").float(), ck.get(P + "hc_attn_base").float(), ck.get(P + "input_layernorm.weight")),
                "ffn": (ck.get(P + "hc_ffn_fn").float(), ck.get(P + "hc_ffn_scale").float(), ck.get(P + "hc_ffn_base").float(), ck.get(P + "post_attention_layernorm.weight")),
            }
            for call, kind in (("hc_pre", "attn"), ("hc_fused", "attn"), ("hc_fused@1", "ffn")):
                key = f"L{L}.{call}.out0"
                if key not in cps:
                    continue
                fn, sc, base, nw = params[kind]
                if call == "hc_pre":
                    residual_cur = g(f"L{L}.hc_pre.in_residual").view(n, hid)
                    res_out = None
                    post_d, comb_d, x_d = g(key), g(f"L{L}.{call}.out1"), g(f"L{L}.{call}.out2")
                else:
                    x = g(f"L{L}.{call}.in_x")
                    residual = g(f"L{L}.{call}.in_residual").view(n, hid)
                    post = g(f"L{L}.{call}.in_post_layer_mix").view(n, 1)
                    comb = g(f"L{L}.{call}.in_comb_res_mix").view(n, n)
                    residual_cur = mk.mhc_post_torch(x, residual, post, comb)
                    res_out = g(key).view(n, hid)
                    post_d, comb_d, x_d = g(f"L{L}.{call}.out1"), g(f"L{L}.{call}.out2"), g(f"L{L}.{call}.out3")
                post_r, comb_r, xin_r = mk.mhc_pre_torch(residual_cur, fn, sc, base, rms_eps, hc_eps, hc_eps, post_mult, sink)
                x_r = rmsnorm(xin_r.reshape(-1, hid), nw, rms_eps)
                e = [rel(res_out, residual_cur) if res_out is not None else float("nan"), rel(post_d, post_r), rel(comb_d, comb_r), rel(x_d, x_r)]
                worst = max(worst, *[v for v in e if v == v])
                flag = "  <--" if any(v == v and v > args.thr for v in e) else ""
                print(f"L{L:<5d} {call:10s} {e[0]:10.2e} {e[1]:10.2e} {e[2]:10.2e} {e[3]:10.2e}{flag}")
            key = f"L{L}.hc_post.out0"
            if key in cps:
                x, residual = g(f"L{L}.hc_post.in0"), g(f"L{L}.hc_post.in1").view(n, hid)
                post, comb = g(f"L{L}.hc_post.in2").view(n, 1), g(f"L{L}.hc_post.in3").view(n, n)
                r = rel(g(key).view(n, hid), mk.mhc_post_torch(x, residual, post, comb))
                worst = max(worst, r)
                print(f"L{L:<5d} {'hc_post':10s} {r:10.2e}{'  <--' if r > args.thr else ''}")
        print(f"worst relative difference: {worst:.3e} (threshold {args.thr})")


if __name__ == "__main__":
    main()
