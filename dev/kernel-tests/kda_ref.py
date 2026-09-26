#!/usr/bin/env python3
"""Offline decode-step reference for every KDA layer in a glm53_step_tap dump.

A decode dump holds, per KDA layer and tapped row, everything the layer's
decode path consumed and produced: `kda.qkv` (merged q|k|v projection, before
the short conv), `kda.g1`, `kda.beta` (raw), the request's `conv_before` and
`rec_before` state blocks, and `kda.out`, `conv_after`, `rec_after`. This
tool replays the step in fp32 from those inputs with the checkpoint's conv
weights, A_log and dt_bias (sliced to the dump's TP rank), using the same
math as T9's reference (l2-normed q/k, bounded sigmoid gate, sigmoid beta),
and reports how far the dumped output and states are from it. Both conv
state layouts (dim-first / state-first) and both recurrent state layouts
([H,V,K] / [H,K,V]) are tried; the matching one is reported.

    python3 kda_ref.py --ckpt /ckpt /hit/vec-rank0-step3257.pt:92 [more specs]
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open

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

    def has(self, name):
        return name in self.map


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return f"l2 {float((a - b).norm() / (b.norm() + 1e-12)):.2e} max {float((a - b).abs().max() / (b.abs().max() + 1e-12)):.2e}"


def step_ref(q, k, v, g1, beta, a_log, dt_bias, h, lb, D):
    """One recurrent step. q,k,v,g1 [H,D]; beta [H]; h [H,V,K]; returns (o [H,V], h)."""
    A = torch.exp(a_log.float()).view(-1)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * (D**-0.5)
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gate = lb / (1.0 + torch.exp(-(A[:, None] * (g1 + dt_bias))))
    h = h * torch.exp(gate)[:, None, :]
    vp = v - torch.einsum("hvk,hk->hv", h, kn)
    vp = vp * torch.sigmoid(beta)[:, None]
    h = h + vp[:, :, None] * kn[:, None, :]
    return torch.einsum("hvk,hk->hv", h, qn), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="", help="comma list; default every KDA layer in the dump")
    ap.add_argument("--state-vs", default="", help="a prefill dump (extended tap): compare this step's rec/conv_after "
                    "with the state that run left after its last position (extra kda.state_after_req)")
    ap.add_argument("specs", nargs="+", help="vec.pt:pos")
    args = ap.parse_args()
    ck = Ckpt(args.ckpt)
    cfg = ck.cfg
    lac = cfg.get("linear_attn_config") or {}
    H_all = int(cfg.get("linear_num_heads") or lac["num_heads"])
    D = int(cfg.get("linear_head_dim") or lac["head_dim"])
    W = int(cfg.get("linear_conv_kernel_dim") or lac["short_conv_kernel_size"])
    lb = float(cfg["linear_lower_bound"] if "linear_lower_bound" in cfg else lac["gate_lower_bound"])
    for spec in args.specs:
        path, _, pos = spec.rpartition(":")
        m = re.search(r"rank(-?\d+)", os.path.basename(path))
        rank = int(m.group(1)) if m else 0
        b = torch.load(path, map_location="cpu")
        rows = [i for i, r in enumerate(b["rows"]) if r["pos"] == int(pos)]
        if not rows:
            print(f"{spec}: no row at pos {pos}")
            continue
        i = rows[-1]
        cps = b["cps"]
        layers = [int(x) for x in args.layers.split(",") if x] or sorted(
            int(k[1:].split(".")[0]) for k in cps if k.endswith(".kda.out")
        )
        other = torch.load(args.state_vs, map_location="cpu") if args.state_vs else None
        print(f"\n== {os.path.basename(path)} pos {pos} rank {rank} (row {b['rows'][i]}) ==")
        print(f"{'layer':6s} {'conv':22s} {'out vs ref':24s} {'rec_after vs ref':24s} layouts   |out|max")
        for L in layers:
            g = lambda n: cps[f"L{L}.kda.{n}"][i].to(dev)
            qkv, g1, beta, out = g("qkv").float(), g("g1").float(), g("beta").float(), g("out").float()
            cb, ca, rb, ra = g("conv_before").float(), g("conv_after").float(), g("rec_before").float(), g("rec_after").float()
            H = beta.numel()
            tp = H_all // H
            P = H * D  # local projection size
            assert qkv.numel() == 3 * P, (qkv.shape, P)
            base = f"{PREFIX}{L}.self_attn."
            if not ck.has(base + "q_conv1d.weight"):
                base = f"{PREFIX}{L}.self_attn.linear_attn."
            ws = []
            for p in ("q", "k", "v"):
                w = ck.get(f"{base}{p}_conv1d.weight").float().reshape(H_all * D, W)
                ws.append(w[rank * P : (rank + 1) * P])
            cw = torch.cat(ws, 0)  # [3P, W]
            a_log = ck.get(base + "A_log").float().reshape(-1)[rank * H : (rank + 1) * H]
            dt_bias = ck.get(base + "dt_bias").float().reshape(-1)[rank * P : (rank + 1) * P].view(H, D)
            # --- short conv, both state layouts ---
            best = None
            for layout in ("DS", "SD"):
                st = cb.view(3 * P, W - 1) if layout == "DS" else cb.view(W - 1, 3 * P).T
                win = torch.cat([st, qkv[:, None]], 1)  # [3P, W]
                y = torch.nn.functional.silu((win * cw).sum(1))
                new_st = win[:, 1:]
                ca_v = ca.view(3 * P, W - 1) if layout == "DS" else ca.view(W - 1, 3 * P).T
                e = float((new_st - ca_v).abs().max())
                if best is None or e < best[1]:
                    best = (layout, e, y)
            conv_layout, conv_err, y = best
            q, k, v = y.split(P)
            q, k, v, g1r = q.view(H, D), k.view(H, D), v.view(H, D), g1.view(H, D)
            # --- recurrent step, both state layouts ---
            res = None
            for hl in ("HVK", "HKV"):
                h0 = rb.view(H, D, D)
                if hl == "HKV":
                    h0 = h0.transpose(1, 2)
                o, h1 = step_ref(q, k, v, g1r, beta, a_log, dt_bias, h0, lb, D)
                h1_cmp = h1 if hl == "HVK" else h1.transpose(1, 2)
                e_o = float((o.reshape(-1) - out).norm() / (out.norm() + 1e-12))
                if res is None or e_o < res[0]:
                    res = (e_o, hl, o, h1_cmp)
            e_o, hl, o, h1 = res
            sv = ""
            if other is not None:
                st = other.get("extra", {}).get(f"L{L}.kda.state_after_req")
                if st:
                    oc, orc = next(iter(st.values()))
                    sv = f"  | vs {os.path.basename(args.state_vs)} state: conv {rel(ca, oc.reshape(-1).to(dev))} rec {rel(ra, orc.reshape(-1).to(dev))}"
            print(f"L{L:<5d} {conv_layout} state maxdiff {conv_err:.1e}   {rel(out, o.reshape(-1)):24s} {rel(ra, h1.reshape(-1)):24s} {conv_layout}/{hl}  {float(out.abs().max()):.3f}{sv}")


if __name__ == "__main__":
    main()
