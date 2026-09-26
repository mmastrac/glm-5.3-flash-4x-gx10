#!/usr/bin/env python3
"""Offline decode-step reference for the MLA layers in a glm53_step_tap dump.

Two halves, per MLA layer and tapped row:

  up-projection  `mla.out` (the SM90 kernel's latent output, [H, kv_lora_rank])
                 times the checkpoint's W_UV (v half of kv_b_proj, per head)
                 against the dumped `mla.attn_out`. Needs only today's dumps.
  kernel         softmax(scale * q_nope . K_j) V_j over the latent rows the
                 kernel attended (K_j = V_j = fp8 cache row * k_scale), against
                 the dumped `mla.out`. Needs `mla.q_nope`, `mla.k_scale` and
                 `mla.kv_rows` from the extended tap (extra dict in the blob).

    python3 mla_ref.py --ckpt /ckpt /hit/vec-rank0-step3257.pt:92
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

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

    def linear(self, base):
        """fp32 [out, in] whether the layer is NVFP4 or BF16 in the checkpoint."""
        if self.has(base + ".weight_scale"):
            w = self.get(base + ".weight")
            s = self.get(base + ".weight_scale")
            g = self.get(base + ".weight_scale_2").reshape(()).float()
            return dequantize_to_dtype(w[None], s[None], g[None], torch.float32, 16, swizzle=False)[0], "nvfp4"
        return self.get(base + ".weight").float(), "bf16"


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return f"l2 {float((a - b).norm() / (b.norm() + 1e-12)):.2e} max {float((a - b).abs().max() / (b.abs().max() + 1e-12)):.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--kv-vs", default="", help="another dump: compare the rows this step READ (extra kv_rows) "
                    "against the cache_row that dump WROTE at the same positions (e.g. the prefill replay)")
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()
    ck = Ckpt(args.ckpt)
    cfg = ck.cfg
    H_all, nope, vd, lora = (int(cfg[k]) for k in ("num_attention_heads", "qk_nope_head_dim", "v_head_dim", "kv_lora_rank"))
    rope = int(cfg.get("qk_rope_head_dim") or 0)
    scale = (nope + rope) ** -0.5
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
        cps, extra = b["cps"], b.get("extra", {})
        layers = [int(x) for x in args.layers.split(",") if x] or sorted(
            int(k[1:].split(".")[0]) for k in cps if k.endswith(".mla.out")
        )
        print(f"\n== {os.path.basename(path)} pos {pos} rank {rank} ==")
        print(f"{'layer':6s} {'kv_b':6s} {'up-proj: attn_out vs ref':28s} {'kernel: out vs ref':28s} rows  |out|max")
        for L in layers:
            out = cps[f"L{L}.mla.out"][i].to(dev).float()
            H = out.numel() // lora
            out = out.view(H, lora)
            attn = cps[f"L{L}.mla.attn_out"][i].to(dev).float().view(H, vd)
            W, kind = ck.linear(f"{PREFIX}{L}.self_attn.kv_b_proj")
            W = W.view(H_all, nope + vd, lora)[rank * H : (rank + 1) * H]
            W_UV = W[:, nope:, :]  # [H, vd, lora]
            attn_ref = torch.einsum("hl,hvl->hv", out, W_UV)
            up = rel(attn, attn_ref)
            kern = "(needs q_nope/kv_rows from the extended tap)"
            nrows = "-"
            qn_key = f"L{L}.mla.q_nope"
            kv_key = f"L{L}.mla.kv_rows"
            if qn_key in cps and kv_key in extra:
                q_nope = cps[qn_key][i].to(dev).float().view(H, lora)
                kv = extra[kv_key][i].to(dev)
                ks = float(extra.get(f"L{L}.mla.k_scale", 1.0))
                if kv.dtype == torch.uint8:
                    kv = kv.view(torch.float8_e4m3fn)
                latent = kv.float() * ks  # [N, lora]
                logits = torch.einsum("hl,nl->hn", q_nope, latent) * scale
                p = torch.softmax(logits, dim=-1)
                out_ref = p @ latent
                kern = rel(out, out_ref)
                nrows = str(latent.shape[0])
            # fp8 KV write side: the cache row read back this step vs the bf16 latent written
            wr = ""
            cr_key, kvc_key = f"L{L}.mla.cache_row", f"L{L}.mla.kvc"
            if cr_key in cps and kvc_key in cps:
                cr = cps[cr_key][i].to(dev)
                kvc = cps[kvc_key][i].to(dev).float()
                if cr.dtype == torch.uint8:
                    cr = cr.view(torch.float8_e4m3fn)
                crf = cr.float()
                ks = float(extra.get(f"L{L}.mla.k_scale", 1.0))
                fit = float((crf * kvc).sum() / ((crf * crf).sum() + 1e-12))  # least-squares scale
                wr = f"kv write: {rel(crf * ks, kvc)} (k_scale {ks:g}, ls-fit {fit:.4g})"
            print(f"L{L:<5d} {kind:6s} {up:28s} {kern:28s} {nrows:5s} {float(out.abs().max()):.3f}  {wr}")
            # rows this step read vs rows another run wrote at the same positions
            if args.kv_vs and kv_key in extra:
                other = torch.load(args.kv_vs, map_location="cpu")
                ocr = other["cps"].get(cr_key)
                topk = cps[f"L{L}.mla.topk"][i]
                kv = extra[kv_key][i]
                if kv.dtype == torch.uint8:
                    kv = kv.view(torch.float8_e4m3fn)
                kvf = kv.float()
                worst, bad = 0.0, []
                for j in range(kvf.shape[0]):
                    p = int(topk[j])
                    orow = [k for k, r in enumerate(other["rows"]) if r["pos"] == p]
                    if not orow or ocr is None:
                        continue
                    o = ocr[orow[-1]]
                    o = (o.view(torch.float8_e4m3fn) if o.dtype == torch.uint8 else o).float()
                    d = float((kvf[j] - o).norm() / (o.norm() + 1e-12))
                    worst = max(worst, d)
                    if d > 0.5:
                        bad.append((p, round(d, 3)))
                print(f"        kv rows read vs {os.path.basename(args.kv_vs)} rows written: worst rel {worst:.3f}; rows > 0.5: {bad[:10]}")


if __name__ == "__main__":
    main()
