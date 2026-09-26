#!/usr/bin/env python3
"""Compare HF's per-layer residual streams (hf_ref.py output) with vLLM tap dumps
at one position, layer by layer.

HF layer L output = the 4 residual streams after layer L's ffn post-mix. In
vLLM's DEEP dump the same tensor is the NEXT layer's fused op output
`L{L+1}.hc_fused.out0` (the deferred post applied), and for the last layer
`L44.hc_post.out0`. The final normed hidden is compared by applying the norm
weight to the stream mean of vLLM's last output.

    python3 hf_compare.py --hf /out/hf_layers.pt --ckpt /ckpt --pos 338 /hit/vec-rank0-step464.pt /hit/vec-rank0-step470.pt
"""

import argparse
import json
import os

import torch
from safetensors import safe_open


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / (b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    hf = torch.load(args.hf, map_location="cpu")
    kp = hf["keep_pos"]
    if args.pos not in kp:
        print("HF kept positions", kp)
        return 1
    hi = kp.index(args.pos)
    nlayers = max(hf["layers"]) + 1
    idx = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))["weight_map"]
    nm = "model.language_model.norm.weight"
    normw = safe_open(os.path.join(args.ckpt, idx[nm]), "pt").get_tensor(nm).float()
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    eps = float(tc.get("rms_norm_eps", 1e-5))
    dumps = []
    for p in args.dumps:
        b = torch.load(p, map_location="cpu")
        rows = [i for i, r in enumerate(b["rows"]) if r["pos"] == args.pos]
        if not rows:
            print(f"{p}: no row at pos {args.pos}")
            return 1
        dumps.append((os.path.basename(p), b["cps"], rows[-1]))
    print(f"pos {args.pos}: relative l2 difference of the residual streams vs HF, per layer")
    print("layer  " + "  ".join(f"{n:>28s}" for n, _, _ in dumps) + "   (vLLM dump vs vLLM dump)")
    for L in range(nlayers):
        h = hf["layers"][L][hi].float()  # [4, 4096]
        key = f"L{L + 1}.hc_fused.out0" if L + 1 < nlayers else f"L{L}.hc_post.out0"
        vals, vecs = [], []
        for n, cps, i in dumps:
            if key not in cps:
                vals.append(float("nan"))
                vecs.append(None)
                continue
            v = cps[key][i].float().view(4, -1)
            vals.append(rel(v, h))
            vecs.append(v)
        cross = rel(vecs[0], vecs[1]) if len(vecs) > 1 and vecs[0] is not None and vecs[1] is not None else float("nan")
        print(f"L{L:<5d}" + "  ".join(f"{x:28.4f}" for x in vals) + f"   {cross:.4f}")
    # final hidden: HF normed vs norm(mean of vLLM last streams)
    hfin = hf["final_hidden"][hi].float()
    print("final normed hidden vs HF:")
    for n, cps, i in dumps:
        key = f"L{nlayers - 1}.hc_post.out0"
        if key in cps:
            m = cps[key][i].float().view(4, -1).mean(0)
            x = m * torch.rsqrt(m.pow(2).mean() + eps) * normw
            print(f"  {n}: rel {rel(x, hfin):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
