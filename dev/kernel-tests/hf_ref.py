#!/usr/bin/env python3
"""Independent ground truth for GLM-5.3-Flash: HF transformers' own glm5_next
modeling code (torch-only paths: no fla, no causal_conv1d, no kernels hub, eager
attention), fed the official FP8 checkpoint dequantised layer by layer.

The text model is built on the meta device; each decoder layer's parameters are
materialised on the GPU (FP8 e4m3 x 128x128 weight_scale_inv -> bf16) by a
forward pre-hook right before the layer runs and released after it, so a
339-token forward of the 320 GB checkpoint fits in GB10 memory. One forward
gives every position at once.

Outputs (in OUT dir):
  hf_top5.json      top-5 (token, logprob) at the requested positions
  hf_top5_all.json  top-5 at every position, plus the logprob of the token the
                    vLLM run actually produced at each generated index
  hf_layers.pt      per-layer residual streams [4, 4096] at the watched
                    positions, the final normed hidden, and the input ids

    python3 hf_ref.py --ckpt /ckpt --tapshot /hit/tapshot.json --out /out \
        --gen 310 --watch-gen 74,142,148,176,310
"""

import argparse
import gc
import json
import os
import resource
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

dev = torch.device("cuda")
BS = 128  # FP8 block size
DTYPE = torch.bfloat16


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def raw(self, name):
        f = self._open.get(self.map[name])
        if f is None:
            f = safe_open(os.path.join(self.d, self.map[name]), "pt", device="cpu")
            self._open[self.map[name]] = f
        return f.get_tensor(name)

    def has(self, name):
        return name in self.map

    def close(self):
        self._open.clear()

    def load(self, name):
        """bf16 tensor on the GPU; FP8 block-quantised weights are dequantised."""
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name[: -len(".weight")] + ".weight_scale_inv").to(dev).float()
            t = t.to(dev).float()
            out, inn = t.shape
            s = s.repeat_interleave(BS, 0)[:out].repeat_interleave(BS, 1)[:, :inn]
            return (t * s).to(DTYPE)
        return t.to(dev, DTYPE)


def layer_tensor(ck, L, hf, E):
    P = f"model.language_model.layers.{L}."
    if hf.startswith("self_attn.forget_gate."):
        return ck.load(P + "self_attn." + hf.split("forget_gate.", 1)[1])
    if hf == "self_attn.conv1d.weight":
        return torch.cat([ck.load(P + f"self_attn.{c}_conv1d.weight") for c in "qkv"], 0)
    hc = {"attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
          "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale"}
    if hf in hc:
        return ck.load(P + hc[hf])
    if hf == "mlp.experts.gate_up_proj":
        return torch.stack([torch.cat([ck.load(P + f"mlp.experts.{e}.gate_proj.weight"),
                                       ck.load(P + f"mlp.experts.{e}.up_proj.weight")], 0) for e in range(E)])
    if hf == "mlp.experts.down_proj":
        return torch.stack([ck.load(P + f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
    return ck.load(P + hf)


def set_param(root, name, tensor, buffer=False):
    mod = root
    parts = name.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    if buffer:
        mod._buffers[parts[-1]] = tensor
    else:
        mod._parameters[parts[-1]] = torch.nn.Parameter(tensor, requires_grad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=310, help="generated tokens to append to the prompt")
    ap.add_argument("--watch-gen", default="74,142,148,176,310", help="generated indices whose predicting position is reported")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="model/weight dtype (fp32 = perturbation/truth run)")
    ap.add_argument("--bias-dtype", default="model", choices=["model", "fp32"],
                    help="dtype of the router e_score_correction_bias buffer: 'model' (bf16 when --dtype bf16, "
                         "as from_pretrained does) or fp32 (what vLLM uses)")
    args = ap.parse_args()
    global DTYPE
    DTYPE = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    snap = json.load(open(args.tapshot))
    prompt = list(snap["prompt"])
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    gen_ids = tok("".join(snap["strs"][: args.gen]), add_special_tokens=False)["input_ids"]
    assert len(gen_ids) == args.gen, f"re-tokenised {len(gen_ids)} ids, expected {args.gen}"
    ids = prompt + gen_ids
    T = len(ids)
    P0 = len(prompt)
    watch_gen = [int(x) for x in args.watch_gen.split(",") if x]
    watch_pos = [P0 + g - 1 for g in watch_gen]  # the row whose logits predict generated index g
    print(f"prompt {P0} + gen {args.gen} = {T} tokens; watch positions {dict(zip(watch_gen, watch_pos))}", flush=True)

    cfg = AutoConfig.from_pretrained(args.ckpt)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    E = int(tc.num_local_experts)
    torch.set_default_dtype(DTYPE)
    with torch.device("meta"):
        model = Glm5NextTextModel(tc)
    torch.set_default_dtype(torch.float32)
    model.eval()
    ck = Ckpt(args.ckpt)
    nb = sum(1 for _ in model.buffers())
    print(f"model built on meta: {len(list(model.parameters()))} params, {nb} buffers", flush=True)
    for name, buf in model.named_buffers():
        print("  buffer", name, tuple(buf.shape), buf.dtype)

    # non-layer weights
    set_param(model, "embed_tokens.weight", ck.load("model.language_model.embed_tokens.weight"))
    set_param(model, "norm.weight", ck.load("model.language_model.norm.weight"))
    lm_head = ck.load("lm_head.weight")

    saved = {"ids": ids, "watch_pos": watch_pos, "layers": {}}
    keep_pos = sorted(set(watch_pos + [T - 2, T - 1]))
    timing = {}

    def pre_hook(L):
        def f(module, args_, kwargs_):
            t0 = time.time()
            for name, p in list(module.named_parameters()):
                if p.device.type == "meta":
                    set_param(module, name, layer_tensor(ck, L, name, E))
            for name, b in list(module.named_buffers()):
                if b.device.type == "meta":
                    t = layer_tensor(ck, L, name, E)
                    if args.bias_dtype == "fp32" and name.endswith("e_score_correction_bias"):
                        t = ck.raw(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").to(dev).float()
                    set_param(module, name, t, buffer=True)
            torch.cuda.synchronize()
            timing[L] = time.time() - t0
        return f

    def post_hook(L):
        def f(module, args_, kwargs_, out):
            hs = out[0] if isinstance(out, tuple) else out  # [1, T, hc, hidden]
            saved["layers"][L] = hs[0, keep_pos].detach().to("cpu")
            for name, p in list(module.named_parameters()):
                set_param(module, name, torch.empty(p.shape, device="meta", dtype=p.dtype))
            for name, b in list(module.named_buffers()):
                set_param(module, name, torch.empty(b.shape, device="meta", dtype=b.dtype), buffer=True)
            ck.close()
            gc.collect()
            torch.cuda.empty_cache()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"layer {L:2d} done: load {timing.get(L, 0):.1f}s  cuda alloc {torch.cuda.memory_allocated() / 1e9:.1f} GB "
                  f"reserved {torch.cuda.memory_reserved() / 1e9:.1f} GB  host maxrss {rss:.1f} GB", flush=True)
        return f

    for L, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(pre_hook(L), with_kwargs=True)
        layer.register_forward_hook(post_hook(L), with_kwargs=True)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        h = out.last_hidden_state[0].float()  # [T, hidden], normed
        logits = F.linear(h, lm_head.float())
        logp = torch.log_softmax(logits, dim=-1)
    print(f"forward done in {time.time() - t0:.0f}s", flush=True)
    saved["final_hidden"] = h[keep_pos].to("cpu")
    saved["keep_pos"] = keep_pos
    torch.save(saved, os.path.join(args.out, "hf_layers.pt"))

    top = torch.topk(logp, 5, dim=-1)
    res = {}
    for g, p in zip(watch_gen, watch_pos):
        row = [(tok.decode([int(i)]), int(i), round(float(v), 4)) for v, i in zip(top.values[p], top.indices[p])]
        res[str(g)] = {"pos": p, "top5": row, "margin": round(float(top.values[p, 0] - top.values[p, 1]), 4)}
        print(f"gen {g} (pos {p}): " + "  ".join(f"{s!r} {v}" for s, i, v in row) + f"   margin {res[str(g)]['margin']}", flush=True)
    json.dump(res, open(os.path.join(args.out, "hf_top5.json"), "w"), indent=1)

    allrows = []
    for g in range(1, args.gen + 1):
        p = P0 + g - 1
        actual = ids[p + 1] if p + 1 < T else None
        row = {"gen": g, "pos": p,
               "top5": [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values[p], top.indices[p])],
               "actual": (tok.decode([actual]), round(float(logp[p, actual]), 4)) if actual is not None else None,
               "margin": round(float(top.values[p, 0] - top.values[p, 1]), 4)}
        allrows.append(row)
    json.dump(allrows, open(os.path.join(args.out, "hf_top5_all.json"), "w"))
    thin = [(r["gen"], r["margin"], r["top5"][0][0]) for r in allrows if r["margin"] < 2.0]
    wrong = [(r["gen"], r["top5"][0][0], r["actual"]) for r in allrows if r["actual"] and r["top5"][0][0] != r["actual"][0]]
    print(f"HF margins over {len(allrows)} generated positions: thin(<2 nats) {thin[:30]}", flush=True)
    print(f"HF argmax != token vLLM produced at: {wrong[:30]}", flush=True)


if __name__ == "__main__":
    main()
