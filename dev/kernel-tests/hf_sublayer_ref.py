#!/usr/bin/env python3
"""HF's own sublayer modules on vLLM's dumped inputs: an implementation-
independent check of the token-local sublayers (mHC hyper-connection, dense
MLP, MoE with router) against what vLLM produced for the same rows.

For each layer in the dump:
  hc_pre   HF Glm5NextTextHyperConnection(layer params) on vLLM's residual
           streams (`L<i>.hc_fused.out0`, i.e. after the deferred post) ->
           (post, comb, collapsed); vs vLLM `hc_fused.out1/out2` and, after
           HF's input_layernorm, `hc_fused.out3`.
  hc_post  HF's combine `post*x + comb^T residual` on vLLM's fused-op inputs
           vs vLLM `hc_fused.out0` (and the ffn-site call `hc_fused@1`).
  mlp      HF Glm5NextTextMLP / Glm5NextTextMoE (dequantised bf16 weights)
           on vLLM `mlp.in0` vs vLLM `mlp.out0`.

    python3 hf_sublayer_ref.py --ckpt /ckpt --layers 0,1,2,3 /hit/vec-rank0-step470.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_ref  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
dev = torch.device("cuda")


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / (b.norm() + 1e-12))


def fill(module, ck, L, prefix, E):
    for name, p in list(module.named_parameters()):
        hf_ref.set_param(module, name, hf_ref.layer_tensor(ck, L, prefix + name, E))
    for name, b in list(module.named_buffers()):
        hf_ref.set_param(module, name, hf_ref.layer_tensor(ck, L, prefix + name, E), buffer=True)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("dump")
    args = ap.parse_args()
    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as M

    tc = AutoConfig.from_pretrained(args.ckpt).text_config
    E = int(tc.num_local_experts)
    n, hid = int(tc.hc_mult), int(tc.hidden_size)
    ck = hf_ref.Ckpt(args.ckpt)
    b = torch.load(args.dump, map_location="cpu")
    cps = b["cps"]
    nrows = len(b["rows"])
    print(f"{os.path.basename(args.dump)}: {nrows} rows, positions {b['rows'][0]['pos']}..{b['rows'][-1]['pos']}")
    torch.set_default_dtype(torch.bfloat16)
    for L in [int(x) for x in args.layers.split(",")]:
        with torch.no_grad():
            # --- hyper-connection, attention site: input = streams after the previous post ---
            for site, key, hcname, normname in (("attn", f"L{L}.hc_fused", "attn_hc.", "input_layernorm."),
                                                 ("ffn", f"L{L}.hc_fused@1", "ffn_hc.", "post_attention_layernorm.")):
                if f"{key}.out0" not in cps:
                    continue
                with torch.device("meta"):
                    hc = M.Glm5NextTextHyperConnection(tc)
                    ln = M.Glm5NextTextRMSNorm(hid, tc.rms_norm_eps)
                fill(hc, ck, L, hcname, E)
                fill(ln, ck, L, normname, E)
                # HF post-combine on vLLM's fused-op inputs vs vLLM out0 (residual streams)
                x = cps[f"{key}.in_x"].to(dev)
                res = cps[f"{key}.in_residual"].to(dev).view(nrows, n, hid)
                post_in = cps[f"{key}.in_post_layer_mix"].to(dev).view(nrows, n)
                comb_in = cps[f"{key}.in_comb_res_mix"].to(dev).view(nrows, n, n)
                streams = post_in.to(x.dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb_in.to(x.dtype).transpose(-1, -2), res)
                e_post = rel(streams, cps[f"{key}.out0"].to(dev).view(nrows, n, hid))
                # HF hyper-connection on vLLM's resulting streams vs vLLM's post/comb/x
                v_streams = cps[f"{key}.out0"].to(dev).view(1, nrows, n, hid)
                post, comb, collapsed = hc(v_streams)
                e_p = rel(post[0], cps[f"{key}.out1"].to(dev).view(nrows, n))
                e_c = rel(comb[0], cps[f"{key}.out2"].to(dev).view(nrows, n, n))
                e_x = rel(ln(collapsed[0]), cps[f"{key}.out3"].to(dev))
                print(f"L{L:<3d} hc {site:4s}: post-combine {e_post:.2e} | pre: post {e_p:.2e} comb {e_c:.2e} x(normed) {e_x:.2e}")
            # --- MLP / MoE on vLLM's mlp input ---
            key = f"L{L}.mlp"
            if f"{key}.in0" in cps:
                dense = L < int(tc.first_k_dense_replace)
                with torch.device("meta"):
                    mlp = M.Glm5NextTextMLP(tc) if dense else M.Glm5NextTextMoE(tc)
                fill(mlp, ck, L, "mlp.", E)
                xin = cps[f"{key}.in0"].to(dev)
                out = mlp(xin.view(1, nrows, hid) if not dense else xin)
                out = out.view(nrows, hid)
                e = rel(out, cps[f"{key}.out0"].to(dev))
                per_row = [(r["pos"], round(rel(out[k], cps[f"{key}.out0"][k].to(dev)), 3)) for k, r in enumerate(b["rows"])]
                worst = sorted(per_row, key=lambda t: -t[1])[:4]
                print(f"L{L:<3d} mlp {'dense' if dense else 'moe  '}: vLLM out vs HF {e:.2e} (worst rows {worst})")
                del mlp
                torch.cuda.empty_cache()
                ck.close()


if __name__ == "__main__":
    main()
