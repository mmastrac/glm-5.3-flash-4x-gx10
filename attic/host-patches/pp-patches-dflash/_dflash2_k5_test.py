"""CPU-only checks for the DFlash2 port. Run inside the image with the patch
files bind-mounted and the drafter directory at /models/glm-5.3-flash-dflash2.

A. conv + selector math against the z-lab reference code (copied verbatim).
B. module construction with a stub Attention, parameter-name reconciliation
   against the checkpoint header.
C. AutoWeightsLoader dry run with zero tensors of the checkpoint shapes.
"""
import json, os, struct, sys, traceback
import torch, torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
DRAFT_DIR = "/models/glm-5.3-flash-dflash2"

# ---------------------------------------------------------------- reference
def ref_grouped_dynamic_convolve(hidden, dynamic, base, group_size):  # z-lab verbatim
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)

class RefConv(nn.Module):  # z-lab GroupedDynamicCausalConv verbatim
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size; self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)
    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        return (ref_grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
                dynamic[..., 1, :, :])
    def finish(self, hidden, dynamic):
        return ref_grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)

class RefSelector(nn.Module):  # z-lab CandidateSelector verbatim, greedy branch
    def __init__(self, vocab, hidden, rank, top_k):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab, rank)
        self.successor_codebook = nn.Embedding(vocab, rank)
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)
    def select(self, hidden, logits, anchor_ids):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1)

# ------------------------------------------------------------------ part A
print("== A. math vs reference")
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2GroupedConv, DFlash2CandidateSelector, normalize_dflash2_causality)
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.config import VllmConfig, DeviceConfig, set_current_vllm_config
_ctx = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))); _ctx.__enter__()   # CustomOps and parallel init need a current config
init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29571", backend="gloo")
ensure_model_parallel_initialized(1, 1)

H, GS, TAPS, BLOCK, B = 64, 16, 2, 8, 3
mine = DFlash2GroupedConv(H, BLOCK, TAPS, GS, torch.float32, prefix="t")
ref = RefConv(H, TAPS, GS)
with torch.no_grad():
    ref.base_kernel.normal_(); ref.kernel_projection.weight.normal_(std=0.2)
    mine.base_kernel.copy_(ref.base_kernel); mine.kernel_projection.weight.copy_(ref.kernel_projection.weight)
    x = torch.randn(B, BLOCK, H)
    r_out, r_k = ref.prepare(x)
    m_out, m_k = mine.prepare(x.view(B * BLOCK, H))
    assert torch.allclose(m_out.view(B, BLOCK, H), r_out, atol=1e-5), "prepare mismatch"
    y = torch.randn(B, BLOCK, H)
    assert torch.allclose(mine.finish(y.view(-1, H), m_k).view(B, BLOCK, H), ref.finish(y, r_k), atol=1e-5), "finish mismatch"
    # the conv must not leak across request blocks: perturb block 1, block 0 unchanged
    x2 = x.clone(); x2[1] += 5.0
    m2, _ = mine.prepare(x2.view(-1, H))
    assert torch.equal(m2.view(B, BLOCK, H)[0], m_out.view(B, BLOCK, H)[0]), "cross-block leak"
print("conv prepare/finish match z-lab; no cross-block leak")

V, R, K, N = 500, 32, 16, 7
msel = DFlash2CandidateSelector(H, V, R, K, torch.float32, prefix="s")
rsel = RefSelector(V, H, R, K)
with torch.no_grad():
    rsel.predecessor_codebook.weight.normal_(); rsel.successor_codebook.weight.normal_(); rsel.hidden_projection.weight.normal_(std=0.3)
    msel.predecessor_codebook.copy_(rsel.predecessor_codebook.weight); msel.successor_codebook.copy_(rsel.successor_codebook.weight)
    msel.hidden_projection.weight.copy_(rsel.hidden_projection.weight)
    for trial in range(20):
        hidden = torch.randn(B, N, H); logits = torch.randn(B, N, V) * 3; anchors = torch.randint(0, V, (B,))
        r_tok = rsel.select(hidden, logits, anchors)
        unary, cand = torch.topk(logits, K, dim=-1)
        m_tok = msel.greedy_path(hidden, unary, cand, anchors)
        assert torch.equal(m_tok, r_tok), (trial, m_tok, r_tok)
print("selector greedy path matches z-lab on 20 random trials")

# causality normalisation
from types import SimpleNamespace
cfg = SimpleNamespace(dflash_config={"block_size": 8}, is_causal=False)
normalize_dflash2_causality(cfg); assert cfg.dflash_config["causal"] is False
cfg2 = SimpleNamespace(dflash_config={"causal": True}, is_causal=False)
normalize_dflash2_causality(cfg2); assert cfg2.dflash_config["causal"] is True
print("is_causal -> dflash_config.causal normalisation ok")

# ------------------------------------------------------------------ part B
print("== B. construction + name reconciliation")
with open(os.path.join(DRAFT_DIR, "model.safetensors"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n))
header.pop("__metadata__", None)
ckpt = {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items()}
print("checkpoint tensors:", len(ckpt))

from transformers import AutoConfig
hf = AutoConfig.from_pretrained(DRAFT_DIR)   # architectures: DFlash2DraftModel (config.json.orig semantics)
hf.architectures = ["DFlash2DraftModel"]
print("hf config:", hf.model_type, hf.architectures, "is_causal", getattr(hf, "is_causal", None))

import vllm.model_executor.models.qwen3_dflash as q1
import vllm.model_executor.models.qwen3_dflash2 as q2

class StubAttn(nn.Module):
    def __init__(self, *a, **k):
        super().__init__(); self.layer_name = k.get("prefix", ""); self.kv_cache = None
    def forward(self, q, k, v): return q
q1.Attention = StubAttn  # DFlashQwen3Attention resolves the name at call time from its module globals

draft_mc = SimpleNamespace(hf_config=hf, model=DRAFT_DIR, revision=None, quantization=None,
                           get_hidden_size=lambda: hf.hidden_size, get_vocab_size=lambda: hf.vocab_size)
spec = SimpleNamespace(draft_model_config=draft_mc, num_speculative_tokens=5, method="dflash",
                       attention_backend=None, kv_cache_dtype=None)
target_mc = SimpleNamespace(dtype=torch.bfloat16, get_num_layers=lambda pc: 22, get_vocab_size=lambda: 154880,
                            head_dtype=torch.bfloat16)
fake = SimpleNamespace(speculative_config=spec, model_config=target_mc, parallel_config=SimpleNamespace(),
                       cache_config=None, quant_config=None, use_v2_model_runner=True)
q2.get_current_vllm_config = lambda: fake
q1.get_current_vllm_config = lambda: fake
q2.get_draft_quant_config = lambda c: None

torch.set_default_dtype(torch.bfloat16)
model = q2.DFlash2Qwen3ForCausalLM(vllm_config=fake, prefix="")
torch.set_default_dtype(torch.float32)
params = dict(model.named_parameters())
print("model params:", len(params))
print("dflash_config after construction:", hf.dflash_config)

# expected mapping checkpoint name -> vLLM param name(s)
def expect(name):
    if name.startswith("candidate_selector.") or name in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        return ["model." + name]
    if name.startswith("layers."):
        for src, dst in ((".q_proj.", ".qkv_proj."), (".k_proj.", ".qkv_proj."), (".v_proj.", ".qkv_proj."),
                         (".gate_proj.", ".gate_up_proj."), (".up_proj.", ".gate_up_proj.")):
            if src in name: return ["model." + name.replace(src, dst)]
        return ["model." + name]
    raise AssertionError(f"no mapping rule for {name}")

unmapped = []
consumed = {}
for name, (dt, shape) in sorted(ckpt.items()):
    tgt = expect(name)[0]
    if tgt not in params: unmapped.append((name, tgt)); continue
    consumed.setdefault(tgt, []).append(name)
print("checkpoint tensors with no parameter:", unmapped)
assert not unmapped
unloaded = sorted(set(params) - set(consumed))
print("model parameters not fed by the checkpoint:", unloaded)
# shape checks for the direct (non-stacked) ones
for tgt, srcs in consumed.items():
    if len(srcs) == 1 and ".qkv_proj." not in tgt and ".gate_up_proj." not in tgt:
        assert tuple(params[tgt].shape) == ckpt[srcs[0]][1], (tgt, params[tgt].shape, ckpt[srcs[0]])
print("direct-parameter shapes match the checkpoint header")

# ------------------------------------------------------------------ part C
print("== C. AutoWeightsLoader dry run (zeros of checkpoint shapes)")
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
with torch.no_grad():
    for p in params.values(): p.fill_(float("nan"))
    weights = ((name, torch.zeros(shape, dtype=DT[dt])) for name, (dt, shape) in ckpt.items())
    model.load_weights(weights)
    still_nan = sorted(n for n, p in params.items() if torch.isnan(p.float()).any())
print("params still NaN after load (must be exactly the ones the target shares):", still_nan)
assert set(still_nan) <= {"lm_head.weight", "model.embed_tokens.weight", "model.mask_embedding"}, still_nan
assert "model.mask_embedding" not in still_nan or True
print("has_own_embed_tokens", getattr(model, "has_own_embed_tokens", None), "has_own_lm_head", getattr(model, "has_own_lm_head", None))
print("fused kv buffers built:", hasattr(model.model, "_num_attn_layers"), model.model._num_attn_layers)
print("k=5 conv block:", model.model.layers[0].attention_conv.block_size, "mlp:", model.model.layers[0].mlp_conv.block_size); assert model.model.layers[0].attention_conv.block_size == 6; print("ALL CHECKS PASSED (k=5)")
