# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafter (architecture "DFlash2DraftModel").

DFlash2 = the DFlash block-diffusion backbone (qwen3_dflash.py) plus:

* a two-tap grouped dynamic causal convolution wrapped around every attention
  and MLP sub-layer (``layers.N.attention_conv.*``, ``layers.N.mlp_conv.*``),
  applied within one draft block (anchor + N mask tokens);
* a candidate path selector (``candidate_selector.*``) that takes the top-k
  lm_head candidates at every mask slot and walks one coherent path through
  them with a bilinear predecessor/successor codebook score.

Reference: github.com/z-lab/dflash ``dflash/model.py`` (``GroupedDynamicCausalConv``,
``CandidateSelector``, ``DFlash2DraftModel``) and SGLang
``python/sglang/srt/models/dflash.py`` (``DFlashGroupedConv``, ``CandidateSelector``).
Both were read line by line; the math here is theirs, the plumbing is vLLM's.

Only the greedy selector walk is implemented (``draft_sample_method="greedy"``,
the default). Under that setting the rejection sampler treats the draft as a
point mass (``rejection_sampler_utils.py``: ``HAS_DRAFT_LOGITS`` False ->
``draft_log_prob = 0``), so the output distribution is exact for any target
temperature; the selector's own sampling mode only changes acceptance length,
not correctness, and is left out.

This module never calls the target model; it consumes the same
``last_hidden_states`` / ``aux_hidden_states`` the DFlash speculator already
provides. The parent classes are reused wherever their code is unchanged.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _get_dflash_fc_input_size,
)
from vllm.model_executor.models.utils import get_draft_quant_config, maybe_prefix

logger = init_logger(__name__)


def dflash2_config(config) -> dict:
    return getattr(config, "dflash_config", None) or {}


def normalize_dflash2_causality(config) -> None:
    """Fold an explicit top-level ``is_causal`` into ``dflash_config.causal``.

    qwen3_dflash._dflash_layer_causal only reads ``dflash_config.causal`` and,
    failing that, treats every ``sliding_attention`` layer as causal. The
    reference implementations (z-lab ``Qwen3DFlashAttention.__init__``, SGLang
    ``_get_dflash_attention_type``) let an explicit ``is_causal`` override that
    default. GLM-5.3-Flash-DFlash2 ships ``is_causal: false`` with five
    ``sliding_attention`` layers: it is a NON-causal block drafter with a
    symmetric 2048-token window, and without this normalisation vLLM would run
    it causal inside the block. Idempotent; the dict is shared by every
    reader (speculator backend choice, attention metadata, layer build), so
    call it before the speculator's __init__ reads it.
    """
    dc = getattr(config, "dflash_config", None)
    if not isinstance(dc, dict) or "causal" in dc:
        return
    is_causal = getattr(config, "is_causal", None)
    if is_causal is None:
        return
    dc["causal"] = bool(is_causal)


class DFlash2GroupedConv(nn.Module):
    """Grouped dynamic causal K-tap convolution across one draft block.

    Checkpoint tensors: ``base_kernel`` [2, taps, hidden] (row 0 for the
    sub-layer input, row 1 for its output) and ``kernel_projection.weight``
    [2 * taps * groups, hidden]. ``prepare(x)`` projects the sub-layer input to
    per-token, per-group deltas for both sides, convolves the input with side
    0 and hands back the side-1 deltas for ``finish`` to convolve the output.

    out[t] = sum_tap (base[tap] + delta[t, tap, group]) * x[t - tap], with
    x[t - tap] = 0 when t - tap falls before the start of t's block. Tokens are
    the flat vLLM layout ``[num_reqs * block, hidden]``; block membership is
    ``t % block`` because the DFlash speculator lays every request out as
    exactly ``1 + num_speculative_tokens`` contiguous query tokens
    (dflash/speculator.py ``num_query_per_req``), padded requests included.
    Matches z-lab ``_grouped_dynamic_convolve`` (which shifts along an explicit
    [batch, block] axis) and SGLang ``_grouped_conv`` (same flat-position trick).
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"dflash_config.conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, self.taps, self.hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            self.hidden_size,
            2 * self.taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.kernel_projection",
            return_bias=False,
        )

    def _convolve(
        self, x: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        # x: [T, H]; delta: [T, taps, G]
        num_tokens = x.shape[0]
        blocks = x.view(num_tokens, self.num_groups, self.group_size)
        base = self.base_kernel[side].to(x.dtype).view(
            1, self.taps, self.num_groups, self.group_size
        )
        coeff = base + delta.to(x.dtype).unsqueeze(-1)  # [T, taps, G, gs]
        out = coeff[:, 0] * blocks
        if self.taps > 1:
            pos = torch.arange(num_tokens, device=x.device)
            if self.block_size & (self.block_size - 1) == 0:
                pos = pos & (self.block_size - 1)
            else:
                pos = pos % self.block_size
            for tap in range(1, self.taps):
                # x[t - tap], zero for the first `tap` rows of the flat buffer...
                shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
                # ...and zero again wherever t - tap crosses into the previous
                # request's block.
                in_block = (pos >= tap).to(x.dtype).view(-1, 1, 1)
                out = out + coeff[:, tap] * shifted * in_block
        return out.view(num_tokens, self.hidden_size)

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.kernel_projection(x).view(
            x.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(x, coeff[:, 0], side=0), coeff[:, 1]

    def finish(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self._convolve(x, delta, side=1)


class DFlash2CandidateSelector(nn.Module):
    """Top-k lattice over the mask slots and a greedy walk through it.

    score[b, e, p, c] = unary[b, e, c]
                      + < A[pred[b, e, p]] * P(h[b, e]), B[cand[b, e, c]] >

    where A/B are the predecessor/successor codebooks [vocab, rank], P is
    ``hidden_projection`` (hidden -> rank), cand[b, e, :] are the top-k
    candidates at slot e, and pred[b, e, p] is cand[b, e-1, p] for e > 0 and
    the verified anchor token (the bonus token at query offset 0) for e = 0.
    The codebooks are replicated on every TP rank: candidate ids are global.

    Greedy walk (z-lab ``CandidateSelector.select`` at temperature 0): slot 0
    takes argmax_c score[b, 0, anchor, c]; each later slot takes
    argmax_c score[b, e, chosen_{e-1}, c]. Building the whole [k, k] lattice
    first and then walking it with gathers is SGLang's formulation; it is
    equivalent and keeps every op shape-static for the FULL CUDA graph.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.top_k = int(top_k)
        # Stored in the checkpoint WITHOUT a ".weight" suffix
        # (candidate_selector.predecessor_codebook), so plain Parameters, not
        # embedding modules.
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, self.rank, dtype=params_dtype),
            requires_grad=False,
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, self.rank, dtype=params_dtype),
            requires_grad=False,
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            self.rank,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.hidden_projection",
            return_bias=False,
        )

    def build_lattice(
        self,
        hidden: torch.Tensor,  # [B, N, H]
        unary: torch.Tensor,  # [B, N, K] float32
        candidates: torch.Tensor,  # [B, N, K] int64
        anchor_ids: torch.Tensor,  # [B] int64
    ) -> torch.Tensor:  # [B, N, K, K] float32
        top_k = self.top_k
        proj = self.hidden_projection(hidden)  # [B, N, r]
        pred_ids = torch.cat(
            [anchor_ids[:, None, None].expand(-1, 1, top_k), candidates[:, :-1]],
            dim=1,
        )  # [B, N, K]
        pred = self.predecessor_codebook[pred_ids]  # [B, N, K, r]
        succ = self.successor_codebook[candidates]  # [B, N, K, r]
        pair = torch.einsum(
            "bnpr,bncr->bnpc",
            (pred * proj[:, :, None, :]).float(),
            succ.float(),
        )
        return unary[:, :, None, :] + pair

    def greedy_path(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidates: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> torch.Tensor:  # [B, N] int64 token ids
        scores = self.build_lattice(hidden, unary, candidates, anchor_ids)
        num_slots = scores.shape[1]
        # Slot 0: every predecessor row is the anchor, so row 0 is the row.
        index = scores[:, 0, 0].argmax(dim=-1)  # [B]
        maps = scores[:, 1:].argmax(dim=-1)  # [B, N-1, K]: best c per p
        path = [index]
        for edge in range(num_slots - 1):
            index = maps[:, edge].gather(-1, index[:, None])[:, 0]
            path.append(index)
        path_indices = torch.stack(path, dim=1)  # [B, N]
        return candidates.gather(-1, path_indices[..., None])[..., 0]


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer with the two dynamic convolutions wrapped around
    attention and MLP, in the reference order:

        h = input_layernorm(x [+ residual])
        h, k_attn = attention_conv.prepare(h)
        a = attention_conv.finish(self_attn(h), k_attn)
        h, residual = post_attention_layernorm(a, residual)   # fused add+norm
        h, k_mlp = mlp_conv.prepare(h)
        out = mlp_conv.finish(mlp(h), k_mlp)                  # added to residual
                                                              # by the next norm
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        block_size: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        dc = dflash2_config(config)
        taps = int(dc["conv_kernel_size"])
        group_size = int(dc["conv_group_size"])
        params_dtype = vllm_config.model_config.dtype
        self.attention_conv = DFlash2GroupedConv(
            config.hidden_size,
            block_size,
            taps,
            group_size,
            params_dtype,
            prefix=f"{prefix}.attention_conv",
        )
        self.mlp_conv = DFlash2GroupedConv(
            config.hidden_size,
            block_size,
            taps,
            group_size,
            params_dtype,
            prefix=f"{prefix}.mlp_conv",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, attn_kernel)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


class DFlash2Qwen3Model(DFlashQwen3Model):
    """DFlashQwen3Model with DFlash2 layers and the candidate selector.

    Inherits the context-KV precompute (``_build_fused_kv_buffers``,
    ``precompute_and_store_context_kv``), ``embed_input_ids``, the weight
    mapper and ``load_weights`` unchanged. ``__init__`` is rewritten rather
    than extended because the parent's builds its own layers (each registers an
    Attention under a fixed prefix; building them twice is an error).

    The parent is decorated with ``@support_torch_compile``, whose ``__call__``
    returns ``self.forward(...)`` as soon as ``self.do_not_compile`` is true.
    This class sets that flag and skips the wrapper's ``__init__`` entirely:
    the drafter runs eager inside the speculator's FULL CUDA graph (DFlash
    never uses PIECEWISE graphs), which is functionally identical and one less
    thing to debug first. Re-enabling torch.compile for the drafter is a
    follow-up, not a prerequisite.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.do_not_compile = True

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.config = spec_config.draft_model_config.hf_config
        normalize_dflash2_causality(self.config)
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        params_dtype = vllm_config.model_config.dtype

        drafter_config = dict(getattr(self.config, "eagle_config", None) or {})
        drafter_config.update(dflash2_config(self.config))
        self.use_aux_hidden_state = drafter_config.get("use_aux_hidden_state", True)

        for key in ("conv_kernel_size", "conv_group_size", "selector_rank",
                    "selector_top_k", "block_size", "mask_token_id"):
            if key not in drafter_config:
                raise ValueError(
                    f"DFlash2DraftModel needs dflash_config.{key}; got "
                    f"dflash_config={dflash2_config(self.config)}"
                )

        # The convolutions and the lattice are defined over one draft block.
        # The speculator lays each request out as 1 + num_speculative_tokens
        # query tokens (anchor + masks); that MUST be the block the checkpoint
        # was trained with, or the causal taps see the wrong neighbours.
        checkpoint_block = int(drafter_config["block_size"])
        query_block = 1 + int(spec_config.num_speculative_tokens)
        if query_block > checkpoint_block:
            raise ValueError(
                f"DFlash2DraftModel: dflash_config.block_size={checkpoint_block} "
                f"but num_speculative_tokens={spec_config.num_speculative_tokens} "
                f"gives a {query_block}-token query block; the drafter was "
                f"trained on blocks of {checkpoint_block}, set "
                f"num_speculative_tokens<={checkpoint_block - 1}"
            )
        if query_block < checkpoint_block:
            logger.info(
                "DFlash2DraftModel: drafting %d-token blocks (anchor + %d masks) "
                "with a checkpoint trained on %d-token blocks; this is the "
                "reference's truncated last block, every op is defined for it, "
                "but acceptance at this size is unmeasured.",
                query_block,
                spec_config.num_speculative_tokens,
                checkpoint_block,
            )
        # The conv's causal period is the block the speculator actually lays
        # out (anchor + num_speculative_tokens), not the checkpoint's nominal
        # size: with k < block_size - 1 each request's block is shorter and
        # the taps must still stop at its start.
        self.block_size = query_block

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.mask_token_id = drafter_config.get("mask_token_id")
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList(
            [
                DFlash2Qwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    block_size=self.block_size,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            self.fc = ReplicatedLinear(
                input_size=_get_dflash_fc_input_size(vllm_config),
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=params_dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.candidate_selector = DFlash2CandidateSelector(
            self.config.hidden_size,
            self.config.vocab_size,
            rank=int(drafter_config["selector_rank"]),
            top_k=int(drafter_config["selector_top_k"]),
            params_dtype=params_dtype,
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """Registered as "DFlash2DraftModel". Same interface as the DFlash drafter
    plus ``propose_greedy``, which DFlash2Speculator calls instead of the
    per-slot argmax."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlash2Qwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

        dc = dflash2_config(self.config)
        self._output_multiplier = float(dc.get("output_multiplier", 1.0))
        softcap = dc.get("final_logit_softcapping")
        self._final_logit_softcapping = (
            float(softcap) if softcap is not None and float(softcap) > 0 else None
        )

    def _transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # z-lab DFlashDraftModel.compute_logits; identity for this checkpoint.
        logits = logits.float()
        if self._output_multiplier != 1.0:
            logits = logits * self._output_multiplier
        if self._final_logit_softcapping is not None:
            logits = torch.tanh(logits / self._final_logit_softcapping) * (
                self._final_logit_softcapping
            )
        return logits

    def propose_greedy(
        self,
        hidden_states: torch.Tensor,  # [num_reqs * N, H], normed mask-slot rows
        anchor_ids: torch.Tensor,  # [num_reqs] int64
        num_reqs: int,
        num_slots: int,
    ) -> torch.Tensor:  # [num_reqs, N] int64
        top_k = self.model.candidate_selector.top_k
        # Full-vocab logits through the (shared, TP-gathered) target lm_head,
        # exactly what the greedy DFlash path already pays for. A TP-local top-k
        # with an O(tp * k) all-gather (SGLang compute_candidates) is a later
        # optimisation.
        logits = self.compute_logits(hidden_states)
        unary, candidates = torch.topk(logits.float(), top_k, dim=-1)
        unary = self._transform_unary_logits(unary)
        hidden = hidden_states.view(num_reqs, num_slots, -1)
        return self.model.candidate_selector.greedy_path(
            hidden,
            unary.view(num_reqs, num_slots, top_k),
            candidates.view(num_reqs, num_slots, top_k).to(torch.int64),
            anchor_ids.to(torch.int64),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # The parent prefixes everything but lm_head with "model.", which is
        # where candidate_selector.* and layers.N.{attention,mlp}_conv.* live.
        return super().load_weights(weights)
