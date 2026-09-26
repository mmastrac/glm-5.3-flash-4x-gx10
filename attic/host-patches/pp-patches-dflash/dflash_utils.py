# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


# DFLASH-PP-PATCH: under PP the drafter sits on the last stage, whose target
# has no embed_tokens (PPMissingLayer), and the sharing branch below is skipped.
# A DFlash checkpoint that ships no embed_tokens of its own (this one does not:
# 2.34 GB is 5 layers + fc + selector, no 1.27 GB embedding) is then left with
# an uninitialised VocabParallelEmbedding: skip_substrs=["embed_tokens"] in
# DFlashQwen3ForCausalLM.load_weights means nothing ever writes it. The drafts
# would be garbage with no error. Read the target's embedding straight out of
# the target checkpoint instead; VocabParallelEmbedding.weight_loader does the
# TP vocab sharding. Same idea as the PP-PATCH in glm5next/nvidia/mtp.py, but
# that drafter shares the target's safetensors stream and this one does not.
_EMBED_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
)


def _load_target_embed_from_checkpoint(
    draft_embed: nn.Module, target_model_config: ModelConfig
) -> None:
    from safetensors import safe_open

    model_dir = target_model_config.model
    if not os.path.isdir(model_dir):
        raise ValueError(
            "DFLASH-PP-PATCH: the DFlash drafter has no embed_tokens of its own "
            "and under PP the last stage's target has none to share; expected "
            f"a local target checkpoint directory to read it from, got "
            f"{model_dir!r}"
        )
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        key = next((k for k in _EMBED_KEYS if k in weight_map), None)
        if key is None:
            raise ValueError(
                f"DFLASH-PP-PATCH: no embed_tokens weight in {index_path} "
                f"(looked for {_EMBED_KEYS})"
            )
        shard = os.path.join(model_dir, weight_map[key])
    else:
        shard = os.path.join(model_dir, "model.safetensors")
        key = None
    with safe_open(shard, framework="pt", device="cpu") as f:
        if key is None:
            key = next((k for k in _EMBED_KEYS if k in f.keys()), None)
            if key is None:
                raise ValueError(
                    f"DFLASH-PP-PATCH: no embed_tokens weight in {shard}"
                )
        loaded = f.get_tensor(key)
    param = draft_embed.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded)
    logger.info(
        "DFLASH-PP-PATCH: loaded drafter embed_tokens from target checkpoint "
        "%s (%s, %s)", shard, key, tuple(loaded.shape),
    )


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed
    elif not getattr(dflash_model, "has_own_embed_tokens", False):
        # DFLASH-PP-PATCH: see _load_target_embed_from_checkpoint.
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if draft_embed is None:
            raise ValueError("DFLASH-PP-PATCH: drafter has no embed_tokens module")
        _load_target_embed_from_checkpoint(draft_embed, vllm_config.model_config)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
