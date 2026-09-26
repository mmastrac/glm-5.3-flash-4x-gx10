# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 speculator: the DFlash speculator with the per-slot argmax replaced
by the candidate path selector (see model_executor/models/qwen3_dflash2.py).

Everything else (query layout, context-KV precompute, attention metadata,
CUDA graphs, draft-token buffers) is the DFlash speculator's. The one
override, ``_generate_draft``, is what ``DFlashCudaGraphManager.capture``
captures, so it keeps the same argument list and stays shape-static in
``num_reqs`` and ``num_tokens_padded``.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash2 import normalize_dflash2_causality
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        # Before the parent reads dflash_has_any_non_causal(): see the
        # docstring of normalize_dflash2_causality.
        normalize_dflash2_causality(spec_config.draft_model_config.hf_config)
        super().__init__(vllm_config, device)
        if self.draft_logits is not None:
            raise NotImplementedError(
                "DFlash2 implements the greedy selector walk only; use "
                "draft_sample_method='greedy' (the default). The output "
                "distribution is exact either way."
            )
        if self.use_local_argmax_reduction:
            raise NotImplementedError(
                "DFlash2 needs full-vocab logits for its top-k candidates; "
                "use_local_argmax_reduction is not supported."
            )
        logger.info(
            "DFlash2 selector: top_k=%d, rank=%d, block=%d, causal=%s",
            self.model.model.candidate_selector.top_k
            if hasattr(self, "model") and self.model is not None
            else -1,
            self.model.model.candidate_selector.rank
            if hasattr(self, "model") and self.model is not None
            else -1,
            self.num_query_per_req,
            not self.requires_non_causal,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_slots = self.num_speculative_steps
        num_sample = num_reqs * num_slots
        # Rows of the N mask slots per request, in slot order
        # (prepare_dflash_inputs: sample_idx = req * N + (query_off - 1)).
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        # The anchor is the bonus token at query offset 0 of each request's
        # 1 + N block; prepare_dflash_inputs wrote it into input_ids.
        anchor_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req
        ].view(num_reqs, self.num_query_per_req)[:, 0]

        draft_tokens = self.model.propose_greedy(
            sample_hidden_states, anchor_ids, num_reqs, num_slots
        )
        self.draft_tokens[:num_reqs] = draft_tokens
