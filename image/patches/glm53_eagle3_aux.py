#!/usr/bin/env python3
"""Give GLM-5.3-Flash the EAGLE3 interface its DFlash2 drafter requires.

vLLM's DFlash2 asks the target model for auxiliary hidden states, and refuses to
start without them: `set_eagle3_aux_hidden_state_layers` raises "Model does not
support EAGLE3 interface" (v1/worker/gpu/spec_decode/eagle/eagle3_utils.py).
Upstream's `glm5next` model carries no aux-hidden-state code, so this adds it.

`EagleModelMixin` supplies the bookkeeping. What it cannot supply is what an
aux hidden state *is* for this architecture. Under mHC a non-final layer never
materialises its output: it returns (x, residual, post, comb) and defers hc_post
to the next layer's fused post+pre kernel. So the mixin's default of
`hidden_states + residual` is the wrong tensor here, and picking it would not
crash -- it would quietly cost draft acceptance. For an aux layer we apply that
hc_post standalone, the same kernel the final layer and the PP boundary use,
then contract the streams with hc_contract. That matches the DFlash reference
capture for this model (sglang PR #36708) and DeepSeek V4's dspark capture.

Aux ids are 1-based: eagle3_utils shifts the drafter's target_layer_ids by +1,
so id L is the completed output of layer L-1.

Pipeline parallelism is not handled. PP is off for this model, and the mixin's
own pack/collect helpers cover it if that ever changes.
"""

from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MODEL = VLLM / "models/glm5next/common/model.py"
LOG = "[glm53-eagle3]"

EDITS: list[tuple[str, str, str]] = [
    (
        "interfaces import",
        """from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)""",
        """from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)""",
    ),
    (
        "Glm5NextModel mixin",
        "class Glm5NextModel(nn.Module):",
        "class Glm5NextModel(nn.Module, EagleModelMixin):",
    ),
    (
        "aux layer loop",
        """        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
""",
        """        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux = self._completed_layer_output(
                    layer, hidden_states, residual, post, comb
                )
                if self.is_sequence_parallel:
                    aux = sp_all_gather(aux)[:full_num_tokens]
                aux_hidden_states.append(aux)
""",
    ),
    (
        "aux return",
        """        hidden_states = self.norm(hidden_states)
        return hidden_states
""",
        """        hidden_states = self.norm(hidden_states)
        if self.aux_hidden_state_layers:
            # The runner unpacks a tuple whenever use_aux_hidden_state_outputs
            # is set, which is exactly when the layers were configured.
            return hidden_states, aux_hidden_states
        return hidden_states
""",
    ),
    (
        "ForCausalLM SupportsEagle3",
        """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):""",
        """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsEagle3, MixtureOfExperts, IsHybrid
):""",
    ),
    (
        "ForConditionalGeneration SupportsEagle3",
        """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts
):""",
        """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts,
    SupportsEagle3
):""",
    ),
]

COMPLETED = '''    def _completed_layer_output(
        self,
        layer: Glm5NextDecoderLayer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> torch.Tensor:
        """The tensor an aux consumer expects: this layer's completed output.

        `post is None` means there is nothing deferred -- a non-mHC layer, whose
        hidden_states already carries residual + mlp, or the final mHC layer,
        already hc_post'ed and contracted.
        """
        if post is None:
            return hidden_states
        recon = layer.hc_post(hidden_states, residual, post, comb)
        return hc_contract(recon, layer.n)

'''

FORWARD_ANCHOR = """    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:"""


def apply(text: str, label: str, old: str, new: str) -> str:
    n_old, n_new = text.count(old), text.count(new)
    if n_old == 0 and n_new >= 1:
        print("%s %s: skipped" % (LOG, label))
        return text
    if n_old != 1:
        raise SystemExit(
            "%s refuse %s (old=%d new=%d); the stock tree moved" % (LOG, label, n_old, n_new)
        )
    print("%s %s: applied" % (LOG, label))
    return text.replace(old, new, 1)


def main() -> None:
    text = MODEL.read_text()
    for label, old, new in EDITS:
        text = apply(text, label, old, new)
    # Insert the helper just above the forward it serves.
    if "def _completed_layer_output" not in text:
        if text.count(FORWARD_ANCHOR) != 1:
            raise SystemExit(
                "%s refuse helper insert: %d forward anchors"
                % (LOG, text.count(FORWARD_ANCHOR))
            )
        text = text.replace(FORWARD_ANCHOR, COMPLETED + FORWARD_ANCHOR, 1)
        print("%s completed-output helper: applied" % LOG)
    else:
        print("%s completed-output helper: skipped" % LOG)
    MODEL.write_text(text)


if __name__ == "__main__":
    main()
