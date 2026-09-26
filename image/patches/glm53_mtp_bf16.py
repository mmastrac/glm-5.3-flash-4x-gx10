#!/usr/bin/env python3
"""Build GLM-5.3's MTP layer unquantised when the checkpoint stores it that way.

nvidia/GLM-5.3-Flash-NVFP4 keeps its MTP layer (layers.45) entirely in BF16 --
889 tensors, no weight_scale anywhere -- but its config.json does not say so:
quant_algo is NVFP4 for every Linear, and `ignore` lists 132 modules without
mentioning layer 45. vLLM follows config.json (modelopt_fp4), builds the MTP
experts as packed FP4, and the load dies in RoutedExperts._load_w2 with

    The size of tensor a (256) must match the size of tensor b (512)

-- 512 is a BF16 w2 shard at TP=4, 256 the packed FP4 one. hf_quant_config.json
(MIXED_PRECISION, whose quantized_layers omit layer 45) is right, but vLLM
does not read it for this checkpoint.

The fix follows the WEIGHTS rather than hard-coding a layer: if the checkpoint
index holds no *weight_scale for this MTP layer, its prefix is appended to the
quant config's exclude_modules as a ModelOpt wildcard before the layer is built.
Every sublayer then takes the ordinary unquantised path -- ModelOpt returns
UnquantizedLinearMethod for excluded Linears and no method for excluded
RoutedExperts. A checkpoint that does quantise its MTP layer is left alone.

One anchor, exactly once, or the build fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/mtp.py")

# The layer passes vllm_config to its decoder layer, which reads the shared
# quant config itself, so the exclusion goes onto that object before it is built.
OLD = """        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
"""
NEW = """        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        # GLM53-MTP-BF16: see image/patches/glm53_mtp_bf16.py in spark-glm53.
        if quant_config is not None and _glm53_mtp_layer_unquantized(
            vllm_config.speculative_config.draft_model_config.model, prefix
        ):
            _excl = getattr(quant_config, "exclude_modules", None)
            if isinstance(_excl, list) and f"{prefix}.*" not in _excl:
                _excl.append(f"{prefix}.*")
                import logging
                logging.getLogger(__name__).info(
                    "GLM53-MTP-BF16: %s has no quantisation scales in the "
                    "checkpoint; building it unquantised", prefix)
"""

HELPER_ANCHOR = "class Glm5NextMultiTokenPredictorLayer(nn.Module):\n"
HELPER = '''def _glm53_mtp_layer_unquantized(model_dir, prefix):
    """True when the checkpoint carries no quantisation scales for this MTP layer.

    GLM53-MTP-BF16. Reads model.safetensors.index.json; any failure to read it
    means "do not second-guess the config", so the answer is False.
    """
    import json
    import os
    try:
        idx = int(prefix.rsplit(".", 1)[-1])
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            keys = json.load(f)["weight_map"]
    except Exception:
        return False
    tag = f".layers.{idx}."
    layer_keys = [k for k in keys if tag in k]
    return bool(layer_keys) and not any(k.endswith("weight_scale") for k in layer_keys)


'''

text = PATH.read_text()
if "GLM53-MTP-BF16" in text:
    print("[glm53-mtp-bf16] already applied")
else:
    assert text.count(OLD) == 1, f"[glm53-mtp-bf16] constructor anchor matched {text.count(OLD)} times"
    assert text.count(HELPER_ANCHOR) == 1, "[glm53-mtp-bf16] class anchor not unique"
    text = text.replace(OLD, NEW).replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    PATH.write_text(text)
    print("[glm53-mtp-bf16] MTP layer excluded from quantisation when the checkpoint has no scales for it")
