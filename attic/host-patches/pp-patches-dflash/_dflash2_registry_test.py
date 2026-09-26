import os
from vllm.model_executor.models.registry import ModelRegistry
info = ModelRegistry._try_inspect_model_cls("DFlash2DraftModel")
print("inspect (subprocess import):", info is not None and info.architecture)
cls = ModelRegistry._try_load_model_cls("DFlash2DraftModel")
print("resolve:", cls.__module__, cls.__name__)
import vllm.v1.worker.gpu.spec_decode as sd
import vllm.v1.worker.gpu.spec_decode.dflash.dflash2_speculator as s2
print("dispatch import ok:", s2.DFlash2Speculator.__mro__[1].__name__, "_generate_draft overridden:", "_generate_draft" in s2.DFlash2Speculator.__dict__)
# the config the drafter would carry after speculative.py's EAGLEConfig wrap keeps the arch name
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from transformers import AutoConfig
hf = AutoConfig.from_pretrained("/models/glm-5.3-flash-dflash2"); hf.architectures = ["DFlash2DraftModel"]
e = EAGLEConfig(hf, method="dflash")
print("EAGLEConfig architectures:", e.architectures, "dflash_config keys:", sorted(e.dflash_config), "is_causal:", getattr(e, "is_causal", None))
