#!/usr/bin/env python3
"""Plan the SM90 sparse-MLA wrapper with the fp8 dtype, not the storage dtype.

On vLLM main the SM90 FlashInfer sparse-MLA state is built by the metadata
BUILDER from kv_cache_spec.dtype. For an fp8 KV cache that is the storage
dtype, torch.uint8, and FlashInfer's MLA plan() rejects it during the first
autotune dummy run:

    ValueError: MLA kv_data_type torch.uint8 is not supported.
    Supported dtypes: [torch.float16, torch.bfloat16, torch.float8_e4m3fn].

The pre-merge image built the same state in the Impl and passed
torch.float8_e4m3fn explicitly when the cache was fp8, which is what this
restores. The forward path already views the cache as float8_e4m3fn, so plan
and run then agree. Any fp8 KV cache on this backend hits this, not only GB10.

One anchor, exactly once, or the build fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py")
OLD = """        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_cache_spec.dtype,
"""
NEW = """        self.state = _SM90State(
            device,
            impl.num_heads,
            # SM90-FP8-KV-DTYPE: fp8 caches are stored as uint8; plan with fp8.
            (torch.float8_e4m3fn if impl.use_fp8_kv_cache else kv_cache_spec.dtype),
"""
text = PATH.read_text()
if "SM90-FP8-KV-DTYPE" in text:
    print("[sm90-fp8-kv-dtype] already applied")
else:
    assert text.count(OLD) == 1, f"[sm90-fp8-kv-dtype] anchor matched {text.count(OLD)} times"
    PATH.write_text(text.replace(OLD, NEW))
    print("[sm90-fp8-kv-dtype] builder plans fp8 KV as float8_e4m3fn")
