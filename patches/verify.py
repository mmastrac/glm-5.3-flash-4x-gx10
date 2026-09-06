#!/usr/bin/env python3
"""Assert the SM121 patches actually landed, at BUILD time.

Reads the files as text rather than importing them. vllm.platforms.cuda pulls in
vllm._C_stable_libtorch, which needs libcuda.so.1 -- and the driver does not
exist during `docker build`, only at run time. An import-based check therefore
fails for a reason that has nothing to do with whether the patch applied.

Without these assertions the image builds, loads 181 GiB, and dies at KV cache
init on `pe_dim must be 64 for fp8_ds_mla`. That failure cost three full load
cycles before the cause was found, so it is worth catching in the build.
"""
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
fail = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fail.append(label)


cuda_py = (VLLM / "platforms/cuda.py").read_text()
key = "device_capability.major == 12"
has_branch = key in cuda_py
check("cuda.py has the capability-12 branch", has_branch)
if has_branch:
    seg = cuda_py.split(key, 1)[1][:600]
    check("SM90 sparse-MLA listed for capability 12",
          "FLASHINFER_MLA_SPARSE_SM90" in seg,
          "-- without it, GB10 gets only the packed SM120 path this checkpoint cannot use")

sm90 = (VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py").read_text()
check("SM90 backend accepts capability 12", "in (9, 12)" in sm90)
check("SM90 backend falls back to FA2 off Hopper", "fa2" in sm90,
      "-- FA3 is Hopper-only; FA2 is what makes this path portable to sm_12x")

reg = (VLLM / "model_executor/models/registry.py").read_text()
check("glm5_next registered", "Glm5NextForConditionalGeneration" in reg)

for pkg, want in (("flashinfer-python", "0.6.18"),
                  ("nvidia-nccl-cu13", "2.30.7"),
                  ("nvidia-cutlass-dsl", "4.6.2")):
    try:
        got = version(pkg)
    except PackageNotFoundError:
        check(f"{pkg} installed", False)
        continue
    check(f"{pkg} == {want}", got.startswith(want), f"(got {got})")

try:
    version("flashinfer-jit-cache")
    check("flashinfer-jit-cache removed", False,
          "-- present, so SM120 cubins could still be loaded")
except PackageNotFoundError:
    check("flashinfer-jit-cache removed", True)

if fail:
    print("\nFAILED:", ", ".join(fail), file=sys.stderr)
    sys.exit(1)
print("\npatched image verified")
