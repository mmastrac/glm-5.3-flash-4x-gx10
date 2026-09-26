#!/usr/bin/env python3
"""Assert the image carries everything glm53 depends on, AFTER the patches ran.

Each item here has failed silently on this model before -- the image booted,
loaded ~90 GiB per node over ten minutes, and then died or served nonsense.
Failing the BUILD is far cheaper. Checks both what the nightly must provide
and that every patch actually landed.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import torch
from vllm.model_executor.models.registry import ModelRegistry
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

V = Path("/usr/local/lib/python3.12/dist-packages/vllm")
problems = []

archs = set(ModelRegistry.get_supported_archs())
if "Glm5NextForConditionalGeneration" not in archs:
    problems.append("Glm5NextForConditionalGeneration missing from the model registry")

try:
    ReasoningParserManager.get_reasoning_parser("glm45")
except Exception as e:
    problems.append(f"reasoning parser glm45 unavailable: {e!r}")

# The fail-closed tool parser is a plugin file, so load it the way
# --tool-parser-plugin will, then ask for it by name.
try:
    ToolParserManager.import_tool_parser("/usr/local/share/glm47_failclosed.py")
    ToolParserManager.get_tool_parser("glm47_failclosed")
except Exception as e:
    problems.append(f"glm47_failclosed plugin does not register: {e!r}")

if "sm_120" not in torch.cuda.get_arch_list():
    problems.append(f"no sm_120 cubins in torch {torch.__version__}")

# Patches landed. MiaAI's and the spin-wait patcher assert their own anchors;
# these re-read the result so a patcher that silently no-ops is caught too.
cuda_py = (V / "platforms/cuda.py").read_text()
if "FLASHINFER_MLA_SPARSE_SM90" not in cuda_py.split("device_capability.major == 12")[1][:600]:
    problems.append("capability 12 does not list FLASHINFER_MLA_SPARSE_SM90 (MiaAI patch)")
sm90 = (V / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py").read_text()
if "capability.major == 9\n" in sm90:
    problems.append("SM90 sparse backend still Hopper-only (MiaAI patch)")
idx = (V / "models/glm5next/nvidia/sparse_indexer.py").read_text()
if "pool_topk = torch.empty(" in idx:
    problems.append("indexer pool_topk still torch.empty (MiaAI indexer -1, retargeted)")
if "SM90-FP8-KV-DTYPE" not in sm90:
    problems.append("SM90 builder still plans fp8 KV as uint8 (sm90_fp8_kv_dtype.py)")
if "GLM53-DFLASH2-KV" not in (V / "v1/core/kv_cache_utils.py").read_text():
    problems.append("GLM-5-Next KV grouper cannot carry drafter layers (glm53_dflash2_kv_groups.py)")
if "GLM53-MTP-BF16" not in (V / "models/glm5next/common/mtp.py").read_text():
    problems.append("MTP bf16 exclusion missing (glm53_mtp_bf16.py)")
if "SupportsEagle3" not in (V / "models/glm5next/common/model.py").read_text():
    problems.append("GLM5next has no aux hidden states (glm53_eagle3_aux.py)")
for f in ("models/glm5next/common/attention.py", "models/glm5next/nvidia/ops/kpool_compress.py"):
    if "GLM53-KPOOL-TAIL-RING" not in (V / f).read_text():
        problems.append(f"kpool tail ring not sized for spec decode in {f} (glm53_kpool_tail_ring.py)")
if "GLM53-REASONING-ALWAYS" not in (V / "parser/glm47_moe.py").read_text():
    problems.append("reasoning parser drops <think> when thinking is off (glm53_reasoning_always_parsed.py)")
if "busy_loop_s: float = 0.002," not in (V / "distributed/device_communicators/shm_broadcast.py").read_text():
    problems.append("shm_broadcast busy_loop_s not 0.002 (spin_wait.py)")

# Multi-node TP needs the real mentat daemon behind the `ray` executable, not
# just the shim package: ranks are placed through it.
try:
    out = subprocess.run(["ray", "--version"], capture_output=True, text=True, timeout=30).stdout
    if "mentat" not in out.lower():
        problems.append(f"`ray` is not mentat: {out.strip()[:120]}")
except Exception as e:
    problems.append(f"`ray` executable missing: {e!r}")

# The shim package, whose ray.register module is how a TP=1 stack joins
# without the daemon binary. Missing, a container starts and serves and never
# appears in the router.
try:
    importlib.import_module("ray.register")
except Exception as e:
    problems.append(f"mentat ray.register not importable: {e!r}")

import shutil
if not shutil.which("mentatd-probe-machine"):
    problems.append("mentatd-probe-machine missing: `ray start` would register zero GPUs")

if problems:
    for p in problems:
        print(f"IMAGE CHECK FAILED: {p}", file=sys.stderr)
    sys.exit(1)
print("image ok: Glm5Next, glm45, glm47_failclosed, sm_120, MiaAI SM90 + indexer, spin-wait, mentat ray + ray.register")
