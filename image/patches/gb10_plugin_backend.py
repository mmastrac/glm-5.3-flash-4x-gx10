#!/usr/bin/env python3
"""Let VLLM_GLM53_CUDA_SPARSE_MLA choose between the two sm_121 MLA kernels.

There are two solutions to the same GB10 problem and they collide silently.

MiaAI-Lab's patcher gates vLLM's existing SM90 sparse-MLA path onto capability
12 by inserting it into the priority list ahead of SM120. LibertAI's plugin
supplies its own kernel and, by its own design note, "deliberately overrides the
existing FLASHINFER_MLA_SPARSE_SM120 enum slot ... so selection needs no
plumbing". Selection takes the first supported entry, so SM90 always wins and
the plugin sits unreachable in a slot nothing reaches.

This makes the SM90 entry conditional on the plugin's own gate, so one image
serves both: unset picks SM90 as before, set drops SM90 and the SM120 slot --
now the plugin -- is selected. Runs after the MiaAI patcher and rewrites what it
produced, so it fails loudly if that output ever changes shape.
"""
from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
LOG = "[gb10-plugin]"

OLD = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""

NEW = """        elif device_capability.major == 12:
            import os as _os

            return [
                AttentionBackendEnum.TRITON_MLA,
                *(
                    []
                    if _os.environ.get("VLLM_GLM53_CUDA_SPARSE_MLA")
                    else [AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90]
                ),
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""


def main() -> None:
    path = VLLM / "platforms/cuda.py"
    text = path.read_text()
    if text.count(NEW) == 1:
        print("%s skip (already in this image)" % LOG)
        return
    if text.count(OLD) != 1:
        raise SystemExit(
            "%s refuse: expected exactly one MiaAI capability-12 list, found %d"
            % (LOG, text.count(OLD))
        )
    path.write_text(text.replace(OLD, NEW, 1))
    print("%s SM90 is now conditional on VLLM_GLM53_CUDA_SPARSE_MLA" % LOG)


if __name__ == "__main__":
    main()
