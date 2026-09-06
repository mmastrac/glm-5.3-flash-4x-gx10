#!/usr/bin/env python3
"""Cap each worker's share of the GB10's memory.

vLLM never calls set_per_process_memory_fraction, so nothing bounds a worker:
gpu_memory_utilization only sizes the KV pool, and every allocation outside it
(the sparse indexer scores chunk x context/kpool, and grows with the session)
takes host memory until the node stops forking and the watchdog resets it. On
unified memory there is no separate VRAM to run out of first, and the OOM
killer cannot reclaim a CUDA allocation.

A fraction here turns that into torch.OutOfMemoryError: one request fails and
the node keeps serving. Leave TORCH_MEM_FRACTION above gpu_memory_utilization,
far enough that weights, KV and a full prefill's scratch all fit under it, and
below 1.0 by whatever the host needs to stay alive.

spark_mem_trace carries the logic and reads its own env at device init, so
both the ceiling and the trace tune from .env without re-patching.
"""

from pathlib import Path

WORKER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
)
ANCHOR = "            torch.accelerator.set_device_index(self.device)\n"
MARKER = "# [spark-mem-cap]"
INSERT = f"""            {MARKER}
            import spark_mem_trace

            spark_mem_trace.arm(self.device)
"""


def main() -> None:
    text = WORKER.read_text()
    if MARKER in text:
        print("glm53: worker memory cap already patched")
        return
    if text.count(ANCHOR) != 1:
        raise SystemExit(
            f"glm53: expected exactly one device-init anchor in {WORKER.name}, "
            f"found {text.count(ANCHOR)} -- re-check the worker after a base bump"
        )
    WORKER.write_text(text.replace(ANCHOR, ANCHOR + INSERT, 1))
    cache = WORKER.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob("gpu_worker*.pyc"):
            pyc.unlink(missing_ok=True)
    print("glm53: patched worker memory cap (TORCH_MEM_FRACTION arms it)")


if __name__ == "__main__":
    main()
