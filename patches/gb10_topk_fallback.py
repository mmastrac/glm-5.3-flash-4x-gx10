#!/usr/bin/env python3
"""Route GB10 off persistent_topk, onto the fallback already in the tree.

Ours, not MiaAI-Lab's -- glm53-flash_SM121.py stays verbatim upstream, so this
lives beside it rather than inside it.

The sparse-MLA indexer picks persistent_topk whenever select_k is 512, 1024 or
2048, with no device gate. On GB10 that kernel refuses once the KV pool grows
past roughly 3.4M tokens:

    launch_persistent_topk, topk.cu:138, persistent_topk would oversubscribe
    and the FilteredTopK fallback requires >=128KB smem per block (have
    101376). total_ctas=62 > num_sms*occupancy=48

Two walls, and neither has a way through on this hardware: the persistent path
wants total_ctas <= 48, and its own FilteredTopK fallback wants 128 KB of shared
memory per block against the 101,376 B a GB10 SM offers. The trigger is pool
size, not context depth -- 197k tokens in one prefill is fine at a 2.0M pool,
while a 4M pool dies during startup profiling.

top_k_per_row_decode is the same computation without the persistent-CTA scheme,
sits in the else branch of both call sites already, and keeps CUDA graphs (the
torch.topk replacement other deployments use needs --enforce-eager and costs
~25%).

The predicate is the one vLLM already applies to cooperative_topk one line
above, so this makes persistent_topk agree with its neighbour rather than
inventing a new rule.
"""
from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
LOG = "[gb10-topk]"
GATE = " and not current_platform.is_device_capability_family(120)"


def apply_once(path: Path, old: str, new: str, label: str) -> str:
    text = path.read_text()
    n_old, n_new = text.count(old), text.count(new)
    if n_old == 0 and n_new == 1:
        return "skipped"
    if n_old != 1:
        raise SystemExit(
            "%s refuse %s (old=%d new=%d); stock tree changed" % (LOG, label, n_old, n_new)
        )
    path.write_text(text.replace(old, new, 1))
    return "applied"


def bake_indexer() -> str:
    path = VLLM / "model_executor/layers/sparse_attn_indexer.py"
    old = (
        "        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (\n"
        "            512,\n"
        "            1024,\n"
        "            2048,\n"
        "        )\n"
    )
    new = (
        "        use_persistent_topk = (\n"
        "            current_platform.is_cuda()\n"
        "            and topk_tokens in (512, 1024, 2048)\n"
        "            " + GATE.strip() + "\n"
        "        )\n"
    )
    return apply_once(path, old, new, "sparse_attn_indexer")


def bake_kpool() -> str:
    path = VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py"
    old = "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):\n"
    new = (
        "        if (\n"
        "            current_platform.is_cuda()\n"
        "            and select_k in (512, 1024, 2048)\n"
        "            " + GATE.strip() + "\n"
        "        ):\n"
    )
    return apply_once(path, old, new, "sparse_attn_indexer_kpool")


def main() -> None:
    results = [bake_indexer(), bake_kpool()]
    if all(r == "skipped" for r in results):
        print("%s skip (already in this image)" % LOG)
    else:
        print("%s persistent_topk gated off capability 12 (%s)" % (LOG, " ".join(results)))


if __name__ == "__main__":
    main()
