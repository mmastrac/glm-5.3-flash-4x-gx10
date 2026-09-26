#!/usr/bin/env python3
"""Size the kpool tail ring for speculative decoding (GLM-5.3-Flash).

A spec-verify step stashes 1 + num_spec rows into each request's kpool tail
ring before acceptance is known. The ring held exactly index_kpool (4) slots,
so when a pool-completing draft was rejected, the drafts behind it had already
overwritten that pool's committed keys, and the redo compressed the pool from
them. The indexer then keeps a wrong key for that pool, which changes pool
selection once the context exceeds index_topk (2048). MTP stashes 4 rows and
DFlash2 8, so both hit it. Measured on the kernel: 106 of 128 key bytes wrong.

The ring now holds kpool * cdiv(kpool + num_spec, kpool) slots (12 for DFlash2
k=7), and the seed and decode kernels address it by pos % RING. The tail slot
mapping takes the spec's block_size and follows. Upstream: the ring-sizing part
of vllm-project/vllm#55219, extracted as github.com/mmastrac/vllm branch
kpool-tail-ring-spec-slots. Drop this patch when that lands.

Same contract as the other patchers: every anchor matches the expected number
of times, or the build fails.
"""
from pathlib import Path

V = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next")
MARK = "GLM53-KPOOL-TAIL-RING"


def patch(path, edits):
    text = path.read_text()
    if MARK in text:
        print(f"[kpool-tail-ring] {path.name}: already patched")
        return
    for old, new, count in edits:
        n = text.count(old)
        assert n == count, f"[kpool-tail-ring] {path}: anchor matched {n}x, want {count}: {old[:70]!r}"
        text = text.replace(old, new)
    path.write_text(f"# {MARK}\n" + text)
    print(f"[kpool-tail-ring] {path.name}: patched")


# The spec: ring capacity, so block_size (and with it the tail slot mapping)
# covers the open pool plus a speculative step's rows.
patch(V / "common/attention.py", [
    ("from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec\n",
     "from vllm.utils.math_utils import cdiv\n"
     "from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec\n", 1),
    ("""        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self._index_kpool,
        )""",
     """        # The open pool's committed keys plus a speculative step's rows, in
        # whole pools: a rejected pool-completing draft must not leave the
        # drafts behind it overwriting the committed keys its redo reads.
        span = self._index_kpool + vllm_config.num_speculative_tokens
        ring = self._index_kpool * cdiv(span, self._index_kpool)
        return KpoolTailSpec(
            block_size=ring,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=ring,
        )""", 1),
])

# The kernels: address the ring by pos % RING. NVIDIA and AMD differ only in
# how the seed kernel spells its score store.
for vendor, seed_base, seed_score in (
    ("nvidia",
     ("    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n",
      "    base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM\n", 1),
     None),
    ("amd",
     ("    base = block_base + (t % KPOOL) * HEAD_DIM\n",
      "    base = block_base + (t % RING) * HEAD_DIM\n", 1),
     ("        tail_ptr + block_base + KPOOL_HEAD + (t % KPOOL) * HEAD_DIM + offs, s, mask=m\n",
      "        tail_ptr + block_base + KPOOL_HEAD + (t % RING) * HEAD_DIM + offs, s, mask=m\n", 1)),
):
    edits = [
        # seed kernel
        ("    KPOOL: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n",
         "    KPOOL: tl.constexpr,\n    RING: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n", 1),
        ("    blk = t // KPOOL  # t >= 0 here, so trunc == floor\n",
         "    blk = t // RING  # t >= 0 here, so trunc == floor\n", 1),
        ("    if ahead >= 0 and ahead // KPOOL == blk:\n",
         "    if ahead >= 0 and ahead // RING == blk:\n", 1),
        seed_base,
        ("        KPOOL=kpool,\n        BLOCK_D=triton.next_power_of_2(head_dim),\n",
         "        KPOOL=kpool,\n        RING=tail_kv_cache.shape[2],\n"
         "        BLOCK_D=triton.next_power_of_2(head_dim),\n", 1),
        # decode kernel
        ("    POOL_SIZE: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n",
         "    POOL_SIZE: tl.constexpr,\n    RING: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n", 1),
        ("        phys_slot = safe_pos % POOL_SIZE\n",
         "        phys_slot = safe_pos % RING  # RING >= POOL_SIZE\n", 1),
        ("        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE\n",
         "        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING\n", 1),
        ("                phys = (pool_logical_start + pool_slot) % POOL_SIZE\n",
         "                phys = (pool_logical_start + pool_slot) % RING\n", 2),
        ("    assert tail_kv_cache.shape[2] == pool_size\n",
         "    ring = tail_kv_cache.shape[2]\n"
         "    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)\n", 1),
        ("        POOL_SIZE=pool_size,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n",
         "        POOL_SIZE=pool_size,\n        RING=ring,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n", 1),
    ]
    if seed_score:
        edits.append(seed_score)
    patch(V / vendor / "ops/kpool_compress.py", edits)
