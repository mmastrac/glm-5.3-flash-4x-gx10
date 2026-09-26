# The dflash mounts, ported to the vLLM-main base

`../pp-patches-dflash/` forks the old vendor base. These are the same changes
against `glm53-spark:main-v1` (vLLM main at `385dce36`), which is what blocked
the rebase.

`kv_cache_utils.py` is ported and imports cleanly in that image, coordinator
included -- the `dcp_world_size_for_kv_cache_spec` ImportError that crash-looped
was the old fork shadowing a file main's `kv_cache_coordinator` imports from.
`port-kv-cache-utils.py` regenerates it from the image's own copy, asserting each
anchor is unique, so the next base bump is a re-run rather than a re-derivation.

Two attribute renames upstream are why a blind `patch` could not work:
`compress_ratio` became `tokens_per_state`, and `KVCacheTensor(size, shared_by)`
became `KVCacheTensor(size, layers, layer_stride, block_stride, offset)` -- one
pool addressed by offset rather than a tensor per group. The draft regions are
laid after the indexer regions.

## Still to port

| file | hunks left | note |
|---|---|---|
| `spec_decode_init.py` | 1 | routes a `DFlash2DraftModel` drafter to our speculator; needed |
| `dflash_utils.py` | 2 | imports and the DFlash2 weight loading; needed |
| `model_runner.py` | 2 | both are `DFLASH-PP-PATCH`, lifting a PP guard. Check whether PP=1 needs them at all -- `pp-mtp-override.yaml` was dropped for exactly this reason |

Redundant on this base, so drop the mount rather than port it: `registry.py`
(main carries the DFlash2 entry) and `qwen3_dflash2.py` (main ships its own,
290 lines against our 554 -- diff before dropping). `model.py` should become
`patches/glm53_eagle3_aux.py`, 42 anchored lines instead of a 351-line fork.
