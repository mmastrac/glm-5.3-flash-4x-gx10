# Vendored SM121 kernel patcher

`glm53-flash_SM121.py` is taken verbatim from
[MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark),
MIT licensed — the licence is kept beside it as `LICENSE.MiaAI-Lab`.

It is what makes GLM-5.3-Flash run on GB10 at all. The stock
`glm53-flash-arm64-cu130` image offers only `FLASHINFER_MLA_SPARSE_SM120` on
compute capability 12, and that backend mandates the packed `fp8_ds_mla` layout
with `pe_dim == 64`. This checkpoint is **NoPE** (`qk_rope_head_dim: 0`), so the
kernel refuses and the engine dies in `concat_and_cache_mla` — after a full
~10 minute weight load.

The seven patches gate paths that already exist onto GB10 rather than writing
new kernels. Two are the crux:

  - `bake_sm90_on_capability_12` lists `FLASHINFER_MLA_SPARSE_SM90` for
    capability 12, ahead of the SM120 entry.
  - `bake_sm90_wrapper_for_gb10` relaxes that backend's
    `capability.major == 9` to `in (9, 12)` **and** swaps FA3 for FA2 off
    Hopper.

That second half is the part worth internalising: the SM90 sparse-MLA backend
is not inherently Hopper-only. Its FA3 kernel is, but FA2 is portable to sm_12x.
An earlier reading of this tree concluded the SM90 path was unreachable on GB10
because of `wgmma`; that was wrong, and it cost a working configuration.

The rest handle FlashInfer's FA2 fp8 gate and CTA tile for GB10's ~101 KB smem,
turn PDL off (KDA race on this hardware), and fix indexer/kpool details.

The script asserts every replacement matches exactly once and aborts otherwise,
so a base-image change fails the build rather than silently producing a
half-patched tree. Drop it once these land upstream.
