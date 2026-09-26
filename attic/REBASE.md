# Rebasing off the patch stack

State as of 2026-09-10. Three layers sit between upstream and what serves: the
vendor base image `vllm/vllm-openai:glm53-flash-arm64-cu130`, patch scripts
baked by the Dockerfile, and bind-mounted modules wired by `overrides/`.

## What is settled

`pp-mtp-override.yaml` is **removed**. Every file in it was PP-only or inert at
`pipeline_parallel_size=1`, and removing it left draft acceptance at 1.79 of 7
against a control of 1.75. See ../README.md.

The base tag was rebuilt 2026-09-09 onto `0.28.1rc1.dev580+g385dce36b`, which is
vLLM main at `385dce36`. Checked inside that image, it already carries the
thinking-budget guard, the kpool `positions=` fix, the kpool tail-slot mapping
with an `out=` buffer, the DFlash2 registry entry and `qwen3_dflash2.py`. So
`thinking-budget-override.yaml`, `kpool-fix-override.yaml`, and `registry.py`
and `qwen3_dflash2.py` from the dflash set all become redundant **the moment we
move to that base** -- not before. They are still mounted because production
still runs the old base, where those fixes are absent.

Still required against main: the seven SM121 steps (all still apply cleanly),
the GB10 topk/plugin/memory-cap patches, and the spin-wait one-liner
(`busy_loop_s` is still 1 upstream).

## What blocks the move

`glm53-spark:main-v1` is built and present on all four nodes. It boots as far as
KV cache grouping and then fails:

    NotImplementedError: Layer language_model.model.layers.3.self_attn.indexer.k_cache:
    page size is not divisible by the maximum page size and cannot be padded.

`patches/glm53_eagle3_aux.py` clears the first blocker -- DFlash2 needs the
target model to implement `SupportsEagle3`, which upstream's glm5next does not.
It is 42 anchored lines replacing a 351-line `model.py` fork, and it initialises:
`Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`.

The remaining blocker is `kv_cache_utils.py`: 13 patch sites adding draft KV
groups on the GLM-5-Next layout, `DraftSlidingWindowSpec` and its manager,
per-draft-layer tensors and the draft block-size rule. Mounting our fork
unmodified over the new base crash-loops on
`ImportError: cannot import name 'dcp_world_size_for_kv_cache_spec'`, so it has
to be genuinely ported site by site. That is the one remaining task.

## Measuring this without fooling yourself

Two greps of the form `grep -A<n> "device_capability.major == 12"` reported SM90
as present when it was not: the `else:` branch below also names
`FLASHINFER_MLA_SPARSE_SM90`. Read the block, not a window.

Raw diff line counts against a newer base are meaningless -- our patch files are
forks of an older one, so a diff mixes our change with upstream drift. Ask
whether the fix is present, then delete rather than rebase.

Single-sample decode readings are worthless here: the run-to-run spread is
6-9 tok/s. Take a median of six. Draft acceptance per step is the metric that
catches a broken drafter, and `repro/perf-repeat.py` reports both.
