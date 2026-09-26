# DFlash + pipeline parallel on GLM-5.3-Flash: source analysis and patch set

Written 2026-09-02 against vLLM 0.1.dev20051+g487ecf187 (image glm53-spark:sm90-v14),
on top of the existing /home/admin/pp-patches set. Nothing here has been run on a GPU.
"Verified" below means executed in a throwaway CPU-only container; everything else
is reasoning from source.

## TL;DR

1. The PP guard is not the only blocker, and not the first one you would have hit
   next. Two more things are wrong for this drafter, independent of PP:
   - **The target model has no aux-hidden-state support at all.** The nvidia fork
     `vllm/models/glm5next/nvidia/model.py` does not implement `SupportsEagle3` /
     `EagleModelMixin` and its forward never collects aux states. With PP=1 the
     load would have failed with `RuntimeError: Model does not support EAGLE3
     interface` from `set_eagle3_aux_hidden_state_layers`. Fixed in this patch set.
   - **This vLLM build has no DFlash2 implementation.** The checkpoint at
     /home/admin/models/glm-5.3-flash-dflash2 carries DFlash2-only weights
     (`candidate_selector.*`, `layers.N.attention_conv.*`, `layers.N.mlp_conv.*`,
     listed below) that `qwen3_dflash.py` has no modules for. The
     `DFlashDraftModel` rename in config.json only gets past the registry;
     `AutoWeightsLoader` will raise at load time. **Now ported**: see
     "Part 2: the DFlash2 drafter" below (`qwen3_dflash2.py`,
     `dflash2_speculator.py`, `registry.py`, `spec_decode_init.py`).
2. The PP transport itself is small and clean: aux states ride in the existing
   `IntermediateTensors` dict as extra keys. No new collective, no new stream, no
   protocol change. Per-step cost across the 10G link is 16 KiB per token (see
   below), on top of the 32 KiB per token the mHC residual stream already costs.
3. A third PP-specific hole: under PP the DFlash loader skips embedding sharing
   and the drafter ships no `embed_tokens`, so the drafter would run on an
   uninitialised embedding table with no error. Fixed by reading the target's
   embedding from the target checkpoint.

## Files in this directory

Each is a full file for bind-mounting. The first two REPLACE the same-named
files in /home/admin/pp-patches (they are supersets: every PP-PATCH block is
still there, unchanged; my additions are tagged `DFLASH-PP-PATCH`).

| file | mount target | what changed |
|---|---|---|
| `model.py` | `vllm/models/glm5next/nvidia/model.py` | aux-state capture under mHC, PP forwarding, `SupportsEagle3` on both wrappers, `make_empty_intermediate_tensors` grows the aux keys |
| `model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | the `with pipeline parallel is not supported` ValueError moves from `__init__` to `load_model` and is gated on a model class flag |
| `dflash_utils.py` | `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | under PP, load the drafter's `embed_tokens` from the target checkpoint |
| `qwen3_dflash2.py` | `vllm/model_executor/models/qwen3_dflash2.py` (NEW file) | the DFlash2 drafter: grouped dynamic convs, candidate selector, model + causal-LM wrapper |
| `dflash2_speculator.py` | `vllm/v1/worker/gpu/spec_decode/dflash/dflash2_speculator.py` (NEW file) | DFlashSpeculator subclass; `_generate_draft` runs the selector instead of per-slot argmax |
| `registry.py` | `vllm/model_executor/models/registry.py` | one added entry: `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` |
| `spec_decode_init.py` | `vllm/v1/worker/gpu/spec_decode/__init__.py` | `init_speculator` picks `DFlash2Speculator` when the draft's architectures contain `DFlash2DraftModel` |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | GLM-5-Next KV grouper accepts drafter attention layers as their own group (Part 3); v2: small-block draft group opted out of prefix caching (Part 4) |
| `_import_test.py`, `_dflash2_cpu_test.py`, `_dflash2_k5_test.py`, `_dflash2_registry_test.py`, `_kv_groups_cpu_test.py` | (not mounted) | the CPU-only checks I ran, for re-running (mount as `/t.py`, see "Verified" sections) |

### Bind mounts for the whole set

Replace the `model.py` and `model_runner.py` lines in
/home/admin/compose/glm53.pp-experiment/pp-mtp-override.yaml with the first two
below, and add the other six. Everything else in that file stays.

    - /home/admin/pp-patches-dflash/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro
    - /home/admin/pp-patches-dflash/model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro
    - /home/admin/pp-patches-dflash/dflash_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:ro
    - /home/admin/pp-patches-dflash/qwen3_dflash2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py:ro
    - /home/admin/pp-patches-dflash/dflash2_speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/dflash2_speculator.py:ro
    - /home/admin/pp-patches-dflash/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py:ro
    - /home/admin/pp-patches-dflash/spec_decode_init.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py:ro
    - /home/admin/pp-patches-dflash/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro

Then restore the drafter's shipped config:
`cp /home/admin/models/glm-5.3-flash-dflash2/config.json.orig
/home/admin/models/glm-5.3-flash-dflash2/config.json` on all four boxes
(architectures back to `DFlash2DraftModel`). The `--speculative-config`
stays `{"method":"dflash","model":"/models/glm-5.3-flash-dflash2","num_speculative_tokens":7}`:
the method is still `dflash` (same worker, same query layout); the
architecture name selects the DFlash2 classes. `num_speculative_tokens` MUST
be 7 = `block_size - 1`; the model refuses anything else (see below).

Diff against the pp-patches versions: `diff -u /home/admin/pp-patches/model.py
/home/admin/pp-patches-dflash/model.py` (about 130 added lines) and the same
for model_runner.py (about 20 lines). dflash_utils.py diffs against the image's
file (about 70 lines).

## Mechanism

### Where aux states come from (answer to question 1)

`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` reads the drafter's
`dflash_config.target_layer_ids` = [5, 14, 24, 33, 42] and adds 1, giving aux ids
(6, 15, 25, 34, 43), then calls `model.set_aux_hidden_state_layers(ids)`. The
convention (from `EagleModelMixin._maybe_add_hidden_state` and DeepSeek V4's
`if idx + 1 in self.aux_hidden_state_layers`) is: **aux id L is the completed
output of global layer L-1**, i.e. captured after layer L-1 runs. The drafter
concatenates the list along the last dim and feeds `fc` (weight [4096, 20480] in
the checkpoint = 5 x 4096), so each aux state is `[num_tokens, 4096]` and the list
order must be ascending layer order.

Shape/dtype per aux state: `[num_tokens, 4096]` bf16.

**mHC complication.** GLM-5.3-Flash's residual stream is rank 3,
`[tokens, 4, 4096]` (mhc_num_residual_streams = 4, default in
`transformers_utils/configs/glm5_next.py`; the target config does not override
it). A non-final layer never materialises its output: it returns
`(x, residual, post, comb)` and defers `hc_post` into the next layer's fused
post+pre kernel. So "the completed output of layer L-1" has to be reconstructed:
`layer.hc_post(x, residual, post, comb)` -> `[tokens, 4, 4096]`, then
`hc_contract` (= `mean(dim=1)`, `layers/mhc.py`) -> `[tokens, 4096]`.

Why mean over the streams and not something else: two independent sources agree.
(a) DeepSeek V4's dspark capture in this same tree does
`mhc_post_tilelang(...).mean(dim=1)`. (b) The DFlash reference capture for this
exact target, sglang PR #36708 (the PR the drafter README tells you to install),
does `hc_contract(hidden_states + residual, hc_mult)` with `hc_contract` being
`unflatten(-1, (hc_mult, -1)).mean(-2)`, verified in that PR's unit test. Note
that PR was closed unmerged on 2026-08-27, but it is what the drafter's authors
point at. I could not check the HF transformers 5.7 modeling code the drafter was
trained against; it is not in the image (transformers 5.15.1 there has no glm5).
This is the single biggest correctness assumption. If it is wrong the drafts will
be systematically bad and acceptance length will sit near 1.0, with no error.

Implemented in `Glm5NextModel._completed_layer_output`. For a non-mHC (70B)
config, or for the final layer (already post'ed and contracted), the layer's
returned `hidden_states` is used as is.

### How it crosses the PP boundary (answer to questions 2 and "can it ride along")

Yes, it rides along. `Glm5NextModel.forward` on a non-last rank now returns

    IntermediateTensors({"hidden_states": <rank-3 residual stream>,
                         "aux_hidden_states.6": [tokens, 4096],
                         "aux_hidden_states.15": [tokens, 4096]})

and the next rank reads the `aux_hidden_states.<L>` keys back out of
`intermediate_tensors` before running its own layers. This composes with the
existing PP-PATCH in model.py untouched: the boundary `hc_post` for the residual
stream still runs exactly as before; the aux states are separate tensors that
were computed at their own layer, not at the boundary.

Why no new collective is needed, from the source:

- `gpu_worker.py execute_model` sends `output.tensors` with
  `get_pp_group().isend_tensor_dict(...)` and receives with
  `irecv_tensor_dict`. `parallel_state.py` `_split_tensor_dict` pickles a
  metadata list (key, shape, dtype, device) and `send_object`s it per call, then
  sends each tensor. Keys are arbitrary strings. Extra keys need nothing.
- `model_runner.py execute_model` copies received tensors into the persistent
  buffer by iterating the persistent buffer's keys:
  `{k: v[:n].copy_(intermediate_tensors.tensors[k][:n]) for k, v in
  self.intermediate_tensors.tensors.items()}`. So the persistent buffer's key set
  must be exactly what arrives. That buffer comes from
  `model.make_empty_intermediate_tensors(batch_size=max_num_tokens, ...)`, called
  at the end of `load_model` — after `set_eagle3_aux_hidden_state_layers`. My
  factory reads `self.model.incoming_aux_ids()` at call time, so it allocates
  exactly the aux ids owned by earlier stages.
- FULL cudagraph capture on a non-last rank (`cudagraph_utils.py
  _build_forward_fn_factory`) copies every key of the output
  `IntermediateTensors` into static buffers created with
  `IntermediateTensors.empty_like` on first capture, and `run_fullgraph` returns
  `self.intermediate_tensors[:num_tokens]`, which slices every key. On the last
  rank the aux list is copied into `self.aux_hidden_states[i][:num_tokens]`
  buffers. So the added tensors are in the captured graphs' static buffers
  already, with no change (answer to the CUDA-graph question). The incoming aux
  tensors on the last rank are views of the persistent recv buffer, which is
  the graph's static input, same as `hidden_states`.
- TP: `isend_tensor_dict(all_gather_group=get_tp_group())` slices any tensor
  whose numel is divisible by tp_size and all-gathers on receive. That is only
  valid for tensors replicated across TP ranks. The aux state is the output of a
  layer whose attention/MoE outputs are all-reduced, so it is replicated, same as
  `hidden_states`. Under sequence-parallel MoE the model does
  `sp_all_gather(aux)[:full_num_tokens]` first (DSv4 pattern); SP is not on in
  this deployment and that branch is untested.

### Layer-id mapping (answer to the global-vs-local question)

The layer loop is `enumerate(self._active_layers, start=self.start_layer)`, so
`idx` is the GLOBAL layer index on every rank; the capture test is
`idx + 1 in self.aux_hidden_state_layers`. Nothing indexes by local position.

Partition with PP=2 and no `VLLM_PP_LAYER_PARTITION`: `get_pp_indices(45, r, 2)`
returns `(0, 23)` and `(23, 45)` (the remainder goes to the second-to-last
partition; verified by calling it in the container). So:

| stage | global layers | aux ids produced (from layer) |
|---|---|---|
| 0 | 0..22 | 6 (layer 5), 15 (layer 14) |
| 1 | 23..44 | 25 (layer 24), 34 (layer 33), 43 (layer 42) |

**This corrects the prompt's premise**: layer 24 is on stage 1, so 2 of the 5
aux states cross the link, not 3. If you set `VLLM_PP_LAYER_PARTITION` to move
the split above layer 24 (e.g. `25,20`), 3 would cross and the cost below grows by
50%; the code handles any partition, the keys are computed from
`start_layer`.

The incoming/own split is `l - 1 < start_layer` (incoming) vs the loop
(own). The sender ships everything it has collected (incoming + own), so PP > 2
works by pass-through. The list handed to the drafter is incoming (ascending)
then own (ascending); incoming ids are always lower than own ids, so it is in
layer order. Verified in the container for start_layer 0 and 23 (see
_import_test.py output in the log at the bottom).

### Bytes per step across the PP link (answer to the cost question)

hidden 4096, bf16 (2 B): one aux state = 8192 B/token = 8 KiB/token.

- Existing PP payload: `[tokens, 4, 4096]` bf16 = 32 KiB/token.
- Added by this patch (default 23/22 split, 2 aux states): **16 KiB/token**, i.e.
  +50%. With 3 states crossing: 24 KiB/token, +75%.
- Decode step, DFlash k=7 => 8 query tokens per request: +128 KiB per request per
  step. At batch 1: +128 KiB, ~0.12 ms on 10 GbE (~1.1 GB/s payload). At batch 8:
  +1 MiB, ~1 ms per step, total PP payload 3 MiB, ~2.7 ms per step.
- Prefill chunk of 8192 tokens: +128 MiB, ~115 ms extra per chunk (existing
  256 MiB, ~230 ms). If prefill TTFT matters this is where you will feel it.

Plus the pickled metadata per step grows by two small tuples (negligible).

### Extra memory (unified memory, so read this)

- Stage 1 (last rank), persistent recv buffer: 2 x `max_num_batched_tokens` x
  8 KiB. At 8192 tokens: 128 MiB, allocated once with `torch.zeros`.
- Stage 1, FULL-graph static aux output buffers: 5 x (first-captured
  num_tokens) x 8 KiB. This is the existing eagle3/dflash behaviour, not new
  with PP.
- Stage 0, FULL-graph static output buffers: 2 x (first-captured num_tokens)
  x 8 KiB (empty_like of the first capture's output).
- Stage 0 transient per aux layer: the standalone `hc_post` reconstruction
  `[tokens, 4, 4096]` bf16 = 32 KiB/token plus the 8 KiB/token mean; freed
  immediately (caching allocator).
- Receive side per step: `irecv_tensor_dict` allocates fresh tensors per key
  (`torch.empty`), then they are copied into the persistent buffer; the caching
  allocator recycles them.
- Load time on the last rank: `_load_target_embed_from_checkpoint` reads the
  1.27 GB bf16 embedding to CPU, `weight_loader` copies the TP shard into the
  param, then the CPU tensor is dropped. Peak +1.27 GB host RAM briefly, before
  the KV-cache budget is measured.

Nothing here is a large standing allocation. The largest is the 128 MiB recv
buffer on stage 1.

### Stage-0 compute cost

Two extra `mhc_post` kernel launches (one per aux layer) plus two means per
forward. The deferred fused post+pre in the next layer still runs; this is a
redundant recompute of the post part, exactly as DSv4 does it (`aux_recon` there).
Small relative to a layer.

## The guard change (model_runner.py)

`__init__` no longer raises for eagle3/dflash/dspark under PP; it still sets
`use_aux_hidden_state_outputs = True` on every rank (that is what makes
`set_eagle3_aux_hidden_state_layers` run on stage 0 too, which is required).
`load_model` raises the same message, naming the model class, unless
`getattr(self.model, "supports_pp_aux_hidden_states", False)`. Both GLM wrapper
classes declare that flag. DeepSeek V4 does not, and its forward drops aux
states on non-last ranks, so it keeps failing loudly as before (its dspark
loader also raises `NotImplementedError` for PP on its own).

Note `use_aux_hidden_state_outputs` is passed to the cudagraph manager on all
ranks; on non-last ranks the manager's non-last branch ignores it, so the flag
being True on stage 0 is harmless (read `cudagraph_utils.py` around line 699).

## The embedding fix (dflash_utils.py)

`load_dflash_model` only shares `embed_tokens` from the target when
`get_pp_group().world_size == 1` ("each rank owns its own embedding" — true for
the target, false for a drafter that ships none). The drafter's `load_weights`
adds `embed_tokens` to `skip_substrs` when the checkpoint has none, so the
`VocabParallelEmbedding` built in `DFlashQwen3Model.__init__` stays
uninitialised. On PP=1 the share hides this. Under PP this would produce garbage
drafts silently; the DFlash mask token (154856) is an entry in that table too.

Patch: when `pp_size > 1` and the drafter has not set `has_own_embed_tokens`,
read `model.language_model.embed_tokens.weight` (bf16 [154880, 4096], in
`model-00001-of-00120.safetensors` per the target's index; checked on disk) with
`safetensors.safe_open` and hand it to the param's `weight_loader`, which does
the TP vocab sharding. It logs
`DFLASH-PP-PATCH: loaded drafter embed_tokens from target checkpoint ...`.
Look for that line on the last rank; if it is absent the fix did not run.

The drafter has no `lm_head` either; that is shared from the target's
`language_model.lm_head`, which the last rank does have (`get_target_lm_head`).
No change needed.

The drafter's `mtp.py`-style equivalent (loading the target embedding through
the shared weights stream) does not apply here because the draft has its own
checkpoint directory.

## What was verified (executed)

In a throwaway container from the image with the three files bind-mounted
(no GPU, `cd /` to avoid the `vllm/tokenizers` shadowing trap):

- The three modules import cleanly next to the existing pp-patches
  `pp_utils.py` and `speculative.py`.
- `supports_eagle3(Glm5NextForCausalLM)` and
  `supports_eagle3(Glm5NextForConditionalGeneration)` are True (runtime Protocol
  check, so the MRO with the Protocol bases is sound).
- `_set_aux_hidden_state_layers((43,6,25,15,34))` sorts to (6,15,25,34,43);
  `incoming_aux_ids()` is `()` at start_layer 0 and `(6, 15)` at start_layer 23;
  keys are `aux_hidden_states.6`, `aux_hidden_states.15`; out-of-range ids raise.
- `hc_contract` on a `[2,4,3]` bf16 tensor is the mean over dim 1, bf16 out.
- `get_pp_indices(45, r, 2)` = (0,23), (23,45).
- The target checkpoint index has `model.language_model.embed_tokens.weight`
  BF16 [154880, 4096]; the drafter safetensors has no embed/lm_head.

## What was NOT verified (needs a GPU run)

- The `hc_post` reconstruction call: `layer.hc_post(x, residual, post, comb)`
  with the tensors as returned by a non-final layer. Same signature the boundary
  PP-PATCH and layer 44 use, so I expect it to work, but I have not executed it.
- That `hc_post` + mean is the feature the drafter was trained on (see above).
- CUDA graph capture on both ranks with the extra keys; piecewise mode on the
  non-last rank in particular (the code comment says the PW graph "handles the
  model outputs internally"; the existing PP path already returns
  `IntermediateTensors` through it, and my change only adds dict keys).
- The draft model's own KV-cache layer registration under PP (its layers are
  named `model.layers.22..26` on rank 1 from `get_num_layers(parallel_config)`
  = 22; the target's are `language_model.model.layers.23..44`, so no name
  collision, but I did not trace kv_cache_config construction for a draft on a
  non-first rank).
- Memory headroom on node0 with the extra buffers listed above.
- Anything about the DFlash2 modules (they do not exist in this build).

## Failure signatures

| symptom | meaning |
|---|---|
| `ValueError: There is no module or parameter named 'candidate_selector' in DFlashQwen3ForCausalLM` (or `layers.0.attention_conv`) at drafter load | expected with this build: DFlash2 not implemented. See next section. |
| `RuntimeError: Model does not support EAGLE3 interface` at load | model.py patch not mounted (old nvidia model.py has no SupportsEagle3) |
| `ValueError: dflash with pipeline parallel is not supported: <cls> does not forward aux hidden states` at load | model_runner.py patched but model.py not, or a different target class |
| `KeyError: 'aux_hidden_states.6'` in `model_runner.execute_model` on rank 1 | stage 0 did not ship the key: `set_eagle3_aux_hidden_state_layers` did not run on stage 0, or a partition mismatch between ranks |
| `RuntimeError: The size of tensor a ... must match ...` in the same `copy_` | shape mismatch: aux state not `[tokens, 4096]` (contraction did not happen) |
| `AssertionError: collected aux ids [...] configured (...)` on rank 1 | fewer/more aux states than configured reached the last rank |
| `ValueError: DFlash drafter expects 20480 concatenated aux hidden features but received 12288` | only stage 1's three states reached the drafter (the incoming two were dropped); 8192 would mean only stage 0's arrived |
| drafts accepted at length ~1.0, no error | either the embedding did not load (grep the log for `DFLASH-PP-PATCH: loaded drafter embed_tokens`) or the mean-contraction assumption is wrong, or (if you strip the DFlash2 weights to get past load) the backbone is running without its dynamic convolutions |
| PP hang with NCCL watchdog `pp_broadcast`/send timeout | not expected from this change (metadata is exchanged by object send, shapes are self-describing); look at the existing draft-token broadcast PP-PATCH first |

## Part 2: the DFlash2 drafter

### What DFlash2 adds, from the reference code

Read in full: github.com/z-lab/dflash `dflash/model.py` (classes
`GroupedDynamicCausalConv`, `CandidateSelector`, `DFlash2DraftModel`,
`Qwen3DFlashDecoderLayer`, and `dflash_generate` for how they are driven) and
SGLang `python/sglang/srt/models/dflash.py` (`_grouped_conv`,
`DFlashGroupedConv`, `DFlashDecoderLayer.forward`, `_score_edges`,
`CandidateSelector`, `DFlash2DraftModel.compute_candidates`). The two agree on
the math; the port follows z-lab for semantics and SGLang for the flat-batch
formulation.

1. **Grouped dynamic causal convolution**, one `attention_conv` and one
   `mlp_conv` per layer. For a sub-layer input x (after the pre-norm), a linear
   `kernel_projection` [2*taps*groups, hidden] gives per-token, per-group deltas
   for both the input side and the output side. Input side:
   `x'[t] = sum_tap (base_kernel[0, tap] + delta_in[t, tap, group]) * x[t - tap]`;
   the sub-layer (attention / MLP) runs on x'; output side: same form with
   `base_kernel[1]` and `delta_out` on the sub-layer output, before the residual
   add. taps = `conv_kernel_size` = 2, `conv_group_size` = 16 -> 256 groups,
   hence `kernel_projection.weight` [1024, 4096] and `base_kernel` [2, 2, 4096].
   x[t - tap] is zero when t - tap leaves the token's own draft block. The
   convolution runs over the draft block only (anchor + 7 masks); the context
   never passes through it.
2. **Candidate selector.** For each of the N = 7 mask slots take the top-k
   (k = `selector_top_k` = 16) lm_head candidates with their logits (`unary`).
   Score every transition from the previous slot's candidate p to this slot's
   candidate c as
   `unary[e, c] + < A[pred] * P(h[e]), B[c] >` with A = `predecessor_codebook`,
   B = `successor_codebook` (both [154880, 256]), P = `hidden_projection`
   ([256, 4096]) applied to the draft's final normed hidden state at slot e, and
   pred = the verified anchor token for slot 0, else the candidate chosen at
   slot e-1. Walk greedily: slot 0 takes the argmax over c, each later slot the
   argmax given the previous choice. This replaces the independent per-slot
   argmax of DFlash v1; it is what "keeps drafting parallel" coherent.
3. Everything else (mask-token block, `fc` + `hidden_norm` context projection,
   context K/V precompute, RoPE, sliding window) is DFlash v1 and reused.

Sampling: the reference also has a temperature > 0 walk that emits a q over
the 16 candidates for a candidate-restricted rejection sampler. Not ported.
vLLM's default `draft_sample_method="greedy"` leaves `draft_logits` None, and
the rejection sampler then scores the draft as a point mass
(`rejection_sampler_utils.py`, `HAS_DRAFT_LOGITS=False` -> `draft_log_prob=0`).
That is exact for any target temperature, so nothing is lost in output
distribution; only acceptance length under sampling could differ from the
reference. `DFlash2Speculator.__init__` raises if `draft_sample_method` is
`probabilistic` or `use_local_argmax_reduction` is set.

### Every checkpoint tensor and the module that consumes it

Checkpoint: /home/admin/models/glm-5.3-flash-dflash2/model.safetensors, 81
tensors (header read on disk). `DFlashQwen3ForCausalLM.load_weights` (inherited)
prefixes every name except `lm_head*` with `model.`, then `AutoWeightsLoader`
with the inherited `hf_to_vllm_mapper` (q/k/v -> `qkv_proj`, gate/up ->
`gate_up_proj`).

| checkpoint tensor (N = 0..4) | dtype, shape | vLLM parameter | module |
|---|---|---|---|
| `fc.weight` | BF16 [4096, 20480] | `model.fc.weight` | `ReplicatedLinear`, aux-state projection (5 x 4096 -> 4096) |
| `hidden_norm.weight` | BF16 [4096] | `model.hidden_norm.weight` | `RMSNorm` on the projected context before K/V |
| `norm.weight` | BF16 [4096] | `model.norm.weight` | final `RMSNorm` |
| `layers.N.input_layernorm.weight` | BF16 [4096] | `model.layers.N.input_layernorm.weight` | `RMSNorm` |
| `layers.N.post_attention_layernorm.weight` | BF16 [4096] | `model.layers.N.post_attention_layernorm.weight` | `RMSNorm` |
| `layers.N.self_attn.q_proj.weight` | BF16 [4096, 4096] | `model.layers.N.self_attn.qkv_proj.weight` (shard q) | `QKVParallelLinear` |
| `layers.N.self_attn.k_proj.weight` | BF16 [1024, 4096] | same (shard k) | `QKVParallelLinear` |
| `layers.N.self_attn.v_proj.weight` | BF16 [1024, 4096] | same (shard v) | `QKVParallelLinear` |
| `layers.N.self_attn.o_proj.weight` | BF16 [4096, 4096] | `model.layers.N.self_attn.o_proj.weight` | `RowParallelLinear` |
| `layers.N.self_attn.q_norm.weight` | BF16 [128] | `model.layers.N.self_attn.q_norm.weight` | `RMSNorm` |
| `layers.N.self_attn.k_norm.weight` | BF16 [128] | `model.layers.N.self_attn.k_norm.weight` | `RMSNorm` (also stacked into the fused context-KV buffers) |
| `layers.N.mlp.gate_proj.weight` | BF16 [12288, 4096] | `model.layers.N.mlp.gate_up_proj.weight` (shard 0) | `MergedColumnParallelLinear` |
| `layers.N.mlp.up_proj.weight` | BF16 [12288, 4096] | same (shard 1) | `MergedColumnParallelLinear` |
| `layers.N.mlp.down_proj.weight` | BF16 [4096, 12288] | `model.layers.N.mlp.down_proj.weight` | `RowParallelLinear` |
| `layers.N.attention_conv.base_kernel` | BF16 [2, 2, 4096] | `model.layers.N.attention_conv.base_kernel` | `DFlash2GroupedConv` (nn.Parameter) |
| `layers.N.attention_conv.kernel_projection.weight` | BF16 [1024, 4096] | `model.layers.N.attention_conv.kernel_projection.weight` | `DFlash2GroupedConv.kernel_projection` (`ReplicatedLinear`) |
| `layers.N.mlp_conv.base_kernel` | BF16 [2, 2, 4096] | `model.layers.N.mlp_conv.base_kernel` | `DFlash2GroupedConv` |
| `layers.N.mlp_conv.kernel_projection.weight` | BF16 [1024, 4096] | `model.layers.N.mlp_conv.kernel_projection.weight` | `DFlash2GroupedConv.kernel_projection` |
| `candidate_selector.hidden_projection.weight` | BF16 [256, 4096] | `model.candidate_selector.hidden_projection.weight` | `DFlash2CandidateSelector.hidden_projection` (`ReplicatedLinear`) |
| `candidate_selector.predecessor_codebook` | BF16 [154880, 256] | `model.candidate_selector.predecessor_codebook` | `DFlash2CandidateSelector` (nn.Parameter; the checkpoint has no `.weight` suffix, so the parameter is named exactly that) |
| `candidate_selector.successor_codebook` | BF16 [154880, 256] | `model.candidate_selector.successor_codebook` | same |

Unconsumed checkpoint tensors: **none** (asserted by `_dflash2_cpu_test.py`,
part B, against the real header).

Model parameters NOT fed by the checkpoint, and why that is right:

| parameter | source |
|---|---|
| `lm_head.weight` | shared from the target (`load_dflash_model` -> `get_target_lm_head`); the last PP rank has it |
| `model.embed_tokens.weight` | PP=1: shared from the target; PP>1: read from the target checkpoint by the `dflash_utils.py` patch (Part 1). Never in the drafter checkpoint. |
| `model.mask_embedding` | only used when a checkpoint ships `mask_embedding.pt`; this one does not, so `embed_tokens[154856]` (the target's row) is the mask embedding, as in the reference (`_raw_input_embeddings(target, ...)`) |
| `draft_id_to_target_id` | not created: draft vocab == target vocab (154880) |

Codebook memory: 2 x 154880 x 256 x 2 B = 158 MB, replicated per TP rank (as
SGLang does; candidate ids are global, any rank may need any row). Plus
`hidden_projection` 2 MB and per-layer conv weights 2 x (16 KB + 8 MB) x 5 =
80 MB. Total new drafter weight memory about 240 MB per rank on the last PP
stage. Unified memory: that is host RAM, at load time.

### Registration mechanism

`registry.py` (a full copy of the image's file with one line added to
`_SPECULATIVE_DECODING_MODELS`):
`"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")`.
`_resolve_module_name` maps the bare name to
`vllm.model_executor.models.qwen3_dflash2`, which is where `qwen3_dflash2.py`
is mounted. The registry inspects model classes in a subprocess
(`python -m vllm.model_executor.models.registry`), which sees the same bind
mounts. Alternatives considered: a `vllm.general_plugins` entry point needs an
installed package (not mountable); mutating the registry from another mounted
module would need an import hook. One line in a copied file is the least magic.

Why the shipped name survives: `speculative.py` wraps the drafter's hf_config
in `EAGLEConfig(method="dflash")`, which rewrites `architectures` to
`f"DFlash{arch}"` unless the name already starts with `DFlash`.
`DFlash2DraftModel` does, so it is kept (checked by `_dflash2_registry_test.py`).
The method stays `dflash` (auto-detected from the path containing "dflash", or
explicit in `--speculative-config`), so every `use_dflash()` branch (parallel
drafting, K extra slots, query layout) applies unchanged.

`spec_decode_init.py`: `init_speculator` returns `DFlash2Speculator` when
`draft_model_config.architectures` contains `DFlash2DraftModel`, else the
original `DFlashSpeculator`.

### How the pieces attach to the existing DFlash worker

- `DFlash2Qwen3Model` (subclass of `DFlashQwen3Model`) rewrites `__init__`
  only: DFlash2 layers, the selector, and `self.do_not_compile = True`. The
  parent is `@support_torch_compile`-decorated; its `__call__` returns
  `self.forward(...)` directly when that flag is set, so the drafter runs eager
  inside the speculator's FULL CUDA graph (DFlash never uses PIECEWISE). Context
  K/V precompute, embedding, weight mapper and loader are inherited untouched.
- `DFlash2Qwen3DecoderLayer` (subclass) adds the two convs and overrides
  `forward` in the reference order: prepare -> attention -> finish -> fused
  residual+norm -> prepare -> MLP -> finish.
- `DFlash2Qwen3ForCausalLM` (subclass) rebuilds `__init__` with the new model
  and adds `propose_greedy(hidden, anchor_ids, num_reqs, N)`: full-vocab logits
  via the shared lm_head (`compute_logits`, TP all-gathered, the same cost the
  v1 greedy path pays), `torch.topk(16)`, lattice, greedy walk.
- `DFlash2Speculator._generate_draft` (the function the DFlash CUDA-graph
  manager captures) runs the backbone, gathers the 7 mask-slot rows per request
  through the existing `sample_indices`, reads the anchor token of each request
  from `input_buffers.input_ids[req * 8]` (written there by
  `prepare_dflash_inputs` as the bonus token), calls `propose_greedy`, and
  writes `draft_tokens[:num_reqs]`. Shapes depend only on `num_reqs` and
  `num_tokens_padded`, as before, so capture/replay semantics are unchanged.

### Block size and causality: two places the port had to be careful

1. `dflash_config.block_size = 8` must equal the worker's query block
   `1 + num_speculative_tokens`. The convs and the lattice are defined over that
   block; the conv's "previous token" mask uses `t % 8` on the flat token axis,
   which is only right because `prepare_dflash_inputs` places request r's
   block at `r * num_query_per_req`. `DFlash2Qwen3Model.__init__` raises a
   `ValueError` naming both numbers if they differ. With
   `num_speculative_tokens=7` they match.
2. **The checkpoint is non-causal, and stock vLLM would have run it causal.**
   config.json has `is_causal: false` and five `sliding_attention` layers with
   `sliding_window: 2048`. `qwen3_dflash._dflash_layer_causal` ignores
   `is_causal` and marks every sliding layer causal; z-lab's
   `Qwen3DFlashAttention.__init__` and SGLang's `_get_dflash_attention_type`
   both let an explicit `is_causal` win. Non-causal is the whole point of block
   drafting (mask slots attend to each other). `normalize_dflash2_causality`
   writes `dflash_config["causal"] = False` once, before the speculator's
   `__init__` reads `dflash_has_any_non_causal` (backend choice) and before the
   layers are built, so all readers agree. Consequence: the draft attention
   backend must support non-causal queries with a sliding window. The reference
   window is symmetric for non-causal layers (`|q - k| < 2048`, z-lab
   `_attention_mask`); what vLLM's backend does with `per_layer_sliding_window`
   under `use_non_causal` I could not check on CPU. Below 2048 tokens of
   context it cannot matter.

### Verified by execution (CPU-only, in the image, files bind-mounted)

`_dflash2_cpu_test.py` (mount at /t.py with the drafter dir at
/models/glm-5.3-flash-dflash2):

- A. `DFlash2GroupedConv.prepare/finish` equal the z-lab
  `GroupedDynamicCausalConv` (its code copied verbatim into the test) on random
  weights and inputs to 1e-5, on a flat `[3 * 8, 64]` buffer vs the reference's
  `[3, 8, 64]`; perturbing block 1 leaves block 0 bit-identical (no cross-request
  leak). `DFlash2CandidateSelector.greedy_path` returns the same tokens as the
  z-lab `CandidateSelector.select` (temperature 0) on 20 random trials,
  including different top-k orderings (`sorted=False` vs `True`).
  `normalize_dflash2_causality` folds `is_causal` and does not override an
  explicit `causal`.
- B. `DFlash2Qwen3ForCausalLM` constructs on CPU with the real drafter config
  (Attention stubbed, single-rank gloo groups): 69 parameters; every one of the
  81 checkpoint tensors maps to a parameter (0 unmapped); the only parameters
  not fed are `lm_head.weight`, `model.embed_tokens.weight`,
  `model.mask_embedding`; direct (non-stacked) parameter shapes equal the
  header shapes.
- C. `AutoWeightsLoader` dry run: `load_weights` over zero tensors with the
  checkpoint's exact names, dtypes and shapes completes; after NaN-filling every
  parameter first, exactly those three remain NaN; `has_own_embed_tokens` and
  `has_own_lm_head` are False (so the target's are shared); the fused
  context-KV buffers build for 5 layers.

`_dflash2_registry_test.py`: `ModelRegistry._try_inspect_model_cls` (the
subprocess import) and `_try_load_model_cls` resolve `DFlash2DraftModel` to
`vllm.model_executor.models.qwen3_dflash2.DFlash2Qwen3ForCausalLM`;
`vllm.v1.worker.gpu.spec_decode` and `dflash2_speculator` import;
`EAGLEConfig(hf, method="dflash")` keeps `architectures=['DFlash2DraftModel']`
and carries `dflash_config` and `is_causal`.

### Not verified (needs a GPU)

- Any forward pass of the assembled drafter: attention backend selection for
  non-causal + sliding window on GB10, the K/V precompute path with these
  weights, FULL CUDA-graph capture of `_generate_draft` with the lattice and
  the walk inside it (all ops are shape-static; `torch.topk`, `einsum`, gathers
  and argmax are capturable, but I have not captured them).
- Numerical agreement of the whole drafter with the reference on real weights
  and real target hidden states (the test compares my modules to the reference
  modules, not end to end).
- Acceptance length. This is the number that tells you whether the mean-over-
  streams aux state (Part 1) and the non-causal fix are right. A working DFlash2
  on this pair should accept well above 3 tokens per step at k=7; ~1 means a
  systematic mismatch somewhere upstream of the selector.
- TP behaviour of `compute_logits` -> `topk` with the shared target lm_head
  (same path the v1 greedy drafter uses, so low risk).
- Drafter memory and the extra 158 MB of replicated codebooks on the last
  stage.

### Residual risk, ranked

1. Attention backend: non-causal + sliding window. If the backend refuses, the
   error is at draft model load / attention metadata build. If it silently
   applies a causal window, drafts degrade with no error.
2. The aux hidden state contraction (Part 1). Same symptom: acceptance ~1.
3. CUDA-graph capture of the selector walk: `maps[:, edge].gather(...)` in a
   Python loop of 6 steps is fine; anything that syncs (`.item()`) would break
   capture, and there is none in the code path.
4. Padded requests: the walk runs on padded rows too; their `sample_indices`
   point at row 0 and their `sample_idx_mapping` is -1, and `draft_tokens` rows
   beyond `num_reqs` are never read. Same contract as v1.
5. `torch.topk` on bf16 logits then `.float()`: the reference takes top-k on
   the raw logits too; ties broken differently would only matter at exact
   equality.

### Failure signatures (DFlash2-specific)

| symptom | meaning |
|---|---|
| `ValueError: DFlash2DraftModel: dflash_config.block_size=8 but num_speculative_tokens=K gives ...` | set `num_speculative_tokens` to 7 |
| `There is no module or parameter named 'candidate_selector'` | registry.py / qwen3_dflash2.py not mounted, or config.json still says `DFlashDraftModel` (v1 class picked) |
| `NotImplementedError: DFlash2 implements the greedy selector walk only` | `draft_sample_method` not greedy |
| log line `DFlash2 selector: top_k=16, rank=256, block=8, causal=False` missing | `DFlash2Speculator` not dispatched: `spec_decode_init.py` not mounted, or architectures do not contain `DFlash2DraftModel` |
| `causal=True` in that log line | `normalize_dflash2_causality` did not run before the speculator read the config; drafts will be causal and bad |
| attention backend error mentioning non-causal / sliding window at load | risk 1 above; try `--speculative-config` with `"attention_backend"` set to a backend that supports `use_non_causal` |


## Part 3: the KV cache grouping failure (`unify_kv_cache_spec_page_size`)

### What is compared (question 1)

`get_kv_cache_groups` (kv_cache_utils.py ~2014) tries, in order: uniform
spec; `UniformTypeKVCacheSpecs.from_specs`; `group_and_unify_kv_cache_specs`
(DeepSeek V4 only: needs a `SlidingWindowMLASpec`); then
**`_get_kv_cache_groups_glm5_next`** (~1401); then the generic path:
`unify_kv_cache_spec_page_size` + `_get_kv_cache_groups_uniform_page_size`.

The GLM-5-Next grouper is what the working MTP config uses. It builds: one
`UniformTypeKVCacheSpecs` group of all MLA + kpool-indexer layers, one tail
group, and mamba groups whose layers co-own the MLA layers' tensors
(slot sharing). Its entry condition is `all(type(s) is MLAAttentionSpec)` for
every non-mamba, non-tail spec. The DFlash2 drafter contributes five
`SlidingWindowSpec` layers, so the grouper returns `None` and the model falls
through to the generic unifier.

The generic unifier takes `max_page_size = max(page_size_bytes over all
layers)` and, for every smaller page, requires `max % page == 0` (it then
multiplies that layer's `block_size` by the ratio). The only escapes are
`MambaSpec` (padded) and an `AttentionSpec` with
`indexes_kv_by_block_stride=True` (padded, read through a strided view; set by
backends that opt in — the DeepSeek V4 sparse backends do, the GLM indexer
does not). Anything else raises the observed `NotImplementedError`.

### The page sizes (question 2)

Read from the spec builders (`mla_attention.py get_kv_cache_spec`,
`glm5next/nvidia/attention.py Glm5NextIndexerCache`/`Glm5NextTailCache`,
`attention.py get_kv_cache_spec`) with the target config (`kv_lora_rank` 512,
`qk_rope_head_dim` 0, `index_kpool` 4, indexer head_dim 128 + 4 fp8-scale bytes)
and the entrypoint's `--block-size 2304`, `--kv-cache-dtype fp8_e4m3` (the
entrypoint default; attempt 1 ran this unless KV_CACHE_DTYPE was overridden):

| layer | spec | formula | page bytes |
|---|---|---|---|
| MLA latent (11 layers) | `MLAAttentionSpec`, head 512, fp8 | 2304 x 1 x 512 x 1 | **1,179,648** (bf16: 2,359,296) |
| kpool indexer `k_cache` (11) | `MLAAttentionSpec`, head 132, uint8, `compress_ratio` 4 | (2304/4) x 132 | **76,032** |
| indexer `tail_cache` (11) | `KpoolTailSpec`, block 4, head 256, bf16 | 4 x 512 x 2 | 4,096 (2,048 logical; the grouper pads it to the indexer page) |
| KDA mamba (34) | `MambaSpec` | state shapes | ~1.0 MB (independent of block size) |
| DFlash2 draft (5) | `SlidingWindowSpec`, 4 kv heads (8/TP2), head 128, fp8, window 2048 | block x 4 x 256 x 1 | 16,384 at the 16-token kernel block the layer reports (bf16: 32,768) |

Two facts follow. (a) The draft spec does NOT carry `--block-size`;
`Attention.get_kv_cache_spec` deliberately emits the backend's smallest
kernel block and leaves scaling to the unifier. So the max page is the
**MLA page**, not the drafter's; the drafter's 16 KiB divides 1,179,648 (x72)
and would simply have been scaled to a 1152-token block. (b) The layer that
fails is the **indexer**: 1,179,648 / 76,032 = 15.515... Per token the MLA
page is 512 B (fp8) or 1024 B (bf16) and the indexer page is 132/4 = 33 B, so
the ratio is 512/33 or 1024/33 **for every block size** (33 = 3 x 11; the MLA
side is a power of two times the block). No `--block-size` makes it divide.
`--kv-cache-dtype-skip-layers` cannot name the indexer (it matches
`sliding_window` and layer indices of `Attention` layers; the indexer is not an
`Attention` layer and its dtype is fixed uint8), and even if the indexer page
did divide, the generic path would drop the slot-shared layout the GLM runner
depends on. So the fix is not a flag combination.

Verified: `_kv_groups_cpu_test.py` builds these specs on CPU (page sizes
printed above match: 1,179,648 / 76,032 / 2,048 / 16,384) and, against the
stock kv_cache_utils.py, raises exactly
`Layer language_model.model.layers.3.self_attn.indexer.k_cache: page size is
not divisible by the maximum page size and cannot be padded` for both PP=1
and PP=2.

### PP-specific? (question 3)

No. `get_kv_cache_configs` merges all workers' specs into one dict before
grouping, so the merged set at PP=2 is the same set a single PP=1 worker
reports. The CPU reproduction fails identically at PP=1. (The published
recipe is SGLang; it never runs this code.) The earlier TP=2/PP=1 attempt
on .70/.77 died before this point (EAGLE3 interface), which is why it was not
seen there.

### The fix (question 4): `kv_cache_utils.py`, four edits, tagged DFLASH2-PATCH

Chosen candidate: **give the drafter its own KV group inside the GLM-5-Next
grouper**, leaving the target's layout byte-for-byte unchanged.

1. `_get_kv_cache_groups_glm5_next`: split non-MLA attention specs (the
   drafter's) out before the `type(s) is MLAAttentionSpec` check; after the
   existing groups are built, append `_glm5_next_draft_groups(draft_specs,
   mla_block_size)`: draft specs get `block_size` raised to the target's
   (2304; must be a multiple of the draft's kernel block, else a clear
   `NotImplementedError`), are bucketed by spec equality, and become plain
   `KVCacheGroupSpec`s **appended last** so every existing group id (MLA/idx
   0, tail 1, mamba 2..5) is unchanged.
2. `_glm5_next_tensor_layout`: classifies plain `AttentionSpec` groups (not
   `UniformTypeKVCacheSpecs`, not `MambaSpec`) as draft groups, accepts them
   in the group-count check, and returns them as a ninth tuple element. A
   PP stage without the drafter sees that group with an empty layer list and
   it still classifies.
3. `get_kv_cache_config_from_groups` (GLM branch): per-block bytes gain
   `sum(len(layers) * page)` over draft groups; one **unshared**
   `KVCacheTensor(size=page * num_blocks, shared_by=[layer])` per draft
   layer. The worker's `_allocate_kv_cache` handles single-owner tensors
   generically; nothing in `attn_utils.py` is GLM-specific.
4. `_pool_bytes_per_block` and `_max_memory_usage_bytes_from_groups` (GLM
   branch): same per-block addition; the memory check adds the draft group's
   window-bounded `cdiv(max_memory_usage_bytes, page)` blocks.

Why the draft block size is raised to 2304 rather than left small: after
grouping, `engine/core.py` sets `cache_config.block_size = min(block_size of
groups that participate in prefix caching)` and `kv_cache_utils` takes the
hash block as the gcd of the same set (the kpool tail opts out of prefix
caching precisely so its block_size 4 does not drag those down and "desync
from mamba"). A 16- or 256-token draft group would lower the scheduler block
and the hash granularity for the whole model, which changes chunked-prefill
boundaries, the mamba-align splitter, and prefix hashing. At 2304 nothing
outside the draft group changes. The cost is memory, see below. Alternatives
rejected: `indexes_kv_by_block_stride` on the draft spec would be a lie about
its backend and still leaves the model on the generic (wrong) layout; naming
the indexer in skip-layers is impossible; a block size that divides does not
exist.

### What the patched layout costs (from the CPU run; same numbers as the
engine will compute, given 10,594,000,000 B per worker)

| | groups | bytes/block | num_blocks | draft tensors | MLA token capacity |
|---|---|---|---|---|---|
| PP=1 (one worker) | MLA+idx, tail, 4 mamba, **draft** | 24.42 MiB | 413 | 5 x 929 MiB = 4.5 GiB | 951,552 |
| PP=2 stage 0 | same groups, draft group empty | 5.99 MiB | 548 (min over workers) | 0 | 1,262,592 |
| PP=2 stage 1 | | 18.44 MiB | 548 | 5 x 1233 MiB = **6.0 GiB** | 1,262,592 |

Stage 1 spends 6.0 of its 9.87 GiB of KV on draft tensors that a
2048-window drafter uses at most ~6 blocks of per request. That is the
price of block 2304 for a sliding-window layer under one shared block pool
(each draft layer's tensor has `num_blocks` pages whatever the window). It
is ugly but harmless for the experiment: capacity stays 1.26 M MLA tokens
and stage 0 is nowhere near full. Per 16k-token request the pool demand is
roughly 8 MLA + 1 tail + ~8 mamba + 6 draft blocks -> ~18 concurrent
requests at 16k on 548 blocks; at the 262k max, 4.25. If this matters later,
the lever is a smaller draft block with the draft group opted out of the
scheduler block-size min (a two-line follow-up in the same function plus a
`participates_in_prefix_caching` decision), not a flag.

### Flags to run

Attempt 1's flags, unchanged: `--block-size 2304`, the entrypoint's default
`--kv-cache-dtype fp8_e4m3` (or bf16, both work; fp8 halves the draft
tensors), **no** `--kv-cache-dtype-skip-layers`. Do not add the skip-layers
flag from the DFlash2 cookbook: it makes `Attention.get_kv_cache_spec` emit
the draft spec with `page_size_padded=skip_page_size_padded`, which my
block-size raise does not handle (`replace(block_size=...)` would trip the
`page_size_padded >= unpadded` assert). Keep `--enable-prefix-caching` as the
entrypoint has it; the draft group participates like any sliding-window
group and the hybrid coordinator's fixed-point hit search is generic over
groups. If the boot dies inside `HybridKVCacheCoordinator` /
`find_longest_cache_hit`, retry with `--no-enable-prefix-caching` as the
isolation step and tell me the trace.

### Verified (executed, CPU, `_kv_groups_cpu_test.py`)

- Stock file: the exact production error, PP=1 and PP=2.
- Patched file, PP=1 and PP=2: 7 groups in the order above; draft group
  `SlidingWindowSpec` block 2304 page 2,359,296; `get_kv_cache_configs`
  succeeds for both workers with the numbers in the table; the memory check
  (`_check_enough_kv_cache_memory`) passes; group ids 0..5 identical to the
  drafter-less layout; stage 0 gets no draft tensor.
- The mamba/tail/MLA specs in the test are hand-built to the documented
  formulas (the mamba state shape is a stand-in with a page below the MLA
  page, which is the only property the grouper checks). The real specs come
  from the model; if their pages differ the arithmetic changes but not the
  control flow.

### Not verified

- The runner allocating and reshaping the five draft tensors at block 2304
  (kernel block split via `kernel_block_sizes`; 2304 is a multiple of every
  power-of-two kernel block up to 256, and of 16/32/64 that FA/FlashInfer use).
- `SlidingWindowManager` + hybrid prefix caching with this group mix at
  runtime.
- The draft attention backend under `use_non_causal` with a 2048 window (Part
  2 risk 1, unchanged).

### Failure signatures (Part 3)

| symptom | meaning |
|---|---|
| same `NotImplementedError ... indexer.k_cache` | `kv_cache_utils.py` not mounted (the message is only reachable via the generic path) |
| `NotImplementedError: Layer model.layers.22...: draft block size N does not divide the target block size 2304` | the draft backend's kernel block is not a divisor of 2304; pick a block size that is a multiple of N (e.g. 2048 is not a multiple of kpool*32=128... 2304 is; 2560 is) |
| `AssertionError` in `_allocate_kv_cache` "Some layers are not correctly initialized" | a draft layer got no tensor: the projected group lost its layers (report the group print from `_kv_groups_cpu_test.py` vs the engine log) |
| assert `page_size_padded >= unpadded` | `--kv-cache-dtype-skip-layers` was passed; drop it |
| `ValueError: ... hash_block_size` / block sizes divisibility | prefix-caching gcd includes a group whose block is not 2304; should not happen (draft is raised to 2304), report the sizes in the message |


## Part 4: small draft block (kv_cache_utils.py v2)

Measured with Part 3 (draft block 2304): 711,162 tokens = 2.71 x 262,144.
That capacity number is `num_blocks / (sum over groups of blocks a
max-length request needs) * max_model_len` (`get_kv_cache_capacity`). With
the Part-3 layout a 262k request needs 114 MLA + 1 tail + 4 mamba x 2 (3 if
`num_speculative_blocks` is 1) + 6 draft = 129-133 blocks, so the run had
about **350-360 pool blocks** on stage 1, i.e. an effective stage-1 KV budget
of ~6.3-6.5 GiB at 18.44 MiB/block (not the 10.59 GB I assumed; the
"Available KV cache memory" log line will say which). Of each block, 11.25
MiB (61%) was the five draft pages.

### 1. The draft's natural block size

`Attention.get_kv_cache_spec` (attention.py ~600) emits a `SlidingWindowSpec`
with `block_size = _largest_kernel_block_within(...)`, which with no padded
page returns the backend's **smallest** supported kernel block: FlashAttention,
FlexAttention and Triton advertise `MultipleOf(16)`, FlashInfer `[16, 32, 64]`
(large pages only on Blackwell datacenter). So the drafter's spec arrives at
16 tokens and vLLM expects the grouper to scale it: the backend requires only a
multiple of 16 (the runner's `prepare_kernel_block_sizes` /
`select_common_block_size` splits any spec block into kernel blocks). There is
no natural size; 16 is a floor. Its page at 16 tokens is 16 KiB (4 kv heads x
128 x 2 (K+V) x 1 B fp8); page scales linearly.

### 2. Exempting the draft group from the block-size minimum

Three places read block sizes across groups (all verified in source):

- `engine/core.py ~320`: `cache_config.block_size = min(block_size)` over
  groups with `participates_in_prefix_caching` (falls back to all groups).
- `kv_cache_utils.resolve_kv_cache_block_sizes`: `scheduler_block_size =
  lcm(all group block sizes)`; `hash_block_size = gcd` over participating
  groups (or `prefix_match_unit`).
- `HybridKVCacheCoordinator`: the `block_size % hash_block_size` assert and
  `verify_and_split_kv_cache_groups` (hit lookup) consider participating
  groups only; `cache_blocks` still calls every manager.

So the clean exemption is exactly the kpool tail's mechanism: a spec whose
`participates_in_prefix_caching` is False, with a manager whose prefix-cache
hooks no-op. With that, the draft block only has to **divide** the target
block (so the lcm stays 2304) and be a multiple of the kernel block.

Is opting out honest? Partly. The claim "the drafter's KV is regenerated each
step" is true only for the tokens the target computes in that step:
`precompute_and_store_context_kv` writes K/V for `num_target_tokens` and
earlier positions persist in the draft cache (that is what the 2048 window
attends to). After a prefix-cache hit the target skips the cached tokens, so
the drafter never writes their K/V; with an opted-out group those positions
are fresh (stale) blocks and the drafter attends to garbage until 2048 new
tokens push them out of the window. Output is unaffected (drafts are
verified); acceptance for that request drops toward 1 for up to 2048 generated
tokens. With participation (Part 3) the cached prefix's draft blocks were
shared and valid. Keeping participation with a small block is not possible in
this framework: `BlockHashListWithBlockSize` only scales hashes UP by an
integer factor, so a participating 576-token group would force the hash
block to 576 for the whole model. The benchmark (fresh prompts) never hits a
prefix; multi-turn traffic would. That is the documented trade; if it matters,
the fallback is Part 3's participating variant (set
`VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE=2304` is NOT that: the opt-out is
unconditional in v2; to get Part 3 back, mount the previous file).

Implementation (all in kv_cache_utils.py, tagged DFLASH2-PATCH):

- `DraftSlidingWindowSpec(SlidingWindowSpec)`: `participates_in_prefix_caching
  = False`, own `is_uniform_with_collection`. No new fields, so `replace_as`
  converts the layer's spec in place; pickles to the workers by module path.
- `_ensure_draft_sliding_window_spec_registered()`: registers it with
  `DraftSlidingWindowManager(SlidingWindowManager)` whose
  `find_longest_cache_hit` returns empty, `cache_blocks` is a no-op and
  `get_num_common_prefix_blocks` is 0; `get_num_skipped_tokens` /
  `remove_skipped_blocks` (window eviction, the thing that keeps its block
  count bounded) are inherited. Registered lazily because
  single_type_kv_cache_manager imports this module at its top.
- `_glm5_next_draft_block_size`: default = the largest multiple of the kernel
  block that divides the target block and is at most a quarter of it
  (**576** for 2304, = 9 x 64); `VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE` overrides
  (must divide). 512 does NOT divide 2304 (2304 = 2^8 x 9); the helper
  refuses it with a message naming both numbers.
- `_glm5_next_draft_groups`: sliding-window draft layers become
  `DraftSlidingWindowSpec` at that block; a full-attention draft layer
  (EAGLE3 without a window) keeps the Part-3 behaviour; a padded draft page
  (`--kv-cache-dtype-skip-layers`) is refused.

Why a quarter of the target block and not 16 or 256: every pool block a draft
layer holds also reserves that block's MLA, indexer and mamba pages in the
other tensors (block ids are global), and the sliding-window manager's
admission cap is `cdiv(window - 1 + max_in_flight_tokens, block) + 1` blocks
per request during prefill (`max_in_flight_tokens` = 8192 here). Per-request
draft blocks: 41 at 256, 28 at 384, 19 at 576, 11 at 1152, 6 at 2304
(prefill); ~9 / 7 / 5 / 3 / 2 at decode (window/block + 1). Stage-1 bytes per
block: 8.44 / 9.07 / 10.00 / 12.8 / 18.44 MiB. Against the fixed per-request
cost of the target (MLA cdiv(len/2304) + 1 tail + 8 mamba), 576 gives the best
concurrency during prefill bursts and ties 256 at decode.

### 3. Predicted memory and pool

From `_kv_groups_cpu_test.py` at 576 (stage-1 bytes/block 10.00 MiB, draft
page 576 KiB):

| | Part 3 (2304) | v2 (576) |
|---|---|---|
| stage-1 bytes/block | 18.44 MiB | 10.00 MiB |
| draft share of a block | 11.25 MiB (61%) | 2.81 MiB (28%) |
| blocks per 262k request | 129-133 | 142-146 (draft 6 -> 19, prefill cap) |
| num_blocks, if stage-1 budget = 10.59 GB | 548 | 1010 |
| num_blocks, scaled from the measured ~350-360 | 350-360 | **645-665** |
| five draft tensors, measured budget | 6.0 GiB | **1.77-1.83 GiB** (5 x 576 KiB x blocks) |
| max concurrency at 262k, measured budget | 2.71 (measured) | **4.5-4.7** |
| "GPU KV cache size" tokens, measured budget | 711,162 (measured) | **1.19-1.23 M** |
| same, if the budget really is 10.59 GB | 1.26 M (predicted, not seen) | 1.86 M (7.11x) |

The check: the engine's `GPU KV cache size` line should read about 1.2 M
tokens and `Maximum concurrency ... 4.5x`. If it reads 1.86 M / 7.1x the
stage-1 budget is the full 10.59 GB and the Part-3 run had something else
eating it. Either way the ratio v2/Part-3 should be **1.65-1.70x** in tokens
(18.44/10.00 on blocks, x 129/142 on per-request demand).

What this does for the concurrency collapse: at 8k-token prompts a request
needs 4 MLA + 1 + 8 mamba + 5 draft (decode) = 18 blocks, 32 during its
prefill; with ~650 blocks that is ~36 decoding / ~20 prefilling requests
before queueing, versus ~19 / ~17 with Part 3 (mamba's 8 blocks per request
is now the largest fixed cost; that is the target's, not the drafter's).

### 4. Scheduler chunking and hash granularity: untouched

Verified in `_kv_groups_cpu_test.py` on the produced config: the engine's
min over participating groups = 2304; `resolve_kv_cache_block_sizes` returns
`(scheduler_block_size, hash_block_size) = (2304, 2304)`; a real
`HybridKVCacheCoordinator` built from `generate_scheduler_kv_cache_config`
(exactly what EngineCore hands the scheduler) instantiates with managers
`[FullAttentionManager, KpoolTailManager, MambaManager x4,
DraftSlidingWindowManager]`, its participating attention groups are
`[MLAAttentionSpec, MambaSpec]` only (the draft group is skipped in hit
lookup like the tail), all seven groups carry the EAGLE drop flag as before
(no group is flagged, so the coordinator flags all, unchanged from the MTP
config). Prefill chunk alignment (scheduler.py uses
`cache_config.block_size`), the mamba-align splitter and the indexer's
pool-aligned chunk starts therefore see the same 2304 as without the drafter.

### Verified (executed, CPU)

- `_kv_groups_cpu_test.py` v2, PP=1 and PP=2: group order unchanged (draft
  last, id 6), draft block 576, page 589,824; num_blocks 632 (PP=1) / 1010
  (PP=2) at 10.59 GB; pickle round-trip of the config; manager registration
  through the real registry; the no-op hooks; `resolve_kv_cache_block_sizes`;
  coordinator construction; per-group blocks per 262k request (114, 1, 2, 2,
  2, 2, 19).
- `_dflash2_cpu_test.py` (k=7) still passes with the relaxed block check;
  `_dflash2_k5_test.py` constructs the drafter at k=5 with conv period 6.

### Not verified

- Runtime behaviour of `DraftSlidingWindowManager` under real
  allocation/free (it only removes three prefix-cache hooks from a stock
  manager; the tail manager is the precedent).
- Draft attention at kernel block 64 inside a 576 storage block (the runner
  splits it; same mechanism as 2304).
- The acceptance drop after prefix-cache hits described in 2.

### Failure signatures (Part 4)

| symptom | meaning |
|---|---|
| `NotImplementedError: no draft block size in [16, 576] divides ...` or `VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE=N must be a multiple of ...` | block-size arithmetic; pick a divisor of `--block-size` that is a multiple of 16 |
| `AssertionError: No manager registered for KVCacheSpec DraftSlidingWindowSpec` | the lazy registration did not run before the scheduler was built (grouping happened in another process); report the traceback |
| `AttributeError ... participates_in_prefix_caching` / `replace_as` import error | wrong vLLM build; the file targets 0.1.dev20051 |
| GPU KV cache size unchanged at ~711k | old kv_cache_utils.py still mounted |
| `Each KV cache group's real block_size must be divisible by hash_block_size ... block_sizes=[..., 576]` | the draft group was NOT excluded from the participating set: `participates_in_prefix_caching` did not survive to the scheduler (pickle produced a plain SlidingWindowSpec); report the traceback |
| acceptance ~1 only on requests that hit the prefix cache, normal otherwise | the documented opt-out cost, not a bug |

## Part 5: is `num_speculative_tokens = 7` baked in?

No. `block_size: 8` is the block the drafter was trained on, not a hard
constraint. In the reference (`dflash_generate`) the last block of a
generation is `min(block_size, remaining)` tokens, so the model already runs
on shorter blocks; the convolutions are causal two-tap within the block and
the selector walks slot by slot, both defined for any block length; attention
is non-causal within the block whatever its length. `qwen3_dflash2.py` now
accepts `1 + k <= 8` and sets the conv's causal period to the actual query
block (`1 + k`) instead of the nominal 8, with an info log when they differ.
`1 + k > 8` is still refused (positions beyond training). Nothing else in the
patch set assumes 7: the speculator uses `num_speculative_steps` throughout,
the mask count is k, the aux-state and KV work is k-independent. Acceptance
at k=5 is unmeasured; on the reference's numbers a shorter block loses a
little acceptance length but wins verification cost per step, which is the
49%-efficiency question you raised. Note `MTP k` and this k are independent
settings; the drafter config is only read for the upper bound.

## Log of the Part 1 container check

    imports ok
    Glm5NextForCausalLM supports_eagle3(class)= True flag= True
    Glm5NextForConditionalGeneration supports_eagle3(class)= True flag= True
    sorted ids (6, 15, 25, 34, 43)
    start_layer 0 incoming () []
    start_layer 23 incoming (6, 15) ['aux_hidden_states.6', 'aux_hidden_states.15']
    range check: aux hidden state layer ids [0, 46] out of range 1..45 (ids are 1-based: id L is the output of layer L-1)
    hc_contract [[4.5, 5.5, 6.5], [16.5, 17.5, 18.5]] dtype torch.bfloat16
    pp indices [(0, 23), (23, 45)]
