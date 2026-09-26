# glm53 step tap

Per-step, per-rank instrumentation for the GLM-5.3-Flash corruption hunt.
Opt-in through `GLM53_TAP=1`; costs one env lookup when off.

Files:

- `glm53_step_tap.py`  the tap (runtime module, no rebuild needed)
- `glm53_step_tap.pth` one-line loader; Python imports it at interpreter start
  from any site directory
- `glm53_tap_report.py` offline report over the per-rank JSONL files

## What it records (per step, per rank)

- rows: request id, state index, position, input token, device `seq_len`,
  host `seq_lens_cpu_upper_bound`, host and device `num_computed_tokens`
- pairs: rows in the same batch with the same position and the same token
  prefix (the S2 setup). Every checkpoint is compared row A vs row B on the
  device, in execution order; `first_diff` names the first checkpoint where
  the rows differ, `diffs` lists them all.
- checkpoints without `GLM53_TAP_DEEP` (work under torch.compile and
  piecewise/breakable graphs, since they sit at custom-op boundaries):
  - `L<i>.kda.{qkv,g1,beta}` inputs, `L<i>.kda.out` output,
    `L<i>.kda.{conv,rec}_{before,after}` the request's own state block,
    `L<i>.kda.state_idx`
  - `L<i>.mla.{q,kvc,kpe}` inputs, `L<i>.mla.attn_out`, `L<i>.mla.slot`,
    `L<i>.mla.cache_row` (the cache row just written, read back),
    `L<i>.mla.cache_reread` (the row written last step, read again)
  - `L<i>.mla.{planned,valid}` SM90 planned kv_len vs the indexer's valid
    count for the row, `L<i>.mla.topk` logical indices, `L<i>.mla.kvidx`
    physical indices, `L<i>.mla.out` the SM90 output
  - `L<i>.idx.{hidden,q,k,w,gate,topk,slot,tail_slot}`
- checkpoints with `GLM53_TAP_DEEP=1` (pure eager only): forward hooks on
  every decoder layer and its `mhc_pre_op`, `mhc_fused_post_pre_op`,
  `self_attn`, `mhc_post_op`, `mlp`, `input_layernorm`,
  `post_attention_layernorm`, as `L<i>.<sub>.in<k>` / `.out<k>`. A second
  call of the same module in a layer gets `@1` (the fused hc op runs twice:
  `hc_fused` = attn pre, `hc_fused@1` = ffn pre).
- events: `host_dev_mismatch`, `plan_mismatch`, `state_changed` (a KDA
  state block read at step t differs from what the same request wrote at
  t-1), `kv_changed` (an MLA cache row changed between the write and the
  next step's re-read), sampler `argmax`/`top2`/`gap`/`sampled` for the rows
  this rank samples, `fed` = `last_sampled_tokens` per request after
  post_update (what this rank feeds next), `nc_after`.

## Running it

Bind-mount the module and loader into the image's site dir on every box,
and set the env. No image rebuild. With the compose layering, put the env in
`.env` (it beats compose and the entrypoint).

```
# on each GLM box, e.g. as an extra compose volume / env, or on a throwaway run:
-v /home/admin/tap/glm53_step_tap.py:/usr/local/lib/python3.12/dist-packages/glm53_step_tap.py:ro
-v /home/admin/tap/glm53_step_tap.pth:/usr/local/lib/python3.12/dist-packages/glm53_step_tap.pth:ro
-v /home/admin/tap/out:/tmp/glm53-tap
-e GLM53_TAP=1 -e GLM53_TAP_DIR=/tmp/glm53-tap -e GLM53_TAP_VECTORS=1
# for sublayer-level localisation:
-e GLM53_TAP_DEEP=1   plus  --enforce-eager  (EXTRA_ARGS; eager still reproduces per EXP2)
```

Worker processes inherit the env and run site at start, so every rank taps
itself; the stderr line `[glm53-tap] rank N armed, writing ...` confirms it.
The vLLM worker log also shows `[glm53-tap] runner patched` / `KDA patched`
/ `MLAAttention patched` / `SM90 forward_mqa patched` / `indexer patched` /
`sampler patched` as the modules import.

Rebuild variant: `COPY patches/tap/glm53_step_tap.py patches/tap/glm53_step_tap.pth /usr/local/lib/python3.12/dist-packages/`.

Then send the S2 pair (two identical raw-token-id completions in flight
together), collect `tap-rank*.jsonl` from all four boxes into one directory,
and run:

```
python3 glm53_tap_report.py DIR                 # pairs, events, cross-rank, tokens
python3 glm53_tap_report.py DIR --req cmpl-abc  # one request
python3 glm53_tap_report.py DIR --steps 200-320 --all-pairs
```

Reading the pair section: `FIRST SPLIT at L7.mlp.out0` with `L7.mlp.in0`
absent from `differing` means the MoE of layer 7 split two identical rows;
`L12.kda.out` first with `L12.kda.qkv` and `L12.kda.{conv,rec}_before`
equal means the KDA kernels did; `L12.kda.rec_before` first means the two
requests' state blocks already differed when the step began (look at the
earlier steps' `state_changed`). `L3.mla.out` first with `L3.mla.q`,
`L3.mla.topk`, `L3.mla.cache_row` equal points at the SM90 kernel or its
plan (`plan_mismatch` says whether the planned length was wrong).

Vectors: `GLM53_TAP_VECTORS=1` dumps every checkpoint's tapped rows to
`vec-rank<r>-step<N>.pt` at the step a pair first splits (2 = every step;
about 140 MB per step per rank with DEEP on). Load with `torch.load`; the
`cps` dict maps checkpoint name to `[rows, bytes-as-int32]` tensors in the
same row order as `rows`.

Self-test (no vLLM):

```
sudo docker run --rm --gpus all -v /home/admin/tap:/tap -e GLM53_TAP=1 \
  --entrypoint python3 localhost:5000/spark-glm53:v4 /tap/glm53_step_tap.py --selftest
```
