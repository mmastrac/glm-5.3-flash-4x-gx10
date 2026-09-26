# GLM-5.3-Flash on 4× ASUS GX10 (GB10), TP=4

GLM-5.3-Flash (NVFP4) served by vLLM at TP=4 across four GB10 boxes over
RoCE, with DFlash2 speculative decoding and a 524k context window. The base
is a stock vLLM nightly plus a handful of small patches and a newer FlashKDA.
Ray is replaced by [mentat](https://github.com/mmastrac/mentat).

Measured 2026-09-23 on the previous nightly (0961bbae), first-touch,
temperature 0, single stream, nvidia/GLM-5.3-Flash-NVFP4, both ConnectX-7 PCIe
roots in use. The current nightly (ddd6fbca) has not been measured yet.

| | |
|---|---|
| prefill @32k | ~2,600 tok/s |
| prefill @126k | 3,365 tok/s (37.6 s) |
| decode, counting | 113.7 tok/s |
| decode, code | 88.6 tok/s |
| decode, prose | 59.9 tok/s |
| KV pool | 2,632,595 tokens (26 GiB, fp8_e4m3) |
| needle recall, 33k and 136k at three depths | 6/6 |

[model.yaml](model.yaml) has the checkpoints and the memory footprint, and
[KNOBS.md](KNOBS.md) lists every environment variable the entrypoint reads.
[NOTES.md](NOTES.md) has the measurements and diagnosis behind the choices
here.

## What you need

- **Four ASUS GX10 or other GB10 boxes** (sm_121a, 128 GB unified memory).
  The model takes all of each box: ~91.9 GiB of GPU allocations per rank, with
  1.5-3 GiB left free (2026-09-23). Nothing else runs beside it.
- **A ConnectX-7 fabric between all four**, through one switch (ours is a
  MikroTik CRS812 at 200G), with RoCE working. Each box needs a static IPv4
  on its ConnectX interface, all in one subnet (`CLUSTER_SUBNET`), MTU 9000.
  For full prefill speed also give the ConnectX-7's second PCIe root an
  address in a second subnet on every box (`FABRIC_SUBNETS`, see Tuning).
- **A LAN between all four** that your clients can reach. mentat identifies
  each box by its LAN address, and the API is served on it.
- **The weights on each box's local disk**, not on NFS: every rank reads the
  whole checkpoint, and an NFS mount races the network at boot.
  - [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4),
    181 GiB, at `MODEL_HOST_DIR`
  - [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
    the drafter, 2.2 GiB, at `DFLASH_HOST_DIR`. It is licensed CC BY-NC-ND
    4.0, non-commercial; check that before you serve it. `SPEC_METHOD=mtp`
    uses the checkpoint's own MTP head instead, slower but with no second
    download.

  `hf download` is resumable.
- **Docker with the NVIDIA container runtime and Compose v2** on every box,
  and `/dev/infiniband` present on the host.

## Ports

| port | what | where |
|---|---|---|
| 6379, 6380 | mentatd control and HTTP | every box |
| 6381 | mentatd-serve: the OpenAI API and merged MCP for clients | one box, the head by default |
| 6382/udp | mentatd announcements | every box |
| 8002 | vLLM's OpenAI API (`/v1/chat/completions`, `/v1/models`, `/metrics`) | head only |
| 8082 | status page and MCP (`/mcp`) | every box |

Point clients at mentatd-serve on `:6381`, not at vLLM on `:8002`. It
health-gates, so a request during a boot or a reload waits instead of
failing, and it routes by model name, so another model on the same boxes
answers on the same address. mentatd-serve is a separate process from the
daemon by design: nothing that routes inference traffic runs inside the thing
that holds cluster membership.

## Layout

| path | what |
|---|---|
| `image/` | Dockerfile, entrypoint, patches, `verify-base.py`, `self-test.py`, chat template, `build.sh` |
| `compose/glm53.yaml` | the model, the same file on every box |
| `.env.example` | per-host and per-box values; copy to `compose/.env` |
| `smoketest/` | `run.sh <base> [served-name]` |
| `.submodules/spark-agent` | the status server, reached through the `vllm` symlink |
| `dev/` | not in the image: the corruption diagnosis and repros, kernel and patch tests, the step tap |
| `attic/` | used by nothing: the pre-nightly host patches and overrides |

mentatd and mentatd-serve come from the [mentat](https://github.com/mmastrac/mentat)
repo, with their own compose files (step 5).

## 1. Get the repo onto every box

    git clone --recursive https://github.com/mmastrac/glm-5.3-flash-4x-gx10
    # or, in an existing clone: git submodule update --init

The status server comes from the
[spark-agent](https://github.com/mmastrac/spark-agent) submodule, and the
image build fails without it.

## 2. Build the image

    image/build.sh                       # -> spark-glm53:v8
    TAG=spark-glm53:v9 image/build.sh
    BASE=... image/build.sh              # override the pinned nightly

The build context is the repo root (`docker build -f image/Dockerfile .`), so
the `vllm` symlink into the submodule stays inside it. Build on a GX10: there
is no cross-build for aarch64 here. A plain `rsync -a` copy of the repo also
builds, because it keeps the hidden `.submodules/` and the symlink; a bare
`scp -r *` misses the first. Build once and copy the image to the other three
boxes (`docker save | ssh ... docker load`, or a registry): every rank must
run the same image.

The base is `vllm/vllm-openai:nightly-ddd6fbca`, which carries `glm5_next`
and DFlash2 upstream, and the patches are small anchored edits that fail the
build if the tree moves under them. The one thing the build compiles is
FlashKDA (see Patches), in a builder stage that took 98 s on a GX10.
`image/verify-base.py` then checks the finished tree. The image embeds mentat
0.12.0, which refuses daemons older than 0.9, so `mentatd` and `mentatd-serve`
should be 0.12.0 too.

## 3. Fill in compose/.env

On every box:

    cp .env.example compose/.env

Compose reads `.env` from `compose/`, beside the compose file, not from the
directory you run it in.

| variable | value |
|---|---|
| `IMAGE` | the tag you built |
| `HEAD_HOST` | the head's LAN address, the same on every box, the head included |
| `CLUSTER_SUBNET` | the fabric subnet prefix with its trailing dot, e.g. `10.0.0.` |
| `MODEL_HOST_DIR`, `DFLASH_HOST_DIR` | where the checkpoint and drafter are on this box |
| `EXT_DIR` | any directory; see below |
| `CACHE_HOME`, `LOG_DIR` | writable directories for JIT caches and logs |
| `ROLE` | `head` on one box, `worker` on the other three |
| `VLLM_HOST_IP` | **this** box's LAN address |

Only `ROLE` and `VLLM_HOST_IP` differ between boxes. Leave every tuned knob
out: each has its default in `image/entrypoint.sh`, and a copy in `.env`
silently wins over the measured value.

`EXT_DIR` is mounted at `/opt/ext` for LibertAI's sparse-MLA kernel plugin,
which is used only when `VLLM_GLM53_CUDA_SPARSE_MLA` is set. This recipe does
not set it and serves on the MiaAI path (see Patches), so an empty directory
is fine.

## 4. Patch the checkpoint's chat template

Thinking off needs a fix in the chat template, and the template the image
uses is not always its own. `image/chat-template.jinja` carries the fix and
is baked into the image, but the entrypoint prefers the checkpoint's own
`chat_template.jinja` whenever that one handles images (it contains
`<|begin_of_image|>`), and the nvidia checkpoint's does. So edit the
checkpoint's template on every box, keeping the original beside it:

    cd "$MODEL_HOST_DIR"
    cp chat_template.jinja chat_template.jinja.orig

Then make the same two changes `image/chat-template.jinja` makes:

1. The line that sets `effective_reasoning_effort` maps thinking off to low
   effort:

   ```jinja
   {%- set thinking_off = (thinking is defined and not thinking) or (enable_thinking is defined and not enable_thinking) -%}
   {%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else ('low' if thinking_off else 'max') -%}
   ```

2. The generation prompt at the end always opens `<think>`, never an empty
   `<think></think>`:

   ```jinja
   {%- if add_generation_prompt -%}
       <|assistant|>{{- '<think>' -}}
   {%- endif -%}
   ```

Check that the template still renders before you boot (it uses
`{% break %}`, so load it with
`jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])`). The edit is
lost whenever the checkpoint is downloaded again. The smoketest's two thinking
cases catch a missing edit; checking only that `reasoning` is empty does not,
because it is empty in the broken state too, with the trace in `content`.

Why this matters: see "Long output repeats or skips when thinking is off"
under Troubleshooting.

## 5. Start mentatd, and mentatd-serve on the head

mentat has its own repo, compose files and `.env`. On every box, in a
checkout of [mmastrac/mentat](https://github.com/mmastrac/mentat) at `v0.12.0`:

    VERSION=0.12.0 ./build.sh
    echo "MENTAT_PEERS=<head LAN address>:6379" > .env
    docker compose -f mentatd.yaml up -d

The daemon names the box by its default route's address, which must be the
`VLLM_HOST_IP` you gave the model. Set `MENTAT_NODE_IP` in mentat's `.env`
when it is not. The model container registers with the daemon on
`127.0.0.1:6379`. Then, on the head only:

    docker compose -f mentatd-serve.yaml up -d

mentat's `mentatd.yaml` explains the optional settings: interface ranking and
fabric tags (`MENTAT_ANNOUNCE_IFACES`), and signing announcements
(`MENTAT_SECRET`, which must then be set on every box).

## 6. Start the model

On every box:

    docker compose -f compose/glm53.yaml up -d

From cold, the boxes may start in any order: registration retries until the
daemon answers, and the head waits for all four GPUs before it loads. The
weight load takes about ten minutes. The status page on `:8082` answers from
container start, so there is something to read while it loads.

**Replacing a running stack is different.** Recreating all four ranks at once
lets a starting rank join a group that is still tearing down, and the whole
group then hangs just past NCCL setup: every rank `running`, restart count 0,
CPU ~1%, no weights loading, and nothing in any log after the
`custom_all_reduce` warning. Seen on 2026-09-10 after several rapid recreate
cycles. Take every rank down, confirm all four containers are gone, then start
the head and the workers a few seconds later:

    # on each box
    docker compose -f compose/glm53.yaml down --timeout 60
    # confirm on all four: docker ps -a | grep glm53  ->  nothing
    # then the head first, the workers after

The project name is pinned to `glm53`. A stack started under another project
name (an older checkout run from a different directory, say) must be taken
down with its own compose file first, or the two collide on the container
name.

## 7. Check it

    smoketest/run.sh http://<head>:6381

Eight cases, each with an answer that can be checked, because this model can
load cleanly, report healthy and serve fluent nonsense. The first fails
against any model that is not a GLM-5.3 checkpoint. Exit status is the number
of failures. See [smoketest/README.md](smoketest/README.md).

Send one throwaway request before timing anything: the first request after a
cold boot JIT-compiles DFlash2 shapes and can take minutes.

## Roll back

The compose file here leaves every tuned knob to the image's entrypoint. The
previous version of this recipe (commit `dde02f4`) did the opposite: its
`.env` set every tuned knob, it needed `compose/dflash2.yaml` as a second
file, and its image's entrypoint defaults described a TP=2 boot. So an older
image goes back with its own tree, not with this one:

    docker compose -f compose/glm53.yaml down --timeout 60   # on all four
    git checkout dde02f4
    # rebuild that tree's image, restore its .env beside its compose files,
    # and bring it up as its README says

Keep the old image and `.env` until the new one has served for a while.

## Diagnostics

Each container runs spark-agent's status server on `:8082` from container
start, before the model is loadable. It does not proxy inference. The router's
`/mcp` merges its tools under the `glm53__` prefix: `node_status`,
`cluster_status`, `metrics`, `serve_args`, `throughput`,
`latency_percentiles`, `cache_sizes`, `ray_status`, `versions`, and
`find_files` / `search_files` over a fixed set of roots. `/memory` on the
status page says whether torch or something else holds a box's memory.

> **`:8082` is unauthenticated and binds `0.0.0.0`.** Anyone who can reach the
> port can read engine state and search file contents under `SEARCH_ROOTS`
> (the installed vLLM package, `/root/.cache`, `/logs` and `/cache` by
> default). Paths are resolved with `realpath` so symlinks cannot escape those
> roots, arguments are passed as argv rather than through a shell, and output
> is capped at 16 KB, but there is no authentication. Fine on an isolated
> network; put it behind something, or narrow `SEARCH_ROOTS`, on any network
> you do not control.

Host-level facts (GPU, PCI, RDMA counters, dmesg, systemd) come from
spark-agent's separate per-machine agent, which is not part of this recipe.
The container tools above cover the engine; diagnosing the fabric or the box
needs that agent or plain ssh.

## Tuning

This is tuned for one or two interactive users, not for a serving fleet. The
bias throughout is that **a long prefill must never block a short request**,
and that a single stream should be fast, rather than maximising aggregate
throughput at concurrency. Every value below is the entrypoint's default.

| knob | value | why |
|---|---|---|
| `LONG_PREFILL_TOKEN_THRESHOLD` | 2304 | Caps one prefill's share of each scheduler step. Left at the default (budget − 256) a 120k prefill takes the whole step and a 12-token request waits 78–90 s; at 2304 it waited 4.83 s (2026-09-06). Must be a multiple of 2304, the KDA block size, because prefix caching snaps chunk ends to it: 2048 yields alternating 2048/256-token chunks. Costs nothing: the 200k prefill got *faster*. |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | Measured the same as 8192 at 200k once chunks are capped (234.1 s against 237.7 s, 2026-09-06). |
| `KV_CACHE_MEMORY` | 26 GiB | 2.63M tokens with DFlash2. Pinned, `--gpu-memory-utilization` no longer sizes the pool, and vLLM says so at startup. At 28 GiB the head sat near 1 GiB free and eight long requests had a worker OOM-killed. |
| `FABRIC_SUBNETS` | both roots | Each GB10's ConnectX-7 sits on two PCIe roots and one root tops out near 110 Gb/s. NCCL over both took a 126k prefill from 2,412 to 3,365 tok/s (2026-09-23); decode did not move. Needs an IPv4 on the second root's interface in its own subnet, MTU 9000, and the same RoCE v2 GID index on both roots. Set it in `.env`; empty uses `CLUSTER_SUBNET` alone. |
| `MAX_NUM_SEQS` | 32 | Capped by the KDA recurrent state: exactly 32 fit after weights. With DFlash2 k=7 a decode step costs 1+k=8 token slots per sequence. |
| DFlash2 `k=7` | | Decodes 109.8 / 88.8 / 52.6 tok/s structured / code / prose, where the checkpoint's own MTP head at k=4 gave 57.2 / 54.4 / 45.6 on an earlier image (both 2026-09-23). Costs ~41% of the KV pool: 3.44M tokens with speculation off, 2.02M with it at the same pin, on the pre-nightly image (2026-09-06). |
| `MOE_BACKEND` | `flashinfer_cutlass` | The native NVFP4 kernel, reading the checkpoint's own input scales. `marlin` (weight-only) also works. |
| `SAFETENSORS_LOAD_STRATEGY` | eager | Loads in 511 s against 690 s for lazy. Unpinned, eager's buffers cost 38% of the KV cache; with the pin they cost nothing. |
| `busy_loop_s` | 0.002 | See Patches. Raises decode and drops the SoC ~20 °C. |
| `GPU_MEM_UTIL` | 0.88 | 0.90 passes every startup check and wedges the box hours later. See Troubleshooting. |

If you are serving many concurrent users instead, raise `MAX_NUM_SEQS` as far
as the KDA state allows, drop the KV pin back toward 20 GiB, and consider a
larger `LONG_PREFILL_TOKEN_THRESHOLD`: the fairness reserve costs a solo user
about 5% and buys nothing when every step has several requests in it anyway.

## Patches

Applied at build time from `image/patches/`; each asserts its anchor matches
exactly once. `image/verify-base.py` then checks the finished tree, reading
files as text (an import-based check needs a GPU driver that does not exist
during `docker build`).

| file | what it does | source |
|---|---|---|
| `glm53-flash_SM121.py` | Makes the model run on GB10 at all. On capability 12 the nightly offers only `FLASHINFER_MLA_SPARSE_SM120`, which needs the packed `fp8_ds_mla` layout with `pe_dim == 64`; this checkpoint is NoPE, so the engine dies in `concat_and_cache_mla` after a full weight load. Lists the SM90 sparse-MLA path for capability 12 and swaps FA3 for FA2. | [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark), MIT |
| `mia_retarget.py` | Two of MiaAI's edits target code that has moved since: the indexer allocation, now in `models/glm5next/nvidia/sparse_indexer.py`, and FlashInfer 0.7.0's FA2 fp8 gate. Rewrites their paths and anchors in a copy, so the vendored file stays verbatim. | ours |
| `sm90_fp8_kv_dtype.py` | The SM90 backend planned an fp8 KV cache as uint8. | ours |
| `gb10_plugin_backend.py` | Lets `VLLM_GLM53_CUDA_SPARSE_MLA` pick between the two sm_121 MLA kernels, which otherwise collide silently. | ours |
| `glm53_mtp_bf16.py` | nvidia's `config.json` says the MTP layer is NVFP4, but its weights are BF16, so MTP dies at load. Excludes it from quantisation when the checkpoint holds no scales for it. | ours |
| `glm53_eagle3_aux.py` | DFlash2 reads auxiliary hidden states from target layers 5, 14, 24, 33 and 42; upstream GLM5next does not expose them. | ours |
| `glm53_dflash2_kv_groups.py` | The GLM-5-Next KV grouper gives up on the drafter's sliding-window layers and the model dies unifying page sizes. Keeps the target's groups and adds the drafter's. | ours |
| `vllm-58720-routed-experts.patch` | Indexes the expert mapping once per load instead of scanning it for every checkpoint tensor. Merged after this nightly. | [vllm#58720](https://github.com/vllm-project/vllm/pull/58720) |
| `image/flashkda/` | Rebuilds `vllm/_flashkda_C` from FlashKDA 17a037d. The nightly's b59532f rounds the KDA recurrent state to bf16 every 16 tokens, and long prefills then corrupt tool-call output; 17a037d keeps it in fp32. | [vllm#58846](https://github.com/vllm-project/vllm/pull/58846), open |
| `glm53_reasoning_always_parsed.py` | Thinking off maps to low reasoning effort (see step 4), so the model always emits a short `<think>` block. Stock `glm47_moe` stops parsing `<think>` when thinking is off, and the trace would land in `content`. | ours |
| `glm47_failclosed.py` | Tool-call parser plugin (`--tool-call-parser glm47_failclosed`). Checks each call against the tools the request offered; a call with a bad name or argument keys comes back as a retryable call whose sentinel argument names the mistake, instead of being dropped or leaking into history. Containment, not a cure. | ours, after [NNNtrance](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark) #7 and #11 |
| `spin_wait.py` | vLLM's shm queue spins for `busy_loop_s` (1 s) after each message; on GB10 the CPU and GPU share one power budget, so the spin costs ~20 °C and decode. 0.002 keeps the fast path; 0 (always block) measured slower. | [nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) |
| `worker_memory_cap.py` | Caps each worker's share of unified memory (`TORCH_MEM_FRACTION`, 0.92). vLLM never calls `set_per_process_memory_fraction`, so nothing else bounds a worker. | ours |
| `spark_mem_trace.py` | Names whatever crosses that bound, instead of leaving an OOM anonymous. | ours |
| `link_cuda_headers.sh` | The base ships CUDA libraries without their headers where nvcc looks, which breaks FlashInfer JIT at link time. | ours |

`dev/patch-tests/` holds tests for two of the patches; the image does not use
them. The old recipe's `gb10_topk_fallback.py` is now a flag
(`--sparse-indexer-topk-backend per_row`). `thinking_budget_guard.py` and
`glm53_kpool_tail_ring.py` (the spec-decode tail ring,
[vllm#58454](https://github.com/vllm-project/vllm/pull/58454)) are upstream.

## Troubleshooting

**Prefill at half speed, every metric healthy.** If NCCL all-reduce crawls
(~12 Gb/s) while `ib_write_bw` reads a healthy 109 Gb/s and no error counter
moves, the ConnectX-7 has latched a slow fallback state from the DAC cables
being hot-plugged. **Power off and unplug for a minute**: a reboot does not
clear it, and neither does a NIC hotplug reset. The tell is that NCCL Tree
beats Ring; healthy is Ring 110 Gb/s, Tree 44. A single unidirectional stream
cannot see this, which is why the RDMA test passes; a ring collective, sending
and receiving at once, can.

**The box wedges hours after a clean start.** `GPU_MEM_UTIL=0.90` passes every
startup check and is worth +28% KV, then takes the box down with no ssh, no
userspace and ping only (2026-08-27). Unified memory means the CUDA allocation
*is* host memory, so none of it is reclaimable and the OOM killer cannot help.
0.88 is the default.

**Long output repeats or skips when thinking is off.** "Count from 1 to 200"
comes back as `35 36 37 37`, or jumps ahead, or starts copying the prompt.
This is the model, not the stack. GLM-5.3-Flash has no non-thinking mode: its
official template always opens `<think>` under `Reasoning Effort: Max`, and an
empty `<think></think>` is out of distribution. Every checkpoint (nvidia,
RedHatAI, the official FP8), both MoE kernels, TP=2 and TP=4, and Hugging
Face's own `glm5_next` implementation fail the same way (2026-09-23). The
template fix in step 4 maps `thinking: false` / `enable_thinking: false` to
`Reasoning Effort: Low` instead: a few dozen tokens of reasoning and a clean
answer. `reasoning_effort` (`low`, `high`, default `max`) also works directly,
at the top level of the request. If long output still breaks with thinking
off, the checkpoint's template has lost the edit.

**Intermittent corrupted tokens.** The LibertAIDAI modelopt NVFP4 build emits
them mid-word, inside rare tokens: invisible in English, reproducible with a
Korean prompt, and identical under both MoE kernels, both attention backends,
and with speculation off (2026-09-06). The nvidia build this recipe uses and
the compressed-tensors builds (RedHatAI NVFP4, INT4 AWQ) are clean on the same
stack. Independently reported by tonyd2wild.

**Boot hangs at `waiting for 4 GPUs, have 1`.** Every box must set `HEAD_HOST`
to the head, not to itself. mentat replicates an agent's *registration*
across the mesh but not its *liveness*: point a box at its own daemon and the
head lists the agent yet marks it `alive=false degraded=true`, while that
box's own daemon sees only its own agent. The GPU gate asks whichever daemon
it was told about, so it never counts more than one (seen 2026-09-06). Confirm
with `docker exec mentatd mentatd status` on the head: every agent should read
`alive=true`.

**The model waits in placement with nothing in the container log.** If
`MENTAT_ANNOUNCE_IFACES` tags a link `rdma` and the daemons' probes over that
link fail, mentat will not place the four ranks, and the group waits for
`MENTAT_PG_PENDING_TIMEOUT_MS` (10 minutes). `pending_reason` in the
daemon's `/status` (HTTP, port 6380) names the constraint. Fix the link, or set `MENTAT_ISLAND_PLACEMENT=off` and restart
the daemons.

**The second boot deadlocks after NCCL setup.** A persisted FlashInfer
autotune cache keys some MoE entries per rank, so rank 0 loads a tuning the
others lack and they wait on each other forever (2026-09-23). The entrypoint
keeps the cache ephemeral at TP>1 and wipes it at every start, which costs
about two minutes of autotune a boot; `AUTOTUNE_CACHE=persist` brings the old
behaviour back.

**The first request after a cold boot takes minutes.** Triton JIT-compiles
DFlash2 shapes mid-serve. Send one throwaway request before timing anything.

**A value in `.env` does nothing.** `.env` only substitutes into the compose
file; a variable reaches the container only if `compose/glm53.yaml` lists it
under `environment`. And `.env` must be in `compose/`, not in the repo root.

**NCCL fails every TP init after a reboot.** Do not pin `NCCL_IB_GID_INDEX`.
The RoCE GID table is indexed by (address, RoCE version) in the order
addresses appeared, so boxes do not agree and a removed address leaves a hole;
the right index moved from 6 to 5 on one box across a reboot (2026-08-26). The
entrypoint derives it at every start. To read a table:

    for i in $(seq 0 9); do p=/sys/class/infiniband/<dev>/ports/1; \
      echo "$i $(cat $p/gid_attrs/types/$i) $(cat $p/gids/$i)"; done

The right entry is the RoCE v2 one for the box's static fabric address.

## Credits

[mmastrac/mentat](https://github.com/mmastrac/mentat) ·
[mmastrac/spark-agent](https://github.com/mmastrac/spark-agent) ·
[tonyd2wild](https://github.com/tonyd2wild) ·
[MiaAI-Lab](https://github.com/MiaAI-Lab) (sm_121 patches, see
`image/patches/LICENSE.MiaAI-Lab`) ·
[tonyliu312](https://github.com/tonyliu312) (28 GiB KV pin) ·
[nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) (spin wait) ·
[alexellis](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
