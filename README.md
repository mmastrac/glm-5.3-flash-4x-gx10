# GLM-5.3-Flash on 4× ASUS GX10 (GB10), TP=4

vLLM TP=4 across four GB10 boxes over RoCE, DFlash2 speculative decoding,
524k context, on a stock vLLM nightly plus a handful of small patches. Ray is
replaced by [mentat](https://github.com/mmastrac/mentat).

Measured 2026-09-23, first-touch, temperature 0, single stream,
nvidia/GLM-5.3-Flash-NVFP4, both ConnectX-7 PCIe roots in use:

| | |
|---|---|
| prefill @32k | ~2,600 tok/s |
| prefill @126k | 3,365 tok/s (37.6 s) |
| decode, counting | 113.7 tok/s |
| decode, code | 88.6 tok/s |
| decode, prose | 59.9 tok/s |
| KV pool | 2,632,595 tokens (26 GiB, fp8_e4m3) |
| needle recall, 33k and 136k at three depths | 6/6 |

Hardware: 4× ASUS GX10 (GB10, sm_121a, 128 GB unified), ConnectX-7 200G RoCE
through a MikroTik CRS812, 10 GbE for management.

## Run it

```
cp .env.example .env        # on every node; set ROLE, and SPARK_HOME if not /home/admin
./scripts/up.sh 192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13   # head first, then workers
```

`up.sh` starts mentatd on every node, mentatd-serve on the head, then vLLM
everywhere, and waits for `:8002`.

**Point clients at mentatd-serve on `:6381`, not at vLLM on `:8002`.** It is a
single OpenAI endpoint for the whole cluster: it health-gates, so a request
during a boot or a reload queues instead of failing, and it routes by model
name, so the other models on the fleet answer on the same address. `:8002` is
one rank of one model and moves whenever the head does. mentatd-serve is a
separate process from the daemon by design — nothing that routes inference
traffic runs inside the thing that holds cluster membership.

Build with `docker build -t spark-glm53:v6 image/`. It compiles nothing: the
base is `vllm/vllm-openai:nightly-0961bbae`, which carries `glm5_next` and
DFlash2 upstream, and the patches are small anchored edits that fail the build
if the tree moves under them. The image embeds mentat 0.12.0, which refuses
daemons older than 0.9, so `mentatd` and `mentatd-serve` must be 0.12.0 too.

## Diagnostics

Each container runs `status-server.py` on `:8082` from container start — before
the model is loadable, so there is something to read during the ~9 minute weight
load. It does not proxy inference. `MENTAT_MCP_API` points at it. Eleven MCP
tools: `node_status`, `cluster_status`, `serve_args`, `ray_status`, `metrics`,
`throughput`, `latency_percentiles`, `cache_sizes`, `versions`, and
`find_files` / `search_files` over a fixed set of roots.

> **`:8082` is unauthenticated and binds `0.0.0.0`.** Anyone who can reach the
> port can read engine state and search file contents under `SEARCH_ROOTS`
> (`/src/vllm,/root/.cache,/logs,/cache` by default). Paths are resolved with
> `realpath` so symlinks cannot escape those roots, arguments are passed as
> argv rather than through a shell, and output is capped at 16 KB — but there
> is no authentication. Fine on an isolated fabric; put it behind something, or
> narrow `SEARCH_ROOTS`, on any network you do not control.

Host-level facts — GPU, PCI, RDMA counters, memory accounting, dmesg, systemd —
come from a separate per-machine agent that is **not** part of this recipe. It
runs outside the container because those facts are the host's, and it
authenticates with the mesh key. The container tools above cover the engine;
diagnosing the fabric or the box needs that agent or plain ssh.

## Tuning

This is tuned for one or two interactive users, not for a serving fleet. The
bias throughout is that **a long prefill must never block a short request**, and
that a single stream should be fast, rather than maximising aggregate
throughput at concurrency.

| knob | value | why |
|---|---|---|
| `LONG_PREFILL_TOKEN_THRESHOLD` | 2304 | Caps one prefill's share of each scheduler step. Left at the default (budget − 256) a 120k prefill takes the whole step and a 12-token request waits 78–90 s; at 2304 it waits ~5 s. Must be a multiple of 2304, the KDA block size, because prefix caching snaps chunk ends to it — 2048 yields alternating 2048/256-token chunks. Costs nothing: the 200k prefill got *faster*. |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | Measured indistinguishable from 8192 at 200k once chunks are capped. |
| `KV_CACHE_MEMORY` | 26 GiB | 2.63M tokens with DFlash2. Pin it explicitly — `--gpu-memory-utilization` then becomes decorative and vLLM says so at startup. |
| `FABRIC_SUBNETS` | both roots | Each GB10's ConnectX-7 sits on two PCIe roots and one root tops out near 110 Gb/s. NCCL over both took a 126k prefill from 2,412 to 3,365 tok/s; decode did not move. Needs an IPv4 on the second root's interface in its own subnet, MTU 9000, and the same RoCE v2 GID index on both roots. At TP=4 it cost no measurable headroom. |
| `MAX_NUM_SEQS` | 32 | With DFlash2 k=7 a decode step costs 1+k=8 token slots per sequence, so 32 sequences exactly consume a 256-token reserve. |
| DFlash2 `k=7` | | Roughly doubles decode against MTP k=4 on this model. Costs ~41% of the KV pool: 3.44M tokens with speculation off, 2.02M with it at the same pin. |
| `busy_loop_s` | 0.002 | See Patches. Raises decode and drops the SoC ~20 °C. |
| `GPU_MEM_UTIL` | 0.88 | 0.90 passes every startup check and wedges the box hours later. See Troubleshooting. |

If you are serving many concurrent users instead, raise `MAX_NUM_SEQS`, drop
the KV pin back toward 20 GiB for single-stream speed, and consider leaving
`LONG_PREFILL_TOKEN_THRESHOLD` unset — the fairness reserve costs a solo user
about 5% and buys nothing when every step has several requests in it anyway.

## Patches

Applied at build time from `image/patches/`; each asserts its anchor matches
exactly once. `image/verify-base.py` then checks the finished tree, reading files
as text (an import-based check needs a GPU driver that does not exist during
`docker build`).

| file | what it does | source |
|---|---|---|
| `glm53-flash_SM121.py` | Makes the model run on GB10 at all. On capability 12 the nightly offers only `FLASHINFER_MLA_SPARSE_SM120`, which needs the packed `fp8_ds_mla` layout with `pe_dim == 64`; this checkpoint is NoPE, so the engine dies in `concat_and_cache_mla` after a full weight load. Lists the SM90 sparse-MLA path for capability 12 and swaps FA3 for FA2. The Dockerfile retargets one of its paths to the nightly's `models/glm5next/` layout. | [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark), MIT |
| `sm90_fp8_kv_dtype.py` | The SM90 backend planned an fp8 KV cache as uint8. | ours |
| `gb10_plugin_backend.py` | Lets `VLLM_GLM53_CUDA_SPARSE_MLA` pick between the two sm_121 MLA kernels, which otherwise collide silently. | ours |
| `glm53_mtp_bf16.py` | nvidia's `config.json` says the MTP layer is NVFP4, but its weights are BF16, so MTP dies at load. Excludes it from quantisation when the checkpoint holds no scales for it. | ours |
| `glm53_eagle3_aux.py` | DFlash2 reads auxiliary hidden states from target layers 5, 14, 24, 33 and 42; upstream GLM5next does not expose them. | ours |
| `glm53_dflash2_kv_groups.py` | The GLM-5-Next KV grouper gives up on the drafter's sliding-window layers and the model dies unifying page sizes. Keeps the target's groups and adds the drafter's. | ours |
| `glm53_reasoning_always_parsed.py` | Thinking off maps to low reasoning effort (see Troubleshooting), so the model always emits a short `<think>` block. Stock `glm47_moe` stops parsing `<think>` when thinking is off, and the trace would land in `content`. | ours |
| `glm47_failclosed.py` | Tool-call parser plugin (`--tool-call-parser glm47_failclosed`). Checks each call against the tools the request offered; a call with a bad name or argument keys comes back as a retryable call whose sentinel argument names the mistake, instead of being dropped or leaking into history. Containment, not a cure. | ours, after [NNNtrance](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark) #7 and #11 |
| `spin_wait.py` | vLLM's shm queue spins for `busy_loop_s` (1 s) after each message; on GB10 the CPU and GPU share one power budget, so the spin costs ~20 °C and decode. 0.002 keeps the fast path; 0 (always block) measured slower. | [nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) |
| `worker_memory_cap.py` | Caps each worker's share of unified memory (`TORCH_MEM_FRACTION`). vLLM never calls `set_per_process_memory_fraction`, so nothing else bounds a worker. | ours |
| `spark_mem_trace.py` | Names whatever crosses that bound, instead of leaving an OOM anonymous. | ours |
| `link_cuda_headers.sh` | The base ships CUDA libraries without their headers where nvcc looks, which breaks FlashInfer JIT at link time. | ours |

The old recipe's `gb10_topk_fallback.py` is now a flag
(`--sparse-indexer-topk-backend per_row`), and `thinking_budget_guard.py` is
upstream.

## Troubleshooting

**Prefill at half speed, every metric healthy.** If NCCL all-reduce crawls
(~12 Gb/s) while `ib_write_bw` reads a healthy 109 Gb/s and no error counter
moves, the ConnectX-7 has latched a slow fallback state from the DAC cables
being hot-plugged. **Power off and unplug for a minute** — a reboot does not
clear it, and neither does a NIC hotplug reset. The tell is that NCCL Tree
beats Ring; healthy is Ring 110 Gb/s, Tree 44. A single unidirectional stream
cannot see this, which is why the RDMA test passes; a ring collective, sending
and receiving at once, can.

**The box wedges hours after a clean start.** `GPU_MEM_UTIL=0.90` passes every
startup check and is worth +28% KV, then takes the node down with no ssh, no
userspace and ping only. Unified memory means the CUDA allocation *is* host
memory, so none of it is reclaimable and the OOM killer cannot help. 0.88 is the
committed value.

**Long output repeats or skips when thinking is off.** "Count from 1 to 200"
comes back as `35 36 37 37`, or jumps ahead, or starts copying the prompt. This
is the model, not the stack. GLM-5.3-Flash has no non-thinking mode: its official
template always opens `<think>` under `Reasoning Effort: Max`, and an empty
`<think></think>` is out of distribution. Every checkpoint (nvidia, RedHatAI,
the official FP8), both MoE kernels, TP=2 and TP=4, and Hugging Face's own
`glm5_next` implementation fail the same way. This recipe's template maps
`thinking: false` / `enable_thinking: false` to `Reasoning Effort: Low` instead:
a few dozen tokens of reasoning and a clean answer. `reasoning_effort` (`low`,
`high`, default `max`) also works directly, at the top level of the request.

**Intermittent corrupted tokens.** The modelopt NVFP4 build emits them
mid-word, inside rare tokens — invisible in English, reproducible with a Korean
prompt, and identical under both MoE kernels, both attention backends, and with
speculation off. The compressed-tensors builds (RedHatAI NVFP4, INT4 AWQ) are
clean on the same stack. Independently reported by tonyd2wild.

**Boot hangs at `waiting for 4 GPUs, have 1`.** Every node must set `HEAD_HOST`
to the head, not to itself. mentat replicates an agent's *registration* across
the mesh but not its *liveness*: point a node at its own daemon and the head
lists the agent yet marks it `alive=false degraded=true`, while that node's own
daemon sees only its own agent. The GPU gate asks whichever daemon it was told
about, so it never counts more than one. Confirm with `mentatd status` on the
head — every agent should read `alive=true`.

**The second boot deadlocks after NCCL setup.** A persisted FlashInfer autotune
cache keys some MoE entries per rank, so rank 0 loads a tuning the others lack
and they wait on each other forever. The entrypoint keeps the cache ephemeral on
TP>1 and wipes it at every start; `AUTOTUNE_CACHE=persist` brings the old
behaviour back.

**The first request after a cold boot takes minutes.** Triton JIT compiling
DFlash2 shapes mid-serve; `jit_monitor` names them in the log. Send one
throwaway request before timing anything.

**A value in `.env` does nothing.** `.env` only substitutes into the compose
files; a variable reaches the container only if `glm53.yaml` lists it under
`environment`. Compose files layer in order and the last wins.

## Layout

```
compose/    glm53.yaml + dflash2.yaml (DFlash2 speculation), mentatd, mentatd-serve
image/      the whole build context: Dockerfile, entrypoint, chat template,
            status-server.py (status page + engine MCP tools on :8082),
            self-test.py, verify-base.py, and patches/
scripts/    up.sh, down.sh
```

## Credits

[mmastrac/mentat](https://github.com/mmastrac/mentat) ·
[tonyd2wild](https://github.com/tonyd2wild) ·
[MiaAI-Lab](https://github.com/MiaAI-Lab) (sm_121 patches, see
`patches/LICENSE.MiaAI-Lab`) ·
[tonyliu312](https://github.com/tonyliu312) (28 GiB KV pin) ·
[nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) (spin wait) ·
[alexellis](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
