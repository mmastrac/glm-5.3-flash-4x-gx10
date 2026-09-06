# GLM-5.3-Flash on 4× ASUS GX10 (GB10), TP=4

vLLM TP=4 across four GB10 boxes over RoCE, DFlash2 speculative decoding,
524k context. Ray is replaced by [mentat](https://github.com/mmastrac/mentat).

Measured, first-touch, temperature 0, single stream:

| | |
|---|---|
| prefill @114k | 2,018 tok/s |
| prefill @200k | 2,004 tok/s |
| decode, counting | 104.5 tok/s |
| decode, code | 70.6 tok/s |
| decode, prose | 44.9 tok/s |
| KV pool | 2,835,245 tokens (28 GiB, fp8_e4m3) |
| 200k needle recall | pass |
| a 12-token request sent during a 120k prefill | ~5 s |

Hardware: 4× ASUS GX10 (GB10, sm_121a, 128 GB unified), ConnectX-7 200G RoCE
through a MikroTik CRS812, 10 GbE for management.

## Run it

```
cp .env.example .env        # on every node; set ROLE, and SPARK_HOME if not /home/admin
./scripts/up.sh 192.168.1.93 192.168.1.70 192.168.1.77 192.168.1.36
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

Build with `docker build -t glm53-spark:sm90-v21 image/` — it compiles nothing,
the vLLM `glm5_next` per-model image is the base — and run
`scripts/patch-spin-wait.sh` once before first boot.

Each container also serves a status page and a set of engine MCP tools on
`:8082` (`status-server.py`), which is what `MENTAT_MCP_API` points at. Use it
to read engine state, memory accounting and the KV pool without attaching to
the container.

## Tuning

This is tuned for one or two interactive users, not for a serving fleet. The
bias throughout is that **a long prefill must never block a short request**, and
that a single stream should be fast, rather than maximising aggregate
throughput at concurrency.

| knob | value | why |
|---|---|---|
| `LONG_PREFILL_TOKEN_THRESHOLD` | 2304 | Caps one prefill's share of each scheduler step. Left at the default (budget − 256) a 120k prefill takes the whole step and a 12-token request waits 78–90 s; at 2304 it waits ~5 s. Must be a multiple of 2304, the KDA block size, because prefix caching snaps chunk ends to it — 2048 yields alternating 2048/256-token chunks. Costs nothing: the 200k prefill got *faster*. |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | Measured indistinguishable from 8192 at 200k once chunks are capped. |
| `--kv-cache-memory` | 28 GiB | 2.84M tokens, 5.41× concurrency at full context. 20 GiB is ~5% faster on a single stream; this trades that for headroom. Pin it explicitly — `--gpu-memory-utilization` then becomes decorative and vLLM says so at startup. |
| `MAX_NUM_SEQS` | 32 | With DFlash2 k=7 a decode step costs 1+k=8 token slots per sequence, so 32 sequences exactly consume a 256-token reserve. |
| DFlash2 `k=7` | | Roughly doubles decode against MTP k=4 on this model. Costs ~41% of the KV pool: 3.44M tokens with speculation off, 2.02M with it at the same pin. |
| `busy_loop_s` | 0.002 | See Patches. Raises decode and drops the SoC ~20 °C. |
| `GPU_MEM_UTIL` | 0.88 | 0.90 passes every startup check and wedges the box hours later. See Troubleshooting. |

If you are serving many concurrent users instead, raise `MAX_NUM_SEQS`, drop
the KV pin back toward 20 GiB for single-stream speed, and consider leaving
`LONG_PREFILL_TOKEN_THRESHOLD` unset — the fairness reserve costs a solo user
about 5% and buys nothing when every step has several requests in it anyway.

## Patches

Runtime patches applied to the stock image. All of them gate paths that already
exist onto GB10 rather than writing kernels.

Applied at build time from `image/patches/`.

| file | what it does | source |
|---|---|---|
| `glm53-flash_SM121.py` | Makes the model run on GB10 at all. The stock image offers only `FLASHINFER_MLA_SPARSE_SM120` on capability 12, and that backend requires the packed `fp8_ds_mla` layout with `pe_dim == 64`; this checkpoint is NoPE, so the kernel refuses and the engine dies in `concat_and_cache_mla` after a full weight load. Lists the SM90 sparse-MLA path for capability 12 and relaxes its `capability.major == 9` check, swapping FA3 for FA2 — the backend is not inherently Hopper-only, only its FA3 kernel is. | [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark), MIT, verbatim |
| `gb10_topk_fallback.py` | Routes GB10 off the indexer's `persistent_topk` onto the fallback already in the tree. That kernel is a cooperative launch whose whole grid must be resident, so it aborts above a pool size GB10 cannot reach. | ours |
| `gb10_plugin_backend.py` | Lets `VLLM_GLM53_CUDA_SPARSE_MLA` pick between the two sm_121 MLA kernels. Two independent solutions to the same problem collide silently otherwise: MiaAI-Lab's patcher lists SM90 *ahead* of SM120, which shadows the LibertAI sparse-MLA plugin. | ours |
| `worker_memory_cap.py` | Caps each worker's share of unified memory. vLLM never calls `set_per_process_memory_fraction`, so nothing bounds a worker — `gpu_memory_utilization` only sizes the KV pool, and the indexer's scores allocation grows with the session. | ours |
| `spark_mem_trace.py` | Names whatever crosses that bound, instead of leaving an OOM anonymous. | ours |
| `link_cuda_headers.sh` | Symlinks the CUDA headers where nvcc looks. The base ships a trimmed `/usr/local/cuda/include`: the libraries are present, the matching headers are not. Without it a JIT compile dies at link time with no obvious cause. | ours |
| `verify.py` | Asserts at **build** time that the SM121 patches landed, reading files as text — an import-based check cannot work, since `vllm.platforms.cuda` needs a driver that does not exist during `docker build`. A base-image change fails the build rather than silently producing a half-patched tree. | ours |
| `scripts/patch-spin-wait.sh` | vLLM's shm queue spins for `busy_loop_s` after the last message before it will block on zmq. The default is 1 s and decode messages arrive every few ms, so the blocking path is never taken and the cores spin at full power. On GB10 the CPU and GPU share one package, so that heat comes out of the GPU's budget. 1 → 0.002 cut vLLM CPU 185%→109%, the SoC ~20 °C, and *raised* decode 66.9→70.6 tok/s. 0 (always block) is cooler and 11% slower — the short spin is worth keeping. | [nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) |

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

**The first request after a cold boot takes minutes.** Triton JIT compiling
DFlash2 shapes mid-serve; `jit_monitor` names them in the log. Send one
throwaway request before timing anything.

**`--kv-cache-memory` cannot be overridden from `.env`.** It is a literal inside
`dflash2-full-override.yaml`. Compose files layer in order and the last wins;
`.env` beats compose beats the entrypoint, and an `.env` value does nothing
unless the compose file passes it through.

## Layout

```
compose/    glm53.yaml + three override files, mentatd, mentatd-serve
image/      the whole build context: Dockerfile, entrypoint, chat template,
            status-server.py (status page + engine MCP tools on :8082),
            self-test.py, and patches/
scripts/    up.sh, down.sh, patch-spin-wait.sh
```

## Credits

[mmastrac/mentat](https://github.com/mmastrac/mentat) ·
[tonyd2wild](https://github.com/tonyd2wild) ·
[MiaAI-Lab](https://github.com/MiaAI-Lab) (sm_121 patches, see
`patches/LICENSE.MiaAI-Lab`) ·
[tonyliu312](https://github.com/tonyliu312) (28 GiB KV pin) ·
[nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) (spin wait) ·
[alexellis](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
