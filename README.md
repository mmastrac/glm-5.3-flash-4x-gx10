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
cp .env.example .env        # on every node; edit ROLE and VLLM_HOST_IP
./scripts/up.sh 192.168.1.93 192.168.1.70 192.168.1.77 192.168.1.36
```

`up.sh` starts mentatd on every node, mentatd-serve on the head, then vLLM
everywhere, and waits for `:8002`. The OpenAI endpoint is `:8002` on the head
directly, or `:6381` through mentatd-serve, which also routes other models.

Build the image with `image/Dockerfile` (it compiles nothing; the vLLM
`glm5_next` per-model image is the base) and apply `scripts/patch-spin-wait.sh`
once before first boot.

## Five things that are not obvious

**Drain the flea power after plugging the DAC cables.** The ConnectX-7 latches a
slow fallback state when cables are hot-plugged. Every link reports 200G and
`ib_write_bw` reads a healthy 109 Gb/s, but NCCL all-reduce crawls at 12 Gb/s and
prefill runs at half speed. A reboot does not clear it and neither does a NIC
hotplug reset — only powering off and pulling the cords. Symptom to look for:
NCCL Tree faster than Ring. Healthy is Ring 110 Gb/s, Tree 44.

**`LONG_PREFILL_TOKEN_THRESHOLD` must be a multiple of 2304.** That is the KDA
block size, and prefix caching snaps chunk ends to it. 2048 produces alternating
2048/256-token chunks. Capping it at 2304 instead of leaving it at the default
(budget − 256) took a 12-token request sent during a 120k prefill from 78–90 s to
~5 s, and made the 200k prefill *faster*, not slower.

**vLLM spins for a full second before it will sleep.** `busy_loop_s` in
`shm_broadcast.py` defaults to 1 s; decode messages arrive every few ms, so the
blocking path is never taken. On GB10 the CPU and GPU share one package, so the
spinning cores take the GPU's thermal budget. 0.002 cut vLLM CPU 185%→109%, the
SoC by ~20 °C, and *raised* decode 66.9→70.6 tok/s. Setting it to 0 is cooler
still and 11% slower — the short spin is worth keeping.
Credit: https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/

**Not every NVFP4 checkpoint is equal.** The modelopt NVFP4 build emits
intermittent corrupted tokens — mid-word, inside rare tokens, invisible in
English and reproducible with a Korean prompt. Same behaviour under both MoE
kernels, both attention backends, with and without speculation. The
compressed-tensors builds (RedHatAI NVFP4, and an INT4 AWQ) are clean on the
identical stack. Independently reported by tonyd2wild.

**`GPU_MEM_UTIL=0.90` passes every startup check and wedges the box hours
later.** Unified memory means the CUDA allocation *is* host memory, so none of it
is reclaimable and the OOM killer cannot help: no ssh, no userspace, ping only.
0.88 is the committed value. Pin the KV pool explicitly with
`--kv-cache-memory` and `--gpu-memory-utilization` becomes decorative.

## Layout

```
compose/    glm53.yaml + four override files, mentatd, mentatd-serve
image/      Dockerfile, entrypoint, chat template
patches/    GB10/sm_121 runtime patches (topk fallback, backend selection)
scripts/    up.sh, down.sh, patch-spin-wait.sh
```

Compose files layer in order and the last one wins; `.env` beats compose beats
the entrypoint, and an `.env` value does nothing unless the compose file passes
it through.

## Credits

[mmastrac/mentat](https://github.com/mmastrac/mentat) ·
[tonyd2wild](https://github.com/tonyd2wild) ·
[MiaAI-Lab](https://github.com/MiaAI-Lab) (sm_121 patches, see
`patches/LICENSE.MiaAI-Lab`) ·
[tonyliu312](https://github.com/tonyliu312) (28 GiB KV pin) ·
[alexellis](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
