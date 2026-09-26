# Notes

The measurements and diagnosis behind the recipe in [README.md](README.md).
Dates are when each was measured; image tags (v4, v6, sm90-v21) are the
builds of this recipe at the time.

## Moving to a vLLM nightly, 2026-09-23

Before the nightly, this ran a per-model base image with 16 vLLM source files
bind-mounted from five compose overrides (now in `attic/`). Every one of those
files is now upstream, a flag, or a small anchored patch baked into the image,
and nothing is mounted over it. Images v4 to v7 used `nightly-0961bbae`; v8
moved to `nightly-ddd6fbca` on 2026-09-26 and has not been measured.

Measured on the same four nodes, thinking off, median of 3, 2026-09-23:

| | old sm90-v21 + DFlash2 | v4 + DFlash2 |
|---|---|---|
| decode structured / code / prose | 105.8 / 71.9 / 50.7 tok/s | 104.4 / 79.1 / 53.4 |
| acceptance | 89.0 / 58.3 / 36.0 % | 88.5 / 65.0 / 38.2 % |
| prefill 31.6k / 126k | 2,229 / 2,176 tok/s | 2,389 / 2,421 |
| KV pool at the 26 GiB pin | 2.02M tokens | 2.63M |
| needle 33k+136k | 6/6 | 6/6 |
| greedy @42k, 8 runs | 7 distinct | 4 distinct |

v6 (thinking off as low effort, below) decodes 109.8 / 88.8 / 52.6 on the same
three, because DFlash2's acceptance rose. MTP (the checkpoint's own head,
`SPEC_METHOD=mtp`) also works, and decoded 57.2 / 54.4 / 45.6 on v4: correct,
but DFlash2 is what carries structured and code.

### What it took, in the order it broke

1. mentat 0.7.0 refuses a bare `ray start --head` while `RAY_ADDRESS` is set.
   The entrypoint passes `--address`.
2. The MTP layer of nvidia/GLM-5.3-Flash-NVFP4 is BF16 but `config.json` says
   NVFP4 (`image/patches/glm53_mtp_bf16.py`).
3. The SM90 sparse-MLA builder planned an fp8 KV cache as uint8
   (`image/patches/sm90_fp8_kv_dtype.py`).
4. The GLM-5-Next KV grouper drops out on the drafter's sliding-window layers
   (`image/patches/glm53_dflash2_kv_groups.py`, tested by
   `dev/patch-tests/_glm53_dflash2_kv_groups_test.py`).
5. A persisted FlashInfer autotune cache deadlocks the next TP=4 boot: rank 0
   saves per-rank MoE entries the others then miss. The entrypoint keeps it
   ephemeral and wipes it on every start.

mentat drops a remote rank's traceback, so a worker that raises during setup
shows only the message on the head. To see the stack, bind-mount a copy of
`ray/_host.py` with `traceback.print_exc()` added to both `except BaseException`
handlers; that is how 2 and 3 were found.

## The model

320B total / 18B active, natively multimodal MoE, `Glm5NextForConditionalGeneration`.
45 layers split **34 KDA linear-attention + 11 DeepSeek-sparse-attention**, 288
routed experts (8 active + 1 shared), an MTP head, 1M context.

Only the 11 sparse layers carry a KV cache and they use `kv_lora_rank: 512`, so
KV is cheap and the memory pressure is essentially all weights. The 26 GiB pin
holds 2,632,595 tokens with DFlash2 at a 524288 window, 5.02 full-length
requests (2026-09-23).

The KDA layers keep one recurrent state per sequence, allocated after weights,
and exactly 32 fit (2026-08-26). That, not throughput, caps `MAX_NUM_SEQS`.

## Why NVFP4 and not the official checkpoint

Measured when the model was released (2026-08-26), at TP=2:

| checkpoint | size | per node at TP=2 | fits? |
|---|---|---|---|
| BF16 | 598.5 GiB | 299 GiB | no |
| FP8 e4m3 (the default) | 305.8 GiB | 153 GiB | **no** — 121.6 GiB per node |
| NVFP4 | **181.3 GiB** | **90.7 GiB** | yes |

Sizes are from the safetensors indexes, not estimates. The routed experts are
NVFP4; both attention flavours, the vision tower, shared experts, routers,
embeddings and `lm_head` stay BF16. The deployed checkpoint is nvidia's
(`nvidia/GLM-5.3-Flash-NVFP4`); the LibertAIDAI and RedHat quants were served
before it.

**There is no usable GGUF.** llama.cpp has no `glm5_next` support, so the GGUF
repos on HF are not loadable by anything.

## What the image is

A pinned vLLM nightly plus anchored patches, each described in the Dockerfile
header, and:

- **mentat** — the daemon binary as `ray`, `mentatd-probe-machine`, and the
  shim wheel. TP=4 across four boxes needs real placement, which a TP=1 tenant
  gets from `ray.register` alone.
- **iproute2 and ibverbs** — the entrypoint finds this node's fabric interface
  with `ip`, and NCCL needs ibverbs for RoCE (without it NCCL falls back to TCP
  over the LAN and every step crawls). Not replaceable by a hardcoded interface
  name: the nodes carry the link on different ports.
- **spark-agent's status server**, plus ripgrep, pciutils and usbutils for its
  tools.

GB10 is sm_121 and the nightly's torch reports up to sm_120. That reads like a
blocker and is not one: sm_120 cubins run on sm_121. `verify-base.py` asserts
it, with every patch marker, the `ray` binary and `ray.register`, so a rebased
base fails the build instead of producing an image that loads ~90 GiB and then
dies.

## Why it loads plain safetensors

`--load-format auto`, and deliberately no sharded-state cache. The fast-boot
cache is dumped *after* the NVFP4 kernel-format transform, so loading one
re-runs that transform, permutes the fused gate/up halves and serves fluent
nonsense with no error. A guard in the loader that skips an already
transformed tensor prevents this. This image has no such guard, so the only
safe load is the one that runs the transform exactly once.

Adopt the sharded-state cache only together with that guard. The cost of not
having it is boot time, which is CPU-bound on weight processing.

## PP is off, and turning it on needs files this repo no longer has

`pp-mtp-override.yaml` was in the old deployment until 2026-09-10. Every file
it mounted was PP-only or inert at `pipeline_parallel_size=1`: two activate
solely when `VLLM_NCCL_NET_*` is set, `mtp.py` is gated on
`get_pp_group().world_size > 1`, `speculative.py` computed the value PP=1
already yields, and its `model.py` and `model_runner.py` were shadowed by the
dflash copies at the same destinations. Removing it left draft acceptance at
1.79 of 7 against a control of 1.75, and decode medians of 41.1 against 38.9
with spreads of 9.4 and 6.0. Its files are in this repo's history
(`git log --all -- '*pp-mtp-override*'`); port them before raising PP above 1.

## The RoCE GID index is per node and per boot

The entrypoint derives `NCCL_IB_GID_INDEX` at start; do not pin it. The RoCE
GID table is indexed by (address, RoCE version) in the order addresses
appeared, so nodes do not agree and a removed address leaves a hole. Read on
2026-08-26 from two nodes:

| node | RoCE device | table | value |
|---|---|---|---|
| A | `rocep1s0f1` | 0,1,2,**hole**,4,5,6 | **6** |
| B | `rocep1s0f0` | 0,1,2,3,4,5 | **5** |

Reboot A with the static profile in place and its table comes up dense, moving
6 to 5, so a pinned 6 names an empty slot and NCCL fails every TP init. To read
a table:

    for i in $(seq 0 9); do p=/sys/class/infiniband/<dev>/ports/1; \
      echo "$i $(cat $p/gid_attrs/types/$i) $(cat $p/gids/$i)"; done

The right entry is the RoCE v2 one for the node's own static fabric address,
unlike the IPv6 link-locals, which regenerate.

## Thinking off is outside what the model was trained on

The official template (zai-org FP8, and the RedHat checkpoint) has no
non-thinking mode at all: every kwarg renders
`<|system|>Reasoning Effort: Max ... <|assistant|><think>`. Our first fix
closed the block empty (`<think></think>`), and long structured output then
degraded. Measured 2026-09-23, "count from 1 to 200", temperature 0:

| mode | nvidia NVFP4, cutlass, MTP | official FP8, triton |
|---|---|---|
| thinking on (default) | clean 8/8 | clean 6/6 |
| thinking off (`<think></think>`) | corrupt 3-8 of 8 | corrupt 8/8 |

The failures are repeats (`35 36 37 37`), skips, or a jump to copying the prompt
text. Every checkpoint (nvidia, RedHat, official FP8), both MoE backends,
weight-only marlin everywhere, fp8 or bf16 KV, TP=2 or 4, graphs or eager, and
the old sm90-v21 image all fail the thinking-off probe the same way. It is not
a serving bug: the per-layer references in `dev/kernel-tests/*_ref.py` match
the decode path. Short answers and the Korean corruption probe are fine with
thinking off; long repetitive output is not.

**Since v6, thinking off means low effort.** The official template takes
`reasoning_effort` (`low`, `high`, anything else is `max`), passed at the top
level of the request or in `chat_template_kwargs`. `image/chat-template.jinja`
maps `thinking: false` / `enable_thinking: false` to `Reasoning Effort: Low`
(an explicit `reasoning_effort` wins) and always opens `<think>`.
`image/patches/glm53_reasoning_always_parsed.py` makes the reasoning parser
track `<think>` on every request, since stock glm47_moe stops parsing it when
either kwarg is false and the short trace would land in `content`. Measured on
v6 (2026-09-23): thinking off gives ~67 reasoning chars and a clean count 6/6.

## v8 against v7, 2026-09-26

Same four boxes and scripts, v7 measured just before the switch. v7 ran with
two overrides that v8 has built in: the SM90 kpool plan (vllm#58704) and fp32
FlashKDA state (vllm#58846).

| | v7 | v8 |
|---|---|---|
| decode counting / code / prose | 101.4 / 75.6 / 34.1 tok/s | 109.9 / 84.4 / 38.4 |
| acceptance | 96.1 / 72.2 / 23.5 % | 97.2 / 74.7 / 24.5 |
| prefill @31k / @117-124k | 2,699 / 2,662 tok/s | 2,730 / 2,679 |
| tool-call probe, 40 runs at 42k | 19 agree with the majority | 22 |
| boot to API | ~10 min | 7.5 min (weights 238 s) |

Prefill is at parity. The first long prompt after a boot measured 2,309 tok/s
and is left out: it pays for JIT compiles. The tool-call probe still diverges,
with 14 and 16 distinct completions, so read 19 against 22 as noise.

Neither image reproduces the 3,365 tok/s at 126k measured on 2026-09-23, with
NCCL on both roots on every box. Check the fabric for a latched slow link
before blaming the image.

## Not yet measured

- `NCCL_MAX_NCHANNELS=8` was chosen on 2026-09-06 while the fabric ran at
  12 Gb/s, before a power drain fixed it. NCCL's own choice is untested since.
- Why production moved from marlin to `flashinfer_cutlass` on 2026-09-21 is not
  recorded. It passes the thinking-on corruption probe; the greedy
  determinism repros in `dev/repro/` were measured on marlin.
