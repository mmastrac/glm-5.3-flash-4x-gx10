# GLM-5.3-Flash on 4x ASUS GX10

GLM-5.3-Flash NVFP4 at TP=4 across four GB10 / ASUS GX10 nodes, one head and
three workers over the ConnectX fabric, with the DFlash2 drafter. The ranks
join through [mentat](https://github.com/mmastrac/mentat), a Ray replacement
that places them and lets its router serve the model.

See [model.yaml](model.yaml) for the weights, footprint and API, and
[KNOBS.md](KNOBS.md) for the environment contract.

| port | what |
|---|---|
| 8002 | OpenAI API, head only (`/v1/chat/completions`, `/v1/models`, `/metrics`) |
| 8082 | status page and MCP (`/mcp`), every rank, from spark-agent |

## Before you start

Every node runs mentatd, and one node runs mentatd-serve if you want the router
in front. Both come from the mentat repo, which has the compose files and
setup: `mentatd.yaml` on every node, `mentatd-serve.yaml` on one. The four
nodes need the ConnectX fabric cabled and addressed; see "The RoCE GID index
is per node and per boot" below.

## Layout

| path | what |
|---|---|
| `image/` | Dockerfile, entrypoint, patches, `verify-base.py`, `self-test.py`, chat template, `build.sh` |
| `compose/glm53.yaml` | the deployment, the same file on every node |
| `.env.example` | per-host and per-node values; copy to `compose/.env` |
| `smoketest/` | `run.sh <base> [served-name]` |
| `.submodules/spark-agent` | the status server, reached through the `vllm` symlink |
| `dev/` | not in the image: the corruption diagnosis and repros, kernel and patch tests, the step tap |
| `attic/` | used by no deployment: the pre-nightly host patches and overrides |

## Build

    git clone --recursive ...        # or: git submodule update --init
    image/build.sh                   # -> spark-glm53:v8
    TAG=spark-glm53:v9 image/build.sh
    BASE=... image/build.sh          # override the pinned nightly

The build context is the repo root, so the `vllm` symlink into the submodule
stays inside it. Build on a box that will run it; there is no cross-build for
aarch64 here. A plain `rsync -a` copy of the repo builds, because it keeps the
hidden `.submodules/` and the symlink; a bare `scp -r *` misses the first.
Build once and pull from the other three nodes; every rank must run the same
image.

## Run

On every node:

    cp .env.example compose/.env     # then fill it in; ROLE and VLLM_HOST_IP differ per node
    docker compose -f compose/glm53.yaml up -d

Compose reads `.env` from `compose/`, beside the compose file. The project name
is pinned to `glm53`; a stack started under another project name (a directory
named after the deployment, say) must be taken down first, or the two collide
on the container name.

GLM's weights alone take ~90.7 GiB of each node's 121.6 GiB, so nothing else
large fits beside it.

Either side may start first from cold: `ray start` under mentat retries until
the daemon answers, and the head waits for all four GPUs before it loads.

**Replacing a running stack is different.** Recreating all four ranks in
parallel lets a starting rank join a group that is still tearing down, and the
whole group then hangs just past NCCL setup: every rank `running`, RestartCount
0, CPU ~1%, no weights loading, and nothing in any log after the
`custom_all_reduce` warning. Seen on 2026-09-10 after several rapid recreate
cycles. Take every rank down, confirm all four containers are gone, then start
the head and the workers a few seconds later.

    # on each node
    docker compose -f compose/glm53.yaml down --timeout 60
    # confirm: docker ps -a | grep glm53  ->  nothing, on all four
    # then head first, workers after

Then check it: `smoketest/run.sh http://<head>:8002`, or through mentat-serve.

## Roll back

The compose file here leaves every tuned knob to the image's entrypoint, and
images up to v7 carry the old defaults (TP=2, MTP, a 262144 window). So an
older image goes back with its own compose file and `.env`, not with this one:
keep the previous deployment directory (`glm53.yaml` + `dflash2.yaml` and a
`.env` that sets every tuned knob) until v8 has served, then `down` here and
`up -d` there. That tree is commit `9c9c3e1`.

The pre-nightly deployment, a per-model base image plus 16 vLLM source files
bind-mounted from five compose overrides, is in `attic/` for the record.

## On a vLLM nightly since 2026-09-23

Stock `vllm/vllm-openai:nightly-ddd6fbca` (v8, 2026-09-26; v4 to v7 used
`nightly-0961bbae`), plus FlashKDA 17a037d and the patches listed in the
Dockerfile header, with nothing mounted over the image. Every file the old
deployment bind-mounted is now upstream, a flag, or a small anchored patch
baked into the image.

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
nonsense with no error. `ds4-flash/vllm-spark.patch` carries the
`IDEMPOTENT-GUARD` that prevents this; **this image does not carry it**, so the
only safe load is the one that runs the transform exactly once.

Adopt the sharded-state cache only together with that patch. The cost of not
having it is boot time, which is CPU-bound on weight processing.

## PP is off, and turning it on needs files this repo no longer has

`pp-mtp-override.yaml` was in the old deployment until 2026-09-10. Every file
it mounted was PP-only or inert at `pipeline_parallel_size=1`: two activate
solely when `VLLM_NCCL_NET_*` is set, `mtp.py` is gated on
`get_pp_group().world_size > 1`, `speculative.py` computed the value PP=1
already yields, and its `model.py` and `model_runner.py` were shadowed by the
dflash copies at the same destinations. Removing it left draft acceptance at
1.79 of 7 against a control of 1.75, and decode medians of 41.1 against 38.9
with spreads of 9.4 and 6.0. Its files are in git history; port them before
raising PP above 1.

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

## Thinking off needs a patch in the checkpoint directory

The template decides this, not the flag. `image/chat-template.jinja` carries
the fix, and the Dockerfile bakes it to
`/usr/local/share/glm53-chat-template.jinja`. **That copy is not always the one
used.** The entrypoint prefers the checkpoint's own template whenever it finds
`<|begin_of_image|>` in it, deliberately, because the baked one would otherwise
cost image support on a checkpoint that has its own. So a checkpoint shipping
an image-capable template silently shadows the baked fix, and thinking off
breaks again with nothing in any log.

The nvidia checkpoint is such a checkpoint, so its own `chat_template.jinja` is
patched in place on each node, with the original kept beside it
(`<MODEL_HOST_DIR>/chat_template.jinja.pre-effortfix`). **This reverts on any
checkpoint swap or model re-sync.** Re-apply the edit, validate that the
template renders before deploying (`jinja2.Environment(extensions=
["jinja2.ext.loopcontrols"])` — the template uses `{% break %}`; run it inside
the container), then run the smoketest, whose thinking cases check both modes.
Checking only that `reasoning` is empty is not enough — it is empty in the
broken state too, with the monologue sitting in `content`.

### Thinking off is outside what the model was trained on

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

## Weights live on local disk, per node

Every rank reads the whole checkpoint under `--load-format auto`, so each node
needs its own copy at `MODEL_HOST_DIR`, and the DFlash2 drafter at
`DFLASH_HOST_DIR`. Local rather than NFS on purpose: the NFS models mount races
the network at boot and fails (seen on one node on 2026-08-26, and on the two
boots before it), and a model that only starts when the NAS is awake will
eventually fail to start. `hf download` is resumable.

## Not yet measured

- `NCCL_MAX_NCHANNELS=8` was chosen on 2026-09-06 while the fabric ran at
  12 Gb/s, before a power drain fixed it. NCCL's own choice is untested since.
- Why production moved from marlin to `flashinfer_cutlass` on 2026-09-21 is not
  recorded. It passes the thinking-on corruption probe; the greedy
  determinism repros in `dev/repro/` were measured on marlin.
- The mentat endpoints in port form (`8002/v1`, `8082/mcp`) have not yet run on
  this model: v7 and earlier announced URLs built from `VLLM_HOST_IP`.
