#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Serve GLM-5.3-Flash (NVFP4) across the ConnectX-linked GX10 boxes. TP and PP
# come from the environment, so the same image serves the TP=2/PP=2 pair-of-
# pairs and a flat TP=4 across one fabric.
#
# Deliberately simpler than ds4-flash/entrypoint.sh in one respect: it loads
# PLAIN SAFETENSORS and builds no sharded-state cache. That costs boot time and
# buys correctness margin -- the sharded-state path needs the modelopt
# IDEMPOTENT-GUARD patch, because a cache is dumped AFTER the NVFP4
# kernel-format transform and re-running that transform on load permutes the
# fused gate/up halves and serves fluent nonsense with no error. This image is
# built on upstream's per-model base and carries none of our patches, so the
# safe load is the one that runs the transform exactly once.
#
# Adopt the sharded-state cache only together with that patch. See
# ds4-flash/vllm-spark.patch.
# ---------------------------------------------------------------------------

TP="${TP:-4}"
MTP="${MTP:-1}"
# SPEC_METHOD picks the drafter: dflash (the separate DFlash2 draft model at
# DFLASH_MODEL, which must be mounted on every node; k must be 7, its block
# size minus one, and the model refuses anything else), mtp (the checkpoint's
# own head, num_nextn_predict_layers: 1), or none. MTP=0 is the older off
# switch and still turns speculation off whichever method is named. Resolved
# here, above the cache tag, because the tag hashes it.
#
# dflash is what serves: 109.8 / 88.8 / 52.6 tok/s on structured / code / prose
# with v6, where MTP gave 57.2 / 54.4 / 45.6 with v4 (both 2026-09-23). Judge a drafter
# on acceptance LENGTH, not acceptance rate. Set here rather than through
# EXTRA_ARGS, which is one string that a compose override restating it
# silently replaces.
SPEC_METHOD="${SPEC_METHOD:-dflash}"
[[ "$MTP" == "0" ]] && SPEC_METHOD=none
ROLE="${ROLE:-head}"
MODEL="${MODEL_DIR:-/models/glm-5.3-flash-nvfp4}"
SERVED="${SERVED_NAME:-glm53}"

# --- cluster networking: everything rides the ConnectX link -----------------
# Copied wholesale from ds4-flash because the constraint is the hardware, not
# the model. No literal address appears here or in the compose file: each node
# finds its own cluster IP by looking for the interface carrying CLUSTER_SUBNET.
# The boxes are NOT symmetric (one carries its link on enp1s0f1np1, the rest
# on enp1s0f0np0), so any hardcoded interface name is wrong somewhere whichever
# you pick.
#
# An uncabled port powers down completely -- no PCI device, no
# /sys/class/infiniband entry. That is not a missing driver and no amount of
# modprobe fixes it, so a node reports only the ports it actually has.
CLUSTER_SUBNET="${CLUSTER_SUBNET:?set CLUSTER_SUBNET to the fabric address prefix, with its trailing dot}"
_cxip=$(ip -o -4 addr show 2>/dev/null | awk -v p="$CLUSTER_SUBNET" \
        '$4 ~ "^"p {split($4,a,"/"); print a[1]; exit}')

# The fabric is also absent while the switch reboots, and exiting then turns a
# two-minute outage into a restart loop racing it: 60 restarts across one
# firmware upgrade, none of which could have succeeded. Wait instead, so the
# outage is a pause. FABRIC_WAIT_S bounds it; 0 waits forever.
if [[ -z "${VLLM_HOST_IP:-}" ]]; then
  _waited=0
  while [[ -z "$_cxip" ]]; do
    if (( ${FABRIC_WAIT_S:-0} > 0 && _waited >= ${FABRIC_WAIT_S:-0} )); then
      echo "FATAL: no interface carries ${CLUSTER_SUBNET}0/24 after ${_waited}s" >&2
      ip -br addr show >&2
      exit 1
    fi
    (( _waited % 60 )) || echo "waiting for an interface on ${CLUSTER_SUBNET}0/24 (${_waited}s)"
    sleep 10
    _waited=$(( _waited + 10 ))
    _cxip=$(ip -o -4 addr show 2>/dev/null | awk -v p="$CLUSTER_SUBNET" \
            '$4 ~ "^"p {split($4,a,"/"); print a[1]; exit}')
  done
fi
export VLLM_HOST_IP="${VLLM_HOST_IP:-$_cxip}"

# Gloo needs an EXACT interface name -- NCCL_SOCKET_IFNAME takes a prefix, this
# does not. Left unset, Gloo binds 127.0.0.1 and the cluster fails silently at
# rendezvous, so derive it from whichever interface actually owns VLLM_HOST_IP.
# Bounded, unlike the fabric wait above: the address is already up, so the
# interface and its RDMA device are local state that either settles in seconds
# or is broken. Waiting past that hides the fault instead of reporting it.
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  _waited=0
  while :; do
    GLOO_SOCKET_IFNAME=$(ip -o -4 addr show 2>/dev/null \
        | awk -v ip="$VLLM_HOST_IP" '$4 ~ "^"ip"/" {print $2; exit}')
    [[ -n "$GLOO_SOCKET_IFNAME" ]] && break
    if (( _waited >= ${ROCE_SETTLE_S:-60} )); then
      echo "FATAL: no interface holds VLLM_HOST_IP=$VLLM_HOST_IP after ${_waited}s" >&2
      ip -br addr show >&2; exit 1
    fi
    sleep 5; _waited=$(( _waited + 5 ))
  done
fi
export GLOO_SOCKET_IFNAME

# --- fabric ports: which RoCE devices carry the cluster, at which GID -------
# FABRIC_SUBNETS names one address prefix per cabled port. A second entry puts
# NCCL on both PCIe roots of the ConnectX-7, which ib_write_bw measured at 196
# Gb/s against 112 for one root alone. It is opt-in because the second root's
# registrations are GPU-resident and land AFTER vLLM profiles, so they eat the
# headroom a long prefill needs rather than the KV cache: allocations sat at
# 111.41 GiB against the 104.6 GiB GPU_MEM_UTIL=0.86 budgets. At TP=2 that
# bought nothing -- the per-token allreduce is ~720 KB, a fraction of a
# millisecond against a 45 ms token -- so weigh it only at TP>2.
#
# Keyed on the subnet rather than on VLLM_HOST_IP because the two are no longer
# the same address. mentat identifies a node by its LAN address and the agent
# and daemon must agree on one string, so VLLM_HOST_IP is the LAN one, and no
# RoCE GID will ever match it.
FABRIC_SUBNETS="${FABRIC_SUBNETS:-$CLUSTER_SUBNET}"

# Echoes "<rdma-device> <gid-index>" for the port holding an address in $1.
# Returns 1 when this node has not cabled that port, or its GID has yet to
# appear.
fabric_port() {
  local prefix="$1" found addr ifname dev="" hex i t g d n
  found=$(ip -o -4 addr show 2>/dev/null \
      | awk -v p="$prefix" '$4 ~ "^"p {split($4,a,"/"); print $2, a[1]; exit}')
  [[ -n "$found" ]] || return 1
  ifname="${found%% *}"; addr="${found##* }"
  for d in /sys/class/infiniband/*; do
    for n in "$d"/ports/1/gid_attrs/ndevs/*; do
      [[ -f "$n" ]] || continue
      [[ "$(cat "$n" 2>/dev/null)" == "$ifname" ]] || continue
      dev="$(basename "$d")"; break 2
    done
  done
  [[ -n "$dev" ]] || return 1
  # Slots 0/1 hold the driver's MAC-derived GIDs whatever the IP configuration,
  # so match this port's own static address in its IPv4-mapped form, v2 only.
  hex=$(printf '%s' "$addr" | awk -F. '{printf "%02x%02x:%02x%02x", $1,$2,$3,$4}')
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/types/$i" 2>/dev/null) || continue
    g=$(cat "/sys/class/infiniband/$dev/ports/1/gids/$i" 2>/dev/null) || continue
    [[ "$t" == "RoCE v2" && "$g" == *"ffff:$hex" ]] || continue
    printf '%s %s\n' "$dev" "$i"; return 0
  done
  return 1
}

# There is no correct constant for the GID index. The table is keyed by
# (address, RoCE version), slots are allocated first-free and freed in place, so
# the index for one address differs per node AND per boot. Observed, not
# theorised: on 2026-08-26 one node held the right entry at 6 and its peer at
# 5, because a `nmcli con delete` / `add` had left the first a hole at slot 3. Reboot
# it with the static profile already in place and the table comes up dense,
# moving that 6 to 5, at which point a pinned 6 names an empty slot and NCCL
# fails every TP init with "unhandled system error".
if [[ -z "${NCCL_IB_HCA:-}" || -z "${NCCL_IB_GID_INDEX:-}" ]]; then
  _waited=0
  while :; do
    _hcas=""; _gid=""; _mismatch=""
    for _p in $FABRIC_SUBNETS; do
      _r=$(fabric_port "$_p") || continue
      _d="${_r%% *}"; _i="${_r##* }"
      [[ -n "$_gid" && "$_i" != "$_gid" ]] && _mismatch="$_d at $_i, expected $_gid"
      _gid="${_gid:-$_i}"
      _hcas="${_hcas:+$_hcas,}$_d"
    done
    [[ -n "$_hcas" ]] && break
    if (( _waited >= ${ROCE_SETTLE_S:-60} )); then
      echo "FATAL: no RoCE v2 GID for any of: $FABRIC_SUBNETS" >&2
      ip -br addr show >&2
      ls /sys/class/infiniband/ >&2 || echo "(no /sys/class/infiniband at all)" >&2
      exit 1
    fi
    sleep 5; _waited=$(( _waited + 5 ))
  done
  # NCCL_IB_GID_INDEX applies to every device in NCCL_IB_HCA, so a device whose
  # index differs gets asked for a GID it does not have and QP setup dies with
  # "local GID ::". One IPv4 per fabric interface keeps them aligned; refuse the
  # list rather than hand NCCL one that cannot work.
  if [[ -n "$_mismatch" ]]; then
    echo "FATAL: fabric ports disagree on GID index ($_mismatch)." >&2
    echo "Each fabric interface must carry exactly one IPv4 address." >&2
    exit 1
  fi
  NCCL_IB_HCA="${NCCL_IB_HCA:-$_hcas}"
  NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-$_gid}"
  echo "fabric: NCCL_IB_HCA=$NCCL_IB_HCA gid=$NCCL_IB_GID_INDEX (subnets: $FABRIC_SUBNETS)"
else
  echo "fabric: NCCL_IB_HCA=$NCCL_IB_HCA gid=$NCCL_IB_GID_INDEX (both pinned; derivation skipped)"
fi
export NCCL_IB_HCA NCCL_IB_GID_INDEX
# The EXACT interface holding VLLM_HOST_IP, not a prefix. This is only the
# out-of-band bootstrap path -- the IB devices above carry the data -- and a
# prefix matches every interface that happens to share it. Once the fabric
# grew a second cable, "enp1s0f" matched both the head link and an unrelated
# one, and NCCL bootstrapped toward the wrong wire: "Connection closed by
# remote peer spark-head".
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"

# Channel count. NCCL picks 64 here, while every published GB10 recipe pins
# 4-12. 8 was chosen on 2026-09-06 while the fabric was stuck at 12 Gb/s, before
# a power drain fixed it; NCCL's own choice has not been re-tested since.
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
# INFO prints the devices NCCL actually selected, which is the only way to
# confirm both roots are in use.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# --- worker memory ---------------------------------------------------------
# Torch's caching allocator never hands a freed block back to the OS, and on
# unified memory that block is host memory. The sparse indexer scores
# chunk x context/kpool, so a session whose context keeps growing asks for a
# slightly bigger block each step and strands the last one.
# expandable_segments:True grows one segment instead of laddering.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Hard ceiling on one worker's device memory, as a fraction of the 121.6 GiB,
# read by spark_mem_trace.py. Keep it above GPU_MEM_UTIL by enough for a full
# prefill's indexer scratch, and below 1.0 by whatever the host needs to keep
# forking: past this a request raises OutOfMemoryError instead of the node
# wedging. 0.92 is ~111.9 GiB against GPU_MEM_UTIL's ~107.
export TORCH_MEM_FRACTION="${TORCH_MEM_FRACTION:-0.92}"

RAY_ADDRESS="${RAY_ADDRESS:-${HEAD_HOST:?set HEAD_HOST to the head node address}:6379}"
export RAY_ADDRESS

# mentat (the Ray replacement in this image) rendezvouses subclusters by
# group; every rank of this deployment must carry the same value. Running the
# same model twice means two compose stacks with DISTINCT MENTAT_GROUP values.
export MENTAT_GROUP="${MENTAT_GROUP:-${SERVICE_NAME:-glm53}}"

# Pin the executor the mentat shim was audited against. It is vLLM's default
# today (verified in this image AND the DS4 tree), but the default is
# undocumented, and if an upstream bump flips it the legacy executor's
# compiled-DAG surface comes alive -- which mentat does not implement and
# fails loudly on. Pinning turns that failure into a grep-able one-liner.
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1

# --- scheduler: per-step token budget, and one prefill's share of it --------
# One setting, not two. The scheduler gives a long prefill
# min(threshold, remaining budget), so a threshold at or above the budget lets
# one big prompt take the entire step -- no decodes of running requests, no
# admissions from the waiting queue. Measured on DS4 2026-08-25: a 90-token
# "ping" sent 20s into a 198K-token prefill took 125.5s to answer.
#
# 16384 with each prefill capped at 2304, so one step carries several prefill
# chunks and every running decode. Measured on TP=4 (2026-09-06): 8192 and
# 16384 prefill a 200k prompt in the same time (234.1 s against 237.7 s); the
# old 9.5x win for 8192 was a PP=2 pipeline bubble, gone at PP=1. Measure this
# at 200k, never at 8k: an 8k prompt fits inside both budgets.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# --- max_num_seqs is capped by the KDA state, not by throughput -------------
# 34 of this model's 45 layers are KDA linear attention, and a linear-attention
# layer keeps a fixed-size RECURRENT STATE per sequence rather than a per-token
# KV entry. vLLM accounts for those as "Mamba cache blocks", one per decode
# sequence, and they are allocated out of whatever is left after weights.
#
# Measured here 2026-08-26: exactly 32 blocks fit at GPU_MEM_UTIL=0.86 with the
# 181 GiB NVFP4 checkpoint. vLLM's default max_num_seqs of 256 therefore aborts
# at CUDA graph capture, AFTER a full 10-minute weight load:
#
#   ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (32).
#   Each decode sequence requires one Mamba cache block, so CUDA graph capture
#   cannot proceed.
#
# This is a hard structural cap, not a tuning preference -- raising
# GPU_MEM_UTIL is the only way to buy more blocks. The default takes all 32.
#
# Keep CUDAGRAPH_CAPTURE_SIZES's largest entry >= this value: with MTP off a
# decode step is one token per sequence, so a full batch is exactly this many
# tokens, and a step larger than the biggest captured size runs uncaptured.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# One prefill's share of the step: 2304, one KDA block. A multiple of the block
# matters -- 2048 produced alternating 2048/256 chunks. With it (2026-09-06),
# prefill ran 2,004 tok/s at 200k and a 12-token ping sent under a 120k prefill
# answered in 4.83 s. DECODE_RESERVE_TOKENS, when set, replaces it with
# budget minus reserve, which is how DS4 expresses the same split.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-2304}"
if [[ -n "${DECODE_RESERVE_TOKENS:-}" ]]; then
  LONG_PREFILL_TOKEN_THRESHOLD=$(( (MAX_NUM_BATCHED_TOKENS - DECODE_RESERVE_TOKENS) / 4 * 4 ))
  if (( LONG_PREFILL_TOKEN_THRESHOLD < 4 )); then
    echo "FATAL: DECODE_RESERVE_TOKENS=${DECODE_RESERVE_TOKENS} leaves only" >&2
    echo "${LONG_PREFILL_TOKEN_THRESHOLD} tokens for prefill out of a ${MAX_NUM_BATCHED_TOKENS} budget." >&2
    exit 1
  fi
fi
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-8 16 32 64 96 128 192 256}"
echo "scheduler: budget=${MAX_NUM_BATCHED_TOKENS} prefill<=${LONG_PREFILL_TOKEN_THRESHOLD}" \
     "reserve=$(( MAX_NUM_BATCHED_TOKENS - LONG_PREFILL_TOKEN_THRESHOLD ))/step"

export MAX_JOBS="${MAX_JOBS:-4}"

# --- JIT/autotune cache tiers ----------------------------------------------
# Two tiers, because the caches have different invalidation domains: the shared
# tier is keyed on version/arch/kv-dtype only and is the expensive one to
# rebuild, so it must stay stable across serving-option experiments; the keyed
# tier holds torch.compile, which vLLM hashes against the whole serving config.
CACHE_ROOT="${CACHE_ROOT:-/root/.cache}"
_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
_arch="${_arch:-unknown}"
# --- KV cache dtype: auto (bf16), NOT fp8 -----------------------------------
# fp8 here selects vLLM's `fp8_ds_mla` packed KV format, which is DeepSeek's and
# hardcodes pe_dim == 64:
#
#   RuntimeError: concat_and_cache_mla, cache_kernels.cu:866,
#   pe_dim must be 64 for fp8_ds_mla
#
# GLM-5.3-Flash is NoPE sparse MLA -- `qk_rope_head_dim: 0`, no rotary on the
# sparse path at all -- so pe_dim is 0 and that kernel refuses. This is what the
# vendor recipe means by "FlashInfer 0.6.17+ is required for NoPE sparse MLA":
# the NoPE variant is a different code path, not just a newer one.
#
# Copying ds4-flash's `--kv-cache-dtype fp8` across is therefore wrong: DS4 is
# DeepSeek geometry (pe_dim 64) and this model is not. It costs a full weight
# load (~10 min) to find out, because the failure is at KV cache init.
#
# The default below is fp8_e4m3, and it is what every measurement on this model
# was taken with: the Korean corruption probe comes back clean and 200k recall
# passes on it. Only 11 of 45 layers carry a KV cache and they use
# kv_lora_rank 512, so bf16 would also fit -- but nothing here has been measured
# on bf16, so do not switch on the assumption that it is the safer default.
_kv="${KV_CACHE_DTYPE:-fp8_e4m3}"
export SHARED_TAG="${SHARED_TAG:-glm53-${_arch}-${_kv}}"
# Resolve ONCE, above every use. Two MAX_MODEL_LEN defaults in one
# file is the "two places to change and one silently winning" trap the DS4
# compose file warns about -- and it was live here: the cache tag said 131072
# while the server was told 262144.
#
# 524288 of the model's 1M. KV is cheap here -- only 11 of 45 layers carry one,
# at kv_lora_rank 512 -- so the 26 GiB pin below holds 2,632,595 tokens with
# DFlash2, 5.02 requests at full length (head boot log, 2026-09-23).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
_optstr="tp=${TP} mtp=${MTP} spec=${SPEC_METHOD} len=${MAX_MODEL_LEN} cg=${CUDAGRAPH_CAPTURE_SIZES}"
_opthash=$(printf '%s' "$_optstr" | sha256sum | cut -c1-8)
export CACHE_TAG="${CACHE_TAG:-${SHARED_TAG}-${_opthash}}"

# FlashInfer autotune: EPHEMERAL on TP>1, and wiped on every start. A persisted
# cache deadlocks the NEXT boot, silently and forever. On the nightly, rank 0
# reads its cache and broadcasts it, but only rank 0 ever saves, and the fused-
# MoE entries it saves key per rank: ranks 1-3 miss what rank 0 hits, go off to
# profile (a CPU-group all_reduce) while rank 0 has moved on to the model's NCCL
# all_reduce, and each waits for the other. Measured 2026-09-23 with py-spy on
# spark-glm53:v3. The first boot always works (nobody has a cache); every boot
# after a good one hangs with the head at 96% GPU and the workers at 0%.
# Wiping on start, not just keeping it out of the bind mount, matters: a head
# engine restart keeps the container's /tmp. Costs ~2 min of autotune a boot.
# AUTOTUNE_CACHE=persist brings the old behaviour back (TP=1 is safe with it).
if [[ "${AUTOTUNE_CACHE:-}" == "persist" || "${TP:-1}" -le 1 ]]; then
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${CACHE_ROOT}/${SHARED_TAG}/flashinfer_autotune"
else
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/flashinfer_autotune
  rm -rf "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
fi
echo "autotune cache: $VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/${CACHE_TAG}/vllm"
mkdir -p "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" "$VLLM_CACHE_ROOT"

# FlashInfer's own JIT cache is deliberately NOT redirected. FLASHINFER_CACHE_DIR
# looks like the knob but is a module-level constant computed from
# FLASHINFER_WORKSPACE_BASE (default $HOME), so setting it does nothing at all.
# $HOME/.cache is the bind mount, so the default already persists.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/${SHARED_TAG}/triton}"
mkdir -p "$TRITON_CACHE_DIR"

# TileLang JIT builds the mHC (Manifold-Constrained Hyper-Connection) kernels
# this architecture adds, and the DSA/MTP kernels. Its default is
# ~/.tilelang/cache -- a SIBLING of the bind-mounted ~/.cache, so left alone it
# lands inside the container and is destroyed on every `compose down`, and every
# boot recompiles. Same invalidation domain as FlashInfer, hence the shared tier.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${CACHE_ROOT}/${SHARED_TAG}/tilelang}"
mkdir -p "$TILELANG_CACHE_DIR"
echo "cache shared: ${CACHE_ROOT}/${SHARED_TAG} (arch=${_arch} kv=${_kv})"
echo "cache keyed : ${CACHE_ROOT}/${CACHE_TAG} ($_optstr)"
echo "cache tilelang: $TILELANG_CACHE_DIR"

# --- status page ------------------------------------------------------------
# Own port, started immediately. Loading 181 GiB of weights takes long enough
# that "connection refused" is a poor answer to "what is it doing?", and this
# model's metrics read zero throughout a long prefill anyway.
export STAGE_FILE="${STAGE_FILE:-/tmp/glm53-stage}"
: > "$STAGE_FILE"
stage() { echo "$1" >> "$STAGE_FILE"; echo "== stage: $1 =="; }
STATUS_PORT="${STATUS_PORT:-8082}" PORT="${API_PORT:-8002}" \
  python3 /usr/local/bin/status-server.py &
stage starting

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "FATAL: no config.json under MODEL_DIR=$MODEL." >&2
  echo "Both ranks read the full checkpoint, so it must be present on THIS node." >&2
  exit 1
fi

if [[ -d "${MCP_LOG_DIR:-/logs}" && -w "${MCP_LOG_DIR:-/logs}" ]]; then
  _log="${MCP_LOG_DIR:-/logs}/vllm-${ROLE}.log"
  [[ -f "$_log" ]] && mv -f "$_log" "${_log%.log}.prev.log" 2>/dev/null || true
  echo "logging to $_log"
  exec > >(tee "$_log") 2>&1
fi

# NEUTRALIZED BY MENTAT, kept for real-ray fallback images (the flag is
# accepted and ignored -- mentat has no object store at all, which is the
# actual fix). The history, because it justifies mentat's existence:
#
# Ray's object store defaults to ~30% of RAM. On a discrete-GPU box that is
# harmless -- it is host memory the model never wanted. GB10 is UNIFIED memory,
# so it comes straight out of the pool the weights and KV cache need.
#
# Measured 2026-08-27: the head was capped at 4 GB but the worker was not, and
# `ray status` reported 40.15 GiB of object store cluster-wide -- roughly 36 GiB
# of it on the worker, for a model that passes tensors over NCCL and puts
# essentially nothing in the object store.
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"

# NEUTRALIZED BY MENTAT, kept for real-ray fallback images (mentat has no
# memory monitor; nothing in it ever kills a worker on a heuristic).
#
# Ray's memory monitor kills worker processes when the NODE crosses 95% memory.
# That heuristic assumes host RAM and GPU memory are separate pools. On GB10
# they are the same pool, so the ~89 GiB of resident model weights count as
# host memory and the node sits at ~95% whenever the model is simply loaded.
#
# Measured 2026-08-27: rank 0 was killed three times (13:41, 13:52, 14:03),
# each time seconds after the engine came up, with the raylet logging
#   "Memory on the node was 115.68GB / 121.63GB (0.951074)"
#   "Selected to kill: vllm_Worker_..._TP0, pid=3889, actual memory used=2.07GB"
# Ray killed the 2 GB process because the 89 GiB of weights pushed the node
# over its threshold. Nothing was leaking and no CUDA allocation had failed.
#
# This looked like a GPU OOM for hours: NV_ERR_NO_MEMORY appears in dmesg from
# vLLM's own profiling probes, which allocate until they fail by design. Those
# are noise. The kill decision is Ray's, and it is logged only in the raylet
# event log, not in vLLM's output.
#
# 0 disables the monitor. Raising the threshold instead only moves the cliff:
# there is no honest value when the weights alone are 73% of the pool.
export RAY_memory_monitor_refresh_ms="${RAY_MEMORY_MONITOR_REFRESH_MS:-0}"

# --- service announcement (mentat-serve) ------------------------------------
# The agent reads these at `ray start` and carries them in its registration.
# mentat-serve routes by what is announced, and probes it before believing
# it. Purely additive: an image or daemon without this support ignores them.
# Every rank announces its management MCP (the status server runs on both
# roles). Only the head announces the OpenAI endpoint: a worker holds a TP
# rank and serves nothing, and a URL announced from it would route requests
# at a port with nothing behind it. The provider names the engine behind that
# endpoint, which two servers answering /v1/chat/completions do not reveal by
# answering it, so it goes on the rank that announces the endpoint.
#
# Port form (8002/v1): the router resolves it against every address the node
# announces, so a router on the LAN and one on the fabric both reach it. A URL
# pins one address, reachable only from that link. Both servers bind 0.0.0.0.
export MENTAT_MCP_API="${MENTAT_MCP_API:-${STATUS_PORT:-8082}/mcp}"
if [[ "$ROLE" == "worker" ]]; then
  unset MENTAT_OPENAI_API
else
  export MENTAT_OPENAI_API="${MENTAT_OPENAI_API:-${API_PORT:-8002}/v1}"
  export MENTAT_MODEL_PROVIDER="${MENTAT_MODEL_PROVIDER:-vllm}"
fi

# --- worker: join and block -------------------------------------------------
# Under mentat, `ray start --block --address` retries forever, so the
# head-first ordering below stopped mattering -- start either side first. The
# stage-file dance is kept because it still reads well on the status page,
# and because a real-ray fallback image (where `ray start --address` does NOT
# retry) needs the head up first.
if [[ "$ROLE" == "worker" ]]; then
  stage joining
  ( for _ in $(seq 1 120); do
      if ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
        stage worker-ready; break
      fi
      sleep 5
    done ) &
  exec ray start --block --address="$RAY_ADDRESS" \
            --object-store-memory="$RAY_OBJECT_STORE_MEMORY"
fi

# --- head -------------------------------------------------------------------
# --address names the daemon explicitly. mentat 0.7.0 refuses a bare --head
# while RAY_ADDRESS is set ("--head names no daemon, so this agent would
# register with the local one while RAY_ADDRESS names ..."): every agent of a
# group and its driver must reach one daemon, and 0.6.0 let the two disagree
# silently. On the head RAY_ADDRESS is its own daemon, so this changes nothing
# about where it registers -- it only stops mentat having to guess.
ray start --head --address="$RAY_ADDRESS" --node-ip-address="$VLLM_HOST_IP" --port=6379 \
          --object-store-memory="$RAY_OBJECT_STORE_MEMORY"

# Starting the engine without its workers reaches NCCL and dies there
# ("invalid usage"), so this waits rather than giving up and proceeding. The
# stage stays visible on the status page throughout. WORKER_WAIT_S bounds it;
# 0 waits forever, which is what a switch reboot wants.
stage waiting-workers
_waited=0
while :; do
  have=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | cut -d/ -f2 | cut -d. -f1 || echo 0)
  [[ "${have:-0}" -ge "$TP" ]] && break
  if (( ${WORKER_WAIT_S:-0} > 0 && _waited >= ${WORKER_WAIT_S:-0} )); then
    echo "FATAL: only ${have:-0} of "$TP" GPUs after ${_waited}s" >&2
    exit 1
  fi
  (( _waited % 60 )) || echo "waiting for "$TP" GPUs, have ${have:-0} (${_waited}s)"
  sleep 5
  _waited=$(( _waited + 5 ))
done

# The drafter was chosen at the top (SPEC_METHOD). A missing drafter would
# otherwise fail only after the ~10 minute target load, so check it first.
SPEC=()
case "$SPEC_METHOD" in
  dflash)
    SPEC_TOKENS="${SPEC_TOKENS:-7}"
    DFLASH_MODEL="${DFLASH_MODEL:-/models/glm-5.3-flash-dflash2}"
    if [[ ! -f "$DFLASH_MODEL/config.json" ]]; then
      echo "FATAL: SPEC_METHOD=dflash but no config.json under DFLASH_MODEL=$DFLASH_MODEL." >&2
      echo "Mount the DFlash2 drafter there on every node, or set SPEC_METHOD=mtp." >&2
      exit 1
    fi
    SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_MODEL}\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  mtp)
    SPEC_TOKENS="${SPEC_TOKENS:-4}"
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  none) ;;
  *) echo "FATAL: SPEC_METHOD=$SPEC_METHOD; want mtp, dflash or none" >&2; exit 1 ;;
esac
[[ ${#SPEC[@]} -gt 0 ]] && echo "speculative: ${SPEC[*]}"

stage loading

if [[ "${SELF_TEST:-1}" == "1" ]]; then
  (
    if python3 /usr/local/bin/self-test.py \
         --base "http://127.0.0.1:${API_PORT:-8002}" --model "$SERVED"; then
      stage serving
    else
      stage self-test-failed
      echo "!! SELF-TEST FAILED. Serving anyway so the model can be probed;" >&2
      echo "!! the status page reports unhealthy. A wrong-but-healthy model is" >&2
      echo "!! the specific failure this gate exists to catch." >&2
    fi
  ) &
else
  ( for _ in $(seq 1 480); do
      curl -sf -o /dev/null --max-time 4 "http://127.0.0.1:${API_PORT:-8002}/v1/models" \
        && { stage serving; break; }
      sleep 5
    done ) &
fi

# --- tool parser --------------------------------------------------------------
# glm47_failclosed by default: a plugin baked at /usr/local/share, loaded with
# --tool-parser-plugin. It used to arrive through EXTRA_ARGS from a compose
# override, and kv-26gib-override.yaml restated EXTRA_ARGS after it -- compose
# keeps the last value -- so production ran the stock glm47 parser. Selected
# here, it cannot be dropped by an override. TOOL_PARSER=glm47 is the stock one.
TOOL_PARSER="${TOOL_PARSER:-glm47_failclosed}"
TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
if [[ "$TOOL_PARSER" == "glm47_failclosed" ]]; then
  TOOL_ARGS=(--tool-parser-plugin /usr/local/share/glm47_failclosed.py "${TOOL_ARGS[@]}")
fi
echo "tool parser: $TOOL_PARSER"

# --- sparse indexer top-k -------------------------------------------------------
# GB10 cannot run persistent_topk once the KV pool passes ~3.4M tokens (it would
# oversubscribe 48 CTAs), and the FilteredTopK fallback wants 128 KB of shared
# memory per block against the 101,376 B an SM has. per_row is the same
# computation without the persistent-CTA scheme. This used to be a patch
# (gb10_topk_fallback.py); on main it is a flag. "auto" tries cooperative,
# then persistent, then per_row.
TOPK_BACKEND="${TOPK_BACKEND:-per_row}"
TOPK_ARGS=(--sparse-indexer-topk-backend "$TOPK_BACKEND")

# --load-format auto, NOT sharded_state -- see the header.
#
# Parser names do not match the model version, which is normal here: the vendor
# recipe for 5.3-Flash specifies glm47 and glm45.
# --- MoE backend and CUDA graphs -------------------------------------------
# flashinfer_cutlass, the native NVFP4 kernel, which runs W4A4 off the
# checkpoint's per-projection input scales. Production has served on it with
# the nvidia checkpoint since 2026-09-21; why it replaced marlin then is not
# recorded. It decodes 109.8 / 88.8 / 52.6 tok/s (structured / code / prose,
# v6 + DFlash2, 2026-09-23) and passes the thinking-on count-to-200 probe 8/8.
# It takes vLLM's own CUDA graph sizes; CUDAGRAPH_CAPTURE_SIZES applies to
# marlin only.
#
# marlin was the default before that. It is weight-only: it dequantises to FP16
# and runs bf16 activations, so a checkpoint's input scales are never read. It
# was adopted when the native kernels died on sm_121 with
# cudaErrorNoKernelImageForDevice, a failure both MiaAI-Lab and LibertAI later
# put down to one checkpoint's uninitialised input_scale. It is also not
# bit-deterministic at M >> 1.
#
# Pairing marlin with --enforce-eager is recipe convention, not a correctness
# constraint. Marlin's workspace helper reuses storage precisely so a captured
# graph's addresses stay valid, and the only eager gate in this vLLM is for
# DeepseekV4. Capture measures 6 s here. CUDA_GRAPHS=0 restores eager.
#
# Capture sizes are in TOKENS and vLLM rounds them to multiples of (1 + draft
# tokens). Above max_num_seqs * (1 + k) the dispatcher returns NONE and decode
# falls back to eager with nothing in the log to say so, which is why the
# ceiling is computed and checked rather than left to whoever edits the list.
MOE=()
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
if [[ "$MOE_BACKEND" == "marlin" ]]; then
  if [[ "${CUDA_GRAPHS:-1}" == "1" ]]; then
    _k=1
    [[ -n "$SPEC_METHOD" && "$SPEC_METHOD" != none ]] && _k=$(( SPEC_TOKENS + 1 ))
    _ceil=$(( MAX_NUM_SEQS * _k ))
    _largest=$(tr ' ' '\n' <<<"$CUDAGRAPH_CAPTURE_SIZES" | sort -n | tail -1)
    if (( _largest > _ceil )); then
      echo "WARNING: largest capture size $_largest exceeds max_num_seqs*(1+k)=$_ceil;" >&2
      echo "decode above $_ceil tokens will run eager and log nothing." >&2
    fi
    MOE=(--moe-backend marlin
         --cudagraph-capture-sizes ${CUDAGRAPH_CAPTURE_SIZES}
         --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\"}")
    echo "MoE backend: marlin, cudagraphs ${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}" \
         "(sizes: $CUDAGRAPH_CAPTURE_SIZES, ceiling ${_ceil})"
  else
    MOE=(--moe-backend marlin --enforce-eager)
    echo "MoE backend: marlin (enforce-eager, CUDA_GRAPHS=0)"
  fi
else
  MOE=(--moe-backend "${MOE_BACKEND}")
  echo "MoE backend: ${MOE_BACKEND} (vLLM's own CUDA graph sizes)"
fi

# Multimodal: up to 16 images a prompt, no video. Exceeding a cap is a clean
# 400 with an OpenAI-shaped error body, not an engine fault, but a client that
# treats any 400 as fatal dies on it.
# The closing brace is escaped: bash ends a :- default at the first unescaped
# one, so it would cut the default short and append the tail as literal text.
LIMIT_MM="${LIMIT_MM:-{\"image\":16,\"video\":0\}}"
python3 -c 'import json, sys; json.loads(sys.argv[1])' "$LIMIT_MM" 2>/dev/null || {
  echo "FATAL: LIMIT_MM is not valid JSON: $LIMIT_MM" >&2
  echo "       Write the whole object, closing brace included." >&2
  exit 1; }
MM=(--limit-mm-per-prompt "$LIMIT_MM")
# 0: run the max-size dummy forward at init, so the vision encoder's peak is
# budgeted at startup instead of coming out of the headroom a long prefill
# needs at run time. 1 skips it, which is faster to boot and was the default
# while images were capped at 4.
[[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && MM+=(--skip-mm-profiling)

# The checkpoint ships a TEXT-ONLY template. Its media branch renders
# "<reminder>You are unable to process this image ...</reminder>" and emits no
# placeholder, so vLLM's processor extracts the image features, scans the
# prompt for <|begin_of_image|><|image|><|end_of_image|> to replace, finds
# nothing, and every image request dies in _apply_prompt_updates with
# "Failed to apply prompt replacement for mm_items['image'][0]". The weights
# are fine: 347 vision tensors and all three token ids are in the checkpoint.
#
# chat-template.jinja beside this file is that same template with the media
# branch emitting what vLLM scans for. A checkpoint that grows its own image
# handling takes precedence again, so this stops applying by itself.
TMPL=()
: "${CHAT_TEMPLATE:=}"
if [[ -z "$CHAT_TEMPLATE" ]]; then
  if grep -qF '<|begin_of_image|>' "$MODEL/chat_template.jinja" 2>/dev/null; then
    CHAT_TEMPLATE="$MODEL/chat_template.jinja"
  elif [[ -f /usr/local/share/glm53-chat-template.jinja ]]; then
    CHAT_TEMPLATE=/usr/local/share/glm53-chat-template.jinja
  elif [[ -f "$MODEL/chat_template.jinja" ]]; then
    CHAT_TEMPLATE="$MODEL/chat_template.jinja"
  fi
fi
if [[ -n "$CHAT_TEMPLATE" ]]; then
  TMPL=(--chat-template "$CHAT_TEMPLATE")
  echo "chat template: $CHAT_TEMPLATE"
fi

# A 320B MoE takes far longer to init than the default allows.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# --- KV cache pin ----------------------------------------------------------
# Left unset, vLLM PROFILES the KV cache and takes everything available -- here
# that was 13.59 GiB, giving 796,361 tokens. FlashInfer's autotune then runs
# during warmup, asks the driver for more, and there is none:
#
#   NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal
#   RayWorkerProc rank=[0] died unexpectedly
#
# On a discrete GPU the profiler's headroom assumptions usually survive this.
# On GB10's unified memory they do not, which is what the upstream recipe means
# by "--kv-cache-memory pin (UMA OOM only)". Pin it below the profiled figure so
# warmup has room.
#
# Measured 2026-08-27: a 10 GiB pin made things WORSE, because skipping the
# profile also discards the gpu_memory_utilization margin autotune relies on.
# That was before TORCH_MEM_FRACTION and the ephemeral autotune cache.
#
# 26 GiB is the pin that serves under DFlash2. At 28 the head sat near 1 GiB
# free and eight concurrent long-context requests had a worker OOM-killed. A
# pin also stops the pool moving by ~0.9 GiB between identical boots with
# page-cache timing at profiling. DFlash2 costs 41% of the pool: 3,437,736
# tokens with speculation off against 2,024,644 with it (pre-nightly image,
# 2026-09-06).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-27917287424}"
KV_ARGS=(--kv-cache-memory "${KV_CACHE_MEMORY}")
echo "kv-cache-memory pinned to ${KV_CACHE_MEMORY} bytes ($(( KV_CACHE_MEMORY / 1073741824 )) GiB)"

# eager stages each shard through anonymous memory and loads faster -- 511 s
# against 690 s for lazy. Unpinned, its buffers were still resident when vLLM
# profiled for KV and cost 38% of the cache: 810,576 tokens against 1,296,922.
# With the KV pin above the pool is not profiled, and production runs eager.
# Go back to lazy if the pin is ever dropped.
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --distributed-executor-backend ray \
  --load-format auto \
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-eager}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.88}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --block-size "${BLOCK_SIZE:-2304}" \
  "${KV_ARGS[@]}" \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" \
  "${MOE[@]}" \
  "${MM[@]}" \
  "${TMPL[@]}" \
  ${ITERATION_DETAILS:+--enable-logging-iteration-details} \
  "${TOOL_ARGS[@]}" \
  "${TOPK_ARGS[@]}" \
  --reasoning-parser glm45 \
  "${SPEC[@]}" \
  --host 0.0.0.0 --port "${API_PORT:-8002}" \
  ${EXTRA_ARGS:-}
