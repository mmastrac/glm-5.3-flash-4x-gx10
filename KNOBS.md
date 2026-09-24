# Knobs

Environment variables read by `entrypoint.sh`, extracted mechanically.
A default of _(none)_ means the variable is referenced without one --
check the entrypoint for whether it is required or merely optional.
Meaning and rationale stay in the entrypoint and compose comments.

| variable | default |
|---|---|
| `API_PORT` | `8002` |
| `AUTOTUNE_CACHE` | _(empty)_ |
| `BLOCK_SIZE` | `2304` |
| `CACHE_ROOT` | `/root/.cache` |
| `CACHE_TAG` | `${SHARED_TAG}-${_opthash}` _(derived)_ |
| `CLUSTER_SUBNET` | _(none)_ |
| `CUDAGRAPH_CAPTURE_SIZES` | `8 16 32 64 96 128 192 256` |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` |
| `CUDA_GRAPHS` | `1` |
| `DECODE_RESERVE_TOKENS` | _(empty)_ |
| `DFLASH_MODEL` | `/models/glm-5.3-flash-dflash2` |
| `EXTRA_ARGS` | _(empty)_ |
| `FABRIC_SUBNETS` | `$CLUSTER_SUBNET` _(derived)_ |
| `FABRIC_WAIT_S` | `0` |
| `GLOO_SOCKET_IFNAME` | _(empty)_ |
| `GPU_MEM_UTIL` | `0.88` |
| `KV_CACHE_DTYPE` | `fp8_e4m3` |
| `KV_CACHE_MEMORY` | `27917287424` |
| `LIMIT_MM` | `{\"image\":16,\"video\":0\}` |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `2304` |
| `MAX_JOBS` | `4` |
| `MAX_MODEL_LEN` | `524288` |
| `MAX_NUM_BATCHED_TOKENS` | `16384` |
| `MAX_NUM_SEQS` | `32` |
| `MCP_LOG_DIR` | `/logs` |
| `MENTAT_GROUP` | `${SERVICE_NAME:-glm53}` _(derived)_ |
| `MENTAT_MCP_API` | `${STATUS_PORT:-8082}/mcp` _(derived)_ |
| `MENTAT_MODEL_PROVIDER` | `vllm` |
| `MENTAT_OPENAI_API` | `${API_PORT:-8002}/v1` _(derived)_ |
| `MODEL_DIR` | `/models/glm-5.3-flash-nvfp4` |
| `MOE_BACKEND` | `flashinfer_cutlass` |
| `MTP` | `1` |
| `NCCL_DEBUG` | `INFO` |
| `NCCL_IB_GID_INDEX` | _(empty)_ |
| `NCCL_IB_HCA` | _(empty)_ |
| `NCCL_MAX_NCHANNELS` | `8` |
| `NCCL_SOCKET_IFNAME` | `$GLOO_SOCKET_IFNAME` _(derived)_ |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| `RAY_ADDRESS` | `${HEAD_HOST:?set HEAD_HOST to the head node address}:6379` _(derived)_ |
| `RAY_MEMORY_MONITOR_REFRESH_MS` | `0` |
| `RAY_OBJECT_STORE_MEMORY` | `4294967296` |
| `ROCE_SETTLE_S` | `60` |
| `ROLE` | `head` |
| `SAFETENSORS_LOAD_STRATEGY` | `eager` |
| `SELF_TEST` | `1` |
| `SERVED_NAME` | `glm53` |
| `SERVICE_NAME` | `glm53` |
| `SHARED_TAG` | `glm53-${_arch}-${_kv}` _(derived)_ |
| `SKIP_MM_PROFILING` | `0` |
| `SPEC_METHOD` | `dflash` |
| `SPEC_TOKENS` | `7` |
| `STAGE_FILE` | `/tmp/glm53-stage` |
| `STATUS_PORT` | `8082` |
| `TILELANG_CACHE_DIR` | `${CACHE_ROOT}/${SHARED_TAG}/tilelang` _(derived)_ |
| `TOOL_PARSER` | `glm47_failclosed` |
| `TOPK_BACKEND` | `per_row` |
| `TORCH_MEM_FRACTION` | `0.92` |
| `TP` | `4` |
| `TRITON_CACHE_DIR` | `${CACHE_ROOT}/${SHARED_TAG}/triton` _(derived)_ |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` |
| `VLLM_HOST_IP` | _(empty)_ |
| `WORKER_WAIT_S` | `0` |
