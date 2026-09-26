# Decode-path kernel tests (GLM-5.3-Flash on GB10, spark-glm53:v4)

Written 2026-09-23 while chasing the count-to-200 token corruption. Each file is
self-contained and runs inside the image on one GB10; none needs the weights.

    scp <file> admin@<box>:/tmp/kvtest/
    sudo docker run --rm --gpus all --ipc=host -v /tmp/kvtest:/kv \
        --entrypoint python3 -w / localhost:5000/spark-glm53:v4 /kv/<file>

| file | what it proves | result on v4 |
|---|---|---|
| `_topk_mhcpost_fp8mla_test.py` | T1 per_row top-k + expand_pools_and_append_tail return exactly the causal set for short rows under any garbage past the row; T3 mhc_post_tilelang is pure and matches the torch reference; T5 SM90 fp8 MLA wrapper planned as float8_e4m3fn (patches/sm90_fp8_kv_dtype.py) matches a bf16 reference at <=0.5% on 8-row verify shapes and across 2048 | PASS |
| `_kda_spec_vs_plain_test.py` | T2 causal_conv1d_update + fused_recurrent_kda spec path (k=3 and k=7, random acceptance) is bit-identical to the one-token-per-step plain path; run at H=64 and H=16 | PASS |
| `_cudagraph_replay_test.py` | T6a SM90 wrapper captured at the dummy lens vLLM uses (1..8), replayed at real lens after replan: exact. T6b DeepGEMM paged MQA logits + top-k under replay still select the full causal pool set (its logits reference is wrong: my test lays scales per row, DeepGEMM keeps them per page; ignore rel_err) | T6a PASS, T6b selection PASS |
| `_cutlass_nvfp4_moe_rows_test.py` | T7 FlashInfer CUTLASS NVFP4 fused MoE at the per-rank shape (E=288,K=4096,N=512,topk 8): every batched row (M up to 64) is bit-identical to the same row run alone; repeat calls are deterministic | PASS |
| `_cutlass_nvfp4_moe_sanity_test.py` | T7b outputs are finite, non-zero, input-dependent and row-local (my torch dequant reference disagrees by ~70%; that is the reference's scale convention, not the kernel) | as described |

`_cudagraph_replay_test.py` needs the import fix already applied (vllm.utils.platform_utils).

Added after the spec-off result (plain decode corrupts too):

| file | what it proves | result on v4 |
|---|---|---|
| `_tail_compaction_test.py` | T8 production-shaped top-k rows (pools, -1 gap, tail at columns 2048+) through the SM90 index remap: every tail slot is inside the first `ctx` compacted entries the plan reads, eager and under graph replay, positions 99..5003 | PASS |
| `_kda_numerics_test.py` | T9 FlashKDA prefill (the nightly's new SM12x default), Triton chunk prefill (old image path) and fused_recurrent_kda plain decode vs an fp32 torch reference of the kernel math; state seeded from prefill; L=20/300/2500 | PASS (<=1.1% bf16 error, decode state rel err ~1e-6) |
| `_cudagraph_replay_shapes_test.py` | T6a at the spec-off shape: one request captured with dummy seq_len 1, replayed at 100/700/2051 and back down to 37 | PASS (the R=4 case aborts on a test-side cache size, not a kernel error) |

## Collective-corruption test on the four GLM boxes (no model loaded)

`nccl_collective_test.py` + `nccl_test_launch.sh`. The launcher derives the
per-box network exactly as entrypoint.sh does (LAN address for the bootstrap
socket, RoCE devices and GID index for the data plane, NCCL_MAX_NCHANNELS=8)
and runs the test, which emulates a TP=4 decode step's collectives (90 bf16
all-reduces of [M,4096] per step, one bf16 logits all-gather [M,38720],
a [2304,4096] "prefill" all-reduce every 200 steps) with deterministic inputs
and checks every result (bit-exact for integer data and all-gathers, 2 bf16
ulps for float data, all ranks identical, repeat identical).

Copy both files to the same path on all four boxes (e.g. /home/admin/kernel-tests).
Ranks: node0 (192.0.2.1) = 0, 192.0.2.2 = 1, 192.0.2.3 = 2, 192.0.2.4 = 3. Run the four
commands within ~60 s of each other (rank 0 first). VLLM_HOST_IP is the box's
LAN address (what production passes); the launcher derives the RoCE side.

    # on each box, with RANK and VLLM_HOST_IP set for that box:
    sudo docker run --rm --gpus all --network host --ipc host --ulimit memlock=-1 \
      --device /dev/infiniband:/dev/infiniband --cap-add SYS_PTRACE \
      -v /home/admin/kernel-tests:/kt \
      -e RANK=<0..3> -e WORLD_SIZE=4 -e MASTER_ADDR=192.0.2.1 -e MASTER_PORT=29555 \
      -e VLLM_HOST_IP=<this box's 192.0.2.x> -e CLUSTER_SUBNET=198.51.100. \
      $VARIANT --entrypoint bash spark-glm53:v5 /kt/nccl_test_launch.sh --mode vllm --iters 2000

Variants (VARIANT):
  default              (nothing)
  simple protocol      -e NCCL_PROTO=Simple
  ring algorithm       -e NCCL_ALGO=Ring
  no RDMA (TCP)        -e NCCL_IB_DISABLE=1
  both HCAs            -e FABRIC_SUBNETS="198.51.100. 203.0.113."   (only if production uses both)
  communicator         --mode pynccl   |   --mode torch          (instead of --mode vllm)

Each rank prints its NCCL_* env, a progress line every 100 steps, every
failure with step/collective/M/first bad index, and rank 0 prints
`RESULT: failures per rank [...], collectives per rank N, CLEAN|CORRUPT`.
2000 steps x 90 all-reduces x 2 = 360k all-reduces plus 2000 all-gathers;
expect a few minutes. NCCL_DEBUG=INFO on top shows the devices NCCL chose.

## Real-weight MoE value test (T10)

`_cutlass_nvfp4_moe_realweights_test.py [nvidia|rh]` needs `/kv/moe-l5-<name>.safetensors`
(16 experts of layer 5 plus the gate, extracted read-only from the checkpoint; the
extractor is scratch code, ~220 MiB each). Result on v4, both checkpoints: the
FlashInfer CUTLASS kernel matches the w4a4 reference (dequantised weights, activations
quantised by vLLM's own scaled_fp4_quant with the checkpoint's global input scale)
within 2-4%; the w4a4 result differs from the w4a16 quantity (what marlin computes)
by 13-20% at |x|<~2, 25-30% at 2x that, 62% once inputs exceed the calibrated amax
(6 * 448 * input_scale = 1.88 for gate/up on both checkpoints), because the static
global activation scale clips every element above it.
