# Greedy decoding is not deterministic at long context

GLM-5.3-Flash-NVFP4, vLLM `0.1.dev20051+g487ecf187`, 4x GB10 (sm121) at TP=4.

## What happens

At `temperature=0` with a fixed seed, identical requests sometimes return
different completions. Most differences are invisible. The one that reaches a
user is a corrupted **tool name**: `glm47_moe` sets `validate_tool_names=True`,
so a call whose name is not in the request's tool list emits **zero deltas** and
finishes `stop` with no content and no tool calls. Clients report it as "model
returned a completed response with no content" and retry it as a dead endpoint.

## Cause

**The marlin NVFP4 MoE is not bit-deterministic at M >> 1**, and the model
amplifies that into whole-token differences.

Measured, in order:

1. Among runs that answer *correctly* at the failing position, P(correct token)
   ranges 1.00 down to 0.48 -- 35 distinct values in 38 runs. A wrong token is
   the tail of that spread, not a separate failure mode.
2. Per-layer tap across identical runs: layers 0-2 and layer 3's attention input
   are bit-identical; layer 3's attention output takes 12 distinct values.
3. Fixing the FA2 input order makes layer 3 bit-identical, and the first
   divergence moves to layer 4 with identical input -- so it arrives in the KDA
   recurrent state, which prefill produced.
4. Prefill tap: layer 3's `mlp_out`, the first marlin MoE, differs in 8 of 8
   runs at M=800.
5. In-engine, evaluating that MoE block twice on the same input differs at
   M=800 and M=2304, and is bit-stable at M=1.
6. Padding the prompt so the uncached recompute is a single row makes decode
   bit-identical across 11 runs, spread 0.000000.

Amplification is what turns ulp noise into a different token: 288-expert
routing plus eleven sparse selections per token. With layer 3 pinned, the
indexer's 512th logit still spreads by 1-30 units by layer 39.

## Do not use the prefill re-score as an oracle

An earlier version of this document treated a token's logprob under
`prompt_logprobs` as ground truth and read a 13-21 logprob gap as proof that
decode was wrong. It is not: the re-score is its own multi-row prefill, so it is
another draw of the same nondeterministic computation, taken at n=1. It shows
that two draws disagreed, not which one is right.

## Ruled out, by measurement

Top-k selection, both kernels: values match `torch.topk` exactly at every
coarse-bin population up to a whole row in one bin, and real indexer logits hold
exact boundary ties in 3 of 34,320 layer-rows, none at the layer where
divergence enters. The FA2 sparse-MLA kernel (exact, bit-repeatable). DeepGEMM
MQA logits (bit-deterministic; paged equals non-paged). CUDA graphs (eager
reproduces). Speculative decoding (off reproduces). KV dtype, memory pressure,
prefix caching, temperature, conversation history, the request payload, the
uninitialised `topk_indices_buffer`, the MoE router's Python fallback (the fused
kernel runs here), and TP/NCCL (ranks agree bitwise).

Consequently neither `hpc.topk_filtered` (Tencent/hpc-ops#93) nor the upstream
sm121 determinism set -- vllm#55122, the MoE finalize knob, vllm#53899 -- touches
this. `use_fused_finalize` belongs to the FlashInfer CUTLASS path; the backend
here is marlin.

## Reading the rates

Per-condition rates ranged 2.5%-22.5%, including the same configuration twice at
13% and 22.5%. At n=40 nothing in that spread is distinguishable, and comparing
conditions on those numbers cost a night. Size the sample from the expected rate
first, or measure determinism directly with a per-layer tap, which decides on a
single pair of runs.

## Writing a tap

Three ways it silently produces nothing, all of them paid for:

- `GLM53_TAP` in the container does not reach the worker: mentat spawns actors
  without forwarding it. Gate on a file under `/logs` instead.
- Gating on `t.shape[0] == 1` to exclude prefill excludes all of decode, because
  speculative decode carries 1 + draft tokens per step.
- `.item()` inside the forward invalidates CUDA graph capture and crash-loops the
  engine. Accumulate into a preallocated device tensor, and keep the step counter
  on device too: host code does not run on graph replay, so a Python counter
  freezes at its capture value.

## The MoE kernel is not the whole story

2026-09-13, NVIDIA's ModelOpt NVFP4 checkpoint with `MOE_BACKEND=flashinfer_cutlass`
(which does run on sm_121 -- the entrypoint's `cudaErrorNoKernelImageForDevice`
warning and the upstream issues are about auto-selection skipping CUTLASS, not
about the kernel being absent):

| condition | distinct completions from 16 |
|---|---|
| marlin, RedHat weights, content only, 48 tokens | 3 |
| CUTLASS, NVIDIA weights, content only, 48 tokens | 9, sizes [6,2,2,1,1,1,1,1,1] |
| CUTLASS, NVIDIA weights, reasoning included, 400 tokens | 16 |

Replacing the kernel this document blames did not restore reproducibility, and
under matched conditions it is worse. Two variables differ between the first two
rows, kernel and checkpoint, so the attribution needs marlin on the NVIDIA
weights -- one flag, the same boot -- before the kernel is cleared or convicted.
One sample of 16 each, and the sizing warning above applies.

The third row is not comparable to the others: it hashes the reasoning trace as
well as the content, which finds divergence a 48-token content hash cannot. That
is the better test of the forward pass and the worse test for comparing against
the baseline.

## Open

A deterministic NVFP4 MoE at M >> 1 on sm121, in any backend. The
single-row-recompute trick removes the variance but only for prompts whose
uncached tail is one token.

## Reproducing

`repro/` holds three scripts and what each one measures. `repro/greedy_nondet.py`
is the small one: standard library, one repeated paragraph, 3 distinct
completions from 16 identical greedy requests at 42k and 1 from 16 at 2k.
