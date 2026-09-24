# GLM-5.3-Flash NVFP4 on ASUS GX10 / GB10 (sm_121), TP=4 over RoCE.
#
# Built on a stock vLLM NIGHTLY. GLM-5.3-Flash (glm5_next) landed on main, so
# the per-model pre-merge image this used to start from
# (vllm/vllm-openai:glm53-flash-arm64-cu130) is no longer needed, and neither
# are most of the patches that sat on top of it. Audited 2026-09-22 against
# nightly 0961bbae, patch by patch:
#
#   dropped, now upstream:
#     thinking_budget_guard.py      the V2 sampler applies thinking_token_budget
#     kpool-fix indexer.py          tail slot mapping written into a persistent
#                                   buffer (out=slot_mapping_buffer)
#     kpool-fix mamba_hybrid.py     positions= passed to build_attn_metadata
#     dflash registry/qwen3_dflash2/speculator/spec_decode_init
#   dropped, now a flag:
#     gb10_topk_fallback.py         --sparse-indexer-topk-backend per_row
#   dropped, no longer needed:
#     flashinfer nightly pin        the nightly ships 0.6.18.post1
#     NCCL 2.30.7 pin               the nightly ships 2.30.7
#     cutlass-dsl 4.6.2 pin         the nightly ships 4.7.1; watch this one
#     model_runner.py mount         PP-only; this runs PP=1
#   added:
#     glm53_mtp_bf16.py             MTP layer is BF16 but config.json says NVFP4
#     sm90_fp8_kv_dtype.py          SM90 builder planned fp8 KV as uint8
#     glm53_reasoning_always_parsed.py  thinking off = low effort, still <think>
#     glm53_kpool_tail_ring.py      tail ring sized for spec decode (vllm#55219 part)
#   kept:
#     glm53-flash_SM121.py (MiaAI)  6 of 7 edits apply unchanged; see below
#     gb10_plugin_backend.py        MiaAI vs LibertAI sparse-MLA switch
#     worker_memory_cap.py          ours: TORCH_MEM_FRACTION
#     spin_wait.py                  was a whole-file shm_broadcast.py mount
#     glm47_failclosed.py           tool parser plugin, import path ported
#     glm53_eagle3_aux.py           upstream GLM5next exposes no aux hidden states;
#                                   retargeted to models/glm5next/common/model.py
#     glm53_dflash2_kv_groups.py    drafter KV groups on the GLM-5-Next layout
#                                   (the old whole-file kv_cache_utils.py fork,
#                                   reduced to what the nightly lacks)
#
# GB10 is sm_121 and this torch reports up to sm_120. That is correct: sm_120
# cubins run on sm_121. Do not widen TORCH_CUDA_ARCH_LIST.
ARG BASE=vllm/vllm-openai:nightly-0961bbae2894d574be790d219651824eb199318e
# Pulled registry-qualified: a bare `mentat-artifacts:<ver>` tag resolves to a
# local image, and those have been mislabelled before -- every image "pinned"
# to 0.4.0 once silently ran 0.1.0. Verify with `mentatd --version`.
# Must match the host daemons (mentatd and mentatd-serve): 0.12.0 speaks
# protocol 0.99, and a shim from 0.8 or older cannot join them.
ARG MENTAT_VERSION=0.12.0

FROM mmastrac/mentat-artifacts:${MENTAT_VERSION} AS mentat
FROM ${BASE}

# nvcc's include path is missing the headers whose libraries this base ships,
# which breaks any FlashInfer JIT path (nvrtc.h, then -lnvrtc).
COPY image/patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# --- sm_121 sparse MLA -------------------------------------------------------
# This checkpoint is NoPE (qk_rope_head_dim 0). On capability 12 the nightly
# still offers only TRITON_MLA and FLASHINFER_MLA_SPARSE_SM120, and SM120
# mandates the packed fp8_ds_mla layout, which NoPE cannot use -- the engine
# dies in concat_and_cache_mla after a ~10 minute weight load. The SM90 backend
# already covers NoPE (kpe in {0, 64}) but is gated to Hopper; FA3 is, FA2 is
# not. MiaAI's patcher lists SM90 on capability 12 and swaps FA2 in off Hopper,
# plus FA2's fp8 gate and CTA tile for ~101 KB smem, PDL off (a KDA race on
# this hardware), a kpool pid clamp, and initialising the indexer's top-k
# buffer to -1 instead of torch.empty.
#
# That last edit is the only one whose file moved: sparse_attn_indexer_kpool.py
# became models/glm5next/nvidia/sparse_indexer.py, with the same two
# allocations byte for byte. The sed retargets MiaAI's path and nothing else,
# so the vendored file stays verbatim (MIT, licence beside it). The patcher
# still asserts every anchor matches exactly once.
COPY image/patches/glm53-flash_SM121.py image/patches/LICENSE.MiaAI-Lab /tmp/mia/
RUN sed -i 's|model_executor/layers/sparse_attn_indexer_kpool.py|models/glm5next/nvidia/sparse_indexer.py|' /tmp/mia/glm53-flash_SM121.py \
 && python3 /tmp/mia/glm53-flash_SM121.py && rm -rf /tmp/mia

# MiaAI's patcher and LibertAI's plugin both solve the sm_121 MLA problem, and
# the first shuts the second out. Make the choice an env var -- see the script.
COPY image/patches/gb10_plugin_backend.py /tmp/gb10_plugin_backend.py
RUN python3 /tmp/gb10_plugin_backend.py && rm /tmp/gb10_plugin_backend.py

# The SM90 backend's metadata builder plans FlashInfer's MLA wrapper with
# kv_cache_spec.dtype, which for an fp8 cache is the uint8 STORAGE dtype, and
# plan() rejects it at the first autotune run. The pre-merge image passed
# float8_e4m3fn; this restores that. Not GB10-specific.
COPY image/patches/sm90_fp8_kv_dtype.py /tmp/sm90_fp8_kv_dtype.py
RUN python3 /tmp/sm90_fp8_kv_dtype.py && rm /tmp/sm90_fp8_kv_dtype.py

# Nothing in vLLM bounds a worker's memory, and on unified memory an unbounded
# worker takes the node down rather than its own request.
COPY image/patches/spark_mem_trace.py /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
COPY image/patches/worker_memory_cap.py /tmp/worker_memory_cap.py
RUN python3 /tmp/worker_memory_cap.py && rm /tmp/worker_memory_cap.py

COPY image/patches/spin_wait.py /tmp/spin_wait.py
RUN python3 /tmp/spin_wait.py && rm /tmp/spin_wait.py

# DFlash2 reads auxiliary hidden states from target layers 5, 14, 24, 33, 42.
# Upstream serves the DFlash2 drafter but its GLM5next model exposes no aux
# states, so without this `--speculative-config dflash` fails at load with
# "Model does not support EAGLE3 interface". Inert unless a drafter asks for
# aux layers. Only the target path changed from the vendor-base version.
COPY image/patches/glm53_eagle3_aux.py /tmp/glm53_eagle3_aux.py
RUN python3 /tmp/glm53_eagle3_aux.py && rm /tmp/glm53_eagle3_aux.py

# nvidia/GLM-5.3-Flash-NVFP4 stores its MTP layer (45) in BF16, but its
# config.json declares every Linear NVFP4 and does not list layer 45 in
# `ignore`, so MTP dies at load: "size of tensor a (256) must match ... (512)"
# in RoutedExperts._load_w2. This excludes the MTP layer from quantisation when
# the checkpoint holds no scales for it -- driven by the weights, so a
# checkpoint that does quantise its MTP layer is left alone.
COPY image/patches/glm53_mtp_bf16.py /tmp/glm53_mtp_bf16.py
RUN python3 /tmp/glm53_mtp_bf16.py && rm /tmp/glm53_mtp_bf16.py

# Thinking off no longer closes an empty <think></think> (out of distribution for
# this model; it breaks long structured output). The template maps it to
# `Reasoning Effort: Low` and still opens <think>, so the reasoning parser must
# track <think> even when the request says thinking is off.
COPY image/patches/glm53_reasoning_always_parsed.py /tmp/glm53_reasoning_always_parsed.py
RUN python3 /tmp/glm53_reasoning_always_parsed.py && rm /tmp/glm53_reasoning_always_parsed.py

# The kpool tail ring held exactly index_kpool (4) slots, but a spec-verify step
# stashes 1 + num_spec rows (4 with MTP, 8 with DFlash2) before acceptance: a
# rejected pool-completing draft let the drafts behind it overwrite that pool's
# committed keys, and the redo wrote a wrong indexer key for good. The ring now
# covers a spec step. Upstream fix pending (the ring part of vllm#55219).
COPY image/patches/glm53_kpool_tail_ring.py /tmp/glm53_kpool_tail_ring.py
RUN python3 /tmp/glm53_kpool_tail_ring.py && rm /tmp/glm53_kpool_tail_ring.py

# DFlash2's drafter adds five sliding-window attention layers, and the nightly's
# GLM-5-Next KV grouper gives up on any spec that is not MLA, so the model falls
# onto the generic path and dies in unify_kv_cache_spec_page_size ("page size is
# not divisible by the maximum page size"). This keeps the target's groups and
# gives the drafter layers a group of their own, laid out after the indexer
# regions and opted out of prefix caching. Inert without a drafter.
COPY image/patches/glm53_dflash2_kv_groups.py /tmp/glm53_dflash2_kv_groups.py
RUN python3 /tmp/glm53_dflash2_kv_groups.py && rm /tmp/glm53_dflash2_kv_groups.py

# The fail-closed GLM tool parser, loaded with --tool-parser-plugin. Baked and
# selected by the entrypoint rather than passed through EXTRA_ARGS: a compose
# override that restated EXTRA_ARGS dropped it, twice, and production ran the
# stock parser without anyone noticing.
COPY image/patches/glm47_failclosed.py /usr/local/share/glm47_failclosed.py

#   iproute2  the entrypoint finds this node's ConnectX address by scanning
#             `ip -o -4 addr` for CLUSTER_SUBNET, and derives GLOO_SOCKET_IFNAME
#             from whichever interface owns it; the port differs per box.
#   ibverbs   NCCL's RoCE data plane. Missing, NCCL silently falls back to TCP
#             over the LAN and every TP step crawls.
#   ripgrep/pciutils/usbutils are for the MCP inventory and search tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 libibverbs1 librdmacm1 ibverbs-providers \
        ripgrep pciutils usbutils \
    && rm -rf /var/lib/apt/lists/* \
    && command -v ip && command -v rg

# mentat replaces Ray. Unlike the single-GPU tenants, TP=4 across four boxes
# really does need placement, so this installs the daemon binary and the `ray`
# symlink as well as the shim package. Real Ray on unified memory caused two
# outages this exists to prevent: its memory monitor killed healthy ranks at 95%
# node memory (the ~89 GiB of weights ARE host memory), and its object store
# took up to ~36 GiB for a model that moves every tensor over NCCL. On a vLLM
# bump, re-audit the shim with: grep -rn 'ray\.' <site-packages>/vllm/v1/executor/
COPY --from=mentat /out/ /tmp/mentat-out/
# mentatd-probe-machine is how the agent learns this box's GPUs from 0.9.0 on
# ("Move the machine inventory into a probe script"). Without it `ray start`
# registers ZERO GPUs, and vLLM's placement dies with "required GPUs exceeds
# the total number of available GPUs". It must sit beside mentatd or on PATH.
RUN install -m755 /tmp/mentat-out/mentatd /usr/local/bin/mentatd \
    && install -m755 /tmp/mentat-out/mentatd-probe-machine /usr/local/bin/mentatd-probe-machine \
    && ln -sf /usr/local/bin/mentatd /usr/local/bin/ray \
    && pip install --no-cache-dir --no-deps /tmp/mentat-out/mentatd-*-py3-none-any.whl \
    && rm -rf /tmp/mentat-out \
    && ray --version

COPY image/verify-base.py /tmp/verify-base.py
RUN python3 /tmp/verify-base.py && rm /tmp/verify-base.py

COPY image/chat-template.jinja /usr/local/share/glm53-chat-template.jinja
COPY vllm/status-server.py image/self-test.py /usr/local/bin/
COPY image/entrypoint.sh /entrypoint.sh
RUN chmod +x /usr/local/bin/status-server.py /usr/local/bin/self-test.py /entrypoint.sh

# 8002 = OpenAI API, 8082 = startup status page, bound from container start
EXPOSE 8002 8082
ENTRYPOINT ["/entrypoint.sh"]
