# GLM-5.3-Flash NVFP4 on ASUS GX10 / GB10 (SM121a), TP=2 over RoCE.
#
# Unlike ds4-flash/Dockerfile this compiles NOTHING. GLM-5.3-Flash landed on
# 2026-08-26 and its glm5_next architecture is not in any vLLM release -- support
# is PR vllm-project/vllm#53906, which vLLM ships ahead of the merge as a
# per-model image. That image is the base here, so the ~67 min SM12x build the
# DS4 image pays for is simply not incurred.
#
# GB10 is NOT on the vendor's verified list (H100/B200/GB200 are), which reads
# like a blocker and is not one. The base image's TORCH_CUDA_ARCH_LIST stops at
# 12.0 while GB10 is sm_121, but its torch reports
#   ['sm_80','sm_90','sm_100','sm_110','sm_120']
# which is byte-identical to the torch in the ds4-flash image already serving on
# these boxes -- sm_120 cubins run on sm_121. Verified by inspection on
# gx10-9722 before this file was written. Do not "fix" the arch list.
#
# The stock image alone is NOT sufficient, which cost a working configuration to
# learn. It offers only FLASHINFER_MLA_SPARSE_SM120 on compute capability 12,
# and that backend mandates the packed fp8_ds_mla layout with pe_dim == 64.
# This checkpoint is NoPE (qk_rope_head_dim: 0), so the kernel refuses and the
# engine dies in concat_and_cache_mla -- after a full ~10 minute weight load.
# Setting --kv-cache-dtype auto does not help: the backend coerces it back.
#
# patches/glm53-flash_SM121.py (MIT, MiaAI-Lab) is what closes that gap. See
# patches/README.md; the short version is that the SM90 sparse-MLA backend is
# not inherently Hopper-only -- its FA3 kernel is, but FA2 is portable to
# sm_12x -- so the fix is gating that path onto capability 12 rather than
# writing a new kernel.
ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130
# Pulled registry-qualified: a bare `mentat-artifacts:<ver>` tag resolves to a
# local image, and those have been mislabelled before -- every image "pinned"
# to 0.4.0 silently ran 0.1.0. Verify with `mentatd --version`, never the tag.
ARG MENTAT_VERSION=0.7.0
FROM mmastrac/mentat-artifacts:${MENTAT_VERSION} AS mentat
FROM ${BASE}

# --- SM121 kernel layer -----------------------------------------------------
# Pin order matters: the FlashInfer nightly drags in an older NCCL and skews
# cutlass-dsl, and both have to be put back for this hardware. The
# flashinfer-jit-cache package is REMOVED so the image cannot load SM120 cubins
# and quietly take the packed path we are trying to avoid.
ARG FLASHINFER_VERSION=0.6.18.dev20260819
RUN pip install --no-cache-dir --pre \
        "flashinfer-python==${FLASHINFER_VERSION}" \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
        --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# The CX7 link between gx10-5818 and gx10-2353 needs this NCCL.
ARG NCCL_VERSION=2.30.7
RUN pip install --no-cache-dir "nvidia-nccl-cu13==${NCCL_VERSION}" \
    && pip show nvidia-nccl-cu13 | grep -q "Version: ${NCCL_VERSION}"

# GB10 CuTeDSL wants this back after the nightly skews it.
ARG CUTLASS_DSL_VERSION=4.6.2
RUN pip install --no-cache-dir "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
    && pip show nvidia-cutlass-dsl | grep -q "Version: ${CUTLASS_DSL_VERSION}"

# Default mode is sm90 (seven patches). Do NOT pass sm120 -- that packed path is
# the one this model cannot use. The script asserts each replacement matches
# exactly once and aborts otherwise, so a moved base image fails the build
# rather than yielding a half-patched tree.
COPY patches/glm53-flash_SM121.py /tmp/glm53-flash_SM121.py
RUN python3 /tmp/glm53-flash_SM121.py && rm /tmp/glm53-flash_SM121.py

# GB10 refuses persistent_topk once the KV pool passes roughly 3.4M tokens, and
# the FilteredTopK fallback it reaches for wants 128 KB of shared memory per
# block against the 101,376 B an SM has. Gate capability 12 onto
# top_k_per_row_decode, which is already the else branch at both call sites.
# Ours rather than MiaAI-Lab's, so their file stays verbatim upstream.
COPY patches/gb10_topk_fallback.py /tmp/gb10_topk_fallback.py
RUN python3 /tmp/gb10_topk_fallback.py && rm /tmp/gb10_topk_fallback.py

# nvcc's include path is missing the headers whose libraries this base already
# ships, which breaks any FlashInfer JIT path -- see the script.
COPY patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# MiaAI's patcher and LibertAI's plugin both solve the sm_121 MLA problem, and
# the first shuts the second out. Make the choice an env var -- see the script.
COPY patches/gb10_plugin_backend.py /tmp/gb10_plugin_backend.py
RUN python3 /tmp/gb10_plugin_backend.py && rm /tmp/gb10_plugin_backend.py

# Nothing in vLLM bounds a worker's memory, and on unified memory an unbounded
# worker takes the node down rather than its own request. Same contract as the
# script above: one anchor or the build fails.
COPY patches/spark_mem_trace.py /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
COPY patches/worker_memory_cap.py /tmp/worker_memory_cap.py
RUN python3 /tmp/worker_memory_cap.py && rm /tmp/worker_memory_cap.py

# The base is a single-node serving image, so it is missing two things this
# cluster needs. Both are cheap; neither is optional.
#
#   iproute2  : the entrypoint finds this node's ConnectX address by scanning
#               `ip -o -4 addr` for CLUSTER_SUBNET, and derives GLOO_SOCKET_IFNAME
#               from whichever interface owns it. Without `ip` the entrypoint
#               cannot start at all -- the pair is asymmetric (5818 carries the
#               link on enp1s0f1np1, 2353 on enp1s0f0np0), so the discovery is
#               not replaceable by a hardcoded name.
#   ibverbs   : NCCL's RoCE data plane. Missing, NCCL silently falls back to
#               TCP over the 1G LAN and every TP step crawls.
#
# ripgrep/pciutils are for the MCP inventory + search tools, matching ds4-flash.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 \
        libibverbs1 librdmacm1 ibverbs-providers \
        ripgrep pciutils usbutils \
    && rm -rf /var/lib/apt/lists/* \
    && command -v ip && command -v rg \
    && echo "runtime toolchain ok: ip + rg present"

# mentat replaces ray 2.57.0 (which is not a vLLM dependency -- this layer is
# the single thing standing between this image and multi-node TP). Real Ray on
# unified-memory GB10 caused two outages this swap exists to prevent: its
# memory monitor killed healthy ranks at 95% node memory (the ~89 GiB of
# weights ARE host memory), and its object store took up to ~36 GiB of the
# pool for a model that moves every tensor over NCCL. mentat is a placement-
# aware process launcher and nothing more, which is all vLLM's RayExecutorV2
# actually uses Ray for.
#
# The shim reports ray.__version__ == 2.57.0 because vLLM version-checks it;
# `ray.init` logs a mentat banner so nobody debugs believing it's real Ray.
# The shim covers ONLY the audited V2-executor surface and raises
# NotImplementedError on anything else -- on a vLLM base-image bump, re-audit
# with:  grep -rn 'ray\.' <site-packages>/vllm/v1/executor/
#
COPY --from=mentat /out/ /tmp/mentat-out/
RUN install -m755 /tmp/mentat-out/mentatd /usr/local/bin/mentatd \
    && ln -sf /usr/local/bin/mentatd /usr/local/bin/ray \
    && pip install --no-cache-dir --no-deps /tmp/mentat-out/mentatd-*-py3-none-any.whl \
    && rm -rf /tmp/mentat-out \
    && ray --version | grep -q "mentat" \
    && python3 -c "import ray, ray.cloudpickle; assert ray.__version__ == '2.57.0'"

# py-spy reads another process's Python stack without stopping it -- the only
# instrument that can see inside a scheduler tick that never returns. Needs
# CAP_SYS_PTRACE at run time, granted in the compose file.
RUN pip install --no-cache-dir py-spy && py-spy --version

# Assert the base is what this file assumes, so a moved tag fails the build
# rather than producing an image that boots and cannot load the model.
COPY patches/verify.py /tmp/verify.py
RUN python3 /tmp/verify.py


# See entrypoint.sh: the checkpoint's own template refuses images outright.
COPY chat-template.jinja /usr/local/share/glm53-chat-template.jinja
COPY status-server.py self-test.py /usr/local/bin/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /usr/local/bin/status-server.py /usr/local/bin/self-test.py /entrypoint.sh

# 8002 = OpenAI API (deliberately NOT ds4-flash's 8001, so both can be
#        configured on one node without a port clash even though only one can
#        actually be resident)
# 8082 = startup status page, bound from container start
EXPOSE 8002 8082
ENTRYPOINT ["/entrypoint.sh"]
