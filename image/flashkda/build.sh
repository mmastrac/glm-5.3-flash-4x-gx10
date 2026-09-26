#!/bin/bash
# Fetch FlashKDA (with cutlass) and vLLM's registration shim, then build
# _flashkda_C.so into /out. The base image has curl but no git.
set -euo pipefail
: "${FLASHKDA_REF:?}" "${CUTLASS_REF:?}" "${VLLM_REF:?}"
cd /build
mkdir -p flashkda/cutlass vllm-csrc/core /out
curl -fsSL "https://github.com/vllm-project/FlashKDA/archive/${FLASHKDA_REF}.tar.gz" \
    | tar xz --strip-components=1 -C flashkda
curl -fsSL "https://github.com/NVIDIA/cutlass/archive/${CUTLASS_REF}.tar.gz" \
    | tar xz --strip-components=1 -C flashkda/cutlass
raw="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_REF}/csrc"
curl -fsSL -o vllm-csrc/flashkda_registration.cpp "$raw/flashkda_registration.cpp"
curl -fsSL -o vllm-csrc/core/registration.h "$raw/core/registration.h"
# setup.py passes -gencode itself, which stops torch guessing arches with no GPU.
MAX_JOBS="$(nproc)" python3 setup.py build_ext --inplace
cp _flashkda_C*.so /out/_flashkda_C.abi3.so
echo "$FLASHKDA_REF" > /out/FLASHKDA_REF
