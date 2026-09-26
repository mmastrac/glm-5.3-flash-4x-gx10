"""Build vLLM's _flashkda_C from a FlashKDA checkout, outside vLLM's cmake.

Mirrors cmake/external_projects/flashkda.cmake for one arch (12.0f, CUDA 13).
Expects ./flashkda (FlashKDA with its cutlass submodule) and ./vllm-csrc
(vLLM's flashkda_registration.cpp and core/registration.h) beside it.
"""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ninja compiles from its own build directory, so includes must be absolute.
HERE = Path(__file__).resolve().parent
K = HERE / "flashkda"
setup(
    name="flashkda_c",
    ext_modules=[
        CUDAExtension(
            "_flashkda_C",
            sources=[
                "vllm-csrc/flashkda_registration.cpp",
                f"{K}/csrc/flash_kda.cpp",
                f"{K}/csrc/smxx/fwd_launch.cu",
            ],
            include_dirs=[
                f"{HERE}/vllm-csrc",
                f"{K}/csrc",
                f"{K}/cutlass/include",
                f"{K}/cutlass/examples/common",
                f"{K}/cutlass/tools/util/include",
            ],
            define_macros=[
                ("USE_CUDA", None),
                ("TORCH_TARGET_VERSION", "0x020B000000000000ULL"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_120f,code=sm_120f",
                    # torch adds these; vLLM's cmake removes them, and so does this.
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
