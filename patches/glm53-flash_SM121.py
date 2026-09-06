#!/usr/bin/env python3
"""SM121 kernel patches for GLM-5.3-Flash-NVFP4 on GB10.

This Spark recipe runs this once while building glm53-flash-sm121:v8
(GB10 / SM121). The serving tag on top is MM + Ray TP2 with the OpenAI API
on :8888. Stock glm53-flash-arm64-cu130 only lists SM120 packed MLA on
capability 12; our checkpoint is NoPE (pe_dim=0), so this image selects
SM90 sparse-MLA + FA2 instead. We did not author FA2 MLA or FlashInfer
0.6.18 — we only gate what is already in the tree onto GB10.

Modes:

  sm90   default. Seven steps the Dockerfile runs with no args.
  sm120  unused packed-cache path (fp8_ds_mla pad / topk 2048 / skip
         warmup). Kept so it is not a second file. Do not bake.

  python3 glm53-flash_SM121.py
  python3 glm53-flash_SM121.py sm90
  python3 glm53-flash_SM121.py sm120
  python3 glm53-flash_SM121.py --legacy-sm120

Unknown args are refused. Each replacement must match stock vLLM / FI 0.6.18
exactly once or we abort. Already-applied trees are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
FI = Path("/usr/local/lib/python3.12/dist-packages/flashinfer")
LOG = "[mia-sm121]"

USAGE = (
    "usage: python3 glm53-flash_SM121.py [sm90|sm120|--legacy-sm120]\n"
    "  sm90 (default): seven GB10 steps baked by Dockerfile\n"
    "  sm120 / --legacy-sm120: unused packed-cache path (do not bake)"
)


# ---------------------------------------------------------------------------
# CLI (parse first so the bake helpers can sit in FI-then-vLLM source order)
# ---------------------------------------------------------------------------


def parse_mode(argv: list[str]) -> str:
    if not argv:
        return "sm90"
    if len(argv) != 1:
        raise SystemExit("refuse extra args\n" + USAGE)
    arg = argv[0]
    if arg == "sm90":
        return "sm90"
    if arg in ("sm120", "--legacy-sm120"):
        return "sm120"
    raise SystemExit("unknown mode %r\n%s" % (arg, USAGE))


def apply_once(path: Path, old: str, new: str, label: str) -> str:
    text = path.read_text()
    n_old = text.count(old)
    n_new = text.count(new)
    if n_old == 0 and n_new == 1:
        return "skipped"
    if n_old != 1:
        raise SystemExit(
            "%s refuse %s (old=%d new=%d); stock tree changed"
            % (LOG, label, n_old, n_new)
        )
    path.write_text(text.replace(old, new, 1))
    return "applied"


def announce(title: str, results: list[str]) -> None:
    if all(r == "skipped" for r in results):
        print("%s skip %s (already in this image)" % (LOG, title))
        return
    print("%s %s (%s)" % (LOG, title, " ".join(results)))


# ---------------------------------------------------------------------------
# FlashInfer 0.6.18 — files exist in this image after the Dockerfile pip.
# Defined first (unlike a v1-then-v8 changelog). apply_sm90 still runs them
# after the vLLM tree so a rebuild matches the same end state.
# ---------------------------------------------------------------------------


def bake_fa2_fp8_cta_tile() -> None:
    old = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;\n"
    new = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u) : CTA_TILE_KV;\n"
    r = apply_once(
        FI / "data/include/flashinfer/attention/mla.cuh",
        old,
        new,
        "mla.cuh fp8 tile",
    )
    announce("FA2 fp8 CTA tile cap for GB10 ~101KB smem", [r])


def bake_fa2_fp8_sm12_gate() -> None:
    old = "            major, minor = get_compute_capability(self.device)\n            if major != 9:\n"
    new = "            major, minor = get_compute_capability(self.device)\n            if major not in (9, 12):\n"
    r = apply_once(FI / "mla/_core.py", old, new, "_core.py fp8 SM12 gate")
    announce("FA2 fp8 gate major in (9, 12)", [r])


# ---------------------------------------------------------------------------
# vLLM tree — SM90 NoPE + FA2 on capability 12, PDL off, indexer/kpool
# ---------------------------------------------------------------------------


def bake_sm90_on_capability_12() -> None:
    old = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
    new = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
    r = apply_once(VLLM / "platforms/cuda.py", old, new, "cuda.py SM90 list")
    announce("cuda.py lists SM90 on capability 12", [r])


def bake_sm90_wrapper_for_gb10() -> None:
    path = VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
    cap_old = (
        "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
        "        return capability.major == 9\n"
    )
    cap_new = (
        "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
        "        return capability.major in (9, 12)\n"
    )
    fa_old = '            backend="fa3",\n'
    fa_new = (
        '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n'
    )
    gate_old = """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
    gate_new = """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
    results = [
        apply_once(path, cap_old, cap_new, "sm90 capability gate"),
        apply_once(path, fa_old, fa_new, "sm90 FA2 wrapper"),
        apply_once(path, gate_old, gate_new, "sm90 fp8 version gate"),
    ]
    announce("sm90 cap (9,12) + FA2 off-Hopper + fp8 gate", results)


def bake_pdl_off_sm12() -> None:
    old = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
"""
    new = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # PDL lowering is unvalidated on SM12x (GB10) and races on KDA
        # state kernels there; keep it to Hopper/Blackwell-datacenter.
        return major in (9, 10)
"""
    r = apply_once(VLLM / "platforms/cuda.py", old, new, "cuda.py PDL")
    announce("PDL gated off on SM12x (KDA race on this kit)", [r])


def bake_indexer_topk_neg1() -> None:
    path = VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py"
    prefill_old = (
        "                pool_topk = torch.empty(\n"
        "                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
        "                )\n"
    )
    prefill_new = (
        "                pool_topk = torch.full(\n"
        "                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
        "                )\n"
    )
    decode_old = (
        "            pool_topk = torch.empty(\n"
        "                (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
        "            )\n"
    )
    decode_new = (
        "            pool_topk = torch.full(\n"
        "                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
        "            )\n"
    )
    results = [
        apply_once(path, prefill_old, prefill_new, "indexer prefill -1"),
        apply_once(path, decode_old, decode_new, "indexer decode -1"),
    ]
    announce("indexer top-k -1 init (prefill + decode)", results)


def bake_kpool_pid_clamp() -> None:
    old = "    hist_out = tl.where(pid >= 0, hist_val, -1)\n"
    new = "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n"
    r = apply_once(
        VLLM / "models/glm5next/nvidia/ops/kpool_compress.py",
        old,
        new,
        "kpool pid clamp",
    )
    announce("kpool pid clamp to pool_len", [r])


def apply_sm90() -> None:
    bake_sm90_on_capability_12()
    bake_sm90_wrapper_for_gb10()
    bake_pdl_off_sm12()
    bake_indexer_topk_neg1()
    bake_kpool_pid_clamp()
    bake_fa2_fp8_cta_tile()
    bake_fa2_fp8_sm12_gate()


# ---------------------------------------------------------------------------
# sm120 — unused packed-cache path (python3 glm53-flash_SM121.py sm120)
# Marker-based _once, not apply_once. Not invoked by the Dockerfile.
# ---------------------------------------------------------------------------

MARKER = "GLM53_NOPE_FP8_DS_MLA_PAD"
TOPK_MARKER = "GLM53_SM120_TOPK2048"
ROOT: Path | None = None


def _once(path: Path, needle: str, replacement: str, marker: str) -> str:
    text = path.read_text()
    if marker in text:
        return "already"
    if needle not in text:
        raise SystemExit(f"{path}: expected snippet not found")
    path.write_text(text.replace(needle, replacement, 1))
    return "patched"


def packed_cache_write() -> str:
    assert ROOT is not None
    path = ROOT / "v1/attention/backend.py"
    needle = """        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
"""
    replacement = f"""        from vllm import _custom_ops as ops

        # {MARKER}: fp8_ds_mla pages always store 64-dim RoPE. NoPE models
        # (qk_rope_head_dim=0) pad zeros so concat_and_cache_mla accepts them.
        pe = k_pe.squeeze(1)
        if kv_cache_dtype == "fp8_ds_mla" and pe.shape[-1] == 0:
            pe = pe.new_zeros(*pe.shape[:-1], 64)
        ops.concat_and_cache_mla(
            kv_c_normed,
            pe,
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
"""
    return _once(path, needle, replacement, MARKER)


def packed_sm120_decode() -> str:
    assert ROOT is not None
    path = ROOT / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    needle = """        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
"""
    replacement = f"""        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        # {MARKER}: match the 656 B fp8_ds_mla page (ckv + 64-dim kpe).
        rope_dim = self.qk_rope_head_dim
        if rope_dim == 0:
            q = torch.nn.functional.pad(q, (0, 64))
            rope_dim = 64

        num_actual_toks = q.shape[0]
"""
    status = _once(path, needle, replacement, MARKER)
    if status == "already":
        return status
    needle2 = """            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
"""
    replacement2 = """            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=rope_dim,
"""
    text = path.read_text()
    if needle2 not in text:
        raise SystemExit(f"{path}: flashinfer rope-dim call site not found")
    path.write_text(text.replace(needle2, replacement2, 1))
    return "patched"


def packed_sm120_topk() -> str:
    """Unused packed path: SM120 helper still wants topk=2048."""
    assert ROOT is not None
    path = ROOT / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
    needle = """        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
"""
    replacement = f"""        # {TOPK_MARKER}: FlashInfer SM120 v32/GLM wants block_tables (N, 1, 2048).
        if topk_indices_physical.shape[-1] != 2048:
            if topk_indices_physical.shape[-1] > 2048:
                topk_indices_physical = topk_indices_physical[..., :2048].contiguous()
            else:
                pad = 2048 - topk_indices_physical.shape[-1]
                topk_indices_physical = torch.nn.functional.pad(
                    topk_indices_physical, (0, pad), value=-1
                )
        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
"""
    status = _once(path, needle, replacement, TOPK_MARKER)
    if status == "already":
        return status
    text = path.read_text()
    text = text.replace(
        "max_seq_len=attn_metadata.topk_tokens,",
        "max_seq_len=2048,",
        1,
    )
    text = text.replace(
        "sparse_mla_top_k=attn_metadata.topk_tokens,",
        "sparse_mla_top_k=2048,",
        1,
    )
    path.write_text(text)
    return "patched"


def packed_skip_fi_warmup() -> str:
    """Unused packed path: fused_moe gemm1 autotune OOMs rank 0 on GB10."""
    assert ROOT is not None
    path = ROOT / "model_executor/warmup/kernel_warmup.py"
    needle = """    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)
"""
    marker = "GLM53_SKIP_FI_SPARSE_WARMUP"
    replacement = f"""    # {marker}: skip FlashInfer SM120 sparse-MLA autotune (kills rank 0 on GB10).
    deepseek_v4_sparse_mla_attention_warmup(worker)
"""
    return _once(path, needle, replacement, marker)


def packed_skip_fi_autotune() -> str:
    assert ROOT is not None
    path = ROOT / "model_executor/warmup/kernel_warmup.py"
    marker = "GLM53_SKIP_FI_AUTOTUNE"
    needle = '''    from flashinfer.autotuner import AutoTuner, set_autotune_process_group
'''
    replacement = f'''    # {marker}: fused_moe gemm1/gemm2 autotune kills rank 0 on GB10.
    logger.info_once("Skipping FlashInfer autotune on SM121")
    return
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group
'''
    return _once(path, needle, replacement, marker)


def apply_sm120() -> None:
    global ROOT
    import vllm

    ROOT = Path(vllm.__file__).resolve().parent
    cache = packed_cache_write()
    decode = packed_sm120_decode()
    topk = packed_sm120_topk()
    warmup = packed_skip_fi_warmup()
    autotune = packed_skip_fi_autotune()
    print(
        "%s packed-path (not baked) cache_write=%s sm120_decode=%s "
        "sm120_topk=%s skip_fi_warmup=%s skip_fi_autotune=%s"
        % (LOG, cache, decode, topk, warmup, autotune)
    )


def main(argv: list[str] | None = None) -> None:
    mode = parse_mode(sys.argv[1:] if argv is None else argv)
    if mode == "sm90":
        apply_sm90()
        return
    apply_sm120()


if __name__ == "__main__":
    main()
