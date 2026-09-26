#!/usr/bin/env python3
"""Rebuild GLM-5.3-Flash TP=4 + DFlash2's KV specs on the nightly and drive
get_kv_cache_groups / get_kv_cache_configs. Mode 'before' expects the stock
tree's NotImplementedError; mode 'after' checks the patched tree end to end
(groups, configs, block sizes, coordinator, pickle, a small GPU allocation).
"""
import os
import pickle
import sys
from dataclasses import replace

import torch

MODE = sys.argv[1]
MODEL = "/models/glm-5.3-flash-nvfp4-nvidia"
DRAFT = "/models/glm-5.3-flash-dflash2"
KV_PIN = 27917287424
TP = 4

from vllm.engine.arg_utils import EngineArgs  # noqa: E402

args = EngineArgs(
    model=MODEL,
    tensor_parallel_size=TP,
    distributed_executor_backend="ray",
    max_model_len=524288,
    kv_cache_dtype="fp8_e4m3",
    block_size=2304,
    kv_cache_memory_bytes=KV_PIN,
    enable_prefix_caching=True,
    max_num_batched_tokens=16384,
    max_num_seqs=32,
    long_prefill_token_threshold=2304,
    trust_remote_code=True,
    gpu_memory_utilization=0.88,
    speculative_config={
        "method": "dflash",
        "model": DRAFT,
        "num_speculative_tokens": 7,
    },
)
vllm_config = args.create_engine_config()
cc = vllm_config.cache_config
mc = vllm_config.model_config
# The engine core resolves the layout before grouping (resolve_kv_cache_layout):
# candidates[0] of the workers' supported list. FlashInferMLASparseSM90Backend,
# DeepseekV32IndexerBackend (GLM indexer) and KpoolTailBackend each declare only
# LBHNC; TritonAttn (the drafter) declares no preference. So: LBHNC.
cc.kv_cache_layout = "LBHNC"
sc = vllm_config.speculative_config
print(f"== vllm {__import__('vllm').__version__}")
print(
    f"cache: block_size={cc.block_size} cache_dtype={cc.cache_dtype} "
    f"mamba_block_size={cc.mamba_block_size} mamba_page_size_padded={cc.mamba_page_size_padded} "
    f"mamba_cache_mode={cc.mamba_cache_mode} mamba_cache_dtype={cc.mamba_cache_dtype} "
    f"mamba_ssm_cache_dtype={cc.mamba_ssm_cache_dtype} use_kda_recoverssm={cc.use_kda_recoverssm} "
    f"kv_cache_memory_bytes={cc.kv_cache_memory_bytes} kv_cache_layout={cc.kv_cache_layout} "
    f"skip_page_size_padded={cc.skip_page_size_padded}"
)
print(
    f"spec: method={sc.method} k={sc.num_speculative_tokens} draft_arch={sc.draft_model_config.architectures} "
    f"use_eagle={sc.use_eagle()} block_drop={sc.use_eagle_block_drop()} "
    f"lookahead={vllm_config.num_prefill_lookahead_tokens} max_in_flight={vllm_config.max_in_flight_tokens} "
    f"max_model_len={mc.max_model_len}"
)

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.models.glm5next.common.attention import (  # noqa: E402
    Glm5NextIndexerCache,
    Glm5NextTailCache,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)

hf = mc.hf_text_config
mla_layers = list(hf.linear_attn_config["full_attn_layers"])
kda_layers = list(hf.linear_attn_config["kda_layers"])
assert hf.kv_lora_rank == 512 and hf.index_kpool == 4 and hf.index_head_dim == 128
P = "language_model.model.layers"

fp8 = kv_cache_dtype_str_to_dtype("fp8_e4m3", mc)
qm = get_kv_quant_mode("fp8_e4m3")
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention  # noqa: E402

kda_type = GatedDeltaNetAttention.mamba_type.fget(None)

specs = {}
with set_current_vllm_config(vllm_config):
    for i in range(hf.num_hidden_layers):
        if i in mla_layers:
            kc = Glm5NextIndexerCache(
                head_dim=hf.index_head_dim + hf.index_head_dim // 128 * 4,
                dtype=torch.uint8,
                prefix=f"{P}.{i}.self_attn.indexer.k_cache",
                cache_config=cc,
                index_kpool=hf.index_kpool,
            )
            specs[kc.prefix] = kc.get_kv_cache_spec(vllm_config)
            tc = Glm5NextTailCache(
                head_dim=hf.index_head_dim,
                dtype=torch.bfloat16,
                prefix=f"{P}.{i}.self_attn.indexer.tail_cache",
                cache_config=cc,
                index_kpool=hf.index_kpool,
            )
            specs[tc.prefix] = tc.get_kv_cache_spec(vllm_config)
            # mla_attention.MLAAttention.get_kv_cache_spec, head_size = kv_lora_rank + qk_rope(0)
            specs[f"{P}.{i}.self_attn.mla_attn"] = MLAAttentionSpec(
                block_size=cc.block_size,
                num_kv_heads=1,
                head_size=hf.kv_lora_rank + hf.qk_rope_head_dim,
                dtype=fp8,
                cache_dtype_str="fp8_e4m3",
                kv_quant_mode=qm,
                state_content_bytes=None,
                is_index_group_leader=True,
                non_causal_multi_token_decode=False,
            )
        else:
            assert i in kda_layers
            shapes = MambaStateShapeCalculator.kda_state_shape(
                TP,
                hf.linear_num_heads,
                hf.linear_head_dim,
                conv_kernel_size=hf.linear_conv_kernel_dim,
                num_spec=vllm_config.num_speculative_tokens,
            )
            dtypes = MambaStateDtypeCalculator.kda_state_dtype(
                mc.dtype, cc.mamba_cache_dtype, cc.mamba_ssm_cache_dtype
            )
            specs[f"{P}.{i}.self_attn"] = MambaSpec(
                shapes=tuple(shapes),
                dtypes=dtypes,
                block_size=cc.mamba_block_size,
                page_size_padded=cc.mamba_page_size_padded,
                mamba_type=kda_type,
                tp_replicated=False,
                mamba_cache_mode=cc.mamba_cache_mode,
                num_speculative_blocks=(
                    0 if cc.use_kda_recoverssm else vllm_config.num_speculative_tokens
                ),
            )

# The drafter: build a real Attention layer under the draft config so the
# backend selection and Attention.get_kv_cache_spec are the nightly's own.
dhf = sc.draft_model_config.hf_config
draft_cfg = replace(
    vllm_config,
    model_config=sc.draft_model_config,
    attention_config=replace(
        vllm_config.attention_config, backend=None, use_non_causal=True
    ),
)
draft_names = [f"model.layers.{i}.self_attn.attn" for i in range(dhf.num_hidden_layers)]
draft_backend = None
try:
    from vllm.model_executor.layers.attention.attention import Attention

    with set_current_vllm_config(draft_cfg):
        attn = Attention(
            dhf.num_attention_heads // TP,
            dhf.head_dim,
            dhf.head_dim**-0.5,
            num_kv_heads=max(1, dhf.num_key_value_heads // TP),
            cache_config=cc,
            quant_config=None,
            per_layer_sliding_window=dhf.sliding_window,
            prefix=draft_names[0],
        )
        draft_spec = attn.get_kv_cache_spec(draft_cfg)
        draft_backend = attn.get_attn_backend()
    print(
        f"draft backend: {draft_backend.get_name()} kernel blocks "
        f"{draft_backend.get_supported_kernel_block_sizes()}"
    )
except Exception as e:  # noqa: BLE001
    print(f"!! real drafter Attention construction failed: {type(e).__name__}: {e}")
    print("   falling back to a hand-built SlidingWindowSpec at the primary block")
    draft_spec = SlidingWindowSpec(
        block_size=cc.block_size,
        num_kv_heads=max(1, dhf.num_key_value_heads // TP),
        head_size=dhf.head_dim,
        dtype=fp8,
        kv_quant_mode=qm,
        sliding_window=dhf.sliding_window,
    )
for name in draft_names:
    specs[name] = draft_spec
print(f"draft spec: {draft_spec}")

print(f"== {len(specs)} layer specs")
seen = set()
for name, spec in specs.items():
    key = (type(spec).__name__, spec.block_size, spec.page_size_bytes)
    if key in seen:
        continue
    seen.add(key)
    print(f"  {name}: {type(spec).__name__} block={spec.block_size} page={spec.page_size_bytes}")

from vllm.v1.core import kv_cache_utils as U  # noqa: E402

specs_list = [dict(specs) for _ in range(TP)]
if MODE == "before":
    try:
        U.get_kv_cache_configs(vllm_config, specs_list, [KV_PIN] * TP)
    except NotImplementedError as e:
        print(f"== BEFORE: NotImplementedError as expected:\n   {e}")
        sys.exit(0)
    print("!! BEFORE: no error raised; the stock tree changed")
    sys.exit(1)

assert MODE == "after"
groups = U.get_kv_cache_groups(vllm_config, dict(specs))
print(f"== AFTER: {len(groups)} groups")
for gid, g in enumerate(groups):
    s = g.kv_cache_spec
    inner = (
        f"{len(s.kv_cache_specs)} inner specs "
        f"{sorted({type(x).__name__ for x in s.kv_cache_specs.values()})}"
        if isinstance(s, UniformTypeKVCacheSpecs)
        else ""
    )
    print(
        f"  [{gid}] {type(s).__name__} block={s.block_size} page={s.page_size_bytes} "
        f"cacheable={s.prefix_cacheable} eagle={g.is_eagle_group} layers={len(g.layer_names)} "
        f"{g.layer_names[0]}..{g.layer_names[-1]} {inner}"
    )

configs = U.get_kv_cache_configs(vllm_config, specs_list, [KV_PIN] * TP)
cfg = configs[0]
assert all(c.num_blocks == cfg.num_blocks for c in configs)
bpb = U._pool_bytes_per_block(cfg.kv_cache_groups)
print(f"== config: num_blocks={cfg.num_blocks} bytes/block={bpb} ({bpb/2**20:.2f} MiB) "
      f"pool={cfg.num_blocks*bpb/2**30:.2f} GiB of {KV_PIN/2**30:.2f} GiB pinned")
layout = U._glm5_next_tensor_layout(cfg.kv_cache_groups)
assert layout is not None
attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page, tail_names, tail_page, draft_groups = layout
print(f"   layout: {len(mla_names)} MLA x {mla_page} + {len(idx_names)} idx x {idx_page}; tail {len(tail_names)} x {tail_page}; "
      f"{len(mamba_groups)} mamba groups; {len(draft_groups)} draft groups "
      f"({sum(len(g.layer_names) for g in draft_groups)} layers x {draft_groups[0].kv_cache_spec.page_size_bytes} B)")
size = cfg.kv_cache_tensors[0].size
assert size == bpb * cfg.num_blocks
mla_end = len(mla_names) * mla_page * cfg.num_blocks
idx_end = mla_end + len(idx_names) * idx_page * cfg.num_blocks
regions = []
for t in cfg.kv_cache_tensors:
    assert t.size == size
    if t.layers[0] in draft_names:
        span = t.layer_stride * len(t.layers)
        assert t.offset >= idx_end and t.offset + span <= size, (t, idx_end, size)
        regions.append((t.offset, t.offset + span, t.layers[0], t.block_stride))
regions.sort()
assert len(regions) == len(draft_names)
for a, b in zip(regions, regions[1:]):
    assert a[1] <= b[0], "draft regions overlap"
assert regions[-1][1] == size and regions[0][0] == idx_end, "draft regions do not tile the tail of the pool"
print(f"   draft regions: {len(regions)} x {regions[0][1]-regions[0][0]} B, block_stride={regions[0][3]}, "
      f"from {idx_end} to {size} (pool end)")

sched_bs, hash_bs = U.resolve_kv_cache_block_sizes(cfg, vllm_config)
participating = [g.kv_cache_spec.block_size for g in cfg.kv_cache_groups if g.kv_cache_spec.prefix_cacheable]
print(f"== block sizes: scheduler={sched_bs} hash={hash_bs} engine-core min(participating)={min(participating)} "
      f"participating={participating}")
assert sched_bs == cc.block_size and min(participating) == cc.block_size

sched_cfg = U.generate_scheduler_kv_cache_config(configs)
tokens, conc = U.get_kv_cache_capacity(vllm_config, sched_cfg)
print(f"== capacity: {tokens:,} tokens, {conc:.2f}x at {mc.max_model_len}")
for gid, g in enumerate(sched_cfg.kv_cache_groups):
    s = g.kv_cache_spec
    need = U.cdiv(s.max_memory_usage_bytes(vllm_config), s.page_size_bytes)
    print(f"   group {gid} {type(s).__name__}: {need} blocks per max-len request")

from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator  # noqa: E402

coord = get_kv_cache_coordinator(
    kv_cache_config=sched_cfg,
    max_model_len=mc.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    use_eagle=sc.use_eagle_block_drop(),
    enable_caching=True,
    enable_kv_cache_events=False,
    dcp_world_size=1,
    pcp_world_size=1,
    scheduler_block_size=sched_bs,
    hash_block_size=hash_bs,
    num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
)
print(f"== coordinator: {type(coord).__name__}; managers "
      f"{[type(m).__name__ for m in coord.single_type_managers]}")
print(f"   hit-lookup groups: {[(type(g.spec).__name__, g.group_ids, g.use_eagle) for g in coord.attention_groups]}")
print(f"   eagle_group_ids={sorted(coord.eagle_group_ids)} partial_hash_hits={coord.enable_partial_hash_hits}")

rt = pickle.loads(pickle.dumps(cfg))
assert rt.kv_cache_groups[-1].kv_cache_spec == cfg.kv_cache_groups[-1].kv_cache_spec
assert type(rt.kv_cache_groups[-1].kv_cache_spec).__module__ == "vllm.v1.core.kv_cache_utils"
assert not rt.kv_cache_groups[-1].kv_cache_spec.prefix_cacheable
print("== pickle round trip ok (draft spec by module path, still opted out)")

# PP projection: a stage that holds no drafter layer must still take the GLM
# layout (empty draft group), count no draft bytes and emit no draft tensor.
stage_specs = {k: v for k, v in specs.items() if k not in draft_names}
proj = U._project_kv_cache_groups_to_worker(groups, stage_specs)
assert proj[-1].layer_names == [] and isinstance(proj[-1].kv_cache_spec, U.DraftSlidingWindowSpec)
pl = U._glm5_next_tensor_layout(proj)
assert pl is not None and pl[-1][0].layer_names == []
stage_cfg = U.get_kv_cache_config_from_groups(vllm_config, proj, KV_PIN)
stage_bpb = U._pool_bytes_per_block(proj)
assert stage_bpb == len(mla_names) * mla_page + len(idx_names) * idx_page
assert not any(t.layers[0] in draft_names for t in stage_cfg.kv_cache_tensors)
assert U._max_memory_usage_bytes_from_groups(vllm_config, proj) < U._max_memory_usage_bytes_from_groups(vllm_config, groups)
print(f"== PP projection without the drafter: {len(proj)} groups (last empty), bytes/block={stage_bpb}, "
      f"num_blocks={stage_cfg.num_blocks}, no draft tensors")

# A small GPU allocation with the real view builder and a kernel split for the drafter.
if torch.cuda.is_available():
    from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

    small = U.get_kv_cache_config_from_groups(vllm_config, cfg.kv_cache_groups, 2 * 2**30)
    kernel = []
    for g in small.kv_cache_groups:
        s = g.kv_cache_spec
        if g.layer_names and g.layer_names[0] in draft_names and draft_backend is not None:
            kernel.append(select_common_block_size(s.block_size, [draft_backend]))
        else:
            kernel.append(s.block_size)
    lay = cc.get_resolved_kv_cache_layout()
    layout_name = lay.name
    caches = allocate_kv_cache(small, torch.device("cuda"), lay, kernel)
    dv = caches[draft_names[0]]
    print(f"== GPU alloc ({small.num_blocks} blocks, layout {layout_name}, kernel blocks {kernel}): "
          f"draft view shape={tuple(dv.shape)} dtype={dv.dtype} strides={dv.stride()}")
    # Draft regions must not alias anything else: poison one and check the rest.
    dv.view(torch.uint8).fill_(0xAB)
    for name, view in caches.items():
        if name in draft_names:
            continue
        v = view.view(torch.uint8) if view.dtype != torch.uint8 else view
        assert int(v.max()) == 0, f"{name} aliases the draft region"
    other = caches[draft_names[1]].view(torch.uint8)
    assert int(other.max()) == 0, "draft layers alias each other"
    print("   draft region is unshared (poison did not leak into any other view)")
    del caches
print("== ALL CHECKS PASSED")
