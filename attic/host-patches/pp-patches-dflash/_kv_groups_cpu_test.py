"""CPU-only reproduction of the KV cache grouping for GLM-5.3-Flash + DFlash2
drafter, PP=1 and PP=2 (23/22 split, drafter on the last stage). Run against
the stock kv_cache_utils.py it must raise the observed NotImplementedError;
against the patched one it must produce the GLM-5-Next slot-shared layout plus
one per-layer-tensor draft group.
"""
import sys, traceback
from types import SimpleNamespace
import torch
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec, KpoolTailSpec, MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs,
    KVQuantMode, AttentionSpec)
from vllm.v1.core import kv_cache_utils as kcu

PATCHED = hasattr(kcu, "DraftSlidingWindowSpec")
print("kv_cache_utils patched:", PATCHED)

BLOCK = 2304
MLA_LAYERS = list(range(3, 45, 4))                 # 11 deepseek_sparse_attention layers
KDA_LAYERS = [i for i in range(45) if i not in MLA_LAYERS]   # 34 linear_attention layers
fp8 = torch.float8_e4m3fn

def target_specs(layers):
    spec = {}
    for i in layers:
        p = f"language_model.model.layers.{i}.self_attn"
        if i in MLA_LAYERS:
            spec[f"{p}.attn"] = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=512, dtype=fp8,
                                                 cache_dtype_str="fp8_e4m3", kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
            spec[f"{p}.indexer.k_cache"] = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=132,
                                                            dtype=torch.uint8, cache_dtype_str="auto", compress_ratio=4)
            spec[f"{p}.indexer.tail_cache"] = KpoolTailSpec(block_size=4, num_kv_heads=1, head_size=256, head_size_v=0,
                                                            dtype=torch.bfloat16, sliding_window=4)
        else:
            spec[p] = MambaSpec(block_size=BLOCK, shapes=((32, 2, 128), (32, 128, 128)), dtypes=(torch.bfloat16, torch.bfloat16),
                                mamba_cache_mode="align")
    return spec

def draft_specs():
    # what Attention.get_kv_cache_spec emits for the DFlash2 layers under --kv-cache-dtype fp8_e4m3, TP=2:
    # smallest kernel block (16), 8/2 kv heads, head 128, window 2048
    return {f"model.layers.{22 + j}.self_attn.attn": SlidingWindowSpec(block_size=16, num_kv_heads=4, head_size=128,
                dtype=fp8, kv_quant_mode=KVQuantMode.FP8_PER_TENSOR, sliding_window=2048) for j in range(5)}

def fake_config(pp):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False, max_num_encoder_input_tokens=0),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp, decode_context_parallel_size=1, tensor_parallel_size=2),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 45, max_model_len=262144, original_max_model_len=262144),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, mamba_cache_mode="align", block_size=BLOCK),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        kv_transfer_config=None,
        max_in_flight_tokens=8192,
    )

for pp in (1, 2):
    print(f"\n===== PP={pp}")
    if pp == 1:
        workers = [{**target_specs(range(45)), **draft_specs()}]
    else:
        workers = [target_specs(range(0, 23)), {**target_specs(range(23, 45)), **draft_specs()}]
    merged = {}
    for w in workers: merged.update(w)
    pages = sorted({(type(s).__name__, s.page_size_bytes) for s in merged.values()})
    print("page sizes:", pages)
    cfg = fake_config(pp)
    try:
        configs = kcu.get_kv_cache_configs(cfg, workers, [10_594_000_000] * len(workers))
    except NotImplementedError as e:
        print("NotImplementedError:", str(e)[:160]); continue
    groups = kcu.get_kv_cache_groups(cfg, dict(merged))
    for i, g in enumerate(groups):
        spec = g.kv_cache_spec
        kind = type(spec).__name__
        if isinstance(spec, UniformTypeKVCacheSpecs):
            kind += "[" + type(next(iter(spec.kv_cache_specs.values()))).__name__ + "]"
        print(f"  group {i}: {kind:40s} layers={len(g.layer_names):3d} block={spec.block_size:5d} page={spec.page_size_bytes}")
    for wi, c in enumerate(configs):
        per_block = kcu._pool_bytes_per_block(cfg, c.kv_cache_groups)
        draft_tensors = [t for t in c.kv_cache_tensors if any("model.layers.2" in n and not n.startswith("language") for n in t.shared_by)]
        print(f"  worker {wi}: num_blocks={c.num_blocks} bytes/block={per_block/2**20:.2f} MiB tensors={len(c.kv_cache_tensors)} "
              f"draft_tensors={len(draft_tensors)} total={sum(t.size for t in c.kv_cache_tensors)/2**30:.2f} GiB "
              f"MLA-token-capacity={c.num_blocks*BLOCK}")
        if draft_tensors:
            print("   draft tensor:", draft_tensors[0].shared_by, draft_tensors[0].size/2**20, "MiB")
    print("  max concurrency (262144-token requests):", round(kcu.get_max_concurrency_for_kv_cache_config(cfg, configs[-1]), 2))
    if not PATCHED:
        continue
    # ---- v2 checks: the draft group must not move the scheduler / hash / engine block sizes
    import inspect, math, pickle
    fn = next(f for n, f in vars(kcu).items() if callable(f) and getattr(f, "__module__", "") == kcu.__name__
              and "math.lcm(*group_block_sizes)" in (inspect.getsource(f) if inspect.isfunction(f) else ""))
    last = configs[-1]
    sig = inspect.signature(fn); print("  block-size fn:", fn.__name__, list(sig.parameters))
    participating = [g.kv_cache_spec.block_size for g in last.kv_cache_groups if g.kv_cache_spec.participates_in_prefix_caching]
    engine_block = min(participating)
    draft_g = [g for g in last.kv_cache_groups if type(g.kv_cache_spec).__name__ == "DraftSlidingWindowSpec"]
    print("  engine cache_config.block_size (min over participating):", engine_block, " draft groups:", [(g.kv_cache_spec.block_size, len(g.layer_names)) for g in draft_g])
    assert engine_block == BLOCK and draft_g and draft_g[0].kv_cache_spec.block_size == 576
    assert not draft_g[0].kv_cache_spec.participates_in_prefix_caching
    # pickle round trip (worker side unpickles the config)
    rt = pickle.loads(pickle.dumps(last)); print("  pickle round-trip ok:", type(rt.kv_cache_groups[-1].kv_cache_spec).__name__)
    # manager registration + no-op hooks
    from vllm.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
    from vllm.v1.core.block_pool import BlockPool
    pool = BlockPool(num_gpu_blocks=last.num_blocks, enable_caching=True, hash_block_size=BLOCK, enable_kv_cache_events=False)
    mgr = get_manager_for_kv_cache_spec(draft_g[0].kv_cache_spec, max_in_flight_tokens=8192, max_model_len=262144,
                                        block_pool=pool, enable_caching=True, kv_cache_group_id=6, scheduler_block_size=BLOCK)
    print("  draft manager:", type(mgr).__name__, "bases:", [b.__name__ for b in type(mgr).__mro__[1:3]],
          "admission cap blocks/req:", mgr._max_admission_blocks_per_request, "skipped@10000:", mgr.get_num_skipped_tokens(10000))
    assert mgr.get_num_common_prefix_blocks("x") == 0
    assert mgr.find_longest_cache_hit([], 100, [6], pool, draft_g[0].kv_cache_spec, False, BLOCK) == (([],), 0)
    # hybrid coordinator asserts (the scheduler builds exactly this)
    cfg.cache_config.prefix_match_unit = None; cfg.cache_config.enable_prefix_caching = True; cfg.kv_transfer_config = None
    sched_bs, hash_bs = fn(last, cfg)   # resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    print("  scheduler_block_size, hash_block_size:", sched_bs, hash_bs)
    assert (sched_bs, hash_bs) == (BLOCK, BLOCK)
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    sched_cfg = kcu.generate_scheduler_kv_cache_config(configs)   # what EngineCore hands the scheduler
    coord = HybridKVCacheCoordinator(sched_cfg, 262144, 8192, True, True, False, 1, 1, sched_bs, hash_bs)
    print("  coordinator attention_groups (participating):", [type(g.spec).__name__ for g in coord.attention_groups],
          "eagle ids:", sorted(coord.eagle_group_ids))
    print("  managers:", [type(m).__name__ for m in coord.single_type_managers])
    # per-request pool-block budget the capacity metric divides by (max_model_len request)
    per_req = [(type(g.kv_cache_spec).__name__, -(-g.kv_cache_spec.max_memory_usage_bytes(cfg) // g.kv_cache_spec.page_size_bytes)) for g in last.kv_cache_groups]
    print("  blocks per 262k request by group:", per_req, "sum", sum(b for _, b in per_req))
print("DONE")
