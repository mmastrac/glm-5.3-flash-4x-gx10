import sys

src = open("main_stock.py").read()
block = open("draft_block.py").read()
edits = []

def sub(old, new, count=1, label=""):
    global src
    n = src.count(old)
    if n != count:
        print(f"FAIL [{label}]: anchor found {n}x, expected {count}")
        sys.exit(1)
    src = src.replace(old, new)
    edits.append(label)

# 1. carve the drafter's plain attention layers out of the target's spec set
sub("""    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None
""",
"""    # DFLASH2-PATCH: a speculative drafter (EAGLE3 / DFlash) adds ordinary
    # attention layers (FullAttentionSpec / SlidingWindowSpec) to the target's
    # MLA(+kpool indexer)+mamba spec set. Left in, they knock this grouper out
    # and the whole model goes down the generic page-size unifier, where the
    # kpool indexer page (storage_block * 132 B) never divides the MLA page
    # (block * 512 B) for any block size -- and which loses the slot-shared
    # tensor layout below regardless. Carve them out into their own group(s)
    # with per-layer tensors; nothing about the target layout changes.
    draft_specs = {
        name: spec
        for name, spec in attn_specs.items()
        if not isinstance(spec, MLAAttentionSpec)
    }
    attn_specs = {
        name: spec for name, spec in attn_specs.items() if name not in draft_specs
    }
    if (
        not mamba_specs
        or not attn_specs
        or not all(type(spec) is MLAAttentionSpec for spec in attn_specs.values())
    ):
        return None
    if not all(isinstance(spec, AttentionSpec) for spec in draft_specs.values()):
        return None
""", label="1 draft carve-out")

# 2. build the draft groups and append them last, so existing group ids hold
sub("""    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
""",
"""    # DFLASH2-PATCH: draft groups go last so every existing group id is
    # unchanged.
    draft_groups = _glm5_next_draft_groups(
        draft_specs, next(iter(mla_specs.values())).block_size
    )
    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + draft_groups
    )
""", label="2 draft groups appended")

# 3. the spec class, its manager, the block-size rule and the byte helper
sub("def _glm5_next_tensor_layout(", block + "\ndef _glm5_next_tensor_layout(",
    label="3 draft spec + helpers inserted")

# 4. the layout tuple gains a ninth element
sub("""        list[str],
        int,
    ]
    | None
):
    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.\"\"\"""",
"""        list[str],
        int,
        list[KVCacheGroupSpec],
    ]
    | None
):
    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.

    The ninth element is the drafter's groups, empty when there is no drafter.
    \"\"\"""", label="4 layout return type")

# 5. classify the drafter's groups, and count them in the exhaustiveness check
sub("""    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None""",
"""    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    # DFLASH2-PATCH: drafter groups are plain AttentionSpec groups; the target's
    # attention and tail groups are both UniformTypeKVCacheSpecs, so none of the
    # target's is misclassified here. A PP stage without the drafter sees the
    # group with an empty layer list, and it still counts.
    draft_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, AttentionSpec)
        and not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    attn_group: KVCacheGroupSpec | None = None""", label="5 draft group detection")

sub("""    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None""",
"""    if len(uniform_groups) + len(mamba_groups) + len(draft_groups) != len(
        kv_cache_groups
    ):
        return None""", label="5b exhaustiveness check")

# 6. return them
sub("""    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
    )""",
"""    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
        draft_groups,
    )""", label="6 layout returns draft groups")

# 7+9. both unpack sites
sub("""            tail_names,
            _,
        ) = glm5_layout""",
"""            tail_names,
            _,
            draft_groups,
        ) = glm5_layout""", count=2, label="7+9 both unpack sites")

# 8. a pool block carries the draft pages too
sub("""        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page""",
"""        bytes_per_block = (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            # DFLASH2-PATCH: every pool block also carries the draft pages.
            + _glm5_next_draft_bytes_per_block(draft_groups)
        )""", label="8 bytes per block")

# 8b. one unshared region per draft layer, after the indexer regions
sub("""                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(""",
"""                add_tensor(tail_name, tail_specs[tail_name], offset)

        # DFLASH2-PATCH: one unshared region per draft layer, laid after the
        # indexer regions. Nothing shares these: a draft layer's contents are
        # per request and never prefix-cached.
        draft_offset = idx_base + len(idx_names) * idx_page * num_blocks
        for group in draft_groups:
            draft_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                add_tensor(layer_name, draft_spec, draft_offset)
                draft_offset += draft_spec.page_size_bytes * num_blocks

        return KVCacheConfig(""", label="8b draft tensors emitted")

# 10. the drafter's window-bounded blocks come from the same pool
sub("""        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)""",
"""        if tail_names:
            total_blocks += 1
        # DFLASH2-PATCH: draft groups draw their window-bounded blocks from the
        # same pool, and every pool block also carries the draft pages.
        for group in draft_groups:
            if not group.layer_names:
                continue
            draft_spec = group.kv_cache_spec
            total_blocks += cdiv(
                draft_spec.max_memory_usage_bytes(vllm_config),
                draft_spec.page_size_bytes,
            )
        return total_blocks * (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + _glm5_next_draft_bytes_per_block(draft_groups)
        )""", label="10 block estimate")

open("ported.py", "w").write(src)
print("applied:", len(edits), "sites")
for e in edits:
    print("  +", e)
