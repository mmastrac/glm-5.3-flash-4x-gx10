#!/usr/bin/env python3
"""Let the GLM-5-Next KV grouper carry a DFlash2 drafter's attention layers.

With `--speculative-config {"method":"dflash",...}` the drafter adds five
SlidingWindowSpec layers (Qwen3, 2 kv heads at TP=4, window 2048) to the
target's spec set. `_get_kv_cache_groups_glm5_next` returns None when any
non-mamba, non-tail spec is not exactly MLAAttentionSpec, so the model falls
off its own layout onto the generic path. The page unifier there takes the
largest page (the MLA latent, 512 B/token) and needs every smaller page to
divide it. The kpool indexer page is 33 B/token (132 B per 4-token pool), and
512/33 is not an integer for any block size. MLA pages cannot be padded, so
`unify_kv_cache_spec_page_size` raises

    Layer language_model.model.layers.3.self_attn.indexer.k_cache: page size
    is not divisible by the maximum page size and cannot be padded.

The full-allocation fallback fails too (draft and target block sizes differ),
and the engine dies after a full weight load. No flag reaches the indexer (it
is not an Attention layer).

The fix: carve the drafter's specs out before the MLA check and append them as
their own group(s) after every existing group, so the target's groups, ids,
tensors and slot sharing are unchanged. `_glm5_next_tensor_layout` learns to
recognise those groups, and the GLM branches of the config builder, the
bytes-per-block helper and the memory estimate add one unshared region per
draft layer after the indexer regions. This is the pre-merge image's fork,
reduced to what the nightly lacks.

The draft group is a `DraftSlidingWindowSpec` whose `prefix_cacheable` is
False, the nightly's own opt-out (the kpool tail uses it). The engine core's
block-size minimum, the hash-block gcd, the coordinator's hit lookup and the
base manager's `cache_blocks` all honour it, so no custom manager and no
registry entry are needed (the registry walks the MRO to SlidingWindowSpec).
The opt-out keeps the target's prefix caching exactly what it is without a
drafter, whatever block the draft group uses. Its cost: after a prefix-cache
hit the drafter never writes K/V for the cached tokens, so drafts attend to
stale blocks until the 2048-token window rolls past them. Output is unaffected
(drafts are verified). Acceptance dips for that request.

Draft block size: the target's (`--block-size`, 2304) by default.
VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE shrinks it to a multiple of 64 that divides the
target block. On the old image 576 gave 1.5x the KV pool and 43% slower
decode. Every non-MLA backend the drafter can land on runs any multiple of 64
(FlashAttention/Triton/Flex: MultipleOf(16). FlashInfer: 16/32/64), and the
runner splits the storage block into kernel blocks itself.

Every anchor must match exactly once, or the build fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
MARK = "GLM53-DFLASH2-KV"

EDITS: list[tuple[str, str, str]] = []


def edit(label: str, old: str, new: str) -> None:
    EDITS.append((label, old, new))


# 1. Carve the drafter's plain attention specs out before the MLA check.
edit(
    "grouper carve-out",
    """    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
""",
    """    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    # GLM53-DFLASH2-KV: a speculative drafter (DFlash2, EAGLE3) adds ordinary
    # attention layers to the target's MLA(+indexer)+mamba set. Left in, they
    # fail the MLA check below and the model falls onto the generic unifier,
    # where the indexer page (33 B/token) never divides the MLA page
    # (512 B/token) and cannot be padded. Carve them out into their own
    # group(s), appended last. The target's layout is untouched.
    draft_specs = {
        name: spec
        for name, spec in attn_specs.items()
        if type(spec) in (SlidingWindowSpec, FullAttentionSpec)
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

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
""",
)

# 2. Draft groups go last so every existing group id is unchanged.
edit(
    "grouper return",
    """    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
""",
    """    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        # GLM53-DFLASH2-KV: draft groups last, so existing group ids hold.
        + _glm5_next_draft_groups(draft_specs, uniform_spec.block_size)
    )
""",
)

# 3. The draft spec and its helpers, inserted before the layout recogniser.
HELPERS = '''@dataclass(frozen=True, kw_only=True)
class DraftSlidingWindowSpec(SlidingWindowSpec):
    """GLM53-DFLASH2-KV: a drafter's sliding-window layer on the GLM-5-Next
    layout. Identical to SlidingWindowSpec but opted out of prefix caching,
    the same opt-out the kpool tail uses: the engine core's block-size
    minimum, the hash-block gcd, the coordinator's hit lookup and the base
    manager's cache_blocks all skip it, so the target's caching behaviour is
    exactly what it is without a drafter, whatever block size this group has.
    The registry resolves it to SlidingWindowManager through the MRO.
    """

    @property
    def prefix_cacheable(self) -> bool:
        return False


def _glm5_next_draft_block_size(target_block_size: int) -> int:
    """GLM53-DFLASH2-KV: block size for the drafter's sliding-window group.

    The target block by default. VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE shrinks the
    draft tensors. It must be a multiple of 64, which every non-MLA backend
    runs (MultipleOf(16) or 16/32/64), and divide the target block, which
    keeps the scheduler block (lcm over groups) at the target's.
    """
    requested = os.environ.get("VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE")
    if requested is None:
        return target_block_size
    block = int(requested)
    if block <= 0 or block % 64 != 0 or target_block_size % block != 0:
        raise ValueError(
            f"VLLM_GLM5NEXT_DRAFT_BLOCK_SIZE={block} must be a positive multiple "
            f"of 64 that divides the target block size {target_block_size}"
        )
    return block


def _glm5_next_draft_groups(
    draft_specs: dict[str, KVCacheSpec], target_block_size: int
) -> list[KVCacheGroupSpec]:
    """GLM53-DFLASH2-KV: KV cache groups for drafter attention layers riding
    on the GLM-5-Next layout. Each draft layer gets its own unshared region
    (see get_kv_cache_config_from_groups). Sliding-window layers become
    DraftSlidingWindowSpec. A full-attention drafter layer (EAGLE3 without a
    window) keeps participating at the target's block size.
    """
    if not draft_specs:
        return []
    scaled: dict[str, KVCacheSpec] = {}
    for name, spec in draft_specs.items():
        assert isinstance(spec, AttentionSpec)
        if spec.page_size_padded is not None:
            raise NotImplementedError(
                f"Layer {name}: a padded draft page (kv_cache_dtype_skip_layers) "
                "is not supported on the GLM-5-Next layout. Drop that flag."
            )
        if isinstance(spec, SlidingWindowSpec):
            scaled[name] = replace_as(
                spec,
                DraftSlidingWindowSpec,
                block_size=_glm5_next_draft_block_size(target_block_size),
            )
            continue
        if target_block_size % spec.block_size != 0:
            raise NotImplementedError(
                f"Layer {name}: draft block size {spec.block_size} does not "
                f"divide the target block size {target_block_size}. Pick a "
                "--block-size that is a multiple of the draft backend's kernel "
                "block."
            )
        scaled[name] = replace(spec, block_size=target_block_size)
    same_spec_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for name, spec in scaled.items():
        same_spec_layers[spec].append(name)
    return create_kv_cache_group_specs(scaled, list(same_spec_layers.values()))


def _glm5_next_draft_bytes_per_block(draft_groups: list[KVCacheGroupSpec]) -> int:
    """GLM53-DFLASH2-KV: bytes one pool block costs across all draft layers."""
    return sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes
        for group in draft_groups
    )


'''
edit(
    "helpers inserted",
    "def _glm5_next_tensor_layout(\n",
    HELPERS + "def _glm5_next_tensor_layout(\n",
)

# 4. The layout tuple gains a ninth element: the draft groups.
edit(
    "layout signature",
    '''        list[str],
        int,
    ]
    | None
):
    """Recognize the GLM-5.3-Flash grouping after optional PP projection."""
''',
    '''        list[str],
        int,
        list[KVCacheGroupSpec],
    ]
    | None
):
    """Recognize the GLM-5.3-Flash grouping after optional PP projection.

    GLM53-DFLASH2-KV: the ninth element is the drafter's groups, empty when
    there is no drafter.
    """
''',
)

edit(
    "layout draft classification",
    """    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None
""",
    """    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    # GLM53-DFLASH2-KV: draft groups carry a plain AttentionSpec. The target's
    # attention and tail groups are UniformTypeKVCacheSpecs, so none of them
    # is misclassified. A PP stage without the drafter sees the group with an
    # empty layer list, and it still classifies.
    draft_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, AttentionSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None
""",
)

edit(
    "layout group count",
    """    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None
""",
    """    if len(uniform_groups) + len(mamba_groups) + len(draft_groups) != len(
        kv_cache_groups
    ):
        return None
""",
)

edit(
    "layout return",
    """        tail_names,
        tail_page,
    )


def unify_kv_cache_spec_page_size(
""",
    """        tail_names,
        tail_page,
        draft_groups,
    )


def unify_kv_cache_spec_page_size(
""",
)

# 5. Bytes per pool block (also feeds the null-block reserve and the re-plan).
edit(
    "bytes per block",
    """        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page
""",
    """        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_groups = (
            glm5_layout
        )
        return (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + _glm5_next_draft_bytes_per_block(draft_groups)
        )
""",
)

# 6. The config builder: count the draft pages and emit their regions.
edit(
    "config builder unpack",
    """            tail_names,
            _,
        ) = glm5_layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = may_override_num_blocks(
""",
    """            tail_names,
            _,
            draft_groups,
        ) = glm5_layout
        bytes_per_block = (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            # GLM53-DFLASH2-KV: every pool block also carries the draft pages.
            + _glm5_next_draft_bytes_per_block(draft_groups)
        )
        num_blocks = may_override_num_blocks(
""",
)

edit(
    "config builder draft regions",
    """                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
""",
    """                add_tensor(tail_name, tail_specs[tail_name], offset)

        # GLM53-DFLASH2-KV: one unshared region per draft layer, after the
        # indexer regions. Nothing aliases these.
        draft_offset = idx_base + len(idx_names) * idx_page * num_blocks
        for group in draft_groups:
            for layer_name in group.layer_names:
                add_tensor(layer_name, group.kv_cache_spec, draft_offset)
                draft_offset += group.kv_cache_spec.page_size_bytes * num_blocks

        return KVCacheConfig(
""",
)

# 7. The memory estimate: draft groups draw window-bounded blocks from the
#    same pool, and every pool block carries the draft pages.
edit(
    "memory estimate unpack",
    """            tail_names,
            _,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
""",
    """            tail_names,
            _,
            draft_groups,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
""",
)

edit(
    "memory estimate total",
    """        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
""",
    """        if tail_names:
            total_blocks += 1
        # GLM53-DFLASH2-KV: draft groups draw their window-bounded blocks from
        # the same pool, and every pool block also carries the draft pages.
        for group in draft_groups:
            if group.layer_names:
                total_blocks += cdiv(
                    group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                    group.kv_cache_spec.page_size_bytes,
                )
        return total_blocks * (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + _glm5_next_draft_bytes_per_block(draft_groups)
        )
""",
)


def main() -> None:
    text = PATH.read_text()
    if MARK in text:
        print("[glm53-dflash2-kv] already applied")
        return
    for label, old, _ in EDITS:
        n = text.count(old)
        assert n == 1, (
            f"[glm53-dflash2-kv] anchor '{label}' matched {n} times. "
            "The stock tree changed."
        )
    for _, old, new in EDITS:
        text = text.replace(old, new)
    PATH.write_text(text)
    print(
        f"[glm53-dflash2-kv] {len(EDITS)} edits: GLM-5-Next grouper carries the "
        "DFlash2 drafter's attention layers as their own trailing group"
    )


if __name__ == "__main__":
    main()
