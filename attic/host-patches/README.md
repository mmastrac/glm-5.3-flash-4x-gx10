# Host patches

vLLM modules bind-mounted read-only over the image's copies, wired by the
compose files in `../overrides/`. They are applied per host under
`/home/admin/`, so `docker compose` picks them up without a rebuild -- and so
they are invisible to anyone reading `../patches/`, which holds the patches
baked into the image instead.

    host-patches/pp-patches-dflash     -> /home/admin/pp-patches-dflash
    host-patches/thinking-budget-patch -> /home/admin/thinking-budget-patch
    host-patches/kpool-fix             -> /home/admin/kpool-fix
    host-patches/spin-patch            -> /home/admin/spin-patch

Byte-identical on all four GB10 nodes as of 2026-09-10, checked by md5. Nothing
enforces that: a node patched alone would serve a different model with no
warning, since only rank 0 reads most of these.

`thinking-budget-patch/sampler.py` supersedes `../patches/thinking_budget_guard.py`.
The bind mount wins over the baked patch, so the sampler that runs is this one.

`pp-patches` was removed on 2026-09-10: every file in it was PP-only or inert at
`pipeline_parallel_size=1`. See ../README.md, and git history for the files.
