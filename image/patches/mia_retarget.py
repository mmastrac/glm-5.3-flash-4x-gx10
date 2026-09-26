#!/usr/bin/env python3
"""Point MiaAI's vendored patcher at where its targets live in this base.

glm53-flash_SM121.py stays verbatim (MIT, licence beside it). Two of its edits
target code that moved, so this rewrites their paths and anchors in a copy
before it runs:

- The indexer's two top-k allocations moved from
  model_executor/layers/sparse_attn_indexer_kpool.py to
  models/glm5next/nvidia/sparse_indexer.py, byte for byte.
- FlashInfer 0.7.0 moved the FA2 "FP8 kv_data_type for MLA requires an SM90"
  gate from mla/_core.py into mla/_batch_mla/_backends/_fa_common.py, as a
  function of `device` at a shallower indent.

Every retarget matches exactly once, or the build fails.

usage: mia_retarget.py <copy of glm53-flash_SM121.py>
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
RETARGETS = [
    (
        "model_executor/layers/sparse_attn_indexer_kpool.py",
        "models/glm5next/nvidia/sparse_indexer.py",
    ),
    ('FI / "mla/_core.py"', 'FI / "mla/_batch_mla/_backends/_fa_common.py"'),
    (
        'old = "            major, minor = get_compute_capability(self.device)\\n'
        '            if major != 9:\\n"',
        'old = "        major, minor = get_compute_capability(device)\\n'
        '        if major != 9:\\n"',
    ),
    (
        'new = "            major, minor = get_compute_capability(self.device)\\n'
        '            if major not in (9, 12):\\n"',
        'new = "        major, minor = get_compute_capability(device)\\n'
        '        if major not in (9, 12):\\n"',
    ),
]
for old, new in RETARGETS:
    n = text.count(old)
    assert n == 1, f"[mia-retarget] {old[:60]!r} matched {n}x, want 1"
    text = text.replace(old, new)
path.write_text(text)
print("[mia-retarget] indexer path and FlashInfer 0.7.0 fp8 gate retargeted")
