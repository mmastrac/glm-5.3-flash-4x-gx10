#!/bin/bash
# thinking_token_budget is applied inside apply_sampling_params, which the V2
# sampler skips unless _requires_logits_processing() is true. That predicate
# never checks the budget, and temperature 0 (greedy) and temperature 1 (this
# model's generation_config default) both miss every other condition -- so the
# budget is silently ignored on the most common requests. Measured before the
# fix, budget=16: temp 0 -> 1426 chars, temp 1.0 -> 1996, temp 0.6 -> 61.
# Upstream main fixes the same gap with a per-request needs_logits_processing
# flag (vllm#46727 added the V2 kernel; vllm#50473 reports the gap).
set -euo pipefail
F=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/sampler.py
OUT=${1:-${SPARK_HOME:-/home/admin}/thinking-budget-patch}
mkdir -p "$OUT"
sudo docker create --name _tb_tmp "${IMAGE:-glm53-spark:sm90-v21}" >/dev/null
sudo docker cp "_tb_tmp:$F" "$OUT/sampler.py"
sudo docker rm _tb_tmp >/dev/null
sudo chown "$(id -u):$(id -g)" "$OUT/sampler.py"
python3 - "$OUT/sampler.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if "thinking_budget_state.use_thinking_budget[idx_mapping_np]" in s.split("def apply_sampling_params")[0]:
    print("already patched"); sys.exit()
anchor = "    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:\n"
if anchor not in s:
    sys.exit("anchor not found; the sampler has changed, re-derive this patch")
add = (anchor +
"        if self.thinking_budget_state.enabled and np.any(\n"
"            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]\n"
"        ):\n"
"            return True\n")
p.write_text(s.replace(anchor, add, 1))
print("patched")
PY
python3 -c "import ast,sys;ast.parse(open('$OUT/sampler.py').read());print('parses OK')"
