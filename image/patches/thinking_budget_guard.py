#!/usr/bin/env python3
"""Let thinking_token_budget reach the V2 sampler's budget kernel.

The V2 sampler applies the thinking budget inside apply_sampling_params, which
only runs when _requires_logits_processing says some request needs logits work
(bias, penalties, bad words, temperature outside {0, 1}, min_p, top_k, top_p).
That guard never looked at the thinking budget. A greedy request resets top_p,
top_k and min_p (sampling_params.py), and this model's generation_config sets
only temperature=1.0, so on a plain request every clause is false and the
budget is recorded and never applied: accepted and ignored, with no warning.

The guard is np.any over the whole batch, so a co-scheduled request with, say,
temperature 0.7 turned the cap on for everyone. That is why it looked random.

Upstream main fixed the same gap with a per-request needs_logits_processing
flag that includes use_thinking_budget. This adds the equivalent clause to the
guard we have. Same contract as the neighbours: exactly one anchor or the
build fails.
"""
from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
LOG = "[thinking-budget-guard]"

OLD = (
    "    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:\n"
    "        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):\n"
    "            return True\n"
)
NEW = OLD + (
    "        # Local patch (home-infra): thinking_token_budget is applied inside\n"
    "        # apply_sampling_params, but the fast path skipped it whenever nothing\n"
    "        # else needed logits processing (e.g. greedy). Upstream main fixed the\n"
    "        # same gap via a per-request needs_logits_processing flag.\n"
    "        if self.thinking_budget_state.enabled and np.any(\n"
    "            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]\n"
    "        ):\n"
    "            return True\n"
)


def apply_once(path: Path, old: str, new: str, label: str) -> str:
    text = path.read_text()
    n_old, n_new = text.count(old), text.count(new)
    if n_old == 0 and n_new == 1:
        return "skipped"
    if n_old != 1:
        raise SystemExit(
            "%s refuse %s (old=%d new=%d); stock tree changed" % (LOG, label, n_old, n_new)
        )
    path.write_text(text.replace(old, new, 1))
    return "applied"


def main() -> None:
    path = VLLM / "v1/worker/gpu/sample/sampler.py"
    text = path.read_text()
    if "self.needs_logits_processing[" in text:
        print("%s sampler already carries upstream's flag; nothing to do" % LOG)
        return
    # OLD is a prefix of NEW, so apply_once alone would insert the hunk again.
    if NEW in text:
        print("%s sampler.py: skipped" % LOG)
        return
    print("%s %s: %s" % (LOG, path.name, apply_once(path, OLD, NEW, "guard")))


if __name__ == "__main__":
    main()
