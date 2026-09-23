#!/usr/bin/env python3
"""Always split reasoning from content, whatever the thinking kwargs say.

GLM-5.3-Flash was trained to think on every turn; its official template has no
non-thinking mode. Our template therefore maps `thinking: false` /
`enable_thinking: false` to `Reasoning Effort: Low` and still opens `<think>`
(the empty `<think></think>` we used before is out of distribution and breaks
long structured output). The model now always emits a short reasoning block.

Stock glm47_moe drops its <think> terminals and starts in CONTENT when either
kwarg is false, so that reasoning would land in `content`. This makes the
parser track <think> on every request: thinking off returns a short `reasoning`
and a clean `content`.

Same contract as the other patchers: one anchor, exactly once, or the build
fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py")
OLD = """        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
"""
NEW = """        # GLM53-REASONING-ALWAYS: the template always opens <think> (thinking
        # off means low effort), so the parser must always track it.
        del thinking, enable_thinking
        self.thinking_enabled = True
"""

text = PATH.read_text()
if text.count(NEW) == 1 and text.count(OLD) == 0:
    print("[reasoning-always] already patched")
else:
    assert text.count(OLD) == 1, f"[reasoning-always] anchor matched {text.count(OLD)} times; stock tree changed"
    PATH.write_text(text.replace(OLD, NEW))
    print("[reasoning-always] glm47_moe always parses <think>")
