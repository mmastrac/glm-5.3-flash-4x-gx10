"""A fail-closed glm47 tool parser, registered as `glm47_failclosed`.

Loaded with `--tool-parser-plugin` and selected with
`--tool-call-parser glm47_failclosed`, so no upstream file is forked or mounted.

## Why

The forward pass is not bit-reproducible at M >> 1 on this hardware (see
../TOPK-CORRUPTION.md), and the token it lands on is occasionally part of a tool
call. Upstream handles a corrupted *name* by dropping the call: the turn ends
`stop` with no content and no `tool_calls`, which clients report as a dead
endpoint. It does not check argument *keys* at all, so a call carrying a garbage
key is emitted as though it were fine, the client echoes it into history, the
chat template re-renders it, and the model imitates its own corruption until a
turn is unparseable.

Containment, not a cure: nothing here stops the first bad token, it only keeps
one from poisoning the rest of the session.

Approach and both of its corrections come from
NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark, issue #7 and PR #11.

## What it does

Validates at TOOL_CALL_END, once the whole call is buffered
(`stream_arg_deltas=False`): the name must be shaped like a tool name and resolve
to one the request offered, the arguments must parse to an object, and every key
must appear in that tool's schema.

A call that fails is **re-offered with a sentinel argument key** rather than
dropped or turned into text. Surfacing it as content clears the tool slot, and an
agent loop reads that as a finished turn and stops -- the bug behind PR #11. A
call no client can accept is a retryable error instead, so the loop continues.
The sentinel's value names the specific mistake and lists the valid keys, because
an unexplained rejection invites the model to repeat it.

Name resolution takes an exact match first, then the longest offered name the
emitted one starts with, so `bash</arg_key>...` resolves to `bash` while
`bash1635` does not.

`GLM47_FAIL_CLOSED=0` restores upstream behaviour. `GLM47_REJECT_AS_TEXT=1`
surfaces refusals as content instead, which is the pre-#11 behaviour and ends
agent loops.
"""

from __future__ import annotations

import json
import os
import re

# Moved from vllm.entrypoints.openai.engine.protocol on vLLM main; the old
# path is why this plugin failed to register on the 2026-09-22 nightly.
from vllm.entrypoints.generate.base.protocol import DeltaFunctionCall, DeltaToolCall
from vllm.logger import init_logger
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties

logger = init_logger(__name__)

# Tool and argument names as these schemas spell them: a leading letter or
# underscore, then word characters, dots or dashes. Narrower than the JSON that
# reaches us, which is the point.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# No real schema declares this, so a client that validates arguments must reject
# the call, and one that does not gets a message saying what happened.
SENTINEL_KEY = "_rejected_by_server"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


ENABLED = _flag("GLM47_FAIL_CLOSED", "1")
REJECT_AS_TEXT = _flag("GLM47_REJECT_AS_TEXT", "0")


class Glm47FailClosedParser(Glm47MoeParser):
    """Glm47MoeParser that refuses any call it cannot fully validate."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        # An argument delta cannot be withdrawn once sent, and the decision needs
        # the whole argument object.
        self._stream_arg_deltas = False

    def _offered_names(self) -> list[str]:
        names: list[str] = []
        for tool in self._tools or []:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def _resolve_name(self, emitted: str) -> str | None:
        """The offered tool this name means, or None.

        Exact match first. Otherwise the longest offered name the emitted one
        starts with, which recovers a name whose tail is corrupted markup without
        matching a different tool that merely shares a prefix.
        """
        if find_tool_name(self._tools, emitted):
            return emitted
        candidates = [n for n in self._offered_names() if emitted.startswith(n)]
        if not candidates:
            return None
        best = max(candidates, key=len)
        tail = emitted[len(best):]
        if not tail:
            return best
        # What follows decides: a name character means a different tool
        # (`bash1635`), anything else means corruption after the real name
        # (`bash</arg_key>`).
        return None if re.match(r"[A-Za-z0-9_.-]", tail) else best

    def _reject_reason(self, name: str, args_json: str) -> str | None:
        """Why this call cannot be trusted, or None if it can."""
        if not NAME_RE.match(name):
            return f"tool name {name!r} is not shaped like a tool name"
        if self._tools and not find_tool_name(self._tools, name):
            return f"tool name {name!r} is not one this request offered"
        text = (args_json or "").strip()
        if not text:
            return None
        try:
            args = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"arguments for {name!r} are not JSON: {exc}"
        if not isinstance(args, dict):
            return f"arguments for {name!r} are {type(args).__name__}, not an object"
        properties = find_tool_properties(self._tools, name) or {}
        for key in args:
            if not isinstance(key, str) or not NAME_RE.match(key):
                return f"argument key {key!r} is not shaped like a key"
            # No declared properties means any key is unverifiable rather than
            # wrong, so it passes.
            if properties and key not in properties:
                valid = ", ".join(sorted(properties)) or "none"
                return (
                    f"argument key {key!r} is not in the schema for {name!r}; "
                    f"valid keys are: {valid}"
                )
        return None

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if not ENABLED or not (0 <= idx < len(self._tool_slots)):
            super()._handle_tool_end(event, deltas)
            return

        slot = self._tool_slots[idx]
        emitted = (slot.name or self._try_extract_name(idx) or "").strip()
        resolved = self._resolve_name(emitted)
        # `args` is a property, so reading it leaves the converter's state alone.
        reason = self._reject_reason(resolved or emitted, slot.args)
        if reason is None:
            if resolved and resolved != emitted:
                slot.name = resolved
            super()._handle_tool_end(event, deltas)
            return

        # Drain this slot's converter so its pending text cannot surface inside
        # the next call.
        self._flush_arg_converter(idx)
        logger.warning("glm47 fail-closed: %s", reason)

        if REJECT_AS_TEXT:
            # With no tool delta for this event the engine flushes deferred
            # content as ordinary content. This ends agent loops; see PR #11.
            self._deferred_content += f"\n[tool call refused: {reason}]\n"
            slot.name_sent = True
            slot.streamed_json = slot.args or ""
            return

        # Re-offer under the resolved name where there is one, so the client
        # recognises the tool and rejects it on the arguments.
        name = resolved or emitted if NAME_RE.match(emitted) else (resolved or "unknown_tool")
        slot.name = name
        slot.name_sent = True
        self._ensure_tool_id(slot, name)
        arguments = json.dumps({SENTINEL_KEY: reason}, ensure_ascii=False)
        slot.streamed_json = arguments
        deltas.append(
            DeltaToolCall(
                index=idx,
                id=slot.id,
                type="function",
                function=DeltaFunctionCall(name=name, arguments=arguments),
            )
        )


_REASONING_ADAPTER, _TOOL_ADAPTER = make_adapters(Glm47FailClosedParser)


@ToolParserManager.register_module("glm47_failclosed")
class Glm47FailClosedToolParser(_TOOL_ADAPTER):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
