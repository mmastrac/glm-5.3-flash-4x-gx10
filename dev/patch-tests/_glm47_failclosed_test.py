"""Model-free checks of the fail-closed validator and name resolution."""
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager as M

M.import_tool_parser("/tmp/plugin.py")
P = M.get_tool_parser("glm47_failclosed")._parser_engine_cls

def tool(name, props):
    return ChatCompletionToolsParam.model_validate({
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": props}},
    })

parser = P.__new__(P)  # only _tools is needed
parser._tools = [
    tool("bash", {"command": {}, "timeout": {}, "workdir": {}}),
    tool("run_code", {"code": {}, "description": {}}),
]

print("== name resolution")
for emitted, want in [("bash", "bash"), ("bash</arg_key>x", "bash"),
                      ("bash1635", None), ("run_code", "run_code"),
                      ("nope", None), ("run_code</arg_value>", "run_code")]:
    got = parser._resolve_name(emitted)
    print(f"  {emitted:<24} -> {str(got):<10} {'ok' if got == want else 'WANTED ' + str(want)}")

print("== validation")
cases = [
    ("good call", "bash", '{"command": "ls", "timeout": 5}', True),
    ("no args", "bash", "", True),
    ("unknown tool", "list_files", '{"path": "/"}', False),
    ("corrupted name", "bash</arg_value><arg_key>x", '{"command": "ls"}', False),
    ("key not in schema", "bash", '{"prefix": "x"}', False),
    ("corrupt key", "run_code",
     '{"print_code_snapshot</arg_value><arg_key>description": "x"}', False),
    ("not json", "bash", '{"command": ', False),
    ("args not an object", "bash", '["command"]', False),
]
for label, name, args, want_ok in cases:
    reason = parser._reject_reason(name, args)
    ok = reason is None
    mark = "ok" if ok == want_ok else "UNEXPECTED"
    print(f"  {label:<20} {'accept' if ok else 'refuse':<7} {mark:<11} {(reason or '')[:60]}")
