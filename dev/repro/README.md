# Reproductions for the greedy-decoding corruption

Three scripts, smallest first. See `../TOPK-CORRUPTION.md` for the diagnosis
these came out of. The two HTTP scripts use the standard library only and take
`--url` and `--model`, so they can be attached to an issue as they are.

## greedy_nondet.py

Identical greedy requests over a ~42k-token prompt return different
completions. The prompt is one paragraph repeated to length, so there is
nothing model-specific or agent-specific in it.

Measured against GLM-5.3-Flash-NVFP4, 4x GB10 (sm121) at TP=4:

    42034 tokens, 16 runs   3 distinct completions
     2104 tokens, 16 runs   1 distinct completion

Use this one for a determinism issue. It shows the effect and nothing else.

## toolcall_corruption.py

The same nondeterminism landing inside a tool call, which is what reaches a
user. It rebuilds a 42272-token coding-agent transcript (47 tools, 24
tool-call/result pairs, generated from fixed seeds) and asks for a Bash call
repeating a command from earlier in the conversation, so the correct output is
fixed and every difference is legible.

One run of 40 at temperature 0, same seed, 5 distinct completions:

    30 runs  correct
     4 runs  pos 23  ' identical' for '>&'       -> pytest -q 2 identical 2>&1
     3 runs  pos 26  ' pytest'    for ' tail'    -> ... 2>&1 | pytest -q 2>&1 | tail -40
     2 runs  pos 23  ' Bash'      for '>&'       -> pytest -q 2 Bash 2>&1
     1 run   pos 31  ' Read'      for 'description'

The last three corrupt a tool name rather than an argument. That is the path to
the symptom clients actually report: `glm47_moe.py` runs with
`validate_tool_names=True`, so an unknown name emits zero deltas and the request
finishes `stop` with no content and no tool calls.

It is 340 lines because shrinking it stops it reproducing. Two reductions were
measured and both went bit-stable over 40 runs: replacing the generated file
listings with the command's own output repeated, and dropping to 3 tools with
one Read/Edit chain. Both make every token of the command high-margin. The
length matters too, and 2k tokens is stable over 16 runs.

## marlin_moe_nondet.py

`fused_marlin_moe` returns different bits for identical inputs at some M. No
checkpoint and no server: random NVFP4 weights on one CUDA device, in a
container that has vLLM importable.

Measured earlier at the GLM-5.3-Flash geometry (288 experts, top-8, hidden
4096, intermediate 2048), 24 repetitions per M:

    M=3104   23/23 runs differ
    M=2304    9/39 runs differ
    M=1536    6/11 runs differ
    M=3072, 2816, 2272, 2048, 1024, 800   bit-identical

It needs roughly 4 GB free, so it cannot run while the engine holds
`gpu_memory_utilization=0.88`. It has not been re-run since the numbers above
were taken, and no measurement yet connects those M values to the ones the
engine actually issues, so treat it as a separate finding rather than the cause
of the two scripts above.
