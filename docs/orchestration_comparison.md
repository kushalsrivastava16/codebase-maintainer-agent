# Orchestration Comparison: Raw Python Loop vs. LangGraph

This document analyses the architectural tradeoffs between the raw Python `while`-loop
orchestrator used in Phase 1 (`agent/orchestrator.py`) and a hypothetical LangGraph-based
replacement. It is intended to inform the decision of whether to adopt LangGraph in Phase 2.

---

## 1. What Phase 1 Actually Does

The current orchestrator in `agent/orchestrator.py` manages an LLM agent loop with three
explicit state variables:

```python
messages: list[dict] = []   # Anthropic conversation history
iteration: int = 0          # loop counter, checked against max_iterations
retry_count: int = 0        # self-correction attempt counter
```

Every state transition is a plain Python statement:

- `messages.append(...)` — history update
- `iteration += 1` — advance the loop counter
- `retry_count += 1` — record a correction attempt (Phase 2)

The loop has two abort guards at the top of each iteration:

```python
if iteration >= self._config.max_iterations:
    return self._abort(task_id, "max_iterations_exceeded", ..., exit_code=3)

if self._cost_tracker.budget_exceeded():
    return self._abort(task_id, "budget_exceeded", ..., exit_code=5)
```

Tool dispatch is a for-loop over `response.content` blocks when `stop_reason == "tool_use"`.
The final answer is extracted from `response.content` with a regex when `stop_reason == "end_turn"`.
The entire functional loop is approximately **90 lines** (excluding helpers and docstrings);
the full class with helpers is **~160 lines**.

---

## 2. Side-by-Side Comparison

| Criterion | Raw Python `while` loop (Phase 1) | LangGraph |
|---|---|---|
| **Transparency** | Every state variable (`messages`, `iteration`, `retry_count`) is an explicit Python local variable. A developer can set a breakpoint on any line and inspect or mutate state with standard tooling. There is no framework between the code and its execution. | State lives inside a `TypedDict` (the LangGraph `State` schema) and transitions are encoded as edges in a graph. Understanding what happens when requires reading both the node functions and the graph's `add_conditional_edges` calls — two levels of indirection instead of one. |
| **Boilerplate** | Phase 1 loop is ~90 lines of active code. Adding a retry branch means adding an `elif` and incrementing `retry_count`. Adding a new abort condition is one `if` block. | Initial StateGraph definition is ~50 lines (graph definition + node wiring), but requires ~20 additional lines of framework setup (state schema, `StateGraph` construction, `compile()`). Total comparable implementation: ~120–140 lines. Framework knowledge is a prerequisite before any line is written. |
| **Debuggability** | Standard Python `pdb` / IDE debugger works without modification. Structured logs (`llm_call`, `tool_dispatch`, `tool_result`, `llm_response`) are emitted inline at the call site in `orchestrator.py`, so log events are co-located with the code that generates them. | LangGraph execution is driven by an internal runner. To add per-node logging you must either add it in every node function or hook into LangGraph's callback system (`StreamingStdOutCallbackHandler` or custom callbacks). The stack trace of a node failure includes framework internals, making the root cause harder to locate. |
| **Testability** | `Orchestrator.run()` can be unit-tested by mocking `self._client.messages.create` (one method). Tool dispatch is exercised by substituting the `tools` dict with test doubles. See `tests/test_orchestrator.py`. | Node functions can be unit-tested independently, which is a genuine advantage. However, testing the graph as a whole requires either running the compiled graph (bringing in the full LangGraph runtime) or mocking at a lower level. |
| **Framework lock-in** | Zero external dependency for orchestration. The agent runs anywhere Python and the `anthropic` SDK are installed. | Requires `langgraph` and its transitive dependencies (`langchain-core`, etc.). LangGraph is under active development; breaking changes between minor versions have been observed in 2024. |
| **Extensibility** | Adding a new task type or conditional branch is an `elif` or a new method. Adding parallelism would require explicit `threading`/`asyncio` — currently not needed. | LangGraph's `Send` API and parallel node execution are built-in. If Phase 3+ needs concurrent issue processing across multiple files, LangGraph's primitives are better suited than hand-rolled threading. |
| **Production readiness** | Retry logic, checkpointing, and persistence must be re-implemented from scratch (Phase 2 adds `MemoryStore`; retry is manual). | LangGraph includes built-in checkpointing (PostgreSQL, SQLite) and retry policies. These would replace parts of `memory.py` and the manual retry counter for free. |

---

## 3. Estimated Lines-of-Code Difference

| Scope | Raw loop (`orchestrator.py`) | Equivalent LangGraph |
|---|---|---|
| Core loop logic | ~90 lines | ~70 lines (node functions are shorter individually) |
| State management | 3 explicit locals | ~25 lines (State TypedDict + graph wiring) |
| Tool dispatch | ~25 lines | ~25 lines (same logic inside a node) |
| Abort / error handling | ~20 lines | ~20 lines |
| Framework setup | 0 | ~20 lines (StateGraph, compile, stream) |
| **Total (active code)** | **~160 lines** | **~190–210 lines** |

The raw loop is shorter overall. LangGraph pays for itself in large, multi-branch graphs.
For a two-branch loop (tool use vs. end turn), the framework overhead is real.

The design document's estimate of "~250 lines raw vs. ~180 lines LangGraph" assumed a
larger orchestrator. The actual Phase 1 implementation is leaner than the design doc
anticipated because helpers (`_extract_text`, `_extract_code_block`, `_build_system_prompt`)
are kept separate from the loop logic.

---

## 4. Author's Recommendation for Phase 2

**Do not adopt LangGraph in Phase 2.**

Phase 2 adds two features to the existing loop: persistent deduplication (`MemoryStore`) and
a self-correction retry loop (`retry_count += 1`, maximum 3 attempts). Both of these fit
naturally into the existing `while` loop as a counter check and a branch — they require
fewer than 30 additional lines and no architectural change.

Migrating to LangGraph at this point would:
1. Add `langchain-core` and `langgraph` to `pyproject.toml`, increasing the dependency
   surface without a corresponding feature gain.
2. Replace readable Python state (`retry_count`, `messages`) with a framework `State` dict
   that requires knowledge of LangGraph's execution model to reason about.
3. Eliminate the pedagogical value of the Phase 1 design: the raw loop is intentionally
   transparent so that every state transition is understandable without framework knowledge.

**Reconsider for Phase 3 or 4**, when the agent may need to:
- Process multiple GitHub issues concurrently (LangGraph's `Send` and parallel branches)
- Persist checkpoints between runs for long-running triage sessions (LangGraph's built-in
  SQLite checkpointer would replace or augment `MemoryStore`)
- Route to different sub-graphs depending on issue type (LangGraph's conditional edges)

At that point, the raw loop's extensibility cost becomes concrete rather than hypothetical,
and the migration would be principled — we would know exactly which LangGraph features we
need because we hit their absence in practice.

---

## 5. Migration Path (When the Time Comes)

The `OrchestratorProtocol` in `agent/protocols.py` makes migration mechanical:

```python
class OrchestratorProtocol(Protocol):
    def run(self, task_type: str, target_path: str) -> TaskResult: ...
```

A `LangGraphOrchestrator` that satisfies this protocol can replace `Orchestrator` in the
CLI wiring (`agent/__main__.py`) without touching any tool, config, or logger code. The
raw loop and the LangGraph version can co-exist in the same codebase during a transition,
switchable by a `--orchestrator` CLI flag.

---

*Document written for Phase 5. Revisit before the Phase 3 GitHub integration milestone.*
