"""
Core data models and Protocol interfaces for the Codebase Maintainer Agent.

WHY Protocols instead of ABCs:
  Python's typing.Protocol enables structural subtyping — any class with the right
  shape satisfies the interface without explicit inheritance. This keeps tools
  decoupled from the orchestrator: we can add a new tool without touching the
  orchestrator code at all, as long as the tool has the right attributes/methods.

WHY frozen dataclasses for value objects (ToolResult, TokenUsage, TaskResult):
  These are pure data carriers — they should never be mutated after creation.
  frozen=True prevents accidental mutation and makes them hashable, which is useful
  for deduplication and caching in later phases.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """
    The return type of every Tool.execute() call.

    WHY is_error as a boolean field instead of raising exceptions?
      Agent loops are long-running. If a tool raises an unhandled exception, it
      propagates up through the orchestrator and kills the entire run. By making
      errors first-class values, we let the LLM reason about the error and
      potentially recover — just like a human would if a command failed.
    """

    is_error: bool
    content: str  # UTF-8 string returned to the LLM; may be error message or data


@dataclass(frozen=True)
class TokenUsage:
    """
    Immutable snapshot of token consumption for a task run or a single LLM call.

    estimated_usd is computed by the CostTracker using per-model pricing constants.
    It is an estimate — actual billing may differ slightly due to API rounding.
    """

    input_tokens: int
    output_tokens: int
    estimated_usd: float  # USD; computed as (input * price_in + output * price_out)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class TaskResult:
    """
    The final outcome of a single task run, returned by Orchestrator.run().

    output_path is None when:
      - The linter found no violations (clean file)
      - The task failed before producing any diff
      - The task was a triage task (output is a GitHub comment, not a local file)
    """

    task_id: str  # UUID4 string, generated at task start
    status: str  # "success" | "failed" | "pending"
    output_path: str | None  # Absolute path to .diff file, or None
    token_usage: TokenUsage
    error: str | None  # Human-readable error message, or None on success


@runtime_checkable
class Tool(Protocol):
    """
    Structural interface for all agent tools.

    WHY runtime_checkable?
      Allows isinstance(obj, Tool) checks at runtime — useful in the orchestrator
      for validating that registered tools implement the full interface before
      starting a task, rather than discovering the missing method mid-run.

    Tool contract:
      - name: snake_case identifier matching the tool name in Anthropic tool definitions
      - description: LLM-facing description (1-2 sentences, present tense)
      - input_schema: valid JSON Schema object passed in the Anthropic tools parameter
      - execute(): NEVER raises; always returns ToolResult (error or success)
    """

    name: str  # e.g. "read_file", "run_lint"
    description: str  # shown to the LLM to explain what the tool does
    input_schema: dict  # JSON Schema for the tool's input parameters

    def execute(self, inputs: dict) -> ToolResult:
        """
        Execute the tool with the given inputs and return a result.

        Contract: this method must NEVER raise an exception. All error conditions
        must be returned as ToolResult(is_error=True, content="<error message>").
        The orchestrator wraps execute() in a try/except as a safety net, but tools
        should handle their own errors explicitly for clarity.
        """
        ...


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """
    Structural interface for the agent orchestrator.

    WHY define this Protocol?
      Phase 1 uses a raw while loop. Phase 2+ may adopt LangGraph or another
      framework. By defining this Protocol, we guarantee that switching the
      orchestration layer only requires changing the class that implements
      OrchestratorProtocol — the CLI, memory store, and tools need zero changes.

      This is the "dependency inversion principle" applied to agent architecture:
      high-level policy (what tasks to run) is decoupled from low-level mechanism
      (how the loop is structured).
    """

    def run(self, task_type: str, target_path: str) -> TaskResult:
        """
        Run a single maintenance task end-to-end.

        task_type: one of "lint_fix", "generate_tests", "convert_todos", "triage_issues"
        target_path: absolute path to the target file or directory

        Returns a TaskResult with status "success" or "failed".
        May raise SystemExit with a numeric exit code on unrecoverable errors.
        """
        ...
