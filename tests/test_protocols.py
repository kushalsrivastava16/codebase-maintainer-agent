"""
Unit tests for agent/protocols.py.

Covers:
  1. ToolResult is immutable (frozen=True) — assigning to a field raises FrozenInstanceError
  2. TokenUsage.total_tokens property returns the correct sum
  3. isinstance(obj, Tool) returns True for an object with all required attributes/methods
  4. isinstance(obj, Tool) returns False for an object missing required attributes
  5. TaskResult fields are accessible and hold the expected values
  6. A class implementing OrchestratorProtocol.run() passes isinstance check
"""
import dataclasses

import pytest

from agent.protocols import (
    OrchestratorProtocol,
    TaskResult,
    TokenUsage,
    Tool,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Helpers / stub classes
# ---------------------------------------------------------------------------


class _GoodTool:
    """Minimal class that satisfies the Tool protocol."""

    name = "read_file"
    description = "Read a file from the repository."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(is_error=False, content="ok")


class _MissingMethodTool:
    """Has name and description but no execute() — should NOT satisfy Tool."""

    name = "broken_tool"
    description = "A broken tool."
    input_schema = {}


class _MissingAttributeTool:
    """Has execute() but none of the required attributes — should NOT satisfy Tool."""

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(is_error=False, content="ok")


class _GoodOrchestrator:
    """Minimal class that satisfies the OrchestratorProtocol."""

    def run(self, task_type: str, target_path: str) -> TaskResult:
        usage = TokenUsage(input_tokens=0, output_tokens=0, estimated_usd=0.0)
        return TaskResult(
            task_id="abc",
            status="success",
            output_path=None,
            token_usage=usage,
            error=None,
        )


class _BadOrchestrator:
    """Does NOT implement run() — should NOT satisfy OrchestratorProtocol."""

    def execute(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_fields_accessible(self):
        result = ToolResult(is_error=False, content="hello")
        assert result.is_error is False
        assert result.content == "hello"

    def test_error_result_fields(self):
        result = ToolResult(is_error=True, content="file_not_found: foo.py")
        assert result.is_error is True
        assert "file_not_found" in result.content

    def test_is_frozen_content(self):
        """Assigning to content on a frozen dataclass must raise FrozenInstanceError."""
        result = ToolResult(is_error=False, content="original")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.content = "mutated"  # type: ignore[misc]

    def test_is_frozen_is_error(self):
        """Assigning to is_error on a frozen dataclass must raise FrozenInstanceError."""
        result = ToolResult(is_error=False, content="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.is_error = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens_sum(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, estimated_usd=0.001)
        assert usage.total_tokens == 150

    def test_total_tokens_zeros(self):
        usage = TokenUsage(input_tokens=0, output_tokens=0, estimated_usd=0.0)
        assert usage.total_tokens == 0

    def test_total_tokens_large_values(self):
        usage = TokenUsage(input_tokens=50_000, output_tokens=10_000, estimated_usd=0.30)
        assert usage.total_tokens == 60_000

    def test_estimated_usd_accessible(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, estimated_usd=0.0123)
        assert usage.estimated_usd == pytest.approx(0.0123)

    def test_is_frozen(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5, estimated_usd=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            usage.input_tokens = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskResult tests
# ---------------------------------------------------------------------------


class TestTaskResult:
    def _make_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=200, output_tokens=100, estimated_usd=0.002)

    def test_fields_success(self):
        usage = self._make_usage()
        result = TaskResult(
            task_id="task-uuid-1234",
            status="success",
            output_path="/agent_output/lint_fix_utils_2024.diff",
            token_usage=usage,
            error=None,
        )
        assert result.task_id == "task-uuid-1234"
        assert result.status == "success"
        assert result.output_path == "/agent_output/lint_fix_utils_2024.diff"
        assert result.token_usage is usage
        assert result.error is None

    def test_fields_failed_with_error(self):
        usage = self._make_usage()
        result = TaskResult(
            task_id="task-uuid-5678",
            status="failed",
            output_path=None,
            token_usage=usage,
            error="max_iterations_exceeded",
        )
        assert result.status == "failed"
        assert result.output_path is None
        assert result.error == "max_iterations_exceeded"

    def test_output_path_none_allowed(self):
        """output_path is explicitly None when lint is clean or task failed pre-diff."""
        usage = self._make_usage()
        result = TaskResult(
            task_id="x",
            status="success",
            output_path=None,
            token_usage=usage,
            error=None,
        )
        assert result.output_path is None

    def test_is_frozen(self):
        usage = self._make_usage()
        result = TaskResult(
            task_id="x",
            status="success",
            output_path=None,
            token_usage=usage,
            error=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = "failed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tool Protocol isinstance checks
# ---------------------------------------------------------------------------


class TestToolProtocol:
    def test_good_tool_passes_isinstance(self):
        """An object with name, description, input_schema, and execute() satisfies Tool."""
        tool = _GoodTool()
        assert isinstance(tool, Tool)

    def test_missing_execute_fails_isinstance(self):
        """An object without execute() does NOT satisfy Tool."""
        bad = _MissingMethodTool()
        assert not isinstance(bad, Tool)

    def test_missing_attributes_fails_isinstance(self):
        """An object with only execute() but no required attributes does NOT satisfy Tool."""
        bad = _MissingAttributeTool()
        assert not isinstance(bad, Tool)

    def test_plain_object_fails_isinstance(self):
        """A plain object with no relevant attributes does not satisfy Tool."""
        assert not isinstance(object(), Tool)

    def test_tool_execute_returns_tool_result(self):
        """Verify the good stub actually returns a ToolResult."""
        tool = _GoodTool()
        result = tool.execute({"path": "foo.py"})
        assert isinstance(result, ToolResult)
        assert result.is_error is False


# ---------------------------------------------------------------------------
# OrchestratorProtocol isinstance checks
# ---------------------------------------------------------------------------


class TestOrchestratorProtocol:
    def test_good_orchestrator_passes_isinstance(self):
        """A class with run(task_type, target_path) -> TaskResult satisfies the protocol."""
        orch = _GoodOrchestrator()
        assert isinstance(orch, OrchestratorProtocol)

    def test_bad_orchestrator_fails_isinstance(self):
        """A class without run() does NOT satisfy OrchestratorProtocol."""
        bad = _BadOrchestrator()
        assert not isinstance(bad, OrchestratorProtocol)

    def test_plain_object_fails_isinstance(self):
        assert not isinstance(object(), OrchestratorProtocol)

    def test_orchestrator_run_returns_task_result(self):
        """Verify the stub orchestrator returns a properly structured TaskResult."""
        orch = _GoodOrchestrator()
        result = orch.run("lint_fix", "/some/path/utils.py")
        assert isinstance(result, TaskResult)
        assert result.status == "success"
        assert result.task_id == "abc"
