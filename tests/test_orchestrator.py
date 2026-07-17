"""
Tests for agent/orchestrator.py.

Tests cover:
  1. _dispatch_tool() returns unknown_tool error for unregistered tool name
  2. _dispatch_tool() returns ToolResult(is_error=True) when tool raises exception
  3. _dispatch_tool() returns tool's result for valid tool call
  4. _extract_code_block() extracts Python code from ```python ... ``` block
  5. _extract_code_block() returns None when no code block present
  6. _extract_code_block() handles plain ``` block (no language tag)
  7. _extract_text() concatenates all TextBlock texts
  8. run() aborts with SystemExit(3) when max_iterations exceeded
  9. run() aborts with SystemExit(5) when budget exceeded
 10. run() returns TaskResult(status="success") when LLM returns end_turn with code block
 11. run() returns TaskResult(status="success", output_path=None) when end_turn without code block
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import AgentConfig
from agent.cost_tracker import CostTracker
from agent.diff_writer import DiffWriter
from agent.logger import StructuredLogger
from agent.orchestrator import Orchestrator
from agent.protocols import TaskResult, ToolResult


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class FakeUsage:
    """Minimal stand-in for anthropic.types.Usage."""

    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeContent:
    """
    Stand-in for Anthropic response content blocks.

    type: "text" or "tool_use"
    text: text content (for text blocks)
    name: tool name (for tool_use blocks)
    input: tool inputs dict (for tool_use blocks)
    id: tool_use_id (for tool_use blocks)
    """

    def __init__(
        self,
        block_type: str = "text",
        text: str = "",
        name: str = "",
        inputs: dict | None = None,
        block_id: str = "tu_001",
    ) -> None:
        self.type = block_type
        self.text = text
        self.name = name
        self.input = inputs or {}
        self.id = block_id


class FakeResponse:
    """Minimal stand-in for anthropic.types.Message."""

    def __init__(
        self,
        stop_reason: str = "end_turn",
        content: list | None = None,
        usage: FakeUsage | None = None,
    ) -> None:
        self.stop_reason = stop_reason
        self.content = content if content is not None else []
        self.usage = usage if usage is not None else FakeUsage()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_orchestrator(
    max_iterations: int = 5,
    max_tokens: int = 50_000,
    tools: dict | None = None,
    original_content: str = "# original",
) -> Orchestrator:
    """Build an Orchestrator with test-friendly defaults."""
    config = AgentConfig(
        max_iterations=max_iterations,
        max_tokens_per_task=max_tokens,
        output_dir="./agent_output",
        model="claude-haiku-4-5-20251001",
    )
    logger = StructuredLogger(min_level="ERROR")  # suppress noise in tests
    cost_tracker = CostTracker(max_tokens=max_tokens)
    diff_writer = MagicMock(spec=DiffWriter)
    diff_writer.write.return_value = "/tmp/fake.diff"

    orch = Orchestrator(
        config=config,
        logger=logger,
        cost_tracker=cost_tracker,
        diff_writer=diff_writer,
        tools=tools or {},
        original_content=original_content,
    )
    return orch


def make_tool(name: str = "my_tool", result: ToolResult | None = None) -> MagicMock:
    """Build a mock Tool that returns the given ToolResult."""
    tool = MagicMock()
    tool.name = name
    tool.description = "A test tool"
    tool.input_schema = {"type": "object", "properties": {}}
    tool.execute.return_value = result or ToolResult(is_error=False, content="ok")
    return tool


# ---------------------------------------------------------------------------
# Tests: _dispatch_tool()
# ---------------------------------------------------------------------------

class TestDispatchTool:

    def test_unknown_tool_returns_error(self):
        """Dispatching an unregistered tool name returns is_error=True."""
        orch = make_orchestrator(tools={})
        result = orch._dispatch_tool("nonexistent_tool", {})

        assert result.is_error is True
        assert "unknown_tool" in result.content
        assert "nonexistent_tool" in result.content

    def test_unknown_tool_lists_available_tools(self):
        """Error message for unknown tool includes the list of available tools."""
        tool = make_tool("read_file")
        orch = make_orchestrator(tools={"read_file": tool})
        result = orch._dispatch_tool("run_lint", {})

        assert result.is_error is True
        assert "read_file" in result.content

    def test_tool_exception_returns_error_result(self):
        """When tool.execute() raises, _dispatch_tool wraps it in ToolResult(is_error=True)."""
        tool = make_tool("bad_tool")
        tool.execute.side_effect = RuntimeError("something exploded")
        orch = make_orchestrator(tools={"bad_tool": tool})

        result = orch._dispatch_tool("bad_tool", {"arg": "val"})

        assert result.is_error is True
        assert "tool_exception" in result.content
        assert "RuntimeError" in result.content
        assert "something exploded" in result.content

    def test_tool_exception_does_not_propagate(self):
        """_dispatch_tool must never raise â€” exceptions become ToolResult errors."""
        tool = make_tool("exploding_tool")
        tool.execute.side_effect = ValueError("boom")
        orch = make_orchestrator(tools={"exploding_tool": tool})

        # Should NOT raise â€” returns ToolResult instead
        result = orch._dispatch_tool("exploding_tool", {})
        assert result.is_error is True

    def test_valid_tool_returns_its_result(self):
        """_dispatch_tool returns exactly the ToolResult from tool.execute()."""
        expected = ToolResult(is_error=False, content="file content here")
        tool = make_tool("read_file", result=expected)
        orch = make_orchestrator(tools={"read_file": tool})

        result = orch._dispatch_tool("read_file", {"path": "foo.py"})

        assert result is expected
        assert result.is_error is False
        assert result.content == "file content here"

    def test_valid_tool_called_with_correct_inputs(self):
        """_dispatch_tool passes the inputs dict directly to tool.execute()."""
        tool = make_tool("read_file")
        orch = make_orchestrator(tools={"read_file": tool})
        inputs = {"path": "src/main.py"}

        orch._dispatch_tool("read_file", inputs)

        tool.execute.assert_called_once_with(inputs)

    def test_error_tool_result_propagated(self):
        """A tool returning is_error=True is forwarded unchanged."""
        error_result = ToolResult(is_error=True, content="permission_denied: /etc/shadow")
        tool = make_tool("read_file", result=error_result)
        orch = make_orchestrator(tools={"read_file": tool})

        result = orch._dispatch_tool("read_file", {"path": "/etc/shadow"})

        assert result.is_error is True
        assert "permission_denied" in result.content


# ---------------------------------------------------------------------------
# Tests: _extract_code_block()
# ---------------------------------------------------------------------------

class TestExtractCodeBlock:

    def test_extracts_python_tagged_block(self):
        """```python\\n...\\n``` block is extracted correctly."""
        text = "Here is the fix:\n```python\ndef hello():\n    pass\n```\nDone."
        result = Orchestrator._extract_code_block(text)
        assert result == "def hello():\n    pass\n"

    def test_returns_none_when_no_code_block(self):
        """Returns None when the response contains no fenced code block."""
        text = "lint_clean: no changes needed."
        result = Orchestrator._extract_code_block(text)
        assert result is None

    def test_plain_fence_no_language_tag(self):
        """Plain ``` block (no language tag) is also matched."""
        text = "Fix:\n```\nx = 1\n```"
        result = Orchestrator._extract_code_block(text)
        assert result == "x = 1\n"

    def test_empty_string_returns_none(self):
        """Empty string has no code block."""
        result = Orchestrator._extract_code_block("")
        assert result is None

    def test_multiline_code_block(self):
        """Multi-line code block content is returned in full."""
        code = "import os\n\ndef foo():\n    return os.getcwd()\n"
        text = f"```python\n{code}```"
        result = Orchestrator._extract_code_block(text)
        assert result == code

    def test_only_first_block_extracted(self):
        """When multiple code blocks exist, only the first match is returned."""
        text = "```python\nfirst = 1\n```\nThen:\n```python\nsecond = 2\n```"
        result = Orchestrator._extract_code_block(text)
        assert result == "first = 1\n"


# ---------------------------------------------------------------------------
# Tests: _extract_text()
# ---------------------------------------------------------------------------

class TestExtractText:

    def test_single_text_block(self):
        """A single text block is returned as-is."""
        blocks = [FakeContent(block_type="text", text="Hello world")]
        result = Orchestrator._extract_text(blocks)
        assert result == "Hello world"

    def test_multiple_text_blocks_joined(self):
        """Multiple text blocks are joined with newlines."""
        blocks = [
            FakeContent(block_type="text", text="First line"),
            FakeContent(block_type="text", text="Second line"),
        ]
        result = Orchestrator._extract_text(blocks)
        assert result == "First line\nSecond line"

    def test_tool_use_blocks_ignored(self):
        """Tool-use blocks (no .text attribute with meaningful content) are skipped."""
        blocks = [
            FakeContent(block_type="tool_use", name="read_file", inputs={"path": "x.py"}),
        ]
        # tool_use FakeContent has text="" â€” it won't contribute meaningful content
        result = Orchestrator._extract_text(blocks)
        # Only the empty string from the text attribute is joined (which becomes "")
        assert result == ""

    def test_mixed_blocks_extracts_only_text(self):
        """Only blocks with non-empty .text attributes contribute."""
        blocks = [
            FakeContent(block_type="text", text="Analysis done."),
            FakeContent(block_type="tool_use", name="run_lint", inputs={}, text=""),
            FakeContent(block_type="text", text="Here is the fix."),
        ]
        result = Orchestrator._extract_text(blocks)
        assert "Analysis done." in result
        assert "Here is the fix." in result

    def test_empty_content_list(self):
        """Empty content list returns empty string."""
        result = Orchestrator._extract_text([])
        assert result == ""

    def test_blocks_without_text_attribute_skipped(self):
        """Blocks lacking a .text attribute are silently skipped."""

        class NoTextBlock:
            type = "weird_type"
            # no .text attribute

        blocks = [NoTextBlock(), FakeContent(block_type="text", text="real content")]
        result = Orchestrator._extract_text(blocks)
        assert result == "real content"


# ---------------------------------------------------------------------------
# Tests: run() â€” abort guards
# ---------------------------------------------------------------------------

class TestRunAbortGuards:

    def test_max_iterations_raises_system_exit_3(self):
        """
        When the LLM always returns tool_use, the loop hits max_iterations and
        raises SystemExit(3).
        """
        orch = make_orchestrator(max_iterations=2, tools={})

        # The LLM will always return tool_use with no actual tool blocks,
        # so the loop just increments iteration each time.
        always_tool_use = FakeResponse(
            stop_reason="tool_use",
            content=[],  # no tool_use blocks â€” loop body appends empty tool_results
            usage=FakeUsage(10, 10),
        )

        with patch.object(orch._client.messages, "create", return_value=always_tool_use):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", "src/foo.py")

        assert exc_info.value.code == 3

    def test_budget_exceeded_raises_system_exit_5(self):
        """
        When the token budget is exceeded after the first LLM call, the next
        iteration raises SystemExit(5).
        """
        # Budget of 1 token â€” immediately exceeded after first response
        orch = make_orchestrator(max_tokens=1, tools={})

        # First call returns tool_use (so iteration continues), then budget check fires
        first_response = FakeResponse(
            stop_reason="tool_use",
            content=[],
            usage=FakeUsage(input_tokens=100, output_tokens=100),
        )

        with patch.object(orch._client.messages, "create", return_value=first_response):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", "src/foo.py")

        assert exc_info.value.code == 5

    def test_max_iterations_exit_code_is_exactly_3(self):
        """Exit code for max_iterations is 3, not 1 or any other value."""
        orch = make_orchestrator(max_iterations=1, tools={})
        always_tool_use = FakeResponse(stop_reason="tool_use", content=[], usage=FakeUsage())

        with patch.object(orch._client.messages, "create", return_value=always_tool_use):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", "src/foo.py")

        assert exc_info.value.code == 3

    def test_budget_exceeded_exit_code_is_exactly_5(self):
        """Exit code for budget exceeded is 5, not 1 or any other value."""
        orch = make_orchestrator(max_tokens=1, tools={})
        big_response = FakeResponse(
            stop_reason="tool_use",
            content=[],
            usage=FakeUsage(input_tokens=9999, output_tokens=9999),
        )

        with patch.object(orch._client.messages, "create", return_value=big_response):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", "src/foo.py")

        assert exc_info.value.code == 5


# ---------------------------------------------------------------------------
# Tests: run() â€” successful end_turn
# ---------------------------------------------------------------------------

class TestRunEndTurn:

    def test_end_turn_with_code_block_returns_success(self):
        """
        When LLM returns end_turn with a ```python ... ``` block, run() returns
        TaskResult(status="success") with a non-None output_path.
        """
        code = "x = 1\n"
        orch = make_orchestrator(original_content="# old content", tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text=f"```python\n{code}```")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            result = orch.run("lint_fix", "src/foo.py")

        assert result.status == "success"
        assert result.output_path is not None
        assert result.error is None

    def test_end_turn_with_code_block_calls_diff_writer(self):
        """DiffWriter.write() is called once when a code block is produced."""
        code = "x = 1\n"
        orch = make_orchestrator(original_content="# old", tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text=f"```python\n{code}```")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            orch.run("lint_fix", "src/foo.py")

        orch._diff_writer.write.assert_called_once()

    def test_end_turn_without_code_block_returns_success_no_output_path(self):
        """
        When LLM returns end_turn without a code block (e.g., lint_clean),
        run() returns TaskResult(status="success", output_path=None).
        """
        orch = make_orchestrator(original_content="# clean", tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="lint_clean: no changes needed")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            result = orch.run("lint_fix", "src/foo.py")

        assert result.status == "success"
        assert result.output_path is None
        assert result.error is None

    def test_end_turn_without_code_block_does_not_call_diff_writer(self):
        """DiffWriter.write() is NOT called when no code block is present."""
        orch = make_orchestrator(original_content="# clean", tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="lint_clean: no changes needed")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            orch.run("lint_fix", "src/foo.py")

        orch._diff_writer.write.assert_not_called()

    def test_end_turn_with_code_block_but_no_original_content_returns_no_diff(self):
        """
        When original_content is empty, no diff is written even if a code block
        was produced (nothing to diff against).
        """
        orch = make_orchestrator(original_content="", tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="```python\nx = 1\n```")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            result = orch.run("lint_fix", "src/foo.py")

        assert result.status == "success"
        assert result.output_path is None
        orch._diff_writer.write.assert_not_called()

    def test_end_turn_result_has_task_id(self):
        """The returned TaskResult has a non-empty task_id."""
        orch = make_orchestrator(tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="lint_clean")],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            result = orch.run("lint_fix", "src/foo.py")

        assert result.task_id
        assert len(result.task_id) > 0

    def test_end_turn_result_has_token_usage(self):
        """The returned TaskResult includes token usage."""
        orch = make_orchestrator(tools={})

        response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="lint_clean")],
            usage=FakeUsage(input_tokens=50, output_tokens=100),
        )

        with patch.object(orch._client.messages, "create", return_value=response):
            result = orch.run("lint_fix", "src/foo.py")

        assert result.token_usage.input_tokens == 50
        assert result.token_usage.output_tokens == 100


# ---------------------------------------------------------------------------
# Tests: run() â€” tool dispatch integration
# ---------------------------------------------------------------------------

class TestRunToolDispatch:

    def test_tool_use_response_triggers_tool_dispatch(self):
        """When LLM returns tool_use, the tool is dispatched and loop continues."""
        tool = make_tool("read_file", result=ToolResult(is_error=False, content="file contents"))
        orch = make_orchestrator(tools={"read_file": tool}, original_content="# orig")

        tool_use_block = FakeContent(
            block_type="tool_use",
            name="read_file",
            inputs={"path": "foo.py"},
            block_id="tu_001",
        )

        # First response: tool_use â†’ triggers dispatch
        # Second response: end_turn â†’ exits loop
        end_turn_response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="lint_clean: all good")],
            usage=FakeUsage(),
        )
        tool_use_response = FakeResponse(
            stop_reason="tool_use",
            content=[tool_use_block],
            usage=FakeUsage(),
        )

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_use_response
            return end_turn_response

        with patch.object(orch._client.messages, "create", side_effect=side_effect):
            result = orch.run("lint_fix", "src/foo.py")

        tool.execute.assert_called_once_with({"path": "foo.py"})
        assert result.status == "success"

    def test_unknown_tool_in_response_does_not_crash_loop(self):
        """An unknown tool name in an LLM tool_use block is handled gracefully."""
        orch = make_orchestrator(tools={}, original_content="")

        unknown_block = FakeContent(
            block_type="tool_use",
            name="nonexistent_tool",
            inputs={},
            block_id="tu_999",
        )

        tool_use_response = FakeResponse(
            stop_reason="tool_use",
            content=[unknown_block],
            usage=FakeUsage(),
        )
        end_turn_response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(block_type="text", text="done")],
            usage=FakeUsage(),
        )

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return tool_use_response if call_count == 1 else end_turn_response

        with patch.object(orch._client.messages, "create", side_effect=side_effect):
            result = orch.run("lint_fix", "src/foo.py")

        # Loop should have continued and eventually returned success
        assert result.status == "success"
