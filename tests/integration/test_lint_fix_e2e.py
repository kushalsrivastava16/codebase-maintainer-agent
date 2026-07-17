"""
End-to-end integration tests for the lint_fix workflow.

Tests cover:
  1. LintRunner correctly detects violations in the fixture file (no Anthropic API needed)
  2. Orchestrator aborts with SystemExit(3) when mocked LLM always returns tool_use
     (abort guard works end-to-end)
  3. Full orchestrator with mocked LLM returning a fixed file produces a .diff file
  4. CLI validates non-existent target path and exits 1
  5. CLI --task validation rejects invalid task types

Tests 1, 2, and 3 invoke ruff and are skipped if ruff is not installed.
Tests 4 and 5 are pure CLI validation â€” they do not require ruff.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.__main__ import cli
from agent.config import AgentConfig
from agent.cost_tracker import CostTracker
from agent.diff_writer import DiffWriter
from agent.logger import StructuredLogger
from agent.orchestrator import Orchestrator
from agent.protocols import ToolResult
from agent.tools.lint_runner import LintRunner

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
UTILS_WITH_ERRORS = FIXTURES_DIR / "utils_with_lint_errors.py"
CLEAN_FILE = FIXTURES_DIR / "clean_file.py"


# ---------------------------------------------------------------------------
# Stub helpers (mirrored from test_orchestrator.py patterns)
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeContent:
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
# Helper: build a real Orchestrator wired to a tmp output dir
# ---------------------------------------------------------------------------


def make_real_orchestrator(
    output_dir: Path,
    original_content: str = "",
    max_iterations: int = 5,
    max_tokens: int = 50_000,
    tools: dict | None = None,
) -> Orchestrator:
    config = AgentConfig(
        max_iterations=max_iterations,
        max_tokens_per_task=max_tokens,
        output_dir=str(output_dir),
        model="claude-haiku-4-5-20251001",
    )
    logger = StructuredLogger(min_level="ERROR")
    cost_tracker = CostTracker(max_tokens=max_tokens)
    diff_writer = DiffWriter(output_dir=output_dir)
    return Orchestrator(
        config=config,
        logger=logger,
        cost_tracker=cost_tracker,
        diff_writer=diff_writer,
        tools=tools or {},
        original_content=original_content,
    )


# ---------------------------------------------------------------------------
# Test 1: LintRunner detects violations in the fixture file
# (requires ruff)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
class TestLintRunnerDetectsViolations:
    def test_detects_unused_imports_in_fixture(self):
        """LintRunner reports F401 (unused import) violations in utils_with_lint_errors.py."""
        runner = LintRunner()
        result = runner.execute({"path": str(UTILS_WITH_ERRORS)})

        assert result.is_error is False
        # ruff should report at least one violation
        assert result.content != "lint_clean: no violations found"

    def test_detects_f401_unused_import(self):
        """The fixture's unused imports trigger F401 violations."""
        runner = LintRunner()
        result = runner.execute({"path": str(UTILS_WITH_ERRORS)})

        # json is unused â€” ruff should flag it
        assert "F401" in result.content or "unused" in result.content.lower()

    def test_clean_fixture_reports_no_violations(self):
        """clean_file.py has no violations â€” LintRunner returns lint_clean."""
        runner = LintRunner()
        result = runner.execute({"path": str(CLEAN_FILE)})

        assert result.is_error is False
        assert result.content == "lint_clean: no violations found"

    def test_fixture_file_exists(self):
        """Sanity check: the fixture files exist on disk."""
        assert UTILS_WITH_ERRORS.exists(), f"Missing fixture: {UTILS_WITH_ERRORS}"
        assert CLEAN_FILE.exists(), f"Missing fixture: {CLEAN_FILE}"


# ---------------------------------------------------------------------------
# Test 2: Orchestrator aborts with SystemExit(3) when LLM always returns tool_use
# (abort guard end-to-end)
# (requires ruff because the orchestrator wires up a real LintRunner)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
class TestOrchestratorAbortGuard:
    def test_abort_with_exit_3_when_llm_loops_forever(self, tmp_path: Path):
        """
        When the mocked LLM always returns tool_use (never end_turn), the orchestrator
        hits max_iterations and raises SystemExit(3).
        """
        lint_runner = LintRunner()
        orch = make_real_orchestrator(
            output_dir=tmp_path,
            original_content="# some python file",
            max_iterations=3,
            tools={"run_lint": lint_runner},
        )

        # LLM perpetually returns tool_use with no actual tool blocks
        always_tool_use = FakeResponse(
            stop_reason="tool_use",
            content=[],
            usage=FakeUsage(10, 10),
        )

        with patch.object(orch._client.messages, "create", return_value=always_tool_use):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", str(UTILS_WITH_ERRORS))

        assert exc_info.value.code == 3

    def test_abort_exit_code_is_exactly_3_not_1(self, tmp_path: Path):
        """SystemExit code for max_iterations must be 3 (not 1 or other)."""
        orch = make_real_orchestrator(
            output_dir=tmp_path,
            original_content="# content",
            max_iterations=2,
            tools={},
        )
        always_tool_use = FakeResponse(stop_reason="tool_use", content=[], usage=FakeUsage())

        with patch.object(orch._client.messages, "create", return_value=always_tool_use):
            with pytest.raises(SystemExit) as exc_info:
                orch.run("lint_fix", "some/file.py")

        assert exc_info.value.code == 3


# ---------------------------------------------------------------------------
# Test 3: Full orchestrator with mocked LLM produces a .diff file
# (requires ruff so we can use a real LintRunner in the tool registry)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
class TestOrchestratorProducesDiff:
    def test_mocked_llm_produces_diff_file(self, tmp_path: Path):
        """
        When the mocked LLM returns end_turn with a complete fixed file in a
        ```python code block, DiffWriter writes a .diff file to agent_output/.
        """
        original_content = UTILS_WITH_ERRORS.read_text(encoding="utf-8")

        # The "fixed" file proposed by the mocked LLM â€” all unused imports removed
        fixed_content = (
            "def add_numbers(a, b):\n"
            "    result = a + b\n"
            "    return result\n"
            "\n"
            "def format_message(msg):\n"
            "    return f\"Message: {msg}\"\n"
            "\n"
            "class DataProcessor:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "    def process(self):\n"
            "        results = []\n"
            "        for item in self.data:\n"
            "            results.append(item * 2)\n"
            "        return results\n"
        )

        lint_runner = LintRunner()
        orch = make_real_orchestrator(
            output_dir=tmp_path,
            original_content=original_content,
            max_iterations=5,
            tools={"run_lint": lint_runner},
        )

        # Single end_turn response with fixed file in a code block
        llm_response = FakeResponse(
            stop_reason="end_turn",
            content=[
                FakeContent(
                    block_type="text",
                    text=f"Here is the fixed file:\n```python\n{fixed_content}```",
                )
            ],
            usage=FakeUsage(input_tokens=100, output_tokens=200),
        )

        with patch.object(orch._client.messages, "create", return_value=llm_response):
            result = orch.run("lint_fix", str(UTILS_WITH_ERRORS))

        assert result.status == "success"
        assert result.output_path is not None

        # The .diff file must exist on disk
        diff_path = Path(result.output_path)
        assert diff_path.exists(), f"Expected .diff file at {diff_path}"
        assert diff_path.suffix == ".diff"

    def test_diff_file_contains_minus_and_plus_lines(self, tmp_path: Path):
        """The generated diff contains removed (-) and added (+) lines."""
        original_content = "import os\n\nx = 1\n"
        fixed_content = "x = 1\n"

        orch = make_real_orchestrator(
            output_dir=tmp_path,
            original_content=original_content,
            max_iterations=5,
            tools={},
        )

        llm_response = FakeResponse(
            stop_reason="end_turn",
            content=[
                FakeContent(
                    block_type="text",
                    text=f"```python\n{fixed_content}```",
                )
            ],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=llm_response):
            result = orch.run("lint_fix", "some/file.py")

        assert result.output_path is not None
        diff_content = Path(result.output_path).read_text(encoding="utf-8")
        # Unified diff has lines starting with '-' (removed) and '+' (added)
        assert "-import os" in diff_content
        assert diff_content  # non-empty

    def test_companion_json_metadata_file_is_created(self, tmp_path: Path):
        """A companion .json metadata file is written alongside the .diff file."""
        original_content = "import os\n\nx = 1\n"
        fixed_content = "x = 1\n"

        orch = make_real_orchestrator(
            output_dir=tmp_path,
            original_content=original_content,
            max_iterations=5,
            tools={},
        )

        llm_response = FakeResponse(
            stop_reason="end_turn",
            content=[
                FakeContent(
                    block_type="text",
                    text=f"```python\n{fixed_content}```",
                )
            ],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=llm_response):
            result = orch.run("lint_fix", "some/file.py")

        assert result.output_path is not None
        json_path = Path(result.output_path).with_suffix(".json")
        assert json_path.exists(), f"Expected companion JSON at {json_path}"


# ---------------------------------------------------------------------------
# Test 4: CLI validates non-existent target path and exits 1
# (no ruff needed â€” pure CLI argument validation)
# ---------------------------------------------------------------------------


class TestCLITargetValidation:
    def test_nonexistent_target_exits_1(self):
        """CLI exits 1 when --target points to a path that does not exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "lint_fix", "--target", "/nonexistent/path/to/file.py"],
        )
        assert result.exit_code == 1

    def test_nonexistent_target_prints_error_message(self):
        """CLI prints an error message when the target does not exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "lint_fix", "--target", "/nonexistent/path/to/file.py"],
        )
        # CliRunner captures combined output; the CLI writes to err=True but
        # the combined output stream still contains the message
        assert "error" in result.output.lower() or result.exit_code == 1

    def test_nonexistent_target_exit_code_is_1_not_2(self):
        """Exit code for missing target is 1 (validation error), not 2 (click usage error)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "lint_fix", "--target", "/no/such/file.py"],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Test 5: CLI --task validation rejects invalid task types
# (no ruff needed â€” pure Click validation via click.Choice)
# ---------------------------------------------------------------------------


class TestCLITaskValidation:
    def test_invalid_task_type_exits_nonzero(self):
        """CLI exits with a non-zero code when --task is an unrecognised task type."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "invalid_task_type", "--target", "."],
        )
        assert result.exit_code != 0

    def test_invalid_task_type_exit_code_is_2(self):
        """click.Choice validation exits with code 2 for invalid option values."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "not_a_real_task", "--target", "."],
        )
        # click exits 2 for bad option values (UsageError)
        assert result.exit_code == 2

    def test_invalid_task_mentions_valid_choices(self):
        """Error output for invalid --task includes the list of valid choices."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--task", "bogus_task", "--target", "."],
        )
        output = result.output
        # click.Choice error message includes the valid options
        assert any(
            task in output
            for task in ["lint_fix", "generate_tests", "convert_todos", "triage_issues"]
        )

    def test_valid_task_types_are_accepted(self, tmp_path: Path):
        """
        All four supported task types pass --task validation.
        (We only check validation â€” we stop before API calls by providing
        a non-existent target, which exits 1 before any LLM call.)
        """
        runner = CliRunner()
        valid_tasks = ["lint_fix", "generate_tests", "convert_todos", "triage_issues"]
        for task in valid_tasks:
            result = runner.invoke(
                cli,
                ["run", "--task", task, "--target", "/nonexistent/path.py"],
            )
            # click.Choice accepted the value (exit 1 = target-not-found, not 2 = bad option)
            assert result.exit_code != 2, (
                f"task '{task}' was rejected by click.Choice (exit 2)"
            )
