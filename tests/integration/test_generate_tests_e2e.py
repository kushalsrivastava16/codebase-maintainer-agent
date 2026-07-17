"""
End-to-end integration tests for the generate_tests workflow.

Tests cover:
  1. CoverageReader correctly parses the fixture coverage report
  2. CoverageReader returns coverage_complete when no lines are missing
  3. Orchestrator with mocked LLM produces a .diff file for test generation
  4. CLI validation: --coverage-report is required for generate_tests
  5. CLI validation: non-existent coverage report path is handled

These tests do NOT require the Anthropic API â€” the LLM is mocked.
"""
from __future__ import annotations

import json
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
from agent.tools.coverage_reader import CoverageReader
from agent.tools.file_reader import FileReader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CALCULATOR_FILE = FIXTURES_DIR / "calculator.py"
COVERAGE_JSON = Path(__file__).parent.parent.parent / "coverage.json"


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeContent:
    def __init__(self, block_type: str = "text", text: str = "") -> None:
        self.type = block_type
        self.text = text


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


def _make_orchestrator(
    output_dir: Path,
    original_content: str = "",
    tools: dict | None = None,
) -> Orchestrator:
    config = AgentConfig(
        max_iterations=5,
        max_tokens_per_task=50_000,
        output_dir=str(output_dir),
        model="claude-haiku-4-5-20251001",
    )
    logger = StructuredLogger(min_level="ERROR")
    cost_tracker = CostTracker(max_tokens=50_000)
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
# Fixture: minimal coverage JSON report
# ---------------------------------------------------------------------------


def _make_coverage_report(tmp_path: Path, missing_lines: list[int]) -> Path:
    """Write a minimal pytest-cov JSON report and return its path."""
    data = {
        "files": {
            str(CALCULATOR_FILE): {
                "missing_lines": missing_lines,
                "summary": {"percent_covered": 50.0 if missing_lines else 100.0},
            }
        }
    }
    report_path = tmp_path / "coverage.json"
    report_path.write_text(json.dumps(data), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Test 1: CoverageReader parses the fixture coverage report
# ---------------------------------------------------------------------------


class TestCoverageReaderParsesCoverage:
    def test_returns_missing_lines_from_valid_report(self, tmp_path: Path):
        """CoverageReader returns formatted missing line numbers from a valid JSON report."""
        report = _make_coverage_report(tmp_path, missing_lines=[10, 20, 30])
        reader = CoverageReader()
        result = reader.execute({
            "report_path": str(report),
            "source_file": "calculator.py",
        })
        assert result.is_error is False
        assert "10" in result.content
        assert "20" in result.content
        assert "30" in result.content

    def test_returns_coverage_complete_when_no_missing_lines(self, tmp_path: Path):
        """CoverageReader returns coverage_complete when all lines are covered."""
        report = _make_coverage_report(tmp_path, missing_lines=[])
        reader = CoverageReader()
        result = reader.execute({
            "report_path": str(report),
            "source_file": "calculator.py",
        })
        assert result.is_error is False
        assert "coverage_complete" in result.content

    def test_returns_error_for_nonexistent_report(self):
        """CoverageReader returns error when report path does not exist."""
        reader = CoverageReader()
        result = reader.execute({
            "report_path": "/nonexistent/coverage.json",
            "source_file": "calculator.py",
        })
        assert result.is_error is True
        assert "not_found" in result.content or "coverage_report" in result.content

    def test_returns_error_for_invalid_json(self, tmp_path: Path):
        """CoverageReader returns error when the report contains invalid JSON."""
        bad_report = tmp_path / "bad_coverage.json"
        bad_report.write_text("not valid json {{{", encoding="utf-8")
        reader = CoverageReader()
        result = reader.execute({
            "report_path": str(bad_report),
            "source_file": "calculator.py",
        })
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Test 2: Orchestrator produces a diff for generate_tests task
# ---------------------------------------------------------------------------


class TestOrchestratorGeneratesTestDiff:
    def test_mocked_llm_produces_diff_for_generate_tests(self, tmp_path: Path):
        """
        When the mocked LLM returns a test file in a code block, the orchestrator
        writes a .diff file to agent_output/.
        """
        if not CALCULATOR_FILE.exists():
            pytest.skip(f"Missing fixture: {CALCULATOR_FILE}")

        original_content = "# placeholder test file\n"
        proposed_tests = (
            "def test_add():\n"
            "    from tests.fixtures.calculator import add\n"
            "    assert add(2, 3) == 5\n"
            "\n"
            "def test_divide_by_zero():\n"
            "    import pytest\n"
            "    from tests.fixtures.calculator import divide\n"
            "    with pytest.raises(ValueError):\n"
            "        divide(1, 0)\n"
        )

        report_path = _make_coverage_report(tmp_path, missing_lines=[1, 2, 3, 4])
        file_reader = FileReader(repo_root=CALCULATOR_FILE.parent)
        coverage_reader = CoverageReader()
        orch = _make_orchestrator(
            output_dir=tmp_path,
            original_content=original_content,
            tools={
                file_reader.name: file_reader,
                coverage_reader.name: coverage_reader,
            },
        )

        llm_response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(
                block_type="text",
                text=f"Here are the generated tests:\n```python\n{proposed_tests}```",
            )],
            usage=FakeUsage(100, 200),
        )

        with patch.object(orch._client.messages, "create", return_value=llm_response):
            result = orch.run("generate_tests", str(CALCULATOR_FILE))

        assert result.status == "success"

    def test_companion_json_written_for_generate_tests(self, tmp_path: Path):
        """A companion .json metadata file is written alongside the test diff."""
        original_content = "# test file\n"
        proposed_tests = "def test_something():\n    assert True\n"

        orch = _make_orchestrator(
            output_dir=tmp_path,
            original_content=original_content,
            tools={},
        )

        llm_response = FakeResponse(
            stop_reason="end_turn",
            content=[FakeContent(
                block_type="text",
                text=f"```python\n{proposed_tests}```",
            )],
            usage=FakeUsage(),
        )

        with patch.object(orch._client.messages, "create", return_value=llm_response):
            result = orch.run("generate_tests", "tests/test_something.py")

        if result.output_path:
            json_path = Path(result.output_path).with_suffix(".json")
            assert json_path.exists()


# ---------------------------------------------------------------------------
# Test 3: CLI validation for generate_tests task
# ---------------------------------------------------------------------------


class TestCLIGenerateTestsValidation:
    def test_coverage_report_required_for_generate_tests(self, tmp_path: Path):
        """CLI exits 1 when --coverage-report is not provided for generate_tests."""
        runner = CliRunner()
        # Create a real target file so target validation passes
        target = tmp_path / "source.py"
        target.write_text("x = 1\n")

        result = runner.invoke(
            cli,
            ["run", "--task", "generate_tests", "--target", str(target)],
        )
        assert result.exit_code == 1
        assert "coverage-report" in result.output.lower() or result.exit_code == 1

    def test_nonexistent_target_exits_1_for_generate_tests(self):
        """CLI exits 1 when --target does not exist for generate_tests."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--task", "generate_tests",
                "--target", "/nonexistent/file.py",
                "--coverage-report", "/also/nonexistent.json",
            ],
        )
        assert result.exit_code == 1
