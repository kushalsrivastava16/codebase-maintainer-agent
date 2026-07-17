"""
Tests for agent/tools/lint_runner.py

Covers:
  1. lint_tool_missing when ruff is not on PATH
  2. lint_clean when both commands succeed with no output
  3. Violation text returned when ruff check reports violations (exit 1)
  4. Violation text returned when ruff format --check reports violations
  5. Both violations combined when both commands report issues
  6. ruff_unexpected_exit error when ruff check exits with code 2
  7. ruff_unexpected_exit error when ruff format exits with code 2
  8. is_error=False for clean lint and violations (violations are data, not errors)
  9. is_error=True for missing binary and unexpected exit codes
  10. Integration test: actually runs ruff on a real temp file with a known violation
      (skipped automatically if ruff is not installed)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, call, patch

import pytest

from agent.tools.lint_runner import LintRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int, stdout: str = "", stderr: str = "") -> CompletedProcess:
    """Build a CompletedProcess stand-in for subprocess.run mock return values."""
    proc = MagicMock(spec=CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# 1. lint_tool_missing when ruff is not on PATH
# ---------------------------------------------------------------------------

class TestRuffNotInstalled:
    def test_returns_lint_tool_missing(self):
        runner = LintRunner()
        with patch("shutil.which", return_value=None):
            result = runner.execute({"path": "some/file.py"})

        assert result.is_error is True
        assert result.content == "lint_tool_missing: ruff not found on PATH"

    # 9. is_error=True for missing binary
    def test_is_error_true_for_missing_binary(self):
        runner = LintRunner()
        with patch("shutil.which", return_value=None):
            result = runner.execute({"path": "."})
        assert result.is_error is True


# ---------------------------------------------------------------------------
# 2. lint_clean when both commands exit 0 with no output
# ---------------------------------------------------------------------------

class TestCleanLint:
    def test_returns_lint_clean_message(self):
        runner = LintRunner()
        clean_check = _make_proc(0, stdout="", stderr="")
        clean_fmt = _make_proc(0, stdout="", stderr="")

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[clean_check, clean_fmt]):
            result = runner.execute({"path": "src/"})

        assert result.content == "lint_clean: no violations found"

    # 8. is_error=False for clean lint
    def test_is_error_false_for_clean_lint(self):
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(0, stdout="", stderr=""),
                 _make_proc(0, stdout="", stderr=""),
             ]):
            result = runner.execute({"path": "src/"})
        assert result.is_error is False


# ---------------------------------------------------------------------------
# 3. Violation text when ruff check reports violations (exit 1, stdout content)
# ---------------------------------------------------------------------------

class TestCheckViolations:
    def test_returns_check_violations(self):
        runner = LintRunner()
        violation_output = "src/foo.py:3:1: F401 'os' imported but unused"
        check_proc = _make_proc(1, stdout=violation_output, stderr="")
        fmt_proc = _make_proc(0, stdout="", stderr="")

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[check_proc, fmt_proc]):
            result = runner.execute({"path": "src/foo.py"})

        assert result.content == violation_output
        assert result.is_error is False  # violations are data, not errors (req 8)

    def test_strips_surrounding_whitespace_from_output(self):
        runner = LintRunner()
        padded = "  src/foo.py:1:1: E501 line too long  \n"
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(1, stdout=padded, stderr=""),
                 _make_proc(0, stdout="", stderr=""),
             ]):
            result = runner.execute({"path": "src/foo.py"})
        assert result.content == padded.strip()


# ---------------------------------------------------------------------------
# 4. Violation text when ruff format --check reports violations
# ---------------------------------------------------------------------------

class TestFormatViolations:
    def test_returns_format_violations(self):
        runner = LintRunner()
        fmt_output = "Would reformat src/bar.py"
        check_proc = _make_proc(0, stdout="", stderr="")
        fmt_proc = _make_proc(1, stdout=fmt_output, stderr="")

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[check_proc, fmt_proc]):
            result = runner.execute({"path": "src/bar.py"})

        assert result.content == fmt_output
        assert result.is_error is False


# ---------------------------------------------------------------------------
# 5. Both violations combined when both commands report issues
# ---------------------------------------------------------------------------

class TestBothViolations:
    def test_returns_combined_output(self):
        runner = LintRunner()
        check_out = "src/baz.py:2:1: F401 'sys' imported but unused"
        fmt_out = "Would reformat src/baz.py"

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(1, stdout=check_out, stderr=""),
                 _make_proc(1, stdout=fmt_out, stderr=""),
             ]):
            result = runner.execute({"path": "src/baz.py"})

        assert check_out in result.content
        assert fmt_out in result.content
        assert result.is_error is False

    def test_combined_output_joined_by_newline(self):
        runner = LintRunner()
        check_out = "check violation"
        fmt_out = "format violation"

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(1, stdout=check_out, stderr=""),
                 _make_proc(1, stdout=fmt_out, stderr=""),
             ]):
            result = runner.execute({"path": "."})

        assert result.content == f"{check_out}\n{fmt_out}"


# ---------------------------------------------------------------------------
# 6. ruff_unexpected_exit error when ruff check exits with code 2
# ---------------------------------------------------------------------------

class TestUnexpectedExitCheck:
    def test_returns_error_on_check_exit_2(self):
        runner = LintRunner()
        stderr_text = "ruff internal error details"
        check_proc = _make_proc(2, stdout="", stderr=stderr_text)

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", return_value=check_proc):
            result = runner.execute({"path": "src/"})

        assert result.is_error is True
        assert "ruff_unexpected_exit" in result.content
        assert "code=2" in result.content
        assert stderr_text in result.content

    # 9. is_error=True for unexpected exit codes
    def test_is_error_true_for_unexpected_exit(self):
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", return_value=_make_proc(2, stderr="err")):
            result = runner.execute({"path": "."})
        assert result.is_error is True

    def test_does_not_run_format_after_check_error(self):
        """When ruff check exits unexpectedly, format should not be called."""
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", return_value=_make_proc(2)) as mock_run:
            runner.execute({"path": "."})
        # Only one subprocess.run call should have been made (the check one)
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# 7. ruff_unexpected_exit error when ruff format exits with code 2
# ---------------------------------------------------------------------------

class TestUnexpectedExitFormat:
    def test_returns_error_on_format_exit_2(self):
        runner = LintRunner()
        stderr_text = "format internal error"
        check_proc = _make_proc(0, stdout="", stderr="")
        fmt_proc = _make_proc(2, stdout="", stderr=stderr_text)

        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[check_proc, fmt_proc]):
            result = runner.execute({"path": "src/"})

        assert result.is_error is True
        assert "ruff_unexpected_exit" in result.content
        assert "code=2" in result.content
        assert stderr_text in result.content

    def test_is_error_true_for_format_unexpected_exit(self):
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(0, stdout="", stderr=""),
                 _make_proc(2, stderr="err"),
             ]):
            result = runner.execute({"path": "."})
        assert result.is_error is True


# ---------------------------------------------------------------------------
# Subprocess call arguments verification
# ---------------------------------------------------------------------------

class TestSubprocessCalls:
    def test_check_called_with_correct_args(self):
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(0, stdout="", stderr=""),
                 _make_proc(0, stdout="", stderr=""),
             ]) as mock_run:
            runner.execute({"path": "myfile.py"})

        first_call_args = mock_run.call_args_list[0]
        assert first_call_args == call(
            ["ruff", "check", "myfile.py"],
            capture_output=True,
            text=True,
        )

    def test_format_called_with_correct_args(self):
        runner = LintRunner()
        with patch("shutil.which", return_value="/usr/bin/ruff"), \
             patch("subprocess.run", side_effect=[
                 _make_proc(0, stdout="", stderr=""),
                 _make_proc(0, stdout="", stderr=""),
             ]) as mock_run:
            runner.execute({"path": "myfile.py"})

        second_call_args = mock_run.call_args_list[1]
        assert second_call_args == call(
            ["ruff", "format", "--check", "myfile.py"],
            capture_output=True,
            text=True,
        )


# ---------------------------------------------------------------------------
# Tool protocol compliance
# ---------------------------------------------------------------------------

class TestToolProtocol:
    def test_has_required_attributes(self):
        runner = LintRunner()
        assert hasattr(runner, "name")
        assert hasattr(runner, "description")
        assert hasattr(runner, "input_schema")
        assert callable(runner.execute)

    def test_name_is_run_lint(self):
        assert LintRunner.name == "run_lint"

    def test_input_schema_requires_path(self):
        schema = LintRunner.input_schema
        assert "path" in schema["properties"]
        assert "path" in schema["required"]


# ---------------------------------------------------------------------------
# 10. Integration test: runs ruff on a real temp file with a known violation
#     Skipped automatically if ruff is not installed on the system.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    shutil.which("ruff") is None,
    reason="ruff not installed — skipping integration test",
)
class TestIntegration:
    def test_detects_unused_import_violation(self, tmp_path: Path):
        """ruff check should flag the unused import in a real file."""
        bad_file = tmp_path / "bad_code.py"
        # Write Python with a known lint violation: unused import
        bad_file.write_text("import os\n\nx = 1\n", encoding="utf-8")

        runner = LintRunner()
        result = runner.execute({"path": str(bad_file)})

        assert result.is_error is False
        # ruff reports F401 for unused imports
        assert "F401" in result.content or "bad_code.py" in result.content

    def test_clean_file_returns_lint_clean(self, tmp_path: Path):
        """A well-formatted file with no violations should return lint_clean."""
        good_file = tmp_path / "good_code.py"
        good_file.write_text("x = 1\n", encoding="utf-8")

        runner = LintRunner()
        result = runner.execute({"path": str(good_file)})

        assert result.is_error is False
        assert result.content == "lint_clean: no violations found"
