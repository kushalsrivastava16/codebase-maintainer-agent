"""
Lint Runner Tool for the Codebase Maintainer Agent.

WHY use ruff instead of flake8/pylint?
  ruff is 10-100x faster than legacy Python linters, handles both style
  (PEP 8) and type-adjacent checks, and is the de-facto standard for new
  Python projects in 2024. It also includes a formatter (like black) so we
  get style + format checking in a single binary.

WHY run two separate subprocess calls (check + format --check)?
  ruff check catches code style violations (unused imports, undefined names,
  etc.). ruff format --check catches formatting violations (line length,
  whitespace, etc.). They have separate exit codes and output formats.
  Running them separately gives the LLM a clear picture of both issue types.

WHY use shutil.which() before subprocess.run()?
  subprocess.run() with a missing binary raises FileNotFoundError, which would
  propagate as an unhandled exception and crash the agent loop. Checking first
  allows us to return a clean ToolResult(is_error=True) that the LLM can reason
  about, e.g., asking the user to install ruff.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from agent.protocols import ToolResult


class LintRunner:
    """
    Runs ruff check and ruff format --check on a file or directory.

    Returns the combined stdout from both commands. If both commands report
    no issues, returns the special message "lint_clean: no violations found"
    so the LLM knows it does not need to propose any changes.
    """

    name = "run_lint"
    description = (
        "Run ruff style and format checks on a Python file or directory. "
        "Returns violation messages with line numbers, or 'lint_clean: no violations found'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the Python file or directory to check",
            }
        },
        "required": ["path"],
    }

    def execute(self, inputs: dict) -> ToolResult:
        # Fast-fail if ruff is not installed — gives the LLM an actionable error
        if shutil.which("ruff") is None:
            return ToolResult(
                is_error=True,
                content="lint_tool_missing: ruff not found on PATH",
            )

        target = inputs.get("path", "")
        output_lines: list[str] = []

        # Run ruff check (style violations: unused imports, undefined names, etc.)
        # ruff check exit codes:
        #   0 = no violations
        #   1 = violations found
        #   2 = internal error or invalid usage
        check_result = subprocess.run(
            ["ruff", "check", target],
            capture_output=True,
            text=True,
        )
        if check_result.returncode not in (0, 1):
            return ToolResult(
                is_error=True,
                content=(
                    f"ruff_unexpected_exit: code={check_result.returncode}\n"
                    f"{check_result.stderr}"
                ),
            )
        if check_result.stdout.strip():
            output_lines.append(check_result.stdout.strip())

        # Run ruff format --check (formatting violations: indentation, line length, etc.)
        # ruff format --check exit codes:
        #   0 = no formatting changes needed
        #   1 = formatting changes would be made
        #   2 = internal error
        fmt_result = subprocess.run(
            ["ruff", "format", "--check", target],
            capture_output=True,
            text=True,
        )
        if fmt_result.returncode not in (0, 1):
            return ToolResult(
                is_error=True,
                content=(
                    f"ruff_unexpected_exit: code={fmt_result.returncode}\n"
                    f"{fmt_result.stderr}"
                ),
            )
        if fmt_result.stdout.strip():
            output_lines.append(fmt_result.stdout.strip())

        if not output_lines:
            # Both commands reported clean — tell the LLM nothing needs fixing
            return ToolResult(is_error=False, content="lint_clean: no violations found")

        return ToolResult(is_error=False, content="\n".join(output_lines))
