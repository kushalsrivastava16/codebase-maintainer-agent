"""
Coverage Reader Tool for the Codebase Maintainer Agent.

WHY this tool exists:
  Test generation is most valuable when focused on untested code. This tool
  reads a pytest-cov JSON coverage report and extracts uncovered lines for a
  specific file, giving the LLM the exact context it needs to write targeted
  tests rather than guessing what is already covered.

WHY parse JSON instead of reading the .coverage SQLite file directly?
  The pytest-cov JSON report (--cov-report=json) is a stable, documented
  format that works across all pytest-cov versions. The .coverage file is a
  SQLite database whose schema changes between versions. Parsing JSON is also
  simpler and avoids a runtime dependency on coverage.py internals.

WHY return missing lines rather than covered lines?
  The LLM's job is to write tests for uncovered code. Returning covered lines
  would require the LLM to subtract them from the full line list — an error-
  prone extra step. Returning missing lines directly gives the model exactly
  what it needs to act.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.protocols import ToolResult


class CoverageReader:
    """
    Reads a pytest-cov JSON coverage report and returns uncovered line numbers
    for a specified source file.

    Returns a formatted string:
        missing_lines: 12, 34, 56, 78
    or:
        coverage_complete: all lines covered

    If the file is not found in the coverage report:
        file_not_in_report: <path>
    """

    name = "read_coverage"
    description = (
        "Read a pytest-cov JSON coverage report and return uncovered line numbers "
        "for a specific source file. Use this to identify which lines lack test coverage "
        "before generating new tests."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": (
                    "Path to the pytest-cov JSON report file "
                    "(generated with --cov-report=json)."
                ),
            },
            "source_file": {
                "type": "string",
                "description": (
                    "Path to the source file to look up in the coverage report. "
                    "Can be absolute or relative — the tool matches by suffix."
                ),
            },
        },
        "required": ["report_path", "source_file"],
    }

    def execute(self, inputs: dict) -> ToolResult:
        report_path_str: str = inputs.get("report_path", "")
        source_file_str: str = inputs.get("source_file", "")

        report_path = Path(report_path_str)
        if not report_path.exists():
            return ToolResult(
                is_error=True,
                content=f"report_not_found: {report_path_str}",
            )

        try:
            raw = report_path.read_text(encoding="utf-8")
        except PermissionError:
            return ToolResult(
                is_error=True,
                content=f"permission_denied: {report_path_str}",
            )

        try:
            report = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolResult(
                is_error=True,
                content=f"invalid_json: {exc}",
            )

        # The coverage JSON has a top-level "files" dict keyed by file path.
        files_section: dict = report.get("files", {})
        if not files_section:
            return ToolResult(
                is_error=True,
                content="coverage_report_empty: no 'files' section found",
            )

        # Match by suffix — coverage reports may use absolute paths while the
        # caller uses relative paths or vice versa.
        source_suffix = Path(source_file_str).as_posix()
        matched_key: str | None = None

        for key in files_section:
            if Path(key).as_posix().endswith(source_suffix):
                matched_key = key
                break
            # Also try matching just the filename as a fallback
            if Path(key).name == Path(source_file_str).name:
                matched_key = key
                # Don't break — a more specific suffix match above is preferred

        if matched_key is None:
            return ToolResult(
                is_error=False,
                content=f"file_not_in_report: {source_file_str}",
            )

        file_data: dict = files_section[matched_key]
        missing_lines: list[int] = file_data.get("missing_lines", [])

        if not missing_lines:
            return ToolResult(
                is_error=False,
                content="coverage_complete: all lines covered",
            )

        formatted = ", ".join(str(ln) for ln in sorted(missing_lines))
        summary = file_data.get("summary", {})
        covered_pct = summary.get("percent_covered", None)
        pct_str = f" ({covered_pct:.1f}% covered)" if covered_pct is not None else ""

        return ToolResult(
            is_error=False,
            content=f"missing_lines{pct_str}: {formatted}",
        )
