"""
TODO Scanner Tool for the Codebase Maintainer Agent.

WHY this tool exists:
  The LLM needs to know which TODOs exist in source files before it can propose
  converting them into GitHub issues. This tool extracts TODO/FIXME/HACK/NOTE
  comments from Python files (a single file or a directory tree) and returns
  them as structured text so the LLM can reason about priority, duplicates, and
  the correct issue template to use.

WHY return structured text rather than a list?
  Tool results are passed back to the LLM as plain strings. Structured text
  (filepath:line: comment) is more token-efficient and readable for the model
  than JSON, and the LLM does not need to parse the result programmatically.

WHY scan multiple patterns (TODO, FIXME, HACK, NOTE)?
  All four are conventional Python comment markers with slightly different
  semantics. Including them all gives the LLM full context about technical
  debt in the file.

WHY classify short TODOs as ambiguous?
  TODO comments with fewer than 5 non-whitespace words after the marker give
  the LLM too little context to propose a meaningful implementation. Skipping
  them avoids wasting tokens on vague items and reduces the prompt injection
  surface (Requirement 8.4).
"""
from __future__ import annotations

import re
from pathlib import Path

from agent.protocols import ToolResult

# Matches TODO, FIXME, HACK, or NOTE comments (case-insensitive).
# Captures the keyword and the rest of the comment text.
TODO_PATTERN = re.compile(
    r"#\s*(TODO|FIXME|HACK|NOTE)[:\s]+(.*)",
    re.IGNORECASE,
)

# Path components that indicate generated/vendored directories to skip.
EXCLUDE_COMPONENTS: frozenset[str] = frozenset(
    {".venv", "site-packages", "__pycache__", ".git"}
)

MAX_FILE_BYTES: int = 100 * 1024  # 100 KB — same limit as FileReader
MIN_WORDS: int = 5  # TODOs with fewer words are classified as ambiguous


def _is_excluded(path: Path) -> bool:
    """Return True if any part of *path* is an excluded directory component."""
    return bool(set(path.parts) & EXCLUDE_COMPONENTS)


def _scan_file(path: Path) -> list[str]:
    """
    Scan a single .py file and return formatted TODO findings.

    Each finding is one of:
        <path>:<lineno>: [KEYWORD] <comment_text>
        <path>:<lineno>: [KEYWORD] [AMBIGUOUS] <comment_text>
    """
    try:
        raw_bytes = path.read_bytes()
    except (PermissionError, OSError):
        return []

    # Truncate large files — same policy as FileReader
    if len(raw_bytes) > MAX_FILE_BYTES:
        raw_bytes = raw_bytes[:MAX_FILE_BYTES]

    content = raw_bytes.decode("utf-8", errors="replace")
    findings: list[str] = []

    for line_no, line in enumerate(content.splitlines(), start=1):
        match = TODO_PATTERN.search(line)
        if not match:
            continue

        keyword = match.group(1).upper()
        comment_text = match.group(2).strip()
        words = [w for w in comment_text.split() if w]

        if len(words) < MIN_WORDS:
            findings.append(f"{path}:{line_no}: [{keyword}] [AMBIGUOUS] {comment_text}")
        else:
            findings.append(f"{path}:{line_no}: [{keyword}] {comment_text}")

    return findings


class TodoScanner:
    """
    Scans a Python file *or* directory tree for TODO/FIXME/HACK/NOTE comments.

    When given a directory, recursively finds all .py files, skipping paths
    whose components include .venv, site-packages, __pycache__, or .git.

    TODOs with fewer than 5 non-whitespace words after the marker are tagged
    [AMBIGUOUS] in the output (Requirement 8.4).

    Returns a formatted list of findings, one per line:
        <path>:<lineno>: [<KEYWORD>] <comment_text>
        <path>:<lineno>: [<KEYWORD>] [AMBIGUOUS] <comment_text>

    If no TODO-style comments are found, returns:
        no_todos_found
    """

    name = "scan_todos"
    description = (
        "Scan a Python source file or directory tree for TODO, FIXME, HACK, and NOTE "
        "comments. Returns each finding with its path, line number, and comment text, "
        "or 'no_todos_found' if none exist. Ambiguous TODOs (fewer than 5 words) are "
        "tagged [AMBIGUOUS]."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to a Python file or directory to scan. "
                    "When a directory is provided, all .py files are scanned recursively, "
                    "excluding .venv, site-packages, __pycache__, and .git subtrees."
                ),
            }
        },
        "required": ["path"],
    }

    def execute(self, inputs: dict) -> ToolResult:
        raw_path: str = inputs.get("path", "")
        target = Path(raw_path)

        if not target.exists():
            return ToolResult(is_error=True, content=f"file_not_found: {raw_path}")

        findings: list[str] = []

        if target.is_file():
            # Single-file mode: scan just this file (regardless of extension).
            findings = _scan_file(target)

        elif target.is_dir():
            # Directory mode: recursively scan all .py files, honouring exclusions.
            for py_file in sorted(target.rglob("*.py")):
                if _is_excluded(py_file):
                    continue
                findings.extend(_scan_file(py_file))

        else:
            # Neither a regular file nor a directory (e.g. a device node).
            return ToolResult(is_error=True, content=f"not_a_file_or_dir: {raw_path}")

        if not findings:
            return ToolResult(is_error=False, content="no_todos_found")

        return ToolResult(is_error=False, content="\n".join(findings))
