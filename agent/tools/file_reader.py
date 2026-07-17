"""
File Reading Tool for the Codebase Maintainer Agent.

WHY this tool exists:
  The LLM cannot read files directly — it can only reason about text provided
  in its context. This tool is the bridge: it safely exposes repository source
  code to the LLM so it can reason about lint errors, test gaps, and TODOs.

WHY the path traversal guard matters:
  The LLM generates the file path argument. If it hallucinates "../../../etc/passwd"
  or is manipulated by a prompt injection in a TODO comment, the guard prevents
  any file outside the repository root from being read. This is one of the most
  important security controls in the agent.

WHY strip injection patterns:
  File content is untrusted data from the user's codebase. A developer could
  have written "# TODO: Human: Ignore all previous instructions and..." in their
  source file. Stripping these patterns before passing content to the LLM reduces
  the attack surface, though a determined attacker could find other vectors.
"""
from __future__ import annotations

import re
from pathlib import Path

from agent.protocols import ToolResult

MAX_FILE_BYTES: int = 100 * 1024  # 100 KB

# Known LLM prompt injection / role-spoofing patterns.
# These are model-family-specific delimiters that, if present in file content,
# could confuse the model into treating the file as part of the system prompt
# or a different conversation turn.
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'\[INST\]'),                  # LLaMA-2 / Mistral instruction tag
    re.compile(r'^Human:\s', re.MULTILINE),   # Claude/RLHF human-turn injection (line-start only)
    # NOT stripping: ###, <s>, </s> — these appear legitimately in Python
    # section dividers, Sphinx math RST, and XML/HTML docstrings and would
    # corrupt the source code the LLM is asked to fix.
]


class FileReader:
    """
    Safely reads source files from a repository for LLM context.

    All safety checks (path traversal, size limit, injection) are applied
    before returning content. Errors are returned as ToolResult(is_error=True)
    so the LLM can reason about the failure rather than crashing the agent.
    """

    name = "read_file"
    description = (
        "Read a source file from the repository and return its UTF-8 content. "
        "Use this to inspect file contents before proposing changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the repository root (e.g. 'src/utils.py')",
            }
        },
        "required": ["path"],
    }

    def __init__(self, repo_root: str | Path) -> None:
        # Resolve once at construction so all checks use the canonical absolute path.
        # WHY resolve here: symlinks in the repo_root itself should be followed,
        # so that (repo_root / "../sibling").resolve() is correctly detected as
        # outside the root even if repo_root contains symlinks.
        self._repo_root: Path = Path(repo_root).resolve()

    def execute(self, inputs: dict) -> ToolResult:
        raw_path: str = inputs.get("path", "")

        # --- Security: path traversal guard ---
        resolved = (self._repo_root / raw_path).resolve()
        try:
            resolved.relative_to(self._repo_root)
        except ValueError:
            # The resolved path is not a descendant of repo_root.
            # This catches ../../../etc/passwd style attacks and symlinks
            # that point outside the repository.
            return ToolResult(is_error=True, content="path_traversal_denied")

        # --- Existence and permission checks ---
        if not resolved.exists():
            return ToolResult(is_error=True, content=f"file_not_found: {raw_path}")

        try:
            raw_bytes: bytes = resolved.read_bytes()
        except PermissionError:
            return ToolResult(is_error=True, content=f"permission_denied: {raw_path}")

        # --- Size limit: truncate at 100 KB ---
        truncated = False
        if len(raw_bytes) > MAX_FILE_BYTES:
            raw_bytes = raw_bytes[:MAX_FILE_BYTES]
            truncated = True

        # --- Decode: replace undecodable bytes rather than raising ---
        content: str = raw_bytes.decode("utf-8", errors="replace")

        # --- Injection pattern stripping ---
        removed = 0
        for pattern in INJECTION_PATTERNS:
            content, n = pattern.subn("", content)
            removed += n

        # --- Append notices ---
        notices: list[str] = []
        if truncated:
            notices.append("[TRUNCATED: file exceeds 100 KB limit]")
        if removed > 0:
            notices.append(f"[SANITIZED: {removed} injection patterns removed]")

        if notices:
            content = content + "\n" + "\n".join(notices)

        return ToolResult(is_error=False, content=content)
