"""
Diff Writer for the Codebase Maintainer Agent.

WHY save diffs rather than applying them directly?
  Human-in-the-loop is the key design constraint. The agent must never modify
  source files directly — it proposes changes that a human reviews and applies.
  Writing diffs to a well-named directory makes the proposal easy to inspect
  with `cat`, `patch --dry-run`, or any diff viewer.

WHY use difflib instead of subprocess `diff`?
  difflib is stdlib — no subprocess, no OS-specific binary, no cross-platform
  issues. The unified_diff output is spec-compliant and identical to `diff -u`
  for text files, which is all we need for Python source files.

WHY enforce Unix line endings?
  Patches applied on Windows with \r\n line endings often fail when the target
  machine uses \n. Normalising to \n on write means diffs work on all platforms.

WHY the collision avoidance loop?
  Multiple runs of the same task in the same second would produce identical
  filenames. The _1, _2 suffix loop guarantees every output file is unique
  without overwriting prior results — important for the self-correction loop
  where _attempt_1, _attempt_2, _attempt_3 must all be preserved.
"""
from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.protocols import TokenUsage


class DiffWriter:
    """
    Generates unified diffs and writes them with companion JSON metadata.

    output_dir is created at construction time if it does not already exist.
    """

    def __init__(self, output_dir: str | Path = "./agent_output") -> None:
        self.output_dir = Path(output_dir)
        # Create immediately so subsequent writes never fail due to missing dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        task_type: str,
        target_path: str,
        original: str,
        proposed: str,
        iteration_count: int,
        token_usage: TokenUsage,
        model: str,
        attempt: int | None = None,
    ) -> str:
        """
        Generate a unified diff between original and proposed content, write it
        to disk, write a companion JSON metadata file, and return the absolute
        path of the .diff file.

        attempt: if provided (e.g. 1, 2, 3), appended as _attempt_N in filename.
        """
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                proposed.splitlines(keepends=True),
                fromfile=f"a/{Path(target_path).name}",
                tofile=f"b/{Path(target_path).name}",
            )
        )
        diff_text = "".join(diff_lines)

        # Normalise to Unix line endings regardless of OS
        # WHY: Windows writes \r\n by default; patches applied on Linux/macOS
        # would fail with "malformed patch" errors if line endings don't match.
        diff_text = diff_text.replace("\r\n", "\n").replace("\r", "\n")

        base_name = self._make_base_name(task_type, target_path, attempt)
        diff_path = self._unique_path(base_name + ".diff")
        json_path = diff_path.with_suffix(".json")

        # Write diff with explicit newline= to prevent Python from translating
        # \n to \r\n on Windows
        diff_path.write_text(diff_text, encoding="utf-8", newline="\n")

        metadata = {
            "task_type": task_type,
            "target_path": str(target_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "llm_model": model,
            "iteration_count": iteration_count,
            "token_usage": {
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "estimated_usd": token_usage.estimated_usd,
            },
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return str(diff_path.resolve())

    def _make_base_name(
        self,
        task_type: str,
        target_path: str,
        attempt: int | None,
    ) -> str:
        """Build the filename stem: {task_type}_{basename}_{ISO8601}[_attempt_N]"""
        # Use UTC time, replace colons so the name is valid on Windows filesystems
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        basename = Path(target_path).stem
        suffix = f"_attempt_{attempt}" if attempt is not None else ""
        return f"{task_type}_{basename}_{timestamp}{suffix}"

    def _unique_path(self, base_name: str) -> Path:
        """
        Return a Path that does not yet exist in output_dir.

        If base_name already exists, try base_name_1, base_name_2, ...
        WHY a loop and not a UUID?  The numeric suffix makes it obvious that
        files are related (same task, different attempts or near-simultaneous
        runs) without making names opaque.
        """
        stem = Path(base_name).stem
        ext = Path(base_name).suffix
        candidate = self.output_dir / base_name
        if not candidate.exists():
            return candidate
        n = 1
        while True:
            candidate = self.output_dir / f"{stem}_{n}{ext}"
            if not candidate.exists():
                return candidate
            n += 1
