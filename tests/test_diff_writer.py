"""
Tests for agent/diff_writer.py.

All tests use pytest's tmp_path fixture to isolate output files.
No mocking — real DiffWriter instances are used throughout.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent.diff_writer import DiffWriter
from agent.protocols import TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_usage(input_tokens: int = 100, output_tokens: int = 50, usd: float = 0.001) -> TokenUsage:
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, estimated_usd=usd)


def _write_default(writer: DiffWriter, **overrides) -> str:
    """Call writer.write() with sensible defaults; caller can override any kwarg."""
    kwargs = dict(
        task_type="lint_fix",
        target_path="mymodule.py",
        original="x = 1\n",
        proposed="x = 2\n",
        iteration_count=1,
        token_usage=_make_usage(),
        model="claude-3-5-sonnet",
    )
    kwargs.update(overrides)
    return writer.write(**kwargs)


# ---------------------------------------------------------------------------
# 1. write() creates both .diff and .json files in output_dir
# ---------------------------------------------------------------------------

def test_write_creates_diff_and_json(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    diff_path_str = _write_default(writer)

    diff_path = Path(diff_path_str)
    json_path = diff_path.with_suffix(".json")

    assert diff_path.exists(), ".diff file should be created"
    assert json_path.exists(), ".json file should be created"


# ---------------------------------------------------------------------------
# 2. The .diff file contains a valid unified diff (starts with --- and +++)
# ---------------------------------------------------------------------------

def test_diff_file_contains_unified_diff_markers(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(_write_default(writer))

    content = diff_path.read_text(encoding="utf-8")
    assert content.startswith("---"), "diff should start with --- header"
    assert "+++" in content, "diff should contain +++ header"


# ---------------------------------------------------------------------------
# 3. The companion .json contains all required metadata fields
# ---------------------------------------------------------------------------

def test_json_contains_required_fields(tmp_path: Path) -> None:
    usage = _make_usage(input_tokens=200, output_tokens=80, usd=0.005)
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(
        writer.write(
            task_type="generate_tests",
            target_path="util.py",
            original="pass\n",
            proposed="pass  # noqa\n",
            iteration_count=3,
            token_usage=usage,
            model="claude-3-opus",
        )
    )
    json_path = diff_path.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["task_type"] == "generate_tests"
    assert data["target_path"] == "util.py"
    assert "timestamp_utc" in data
    assert data["llm_model"] == "claude-3-opus"
    assert data["iteration_count"] == 3
    assert "token_usage" in data


# ---------------------------------------------------------------------------
# 4. Filename follows the pattern {task_type}_{basename}_{timestamp}.diff
# ---------------------------------------------------------------------------

def test_filename_pattern(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(
        writer.write(
            task_type="lint_fix",
            target_path="some/path/my_module.py",
            original="a = 1\n",
            proposed="a = 2\n",
            iteration_count=1,
            token_usage=_make_usage(),
            model="claude-3-5-sonnet",
        )
    )
    name = diff_path.name
    # Pattern: lint_fix_my_module_YYYYMMDDTHHMMSSZ.diff  (no attempt suffix)
    pattern = r"^lint_fix_my_module_\d{8}T\d{6}Z\.diff$"
    assert re.match(pattern, name), f"Filename '{name}' does not match expected pattern"


# ---------------------------------------------------------------------------
# 5. Filename includes _attempt_N when attempt is provided
# ---------------------------------------------------------------------------

def test_filename_includes_attempt_suffix(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(_write_default(writer, attempt=2))
    name = diff_path.name
    assert "_attempt_2" in name, f"Expected '_attempt_2' in filename, got: {name}"


# ---------------------------------------------------------------------------
# 6. Collision avoidance: two writes in the same second produce distinct files
# ---------------------------------------------------------------------------

def test_collision_avoidance_produces_unique_files(tmp_path: Path, monkeypatch) -> None:
    """
    Freeze time so both calls produce the same base timestamp, forcing the
    collision avoidance loop to append _1 on the second call.
    """
    from datetime import datetime, timezone
    import agent.diff_writer as dw_module

    fixed_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Patch datetime inside the module so strftime always returns the same stamp
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt

    monkeypatch.setattr(dw_module, "datetime", FrozenDatetime)

    writer = DiffWriter(output_dir=tmp_path)
    path1 = Path(_write_default(writer))
    path2 = Path(_write_default(writer))

    assert path1 != path2, "Two writes should produce different file paths"
    assert path1.exists()
    assert path2.exists()

    # The second file should have the _1 collision suffix
    assert path2.stem.endswith("_1"), f"Expected collision suffix '_1' on second file, got: {path2.stem}"


# ---------------------------------------------------------------------------
# 7. UTF-8 encoding is used for the diff file
# ---------------------------------------------------------------------------

def test_diff_file_is_utf8(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(
        writer.write(
            task_type="lint_fix",
            target_path="unicode_file.py",
            original='msg = "héllo"\n',
            proposed='msg = "wörld"\n',
            iteration_count=1,
            token_usage=_make_usage(),
            model="claude-3-5-sonnet",
        )
    )
    # Reading as UTF-8 should not raise
    content = diff_path.read_bytes().decode("utf-8")
    assert "héllo" in content or "wörld" in content


# ---------------------------------------------------------------------------
# 8. Unix line endings (\n) even if diff contains \r\n sequences
# ---------------------------------------------------------------------------

def test_unix_line_endings_normalised(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    # Inject \r\n into original/proposed so difflib may include them in output
    diff_path = Path(
        writer.write(
            task_type="lint_fix",
            target_path="crlf_file.py",
            original="line1\r\nline2\r\n",
            proposed="line1\r\nline3\r\n",
            iteration_count=1,
            token_usage=_make_usage(),
            model="claude-3-5-sonnet",
        )
    )
    raw_bytes = diff_path.read_bytes()
    assert b"\r\n" not in raw_bytes, "diff file must not contain CRLF line endings"


# ---------------------------------------------------------------------------
# 9. output_dir is created automatically if it does not exist
# ---------------------------------------------------------------------------

def test_output_dir_created_automatically(tmp_path: Path) -> None:
    new_dir = tmp_path / "deeply" / "nested" / "output"
    assert not new_dir.exists()
    DiffWriter(output_dir=new_dir)
    assert new_dir.exists(), "DiffWriter should create output_dir on construction"


# ---------------------------------------------------------------------------
# 10. Return value is the absolute path of the .diff file as a string
# ---------------------------------------------------------------------------

def test_return_value_is_absolute_string_path(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    result = _write_default(writer)

    assert isinstance(result, str), "write() should return a str"
    assert Path(result).is_absolute(), "returned path should be absolute"
    assert result.endswith(".diff"), "returned path should end with .diff"


# ---------------------------------------------------------------------------
# 11. Empty diff (identical original and proposed) writes file with no diff lines
# ---------------------------------------------------------------------------

def test_empty_diff_for_identical_content(tmp_path: Path) -> None:
    writer = DiffWriter(output_dir=tmp_path)
    same = "x = 1\ny = 2\n"
    diff_path = Path(
        writer.write(
            task_type="lint_fix",
            target_path="unchanged.py",
            original=same,
            proposed=same,
            iteration_count=1,
            token_usage=_make_usage(),
            model="claude-3-5-sonnet",
        )
    )
    assert diff_path.exists(), "diff file should still be written for identical content"
    content = diff_path.read_text(encoding="utf-8")
    assert content == "", "diff content should be empty when original == proposed"


# ---------------------------------------------------------------------------
# 12. JSON metadata has correct token_usage sub-fields
# ---------------------------------------------------------------------------

def test_json_token_usage_subfields(tmp_path: Path) -> None:
    usage = TokenUsage(input_tokens=123, output_tokens=456, estimated_usd=0.0099)
    writer = DiffWriter(output_dir=tmp_path)
    diff_path = Path(
        writer.write(
            task_type="convert_todos",
            target_path="tasks.py",
            original="# TODO: fix\n",
            proposed="# FIXME: fix\n",
            iteration_count=2,
            token_usage=usage,
            model="claude-3-haiku",
        )
    )
    data = json.loads(diff_path.with_suffix(".json").read_text(encoding="utf-8"))
    tu = data["token_usage"]

    assert tu["input_tokens"] == 123
    assert tu["output_tokens"] == 456
    assert abs(tu["estimated_usd"] - 0.0099) < 1e-9
