"""
Tests for agent/tools/todo_scanner.py

Covers:
1.  Single file with TODOs — correct line numbers and comment text
2.  Directory scan — finds TODOs in multiple .py files recursively
3.  .venv exclusion
4.  site-packages exclusion
5.  __pycache__ exclusion
6.  .git exclusion
7.  Ambiguous TODO (fewer than 5 words) — tagged [AMBIGUOUS]
8.  Non-ambiguous TODO (5+ words) — no [AMBIGUOUS] tag
9.  no_todos_found when nothing matches
10. Missing file → is_error=True
11. FIXME, HACK, NOTE patterns are detected
12. Case-insensitive matching (todo, Todo, TODO all match)
"""
from __future__ import annotations

import pytest

from agent.tools.todo_scanner import TodoScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, text: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def scanner() -> TodoScanner:
    return TodoScanner()


# ---------------------------------------------------------------------------
# 1. Single file — correct line numbers and comment text
# ---------------------------------------------------------------------------

def test_single_file_line_numbers_and_text(tmp_path, scanner):
    py_file = tmp_path / "module.py"
    _write(py_file, (
        "def foo():\n"                                   # line 1
        "    pass\n"                                     # line 2
        "# TODO: refactor this to use a strategy pattern\n"  # line 3
        "def bar():\n"                                   # line 4
        "    # FIXME: handle the edge case where x is None here\n"  # line 5
        "    pass\n"                                     # line 6
    ))

    result = scanner.execute({"path": str(py_file)})

    assert not result.is_error
    lines = result.content.splitlines()
    assert len(lines) == 2

    # Line 3
    assert ":3:" in lines[0]
    assert "[TODO]" in lines[0]
    assert "refactor this to use a strategy pattern" in lines[0]

    # Line 5
    assert ":5:" in lines[1]
    assert "[FIXME]" in lines[1]
    assert "handle the edge case where x is None here" in lines[1]


# ---------------------------------------------------------------------------
# 2. Directory scan — finds TODOs in multiple .py files recursively
# ---------------------------------------------------------------------------

def test_directory_scan_multiple_files(tmp_path, scanner):
    _write(tmp_path / "a.py", "# TODO: first task that needs to be done soon\n")
    _write(tmp_path / "sub" / "b.py", "# TODO: second task that must be finished today\n")
    _write(tmp_path / "sub" / "deep" / "c.py", "# HACK: ugly workaround for the serialiser bug\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    content = result.content

    assert "[TODO]" in content
    assert "first task" in content
    assert "second task" in content
    assert "[HACK]" in content
    assert "ugly workaround" in content


# ---------------------------------------------------------------------------
# 3. .venv files are excluded from directory scan
# ---------------------------------------------------------------------------

def test_venv_excluded(tmp_path, scanner):
    _write(tmp_path / ".venv" / "lib" / "pkg.py",
           "# TODO: vendored code that must be skipped entirely here\n")
    # Real file — should still be found
    _write(tmp_path / "real.py",
           "# TODO: this is a real task that actually needs doing now\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    assert "vendored code" not in result.content
    assert "real task" in result.content


# ---------------------------------------------------------------------------
# 4. site-packages files are excluded
# ---------------------------------------------------------------------------

def test_site_packages_excluded(tmp_path, scanner):
    _write(tmp_path / "site-packages" / "third_party.py",
           "# TODO: upstream library todo that should not appear here\n")
    _write(tmp_path / "src.py",
           "# TODO: our own code task that belongs in the results today\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    assert "upstream library" not in result.content
    assert "our own code task" in result.content


# ---------------------------------------------------------------------------
# 5. __pycache__ files are excluded
# ---------------------------------------------------------------------------

def test_pycache_excluded(tmp_path, scanner):
    _write(tmp_path / "__pycache__" / "compiled.py",
           "# TODO: bytecode artefact that must never appear in scan output\n")
    _write(tmp_path / "source.py",
           "# TODO: genuine source code comment that should definitely appear\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    assert "bytecode artefact" not in result.content
    assert "genuine source code comment" in result.content


# ---------------------------------------------------------------------------
# 6. .git files are excluded
# ---------------------------------------------------------------------------

def test_git_excluded(tmp_path, scanner):
    _write(tmp_path / ".git" / "hooks" / "pre_commit.py",
           "# TODO: git hook script that should be excluded from scan always\n")
    _write(tmp_path / "app.py",
           "# TODO: application logic that should appear in the scan results\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    assert "git hook script" not in result.content
    assert "application logic" in result.content


# ---------------------------------------------------------------------------
# 7. Ambiguous TODO (< 5 non-whitespace words) → [AMBIGUOUS] tag
# ---------------------------------------------------------------------------

def test_ambiguous_todo_tagged(tmp_path, scanner):
    py_file = tmp_path / "ambig.py"
    # Comment text after "TODO:" has only 3 words — ambiguous
    _write(py_file, "# TODO: fix this later\n")

    result = scanner.execute({"path": str(py_file)})

    assert not result.is_error
    assert "[AMBIGUOUS]" in result.content
    assert "fix this later" in result.content


# ---------------------------------------------------------------------------
# 8. Non-ambiguous TODO (≥ 5 non-whitespace words) → no [AMBIGUOUS] tag
# ---------------------------------------------------------------------------

def test_non_ambiguous_todo_not_tagged(tmp_path, scanner):
    py_file = tmp_path / "clear.py"
    # Comment text has exactly 5 words — not ambiguous
    _write(py_file, "# TODO: refactor to use the factory pattern\n")

    result = scanner.execute({"path": str(py_file)})

    assert not result.is_error
    assert "[AMBIGUOUS]" not in result.content
    assert "refactor to use the factory pattern" in result.content


def test_non_ambiguous_todo_many_words(tmp_path, scanner):
    py_file = tmp_path / "verbose.py"
    _write(py_file, "# TODO: replace this naive O(n^2) loop with a hash map for performance\n")

    result = scanner.execute({"path": str(py_file)})

    assert not result.is_error
    assert "[AMBIGUOUS]" not in result.content


# ---------------------------------------------------------------------------
# 9. no_todos_found when there are no matching comments
# ---------------------------------------------------------------------------

def test_no_todos_found_in_file(tmp_path, scanner):
    py_file = tmp_path / "clean.py"
    _write(py_file, "def add(a, b):\n    return a + b\n")

    result = scanner.execute({"path": str(py_file)})

    assert not result.is_error
    assert result.content == "no_todos_found"


def test_no_todos_found_in_directory(tmp_path, scanner):
    _write(tmp_path / "a.py", "x = 1\n")
    _write(tmp_path / "b.py", "y = 2\n")

    result = scanner.execute({"path": str(tmp_path)})

    assert not result.is_error
    assert result.content == "no_todos_found"


# ---------------------------------------------------------------------------
# 10. Missing file → is_error=True
# ---------------------------------------------------------------------------

def test_missing_file_returns_error(tmp_path, scanner):
    missing = tmp_path / "nonexistent.py"

    result = scanner.execute({"path": str(missing)})

    assert result.is_error
    assert "file_not_found" in result.content


# ---------------------------------------------------------------------------
# 11. FIXME, HACK, NOTE patterns are detected
# ---------------------------------------------------------------------------

def test_fixme_detected(tmp_path, scanner):
    py_file = tmp_path / "f.py"
    _write(py_file, "# FIXME: address the race condition in the connection pool\n")

    result = scanner.execute({"path": str(py_file)})
    assert "[FIXME]" in result.content


def test_hack_detected(tmp_path, scanner):
    py_file = tmp_path / "h.py"
    _write(py_file, "# HACK: temporary shim until the upstream API is stable\n")

    result = scanner.execute({"path": str(py_file)})
    assert "[HACK]" in result.content


def test_note_detected(tmp_path, scanner):
    py_file = tmp_path / "n.py"
    _write(py_file, "# NOTE: this function assumes the caller holds the GIL lock\n")

    result = scanner.execute({"path": str(py_file)})
    assert "[NOTE]" in result.content


def test_all_four_patterns_in_one_file(tmp_path, scanner):
    py_file = tmp_path / "mixed.py"
    _write(py_file, (
        "# TODO: implement the retry logic with exponential backoff soon\n"
        "# FIXME: address the race condition in the connection pool manager\n"
        "# HACK: temporary shim until the upstream API version is stable\n"
        "# NOTE: this function assumes the caller holds the GIL lock always\n"
    ))

    result = scanner.execute({"path": str(py_file)})
    assert "[TODO]" in result.content
    assert "[FIXME]" in result.content
    assert "[HACK]" in result.content
    assert "[NOTE]" in result.content


# ---------------------------------------------------------------------------
# 12. Case-insensitive matching (todo, Todo, TODO all match)
# ---------------------------------------------------------------------------

def test_lowercase_todo_matched(tmp_path, scanner):
    py_file = tmp_path / "lower.py"
    _write(py_file, "# todo: implement the missing validation for the email field\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[TODO]" in result.content


def test_titlecase_todo_matched(tmp_path, scanner):
    py_file = tmp_path / "title.py"
    _write(py_file, "# Todo: add proper error handling for the network timeout\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[TODO]" in result.content


def test_uppercase_todo_matched(tmp_path, scanner):
    py_file = tmp_path / "upper.py"
    _write(py_file, "# TODO: replace this placeholder with the real implementation\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[TODO]" in result.content


def test_mixed_case_fixme_matched(tmp_path, scanner):
    py_file = tmp_path / "fixme_lower.py"
    _write(py_file, "# fixme: resolve the deadlock in the transaction manager layer\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[FIXME]" in result.content


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_empty_file_returns_no_todos(tmp_path, scanner):
    py_file = tmp_path / "empty.py"
    _write(py_file, "")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert result.content == "no_todos_found"


def test_exact_five_words_not_ambiguous(tmp_path, scanner):
    """Exactly 5 non-whitespace words → not ambiguous (boundary check)."""
    py_file = tmp_path / "exact5.py"
    # "implement", "the", "new", "login", "flow" = 5 words
    _write(py_file, "# TODO: implement the new login flow\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[AMBIGUOUS]" not in result.content


def test_four_words_is_ambiguous(tmp_path, scanner):
    """4 non-whitespace words → ambiguous (boundary check)."""
    py_file = tmp_path / "four.py"
    # "implement", "new", "login", "flow" = 4 words
    _write(py_file, "# TODO: implement new login flow\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert "[AMBIGUOUS]" in result.content


def test_directory_with_no_py_files_returns_no_todos(tmp_path, scanner):
    """A directory containing only non-.py files returns no_todos_found."""
    (tmp_path / "readme.txt").write_text("TODO: this is a text file, not Python\n")

    result = scanner.execute({"path": str(tmp_path)})
    assert not result.is_error
    assert result.content == "no_todos_found"


def test_deeply_nested_file_found(tmp_path, scanner):
    """Scanner reaches .py files nested several levels deep."""
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    _write(deep / "deep.py", "# TODO: refactor this deeply nested module for better clarity\n")

    result = scanner.execute({"path": str(tmp_path)})
    assert not result.is_error
    assert "refactor this deeply nested module" in result.content


def test_result_includes_file_path(tmp_path, scanner):
    """Each result line includes the path to the source file."""
    py_file = tmp_path / "located.py"
    _write(py_file, "# TODO: add input sanitisation to the public API endpoint now\n")

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert str(py_file) in result.content


def test_result_includes_line_number(tmp_path, scanner):
    """Line number appears in result string."""
    py_file = tmp_path / "lined.py"
    _write(py_file, (
        "# a blank first line\n"   # line 1
        "x = 1\n"                  # line 2
        "# TODO: write tests for the edge cases in the parser module\n"  # line 3
    ))

    result = scanner.execute({"path": str(py_file)})
    assert not result.is_error
    assert ":3:" in result.content
