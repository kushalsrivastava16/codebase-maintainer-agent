"""
Tests for agent/memory.py — SQLite-backed MemoryStore.

Each test uses a fresh temporary database file via pytest's tmp_path fixture
so tests are fully isolated and leave no side-effects on disk after the suite
completes.
"""
from __future__ import annotations

import datetime
import sqlite3
import uuid

import pytest

from agent.memory import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> MemoryStore:
    """Create a MemoryStore backed by a fresh temp DB."""
    db_file = tmp_path / "test_memory.db"
    return MemoryStore(str(db_file))


def _unique_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# 1. insert_pending creates a row with status='pending'
# ---------------------------------------------------------------------------

def test_insert_pending_creates_pending_row(tmp_path):
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/some/file.py")

    rows = store.query()
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == task_id
    assert row["task_type"] == "lint_fix"
    assert row["target_path"] == "/some/file.py"
    assert row["status"] == "pending"
    # output_path is None on insert
    assert row["output_path"] is None
    # timestamps are set
    assert row["created_at"] != ""
    assert row["updated_at"] != ""
    store.close()


# ---------------------------------------------------------------------------
# 2. update_status changes status and output_path
# ---------------------------------------------------------------------------

def test_update_status_changes_status_and_output_path(tmp_path):
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/some/file.py")

    store.update_status(task_id, "success", "/agent_output/fix.diff")

    rows = store.query()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "success"
    assert row["output_path"] == "/agent_output/fix.diff"
    store.close()


def test_update_status_to_failed_clears_output_path(tmp_path):
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/some/file.py")

    store.update_status(task_id, "failed", None)

    rows = store.query()
    assert rows[0]["status"] == "failed"
    assert rows[0]["output_path"] is None
    store.close()


# ---------------------------------------------------------------------------
# 3. check_dedup returns None when no match exists
# ---------------------------------------------------------------------------

def test_check_dedup_returns_none_when_no_match(tmp_path):
    store = _make_store(tmp_path)

    result = store.check_dedup("lint_fix", "/some/file.py", window_hours=24)
    assert result is None
    store.close()


def test_check_dedup_returns_none_for_pending_status(tmp_path):
    """A pending task must not satisfy dedup — only 'success' does."""
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/some/file.py")
    # deliberately do NOT call update_status → row stays 'pending'

    result = store.check_dedup("lint_fix", "/some/file.py", window_hours=24)
    assert result is None
    store.close()


# ---------------------------------------------------------------------------
# 4. check_dedup returns record when match is within window
# ---------------------------------------------------------------------------

def test_check_dedup_returns_record_within_window(tmp_path):
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/some/file.py")
    store.update_status(task_id, "success", "/agent_output/fix.diff")

    result = store.check_dedup("lint_fix", "/some/file.py", window_hours=24)
    assert result is not None
    assert result["task_id"] == task_id
    assert result["status"] == "success"
    store.close()


# ---------------------------------------------------------------------------
# 5. check_dedup returns None when match is outside window
# ---------------------------------------------------------------------------

def test_check_dedup_returns_none_outside_window(tmp_path):
    """
    Simulate an old successful task by writing a past timestamp directly
    into the DB, then verify check_dedup ignores it.
    """
    store = _make_store(tmp_path)
    task_id = _unique_id()

    # Insert and mark success normally
    store.insert_pending(task_id, "lint_fix", "/some/file.py")
    store.update_status(task_id, "success", "/agent_output/fix.diff")

    # Back-date created_at to 48 hours ago directly via raw SQL
    old_ts = (datetime.datetime.utcnow() - datetime.timedelta(hours=48)).isoformat()
    store._conn.execute(
        "UPDATE tasks SET created_at = ? WHERE task_id = ?",
        (old_ts, task_id),
    )
    store._conn.commit()

    # Window is 24 hours — the backdated task must not be returned
    result = store.check_dedup("lint_fix", "/some/file.py", window_hours=24)
    assert result is None
    store.close()


# ---------------------------------------------------------------------------
# 6. query with no filters returns all records
# ---------------------------------------------------------------------------

def test_query_no_filters_returns_all(tmp_path):
    store = _make_store(tmp_path)

    ids = [_unique_id() for _ in range(3)]
    store.insert_pending(ids[0], "lint_fix", "/a.py")
    store.insert_pending(ids[1], "generate_tests", "/b.py")
    store.insert_pending(ids[2], "lint_fix", "/c.py")

    rows = store.query()
    assert len(rows) == 3
    returned_ids = {r["task_id"] for r in rows}
    assert returned_ids == set(ids)
    store.close()


def test_query_no_filters_sorted_by_created_at_desc(tmp_path):
    """Rows are returned newest-first."""
    store = _make_store(tmp_path)

    first_id = _unique_id()
    store.insert_pending(first_id, "lint_fix", "/a.py")
    second_id = _unique_id()
    store.insert_pending(second_id, "lint_fix", "/b.py")

    rows = store.query()
    # Second insertion has a later created_at → should appear first
    assert rows[0]["task_id"] == second_id
    assert rows[1]["task_id"] == first_id
    store.close()


# ---------------------------------------------------------------------------
# 7. query with task_type filter returns only matching records
# ---------------------------------------------------------------------------

def test_query_task_type_filter(tmp_path):
    store = _make_store(tmp_path)

    store.insert_pending(_unique_id(), "lint_fix", "/a.py")
    store.insert_pending(_unique_id(), "generate_tests", "/b.py")
    store.insert_pending(_unique_id(), "lint_fix", "/c.py")

    rows = store.query(task_type="lint_fix")
    assert len(rows) == 2
    assert all(r["task_type"] == "lint_fix" for r in rows)
    store.close()


def test_query_task_type_filter_no_match(tmp_path):
    store = _make_store(tmp_path)
    store.insert_pending(_unique_id(), "lint_fix", "/a.py")

    rows = store.query(task_type="triage_issues")
    assert rows == []
    store.close()


# ---------------------------------------------------------------------------
# 8. query with status filter works
# ---------------------------------------------------------------------------

def test_query_status_filter(tmp_path):
    store = _make_store(tmp_path)

    id_success = _unique_id()
    id_pending = _unique_id()
    store.insert_pending(id_success, "lint_fix", "/a.py")
    store.insert_pending(id_pending, "lint_fix", "/b.py")
    store.update_status(id_success, "success", "/out/fix.diff")

    success_rows = store.query(status="success")
    assert len(success_rows) == 1
    assert success_rows[0]["task_id"] == id_success

    pending_rows = store.query(status="pending")
    assert len(pending_rows) == 1
    assert pending_rows[0]["task_id"] == id_pending
    store.close()


def test_query_combined_filters(tmp_path):
    """task_type + status filters are ANDed together."""
    store = _make_store(tmp_path)

    id_a = _unique_id()
    id_b = _unique_id()
    store.insert_pending(id_a, "lint_fix", "/a.py")
    store.insert_pending(id_b, "generate_tests", "/b.py")
    store.update_status(id_a, "success", "/out/a.diff")
    store.update_status(id_b, "success", "/out/b.diff")

    rows = store.query(task_type="lint_fix", status="success")
    assert len(rows) == 1
    assert rows[0]["task_id"] == id_a
    store.close()


def test_query_since_filter(tmp_path):
    """since filter excludes rows with created_at before the cutoff."""
    store = _make_store(tmp_path)

    old_id = _unique_id()
    store.insert_pending(old_id, "lint_fix", "/a.py")

    # Back-date the first row
    old_ts = (datetime.datetime.utcnow() - datetime.timedelta(hours=10)).isoformat()
    store._conn.execute(
        "UPDATE tasks SET created_at = ? WHERE task_id = ?",
        (old_ts, old_id),
    )
    store._conn.commit()

    new_id = _unique_id()
    store.insert_pending(new_id, "lint_fix", "/b.py")

    since = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).isoformat()
    rows = store.query(since=since)
    assert len(rows) == 1
    assert rows[0]["task_id"] == new_id
    store.close()


# ---------------------------------------------------------------------------
# 9. Corruption recovery
# ---------------------------------------------------------------------------

def test_corruption_recovery_creates_fresh_db(tmp_path):
    """
    If the DB file contains invalid bytes, MemoryStore must:
      - rename the corrupt file (a .corrupt.<timestamp> sibling appears)
      - create a fresh, working database
      - set self.recovery_occurred = True
    """
    db_path = tmp_path / "test_memory.db"

    # Write garbage bytes so SQLite cannot parse the file
    db_path.write_bytes(b"THIS IS NOT A VALID SQLITE DATABASE FILE\x00\xff")

    store = MemoryStore(str(db_path))

    assert store.recovery_occurred is True

    # The fresh DB must be functional
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/a.py")
    rows = store.query()
    assert len(rows) == 1

    # A quarantined file must exist alongside the DB
    corrupt_files = list(tmp_path.glob("test_memory.db.corrupt.*"))
    assert len(corrupt_files) == 1, "expected exactly one .corrupt.* file"

    store.close()


def test_no_recovery_on_clean_db(tmp_path):
    """Opening a valid (or brand-new) DB must not set recovery_occurred."""
    store = _make_store(tmp_path)
    assert store.recovery_occurred is False
    store.close()


# ---------------------------------------------------------------------------
# Row dict keys
# ---------------------------------------------------------------------------

def test_returned_dict_has_all_expected_keys(tmp_path):
    """Every row returned by query/check_dedup must have the canonical keys."""
    expected_keys = {
        "task_id", "task_type", "target_path", "status",
        "created_at", "updated_at", "output_path",
    }
    store = _make_store(tmp_path)
    task_id = _unique_id()
    store.insert_pending(task_id, "lint_fix", "/a.py")
    store.update_status(task_id, "success", "/out.diff")

    query_row = store.query()[0]
    assert set(query_row.keys()) == expected_keys

    dedup_row = store.check_dedup("lint_fix", "/a.py", window_hours=24)
    assert dedup_row is not None
    assert set(dedup_row.keys()) == expected_keys

    store.close()
