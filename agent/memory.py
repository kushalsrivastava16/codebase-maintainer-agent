"""
SQLite-backed persistent memory store for task history and deduplication.

WHY SQLite instead of a flat file or in-memory dict?
  SQLite gives us ACID transactions, SQL filtering, and WAL mode for concurrent
  readers without locking writers. A flat JSON file would require full rewrites
  on every update and provides no query capability. An in-memory dict is lost
  between runs, which defeats the purpose of deduplication across invocations.

WHY WAL (Write-Ahead Logging) mode?
  WAL allows concurrent reads and a single writer to coexist without readers
  blocking writers or vice versa. Since the agent may someday process multiple
  tasks concurrently (Phase 3), enabling WAL now is cheap insurance.

WHY corruption recovery instead of raising?
  A corrupt DB on agent startup would crash every subsequent run, rendering the
  agent inoperable until a human intervenes. By renaming the corrupt file and
  starting fresh, the agent degrades gracefully: it loses dedup history but
  keeps running. The caller is responsible for logging the recovery event
  (via self.recovery_occurred) because the logger may not yet be available
  when __init__ runs.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
from pathlib import Path
from typing import Any


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    task_type   TEXT NOT NULL,
    target_path TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    output_path TEXT
);
"""

_ROW_KEYS = ("task_id", "task_type", "target_path", "status", "created_at", "updated_at", "output_path")


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a raw SQLite row tuple to a keyed dict with the canonical schema."""
    return dict(zip(_ROW_KEYS, row))


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string (no tzinfo suffix)."""
    return datetime.datetime.utcnow().isoformat()


class MemoryStore:
    """
    Persistent task history store backed by SQLite.

    Responsibilities:
      - Insert new tasks with status='pending'
      - Update task status and output path on completion
      - Deduplicate tasks by (task_type, target_path) within a rolling window
      - Query task history with optional filters

    Thread safety: SQLite's WAL mode handles concurrent readers; a single
    connection per process is safe for sequential access. For concurrent writes
    in Phase 3, callers should instantiate one MemoryStore per thread or use
    a connection pool.
    """

    recovery_occurred: bool  # True if the DB was corrupt and was replaced

    def __init__(self, db_path: str) -> None:
        """
        Open or create the SQLite database at db_path.

        If the file exists but is corrupt, it is renamed to
        <db_path>.corrupt.<timestamp> and a fresh database is created.
        After construction, check self.recovery_occurred and log accordingly.
        """
        self.db_path = db_path
        self.recovery_occurred = False
        self._conn = self._open()
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        """
        Open the SQLite connection, enabling WAL mode.

        On sqlite3.DatabaseError (corrupt file), rename and reinitialise.

        WHY close the connection before quarantining?
          On Windows, a file that is open by a sqlite3 connection cannot be
          renamed — the OS holds an exclusive lock. We must explicitly close
          the bad connection object before calling os.rename(), otherwise the
          rename silently fails (or raises OSError on Windows) and the second
          sqlite3.connect() call re-opens the same corrupt file.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            # Validate the file is actually a readable SQLite database by
            # executing a trivial statement before we return.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA integrity_check(1);")
            return conn
        except sqlite3.DatabaseError:
            # Release the file handle BEFORE renaming so the OS lock is cleared.
            try:
                conn.close()
            except Exception:
                pass
            self._quarantine_corrupt_db()
            fresh = sqlite3.connect(self.db_path, check_same_thread=False)
            fresh.execute("PRAGMA journal_mode=WAL;")
            return fresh

    def _quarantine_corrupt_db(self) -> None:
        """
        Rename the corrupt database file so the agent can start fresh.

        WHY keep the corrupt file instead of deleting it?
          The corrupt file may contain recoverable data or be useful for
          post-mortem debugging. Renaming preserves it without blocking the
          agent from continuing.
        """
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        corrupt_path = f"{self.db_path}.corrupt.{timestamp}"
        try:
            os.rename(self.db_path, corrupt_path)
        except OSError:
            # If we cannot rename (e.g., permissions), just delete so we can
            # create a fresh DB. The data is already lost at this point.
            try:
                os.remove(self.db_path)
            except OSError:
                pass
        self.recovery_occurred = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_pending(self, task_id: str, task_type: str, target_path: str) -> None:
        """
        Insert a new task row with status='pending'.

        Both created_at and updated_at are set to the current UTC time.
        Raises sqlite3.IntegrityError if task_id already exists (PRIMARY KEY
        collision). Callers should generate unique task_ids (UUIDs).
        """
        now = _utcnow_iso()
        self._conn.execute(
            """
            INSERT INTO tasks (task_id, task_type, target_path, status, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?)
            """,
            (task_id, task_type, target_path, now, now),
        )
        self._conn.commit()

    def update_status(self, task_id: str, status: str, output_path: str | None) -> None:
        """
        Update the status, updated_at, and output_path of an existing task.

        Does nothing (silently) if task_id does not exist, matching the
        convention of other update helpers in this codebase that treat missing
        rows as a no-op rather than an error.
        """
        now = _utcnow_iso()
        self._conn.execute(
            """
            UPDATE tasks
               SET status      = ?,
                   updated_at  = ?,
                   output_path = ?
             WHERE task_id = ?
            """,
            (status, now, output_path, task_id),
        )
        self._conn.commit()

    def check_dedup(
        self,
        task_type: str,
        target_path: str,
        window_hours: int,
    ) -> dict | None:
        """
        Return the most recent successful task for (task_type, target_path)
        within the last window_hours, or None if no such task exists.

        WHY deduplicate by success only?
          A previously failed task should not block a retry. We only skip work
          when we know the target was already successfully processed recently.

        WHY use created_at for the window instead of updated_at?
          created_at is set once and never changes, making it a stable anchor
          for "when did we start working on this". updated_at can be bumped by
          status transitions that happen well after the task started.
        """
        cutoff = (
            datetime.datetime.utcnow() - datetime.timedelta(hours=window_hours)
        ).isoformat()

        cursor = self._conn.execute(
            """
            SELECT task_id, task_type, target_path, status, created_at, updated_at, output_path
              FROM tasks
             WHERE task_type   = ?
               AND target_path = ?
               AND status      = 'success'
               AND created_at  >= ?
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (task_type, target_path, cutoff),
        )
        row = cursor.fetchone()
        return _row_to_dict(row) if row is not None else None

    def query(
        self,
        task_type: str | None = None,
        status: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """
        Return task rows matching the given filters, sorted by created_at DESC.

        All filters are optional and ANDed together:
          task_type: exact match on the task_type column
          status:    exact match on the status column
          since:     ISO 8601 timestamp; only rows with created_at >= since are included

        Returns an empty list when no rows match.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if task_type is not None:
            clauses.append("task_type = ?")
            params.append(task_type)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT task_id, task_type, target_path, status, created_at, updated_at, output_path
              FROM tasks
             {where}
             ORDER BY created_at DESC
        """
        cursor = self._conn.execute(sql, params)
        return [_row_to_dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
