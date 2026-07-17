"""
FastAPI web server for the Codebase Maintainer Agent.

Provides:
  - Static file serving for the frontend SPA
  - POST /api/run       — start a task, stream logs via SSE
  - GET  /api/stream    — SSE endpoint that replays logs for a run
  - GET  /api/history   — task history from the SQLite memory store
  - GET  /api/diffs     — list generated diff files
  - GET  /api/diff      — serve a specific diff file
  - GET  /api/files     — list Python files available to target
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import AsyncIterator

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHF]|\x1b\[[?][0-9;]*[hl]|\x1b\[[\d;]*[A-Za-z]')

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
AGENT_OUTPUT_DIR = BASE_DIR / "agent_output"
MEMORY_DB = BASE_DIR / "agent_memory.db"
AGENT_OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Codebase Maintainer Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory log buffer: run_id -> list of SSE lines
_run_logs: dict[str, list[str]] = {}
_run_status: dict[str, str] = {}  # run_id -> "running" | "done" | "error"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str
    target: str
    verbose: bool = True
    max_tokens: int | None = None
    coverage_report: str | None = None
    api_key: str | None = None  # ANTHROPIC_API_KEY, passed to subprocess env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_py_files() -> list[str]:
    """Return repo-relative paths of all .py files (excluding venv/cache)."""
    EXCLUDE = {".venv", "venv", "__pycache__", ".git", ".pytest_cache"}
    result = []
    for p in sorted(BASE_DIR.rglob("*.py")):
        parts = set(p.relative_to(BASE_DIR).parts)
        if parts & EXCLUDE:
            continue
        result.append(str(p.relative_to(BASE_DIR)).replace("\\", "/"))
    return result


def _read_history() -> list[dict]:
    """Read task history from the SQLite memory store."""
    if not MEMORY_DB.exists():
        return []
    try:
        con = sqlite3.connect(str(MEMORY_DB))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT task_id, task_type, target_path, status, output_path, "
            "created_at, updated_at FROM tasks ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _run_agent_thread(run_id: str, cmd: list[str], extra_env: dict | None = None) -> None:
    """Run the agent in a background thread, buffering all output for SSE replay.

    Using a thread (not asyncio subprocess) because on Windows the default
    uvicorn event loop is SelectorEventLoop which does not support
    asyncio.create_subprocess_exec — ProactorEventLoop would be required.
    Threads + blocking subprocess work everywhere without event-loop concerns.
    """
    _run_logs[run_id] = []
    _run_status[run_id] = "running"
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout for simplicity
            cwd=str(BASE_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            text = _ANSI_RE.sub("", line.rstrip())
            if text:
                _run_logs[run_id].append(text)
        proc.wait()
        _run_status[run_id] = "done" if proc.returncode == 0 else "error"
        _run_logs[run_id].append(
            json.dumps({"event": "stream_end", "exit_code": proc.returncode})
        )
    except Exception as exc:
        _run_status[run_id] = "error"
        _run_logs[run_id].append(
            json.dumps({"event": "stream_end", "exit_code": -1, "error": str(exc)})
        )


async def _sse_generator(run_id: str) -> AsyncIterator[str]:
    """Yield SSE-formatted lines for a given run, waiting for new ones."""
    sent = 0
    while True:
        logs = _run_logs.get(run_id, [])
        while sent < len(logs):
            line = logs[sent]
            sent += 1
            yield f"data: {line}\n\n"
        if _run_status.get(run_id) in ("done", "error") and sent >= len(logs):
            break
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not built")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/files")
async def list_files():
    return {"files": _list_py_files()}


@app.get("/api/history")
async def history():
    return {"tasks": _read_history()}


@app.get("/api/diffs")
async def list_diffs():
    diffs = []
    for p in sorted(AGENT_OUTPUT_DIR.glob("*.diff"), key=lambda x: x.stat().st_mtime, reverse=True):
        meta_path = p.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        diffs.append({"name": p.name, "size": p.stat().st_size, "meta": meta})
    return {"diffs": diffs}


@app.get("/api/diff")
async def get_diff(name: str = Query(...)):
    path = AGENT_OUTPUT_DIR / name
    if not path.exists() or not path.suffix == ".diff":
        raise HTTPException(404, "Diff not found")
    return {"content": path.read_text(encoding="utf-8")}


@app.post("/api/run")
async def run_task(req: RunRequest):
    VALID_TASKS = ["lint_fix", "generate_tests", "convert_todos", "triage_issues"]
    if req.task not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task: {req.task}")

    run_id = str(uuid.uuid4())

    cmd = [
        sys.executable, "-m", "agent", "run",
        "--task", req.task,
        "--target", req.target,
    ]
    if req.verbose:
        cmd.append("--verbose")
    if req.max_tokens:
        cmd += ["--max-tokens", str(req.max_tokens)]
    if req.coverage_report:
        cmd += ["--coverage-report", req.coverage_report]

    # Build extra env — inject API key if provided
    extra_env: dict | None = None
    if req.api_key:
        extra_env = {"ANTHROPIC_API_KEY": req.api_key}

    # Fire-and-forget in a daemon thread — frontend streams via /api/stream/{run_id}
    t = threading.Thread(target=_run_agent_thread, args=(run_id, cmd, extra_env), daemon=True)
    t.start()
    return {"run_id": run_id}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str):
    # Wait briefly for the run to be registered
    for _ in range(20):
        if run_id in _run_logs:
            break
        await asyncio.sleep(0.05)
    else:
        raise HTTPException(404, "Run not found")

    return StreamingResponse(
        _sse_generator(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Static files (must come last so /api routes take precedence)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
