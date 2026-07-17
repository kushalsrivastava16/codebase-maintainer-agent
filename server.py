"""
FastAPI web server for the Codebase Maintainer Agent.

Provides:
  - Static file serving for the frontend SPA
  - POST /api/clone     — git clone a GitHub repo into /tmp and return its file list
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
import shutil
import sqlite3
import subprocess
import sys
import tempfile
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

# Cloned GitHub repos: normalised_url -> local absolute path
# Persists for the lifetime of the server process.
_cloned_repos: dict[str, str] = {}
_CLONE_BASE = Path(tempfile.gettempdir()) / "agent_repos"
_CLONE_BASE.mkdir(exist_ok=True)
_clone_lock = asyncio.Lock()  # prevents concurrent clones of the same URL racing

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

class CloneRequest(BaseModel):
    url: str        # GitHub HTTPS URL, e.g. https://github.com/owner/repo
    refresh: bool = False  # if True, re-clone even if already cached


class RunRequest(BaseModel):
    task: str
    target: str
    verbose: bool = True
    max_tokens: int | None = None
    coverage_report: str | None = None
    api_key: str | None = None         # ANTHROPIC_API_KEY, passed to subprocess env
    repo_path: str | None = None       # absolute local path to a repo
    repo_url: str | None = None        # GitHub URL — resolved to cloned path at run time
    github_repo: str | None = None     # "owner/repo" — required for triage_issues
    github_token: str | None = None    # GITHUB_TOKEN for triage_issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_github_url(url: str) -> str:
    """Strip trailing .git and whitespace so the same repo always maps to one cache key."""
    return url.strip().rstrip("/").removesuffix(".git")


def _list_py_files(base: Path | None = None) -> list[str]:
    """Return repo-relative paths of all .py files under base (default: this project)."""
    EXCLUDE = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", ".tox", "node_modules"}
    root = base or BASE_DIR
    result = []
    for p in sorted(root.rglob("*.py")):
        parts = set(p.relative_to(root).parts)
        if parts & EXCLUDE:
            continue
        result.append(str(p.relative_to(root)).replace("\\", "/"))
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


def _run_agent_thread(
    run_id: str,
    cmd: list[str],
    extra_env: dict | None = None,
    cwd: str | None = None,
) -> None:
    """Run the agent in a background thread, buffering all output for SSE replay.

    cwd defaults to BASE_DIR (this project). Pass an external repo path to run
    the agent against a different codebase — the agent uses Path.cwd() as its
    repo_root, so changing cwd is all that's needed.
    """
    _run_logs[run_id] = []
    _run_status[run_id] = "running"
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        # Always make the agent package importable, even when cwd is an external repo
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(BASE_DIR) + os.pathsep + existing_pythonpath
            if existing_pythonpath
            else str(BASE_DIR)
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout for simplicity
            cwd=cwd or str(BASE_DIR),
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


@app.post("/api/clone")
async def clone_repo(req: CloneRequest):
    """Clone a public GitHub repository (shallow, depth=1) and return its .py file list.

    Caches the clone for the server process lifetime; pass refresh=true to re-clone.
    Uses an asyncio lock to prevent concurrent requests for the same URL racing on
    rmtree + git clone.
    """
    url = _normalise_github_url(req.url)

    if not (url.startswith("https://github.com/") or url.startswith("http://github.com/")):
        raise HTTPException(400, "Only https://github.com/ URLs are supported")

    slug = url.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    parts = slug.split("/")
    if len(parts) < 2:
        raise HTTPException(400, "URL must be https://github.com/owner/repo")

    folder_name = f"{parts[0]}_{parts[1]}"
    target = _CLONE_BASE / folder_name

    # Fast-path: serve from cache without acquiring the lock
    if not req.refresh and url in _cloned_repos and Path(_cloned_repos[url]).exists():
        files = _list_py_files(Path(_cloned_repos[url]))
        return {"files": files, "repo_root": _cloned_repos[url], "cached": True, "repo_name": folder_name}

    if shutil.which("git") is None:
        raise HTTPException(500, "git is not available on this server")

    async with _clone_lock:
        # Re-check cache after acquiring lock in case another request just finished
        if not req.refresh and url in _cloned_repos and Path(_cloned_repos[url]).exists():
            files = _list_py_files(Path(_cloned_repos[url]))
            return {"files": files, "repo_root": _cloned_repos[url], "cached": True, "repo_name": folder_name}

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch",
             "--filter=blob:limit=10m", url, str(target)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            # Surface common errors with a clear message
            if "Repository not found" in err or "not found" in err.lower():
                raise HTTPException(404, "Repository not found or is private")
            if "rate limit" in err.lower():
                raise HTTPException(429, "GitHub rate limit hit — try again in a minute")
            raise HTTPException(400, f"Clone failed: {err[:400]}")

        _cloned_repos[url] = str(target)

    files = _list_py_files(target)
    return {"files": files, "repo_root": str(target), "cached": False, "repo_name": folder_name}


@app.get("/api/files")
async def list_files(repo: str | None = Query(None)):
    """List .py files in the project (default) or in an external repository.

    Pass ?repo=/absolute/path/to/repo to scan a different codebase.
    Returns {files: [...], repo_root: "..."} so the UI can show which root
    is active.
    """
    if repo:
        repo_path = Path(repo)
        if not repo_path.exists():
            raise HTTPException(400, f"Path does not exist: {repo}")
        if not repo_path.is_dir():
            raise HTTPException(400, f"Path is not a directory: {repo}")
        return {"files": _list_py_files(repo_path), "repo_root": str(repo_path)}
    return {"files": _list_py_files(), "repo_root": str(BASE_DIR)}


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
    # Resolve and check the path is within AGENT_OUTPUT_DIR to prevent traversal
    path = (AGENT_OUTPUT_DIR / name).resolve()
    if not path.is_relative_to(AGENT_OUTPUT_DIR.resolve()):
        raise HTTPException(400, "Invalid diff name")
    if not path.exists() or path.suffix != ".diff":
        raise HTTPException(404, "Diff not found")
    return {"content": path.read_text(encoding="utf-8")}


@app.post("/api/run")
async def run_task(req: RunRequest):
    VALID_TASKS = ["lint_fix", "generate_tests", "convert_todos", "triage_issues"]
    if req.task not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task: {req.task}")

    # triage_issues requires a GitHub repo slug ("owner/repo")
    if req.task == "triage_issues":
        if not req.github_repo:
            raise HTTPException(400, "github_repo (owner/repo) is required for triage_issues")
        if "/" not in req.github_repo or req.github_repo.count("/") != 1:
            raise HTTPException(400, "github_repo must be in 'owner/repo' format")

    # Reject absolute paths and path traversal — target must be relative and
    # stay within the repo root (enforced again by FileReader, but fail early here)
    target_path = Path(req.target)
    if target_path.is_absolute():
        raise HTTPException(400, "target must be a relative path, not absolute")
    if any(part == ".." for part in target_path.parts):
        raise HTTPException(400, "target path cannot contain '..'")

    run_id = str(uuid.uuid4())

    cmd = [
        sys.executable, "-m", "agent", "run",
        "--task", req.task,
        "--target", req.target,
        # Always use the project's own agent_config.yaml, not one that may exist
        # inside a cloned repo (which could override model, max_iterations, etc.)
        "--config", str(BASE_DIR / "agent_config.yaml"),
    ]
    if req.verbose:
        cmd.append("--verbose")
    if req.max_tokens:
        cmd += ["--max-tokens", str(req.max_tokens)]
    if req.coverage_report:
        cmd += ["--coverage-report", req.coverage_report]
    if req.github_repo:
        cmd += ["--github-repo", req.github_repo]

    # Build extra env — inject API key and GitHub token if provided
    extra_env: dict = {}
    if req.api_key:
        extra_env["ANTHROPIC_API_KEY"] = req.api_key
    if req.github_token:
        extra_env["GITHUB_TOKEN"] = req.github_token
    extra_env = extra_env or None

    # Resolve cwd: repo_url (cloned) > repo_path (local) > this project
    cwd: str | None = None
    if req.repo_url:
        norm = _normalise_github_url(req.repo_url)
        cloned = _cloned_repos.get(norm)
        if not cloned or not Path(cloned).exists():
            raise HTTPException(400, "Repository not cloned yet — call POST /api/clone first")
        cwd = cloned
    elif req.repo_path:
        rp = Path(req.repo_path)
        if not rp.exists() or not rp.is_dir():
            raise HTTPException(400, f"Repository path does not exist: {req.repo_path}")
        cwd = str(rp)

    # Fire-and-forget in a daemon thread — frontend streams via /api/stream/{run_id}
    t = threading.Thread(target=_run_agent_thread, args=(run_id, cmd, extra_env, cwd), daemon=True)
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
