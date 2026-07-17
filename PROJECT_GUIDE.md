# Codebase Maintainer Agent — Complete Project Guide

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How It Works — End to End](#3-how-it-works--end-to-end)
4. [Architecture & File Map](#4-architecture--file-map)
5. [Tech Stack — From Basics to This Project](#5-tech-stack--from-basics-to-this-project)
6. [API Reference — Every Endpoint](#6-api-reference--every-endpoint)
7. [Deployment Guide (Render Free Tier)](#7-deployment-guide-render-free-tier)
8. [How to Use the Agent on a Different Codebase](#8-how-to-use-the-agent-on-a-different-codebase)
9. [Configuration Reference](#9-configuration-reference)
10. [Exit Codes & Debugging](#10-exit-codes--debugging)

---

## 1. What This Project Is

The **Codebase Maintainer Agent** is an AI-powered software engineering assistant that automatically performs routine maintenance tasks on Python codebases. It uses Anthropic's Claude LLM as its "brain" and wraps it in a tool-use loop so the model can read files, run checks, and propose code fixes — all without human intervention.

Tasks it can perform:

| Task | What It Does |
|---|---|
| `lint_fix` | Reads a file, finds ruff violations (unused imports, bad style), proposes a fixed version |
| `generate_tests` | Reads source + coverage report, generates pytest tests for uncovered lines |
| `convert_todos` | Scans a file for TODO comments, converts them to structured issues |
| `triage_issues` | Reads open GitHub issues via the GitHub API, classifies and prioritises them |

The web frontend lets you pick a task, pick a file, paste your API key, hit Run, and watch the agent work in real time.

---

## 2. The Problem It Solves

In real software teams, code quality work is tedious and repetitive:
- Developers forget to remove unused imports before pushing.
- Test coverage stays low because writing tests is boring.
- TODO comments rot in the codebase for months.
- GitHub issue triage takes hours of reading and labelling.

These tasks require reading code, understanding context, and making mechanical changes — exactly what an LLM is good at. This project automates that loop:

```
Human notices problem
  → Human asks agent
    → Agent reads code
      → Agent proposes fix
        → Human reviews diff
          → Human applies if correct
```

The output is always a `.diff` file — the agent never modifies source files directly. A human reviews and applies. This keeps the LLM in an advisory role where it belongs.

---

## 3. How It Works — End to End

### Step-by-step for `lint_fix`

```
1. You click "Run" in the web UI
   └─ Frontend POSTs to /api/run with {task, target, api_key}

2. FastAPI (server.py) receives the request
   └─ Generates a run_id (UUID)
   └─ Spawns a background thread running:
      python -m agent run --task lint_fix --target tests/fixtures/...

3. The agent CLI (__main__.py) starts
   └─ Loads config (defaults + agent_config.yaml + CLI flags)
   └─ Checks SQLite dedup — skip if same task ran successfully in last 24h
   └─ Reads original file content for later diffing
   └─ Builds tools: {read_file: FileReader, run_lint: LintRunner}
   └─ Creates Orchestrator and calls .run()

4. Orchestrator loop iteration 0
   └─ Sends to Claude: "Task: lint_fix, Target: ..., fix this file"
   └─ Claude responds: stop_reason=tool_use, calls read_file('tests/fixtures/...')
   └─ FileReader reads and sanitises the file, returns content
   └─ Claude responds: stop_reason=tool_use, calls run_lint('tests/fixtures/...')
   └─ LintRunner runs `ruff check` + `ruff format --check`, returns violations

5. Orchestrator loop iteration 1
   └─ Claude has the file + violations in context
   └─ Claude responds: stop_reason=end_turn, with a ```python code block```

6. Self-correction check
   └─ Orchestrator extracts code block
   └─ Runs `ruff check --fix --unsafe-fixes` on proposed content (temp file)
   └─ Runs `ruff check` again to verify
   └─ If current_codes is empty → CLEAN → write diff
   └─ If current_codes non-empty → feed violations back to LLM, retry (max 3x)

7. DiffWriter generates unified diff (original vs proposed)
   └─ Writes agent_output/lint_fix_filename_20260717T221948Z.diff
   └─ Writes companion .json with metadata (tokens, cost, model, timestamp)

8. SQLite MemoryStore records task as success

9. Thread finishes, appends {"event":"stream_end","exit_code":0} to log buffer

10. Frontend EventSource receives stream_end, shows "✓ done"
    └─ Diff tab auto-loads the new diff file
```

### The LLM Tool-Use Loop

Claude does not have internet access or file system access. It can only reason about text in its context. The tool-use protocol lets it request actions:

```
Agent sends:  [system prompt] + [user message with task]
Claude says:  "I want to call read_file with path=..."
Agent runs:   FileReader.execute({"path": "..."})
Agent sends:  [same history] + [tool result]
Claude says:  "I want to call run_lint with path=..."
Agent runs:   LintRunner.execute({"path": "..."})
Agent sends:  [same history] + [tool result]
Claude says:  "Here is the fixed file: ```python ... ```"
Agent:        Extracts code, verifies with ruff, writes diff
```

This is the standard Anthropic tool-use pattern. The Orchestrator is a `while True` loop that dispatches tool calls and feeds results back until `stop_reason == "end_turn"`.

---

## 4. Architecture & File Map

```
Project/
├── agent/                    ← All agent logic
│   ├── __main__.py           ← CLI entry point (click commands: run, eval)
│   ├── config.py             ← Configuration loading (defaults → YAML → CLI)
│   ├── orchestrator.py       ← LLM tool-use loop + self-correction
│   ├── logger.py             ← Structured JSON logger (outputs to stderr/stdout)
│   ├── cost_tracker.py       ← Token usage + USD cost tracking
│   ├── diff_writer.py        ← Generates unified diffs, writes .diff + .json
│   ├── memory.py             ← SQLite task history + deduplication
│   ├── protocols.py          ← Protocol interfaces (Tool, ToolResult, TaskResult)
│   ├── sandbox.py            ← Docker sandbox for isolated code execution
│   ├── triage.py             ← GitHub issue triage helpers
│   ├── tools/
│   │   ├── file_reader.py    ← read_file tool: safe file reading with security guards
│   │   ├── lint_runner.py    ← run_lint tool: ruff check + ruff format --check
│   │   ├── coverage_reader.py← read_coverage tool: parses pytest-cov JSON
│   │   ├── todo_scanner.py   ← scan_todos tool: finds TODO/FIXME comments
│   │   └── github_client.py  ← list_issues tool: reads GitHub issues via API
│   └── eval/
│       └── harness.py        ← Evaluation harness for benchmarking
│
├── server.py                 ← FastAPI web server (REST + SSE)
├── frontend/
│   └── index.html            ← Single Page App (all CSS + JS inline)
│
├── tests/
│   ├── fixtures/
│   │   └── utils_with_lint_errors.py  ← Demo file with intentional violations
│   └── test_*.py             ← Pytest test suite
│
├── agent_output/             ← Generated .diff and .json files go here
├── agent_memory.db           ← SQLite database (task history + dedup)
├── agent_config.yaml         ← Optional YAML config overrides
├── requirements.txt          ← Python dependencies
├── render.yaml               ← Render deployment config
└── Procfile                  ← Process start command for Render
```

---

## 5. Tech Stack — From Basics to This Project

### Python
**What it is:** General-purpose programming language.
**How it's used:** Every component is Python. The agent, the web server, the tools, the CLI — all Python 3.11+.

---

### Anthropic Claude API
**What it is:** Anthropic's REST API for the Claude family of LLMs.
**Basics:** You send HTTP POST to `api.anthropic.com/v1/messages` with a model name, a system prompt, a list of messages, and optional tool definitions. You get back a response with text or tool call requests.
**How it's used:**
- Model: `claude-haiku-4-5-20251001` (fast, cheap, good at mechanical code tasks)
- Tools are declared as JSON schema — Claude picks which tool to call, the agent executes it
- `anthropic.NOT_GIVEN` is passed instead of an empty tools list to suppress tool use during self-correction turns
- Cost: ~$0.004 per lint_fix run (3,000 tokens at Haiku pricing)

```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    system="You are a code maintainer...",
    tools=[{"name": "read_file", "description": "...", "input_schema": {...}}],
    messages=[{"role": "user", "content": "Fix lint errors in foo.py"}],
)
```

---

### FastAPI
**What it is:** A modern Python web framework for building REST APIs.
**Basics:** You define functions decorated with `@app.get("/path")` or `@app.post("/path")`. FastAPI automatically validates request/response shapes using Pydantic models and generates OpenAPI docs at `/docs`.
**How it's used:** `server.py` is a FastAPI app. It has 7 routes (see API Reference below). The key feature used here is `StreamingResponse` with `media_type="text/event-stream"` for real-time log streaming.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/api/run")
async def run_task(req: RunRequest):
    run_id = str(uuid.uuid4())
    threading.Thread(target=_run_agent_thread, args=(run_id, cmd)).start()
    return {"run_id": run_id}
```

---

### Uvicorn
**What it is:** An ASGI web server (runs FastAPI apps).
**Basics:** FastAPI is a framework that defines what your app does. Uvicorn is the server that actually listens on a port and handles HTTP connections. Like Django needs gunicorn, FastAPI needs uvicorn.
**How it's used:** Start command is `uvicorn server:app --host 0.0.0.0 --port $PORT`. On Render, `$PORT` is set by the platform.

---

### Server-Sent Events (SSE)
**What it is:** A web protocol where the server streams a continuous text response to the browser, one line at a time, without the browser re-requesting.
**Basics:** The browser opens an `EventSource("url")` connection. The server responds with `Content-Type: text/event-stream` and sends lines formatted as `data: <payload>\n\n`. The browser fires an `onmessage` event for each line.
**Why not WebSockets?** SSE is one-directional (server → client), simpler, and works over HTTP/1.1. WebSockets need a separate handshake and are overkill for log streaming.
**How it's used:**

Server side (`server.py`):
```python
async def _sse_generator(run_id):
    sent = 0
    while True:
        logs = _run_logs.get(run_id, [])
        while sent < len(logs):
            yield f"data: {logs[sent]}\n\n"
            sent += 1
        if _run_status.get(run_id) in ("done", "error"):
            break
        await asyncio.sleep(0.1)

@app.get("/api/stream/{run_id}")
async def stream(run_id: str):
    return StreamingResponse(_sse_generator(run_id), media_type="text/event-stream")
```

Client side (`frontend/index.html`):
```javascript
const es = new EventSource(`/api/stream/${runId}`);
es.onmessage = (e) => {
    const event = JSON.parse(e.data);
    renderLogCard(event);
};
```

---

### Threading vs asyncio
**What it is:** Python has two concurrency models — threads (OS-level, true parallelism for I/O) and asyncio (cooperative coroutines, single-threaded).
**The problem:** `subprocess.Popen` is blocking. In an async FastAPI handler, calling a blocking function freezes the event loop. The standard fix is `asyncio.create_subprocess_exec`, but on Windows, uvicorn runs a `SelectorEventLoop` which doesn't support subprocess. Only `ProactorEventLoop` does.
**How it's used:** The agent subprocess runs in a `threading.Thread` (a separate OS thread). The thread blocks on `proc.stdout` line-by-line, while the FastAPI event loop stays unblocked and serves other requests. Logs are accumulated in a list. The SSE generator reads from that list asynchronously.

---

### Ruff
**What it is:** An extremely fast Python linter and formatter, written in Rust. Replaces flake8, pylint, isort, and black.
**Basics:**
- `ruff check file.py` — finds code style violations (F401 = unused import, F841 = unused variable, E501 = line too long, etc.)
- `ruff check --fix file.py` — auto-fixes violations it knows how to fix safely
- `ruff check --fix --unsafe-fixes file.py` — also fixes violations that are safe but marked "unsafe" (like removing unused variables where the RHS might have side effects)
- `ruff format --check file.py` — checks if the file would be reformatted (like black)
- Exit code 0 = clean, 1 = violations found, 2 = internal error
**How it's used:** Two places:
1. `LintRunner` tool — runs ruff on the original file, gives violations to the LLM
2. `Orchestrator._apply_ruff_fixes()` + `_check_proposed_content()` — runs ruff on the LLM's proposed fix before accepting it

---

### SQLite (via Python's `sqlite3` module)
**What it is:** A file-based relational database, built into Python. No separate server needed.
**Basics:** You connect with `sqlite3.connect("file.db")`, execute SQL, and commit. WAL (Write-Ahead Logging) mode allows concurrent reads.
**How it's used:** `agent/memory.py` maintains a `tasks` table with columns: `task_id`, `task_type`, `target_path`, `status`, `created_at`, `updated_at`, `output_path`. Used for:
- **Deduplication:** if the same `(task_type, target_path)` succeeded in the last 24 hours, skip the run
- **History:** the `/api/history` endpoint queries this table for the task history sidebar
- **Corruption recovery:** on startup, if the DB file is corrupt, it's renamed and a fresh one is created

---

### Click
**What it is:** Python library for building command-line interfaces with decorators.
**How it's used:** `agent/__main__.py` defines two CLI commands:
- `python -m agent run --task lint_fix --target foo.py`
- `python -m agent eval --benchmark benchmark.yaml`

---

### Pydantic
**What it is:** Python data validation library. FastAPI uses it to parse and validate request/response JSON automatically.
**How it's used:** `RunRequest` in `server.py` is a Pydantic model. When you POST JSON to `/api/run`, FastAPI automatically validates it against the model and returns 422 with details if it's invalid.

```python
class RunRequest(BaseModel):
    task: str
    target: str
    verbose: bool = True
    max_tokens: int | None = None
    api_key: str | None = None
```

---

### Unified Diff Format
**What it is:** A standard text format for showing what changed between two versions of a file. Used by git, patch, and code review tools.
**Basics:** Lines starting with `-` are removed, `+` are added, ` ` (space) are context (unchanged), `@@` headers show line numbers.
**How it's used:** `agent/diff_writer.py` uses Python's built-in `difflib.unified_diff()` to compute the diff between `original_content` and `proposed` (LLM's output). Written to `agent_output/*.diff`.

---

### Render (deployment platform)
**What it is:** A cloud platform for deploying web services. Free tier supports one always-on web service.
**How it's used:** `render.yaml` tells Render how to build and start the app:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- `ANTHROPIC_API_KEY` is set as a secret environment variable in the Render dashboard
**Free tier limits:** 512 MB RAM, shared CPU, service sleeps after 15 min of inactivity (cold start ~30s)

---

## 6. API Reference — Every Endpoint

Base URL: `https://your-render-app.onrender.com` (or `http://localhost:8000` locally)

---

### `GET /`
**Purpose:** Serve the frontend SPA.
**Returns:** HTML (the full `frontend/index.html` content)
**Used by:** Browser navigating to the app URL

---

### `GET /api/files`
**Purpose:** List all Python files in the project that can be targeted.
**Returns:**
```json
{
  "files": [
    "agent/__main__.py",
    "agent/config.py",
    "tests/fixtures/utils_with_lint_errors.py"
  ]
}
```
**Used by:** Frontend dropdown on page load. Files are relative to the project root. Hidden directories (`.venv`, `__pycache__`, `.git`) are excluded.

---

### `POST /api/run`
**Purpose:** Start a task run.
**Request body:**
```json
{
  "task": "lint_fix",
  "target": "tests/fixtures/utils_with_lint_errors.py",
  "verbose": true,
  "max_tokens": null,
  "coverage_report": null,
  "api_key": "sk-ant-..."
}
```
**Field notes:**
- `task`: one of `lint_fix`, `generate_tests`, `convert_todos`, `triage_issues`
- `target`: path relative to project root (must match a value from `/api/files`)
- `api_key`: your Anthropic API key — passed as `ANTHROPIC_API_KEY` env var to the subprocess. If not provided, the server's own env var is used (on Render, set in dashboard)
- `coverage_report`: only needed for `generate_tests`

**Returns:**
```json
{"run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"}
```

**Error cases:**
- `400` if `task` is not one of the 4 valid values

---

### `GET /api/stream/{run_id}`
**Purpose:** Stream real-time logs for a run using Server-Sent Events.
**Path param:** `run_id` from the `/api/run` response.
**Response:** `Content-Type: text/event-stream`, each event is a JSON line:

```
data: {"event":"task_start","level":"INFO","task_type":"lint_fix","target_path":"..."}

data: {"event":"llm_call","level":"INFO","call_number":0,"model":"claude-haiku-4-5-20251001"}

data: {"event":"llm_response","level":"INFO","stop_reason":"tool_use","input_tokens":995}

data: {"event":"tool_dispatch","level":"INFO","tool_name":"read_file"}

data: {"event":"tool_result","level":"INFO","tool_name":"read_file","result_summary":"..."}

data: {"event":"token_usage","level":"INFO","cumulative_tokens":3219,"cumulative_usd_cost":0.0039}

data: {"event":"task_success","level":"INFO","output_path":"/opt/render/.../lint_fix_foo.diff"}

data: {"event":"stream_end","exit_code":0}
```

**How to consume in JavaScript:**
```javascript
const es = new EventSource(`/api/stream/${runId}`);
es.onmessage = (e) => {
    const obj = JSON.parse(e.data);
    if (obj.event === "stream_end") { es.close(); return; }
    // render obj.event, obj.level, other fields...
};
```

---

### `GET /api/history`
**Purpose:** Get the last 50 task runs from the SQLite memory store.
**Returns:**
```json
{
  "tasks": [
    {
      "task_id": "659946d2-...",
      "task_type": "lint_fix",
      "target_path": "/opt/render/.../utils_with_lint_errors.py",
      "status": "success",
      "output_path": "/opt/render/.../lint_fix_foo.diff",
      "created_at": "2026-07-17T22:19:48.123",
      "updated_at": "2026-07-17T22:19:50.456"
    }
  ]
}
```

---

### `GET /api/diffs`
**Purpose:** List all generated diff files.
**Returns:**
```json
{
  "diffs": [
    {
      "name": "lint_fix_utils_with_lint_errors_20260717T221948Z.diff",
      "size": 842,
      "meta": {
        "task_type": "lint_fix",
        "model": "claude-haiku-4-5-20251001",
        "total_tokens": 3219,
        "estimated_usd": 0.0039
      }
    }
  ]
}
```

---

### `GET /api/diff?name=<filename>`
**Purpose:** Get the raw content of a specific diff file.
**Query param:** `name` — filename from `/api/diffs` response.
**Returns:**
```json
{
  "content": "--- a/tests/fixtures/utils_with_lint_errors.py\n+++ b/...\n@@...\n-import os\n..."
}
```
**Error:** `404` if file not found or not a `.diff` file.

---

### `GET /docs`
**Purpose:** Auto-generated interactive API documentation (Swagger UI).
**Available at:** `http://localhost:8000/docs` — lets you try all endpoints in the browser with forms.

---

## 7. Deployment Guide (Render Free Tier)

### First-time setup

1. Push your code to a GitHub repository.

2. Go to [render.com](https://render.com) → New → Web Service → Connect your GitHub repo.

3. Render reads `render.yaml` automatically and configures:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

4. In the Render dashboard → Environment → Add secret variable:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-api03-...` (your actual key)

5. Click **Deploy**. First deploy takes ~3 minutes. Your URL is `https://your-service-name.onrender.com`.

### Redeploying after a code change

```bash
git add .
git commit -m "your message"
git push origin main
```

Render auto-deploys on every push to `main`. Or manually: Dashboard → Manual Deploy → Deploy latest commit.

### Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
# or
export ANTHROPIC_API_KEY="sk-ant-..."   # bash

# Start the server
uvicorn server:app --reload --port 8000

# Open in browser
# http://localhost:8000
```

### Running the agent CLI directly (no web UI)

```bash
# Lint fix
python -m agent run --task lint_fix --target tests/fixtures/utils_with_lint_errors.py

# Generate tests (requires coverage report)
pytest --cov=agent --cov-report=json:coverage.json
python -m agent run --task generate_tests --target agent/config.py --coverage-report coverage.json

# Convert TODOs
python -m agent run --task convert_todos --target agent/orchestrator.py

# Triage GitHub issues
python -m agent run --task triage_issues --github-repo owner/repo-name
```

---

## 8. How to Use the Agent on a Different Codebase

### Why the dropdown only shows this project's files

The `/api/files` endpoint calls `BASE_DIR.rglob("*.py")` where `BASE_DIR` is the directory where `server.py` lives. On Render that is `/opt/render/project/src/`. So the dropdown only shows files that are deployed alongside the server.

This is by design for the demo — but there are three ways to use the agent on external codebases:

---

### Option A — CLI directly (any codebase, no web UI)

Run the agent CLI from inside any Python project directory. It uses `Path.cwd()` as the repo root, so it can read any file in that directory.

```bash
# Navigate to your other project
cd C:/Users/you/your-other-project

# Run the agent (with ANTHROPIC_API_KEY set)
python -m agent run --task lint_fix --target src/utils.py
# or target a whole directory
python -m agent run --task lint_fix --target src/
```

The diff is written to `./agent_output/` inside that project. No web UI needed.

---

### Option B — Deploy the server alongside your codebase

Instead of deploying the agent as a standalone service, add `server.py`, `requirements.txt`, and `render.yaml` to your actual project's repository. When deployed, the `/api/files` dropdown will list your project's Python files.

Steps:
1. Copy `server.py`, `frontend/`, `agent/`, `requirements.txt`, `render.yaml`, `Procfile` into your target project's repo.
2. Push and deploy to Render.
3. The agent now sees and can target your project's files.

---

### Option C — Add a `repo_path` parameter to the API (recommended extension)

Modify `server.py` and the frontend to accept an external repository path or URL:

**In `server.py`**, change `RunRequest` to add a `repo_path` field:
```python
class RunRequest(BaseModel):
    task: str
    target: str
    repo_path: str | None = None   # e.g. "/home/user/my-project"
    api_key: str | None = None
```

**In `_run_agent_thread`**, set `cwd=req.repo_path or str(BASE_DIR)` in `subprocess.Popen`.

**In the frontend**, add a text input for "Repository path" next to the file dropdown. When filled in, the dropdown can be replaced by a free-text path input.

**The `/api/files` endpoint** would need to accept a `?repo=` query parameter and list files from that directory:
```python
@app.get("/api/files")
async def list_files(repo: str | None = None):
    base = Path(repo) if repo else BASE_DIR
    # ... rglob("*.py") on base
```

This approach works for any local repository path accessible to the server process.

---

### Option D — GitHub integration (cloud repos)

For repositories that live on GitHub but are not deployed locally:

1. Add a `git clone` step to the agent startup — clone the target repo into a temp directory.
2. Run the agent against the cloned files.
3. Instead of writing a local diff, open a GitHub PR with the changes using the GitHub API.

This is the full "agentic code review bot" pattern. The `GitHubClient` tool in `agent/tools/github_client.py` already handles GitHub API auth — the same pattern can be extended for creating PRs.

---

### Summary: which option to choose

| Your situation | Best option |
|---|---|
| Quick test on a local project | Option A (CLI) |
| Demo/portfolio with your own code | Option B (deploy alongside) |
| You want a proper multi-repo web UI | Option C (add repo_path param) |
| You want a GitHub bot | Option D (clone + PR) |

---

## 9. Configuration Reference

Create `agent_config.yaml` in the project root to override defaults without changing code:

```yaml
# agent_config.yaml
model: claude-haiku-4-5-20251001    # or claude-sonnet-5 for harder tasks
max_tokens_per_task: 50000          # abort if cumulative tokens exceed this
max_iterations: 10                  # abort if loop runs this many turns
output_dir: ./agent_output          # where .diff files are written
memory_db: ./agent_memory.db        # SQLite task history
dedup_window_hours: 24              # skip re-run if success in last 24h
log_level: INFO                     # DEBUG, INFO, WARNING, ERROR
sandbox_enabled: false              # true = run ruff checks in Docker
github_repo: owner/repo             # for triage_issues task
```

**Config merge order (later wins):**
1. Built-in defaults (in `agent/config.py`)
2. `agent_config.yaml` (if it exists)
3. CLI flags (`--max-tokens`, `--sandbox`, etc.)

---

## 10. Exit Codes & Debugging

| Exit code | Meaning |
|---|---|
| `0` | Task succeeded, diff written |
| `1` | Invalid arguments or missing target file |
| `3` | Task failed (max retries, or agent gave up) |
| `5` | Token budget exceeded |
| `6` | Config file parse error or invalid value |

**Common issues:**

| Symptom | Cause | Fix |
|---|---|---|
| `dedup_skip` — task skipped immediately | Same task ran successfully in last 24h | Delete `agent_memory.db` or wait 24h |
| `file_not_found` on every read_file | Wrong `repo_root` or wrong path passed to `--target` | Ensure `--target` is relative to CWD |
| `ABORTED max retries reached` | LLM keeps producing code with ruff violations | Check `ruff_check_failed` event in logs for exact violations |
| `404` from Anthropic API | Wrong model name | Use `claude-haiku-4-5-20251001` |
| Cold start delay on Render | Free tier sleeps after 15 min idle | First request takes ~30s; subsequent requests are fast |
| Log stream shows ANSI codes | ANSI stripping regex missed a pattern | Check `_ANSI_RE` in `server.py` |

---

*Document generated from the live project codebase — Codebase Maintainer Agent v1.0, July 2026.*
