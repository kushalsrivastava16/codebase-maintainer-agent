# Requirements Document

## Introduction

The Codebase Maintainer Agent is an autonomous software maintenance system built as a portfolio project. It targets a real or sample GitHub repository and performs maintenance tasks including issue triage, lint error remediation, test generation for uncovered code paths, and TODO-to-fix conversion. The agent proposes changes as unified diffs or draft pull requests rather than applying them directly, keeping a human in the loop for all modifications.

The project is structured across five incremental phases. Phase 1 establishes a transparent, raw-Python orchestration loop to maximize learning. Subsequent phases layer in persistent memory, self-correction, GitHub API integration, Docker sandboxing, cost budgeting, and an evaluation harness. The final phase produces documented engineering insights for the practitioner's portfolio.

---

## Glossary

- **Agent**: The autonomous software system that reads a repository, reasons about maintenance tasks, and proposes changes.
- **Orchestrator**: The top-level Python class that manages the agent loop — iterating over LLM responses, dispatching tool calls, and enforcing abort guards.
- **LLM**: Large Language Model; specifically Claude via the Anthropic API, used as the agent's reasoning engine.
- **Tool**: A discrete capability exposed to the LLM (e.g., `read_file`, `run_lint`). Each tool has a defined input schema and return schema.
- **Task**: A single unit of work the agent is asked to perform (e.g., "fix lint errors in `utils.py`").
- **Diff**: A unified diff (GNU diff `-u` format) representing a proposed change to one or more files.
- **Memory Store**: A SQLite database that persists task history, outcomes, and deduplication records.
- **Coverage Report**: A JSON file produced by `pytest-cov` describing which source lines are covered by existing tests.
- **TODO Comment**: A Python source comment beginning with `# TODO` that describes work to be done.
- **Triage**: The act of reading a GitHub issue and posting a structured analysis comment with priority, type, and reproduction status.
- **Eval Harness**: A test suite of fixed benchmark tasks with pass/fail criteria used to measure agent quality over time.
- **Sandbox**: An isolated Docker container used to execute agent-proposed code changes without modifying the host filesystem.
- **Token Budget**: A configurable maximum number of LLM tokens that may be consumed per task run.
- **Ruff**: A fast Python linter and formatter invoked by the agent to detect style and type errors.
- **Draft PR**: A GitHub pull request marked as a draft, signaling it is not yet ready for merge.
- **agent_output**: The local directory where the agent writes diff files and metadata JSON.
- **GITHUB_TOKEN**: An environment variable containing a GitHub personal access token with the required repository scopes.
- **Structured Log**: A JSON-formatted log entry written to stderr containing event type, timestamp, and relevant payload.

---

## Requirements

### Requirement 1: Single-Task Execution Loop

**User Story:** As a practitioner learning agent engineering, I want a transparent Python orchestration loop that runs a single maintenance task end-to-end, so that I can understand every step of the agent cycle without framework abstractions.

#### Acceptance Criteria

1. THE Orchestrator SHALL expose a CLI entry point (`python -m agent run --task <task_type> --target <path>`) that accepts a task type and a target file or directory path.
2. WHEN the CLI is invoked, THE Orchestrator SHALL instantiate exactly one tool appropriate for the specified task type and pass it to the LLM context.
3. WHEN the LLM returns a tool-call response, THE Orchestrator SHALL dispatch the call to the corresponding tool, collect the result, and append both the tool call and the result to the conversation context before the next LLM call.
4. WHEN the agent loop has executed 10 LLM iterations without producing a final answer, THE Orchestrator SHALL abort the loop, emit an `abort` structured log event with the reason `max_iterations_exceeded`, and exit with a non-zero status code.
5. WHEN a task completes successfully within the iteration limit, THE Orchestrator SHALL complete the full loop in under 60 seconds on a machine with a standard broadband connection, excluding LLM API latency.
6. IF the `--task` argument is not one of the supported task types, THEN THE Orchestrator SHALL print the list of supported task types to stderr and exit with status code 1 without invoking the LLM.
7. IF the `--target` path does not exist on the filesystem, THEN THE Orchestrator SHALL exit with status code 1 and emit an error log entry before invoking the LLM.

---

### Requirement 2: File Reading Tool

**User Story:** As a practitioner, I want a file-reading tool that safely exposes repository source code to the LLM, so that the agent can reason about file contents without risking path traversal attacks or prompt injection.

#### Acceptance Criteria

1. WHEN the LLM invokes the `read_file` tool with a file path, THE File_Reader SHALL resolve the path relative to the repository root and return the file contents as a UTF-8 string.
2. WHEN the resolved absolute path falls outside the repository root directory, THE File_Reader SHALL return an error result containing the message `path_traversal_denied` and SHALL NOT read any file contents.
3. WHEN the file to be read is larger than 100 KB, THE File_Reader SHALL return the first 100 KB of the file content and append a truncation notice `[TRUNCATED: file exceeds 100 KB limit]` to the returned content.
4. WHEN file content is returned to the LLM, THE File_Reader SHALL strip any substring matching the pattern `\[INST\]`, `</s>`, `<s>`, `###`, or `Human:` to mitigate prompt injection, and SHALL append a sanitization notice `[SANITIZED: N injection patterns removed]` where N is the count of removed patterns.
5. IF the target file does not exist, THEN THE File_Reader SHALL return an error result containing the message `file_not_found: <path>`.
6. IF the target file cannot be read due to a permission error, THEN THE File_Reader SHALL return an error result containing the message `permission_denied: <path>`.

---

### Requirement 3: Lint Error Detection and Fix Proposal

**User Story:** As a practitioner, I want the agent to detect lint errors using ruff and propose a unified diff that resolves them, so that I can review and apply clean fixes without running linters manually.

#### Acceptance Criteria

1. WHEN the LLM invokes the `run_lint` tool with a target path, THE Lint_Tool SHALL invoke `ruff check <path>` and `ruff format --check <path>` as subprocess calls and collect their combined stdout output.
2. WHEN ruff reports one or more violations, THE Lint_Tool SHALL return the full ruff output to the LLM so it can reason about the specific violations and line numbers.
3. WHEN the LLM produces a fix, THE Orchestrator SHALL generate a unified diff between the original file content and the proposed fixed content and write it to `agent_output`.
4. WHEN ruff reports zero violations on the target path, THE Lint_Tool SHALL return the message `lint_clean: no violations found` and THE Orchestrator SHALL exit with status code 0 without writing a diff.
5. IF the `ruff` binary is not found on `PATH`, THEN THE Lint_Tool SHALL return an error result containing the message `lint_tool_missing: ruff not found on PATH` and THE Orchestrator SHALL exit with status code 2.
6. IF the ruff subprocess exits with an unexpected non-zero status code other than a violations-found exit code, THEN THE Lint_Tool SHALL return the stderr output and exit code in the error result.

---

### Requirement 4: Diff and Output Persistence

**User Story:** As a practitioner, I want every proposed diff to be saved to a well-named file with companion metadata, so that I can audit, compare, and apply proposals across sessions.

#### Acceptance Criteria

1. THE Orchestrator SHALL create the `./agent_output` directory relative to the working directory if it does not already exist before writing any output file.
2. WHEN a task produces a proposed diff, THE Orchestrator SHALL write the diff to a file named `<task_type>_<target_basename>_<ISO8601_date>.diff` inside `agent_output`.
3. WHEN a diff file is written, THE Orchestrator SHALL write a companion JSON metadata file with the same base name and a `.json` extension containing: `task_type`, `target_path`, `timestamp_utc`, `llm_model`, `iteration_count`, and `token_usage`.
4. WHEN a file with the intended name already exists in `agent_output`, THE Orchestrator SHALL append a numeric suffix (`_1`, `_2`, …) to the base name until a non-colliding filename is found, then write both the diff and the metadata file using that suffixed name.
5. THE Orchestrator SHALL write the diff file using UTF-8 encoding and Unix line endings (`\n`) regardless of the host operating system.

---

### Requirement 5: Persistent Memory Store

**User Story:** As a practitioner, I want the agent to remember past task outcomes in a SQLite database, so that it avoids redundant work and I can query what it has already attempted.

#### Acceptance Criteria

1. THE Memory_Store SHALL use a SQLite database file at the path `./agent_memory.db` relative to the working directory, creating the file and schema on first use if it does not exist.
2. THE Memory_Store SHALL store one record per task execution containing: `task_id` (UUID), `task_type`, `target_path`, `status` (one of `pending`, `success`, `failed`), `created_at` (UTC timestamp), `updated_at` (UTC timestamp), and `output_path`.
3. WHEN a task is initiated, THE Memory_Store SHALL insert a record with `status = pending` before the first LLM call.
4. WHEN a task completes or fails, THE Memory_Store SHALL update the corresponding record's `status` and `updated_at` fields.
5. WHEN a task is requested for a `(task_type, target_path)` pair whose most recent record has `status = success` and `created_at` is within the last 24 hours, THE Orchestrator SHALL skip execution, log a `dedup_skip` event with the original `task_id`, and exit with status code 0.
6. THE Memory_Store SHALL expose a `query(task_type=None, status=None, since=None)` interface that returns matching records sorted by `created_at` descending.
7. IF the SQLite database file is corrupt or cannot be opened, THEN THE Memory_Store SHALL log a `db_recovery` event, rename the corrupt file to `agent_memory.db.corrupt.<timestamp>`, and create a fresh database before proceeding.

---

### Requirement 6: Self-Correction Loop

**User Story:** As a practitioner, I want the agent to automatically retry a failed task by feeding the failure back to the LLM, so that I can observe how the agent handles and learns from its own mistakes.

#### Acceptance Criteria

1. WHEN a tool call returns an error result, THE Orchestrator SHALL append the error result to the conversation context and continue the LLM loop without incrementing the retry counter.
2. WHEN the LLM produces a diff that, when applied, still causes ruff to report violations, THE Orchestrator SHALL treat this as a correction attempt, increment the retry counter, and re-enter the loop with the new ruff output appended to context.
3. WHEN a new ruff error code appears in the post-apply output that was not present in the pre-apply output, THE Orchestrator SHALL log a `new_error_introduced` event containing the new error codes before re-entering the correction loop.
4. WHEN the correction retry counter reaches 3, THE Orchestrator SHALL abort the loop, set the Memory_Store record `status` to `failed`, and write a partial diff (the last proposed change) to `agent_output` with the suffix `_attempt_3`.
5. WHEN writing diff files during a correction loop, THE Orchestrator SHALL include the attempt number in the filename suffix (`_attempt_1`, `_attempt_2`, `_attempt_3`) so that all attempts are preserved.
6. IF the correction loop aborts after 3 retries, THEN THE Orchestrator SHALL exit with status code 3.

---

### Requirement 7: Test Generation for Uncovered Code

**User Story:** As a practitioner, I want the agent to generate pytest tests for source lines not covered by the existing test suite, so that I can improve coverage without manually identifying gaps.

#### Acceptance Criteria

1. WHEN the `generate_tests` task is invoked, THE Test_Generator SHALL parse the `pytest-cov` JSON coverage report at the path specified by `--coverage-report` to identify source files with uncovered lines.
2. WHEN uncovered lines are identified, THE Test_Generator SHALL read the source lines and their surrounding context (up to 20 lines above and below each uncovered segment) using the File_Reader and pass the context to the LLM.
3. WHEN the LLM proposes a test, THE Test_Generator SHALL write the proposed test code as a unified diff targeting the appropriate `tests/` file and save it to `agent_output`.
4. WHEN a test file already contains a test function with the same name as a proposed test, THE Test_Generator SHALL skip writing that test, log a `duplicate_test_skipped` event, and continue with the next uncovered segment.
5. IF the `--coverage-report` path does not exist or is not valid JSON, THEN THE Test_Generator SHALL exit with status code 1 and log a `coverage_report_invalid` event.
6. IF no uncovered lines exist in any source file in the report, THEN THE Test_Generator SHALL log a `coverage_complete` event and exit with status code 0 without writing any diffs.

---

### Requirement 8: TODO-to-Fix Conversion

**User Story:** As a practitioner, I want the agent to scan Python source files for TODO comments and propose concrete implementations, so that deferred work items become actionable code changes.

#### Acceptance Criteria

1. WHEN the `convert_todos` task is invoked with a target directory, THE TODO_Scanner SHALL recursively scan all `.py` files in the directory for lines matching the pattern `#\s*TODO[\s:]` (case-insensitive).
2. WHEN a TODO comment is found, THE TODO_Scanner SHALL extract the comment text, the file path, and the line number, then pass them to the LLM with the surrounding function context (up to 30 lines above and below the TODO line).
3. WHEN the LLM proposes an implementation, THE TODO_Scanner SHALL write the proposal as a unified diff to `agent_output`.
4. WHEN a TODO comment text contains fewer than 5 non-whitespace words after the TODO marker, THE TODO_Scanner SHALL classify the TODO as ambiguous, skip it without invoking the LLM, and log a `todo_skipped_ambiguous` event with the file path and line number.
5. THE TODO_Scanner SHALL exclude any `.py` files whose resolved path contains `.venv`, `site-packages`, `__pycache__`, or `.git` path components from the scan.
6. IF no TODO comments are found in the target directory after exclusions, THEN THE TODO_Scanner SHALL log a `no_todos_found` event and exit with status code 0 without invoking the LLM.

---

### Requirement 9: GitHub API Integration

**User Story:** As a practitioner, I want the agent to read GitHub issues and open draft pull requests with proposed fixes, so that the full maintenance cycle from issue discovery to PR proposal is automated.

#### Acceptance Criteria

1. THE GitHub_Client SHALL authenticate all API requests using the bearer token read from the `GITHUB_TOKEN` environment variable.
2. WHEN the `GITHUB_TOKEN` environment variable is not set or is an empty string, THE GitHub_Client SHALL exit with status code 1 and log a `github_auth_missing` event before making any API calls.
3. WHEN the `fetch_issues` operation is invoked, THE GitHub_Client SHALL retrieve open issues from the target repository using the GitHub REST API v3 and return them as a list of structured records containing: `issue_number`, `title`, `body`, `labels`, and `comment_count`.
4. WHEN the agent has produced a proposed fix diff, THE GitHub_Client SHALL create a new branch named `agent/fix-<task_type>-<ISO8601_date>`, commit the diff as a file change on that branch, and open a draft pull request with the title `[Agent] <task_type>: <target_basename>` against the default branch.
5. WHEN a GitHub API response returns HTTP status 429 or a `Retry-After` header, THE GitHub_Client SHALL wait for the duration specified in the `Retry-After` header (or 60 seconds if the header is absent) and retry the request once before logging a `rate_limit_abort` event and exiting with status code 4.
6. THE GitHub_Client SHALL NOT use `git push --force` or any force-push equivalent when creating or updating branches.
7. IF a branch with the intended name already exists in the remote repository, THEN THE GitHub_Client SHALL append a numeric suffix (`-1`, `-2`, …) to the branch name until a non-colliding name is found.

---

### Requirement 10: Sandboxed Execution

**User Story:** As a practitioner, I want agent-proposed code to be executed inside an isolated Docker container, so that untrusted LLM-generated code cannot modify the host filesystem or network.

#### Acceptance Criteria

1. WHEN the `--sandbox` flag is passed to the CLI, THE Sandbox SHALL execute any code evaluation step (e.g., running tests after applying a diff) inside a Docker container built from the project's `Dockerfile.sandbox`.
2. WHEN running inside the sandbox, THE Sandbox SHALL mount the repository directory as a read-only volume at `/repo` and copy proposed diffs to a separate writable working directory `/workspace` before applying them.
3. WHEN a sandboxed execution exceeds 120 seconds, THE Sandbox SHALL kill the Docker container, log a `sandbox_timeout` event, and return a failure result to the Orchestrator.
4. WHEN the `docker` binary is not found on `PATH` and `--sandbox` is specified, THE Sandbox SHALL log a `sandbox_unavailable` event with the message `docker not found on PATH` and fall back to executing on the host without a sandbox, logging a `sandbox_fallback` warning.
5. IF a sandboxed Docker container exits with a non-zero status code, THEN THE Sandbox SHALL capture the container stdout and stderr, log a `sandbox_execution_failed` event, and return the captured output as the tool result.

---

### Requirement 11: Token and Cost Budgeting

**User Story:** As a practitioner, I want each task run to enforce a token limit and record cumulative API cost, so that I can control expenses and understand the cost profile of each task type.

#### Acceptance Criteria

1. THE Cost_Tracker SHALL maintain a running total of input tokens, output tokens, and estimated USD cost for the current task run, updating after every LLM API response.
2. WHEN cumulative token usage for a task run exceeds the configured `max_tokens_per_task` limit (default: 50,000 tokens), THE Orchestrator SHALL abort the loop, log a `budget_exceeded` event with the current token totals, set the Memory_Store record to `failed`, and exit with status code 5.
3. WHEN a task completes (success or failure), THE Cost_Tracker SHALL write the final `input_tokens`, `output_tokens`, `estimated_usd_cost`, and `model` fields to the companion metadata JSON file in `agent_output`.
4. THE Cost_Tracker SHALL log a `token_usage` structured event after every LLM call containing `call_number`, `input_tokens`, `output_tokens`, `cumulative_tokens`, and `cumulative_usd_cost`.
5. WHERE the `--max-tokens` CLI flag is provided, THE Cost_Tracker SHALL use the flag value as `max_tokens_per_task` instead of the default, and the flag value MUST be a positive integer between 1,000 and 500,000.
6. IF the Anthropic API response does not include token usage fields, THEN THE Cost_Tracker SHALL log a `token_usage_unavailable` warning and continue execution without enforcing the budget for that call.

---

### Requirement 12: Issue Triage

**User Story:** As a practitioner, I want the agent to read GitHub issues and post structured triage comments, so that open issues are categorized and prioritized without manual review.

#### Acceptance Criteria

1. WHEN the `triage_issues` task is invoked, THE Triage_Agent SHALL fetch all open issues from the target repository using the GitHub_Client and process them in ascending issue-number order.
2. WHEN an issue is processed, THE Triage_Agent SHALL read the issue body and all existing comments, pass them to the LLM, and receive a triage assessment containing: `priority` (one of `critical`, `high`, `medium`, `low`), `type` (one of `bug`, `feature`, `question`, `documentation`), and `reproduction_status` (one of `confirmed`, `unconfirmed`, `not_applicable`).
3. WHEN a triage assessment is produced, THE Triage_Agent SHALL post it as a comment to the GitHub issue in a structured markdown format and apply the `agent-triaged` label to the issue via the GitHub API.
4. WHEN an issue already has the `agent-triaged` label and the label was applied within the last 7 days, THE Triage_Agent SHALL skip the issue, log a `triage_skipped_recent` event with the issue number, and proceed to the next issue.
5. IF the GitHub API returns an error when posting a triage comment, THEN THE Triage_Agent SHALL log a `triage_comment_failed` event with the issue number and HTTP status code, skip applying the label for that issue, and continue to the next issue.
6. IF the issue body is empty or contains fewer than 10 non-whitespace characters, THEN THE Triage_Agent SHALL post a triage comment noting `insufficient_detail` and assign `priority = low` without invoking the LLM.

---

### Requirement 13: Eval Harness

**User Story:** As a practitioner, I want a benchmark suite that scores the agent on fixed tasks and compares results across runs, so that I can measure improvement and detect regressions as I iterate on the agent.

#### Acceptance Criteria

1. THE Eval_Harness SHALL load benchmark task definitions from a YAML file at the path specified by `--benchmark <file>`, where each task definition contains: `id`, `task_type`, `target_path`, `expected_diff_contains` (list of strings), and `pass_criteria` (one of `diff_contains_all`, `lint_clean_after`, `tests_pass_after`).
2. WHEN the eval is run, THE Eval_Harness SHALL execute each benchmark task in sequence using the Orchestrator and record: `task_id`, `status` (one of `pass`, `fail`, `error`), `duration_seconds`, and `token_usage`.
3. WHEN all tasks complete, THE Eval_Harness SHALL write a JSON results file to `./eval_results/<run_id>.json` containing the per-task records and a summary with `total`, `passed`, `failed`, `error_count`, and `pass_rate`.
4. WHEN the `--compare <run_id_a> <run_id_b>` flag is provided, THE Eval_Harness SHALL load the two result files, compute per-task status changes (e.g., `pass→fail`, `fail→pass`), and print a diff-style comparison report to stdout.
5. THE benchmark YAML file used for CI SHALL contain a minimum of 10 task definitions covering all four task types (`lint_fix`, `generate_tests`, `convert_todos`, `triage_issues`).
6. IF a benchmark task's `target_path` does not exist, THEN THE Eval_Harness SHALL record `status = error` with `reason = target_not_found` for that task and continue to the next task without halting the run.

---

### Requirement 14: Observability and Logging

**User Story:** As a practitioner, I want every significant agent action logged as structured JSON to stderr, so that I can trace exactly what the agent did, debug failures, and understand LLM reasoning without modifying source code.

#### Acceptance Criteria

1. THE Logger SHALL emit all log entries as newline-delimited JSON objects to stderr, where each entry contains at minimum: `timestamp_utc` (ISO 8601), `level` (one of `DEBUG`, `INFO`, `WARNING`, `ERROR`), `event`, and `payload`.
2. WHEN an LLM API call is made, THE Logger SHALL emit an `llm_call` event at `INFO` level containing `model`, `input_tokens_estimate`, and `call_number`.
3. WHEN an LLM API response is received, THE Logger SHALL emit an `llm_response` event at `INFO` level containing `call_number`, `stop_reason`, `input_tokens`, and `output_tokens`.
4. WHEN a tool is dispatched, THE Logger SHALL emit a `tool_dispatch` event at `INFO` level containing `tool_name` and `arguments`.
5. WHEN a tool returns a result, THE Logger SHALL emit a `tool_result` event at `INFO` level containing `tool_name` and `result_summary` (first 200 characters of the result string).
6. WHEN any state change occurs in the Orchestrator loop (e.g., loop iteration, abort, retry), THE Logger SHALL emit a corresponding event at `INFO` or `WARNING` level with the relevant state fields.
7. WHERE the `--verbose` CLI flag is provided, THE Logger SHALL additionally emit `DEBUG` level events including full LLM prompt content and complete tool result payloads.
8. IF a log entry cannot be serialized to JSON due to a non-serializable value, THEN THE Logger SHALL replace the non-serializable field with the string `<non-serializable: <type>>` and emit the entry rather than dropping it.

---

### Requirement 15: Configuration Management

**User Story:** As a practitioner, I want a YAML configuration file with CLI override support, so that I can adjust agent behavior across different repositories without modifying source code.

#### Acceptance Criteria

1. THE Config_Loader SHALL load configuration from a YAML file at `./agent_config.yaml` relative to the working directory if the file exists, applying documented default values for any missing fields.
2. WHEN both a config file and CLI flags are present, THE Config_Loader SHALL apply CLI flag values as overrides with higher precedence than the config file values for the same settings.
3. THE Config_Loader SHALL validate all configuration values on startup and report all validation errors as a structured list before exiting with status code 6 if any validation error is present.
4. THE Config_Loader SHALL support the following fields with their defaults: `max_tokens_per_task: 50000`, `max_iterations: 10`, `output_dir: ./agent_output`, `memory_db: ./agent_memory.db`, `sandbox_enabled: false`, `sandbox_timeout_seconds: 120`, `dedup_window_hours: 24`, `log_level: INFO`.
5. IF the `agent_config.yaml` file exists but contains invalid YAML syntax, THEN THE Config_Loader SHALL log a `config_parse_error` event with the parse error detail and exit with status code 6.
6. IF a numeric configuration field is provided with a non-numeric value, THEN THE Config_Loader SHALL include that field in the validation error list with the message `<field>: expected integer, got <type>`.

---

### Requirement 16: Orchestration Architecture

**User Story:** As a practitioner, I want the Phase 1 orchestrator implemented as a transparent raw Python loop with a documented comparison against LangGraph, so that I understand the trade-offs before adopting a higher-level framework.

#### Acceptance Criteria

1. THE Orchestrator SHALL implement the agent loop as a plain Python `while` loop with explicit state variables (conversation history list, iteration counter, retry counter) rather than using LangGraph, LangChain, or any other agent framework in Phase 1.
2. THE Orchestrator SHALL define a `OrchestratorProtocol` abstract base class (Python `Protocol`) specifying the `run(task_type, target_path) -> TaskResult` interface, enabling future drop-in replacement with a framework-based implementation.
3. THE Orchestrator SHALL define a `Tool` abstract base class (Python `Protocol`) specifying the `name: str`, `description: str`, `input_schema: dict`, and `execute(inputs: dict) -> ToolResult` interface so that all tools are interchangeable.
4. WHEN Phase 1 implementation is complete, THE Agent SHALL include a `docs/orchestration_comparison.md` file containing: a side-by-side comparison of the raw loop vs. LangGraph for at least 3 criteria (transparency, boilerplate, debuggability), the author's recommendation for Phase 2, and estimated lines-of-code difference.
5. IF a tool raises an unhandled Python exception during `execute`, THEN THE Orchestrator SHALL catch the exception, log a `tool_exception` event at `ERROR` level with the exception type and message, and return a `ToolResult` with `is_error=True` rather than propagating the exception.

---

### Requirement 17: Engineering Challenges Documentation

**User Story:** As a practitioner building a portfolio project, I want a comprehensive README that documents the hard engineering problems I solved, failure modes I encountered, and lessons I learned, so that technical reviewers can see my depth of understanding.

#### Acceptance Criteria

1. THE README SHALL include a "Hard Engineering Problems" section containing descriptions of at least 5 distinct technical challenges encountered during development, each with a problem statement, the approach taken, and the outcome.
2. THE README SHALL include a "Failure Modes" section documenting at least 3 observed or anticipated failure modes of the agent (e.g., LLM hallucination of file paths, infinite correction loops, prompt injection via TODO text), each with a description, observed impact, and the mitigation implemented.
3. THE README SHALL include a "What I Learned" section containing at least 3 distinct insights about agent engineering that the practitioner gained from this project, written in first-person reflective style.
4. WHEN the project reaches Phase 5, THE README SHALL include a "Quick Start" section with commands to install dependencies, configure `GITHUB_TOKEN` and `ANTHROPIC_API_KEY`, run the agent against a sample repository, and run the eval harness.
5. THE README SHALL include a "Phase Roadmap" section listing all five phases with a one-sentence description of each phase's primary goal and the current completion status.
