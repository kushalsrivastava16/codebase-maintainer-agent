# Implementation Tasks

## Phase 1: Core Agent Loop

- [x] 1. Project scaffolding and pyproject.toml
  - Create project directory structure with all __init__.py files
  - Write pyproject.toml with pinned dependencies (anthropic==0.34.0, httpx==0.27.0, click==8.1.7, PyYAML==6.0.1, docker==7.1.0)
  - Create agent_config.yaml example file
  - _Requirements: 15, 16_

- [x] 2. Core data models (agent/protocols.py)
  - Implement ToolResult, TokenUsage, TaskResult dataclasses
  - Implement Tool and OrchestratorProtocol as typing.Protocol classes
  - Write tests/test_protocols.py
  - _Requirements: 16_

- [x] 3. Structured logger (agent/logger.py)
  - Implement StructuredLogger with newline-delimited JSON output to stderr
  - Support DEBUG/INFO/WARNING/ERROR levels and verbose mode
  - Implement _sanitize() for non-serializable values
  - Write tests/test_logger.py
  - _Requirements: 14_

- [x] 4. Configuration loader (agent/config.py)
  - Implement load_config() with YAML loading, default merging, CLI override support
  - Implement _validate() for numeric field type checks
  - Exit with code 6 on validation error or YAML parse error
  - Write tests/test_config.py
  - _Requirements: 15_

- [x] 5. File Reader tool (agent/tools/file_reader.py)
  - Implement path traversal guard using Path.resolve() and relative_to()
  - Implement 100KB truncation with truncation notice
  - Implement injection pattern stripping ([INST], </s>, <s>, ###, Human:)
  - Write tests/tools/test_file_reader.py covering all error paths
  - _Requirements: 2_

- [x] 6. Lint Runner tool (agent/tools/lint_runner.py)
  - Implement subprocess calls to ruff check and ruff format --check
  - Handle missing ruff binary via shutil.which
  - Return lint_clean message on zero violations
  - Write tests/tools/test_lint_runner.py
  - _Requirements: 3_

- [x] 7. Diff Writer (agent/diff_writer.py)
  - Implement unified diff generation using difflib.unified_diff
  - Implement filename pattern: {task_type}_{basename}_{ISO8601}.diff
  - Implement collision avoidance with _1, _2 numeric suffixes
  - Write companion JSON metadata file
  - Enforce UTF-8 encoding and Unix line endings
  - Write tests/test_diff_writer.py
  - _Requirements: 4_

- [x] 8. Cost Tracker (agent/cost_tracker.py)
  - Implement running token totals updated from Anthropic usage response
  - Implement budget_exceeded() check
  - Implement totals() returning TokenUsage with USD estimate
  - Write tests/test_cost_tracker.py
  - _Requirements: 11_

- [x] 9. Core Orchestrator (agent/orchestrator.py)
  - Implement raw while loop with messages list, iteration counter, retry_count
  - Implement _call_llm() using anthropic SDK messages.create()
  - Implement _dispatch_tool() with unknown-tool error handling and exception wrapping
  - Implement _abort() with exit code mapping
  - Implement _extract_diff() from end_turn response
  - Implement max_iterations abort guard (exit code 3)
  - Implement budget_exceeded abort guard (exit code 5)
  - Write tests/test_orchestrator.py with mocked LLM client
  - _Requirements: 1, 6, 11, 14, 16_

- [x] 10. CLI entry point (agent/__main__.py)
  - Implement click CLI with run command: --task, --target, --max-tokens, --sandbox, --verbose, --config
  - Validate --task against SUPPORTED_TASK_TYPES, exit code 1 if invalid
  - Validate --target exists, exit code 1 if missing
  - Wire all components: config, logger, diff_writer, cost_tracker, tools, orchestrator
  - Implement TASK_TOOL_MAP for tool registration per task type
  - _Requirements: 1, 15_

- [x] 11. End-to-end lint_fix integration test
  - Create tests/fixtures/utils_with_lint_errors.py with known ruff violations
  - Write tests/integration/test_lint_fix_e2e.py that runs full orchestrator against fixture
  - Verify agent_output/ contains a .diff file
  - Verify ruff is clean after applying the diff
  - _Requirements: 1, 2, 3, 4_

- [x] 12. TODO Scanner tool (agent/tools/todo_scanner.py)
  - Implement recursive .py file scanner with TODO_PATTERN regex
  - Implement exclusion of .venv, site-packages, __pycache__, .git path components
  - Implement ambiguous TODO detection (fewer than 5 non-whitespace words)
  - Write tests/tools/test_todo_scanner.py
  - _Requirements: 8_

- [x] 13. Coverage Reader tool (agent/tools/coverage_reader.py)
  - Implement pytest-cov JSON report parser extracting missing_lines per file
  - Return coverage_complete when all lines are covered
  - Return error on missing or invalid JSON
  - Write tests/tools/test_coverage_reader.py
  - _Requirements: 7_

- [x] 14. Orchestration comparison doc (docs/orchestration_comparison.md)
  - Write side-by-side comparison: raw loop vs LangGraph
  - Cover: transparency, boilerplate, debuggability criteria
  - Include author recommendation for Phase 2 and estimated LoC difference
  - _Requirements: 16_

## Phase 2: Memory, Self-Correction, and Additional Tasks

- [x] 15. Memory Store (agent/memory.py)
  - Implement SQLite schema with CREATE TABLE IF NOT EXISTS
  - Implement insert_pending(), update_status(), check_dedup(), query() methods
  - Implement corruption recovery: rename to .corrupt.<timestamp>, reinitialize
  - Enable WAL mode for durability
  - Write tests/test_memory.py covering dedup window, corruption recovery, query filters
  - _Requirements: 5_

- [x] 16. Self-correction loop integration into Orchestrator
  - Add retry_count tracking and post-apply ruff re-check
  - Implement new_error_introduced detection and logging
  - Write _attempt_N suffix logic for diff filenames
  - Abort at retry_count == 3 with exit code 3
  - _Requirements: 6_

- [x] 17. Memory deduplication integration into CLI
  - Add check_dedup() call in __main__.py before orchestrator.run()
  - Log dedup_skip event and exit 0 on match
  - Log db_recovery when MemoryStore recovers from corruption
  - _Requirements: 5_

- [x] 18. Test generation task support
  - Wire generate_tests task type to FileReader + CoverageReader tools in TASK_TOOL_MAP
  - Add --coverage-report CLI flag
  - Validate coverage report path exists when task == generate_tests
  - Write tests/integration/test_generate_tests_e2e.py with fixture coverage report
  - _Requirements: 7_

- [x] 19. TODO conversion task support
  - Wire convert_todos task type to FileReader + TodoScanner tools in TASK_TOOL_MAP
  - Write tests/integration/test_convert_todos_e2e.py with fixture .py files containing TODOs
  - _Requirements: 8_

## Phase 3: GitHub Integration and Issue Triage

- [x] 20. GitHub Client tool (agent/tools/github_client.py)
  - Implement httpx-based REST client with GITHUB_TOKEN auth
  - Implement fetch_issues(), create_branch(), commit_diff(), open_pr() methods
  - Implement _request() with Retry-After retry on 429
  - Implement _unique_branch_name() with numeric suffix collision avoidance
  - Never use force-push in branch update operations
  - Write tests/tools/test_github_client.py with httpx mock (37 tests)
  - _Requirements: 9_

- [x] 21. Issue Triage agent (agent/triage.py)
  - Implement TriageAgent with GitHub issue fetch and LLM-based assessment
  - Post structured markdown triage comment via GitHub API
  - Apply agent-triaged label after successful comment post
  - Skip recently-triaged issues (within 7 days)
  - Handle insufficient_detail issues without LLM call
  - Write tests/test_triage.py (24 tests)
  - _Requirements: 12_

- [x] 22. triage_issues task type wiring
  - Add triage_issues to SUPPORTED_TASK_TYPES and TASK_TOOL_MAP
  - Require --github-repo flag for triage_issues task
  - Handle github_auth_missing with exit code 1
  - _Requirements: 9, 12_

- [x] 23. Token budgeting full integration
  - CostTracker integrated into Orchestrator loop after every LLM call
  - Log token_usage event after every LLM call
  - Write budget_exceeded outcome to Memory_Store (via memory.update_status)
  - Token totals written to companion metadata JSON via DiffWriter
  - _Requirements: 11_

## Phase 4: Docker Sandbox

- [x] 24. Sandbox implementation (agent/sandbox.py)
  - Implement Docker container execution with read-only /repo and writable /workspace volumes
  - Implement timeout kill with sandbox_timeout event logging
  - Implement fallback to host execution when docker not found on PATH
  - Capture stdout + stderr as single UTF-8 string
  - Write tests/test_sandbox.py with docker mock (12 tests)
  - _Requirements: 10_

- [x] 25. Dockerfile.sandbox
  - Write Dockerfile.sandbox with python:3.11-slim base
  - Install all project dependencies at build time
  - Add non-root sandbox user
  - Set default CMD to pytest /workspace
  - _Requirements: 10_

- [x] 26. Sandbox integration into self-correction loop
  - Use Sandbox.run() for ruff re-check step when sandbox_enabled=True
  - Orchestrator._check_proposed_content() routes through sandbox when configured
  - CLI creates Sandbox instance when --sandbox flag is passed
  - _Requirements: 10_

## Phase 5: Eval Harness and Documentation

- [x] 27. Eval Harness (agent/eval/harness.py)
  - Implement YAML benchmark loader with BenchmarkTask dataclass
  - Implement _run_task() using Orchestrator
  - Implement _score() for diff_contains_all, lint_clean_after, tests_pass_after criteria
  - Write JSON results to eval_results/<run_id>.json
  - Implement compare() for side-by-side run diff
  - Write tests/test_eval_harness.py (23 tests)
  - _Requirements: 13_

- [x] 28. Eval CLI command (agent/__main__.py eval subcommand)
  - Add eval subcommand with --benchmark and --compare flags
  - Wire EvalHarness with orchestrator factory
  - Handle missing benchmark file with exit code 1
  - _Requirements: 13_

- [x] 29. Benchmark suite (benchmarks/benchmark_suite.yaml)
  - 12 benchmark task definitions covering all 4 task types
  - Created fixture files: models_with_errors.py, calculator.py, todo_sample.py
  - Includes 5 lint_fix, 3 generate_tests, 2 convert_todos, 2 triage_issues tasks
  - _Requirements: 13_

- [x] 30. README with engineering documentation
  - Hard Engineering Problems section (6 challenges with code references)
  - Failure Modes section (3 failure modes with mitigations)
  - What I Learned section (3 first-person insights)
  - Quick Start section with install and run commands
  - Phase Roadmap section with completion status
  - _Requirements: 17_
