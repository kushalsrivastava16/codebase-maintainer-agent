"""
CLI entry point for the Codebase Maintainer Agent.

WHY click instead of argparse?
  click provides decorator-based argument declaration, automatic help text
  generation, and better error messages. It's the de-facto standard for Python
  CLI tools and integrates cleanly with the pyproject.toml [project.scripts] entry.

WHY is the TASK_TOOL_MAP defined here rather than in orchestrator.py?
  Tool construction (FileReader, LintRunner, etc.) requires the repo_root and
  other runtime context from the CLI arguments. Keeping the map here avoids
  passing cli-specific state into the orchestrator, which should only know about
  the abstract Tool protocol.

HOW the wiring works:
  CLI args → AgentConfig → tools dict → Orchestrator → TaskResult → exit code
  Each component is constructed in order, with dependencies injected explicitly.
  This is "poor man's dependency injection" — no framework, just constructor args.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from agent.config import load_config
from agent.cost_tracker import CostTracker
from agent.diff_writer import DiffWriter
from agent.logger import StructuredLogger
from agent.memory import MemoryStore
from agent.orchestrator import Orchestrator
from agent.tools.coverage_reader import CoverageReader
from agent.tools.file_reader import FileReader
from agent.tools.lint_runner import LintRunner
from agent.tools.todo_scanner import TodoScanner

SUPPORTED_TASK_TYPES = ["lint_fix", "generate_tests", "convert_todos", "triage_issues"]


def _build_tools(
    task_type: str,
    target_path: Path,
    coverage_report: str | None,
    github_repo: str | None = None,
) -> dict:
    """
    Construct the tool instances for the given task type.

    WHY use CWD as the repo root for FileReader?
      The LLM receives the full relative path to the target (e.g.
      'tests/fixtures/foo.py') and calls read_file with that same relative
      path. FileReader resolves it as (repo_root / path), so repo_root must
      be the project root (CWD) — not the file's parent directory — otherwise
      the join produces a doubled path that doesn't exist.
    """
    repo_root = Path.cwd()

    if task_type == "lint_fix":
        file_reader = FileReader(repo_root=repo_root)
        lint_runner = LintRunner()
        return {file_reader.name: file_reader, lint_runner.name: lint_runner}

    elif task_type == "generate_tests":
        file_reader = FileReader(repo_root=repo_root)
        coverage_reader = CoverageReader()
        return {file_reader.name: file_reader, coverage_reader.name: coverage_reader}

    elif task_type == "convert_todos":
        file_reader = FileReader(repo_root=repo_root)
        todo_scanner = TodoScanner()
        return {file_reader.name: file_reader, todo_scanner.name: todo_scanner}

    elif task_type == "triage_issues":
        if github_repo:
            from agent.tools.github_client import GitHubClient
            github_client = GitHubClient(repo=github_repo)
            return {github_client.name: github_client}
        return {}

    return {}


@click.group()
def cli() -> None:
    """Codebase Maintainer Agent — autonomous code maintenance."""


@cli.command("run")
@click.option(
    "--task",
    required=True,
    type=click.Choice(SUPPORTED_TASK_TYPES),
    help="Type of maintenance task to run.",
)
@click.option(
    "--target",
    required=True,
    help="Path to the file or directory to operate on.",
)
@click.option(
    "--coverage-report",
    default=None,
    help="Path to pytest-cov JSON report (required for generate_tests).",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="Override max_tokens_per_task (1000-500000).",
)
@click.option(
    "--sandbox",
    is_flag=True,
    default=False,
    help="Enable Docker sandbox for code execution.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose DEBUG logging.",
)
@click.option(
    "--config",
    default="./agent_config.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Override output directory for diffs.",
)
@click.option(
    "--github-repo",
    default=None,
    help="GitHub repository in 'owner/repo' format (required for triage_issues).",
)
def run_cmd(
    task: str,
    target: str,
    coverage_report: str | None,
    max_tokens: int | None,
    sandbox: bool,
    verbose: bool,
    config: str,
    output_dir: str | None,
    github_repo: str | None,
) -> None:
    """Run a single maintenance task against a repository file or directory."""
    # --- Validate target path ---
    target_path = Path(target)
    if not target_path.exists():
        click.echo(f"error: target path does not exist: {target}", err=True)
        sys.exit(1)

    # --- Validate generate_tests requires coverage report ---
    if task == "generate_tests" and coverage_report is None:
        click.echo(
            "error: --coverage-report is required for generate_tests task", err=True
        )
        sys.exit(1)

    # --- Validate triage_issues requires github-repo ---
    if task == "triage_issues" and not github_repo:
        click.echo(
            "error: --github-repo is required for triage_issues task", err=True
        )
        sys.exit(1)

    # --- Validate max-tokens range ---
    if max_tokens is not None and not (1_000 <= max_tokens <= 500_000):
        click.echo("error: --max-tokens must be between 1000 and 500000", err=True)
        sys.exit(1)

    # --- Build CLI overrides dict ---
    cli_overrides: dict = {}
    if max_tokens is not None:
        cli_overrides["max_tokens_per_task"] = max_tokens
    if sandbox:
        cli_overrides["sandbox_enabled"] = True
    if output_dir is not None:
        cli_overrides["output_dir"] = output_dir
    if github_repo is not None:
        cli_overrides["github_repo"] = github_repo

    # --- Load configuration ---
    agent_config = load_config(config_path=config, cli_overrides=cli_overrides)

    # --- Build components ---
    logger = StructuredLogger(min_level=agent_config.log_level, verbose=verbose)
    cost_tracker = CostTracker(max_tokens=agent_config.max_tokens_per_task)
    diff_writer = DiffWriter(output_dir=agent_config.output_dir)

    # --- Memory store (task deduplication and history) ---
    memory = MemoryStore(db_path=agent_config.memory_db)
    if memory.recovery_occurred:
        logger.log("db_recovery", "WARNING",
                   db_path=agent_config.memory_db,
                   message="Database was corrupt and has been replaced")

    # --- Deduplication check (Task 17) ---
    canonical_target = str(target_path.resolve())
    existing = memory.check_dedup(
        task_type=task,
        target_path=canonical_target,
        window_hours=agent_config.dedup_window_hours,
    )
    if existing:
        logger.log("dedup_skip", "INFO",
                   task_id=existing["task_id"],
                   task_type=task,
                   target_path=canonical_target,
                   message="Recent successful run found — skipping")
        sys.exit(0)

    # --- Read original file content for diff generation ---
    original_content = ""
    if target_path.is_file():
        try:
            original_content = target_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.log("target_read_error", "ERROR",
                       target=target, error=str(exc))
            sys.exit(1)

    # --- Build tools for task type ---
    repo = github_repo or agent_config.github_repo
    try:
        tools = _build_tools(task, target_path, coverage_report, github_repo=repo)
    except RuntimeError as exc:
        if "github_auth_missing" in str(exc):
            logger.log("github_auth_missing", "ERROR",
                       message="GITHUB_TOKEN environment variable is not set or empty")
            sys.exit(1)
        raise

    # --- Optional sandbox (Task 26: sandbox integration into correction loop) ---
    sandbox = None
    if agent_config.sandbox_enabled:
        from agent.sandbox import Sandbox
        sandbox = Sandbox(
            repo_path=str(target_path.resolve() if target_path.is_file() else target_path),
            timeout=agent_config.sandbox_timeout_seconds,
            logger=logger,
        )

    # --- Create and run orchestrator ---
    orchestrator = Orchestrator(
        config=agent_config,
        logger=logger,
        cost_tracker=cost_tracker,
        diff_writer=diff_writer,
        tools=tools,
        original_content=original_content,
        memory=memory,
        sandbox=sandbox,
    )

    logger.log("run_start", "INFO",
               task_type=task, target=target, model=agent_config.model)

    result = orchestrator.run(task, canonical_target)

    logger.log("run_complete", "INFO",
               status=result.status,
               output_path=result.output_path,
               total_tokens=result.token_usage.total_tokens,
               estimated_usd=result.token_usage.estimated_usd)

    sys.exit(0 if result.status == "success" else 3)


@cli.command("eval")
@click.option("--benchmark", required=True, help="Path to benchmark YAML file.")
@click.option(
    "--compare",
    nargs=2,
    default=None,
    metavar="RUN_ID_A RUN_ID_B",
    help="Compare two eval runs by their run IDs.",
)
@click.option(
    "--config",
    default="./agent_config.yaml",
    help="Path to YAML config file.",
)
def eval_cmd(benchmark: str, compare: tuple | None, config: str) -> None:
    """Run the eval harness against a benchmark suite, or compare two runs."""
    from agent.eval.harness import EvalHarness

    if compare:
        harness = EvalHarness(orchestrator_factory=lambda: None)
        harness.compare(compare[0], compare[1])
        return

    if not Path(benchmark).exists():
        click.echo(f"error: benchmark file not found: {benchmark}", err=True)
        sys.exit(1)

    agent_config = load_config(config_path=config)

    def make_orchestrator():
        logger = StructuredLogger(min_level=agent_config.log_level)
        cost_tracker = CostTracker(max_tokens=agent_config.max_tokens_per_task)
        diff_writer = DiffWriter(output_dir=agent_config.output_dir)
        return Orchestrator(
            config=agent_config,
            logger=logger,
            cost_tracker=cost_tracker,
            diff_writer=diff_writer,
            tools={},
        )

    harness = EvalHarness(orchestrator_factory=make_orchestrator)
    run_id = harness.run(benchmark)
    click.echo(f"Eval complete. Run ID: {run_id}")
    click.echo(f"Results: eval_results/{run_id}.json")


if __name__ == "__main__":
    cli()
