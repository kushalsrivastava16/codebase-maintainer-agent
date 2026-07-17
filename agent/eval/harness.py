"""
Eval harness for benchmarking the Codebase Maintainer Agent.

Loads benchmark task definitions from YAML, runs each through the Orchestrator,
scores results against pass criteria, and writes JSON reports.

WHY a separate eval harness instead of pytest fixtures?
  pytest fixtures are great for unit tests, but benchmarking requires:
  - Running the full orchestrator end-to-end (not mocked)
  - Tracking wall-clock time and token costs per task
  - Writing persistent JSON reports for cross-run comparisons
  - A YAML-driven task definition format that non-engineers can edit

  The harness handles all of this; pytest handles the unit tests that
  validate the harness itself.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from agent.protocols import TaskResult, TokenUsage


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkTask:
    """
    A single benchmark task definition loaded from a YAML file.

    WHY expected_diff_contains as a list of strings?
      A single diff may need to satisfy multiple independent criteria — for
      example, the fixed lint output must remove an unused import AND add a
      blank line. Checking each string independently is more readable than
      a single regex, and gives clearer failure messages.
    """

    id: str
    task_type: str
    target_path: str
    expected_diff_contains: list[str]
    pass_criteria: str  # "diff_contains_all" | "lint_clean_after" | "tests_pass_after"


@dataclass
class EvalResult:
    """
    The outcome of a single benchmark task run.

    WHY estimated_usd on the result rather than just on TokenUsage?
      EvalResult is serialised to JSON and read by humans comparing runs.
      Surfacing the cost at the top level avoids nesting into token_usage.
    """

    task_id: str
    status: str            # "pass" | "fail" | "error"
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_usd: float
    reason: str | None     # error message when status == "error"; None otherwise


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------


class EvalHarness:
    """
    Orchestrates the full benchmark lifecycle:
      load YAML → run tasks → score results → write JSON report

    The orchestrator_factory is called once per task so each task gets a
    fresh orchestrator with a clean CostTracker and message history.
    """

    def __init__(
        self,
        orchestrator_factory: Callable[[], object],  # factory → OrchestratorProtocol
        results_dir: str = "./eval_results",
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, benchmark_path: str) -> str:
        """
        Run all tasks in the benchmark YAML and write a JSON report.

        Returns the run_id (timestamp string) so callers can locate the
        report at ``{results_dir}/{run_id}.json``.
        """
        tasks = self._load_benchmark(benchmark_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        results: list[EvalResult] = []
        for task in tasks:
            result = self._run_task(task)
            results.append(result)

        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        error_count = sum(1 for r in results if r.status == "error")
        total = len(results)
        pass_rate = (passed / total) if total > 0 else 0.0

        summary = {
            "run_id": run_id,
            "total": total,
            "passed": passed,
            "failed": failed,
            "error_count": error_count,
            "pass_rate": round(pass_rate, 4),
            "tasks": [asdict(r) for r in results],
        }

        report_path = self._results_dir / f"{run_id}.json"
        report_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        return run_id

    def compare(self, run_id_a: str, run_id_b: str) -> None:
        """
        Compare two eval runs and print tasks whose status changed.

        Output format per changed task:
          {task_id}: {old_status} → {new_status}

        A summary line showing the pass_rate of each run is printed at the end.
        """
        path_a = self._results_dir / f"{run_id_a}.json"
        path_b = self._results_dir / f"{run_id_b}.json"

        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))

        # Build task_id → status dicts
        status_a: dict[str, str] = {
            t["task_id"]: t["status"] for t in data_a.get("tasks", [])
        }
        status_b: dict[str, str] = {
            t["task_id"]: t["status"] for t in data_b.get("tasks", [])
        }

        all_ids = sorted(set(status_a) | set(status_b))
        changes_found = False
        for task_id in all_ids:
            old = status_a.get(task_id, "missing")
            new = status_b.get(task_id, "missing")
            if old != new:
                print(f"{task_id}: {old} → {new}")
                changes_found = True

        if not changes_found:
            print("No status changes between runs.")

        print(
            f"\nSummary: {run_id_a} pass_rate={data_a.get('pass_rate', 'N/A')}  "
            f"{run_id_b} pass_rate={data_b.get('pass_rate', 'N/A')}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_task(self, task: BenchmarkTask) -> EvalResult:
        """
        Run a single benchmark task end-to-end.

        Error conditions that prevent any orchestrator call (e.g., missing
        target) are returned as status="error" so the run continues with
        the remaining tasks rather than aborting the whole benchmark.
        """
        # Guard: target must exist before we spend any tokens
        if not Path(task.target_path).exists():
            return EvalResult(
                task_id=task.id,
                status="error",
                duration_seconds=0.0,
                input_tokens=0,
                output_tokens=0,
                estimated_usd=0.0,
                reason="target_not_found",
            )

        orchestrator = self._orchestrator_factory()
        t0 = time.monotonic()

        try:
            result: TaskResult = orchestrator.run(task.task_type, task.target_path)
        except Exception as exc:
            duration = time.monotonic() - t0
            return EvalResult(
                task_id=task.id,
                status="error",
                duration_seconds=round(duration, 3),
                input_tokens=0,
                output_tokens=0,
                estimated_usd=0.0,
                reason=str(exc),
            )

        duration = time.monotonic() - t0
        passed = self._score(task, result)

        return EvalResult(
            task_id=task.id,
            status="pass" if passed else "fail",
            duration_seconds=round(duration, 3),
            input_tokens=result.token_usage.input_tokens,
            output_tokens=result.token_usage.output_tokens,
            estimated_usd=result.token_usage.estimated_usd,
            reason=None,
        )

    def _score(self, task: BenchmarkTask, result: TaskResult) -> bool:
        """
        Score a TaskResult against the task's pass_criteria.

        WHY three separate criteria instead of a single boolean?
          Different task types have different natural success signals:
          - lint_fix → the diff contains the expected removals/additions
          - convert_todos → run ruff on the patched file, expect zero errors
          - generate_tests → run pytest on the patched file, expect zero failures
          Having distinct criteria keeps each scoring path simple and testable.
        """
        if task.pass_criteria == "diff_contains_all":
            return self._score_diff_contains_all(task, result)
        elif task.pass_criteria == "lint_clean_after":
            return self._score_lint_clean_after(task, result)
        elif task.pass_criteria == "tests_pass_after":
            return self._score_tests_pass_after(task, result)
        return False

    def _score_diff_contains_all(
        self, task: BenchmarkTask, result: TaskResult
    ) -> bool:
        """Pass if the diff file contains every string in expected_diff_contains."""
        if result.output_path is None:
            return False
        diff_path = Path(result.output_path)
        if not diff_path.exists():
            return False
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
        return all(s in diff_text for s in task.expected_diff_contains)

    def _score_lint_clean_after(
        self, task: BenchmarkTask, result: TaskResult
    ) -> bool:
        """
        Pass if applying the diff to a temp copy of the target file yields a
        ruff-clean file.

        If the diff is empty/absent (meaning no changes were needed), treat the
        file as already clean and return True.
        """
        if result.output_path is None:
            return False

        diff_text = Path(result.output_path).read_text(
            encoding="utf-8", errors="replace"
        )

        # Empty diff means the agent determined the file is already clean
        if not diff_text.strip():
            return True

        target = Path(task.target_path)
        if not target.exists():
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / target.name
            shutil.copy2(target, tmp_file)

            diff_file = Path(tmpdir) / "changes.diff"
            diff_file.write_text(diff_text, encoding="utf-8")

            patch_result = subprocess.run(
                ["patch", str(tmp_file), str(diff_file)],
                capture_output=True,
                text=True,
            )
            if patch_result.returncode != 0:
                return False

            ruff_result = subprocess.run(
                ["ruff", "check", str(tmp_file)],
                capture_output=True,
                text=True,
            )
            return ruff_result.returncode == 0

    def _score_tests_pass_after(
        self, task: BenchmarkTask, result: TaskResult
    ) -> bool:
        """
        Pass if applying the diff and running pytest on the patched file
        exits cleanly (returncode 0).
        """
        if result.output_path is None:
            return False

        diff_text = Path(result.output_path).read_text(
            encoding="utf-8", errors="replace"
        )

        if not diff_text.strip():
            # No changes proposed — run pytest on the original to see if it's already passing
            pytest_result = subprocess.run(
                ["pytest", task.target_path, "--tb=no", "-q"],
                capture_output=True,
                text=True,
            )
            return pytest_result.returncode == 0

        target = Path(task.target_path)
        if not target.exists():
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / target.name
            shutil.copy2(target, tmp_file)

            diff_file = Path(tmpdir) / "changes.diff"
            diff_file.write_text(diff_text, encoding="utf-8")

            patch_result = subprocess.run(
                ["patch", str(tmp_file), str(diff_file)],
                capture_output=True,
                text=True,
            )
            if patch_result.returncode != 0:
                return False

            pytest_result = subprocess.run(
                ["pytest", str(tmp_file), "--tb=no", "-q"],
                capture_output=True,
                text=True,
            )
            return pytest_result.returncode == 0

    def _load_benchmark(self, path: str) -> list[BenchmarkTask]:
        """
        Parse a YAML benchmark file and return a list of BenchmarkTask objects.

        Expected YAML structure::

            tasks:
              - id: lint_fix_001
                task_type: lint_fix
                target_path: ./src/example.py
                expected_diff_contains:
                  - "-import os"
                pass_criteria: diff_contains_all

        WHY yaml.safe_load?
          safe_load prevents arbitrary Python object construction from YAML
          tags, which is a known YAML deserialization vulnerability.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tasks_raw = raw.get("tasks", []) if isinstance(raw, dict) else raw
        return [BenchmarkTask(**t) for t in tasks_raw]
