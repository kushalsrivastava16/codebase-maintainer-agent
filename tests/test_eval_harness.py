"""
Tests for agent/eval/harness.py.

Tests cover:
  1. _load_benchmark() parses a YAML benchmark file into BenchmarkTask objects
  2. run() writes a results JSON file with the correct structure (mock orchestrator)
  3. _run_task() returns status="error" when target_path does not exist
  4. compare() prints status changes between two run JSON files (mock file reads)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.eval.harness import BenchmarkTask, EvalHarness, EvalResult
from agent.protocols import TaskResult, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_BENCHMARK_YAML = textwrap.dedent("""\
    tasks:
      - id: lint_fix_001
        task_type: lint_fix
        target_path: ./src/example.py
        expected_diff_contains:
          - "-import os"
          - "+  # removed unused import"
        pass_criteria: diff_contains_all
      - id: todo_convert_002
        task_type: convert_todos
        target_path: ./src/utils.py
        expected_diff_contains:
          - "# TODO"
        pass_criteria: lint_clean_after
""")


def _make_token_usage(
    input_tokens: int = 100,
    output_tokens: int = 200,
    estimated_usd: float = 0.0012,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_usd=estimated_usd,
    )


def _make_task_result(
    task_id: str = "task-abc",
    status: str = "success",
    output_path: str | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,
        output_path=output_path,
        token_usage=_make_token_usage(),
        error=None,
    )


def _make_harness(tmp_path: Path, orchestrator=None) -> EvalHarness:
    """Return an EvalHarness whose results land in tmp_path."""
    if orchestrator is None:
        orchestrator = MagicMock()
        orchestrator.run.return_value = _make_task_result()

    return EvalHarness(
        orchestrator_factory=lambda: orchestrator,
        results_dir=str(tmp_path / "eval_results"),
    )


# ---------------------------------------------------------------------------
# Test 1: _load_benchmark() parses YAML correctly
# ---------------------------------------------------------------------------

class TestLoadBenchmark:

    def test_parses_task_ids(self, tmp_path):
        """_load_benchmark returns one BenchmarkTask per YAML entry with correct id."""
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(SAMPLE_BENCHMARK_YAML, encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        assert len(tasks) == 2
        assert tasks[0].id == "lint_fix_001"
        assert tasks[1].id == "todo_convert_002"

    def test_parses_task_types(self, tmp_path):
        """task_type is correctly loaded for each task."""
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(SAMPLE_BENCHMARK_YAML, encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        assert tasks[0].task_type == "lint_fix"
        assert tasks[1].task_type == "convert_todos"

    def test_parses_expected_diff_contains_as_list(self, tmp_path):
        """expected_diff_contains is loaded as a list of strings."""
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(SAMPLE_BENCHMARK_YAML, encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        assert isinstance(tasks[0].expected_diff_contains, list)
        assert "-import os" in tasks[0].expected_diff_contains
        assert "+  # removed unused import" in tasks[0].expected_diff_contains

    def test_parses_pass_criteria(self, tmp_path):
        """pass_criteria is loaded as a string."""
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(SAMPLE_BENCHMARK_YAML, encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        assert tasks[0].pass_criteria == "diff_contains_all"
        assert tasks[1].pass_criteria == "lint_clean_after"

    def test_returns_benchmark_task_instances(self, tmp_path):
        """Each loaded entry is a BenchmarkTask dataclass instance."""
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(SAMPLE_BENCHMARK_YAML, encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        for task in tasks:
            assert isinstance(task, BenchmarkTask)

    def test_empty_tasks_returns_empty_list(self, tmp_path):
        """A benchmark YAML with no tasks yields an empty list."""
        benchmark_file = tmp_path / "empty.yaml"
        benchmark_file.write_text("tasks: []\n", encoding="utf-8")

        harness = _make_harness(tmp_path)
        tasks = harness._load_benchmark(str(benchmark_file))

        assert tasks == []


# ---------------------------------------------------------------------------
# Test 2: run() writes a results JSON with correct structure
# ---------------------------------------------------------------------------

class TestRun:

    def _write_benchmark(self, tmp_path: Path, target: Path) -> Path:
        """Write a minimal single-task benchmark YAML pointing at an existing target.

        Uses forward-slash paths so the YAML scalar is safe on all platforms —
        Python's Path() accepts them on Windows just fine.
        """
        # Use POSIX-style forward slashes to avoid YAML escape issues on Windows
        target_posix = target.as_posix()
        benchmark_file = tmp_path / "bench.yaml"
        benchmark_file.write_text(
            textwrap.dedent(f"""\
                tasks:
                  - id: task_001
                    task_type: lint_fix
                    target_path: "{target_posix}"
                    expected_diff_contains: []
                    pass_criteria: diff_contains_all
            """),
            encoding="utf-8",
        )
        return benchmark_file

    def test_run_creates_json_file(self, tmp_path):
        """run() writes a .json file in the results directory."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result(output_path=None)

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)

        run_id = harness.run(str(benchmark_file))

        results_path = tmp_path / "eval_results" / f"{run_id}.json"
        assert results_path.exists(), f"Expected {results_path} to exist"

    def test_run_returns_run_id_string(self, tmp_path):
        """run() returns a non-empty run_id string."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result()

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)

        run_id = harness.run(str(benchmark_file))

        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_run_json_has_required_top_level_keys(self, tmp_path):
        """The JSON report contains run_id, total, passed, failed, error_count, pass_rate, tasks."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result(output_path=None)

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)
        run_id = harness.run(str(benchmark_file))

        data = json.loads(
            (tmp_path / "eval_results" / f"{run_id}.json").read_text(encoding="utf-8")
        )

        for key in ("run_id", "total", "passed", "failed", "error_count", "pass_rate", "tasks"):
            assert key in data, f"Missing key: {key}"

    def test_run_json_total_matches_task_count(self, tmp_path):
        """total in the JSON equals the number of tasks in the benchmark."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result(output_path=None)

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)
        run_id = harness.run(str(benchmark_file))

        data = json.loads(
            (tmp_path / "eval_results" / f"{run_id}.json").read_text(encoding="utf-8")
        )

        assert data["total"] == 1

    def test_run_json_tasks_list_has_expected_fields(self, tmp_path):
        """Each entry in the tasks list has task_id, status, duration_seconds, etc."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result(output_path=None)

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)
        run_id = harness.run(str(benchmark_file))

        data = json.loads(
            (tmp_path / "eval_results" / f"{run_id}.json").read_text(encoding="utf-8")
        )

        assert len(data["tasks"]) == 1
        task_entry = data["tasks"][0]
        for field in ("task_id", "status", "duration_seconds", "input_tokens",
                      "output_tokens", "estimated_usd", "reason"):
            assert field in task_entry, f"Missing field in task entry: {field}"

    def test_run_pass_rate_is_zero_when_all_fail(self, tmp_path):
        """pass_rate is 0.0 when no tasks pass (diff contains all fails on None output_path)."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_orch = MagicMock()
        # output_path=None means diff_contains_all will return False
        mock_orch.run.return_value = _make_task_result(output_path=None)

        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )
        benchmark_file = self._write_benchmark(tmp_path, target)
        run_id = harness.run(str(benchmark_file))

        data = json.loads(
            (tmp_path / "eval_results" / f"{run_id}.json").read_text(encoding="utf-8")
        )

        assert data["pass_rate"] == 0.0
        assert data["passed"] == 0
        assert data["failed"] == 1


# ---------------------------------------------------------------------------
# Test 3: _run_task() returns error when target_path does not exist
# ---------------------------------------------------------------------------

class TestRunTask:

    def test_missing_target_returns_error_status(self, tmp_path):
        """_run_task returns status='error' when the target file does not exist."""
        task = BenchmarkTask(
            id="missing_target_001",
            task_type="lint_fix",
            target_path="/nonexistent/path/that/does/not/exist.py",
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        harness = _make_harness(tmp_path)
        result = harness._run_task(task)

        assert result.status == "error"

    def test_missing_target_reason_is_target_not_found(self, tmp_path):
        """_run_task sets reason='target_not_found' for a missing target."""
        task = BenchmarkTask(
            id="missing_target_002",
            task_type="lint_fix",
            target_path="/nonexistent/path/that/does/not/exist.py",
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        harness = _make_harness(tmp_path)
        result = harness._run_task(task)

        assert result.reason == "target_not_found"

    def test_missing_target_has_zero_tokens(self, tmp_path):
        """_run_task returns zero token counts when target is missing."""
        task = BenchmarkTask(
            id="missing_target_003",
            task_type="lint_fix",
            target_path="/nonexistent/does/not/exist.py",
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        harness = _make_harness(tmp_path)
        result = harness._run_task(task)

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.estimated_usd == 0.0

    def test_missing_target_does_not_call_orchestrator(self, tmp_path):
        """The orchestrator factory is NOT called when the target is missing."""
        task = BenchmarkTask(
            id="missing_target_004",
            task_type="lint_fix",
            target_path="/nonexistent/path.py",
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        factory_mock = MagicMock(return_value=MagicMock())
        harness = EvalHarness(
            orchestrator_factory=factory_mock,
            results_dir=str(tmp_path / "eval_results"),
        )
        harness._run_task(task)

        factory_mock.assert_not_called()

    def test_orchestrator_exception_returns_error(self, tmp_path):
        """_run_task returns status='error' when the orchestrator raises an exception."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        task = BenchmarkTask(
            id="orch_exception_001",
            task_type="lint_fix",
            target_path=str(target),
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        mock_orch = MagicMock()
        mock_orch.run.side_effect = RuntimeError("orchestrator exploded")
        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )

        result = harness._run_task(task)

        assert result.status == "error"
        assert "orchestrator exploded" in result.reason

    def test_task_id_in_result_matches_benchmark_task_id(self, tmp_path):
        """The EvalResult task_id matches the BenchmarkTask id field."""
        target = tmp_path / "src.py"
        target.write_text("x = 1\n", encoding="utf-8")

        task = BenchmarkTask(
            id="my_unique_task_id",
            task_type="lint_fix",
            target_path=str(target),
            expected_diff_contains=[],
            pass_criteria="diff_contains_all",
        )

        mock_orch = MagicMock()
        mock_orch.run.return_value = _make_task_result(output_path=None)
        harness = EvalHarness(
            orchestrator_factory=lambda: mock_orch,
            results_dir=str(tmp_path / "eval_results"),
        )

        result = harness._run_task(task)

        assert result.task_id == "my_unique_task_id"


# ---------------------------------------------------------------------------
# Test 4: compare() prints status changes between two runs
# ---------------------------------------------------------------------------

class TestCompare:

    def _write_run(
        self,
        results_dir: Path,
        run_id: str,
        tasks: list[dict],
        pass_rate: float = 0.5,
    ) -> None:
        """Write a fake run JSON to results_dir."""
        results_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": run_id,
            "total": len(tasks),
            "passed": sum(1 for t in tasks if t["status"] == "pass"),
            "failed": sum(1 for t in tasks if t["status"] == "fail"),
            "error_count": sum(1 for t in tasks if t["status"] == "error"),
            "pass_rate": pass_rate,
            "tasks": tasks,
        }
        (results_dir / f"{run_id}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def test_compare_prints_changed_task(self, tmp_path, capsys):
        """compare() prints a line for each task whose status differs between runs."""
        results_dir = tmp_path / "eval_results"

        self._write_run(results_dir, "run_A", [
            {"task_id": "task_001", "status": "fail"},
        ], pass_rate=0.0)
        self._write_run(results_dir, "run_B", [
            {"task_id": "task_001", "status": "pass"},
        ], pass_rate=1.0)

        harness = EvalHarness(
            orchestrator_factory=lambda: None,
            results_dir=str(results_dir),
        )
        harness.compare("run_A", "run_B")

        captured = capsys.readouterr()
        assert "task_001" in captured.out
        assert "fail" in captured.out
        assert "pass" in captured.out

    def test_compare_shows_arrow_between_statuses(self, tmp_path, capsys):
        """compare() uses the → separator between old and new status."""
        results_dir = tmp_path / "eval_results"

        self._write_run(results_dir, "run_A", [
            {"task_id": "task_002", "status": "error"},
        ], pass_rate=0.0)
        self._write_run(results_dir, "run_B", [
            {"task_id": "task_002", "status": "pass"},
        ], pass_rate=1.0)

        harness = EvalHarness(
            orchestrator_factory=lambda: None,
            results_dir=str(results_dir),
        )
        harness.compare("run_A", "run_B")

        captured = capsys.readouterr()
        assert "→" in captured.out

    def test_compare_no_changes_prints_message(self, tmp_path, capsys):
        """compare() prints 'No status changes' when both runs have identical task statuses."""
        results_dir = tmp_path / "eval_results"

        same_tasks = [{"task_id": "task_003", "status": "pass"}]
        self._write_run(results_dir, "run_A", same_tasks, pass_rate=1.0)
        self._write_run(results_dir, "run_B", same_tasks, pass_rate=1.0)

        harness = EvalHarness(
            orchestrator_factory=lambda: None,
            results_dir=str(results_dir),
        )
        harness.compare("run_A", "run_B")

        captured = capsys.readouterr()
        assert "No status changes" in captured.out

    def test_compare_prints_pass_rates_in_summary(self, tmp_path, capsys):
        """compare() prints both run IDs and their pass_rates in the summary line."""
        results_dir = tmp_path / "eval_results"

        self._write_run(results_dir, "run_A", [
            {"task_id": "task_004", "status": "fail"},
        ], pass_rate=0.25)
        self._write_run(results_dir, "run_B", [
            {"task_id": "task_004", "status": "pass"},
        ], pass_rate=0.75)

        harness = EvalHarness(
            orchestrator_factory=lambda: None,
            results_dir=str(results_dir),
        )
        harness.compare("run_A", "run_B")

        captured = capsys.readouterr()
        assert "run_A" in captured.out
        assert "run_B" in captured.out
        assert "0.25" in captured.out
        assert "0.75" in captured.out

    def test_compare_unchanged_tasks_not_printed(self, tmp_path, capsys):
        """Tasks that have the same status in both runs produce no output line."""
        results_dir = tmp_path / "eval_results"

        self._write_run(results_dir, "run_A", [
            {"task_id": "stable_task", "status": "pass"},
            {"task_id": "changing_task", "status": "fail"},
        ], pass_rate=0.5)
        self._write_run(results_dir, "run_B", [
            {"task_id": "stable_task", "status": "pass"},
            {"task_id": "changing_task", "status": "pass"},
        ], pass_rate=1.0)

        harness = EvalHarness(
            orchestrator_factory=lambda: None,
            results_dir=str(results_dir),
        )
        harness.compare("run_A", "run_B")

        captured = capsys.readouterr()
        # changing_task should appear, stable_task should not appear in change lines
        assert "changing_task" in captured.out
        # stable_task might appear in the summary line but not as a change entry
        lines_with_arrow = [line for line in captured.out.splitlines() if "→" in line]
        task_ids_in_changes = [line.split(":")[0].strip() for line in lines_with_arrow]
        assert "stable_task" not in task_ids_in_changes
