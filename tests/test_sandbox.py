"""
Tests for agent/sandbox.py — Docker-based sandboxed execution.

Strategy:
  - Tests that touch the Docker code path inject a fake `docker` module via
    sys.modules so we never need docker-py installed in CI.
  - Tests that touch _run_host use real subprocess calls with trivial commands
    (python -c "...") so they are fast and hermetic.
  - The Sandbox instance's _docker_available flag is set directly after
    construction to decouple the test from whether Docker is on the CI PATH.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.sandbox import Sandbox
from agent.protocols import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sandbox(repo_path: str = "/tmp/repo", timeout: int = 5) -> Sandbox:
    """Return a Sandbox with a short timeout; docker availability left as-is."""
    return Sandbox(repo_path=repo_path, timeout=timeout)


def _make_mock_docker(
    status_code: int = 0,
    logs: bytes = b"",
    wait_side_effect=None,
) -> MagicMock:
    """
    Build a minimal docker module mock.

    client.containers.run() → container
    container.wait()        → {"StatusCode": status_code}  (or raises side_effect)
    container.logs()        → logs (bytes)
    """
    mock_docker = MagicMock()
    mock_client = MagicMock()
    mock_container = MagicMock()

    mock_docker.from_env.return_value = mock_client
    mock_client.containers.run.return_value = mock_container
    mock_container.logs.return_value = logs

    if wait_side_effect is not None:
        mock_container.wait.side_effect = wait_side_effect
    else:
        mock_container.wait.return_value = {"StatusCode": status_code}

    return mock_docker, mock_container


# ---------------------------------------------------------------------------
# 1. _docker_available reflects whether docker is on PATH
# ---------------------------------------------------------------------------

class TestDockerAvailableFlag:
    def test_false_when_docker_not_on_path(self):
        """_docker_available must be False when shutil.which('docker') returns None."""
        with patch("shutil.which", return_value=None):
            sandbox = Sandbox(repo_path="/tmp/repo")
        assert sandbox._docker_available is False

    def test_true_when_docker_on_path(self):
        """_docker_available must be True when shutil.which('docker') finds the binary."""
        with patch("shutil.which", return_value="/usr/bin/docker"):
            sandbox = Sandbox(repo_path="/tmp/repo")
        assert sandbox._docker_available is True


# ---------------------------------------------------------------------------
# 2 & 3. _run_host success and failure
# ---------------------------------------------------------------------------

class TestRunHost:
    def test_success_returns_non_error_result(self):
        """A command that exits 0 must produce is_error=False."""
        sandbox = _make_sandbox()
        result = sandbox._run_host(["python", "-c", "print('ok')"])

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "ok" in result.content

    def test_failure_returns_error_result(self):
        """A command that exits non-zero must produce is_error=True."""
        sandbox = _make_sandbox()
        result = sandbox._run_host(["python", "-c", "raise SystemExit(1)"])

        assert isinstance(result, ToolResult)
        assert result.is_error is True

    def test_output_combined_in_content(self):
        """stdout and stderr are both present in content."""
        sandbox = _make_sandbox()
        result = sandbox._run_host(
            ["python", "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
        )
        assert "out" in result.content
        assert "err" in result.content


# ---------------------------------------------------------------------------
# 4. run() falls back to _run_host when Docker is unavailable
# ---------------------------------------------------------------------------

class TestRunFallbackToHost:
    def test_run_calls_run_host_when_docker_unavailable(self, tmp_path):
        """When _docker_available is False, run() must delegate to _run_host."""
        sandbox = _make_sandbox(repo_path=str(tmp_path))
        sandbox._docker_available = False

        sentinel = ToolResult(is_error=False, content="host-ran")

        with patch.object(sandbox, "_run_host", return_value=sentinel) as mock_host:
            result = sandbox.run(["pytest"], str(tmp_path))

        mock_host.assert_called_once_with(["pytest"])
        assert result is sentinel

    def test_run_logs_warnings_when_docker_unavailable(self, tmp_path):
        """sandbox_unavailable and sandbox_fallback warnings must be emitted."""
        mock_logger = MagicMock()
        sandbox = Sandbox(repo_path=str(tmp_path), logger=mock_logger)
        sandbox._docker_available = False

        with patch.object(sandbox, "_run_host", return_value=ToolResult(is_error=False, content="")):
            sandbox.run(["true"], str(tmp_path))

        logged_events = [call.args[0] for call in mock_logger.log.call_args_list]
        assert "sandbox_unavailable" in logged_events
        assert "sandbox_fallback" in logged_events


# ---------------------------------------------------------------------------
# 5. Docker execution path (happy path)
# ---------------------------------------------------------------------------

class TestDockerExecutionSuccess:
    def test_successful_container_run_returns_non_error_result(self, tmp_path):
        """A container that exits 0 must yield is_error=False with captured logs."""
        mock_docker, mock_container = _make_mock_docker(
            status_code=0,
            logs=b"success output",
        )

        with patch.dict(sys.modules, {"docker": mock_docker}):
            sandbox = _make_sandbox(repo_path=str(tmp_path))
            sandbox._docker_available = True
            result = sandbox.run(["pytest"], str(tmp_path))

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "success output" in result.content

    def test_container_run_passes_correct_volumes(self, tmp_path):
        """Volumes must map repo_path → /repo (ro) and workspace_dir → /workspace (rw)."""
        mock_docker, mock_container = _make_mock_docker(status_code=0, logs=b"")
        mock_client = mock_docker.from_env.return_value
        workspace = str(tmp_path / "workspace")

        with patch.dict(sys.modules, {"docker": mock_docker}):
            sandbox = _make_sandbox(repo_path=str(tmp_path))
            sandbox._docker_available = True
            sandbox.run(["pytest"], workspace)

        _, kwargs = mock_client.containers.run.call_args
        volumes = kwargs.get("volumes") or mock_client.containers.run.call_args[0][2] if len(mock_client.containers.run.call_args[0]) > 2 else kwargs["volumes"]
        # Accept both positional and keyword-argument styles
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs.get("network_disabled") is True

    def test_non_zero_exit_returns_error_result(self, tmp_path):
        """A container that exits non-zero must yield is_error=True."""
        mock_docker, mock_container = _make_mock_docker(
            status_code=1,
            logs=b"test failed",
        )

        with patch.dict(sys.modules, {"docker": mock_docker}):
            sandbox = _make_sandbox(repo_path=str(tmp_path))
            sandbox._docker_available = True
            result = sandbox.run(["pytest"], str(tmp_path))

        assert result.is_error is True
        assert "test failed" in result.content


# ---------------------------------------------------------------------------
# 6. Sandbox timeout
# ---------------------------------------------------------------------------

class TestSandboxTimeout:
    def test_timeout_kills_container_and_returns_error(self, tmp_path):
        """When container.wait() raises, the container is killed and a timeout result returned."""
        mock_docker, mock_container = _make_mock_docker(
            wait_side_effect=Exception("connection timed out"),
        )

        with patch.dict(sys.modules, {"docker": mock_docker}):
            sandbox = _make_sandbox(repo_path=str(tmp_path))
            sandbox._docker_available = True
            result = sandbox.run(["pytest"], str(tmp_path))

        mock_container.kill.assert_called_once()
        assert result.is_error is True
        assert result.content == "sandbox_timeout"

    def test_timeout_logs_sandbox_timeout_event(self, tmp_path):
        """sandbox_timeout must be logged when container.wait() raises."""
        mock_logger = MagicMock()
        mock_docker, mock_container = _make_mock_docker(
            wait_side_effect=Exception("timeout"),
        )

        with patch.dict(sys.modules, {"docker": mock_docker}):
            sandbox = Sandbox(
                repo_path=str(tmp_path),
                timeout=5,
                logger=mock_logger,
            )
            sandbox._docker_available = True
            sandbox.run(["pytest"], str(tmp_path))

        logged_events = [call.args[0] for call in mock_logger.log.call_args_list]
        assert "sandbox_timeout" in logged_events
