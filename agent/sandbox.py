"""
Docker-based sandboxed execution for the Codebase Maintainer Agent.

WHY Docker sandboxing?
  Running arbitrary test suites or lint commands on host carries risk — a
  misbehaving script could read secrets, make network calls, or modify the
  repository.  Wrapping execution inside a Docker container with a read-only
  repo mount and network_disabled=True eliminates those attack surfaces.

WHY import docker inside run() rather than at module level?
  docker-py is an optional dependency.  Importing it at the top of the module
  would force every caller to have docker-py installed even when Docker is not
  available.  Deferring the import means the module loads cleanly on machines
  without docker-py; the ImportError only surfaces at runtime when Docker is
  actually requested.
"""
import shutil
import subprocess
from pathlib import Path

from agent.protocols import ToolResult


class Sandbox:
    """
    Runs commands inside a Docker container mounted on the target repo.

    Falls back to direct host execution when Docker is not available on PATH,
    emitting structured warning log entries so operators know the sandbox was
    bypassed.
    """

    def __init__(
        self,
        repo_path: str,
        timeout: int = 120,
        logger: "StructuredLogger | None" = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.logger = logger
        self._docker_available = shutil.which("docker") is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, command: list[str], workspace_dir: str) -> ToolResult:
        """
        Execute *command* inside the sandbox container.

        repo_path  is mounted read-only at /repo.
        workspace_dir is mounted read-write at /workspace (test output, etc.).

        Returns a ToolResult whose *content* is the combined container logs
        (stdout + stderr merged by Docker).
        """
        if not self._docker_available:
            if self.logger:
                self.logger.log(
                    "sandbox_unavailable",
                    "WARNING",
                    reason="docker binary not found on PATH",
                )
                self.logger.log(
                    "sandbox_fallback",
                    "WARNING",
                    reason="running command directly on host — no isolation",
                )
            return self._run_host(command)

        # Import deferred so the module loads without docker-py installed.
        import docker  # noqa: PLC0415

        client = docker.from_env()
        volumes = {
            str(self.repo_path): {"bind": "/repo", "mode": "ro"},
            workspace_dir: {"bind": "/workspace", "mode": "rw"},
        }

        container = client.containers.run(
            "codebase-maintainer-sandbox:latest",
            command,
            volumes=volumes,
            network_disabled=True,
            detach=True,
        )

        try:
            result = container.wait(timeout=self.timeout)
            logs: str = container.logs().decode("utf-8", errors="replace")
            exit_code: int = result["StatusCode"]

            if exit_code != 0:
                if self.logger:
                    self.logger.log(
                        "sandbox_execution_failed",
                        "ERROR",
                        exit_code=exit_code,
                        log_tail=logs[-500:],
                    )
                return ToolResult(is_error=True, content=logs)

            return ToolResult(is_error=False, content=logs)

        except Exception:
            # Any exception from container.wait() is treated as a timeout;
            # the container must be killed to prevent it from running forever.
            container.kill()
            if self.logger:
                self.logger.log(
                    "sandbox_timeout",
                    "ERROR",
                    command=command,
                    timeout_seconds=self.timeout,
                )
            return ToolResult(is_error=True, content="sandbox_timeout")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_host(self, command: list[str]) -> ToolResult:
        """
        Run *command* directly on the host process without any isolation.

        stdout and stderr are merged into a single string in ToolResult.content
        so callers get the same interface regardless of whether Docker was used.
        """
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ToolResult(
                is_error=proc.returncode != 0,
                content=proc.stdout + proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(is_error=True, content="sandbox_timeout")
