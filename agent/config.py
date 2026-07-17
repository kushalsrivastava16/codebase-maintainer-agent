"""
Configuration loader for the Codebase Maintainer Agent.

WHY this module exists as a separate layer:
  The orchestrator and CLI need a single, validated AgentConfig object. Keeping
  all loading, merging, and validation here means the rest of the codebase can
  simply import load_config() without knowing where values come from. It also
  makes it trivial to unit-test the config pipeline in isolation.

Merge order (later overrides earlier):
  1. DEFAULT_CONFIG  — safe, conservative values baked into source
  2. YAML file       — user-level tuning without touching the CLI
  3. CLI overrides   — per-invocation flags (highest precedence)

  This three-layer design is standard for twelve-factor apps: defaults ship with
  the code, configuration ships with the deployment, and runtime flags override
  both without requiring a file edit.

Exit code 6:
  The agent uses numbered exit codes so shell scripts can branch on the exact
  failure mode. Code 6 is reserved exclusively for configuration errors — both
  YAML parse failures and field validation failures — so callers can distinguish
  "bad config file" from "bad runtime arguments" (code 1) or "task failed"
  (code 3) without parsing stderr text.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # 50 000 tokens is large enough for most single-file maintenance tasks
    # (a 1 000-line Python file + ruff output + a few back-and-forth turns
    # fits well within this budget) but small enough to keep per-task costs
    # under ~$0.20 on Haiku-class models.
    "max_tokens_per_task": 50_000,
    # 10 iterations is sufficient for a lint fix or test-generation task that
    # requires at most a handful of read→lint→fix→verify cycles.  Keeping this
    # low prevents runaway loops from consuming budget silently.
    "max_iterations": 10,
    # Relative paths are intentional — the agent is expected to be run from
    # the repository root, keeping all artefacts co-located with the project.
    "output_dir": "./agent_output",
    "memory_db": "./agent_memory.db",
    # Sandboxing is off by default so the agent works out of the box without
    # Docker.  Users who want isolation enable it explicitly.
    "sandbox_enabled": False,
    # 120 seconds gives the container enough time to install nothing (all deps
    # are baked into the image) and run a reasonably large pytest suite.
    "sandbox_timeout_seconds": 120,
    # 24 hours is a natural "working day" boundary — if the agent successfully
    # fixed lint errors in a file today, there is no point re-running it until
    # new code has been committed.
    "dedup_window_hours": 24,
    "log_level": "INFO",
    # claude-3-5-haiku-20241022 is the default because it is the fastest and cheapest
    # Anthropic model suitable for mechanical code-maintenance tasks (lint fix,
    # TODO conversion).  Upgrade to Sonnet in agent_config.yaml for tasks that
    # require deeper reasoning (test generation, triage).
    "model": "claude-haiku-4-5-20251001",
    # github_repo is intentionally None in defaults — triage_issues tasks will
    # fail fast with a clear error rather than silently operating on the wrong
    # repository.
    "github_repo": None,
}

# Fields that must be Python ints when present in the merged config dict.
# Booleans are excluded: YAML "true"/"false" already deserialises to bool, and
# sandbox_enabled is never expected to be a bare integer.
NUMERIC_FIELDS: frozenset[str] = frozenset(
    {
        "max_tokens_per_task",
        "max_iterations",
        "sandbox_timeout_seconds",
        "dedup_window_hours",
    }
)

# Accepted values for the log_level field (case-sensitive, uppercase).
VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


# ---------------------------------------------------------------------------
# AgentConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """
    Validated, fully-typed snapshot of all agent configuration.

    Instances are created exclusively by load_config() after validation, so any
    AgentConfig object that exists in memory is guaranteed to be valid.  This
    removes the need for defensive checks elsewhere in the codebase.
    """

    # Token budget per task run.  Abort guard fires before the next LLM call
    # when cumulative (input + output) tokens exceed this value.
    max_tokens_per_task: int = 50_000

    # Hard cap on the number of LLM turns per task.  Prevents infinite loops
    # when the model keeps requesting tools without converging.
    max_iterations: int = 10

    # Directory where .diff and companion .json metadata files are written.
    output_dir: str = "./agent_output"

    # SQLite database file path for task history and deduplication.
    memory_db: str = "./agent_memory.db"

    # When True, code execution is routed through the Docker sandbox.
    sandbox_enabled: bool = False

    # Maximum wall-clock seconds the Docker container may run before being
    # killed and logged as sandbox_timeout.
    sandbox_timeout_seconds: int = 120

    # Deduplication window: if the same (task_type, target_path) pair succeeded
    # within this many hours, the current invocation is skipped (exit 0).
    dedup_window_hours: int = 24

    # Minimum log severity emitted to stderr.
    log_level: str = "INFO"

    # Anthropic model identifier passed to messages.create().
    model: str = "claude-haiku-4-5-20251001"

    # GitHub repository in "owner/repo" format.  Required for triage_issues.
    github_repo: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    config_path: str = "./agent_config.yaml",
    cli_overrides: dict | None = None,
) -> AgentConfig:
    """
    Build a validated AgentConfig from defaults, an optional YAML file, and
    optional CLI overrides.

    Merge order (later wins):
      1. DEFAULT_CONFIG
      2. YAML file at config_path (skipped silently if the file does not exist)
      3. cli_overrides (keys with None values are ignored so callers can pass
         the raw click option dict without pre-filtering)

    WHY None values in cli_overrides are ignored:
      Click sets unprovided optional flags to None.  Propagating None would
      overwrite a valid YAML value with None, effectively deleting the user's
      configuration.  Only explicitly-supplied, non-None values should win.

    Exits:
      sys.exit(6) if the YAML file exists but has a syntax error.
      sys.exit(6) if any field fails type or value validation.
    """
    merged = dict(DEFAULT_CONFIG)

    # --- Layer 2: YAML file ---
    yaml_path = Path(config_path)
    if yaml_path.exists():
        try:
            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            # Print to stderr so the caller can capture it; then exit.
            print(f"config_parse_error: {exc}", file=sys.stderr)
            sys.exit(6)
        merged.update(loaded)

    # --- Layer 3: CLI overrides (skip None values) ---
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                merged[key] = value

    # --- Validation ---
    errors = _validate(merged)
    if errors:
        for err in errors:
            print(f"config_error: {err}", file=sys.stderr)
        sys.exit(6)

    # Build the dataclass from only the fields it declares, ignoring any
    # extra keys that may have come from a future YAML version.
    known_fields = AgentConfig.__dataclass_fields__
    return AgentConfig(**{k: merged[k] for k in known_fields if k in merged})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(data: dict) -> list[str]:
    """
    Return a list of human-readable error strings.  An empty list means valid.

    WHY a list and not a single string:
      When multiple fields are invalid (e.g., both max_iterations and log_level
      are wrong), reporting all errors at once lets the user fix everything in
      one edit rather than discovering problems one at a time.
    """
    errors: list[str] = []

    # Validate numeric fields: must be int, not float, not str, not bool.
    # WHY exclude bool: in Python, bool is a subclass of int, so isinstance(True, int)
    # is True.  A YAML value of "true" would pass unnoticed and silently set
    # max_tokens_per_task to 1, which would be very confusing.
    for field in NUMERIC_FIELDS:
        val = data.get(field)
        if val is not None:
            if isinstance(val, bool) or not isinstance(val, int):
                errors.append(
                    f"{field}: expected integer, got {type(val).__name__!r} ({val!r})"
                )

    # Validate log_level
    log_level = data.get("log_level")
    if log_level is not None and log_level not in VALID_LOG_LEVELS:
        errors.append(
            f"log_level: must be one of {sorted(VALID_LOG_LEVELS)!r}, got {log_level!r}"
        )

    return errors
