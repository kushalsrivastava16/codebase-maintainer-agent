"""
Tests for agent/config.py â€” load_config() and AgentConfig.

Test matrix:
  1. Default config is returned when no YAML file exists
  2. YAML file values override defaults
  3. CLI overrides take precedence over YAML file values
  4. Numeric field with a non-numeric value â†’ exits with code 6
  5. Invalid log_level value â†’ exits with code 6
  6. Malformed YAML syntax â†’ exits with code 6
  7. None values in cli_overrides are ignored (don't overwrite YAML values)
  8. github_repo=None is accepted (it is an optional field)

All tests that exercise SystemExit use pytest.raises(SystemExit) and verify
the exit code is exactly 6.

Temp YAML files are created using the tmp_path fixture so they are isolated
per test and cleaned up automatically.
"""
from __future__ import annotations

import textwrap

import pytest

from agent.config import AgentConfig, load_config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, content: str):
    """Write a YAML file under tmp_path and return its string path."""
    p = tmp_path / "agent_config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. Default config is returned when no YAML file exists
# ---------------------------------------------------------------------------


def test_defaults_when_no_file(tmp_path):
    """A nonexistent config path should silently return DEFAULT_CONFIG values."""
    nonexistent = str(tmp_path / "no_such_file.yaml")
    cfg = load_config(config_path=nonexistent)

    assert isinstance(cfg, AgentConfig)
    assert cfg.max_tokens_per_task == 50_000
    assert cfg.max_iterations == 10
    assert cfg.output_dir == "./agent_output"
    assert cfg.memory_db == "./agent_memory.db"
    assert cfg.sandbox_enabled is False
    assert cfg.sandbox_timeout_seconds == 120
    assert cfg.dedup_window_hours == 24
    assert cfg.log_level == "INFO"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.github_repo is None


# ---------------------------------------------------------------------------
# 2. YAML file values override defaults
# ---------------------------------------------------------------------------


def test_yaml_overrides_defaults(tmp_path):
    """Values present in the YAML file should replace the DEFAULT_CONFIG values."""
    path = _write_yaml(
        tmp_path,
        """
        max_tokens_per_task: 100000
        max_iterations: 5
        output_dir: /tmp/custom_output
        log_level: DEBUG
        model: claude-sonnet-4-5
        github_repo: owner/repo
        """,
    )
    cfg = load_config(config_path=path)

    assert cfg.max_tokens_per_task == 100_000
    assert cfg.max_iterations == 5
    assert cfg.output_dir == "/tmp/custom_output"
    assert cfg.log_level == "DEBUG"
    assert cfg.model == "claude-sonnet-4-5"
    assert cfg.github_repo == "owner/repo"
    # Fields not in YAML should retain defaults
    assert cfg.sandbox_enabled is False
    assert cfg.sandbox_timeout_seconds == 120


def test_yaml_partial_override_keeps_unmentioned_defaults(tmp_path):
    """A YAML file that sets only one field should leave all others at default."""
    path = _write_yaml(tmp_path, "max_iterations: 3\n")
    cfg = load_config(config_path=path)

    assert cfg.max_iterations == 3
    assert cfg.max_tokens_per_task == 50_000  # still default
    assert cfg.log_level == "INFO"  # still default


# ---------------------------------------------------------------------------
# 3. CLI overrides take precedence over YAML file values
# ---------------------------------------------------------------------------


def test_cli_overrides_yaml(tmp_path):
    """CLI overrides must win over YAML values (highest precedence layer)."""
    path = _write_yaml(
        tmp_path,
        """
        max_tokens_per_task: 20000
        log_level: DEBUG
        """,
    )
    cfg = load_config(
        config_path=path,
        cli_overrides={"max_tokens_per_task": 99_000, "log_level": "WARNING"},
    )

    assert cfg.max_tokens_per_task == 99_000
    assert cfg.log_level == "WARNING"


def test_cli_overrides_no_yaml(tmp_path):
    """CLI overrides should also work when there is no YAML file."""
    nonexistent = str(tmp_path / "missing.yaml")
    cfg = load_config(
        config_path=nonexistent,
        cli_overrides={"max_iterations": 20, "model": "claude-opus"},
    )

    assert cfg.max_iterations == 20
    assert cfg.model == "claude-opus"
    assert cfg.max_tokens_per_task == 50_000  # untouched default


# ---------------------------------------------------------------------------
# 4. Numeric field with non-numeric value â†’ exit(6)
# ---------------------------------------------------------------------------


def test_non_numeric_max_tokens_exits_6(tmp_path):
    """A string value for max_tokens_per_task must trigger validation exit 6."""
    path = _write_yaml(tmp_path, 'max_tokens_per_task: "not_a_number"\n')
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


def test_non_numeric_max_iterations_exits_6(tmp_path):
    """A float value for max_iterations must trigger validation exit 6."""
    path = _write_yaml(tmp_path, "max_iterations: 3.5\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


def test_non_numeric_sandbox_timeout_exits_6(tmp_path):
    """A boolean value for sandbox_timeout_seconds must trigger exit 6.

    WHY test bool? In Python, bool is a subclass of int, so isinstance(True, int)
    is True.  The validator explicitly rejects booleans for numeric fields to
    prevent YAML "true" from silently setting sandbox_timeout_seconds = 1.
    """
    path = _write_yaml(tmp_path, "sandbox_timeout_seconds: true\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


def test_non_numeric_dedup_window_exits_6(tmp_path):
    """A list value for dedup_window_hours must trigger validation exit 6."""
    path = _write_yaml(tmp_path, "dedup_window_hours: [1, 2, 3]\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


def test_non_numeric_via_cli_override_exits_6(tmp_path):
    """A non-int passed through cli_overrides should also fail validation."""
    nonexistent = str(tmp_path / "missing.yaml")
    with pytest.raises(SystemExit) as exc_info:
        load_config(
            config_path=nonexistent,
            cli_overrides={"max_tokens_per_task": "fifty_thousand"},
        )
    assert exc_info.value.code == 6


# ---------------------------------------------------------------------------
# 5. Invalid log_level value â†’ exit(6)
# ---------------------------------------------------------------------------


def test_invalid_log_level_exits_6(tmp_path):
    """An unrecognised log_level string must trigger validation exit 6."""
    path = _write_yaml(tmp_path, "log_level: VERBOSE\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


def test_lowercase_log_level_exits_6(tmp_path):
    """log_level values are case-sensitive; 'info' (lowercase) is invalid."""
    path = _write_yaml(tmp_path, "log_level: info\n")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_valid_log_levels_accepted(tmp_path, level):
    """All four accepted log level strings should load without error."""
    path = _write_yaml(tmp_path, f"log_level: {level}\n")
    cfg = load_config(config_path=path)
    assert cfg.log_level == level


# ---------------------------------------------------------------------------
# 6. Malformed YAML â†’ exit(6)
# ---------------------------------------------------------------------------


def test_malformed_yaml_exits_6(tmp_path):
    """A YAML syntax error in the config file must exit with code 6."""
    path = tmp_path / "agent_config.yaml"
    # Deliberately invalid YAML (unbalanced bracket / tab indentation error)
    path.write_text("key: [unclosed bracket\nanother: value\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=str(path))
    assert exc_info.value.code == 6


def test_malformed_yaml_with_tab_indentation_exits_6(tmp_path):
    """YAML does not allow tab characters for indentation."""
    path = tmp_path / "agent_config.yaml"
    path.write_text("key:\n\tvalue: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=str(path))
    assert exc_info.value.code == 6


# ---------------------------------------------------------------------------
# 7. None values in cli_overrides are ignored
# ---------------------------------------------------------------------------


def test_none_cli_overrides_ignored(tmp_path):
    """None values in cli_overrides must not overwrite YAML-supplied values."""
    path = _write_yaml(tmp_path, "max_iterations: 7\nlog_level: WARNING\n")
    # Simulate click passing unprovided options as None
    cfg = load_config(
        config_path=path,
        cli_overrides={
            "max_iterations": None,  # should NOT overwrite the YAML value of 7
            "log_level": None,  # should NOT overwrite WARNING
            "model": None,  # should fall back to default
        },
    )
    assert cfg.max_iterations == 7
    assert cfg.log_level == "WARNING"
    assert cfg.model == "claude-haiku-4-5-20251001"  # default, not None


def test_none_cli_overrides_with_no_yaml(tmp_path):
    """None values in cli_overrides must not overwrite DEFAULT_CONFIG values."""
    nonexistent = str(tmp_path / "missing.yaml")
    cfg = load_config(
        config_path=nonexistent,
        cli_overrides={"max_tokens_per_task": None, "log_level": None},
    )
    assert cfg.max_tokens_per_task == 50_000
    assert cfg.log_level == "INFO"


# ---------------------------------------------------------------------------
# 8. github_repo can be None
# ---------------------------------------------------------------------------


def test_github_repo_default_is_none(tmp_path):
    """github_repo must default to None when not specified anywhere."""
    nonexistent = str(tmp_path / "missing.yaml")
    cfg = load_config(config_path=nonexistent)
    assert cfg.github_repo is None


def test_github_repo_null_in_yaml_is_accepted(tmp_path):
    """Explicitly setting github_repo: null in YAML should keep it None."""
    path = _write_yaml(tmp_path, "github_repo: null\n")
    cfg = load_config(config_path=path)
    assert cfg.github_repo is None


def test_github_repo_set_via_yaml(tmp_path):
    """github_repo can be set to a string value via YAML."""
    path = _write_yaml(tmp_path, "github_repo: myorg/myrepo\n")
    cfg = load_config(config_path=path)
    assert cfg.github_repo == "myorg/myrepo"


def test_github_repo_set_via_cli_override(tmp_path):
    """github_repo can be set via cli_overrides and takes precedence over YAML None."""
    path = _write_yaml(tmp_path, "github_repo: null\n")
    cfg = load_config(
        config_path=path,
        cli_overrides={"github_repo": "cli-owner/cli-repo"},
    )
    assert cfg.github_repo == "cli-owner/cli-repo"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_empty_yaml_file_returns_defaults(tmp_path):
    """An empty YAML file is valid (yaml.safe_load returns None); defaults apply."""
    path = tmp_path / "agent_config.yaml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(config_path=str(path))
    assert cfg.max_tokens_per_task == 50_000
    assert cfg.log_level == "INFO"


def test_multiple_validation_errors_all_reported(tmp_path, capsys):
    """When multiple fields are invalid, all errors should be printed before exit."""
    path = _write_yaml(
        tmp_path,
        """
        max_tokens_per_task: "bad"
        log_level: TRACE
        """,
    )
    with pytest.raises(SystemExit) as exc_info:
        load_config(config_path=path)
    assert exc_info.value.code == 6
    captured = capsys.readouterr()
    # Both errors should appear in stderr output
    assert "max_tokens_per_task" in captured.err
    assert "log_level" in captured.err


def test_cli_overrides_none_dict_is_safe(tmp_path):
    """Passing cli_overrides=None should not raise any error."""
    nonexistent = str(tmp_path / "missing.yaml")
    cfg = load_config(config_path=nonexistent, cli_overrides=None)
    assert isinstance(cfg, AgentConfig)
