"""
Tests for agent/logger.py — StructuredLogger.

All tests capture stderr via pytest's capsys fixture and parse the captured
output as JSON to verify structure and content.
"""
import json
import threading
from datetime import datetime

import pytest

from agent.logger import StructuredLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_lines(capsys) -> list[dict]:
    """Read stderr, split into lines, parse each non-empty line as JSON."""
    captured = capsys.readouterr()
    lines = [l for l in captured.err.splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def _single_entry(capsys) -> dict:
    """Assert exactly one log line was emitted and return it as a dict."""
    entries = _capture_lines(capsys)
    assert len(entries) == 1, f"Expected 1 log entry, got {len(entries)}: {entries}"
    return entries[0]


# ---------------------------------------------------------------------------
# 1. Required fields present
# ---------------------------------------------------------------------------

def test_required_fields_present(capsys):
    logger = StructuredLogger()
    logger.log("test_event", "INFO", key="value")
    entry = _single_entry(capsys)
    assert "timestamp_utc" in entry
    assert "level" in entry
    assert "event" in entry
    assert "payload" in entry


# ---------------------------------------------------------------------------
# 2. timestamp_utc is valid ISO 8601
# ---------------------------------------------------------------------------

def test_timestamp_is_iso8601(capsys):
    logger = StructuredLogger()
    logger.log("ts_check", "INFO")
    entry = _single_entry(capsys)
    # datetime.fromisoformat raises ValueError if not valid ISO 8601
    parsed = datetime.fromisoformat(entry["timestamp_utc"])
    assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# 3. Events below min_level are NOT emitted
# ---------------------------------------------------------------------------

def test_below_min_level_suppressed(capsys):
    logger = StructuredLogger(min_level="WARNING")
    logger.log("low_event", "INFO")
    logger.log("debug_event", "DEBUG")
    entries = _capture_lines(capsys)
    assert entries == [], f"Expected no output, got: {entries}"


# ---------------------------------------------------------------------------
# 4. Events at or above min_level ARE emitted
# ---------------------------------------------------------------------------

def test_at_min_level_emitted(capsys):
    logger = StructuredLogger(min_level="WARNING")
    logger.log("warn_event", "WARNING")
    entry = _single_entry(capsys)
    assert entry["level"] == "WARNING"
    assert entry["event"] == "warn_event"


def test_above_min_level_emitted(capsys):
    logger = StructuredLogger(min_level="WARNING")
    logger.log("error_event", "ERROR")
    entry = _single_entry(capsys)
    assert entry["level"] == "ERROR"


# ---------------------------------------------------------------------------
# 5. DEBUG suppressed by default; emitted when verbose=True
# ---------------------------------------------------------------------------

def test_debug_suppressed_by_default(capsys):
    logger = StructuredLogger(min_level="INFO", verbose=False)
    logger.log("debug_msg", "DEBUG")
    entries = _capture_lines(capsys)
    assert entries == []


def test_debug_emitted_when_verbose(capsys):
    logger = StructuredLogger(min_level="INFO", verbose=True)
    logger.log("debug_msg", "DEBUG")
    entry = _single_entry(capsys)
    assert entry["level"] == "DEBUG"
    assert entry["event"] == "debug_msg"


def test_verbose_does_not_suppress_higher_levels(capsys):
    logger = StructuredLogger(min_level="INFO", verbose=True)
    logger.log("info_msg", "INFO")
    logger.log("warn_msg", "WARNING")
    entries = _capture_lines(capsys)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# 6. Non-serializable payload values are replaced with placeholder
# ---------------------------------------------------------------------------

class _Unserializable:
    """Helper class that is not JSON-serializable."""
    pass


def test_non_serializable_top_level_replaced(capsys):
    logger = StructuredLogger()
    obj = _Unserializable()
    logger.log("bad_payload", "INFO", thing=obj)
    entry = _single_entry(capsys)
    assert entry["payload"]["thing"] == "<non-serializable: _Unserializable>"


def test_serializable_values_pass_through(capsys):
    logger = StructuredLogger()
    logger.log("good_payload", "INFO", num=42, text="hello", flag=True, nothing=None)
    entry = _single_entry(capsys)
    assert entry["payload"]["num"] == 42
    assert entry["payload"]["text"] == "hello"
    assert entry["payload"]["flag"] is True
    assert entry["payload"]["nothing"] is None


# ---------------------------------------------------------------------------
# 7. Nested non-serializable values in dicts and lists
# ---------------------------------------------------------------------------

def test_nested_non_serializable_in_dict(capsys):
    logger = StructuredLogger()
    logger.log("nested_dict", "INFO", data={"ok": 1, "bad": _Unserializable()})
    entry = _single_entry(capsys)
    assert entry["payload"]["data"]["ok"] == 1
    assert entry["payload"]["data"]["bad"] == "<non-serializable: _Unserializable>"


def test_nested_non_serializable_in_list(capsys):
    logger = StructuredLogger()
    logger.log("nested_list", "INFO", items=[1, _Unserializable(), "three"])
    entry = _single_entry(capsys)
    assert entry["payload"]["items"][0] == 1
    assert entry["payload"]["items"][1] == "<non-serializable: _Unserializable>"
    assert entry["payload"]["items"][2] == "three"


def test_deeply_nested_non_serializable(capsys):
    logger = StructuredLogger()
    logger.log("deep_nest", "INFO", outer={"inner": [_Unserializable()]})
    entry = _single_entry(capsys)
    assert entry["payload"]["outer"]["inner"][0] == "<non-serializable: _Unserializable>"


def test_tuple_values_treated_like_list(capsys):
    logger = StructuredLogger()
    logger.log("tuple_val", "INFO", coords=(1, 2, 3))
    entry = _single_entry(capsys)
    assert entry["payload"]["coords"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 8. Output goes to stderr, not stdout
# ---------------------------------------------------------------------------

def test_output_goes_to_stderr_not_stdout(capsys):
    logger = StructuredLogger()
    logger.log("stderr_check", "INFO")
    captured = capsys.readouterr()
    # stderr should have content, stdout should be empty
    assert captured.err.strip() != ""
    assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# 9. Invalid min_level raises ValueError at construction
# ---------------------------------------------------------------------------

def test_invalid_min_level_raises():
    with pytest.raises(ValueError, match="Invalid log level"):
        StructuredLogger(min_level="TRACE")


def test_invalid_min_level_empty_string_raises():
    with pytest.raises(ValueError, match="Invalid log level"):
        StructuredLogger(min_level="")


def test_valid_min_levels_do_not_raise():
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        StructuredLogger(min_level=level)  # should not raise


# ---------------------------------------------------------------------------
# 10. Thread safety — no interleaved JSON lines
# ---------------------------------------------------------------------------

def test_thread_safety_no_interleaved_json(capsys):
    """
    Spin up 20 threads each emitting 10 log lines.
    Every captured stderr line must be valid, complete JSON.
    """
    logger = StructuredLogger(min_level="DEBUG")
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(10):
                logger.log("thread_event", "DEBUG", thread=thread_id, iteration=i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Worker threads raised exceptions: {errors}"

    entries = _capture_lines(capsys)
    # 20 threads × 10 iterations = 200 entries
    assert len(entries) == 200, f"Expected 200 entries, got {len(entries)}"
    # Every entry must have the required fields — proves no interleaving corruption
    for entry in entries:
        assert "timestamp_utc" in entry
        assert "level" in entry
        assert "event" in entry
        assert "payload" in entry


# ---------------------------------------------------------------------------
# 11. event and level values are preserved correctly
# ---------------------------------------------------------------------------

def test_event_name_preserved(capsys):
    logger = StructuredLogger()
    logger.log("my_special_event", "WARNING", x=99)
    entry = _single_entry(capsys)
    assert entry["event"] == "my_special_event"
    assert entry["level"] == "WARNING"
    assert entry["payload"]["x"] == 99


def test_default_level_is_info(capsys):
    logger = StructuredLogger()
    logger.log("default_level_event")  # no level kwarg
    entry = _single_entry(capsys)
    assert entry["level"] == "INFO"
