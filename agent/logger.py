"""
Structured JSON logger for the Codebase Maintainer Agent.

WHY structured logging instead of Python's stdlib logging module?
  The stdlib logging module emits human-readable text by default. Structured
  JSON logs are machine-parseable â€” you can grep, filter with jq, or feed them
  into a log aggregator without writing a parser. For an agent that makes many
  LLM calls and tool dispatches, structured logs are essential for post-hoc
  debugging: you can replay exactly what the agent did by filtering log events.

WHY stderr instead of stdout?
  The agent writes diff content to stdout (via the CLI) so it can be piped to
  patch/apply commands. Writing logs to stderr keeps the two streams separate
  and avoids corrupting diff output with log lines.

WHY a thread lock?
  Phase 3 may introduce concurrent issue processing. The lock ensures log lines
  are never interleaved even if multiple threads call log() simultaneously.
"""
import datetime
import json
import sys
import threading
from typing import Any


# Ordered from least to most severe â€” used to filter by minimum level
_LEVEL_ORDER: dict[str, int] = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
}


class StructuredLogger:
    """
    Emits newline-delimited JSON log entries to stderr.

    Each entry has the shape:
        {
            "timestamp_utc": "<ISO 8601>",
            "level": "INFO",
            "event": "<event_name>",
            "payload": { ... arbitrary key-value pairs ... }
        }

    Usage:
        logger = StructuredLogger(min_level="INFO", verbose=False)
        logger.log("llm_call", "INFO", model="claude-haiku-4-5-20251001", call_number=1)
        logger.log("tool_dispatch", "INFO", tool_name="read_file", arguments={"path": "foo.py"})
    """

    def __init__(self, min_level: str = "INFO", verbose: bool = False) -> None:
        """
        min_level: minimum severity to emit. Events below this level are silently dropped.
        verbose:   when True, DEBUG-level events are also emitted regardless of min_level.
                   This powers the --verbose CLI flag.
        """
        if min_level not in _LEVEL_ORDER:
            raise ValueError(
                f"Invalid log level {min_level!r}. Must be one of: {list(_LEVEL_ORDER)}"
            )
        self._min_level = _LEVEL_ORDER[min_level]
        self._verbose = verbose
        self._lock = threading.Lock()

    def log(self, event: str, level: str = "INFO", **payload: Any) -> None:
        """
        Emit a structured log entry.

        event:   short snake_case identifier for the event type (e.g. "llm_call")
        level:   one of DEBUG, INFO, WARNING, ERROR
        payload: arbitrary key=value pairs included in the "payload" field
        """
        # --verbose enables DEBUG regardless of min_level
        effective_min = 0 if self._verbose else self._min_level
        if _LEVEL_ORDER.get(level, 1) < effective_min:
            return

        entry = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "event": event,
            "payload": self._sanitize(payload),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            print(line, file=sys.stderr, flush=True)

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """
        Recursively replace non-JSON-serializable values with a descriptive placeholder.

        WHY replace instead of dropping?
          Dropping fields silently makes log analysis confusing â€” you see an event
          but some payload fields are missing with no indication why. Replacing with
          a typed placeholder like "<non-serializable: MyClass>" makes it obvious
          that a field existed but couldn't be serialized, and tells you what type
          it was so you can fix the call site.
        """
        if isinstance(obj, dict):
            return {str(k): StructuredLogger._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [StructuredLogger._sanitize(v) for v in obj]
        try:
            # Fast-path: try serializing to catch all non-serializable types at once
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return f"<non-serializable: {type(obj).__name__}>"
