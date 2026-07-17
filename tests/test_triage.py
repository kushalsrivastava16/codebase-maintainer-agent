"""
Tests for agent/triage.py.

Tests cover:
  1. run() fetches issues and processes them in ascending issue_number order
  2. run() returns early and logs triage_fetch_failed when the GitHub fetch fails
  3. _process_issue() skips issues already labelled agent-triaged (logs triage_skipped_recent)
  4. _process_issue() posts an insufficient_detail comment without calling LLM
     when body has fewer than MIN_BODY_CHARS non-whitespace characters
  5. _process_issue() calls _llm_triage and _post_comment for normal issues
  6. _llm_triage() returns a dict with priority / type / reproduction_status keys
  7. _llm_triage() returns safe defaults when the LLM response is not valid JSON
  8. _post_comment() returns True and posts comment + label on success
  9. _post_comment() logs triage_comment_failed and returns False on HTTP error
 10. _post_comment() prepends the Note line when note is provided
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from agent.logger import StructuredLogger
from agent.protocols import ToolResult
from agent.triage import MIN_BODY_CHARS, TRIAGE_LABEL, TriageAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(
    issues: list[dict] | None = None,
    fetch_error: bool = False,
) -> tuple[TriageAgent, MagicMock, MagicMock]:
    """
    Build a TriageAgent with fully-mocked GitHub client and logger.

    Returns (agent, mock_github, mock_logger).

    The mock_github.execute() returns either an error ToolResult (fetch_error=True)
    or a success ToolResult whose content is a JSON-encoded list of issues.
    """
    mock_github = MagicMock()
    mock_github.repo = "owner/repo"

    if fetch_error:
        mock_github.execute.return_value = ToolResult(
            is_error=True, content="rate_limit_abort"
        )
    else:
        payload = json.dumps(issues or [])
        mock_github.execute.return_value = ToolResult(is_error=False, content=payload)

    mock_logger = MagicMock(spec=StructuredLogger)

    # Patch anthropic.Anthropic so __init__ does not require ANTHROPIC_API_KEY.
    with patch("agent.triage.anthropic.Anthropic"):
        agent = TriageAgent(mock_github, mock_logger)

    return agent, mock_github, mock_logger


def make_llm_response(text: str) -> MagicMock:
    """Build a minimal stand-in for an anthropic.types.Message."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


def make_http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError with the given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=MagicMock(),
        response=mock_response,
    )


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------

class TestRun:

    def test_run_fetches_issues_via_github_execute(self):
        """run() calls github.execute with operation=fetch_issues."""
        agent, mock_github, _ = make_agent(issues=[])
        agent.run()
        mock_github.execute.assert_called_once_with({"operation": "fetch_issues"})

    def test_run_processes_issues_in_ascending_order(self):
        """Issues are processed in ascending issue_number order regardless of fetch order."""
        issues = [
            {"issue_number": 3, "title": "C", "body": "body for C issue here", "labels": []},
            {"issue_number": 1, "title": "A", "body": "body for A issue here", "labels": []},
            {"issue_number": 2, "title": "B", "body": "body for B issue here", "labels": []},
        ]
        agent, _, mock_logger = make_agent(issues=issues)

        processed_order: list[int] = []

        def fake_process(issue: dict) -> None:
            processed_order.append(issue["issue_number"])

        agent._process_issue = fake_process  # type: ignore[method-assign]
        agent.run()

        assert processed_order == [1, 2, 3]

    def test_run_logs_fetch_failure_and_returns_early(self):
        """When fetch fails, triage_fetch_failed is logged and no issues are processed."""
        agent, _, mock_logger = make_agent(fetch_error=True)

        called = []
        agent._process_issue = lambda i: called.append(i)  # type: ignore[method-assign]
        agent.run()

        mock_logger.log.assert_called_once_with(
            "triage_fetch_failed", "ERROR", error="rate_limit_abort"
        )
        assert called == []

    def test_run_with_empty_issue_list_does_nothing(self):
        """run() completes silently when there are no open issues."""
        agent, _, mock_logger = make_agent(issues=[])
        agent.run()
        mock_logger.log.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _process_issue()
# ---------------------------------------------------------------------------

class TestProcessIssue:

    def _make_agent_no_api(self) -> tuple[TriageAgent, MagicMock]:
        """Return an agent with mocked github and a real (silent) logger."""
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        logger = StructuredLogger(min_level="ERROR")
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, logger)
        return agent, mock_github

    def test_skips_already_triaged_issue(self):
        """Issues with TRIAGE_LABEL are skipped; triage_skipped_recent is logged."""
        mock_logger = MagicMock(spec=StructuredLogger)
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, mock_logger)

        issue = {
            "issue_number": 5,
            "title": "Old issue",
            "body": "This issue has enough body text.",
            "labels": [TRIAGE_LABEL],
        }
        agent._process_issue(issue)

        mock_logger.log.assert_called_once_with(
            "triage_skipped_recent", "INFO", issue_number=5
        )

    def test_skips_without_calling_llm_or_posting_comment(self):
        """When skipping, neither _llm_triage nor _post_comment are called."""
        agent, mock_github = self._make_agent_no_api()

        called_llm = []
        called_post = []
        agent._llm_triage = lambda i: called_llm.append(i) or {}  # type: ignore[method-assign]
        agent._post_comment = lambda *a, **kw: called_post.append(a)  # type: ignore[method-assign]

        issue = {
            "issue_number": 7,
            "title": "Triaged",
            "body": "Some detailed body.",
            "labels": [TRIAGE_LABEL],
        }
        agent._process_issue(issue)

        assert called_llm == []
        assert called_post == []

    def test_insufficient_detail_posts_comment_without_llm(self):
        """Body with fewer than MIN_BODY_CHARS non-whitespace chars: comment posted, LLM skipped."""
        mock_logger = MagicMock(spec=StructuredLogger)
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, mock_logger)

        called_llm = []
        posted_args: list[tuple] = []

        agent._llm_triage = lambda i: called_llm.append(i) or {}  # type: ignore[method-assign]

        def fake_post(issue_number, assessment, note=None):
            posted_args.append((issue_number, assessment, note))
            return True

        agent._post_comment = fake_post  # type: ignore[method-assign]

        # Body with fewer than MIN_BODY_CHARS non-whitespace characters
        short_body = "   \n  "  # only whitespace — 0 non-ws chars
        issue = {
            "issue_number": 9,
            "title": "Short",
            "body": short_body,
            "labels": [],
        }
        agent._process_issue(issue)

        assert called_llm == [], "LLM should not be called for insufficient_detail"
        assert len(posted_args) == 1
        issue_number, assessment, note = posted_args[0]
        assert issue_number == 9
        assert assessment["priority"] == "low"
        assert assessment["type"] == "question"
        assert assessment["reproduction_status"] == "not_applicable"
        assert note == "insufficient_detail"

    def test_insufficient_detail_boundary_exactly_min_body_chars(self):
        """Body with exactly MIN_BODY_CHARS non-whitespace chars is NOT flagged insufficient."""
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        logger = StructuredLogger(min_level="ERROR")
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, logger)

        llm_called = []
        posted = []

        agent._llm_triage = lambda i: (llm_called.append(i), {"priority": "low", "type": "bug", "reproduction_status": "unconfirmed"})[1]  # type: ignore[method-assign]
        agent._post_comment = lambda *a, **kw: posted.append(a)  # type: ignore[method-assign]

        # Exactly MIN_BODY_CHARS non-whitespace characters
        body = "x" * MIN_BODY_CHARS
        issue = {"issue_number": 11, "title": "Borderline", "body": body, "labels": []}
        agent._process_issue(issue)

        assert len(llm_called) == 1, "LLM should be called when body meets the minimum"

    def test_normal_issue_calls_llm_and_posts_comment(self):
        """A well-described issue triggers LLM triage and posts the result."""
        mock_logger = MagicMock(spec=StructuredLogger)
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, mock_logger)

        fake_assessment = {"priority": "high", "type": "bug", "reproduction_status": "confirmed"}

        llm_called_with = []
        posted_with = []

        agent._llm_triage = lambda i: (llm_called_with.append(i), fake_assessment)[1]  # type: ignore[method-assign]
        agent._post_comment = lambda *a, **kw: (posted_with.append((a, kw)), True)[1]  # type: ignore[method-assign]

        issue = {
            "issue_number": 42,
            "title": "App crashes on startup",
            "body": "Steps to reproduce: run the app and it crashes immediately on launch.",
            "labels": [],
        }
        agent._process_issue(issue)

        assert len(llm_called_with) == 1
        assert llm_called_with[0]["issue_number"] == 42
        assert len(posted_with) == 1
        args, kwargs = posted_with[0]
        assert args[0] == 42
        assert args[1] == fake_assessment
        assert kwargs.get("note") is None


# ---------------------------------------------------------------------------
# Tests: _llm_triage()
# ---------------------------------------------------------------------------

class TestLlmTriage:

    def _make_agent(self) -> TriageAgent:
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        logger = StructuredLogger(min_level="ERROR")
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, logger)
        return agent

    def test_returns_dict_with_expected_keys(self):
        """_llm_triage() always returns a dict with priority, type, reproduction_status."""
        agent = self._make_agent()
        llm_json = json.dumps(
            {"priority": "high", "type": "bug", "reproduction_status": "confirmed"}
        )
        agent._anthropic.messages.create.return_value = make_llm_response(llm_json)

        result = agent._llm_triage(
            {"issue_number": 1, "title": "Crash", "body": "App crashes on login."}
        )

        assert set(result.keys()) == {"priority", "type", "reproduction_status"}

    def test_parses_llm_json_correctly(self):
        """Values from the LLM JSON are mapped to the returned dict."""
        agent = self._make_agent()
        llm_json = json.dumps(
            {"priority": "critical", "type": "bug", "reproduction_status": "confirmed"}
        )
        agent._anthropic.messages.create.return_value = make_llm_response(llm_json)

        result = agent._llm_triage(
            {"issue_number": 2, "title": "Security", "body": "SQL injection discovered."}
        )

        assert result["priority"] == "critical"
        assert result["type"] == "bug"
        assert result["reproduction_status"] == "confirmed"

    def test_returns_safe_defaults_on_invalid_json(self):
        """When the LLM returns non-JSON text, safe defaults are returned."""
        agent = self._make_agent()
        agent._anthropic.messages.create.return_value = make_llm_response(
            "Sorry, I cannot classify this issue."
        )

        result = agent._llm_triage(
            {"issue_number": 3, "title": "Odd", "body": "Something is wrong."}
        )

        assert result == {
            "priority": "low",
            "type": "question",
            "reproduction_status": "unconfirmed",
        }

    def test_returns_safe_defaults_on_api_exception(self):
        """When the Anthropic API raises, safe defaults are returned (no propagation)."""
        agent = self._make_agent()
        agent._anthropic.messages.create.side_effect = RuntimeError("connection refused")

        result = agent._llm_triage(
            {"issue_number": 4, "title": "Net error", "body": "API call blows up."}
        )

        assert result == {
            "priority": "low",
            "type": "question",
            "reproduction_status": "unconfirmed",
        }

    def test_missing_keys_in_llm_json_fall_back_to_defaults(self):
        """Partial JSON from the LLM uses defaults for missing keys."""
        agent = self._make_agent()
        # Only 'type' is present; the other two keys are absent.
        agent._anthropic.messages.create.return_value = make_llm_response(
            json.dumps({"type": "feature"})
        )

        result = agent._llm_triage(
            {"issue_number": 5, "title": "Feature request", "body": "Please add dark mode."}
        )

        assert result["type"] == "feature"
        assert result["priority"] == "low"
        assert result["reproduction_status"] == "unconfirmed"

    def test_includes_issue_title_in_llm_prompt(self):
        """The issue title is included in the user message sent to the LLM."""
        agent = self._make_agent()
        agent._anthropic.messages.create.return_value = make_llm_response(
            json.dumps({"priority": "low", "type": "question", "reproduction_status": "not_applicable"})
        )

        agent._llm_triage(
            {"issue_number": 6, "title": "My Special Title", "body": "Some body text here."}
        )

        call_kwargs = agent._anthropic.messages.create.call_args
        user_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "My Special Title" in user_content


# ---------------------------------------------------------------------------
# Tests: _post_comment()
# ---------------------------------------------------------------------------

class TestPostComment:

    def _make_agent(self) -> tuple[TriageAgent, MagicMock, MagicMock]:
        mock_github = MagicMock()
        mock_github.repo = "owner/repo"
        mock_logger = MagicMock(spec=StructuredLogger)
        with patch("agent.triage.anthropic.Anthropic"):
            agent = TriageAgent(mock_github, mock_logger)
        return agent, mock_github, mock_logger

    def _good_assessment(self) -> dict:
        return {"priority": "medium", "type": "bug", "reproduction_status": "unconfirmed"}

    def test_returns_true_on_success(self):
        """_post_comment() returns True when both POST requests succeed."""
        agent, mock_github, _ = self._make_agent()
        # Both post calls succeed (raise_for_status is a no-op on MagicMock by default)
        result = agent._post_comment(10, self._good_assessment())
        assert result is True

    def test_posts_comment_to_correct_url(self):
        """Comment is POSTed to /repos/{repo}/issues/{number}/comments."""
        agent, mock_github, _ = self._make_agent()
        agent._post_comment(10, self._good_assessment())

        first_call = mock_github._client.post.call_args_list[0]
        url = first_call.args[0] if first_call.args else first_call.kwargs.get("url")
        assert "owner/repo" in url
        assert "/issues/10/comments" in url

    def test_applies_triage_label_after_comment(self):
        """TRIAGE_LABEL is applied via a second POST to the labels endpoint."""
        agent, mock_github, _ = self._make_agent()
        agent._post_comment(10, self._good_assessment())

        # There should be at least two POST calls: one for comment, one for label.
        assert mock_github._client.post.call_count >= 2

        label_call = mock_github._client.post.call_args_list[1]
        label_url = label_call.args[0] if label_call.args else label_call.kwargs.get("url")
        assert "/issues/10/labels" in label_url
        label_body = label_call.kwargs.get("json", {})
        assert TRIAGE_LABEL in label_body.get("labels", [])

    def test_logs_failure_and_returns_false_on_comment_http_error(self):
        """
        When the comment POST returns a 4xx/5xx response, triage_comment_failed is
        logged with the issue_number and status_code, and the method returns False.
        """
        agent, mock_github, mock_logger = self._make_agent()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = make_http_error(422)
        mock_github._client.post.return_value = mock_response

        result = agent._post_comment(99, self._good_assessment())

        assert result is False
        mock_logger.log.assert_called_once_with(
            "triage_comment_failed",
            "ERROR",
            issue_number=99,
            status_code=422,
        )

    def test_label_not_applied_when_comment_fails(self):
        """When the comment POST fails, no label POST is attempted."""
        agent, mock_github, _ = self._make_agent()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = make_http_error(500)
        mock_github._client.post.return_value = mock_response

        agent._post_comment(99, self._good_assessment())

        # Only one POST attempted (the failed comment); label POST is skipped.
        assert mock_github._client.post.call_count == 1

    def test_note_is_prepended_when_provided(self):
        """When note is given, the comment body starts with **Note**: <note>."""
        agent, mock_github, _ = self._make_agent()
        agent._post_comment(55, self._good_assessment(), note="insufficient_detail")

        comment_call = mock_github._client.post.call_args_list[0]
        body_text = comment_call.kwargs["json"]["body"]
        assert body_text.startswith("**Note**: insufficient_detail")

    def test_no_note_when_not_provided(self):
        """When note is None, the comment body does not contain the Note prefix."""
        agent, mock_github, _ = self._make_agent()
        agent._post_comment(56, self._good_assessment(), note=None)

        comment_call = mock_github._client.post.call_args_list[0]
        body_text = comment_call.kwargs["json"]["body"]
        assert "**Note**" not in body_text

    def test_comment_body_contains_assessment_values(self):
        """The formatted comment includes the priority, type, and reproduction_status."""
        agent, mock_github, _ = self._make_agent()
        assessment = {"priority": "high", "type": "bug", "reproduction_status": "confirmed"}
        agent._post_comment(77, assessment)

        comment_call = mock_github._client.post.call_args_list[0]
        body_text = comment_call.kwargs["json"]["body"]
        assert "high" in body_text
        assert "bug" in body_text
        assert "confirmed" in body_text

    def test_label_failure_does_not_cause_false_return(self):
        """Even if the label POST fails, _post_comment still returns True."""
        agent, mock_github, mock_logger = self._make_agent()

        good_response = MagicMock()
        # First call (comment) succeeds; second call (label) raises HTTPStatusError.
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = make_http_error(403)

        mock_github._client.post.side_effect = [good_response, bad_response]

        result = agent._post_comment(33, self._good_assessment())

        assert result is True
        mock_logger.log.assert_called_once()
        event_name = mock_logger.log.call_args.args[0]
        assert event_name == "triage_label_failed"
