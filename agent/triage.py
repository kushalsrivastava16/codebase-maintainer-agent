"""
Issue triage agent for the Codebase Maintainer Agent.

Fetches open GitHub issues and posts structured triage assessments as comments.

WHY a separate TriageAgent instead of routing through Orchestrator?
  The Orchestrator is designed for file-level tasks (lint, tests, TODOs) that
  follow a tight read-modify-write loop. Triage is fundamentally different: it
  iterates over GitHub issues, calls the LLM once per issue for classification,
  then writes a comment back to GitHub. There is no local file diff to produce.
  A dedicated TriageAgent keeps that logic cohesive and avoids bloating the
  Orchestrator with GitHub-specific branching.

WHY call the Anthropic client directly instead of going through Orchestrator?
  The triage LLM call is a single-shot classification — one prompt in, one JSON
  blob out. There is no tool loop, no diff writer, no budget guard needed. Calling
  the client directly is simpler and easier to test than setting up a full
  Orchestrator instance just to classify an issue.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import anthropic
import httpx

from agent.logger import StructuredLogger
from agent.protocols import ToolResult

# Label applied to every issue after triage so we don't double-process it.
TRIAGE_LABEL = "agent-triaged"

# Only issues created within this window are re-eligible for triage
# (unused in the current label-based gate, reserved for future use).
TRIAGE_WINDOW_DAYS = 7

# Issues whose body has fewer than this many non-whitespace characters are
# classified as insufficient_detail without calling the LLM.
MIN_BODY_CHARS = 10

TRIAGE_COMMENT_TEMPLATE = """\
## Agent Triage Report

| Field | Value |
|-------|-------|
| **Priority** | {priority} |
| **Type** | {type} |
| **Reproduction Status** | {reproduction_status} |

*Posted by codebase-maintainer-agent*
"""

_GITHUB_API = "https://api.github.com"


class TriageAgent:
    """
    Orchestrates issue-by-issue triage using the GitHub API.

    For each open issue that has not already been labelled with TRIAGE_LABEL,
    the agent calls the Anthropic LLM directly for a single-shot classification
    (priority / type / reproduction_status), then posts a structured Markdown
    comment and applies the TRIAGE_LABEL so the issue is skipped on future runs.
    """

    def __init__(
        self,
        github_client,  # GitHubClient instance — duck-typed to avoid circular import
        logger: StructuredLogger,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._github = github_client
        self._logger = logger
        self._model = model
        self._anthropic = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Fetch all open issues and triage each one in ascending issue-number order.

        Skips issues already bearing the TRIAGE_LABEL so re-runs are idempotent.
        Logs triage_fetch_failed and returns early if the GitHub fetch fails.
        """
        result = self._github.execute({"operation": "fetch_issues"})
        if result.is_error:
            self._logger.log("triage_fetch_failed", "ERROR", error=result.content)
            return

        issues: list[dict] = json.loads(result.content)
        issues.sort(key=lambda i: i["issue_number"])

        pending = [i for i in issues if TRIAGE_LABEL not in i.get("labels", [])]
        self._logger.log(
            "triage_issues_found", "INFO",
            total=len(issues),
            pending=len(pending),
            repo=self._github.repo,
        )

        if not pending:
            self._logger.log("triage_no_pending", "INFO",
                             message="All issues already triaged or no open issues found")
            return

        for issue in issues:
            self._process_issue(issue)

    # ------------------------------------------------------------------
    # Per-issue processing
    # ------------------------------------------------------------------

    def _process_issue(self, issue: dict) -> None:
        """
        Triage a single issue.

        Gate 1 — already triaged: if the issue already has TRIAGE_LABEL, skip it
          and emit triage_skipped_recent.

        Gate 2 — insufficient detail: if the non-whitespace character count of the
          body is below MIN_BODY_CHARS, post a low-priority comment without calling
          the LLM.

        Otherwise: call _llm_triage() for classification, then post the comment.
        """
        issue_number: int = issue["issue_number"]

        if TRIAGE_LABEL in issue.get("labels", []):
            self._logger.log("triage_skipped_recent", "INFO", issue_number=issue_number)
            return

        body: str = issue.get("body", "") or ""
        non_ws_count = sum(1 for ch in body if not ch.isspace())

        if non_ws_count < MIN_BODY_CHARS:
            self._post_comment(
                issue_number,
                {
                    "priority": "low",
                    "type": "question",
                    "reproduction_status": "not_applicable",
                },
                note="insufficient_detail",
            )
            return

        assessment = self._llm_triage(issue)
        self._post_comment(issue_number, assessment)
        self._logger.log(
            "triage_issue_done", "INFO",
            issue_number=issue_number,
            title=issue.get("title", "")[:80],
            priority=assessment["priority"],
            type=assessment["type"],
        )

    # ------------------------------------------------------------------
    # LLM classification
    # ------------------------------------------------------------------

    def _llm_triage(self, issue: dict) -> dict:
        """
        Ask the LLM to classify the issue and return a triage dict.

        The LLM is instructed to respond with a JSON object containing exactly:
          priority             — critical | high | medium | low
          type                 — bug | feature | question | documentation
          reproduction_status  — confirmed | unconfirmed | not_applicable

        On any parse or API error the method returns safe low-priority defaults
        rather than propagating the exception, so the agent loop can continue
        with the remaining issues.
        """
        system_prompt = (
            "You are a triage assistant for a software project. "
            "Analyse the GitHub issue and return a JSON object with these exact fields:\n"
            '  "priority": one of "critical", "high", "medium", "low"\n'
            '  "type": one of "bug", "feature", "question", "documentation"\n'
            '  "reproduction_status": one of "confirmed", "unconfirmed", "not_applicable"\n'
            "Respond with only valid JSON — no markdown fences, no extra text."
        )
        user_message = (
            f"Issue title: {issue.get('title', '')}\n\n"
            f"Issue body:\n{issue.get('body', '')}"
        )

        try:
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text: str = response.content[0].text
            data: dict = json.loads(text)
            return {
                "priority": data.get("priority", "low"),
                "type": data.get("type", "question"),
                "reproduction_status": data.get("reproduction_status", "unconfirmed"),
            }
        except Exception:  # noqa: BLE001
            # Return safe defaults so the caller can still post a comment.
            return {
                "priority": "low",
                "type": "question",
                "reproduction_status": "unconfirmed",
            }

    # ------------------------------------------------------------------
    # GitHub comment + label
    # ------------------------------------------------------------------

    def _post_comment(
        self,
        issue_number: int,
        assessment: dict,
        note: str | None = None,
    ) -> bool:
        """
        Post a triage comment on the issue and apply TRIAGE_LABEL.

        Uses the underlying httpx.Client stored on the GitHubClient (_client)
        so we can send arbitrary POST requests that the GitHubClient.execute()
        dispatch table does not expose.

        Returns True on success.  On HTTP error when posting the comment, logs
        triage_comment_failed (with status code) and returns False without
        attempting to apply the label.  Label failures are logged as warnings
        but do not cause the method to return False — the comment is already
        posted and is the primary deliverable.
        """
        comment_text = TRIAGE_COMMENT_TEMPLATE.format(
            priority=assessment["priority"],
            type=assessment["type"],
            reproduction_status=assessment["reproduction_status"],
        )
        if note:
            comment_text = f"**Note**: {note}\n\n" + comment_text

        base = f"{_GITHUB_API}/repos/{self._github.repo}"
        comment_url = f"{base}/issues/{issue_number}/comments"
        labels_url = f"{base}/issues/{issue_number}/labels"

        # --- Post the comment ---
        try:
            comment_resp = self._github._client.post(
                comment_url, json={"body": comment_text}
            )
            comment_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._logger.log(
                "triage_comment_failed",
                "ERROR",
                issue_number=issue_number,
                status_code=exc.response.status_code,
            )
            return False

        # --- Apply the triage label ---
        try:
            label_resp = self._github._client.post(
                labels_url, json={"labels": [TRIAGE_LABEL]}
            )
            label_resp.raise_for_status()
        except httpx.HTTPStatusError:
            self._logger.log(
                "triage_label_failed",
                "WARNING",
                issue_number=issue_number,
            )

        return True
