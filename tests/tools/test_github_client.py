"""
Tests for agent/tools/github_client.py

Covers:
  1. Missing GITHUB_TOKEN raises RuntimeError("github_auth_missing")
  2. fetch_issues returns structured data (issue_number, title, body, labels, comment_count)
  3. _request retries once on 429 with the Retry-After header value
  4. _request returns None on a second consecutive 429 (no infinite loop)
  5. _unique_branch_name appends -1, -2, etc. when the branch already exists
  6. execute() dispatches to the correct operation handler
  7. execute() returns is_error=True for an unknown operation
  8. create_branch returns the actual branch name used
  9. open_pr returns the PR html_url
  10. commit_diff chains the Git Data API calls in the correct order
  11. Tool protocol compliance (name, description, input_schema, execute callable)
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, call, patch

import pytest

from agent.tools.github_client import GitHubClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO = "owner/testrepo"


def _make_response(status_code: int = 200, json_body=None, headers=None) -> MagicMock:
    """Build a minimal mock that looks like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_body or {}
    # raise_for_status() does nothing for 2xx; raise for 4xx/5xx
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(
            f"HTTP {status_code}"
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_client(token: str = "test-token") -> GitHubClient:
    """Create a GitHubClient with a patched env token and a mocked httpx.Client."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": token}):
        client = GitHubClient(_REPO)
    # Replace the real httpx.Client with a MagicMock so no network calls occur
    client._client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# 1. Missing GITHUB_TOKEN raises RuntimeError
# ---------------------------------------------------------------------------

class TestMissingToken:
    def test_raises_runtime_error_when_token_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GITHUB_TOKEN is definitely absent
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(RuntimeError, match="github_auth_missing"):
                GitHubClient(_REPO)

    def test_raises_runtime_error_when_token_empty_string(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            with pytest.raises(RuntimeError, match="github_auth_missing"):
                GitHubClient(_REPO)


# ---------------------------------------------------------------------------
# 2. fetch_issues returns structured data
# ---------------------------------------------------------------------------

class TestFetchIssues:
    def _raw_issue(self, number: int, title: str, body: str, labels: list, comments: int):
        return {
            "number": number,
            "title": title,
            "body": body,
            "labels": [{"name": lbl} for lbl in labels],
            "comments": comments,
        }

    def test_returns_structured_issue_list(self):
        client = _make_client()
        raw = [
            self._raw_issue(42, "Fix bug", "description here", ["bug", "help wanted"], 3),
            self._raw_issue(7, "Add feature", None, [], 0),
        ]
        client._client.request.return_value = _make_response(200, json_body=raw)

        result = client.fetch_issues({})

        assert result.is_error is False
        issues = json.loads(result.content)
        assert len(issues) == 2

        first = issues[0]
        assert first["issue_number"] == 42
        assert first["title"] == "Fix bug"
        assert first["body"] == "description here"
        assert first["labels"] == ["bug", "help wanted"]
        assert first["comment_count"] == 3

    def test_null_body_becomes_empty_string(self):
        client = _make_client()
        raw = [self._raw_issue(1, "No body", None, [], 0)]
        client._client.request.return_value = _make_response(200, json_body=raw)

        result = client.fetch_issues({})

        issues = json.loads(result.content)
        assert issues[0]["body"] == ""

    def test_empty_issue_list_returns_empty_json_array(self):
        client = _make_client()
        client._client.request.return_value = _make_response(200, json_body=[])

        result = client.fetch_issues({})

        assert result.is_error is False
        assert json.loads(result.content) == []

    def test_rate_limit_abort_returns_error(self):
        client = _make_client()
        # Two consecutive 429s → _request returns None
        r429 = _make_response(429, headers={"Retry-After": "1"})
        client._client.request.return_value = r429

        with patch("time.sleep"):
            result = client.fetch_issues({})

        assert result.is_error is True
        assert "rate_limit_abort" in result.content


# ---------------------------------------------------------------------------
# 3. _request retries on 429 with Retry-After header
# ---------------------------------------------------------------------------

class TestRequestRetryOn429:
    def test_sleeps_for_retry_after_seconds(self):
        client = _make_client()
        first_429 = _make_response(429, headers={"Retry-After": "30"})
        success = _make_response(200, json_body={"ok": True})
        client._client.request.side_effect = [first_429, success]

        with patch("time.sleep") as mock_sleep:
            resp = client._request("GET", "https://api.github.com/repos/owner/testrepo")

        mock_sleep.assert_called_once_with(30)
        assert resp is not None
        assert resp.status_code == 200

    def test_uses_default_60s_when_retry_after_missing(self):
        client = _make_client()
        first_429 = _make_response(429, headers={})
        success = _make_response(200, json_body={})
        client._client.request.side_effect = [first_429, success]

        with patch("time.sleep") as mock_sleep:
            client._request("GET", "https://api.github.com/repos/owner/testrepo")

        mock_sleep.assert_called_once_with(60)

    def test_makes_exactly_two_requests_on_first_429(self):
        client = _make_client()
        first_429 = _make_response(429, headers={"Retry-After": "1"})
        success = _make_response(200, json_body={})
        client._client.request.side_effect = [first_429, success]

        with patch("time.sleep"):
            client._request("GET", "https://api.github.com/repos/owner/testrepo")

        assert client._client.request.call_count == 2


# ---------------------------------------------------------------------------
# 4. _request returns None on second 429
# ---------------------------------------------------------------------------

class TestRequestReturnsNoneOnSecond429:
    def test_returns_none_when_both_attempts_are_429(self):
        client = _make_client()
        r429_a = _make_response(429, headers={"Retry-After": "1"})
        r429_b = _make_response(429, headers={"Retry-After": "1"})
        client._client.request.side_effect = [r429_a, r429_b]

        with patch("time.sleep"):
            result = client._request("GET", "https://api.github.com/repos/owner/testrepo")

        assert result is None

    def test_does_not_sleep_a_second_time_on_second_429(self):
        client = _make_client()
        r429_a = _make_response(429, headers={"Retry-After": "5"})
        r429_b = _make_response(429, headers={"Retry-After": "5"})
        client._client.request.side_effect = [r429_a, r429_b]

        with patch("time.sleep") as mock_sleep:
            client._request("GET", "https://api.github.com/repos/owner/testrepo")

        # sleep is called exactly once (after the first 429), not twice
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# 5. _unique_branch_name appends suffix when branch exists
# ---------------------------------------------------------------------------

class TestUniqueBranchName:
    def test_returns_base_name_when_branch_does_not_exist(self):
        client = _make_client()
        client._client.request.return_value = _make_response(404)

        name = client._unique_branch_name("feature-x")

        assert name == "feature-x"

    def test_appends_1_when_base_exists(self):
        client = _make_client()
        # First call (base name) → exists (200); second call (-1) → 404
        client._client.request.side_effect = [
            _make_response(200, json_body={}),
            _make_response(404),
        ]

        name = client._unique_branch_name("feature-x")

        assert name == "feature-x-1"

    def test_appends_2_when_base_and_1_both_exist(self):
        client = _make_client()
        client._client.request.side_effect = [
            _make_response(200, json_body={}),  # feature-x exists
            _make_response(200, json_body={}),  # feature-x-1 exists
            _make_response(404),                # feature-x-2 does not exist
        ]

        name = client._unique_branch_name("feature-x")

        assert name == "feature-x-2"

    def test_calls_correct_ref_url(self):
        client = _make_client()
        client._client.request.return_value = _make_response(404)

        client._unique_branch_name("my-branch")

        expected_url = (
            f"https://api.github.com/repos/{_REPO}/git/refs/heads/my-branch"
        )
        client._client.request.assert_called_with("GET", expected_url)


# ---------------------------------------------------------------------------
# 6. execute() dispatches to the correct handler
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def test_dispatches_fetch_issues(self):
        client = _make_client()
        client._client.request.return_value = _make_response(200, json_body=[])

        result = client.execute({"operation": "fetch_issues"})

        assert result.is_error is False

    def test_dispatches_create_branch(self):
        client = _make_client()
        repo_resp = _make_response(200, json_body={"default_branch": "main"})
        ref_resp = _make_response(200, json_body={"object": {"sha": "abc123"}})
        check_404 = _make_response(404)  # _unique_branch_name: branch free
        create_resp = _make_response(201, json_body={"ref": "refs/heads/my-branch"})
        client._client.request.side_effect = [repo_resp, ref_resp, check_404, create_resp]

        result = client.execute({"operation": "create_branch", "params": {"name": "my-branch"}})

        assert result.is_error is False
        assert result.content == "my-branch"

    def test_dispatches_open_pr(self):
        client = _make_client()
        pr_url = "https://github.com/owner/testrepo/pull/1"
        client._client.request.return_value = _make_response(201, json_body={"html_url": pr_url})

        result = client.execute({
            "operation": "open_pr",
            "params": {"title": "My PR", "branch": "feature-x"},
        })

        assert result.is_error is False
        assert result.content == pr_url


# ---------------------------------------------------------------------------
# 7. execute() returns is_error=True for an unknown operation
# ---------------------------------------------------------------------------

class TestExecuteUnknownOperation:
    def test_unknown_operation_returns_error(self):
        client = _make_client()
        result = client.execute({"operation": "teleport"})

        assert result.is_error is True
        assert "unknown_operation" in result.content
        assert "teleport" in result.content

    def test_missing_operation_key_returns_error(self):
        client = _make_client()
        result = client.execute({})

        assert result.is_error is True


# ---------------------------------------------------------------------------
# 8. create_branch returns the final branch name
# ---------------------------------------------------------------------------

class TestCreateBranch:
    def test_returns_unique_branch_name(self):
        client = _make_client()
        repo_resp = _make_response(200, json_body={"default_branch": "main"})
        ref_resp = _make_response(200, json_body={"object": {"sha": "deadbeef"}})
        check_200 = _make_response(200, json_body={})  # base name taken
        check_404 = _make_response(404)                # -1 suffix free
        create_resp = _make_response(201, json_body={})
        client._client.request.side_effect = [
            repo_resp, ref_resp, check_200, check_404, create_resp,
        ]

        result = client.create_branch({"name": "fix-bug"})

        assert result.is_error is False
        assert result.content == "fix-bug-1"

    def test_missing_name_returns_error(self):
        client = _make_client()
        result = client.create_branch({})

        assert result.is_error is True
        assert "name is required" in result.content


# ---------------------------------------------------------------------------
# 9. open_pr returns the PR html_url
# ---------------------------------------------------------------------------

class TestOpenPr:
    def test_returns_pr_url(self):
        client = _make_client()
        expected_url = "https://github.com/owner/testrepo/pull/99"
        client._client.request.return_value = _make_response(
            201, json_body={"html_url": expected_url}
        )

        result = client.open_pr({"title": "Fix everything", "branch": "fix-all"})

        assert result.is_error is False
        assert result.content == expected_url

    def test_posts_as_draft(self):
        client = _make_client()
        client._client.request.return_value = _make_response(
            201, json_body={"html_url": "https://github.com/owner/testrepo/pull/1"}
        )

        client.open_pr({"title": "Draft PR", "branch": "feature"})

        _, kwargs = client._client.request.call_args
        assert kwargs["json"]["draft"] is True

    def test_uses_default_base_main(self):
        client = _make_client()
        client._client.request.return_value = _make_response(
            201, json_body={"html_url": "https://github.com/owner/testrepo/pull/2"}
        )

        client.open_pr({"title": "PR", "branch": "branch-x"})

        _, kwargs = client._client.request.call_args
        assert kwargs["json"]["base"] == "main"

    def test_missing_title_returns_error(self):
        client = _make_client()
        result = client.open_pr({"branch": "some-branch"})

        assert result.is_error is True
        assert "required" in result.content

    def test_missing_branch_returns_error(self):
        client = _make_client()
        result = client.open_pr({"title": "Title only"})

        assert result.is_error is True


# ---------------------------------------------------------------------------
# 10. commit_diff chains Git Data API calls in the correct order
# ---------------------------------------------------------------------------

class TestCommitDiff:
    def _setup_commit_diff_mocks(self, client: GitHubClient) -> None:
        """Wire up the six sequential responses that commit_diff expects."""
        client._client.request.side_effect = [
            _make_response(200, json_body={"object": {"sha": "head-sha"}}),   # get ref
            _make_response(200, json_body={"tree": {"sha": "base-tree-sha"}}), # get commit
            _make_response(201, json_body={"sha": "blob-sha"}),                # create blob
            _make_response(201, json_body={"sha": "new-tree-sha"}),            # create tree
            _make_response(201, json_body={"sha": "new-commit-sha"}),          # create commit
            _make_response(200, json_body={"ref": "refs/heads/my-branch"}),    # update ref
        ]

    def test_returns_new_commit_sha(self):
        client = _make_client()
        self._setup_commit_diff_mocks(client)

        result = client.commit_diff({
            "branch": "my-branch",
            "path": "src/foo.py",
            "content": "x = 1\n",
            "message": "fix: update foo",
        })

        assert result.is_error is False
        assert result.content == "new-commit-sha"

    def test_makes_six_api_calls(self):
        client = _make_client()
        self._setup_commit_diff_mocks(client)

        client.commit_diff({
            "branch": "my-branch",
            "path": "README.md",
            "content": "hello",
            "message": "docs: update readme",
        })

        assert client._client.request.call_count == 6

    def test_missing_branch_returns_error(self):
        client = _make_client()
        result = client.commit_diff({"path": "src/foo.py", "content": "x=1"})

        assert result.is_error is True
        assert "required" in result.content

    def test_missing_path_returns_error(self):
        client = _make_client()
        result = client.commit_diff({"branch": "main", "content": "x=1"})

        assert result.is_error is True


# ---------------------------------------------------------------------------
# 11. Tool protocol compliance
# ---------------------------------------------------------------------------

class TestToolProtocol:
    def test_has_name_attribute(self):
        assert GitHubClient.name == "github"

    def test_has_description_attribute(self):
        assert isinstance(GitHubClient.description, str)
        assert len(GitHubClient.description) > 0

    def test_has_input_schema(self):
        schema = GitHubClient.input_schema
        assert schema["type"] == "object"
        assert "operation" in schema["properties"]
        assert "operation" in schema["required"]

    def test_input_schema_enumerates_operations(self):
        ops = GitHubClient.input_schema["properties"]["operation"]["enum"]
        assert "fetch_issues" in ops
        assert "create_branch" in ops
        assert "commit_diff" in ops
        assert "open_pr" in ops

    def test_execute_is_callable(self):
        client = _make_client()
        assert callable(client.execute)

    def test_execute_never_raises(self):
        """Even if the underlying mock throws, execute() must return ToolResult."""
        client = _make_client()
        client._client.request.side_effect = RuntimeError("network down")

        result = client.execute({"operation": "fetch_issues"})

        assert result.is_error is True
