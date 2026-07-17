"""
GitHub Client Tool for the Codebase Maintainer Agent.

WHY httpx instead of PyGithub?
  httpx gives us explicit control over request headers, retry logic, and the
  raw JSON payloads we pass to the Git Data API (blobs, trees, commits, refs).
  PyGithub's abstraction layer hides those details and makes it harder to
  reason about exactly what is being sent — especially important when using
  the low-level Git Data API for creating commits without a working tree.

WHY raise RuntimeError("github_auth_missing") in __init__?
  Failing fast at construction time makes misconfiguration visible immediately
  (e.g., during agent startup) rather than on the first API call deep inside
  a task run. This prevents subtle failures where the agent appears to start
  normally but then silently fails on every GitHub operation.

WHY retry once on 429?
  GitHub's rate-limit response includes a Retry-After header with the number
  of seconds to wait. A single retry covers transient rate-limit spikes (e.g.,
  burst window resets) without looping indefinitely. On a second 429 we abort
  and return an error ToolResult so the orchestrator can surface it to the LLM.

WHY never force-push?
  Force-pushing rewrites history and can silently destroy teammates' work.
  The agent only creates new commits on dedicated branches, so there is never
  a legitimate reason to force-push.
"""
from __future__ import annotations

import json
import os
import time

import httpx

from agent.protocols import ToolResult

_GITHUB_API = "https://api.github.com"


class GitHubClient:
    """
    Wraps the GitHub REST API v3 for the four operations needed by the agent:
    fetching issues, creating branches, committing file changes, and opening PRs.

    All public methods (except __init__) satisfy the Tool.execute() contract:
    they never raise and always return a ToolResult.
    """

    name = "github"
    description = (
        "Interact with GitHub: fetch issues, create branches, commit diffs, open draft PRs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["fetch_issues", "create_branch", "commit_diff", "open_pr"],
            },
            "params": {"type": "object"},
        },
        "required": ["operation"],
    }

    def __init__(self, repo: str) -> None:
        """
        Initialise the client for a given repo (format: "owner/repo").

        Reads GITHUB_TOKEN from the environment. Raises RuntimeError immediately
        if the token is absent so misconfiguration is caught at startup.
        """
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("github_auth_missing")

        self.repo = repo
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Tool protocol: entry point
    # ------------------------------------------------------------------

    def execute(self, inputs: dict) -> ToolResult:
        """
        Dispatch to the appropriate operation method.

        Catches unexpected exceptions so the agent loop is never killed by
        an unhandled error in this tool.
        """
        operation = inputs.get("operation", "")
        params = inputs.get("params") or {}

        dispatch = {
            "fetch_issues": self.fetch_issues,
            "create_branch": self.create_branch,
            "commit_diff": self.commit_diff,
            "open_pr": self.open_pr,
        }

        handler = dispatch.get(operation)
        if handler is None:
            return ToolResult(
                is_error=True,
                content=f"unknown_operation: {operation!r}",
            )

        try:
            return handler(params)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(is_error=True, content=f"github_tool_error: {exc}")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def fetch_issues(self, params: dict) -> ToolResult:  # noqa: ARG002
        """
        Fetch open issues from the repository (up to 100 per page).

        Returns a JSON list of objects with: issue_number, title, body,
        labels (list of name strings), comment_count.
        """
        url = f"{_GITHUB_API}/repos/{self.repo}/issues"
        response = self._request("GET", url, params={"state": "open", "per_page": 100})
        if response is None:
            return ToolResult(is_error=True, content="rate_limit_abort: fetch_issues")

        issues = response.json()
        simplified = [
            {
                "issue_number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body") or "",
                "labels": [lbl["name"] for lbl in issue.get("labels", [])],
                "comment_count": issue.get("comments", 0),
            }
            for issue in issues
        ]
        return ToolResult(is_error=False, content=json.dumps(simplified))

    def create_branch(self, params: dict) -> ToolResult:
        """
        Create a new branch off the repository's default branch.

        params:
          name (str): desired branch name; a suffix (-1, -2, …) is appended
                      automatically if the name is already in use.

        Returns a ToolResult whose content is the actual branch name created.
        """
        base_name = params.get("name", "")
        if not base_name:
            return ToolResult(is_error=True, content="create_branch_error: name is required")

        # 1. Get default branch name
        repo_url = f"{_GITHUB_API}/repos/{self.repo}"
        repo_resp = self._request("GET", repo_url)
        if repo_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: create_branch (repo)")
        default_branch = repo_resp.json()["default_branch"]

        # 2. Get current SHA of the default branch tip
        ref_url = f"{_GITHUB_API}/repos/{self.repo}/git/ref/heads/{default_branch}"
        ref_resp = self._request("GET", ref_url)
        if ref_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: create_branch (ref)")
        sha = ref_resp.json()["object"]["sha"]

        # 3. Find a unique branch name (appends -1, -2, … on collision)
        final_name = self._unique_branch_name(base_name)

        # 4. Create the ref
        create_url = f"{_GITHUB_API}/repos/{self.repo}/git/refs"
        create_resp = self._request(
            "POST",
            create_url,
            json={"ref": f"refs/heads/{final_name}", "sha": sha},
        )
        if create_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: create_branch (create)")

        return ToolResult(is_error=False, content=final_name)

    def commit_diff(self, params: dict) -> ToolResult:
        """
        Commit a single file change to an existing branch using the Git Data API.

        params:
          branch  (str): target branch name
          path    (str): file path relative to the repository root
          content (str): full UTF-8 text content of the file
          message (str): commit message

        Steps: create blob → create tree (referencing parent tree) →
               create commit → update ref (never force-pushes).
        """
        branch = params.get("branch", "")
        path = params.get("path", "")
        content = params.get("content", "")
        message = params.get("message", "chore: automated update")

        if not branch or not path:
            return ToolResult(
                is_error=True,
                content="commit_diff_error: branch and path are required",
            )

        base = f"{_GITHUB_API}/repos/{self.repo}"

        # 1. Get the current commit SHA for the branch
        ref_resp = self._request("GET", f"{base}/git/ref/heads/{branch}")
        if ref_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (ref)")
        head_sha = ref_resp.json()["object"]["sha"]

        # 2. Get the tree SHA from the commit
        commit_resp = self._request("GET", f"{base}/git/commits/{head_sha}")
        if commit_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (commit)")
        base_tree_sha = commit_resp.json()["tree"]["sha"]

        # 3. Create a blob with the new file content
        blob_resp = self._request(
            "POST",
            f"{base}/git/blobs",
            json={"content": content, "encoding": "utf-8"},
        )
        if blob_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (blob)")
        blob_sha = blob_resp.json()["sha"]

        # 4. Create a new tree referencing the parent tree and the new blob
        tree_resp = self._request(
            "POST",
            f"{base}/git/trees",
            json={
                "base_tree": base_tree_sha,
                "tree": [
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                ],
            },
        )
        if tree_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (tree)")
        new_tree_sha = tree_resp.json()["sha"]

        # 5. Create the commit
        new_commit_resp = self._request(
            "POST",
            f"{base}/git/commits",
            json={
                "message": message,
                "tree": new_tree_sha,
                "parents": [head_sha],
            },
        )
        if new_commit_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (new_commit)")
        new_commit_sha = new_commit_resp.json()["sha"]

        # 6. Update the branch ref — never use force (no `force: True`)
        update_resp = self._request(
            "PATCH",
            f"{base}/git/refs/heads/{branch}",
            json={"sha": new_commit_sha},
        )
        if update_resp is None:
            return ToolResult(is_error=True, content="rate_limit_abort: commit_diff (update_ref)")

        return ToolResult(is_error=False, content=new_commit_sha)

    def open_pr(self, params: dict) -> ToolResult:
        """
        Open a draft pull request.

        params:
          title  (str): PR title
          branch (str): head branch
          base   (str, optional): base branch (default "main")
          body   (str, optional): PR description body
        """
        title = params.get("title", "")
        branch = params.get("branch", "")
        base = params.get("base", "main")
        body = params.get("body", "")

        if not title or not branch:
            return ToolResult(
                is_error=True,
                content="open_pr_error: title and branch are required",
            )

        url = f"{_GITHUB_API}/repos/{self.repo}/pulls"
        response = self._request(
            "POST",
            url,
            json={
                "title": title,
                "head": branch,
                "base": base,
                "body": body,
                "draft": True,
            },
        )
        if response is None:
            return ToolResult(is_error=True, content="rate_limit_abort: open_pr")

        pr_url = response.json()["html_url"]
        return ToolResult(is_error=False, content=pr_url)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response | None:
        """
        Make an HTTP request, retrying once on HTTP 429 (rate limited).

        On the first 429: sleeps for Retry-After seconds (default 60) then retries.
        On a second 429: returns None so the caller can log rate_limit_abort and
        return an error ToolResult.

        Raises httpx.HTTPStatusError for all other 4xx/5xx responses.
        """
        response = self._client.request(method, url, **kwargs)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            time.sleep(retry_after)

            # One retry
            response = self._client.request(method, url, **kwargs)
            if response.status_code == 429:
                # Caller should log rate_limit_abort and return an error ToolResult
                return None

        response.raise_for_status()
        return response

    def _unique_branch_name(self, base: str) -> str:
        """
        Return a branch name that does not yet exist in the repository.

        Checks HEAD for the exact name first. If it exists, tries base-1,
        base-2, … until finding an unused name.
        """
        candidate = base
        counter = 0
        while True:
            check_url = f"{_GITHUB_API}/repos/{self.repo}/git/refs/heads/{candidate}"
            resp = self._client.request("GET", check_url)
            if resp.status_code == 404:
                return candidate
            # Branch exists — try the next suffix
            counter += 1
            candidate = f"{base}-{counter}"
