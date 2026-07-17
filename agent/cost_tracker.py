"""
Cost Tracker for the Codebase Maintainer Agent.

WHY track costs per task?
  Agent loops can silently consume large amounts of tokens if the LLM gets
  stuck in a correction loop or hallucinates many tool calls. Without a budget
  cap, a single misconfigured run could cost hundreds of dollars. The budget
  guard is the last line of defence after the max_iterations abort guard.

WHY hardcode pricing constants?
  Token prices for claude-haiku-4-5-20251001 are stable and publicly documented. Hardcoding
  them makes the cost calculation explicit and auditable, rather than hiding it
  in a config file where it could be accidentally set to zero. Users who need
  a different model's pricing can update the constants or pass them at
  construction time.

WHY not raise when budget is exceeded?
  budget_exceeded() is a query, not an assertion. The orchestrator checks it
  after each LLM call and decides what to do â€” log an event, write a partial
  result, exit with code 5. Keeping the tracker pure (no side effects) makes
  it easier to test and reason about.
"""
from __future__ import annotations

from agent.protocols import TokenUsage

# claude-haiku-4-5-20251001 pricing as of 2024-Q4 (USD per token)
# Source: https://www.anthropic.com/pricing
# Update these constants if you switch to a different model.
INPUT_PRICE_PER_TOKEN: float = 0.80 / 1_000_000   # $0.80 per million input tokens
OUTPUT_PRICE_PER_TOKEN: float = 4.00 / 1_000_000  # $4.00 per million output tokens


class CostTracker:
    """
    Maintains running token totals and enforces a configurable per-task budget.

    Usage:
        tracker = CostTracker(max_tokens=50_000)
        # after each LLM response:
        tracker.record(response.usage)   # response.usage has input_tokens / output_tokens
        if tracker.budget_exceeded():
            raise SystemExit(5)
        totals = tracker.totals()        # TokenUsage with final counts + USD estimate
    """

    def __init__(
        self,
        max_tokens: int = 50_000,
        input_price: float = INPUT_PRICE_PER_TOKEN,
        output_price: float = OUTPUT_PRICE_PER_TOKEN,
    ) -> None:
        """
        max_tokens: combined (input + output) token cap.
                    Set to 0 to disable budget enforcement.
        input_price / output_price: USD per token.  Override for non-Haiku models.
        """
        self._max_tokens = max_tokens
        self._input_price = input_price
        self._output_price = output_price

        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._call_count: int = 0

    def record(self, usage: object) -> bool:
        """
        Add token counts from an LLM response's usage object.

        Accepts any object with .input_tokens and .output_tokens attributes
        (e.g. Anthropic's Usage object, or a simple dataclass in tests).

        Returns True if the token counts were recorded, False if the usage
        object did not have the expected attributes (caller should log
        'token_usage_unavailable').

        WHY return a bool instead of raising?
          If the Anthropic SDK changes its response format, we want the agent
          to keep running with degraded budget enforcement rather than crashing.
          The bool return lets the orchestrator decide how to handle it.
        """
        if not (hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens")):
            return False
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._call_count += 1
        return True

    def budget_exceeded(self) -> bool:
        """
        Return True if cumulative (input + output) tokens exceed the budget.

        WHY check AFTER recording, not before?
          We always let the current LLM call complete before checking the budget.
          Cutting off mid-response would give us a partial answer that's useless.
          The next call is where we abort.

        WHY return False when max_tokens == 0?
          Zero means "unlimited" by convention â€” matches agent_config.yaml docs.
        """
        if self._max_tokens == 0:
            return False
        return (self._input_tokens + self._output_tokens) > self._max_tokens

    def totals(self) -> TokenUsage:
        """
        Return an immutable snapshot of current token usage with USD estimate.
        """
        usd = round(
            self._input_tokens * self._input_price
            + self._output_tokens * self._output_price,
            6,  # 6 decimal places â†’ precision to fractions of a cent
        )
        return TokenUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            estimated_usd=usd,
        )

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded so far."""
        return self._call_count
