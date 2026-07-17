"""
Tests for agent/cost_tracker.py

Uses a simple FakeUsage stub to avoid a dependency on the Anthropic SDK.
"""
from __future__ import annotations

import pytest
from agent.cost_tracker import (
    CostTracker,
    INPUT_PRICE_PER_TOKEN,
    OUTPUT_PRICE_PER_TOKEN,
)
from agent.protocols import TokenUsage


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class FakeUsage:
    """Minimal stand-in for anthropic.types.Usage."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class BadUsage:
    """Object that is missing the expected token attributes."""
    pass


# ---------------------------------------------------------------------------
# record() — accumulation
# ---------------------------------------------------------------------------

def test_record_accumulates_input_tokens():
    tracker = CostTracker()
    tracker.record(FakeUsage(100, 0))
    tracker.record(FakeUsage(200, 0))
    assert tracker.totals().input_tokens == 300


def test_record_accumulates_output_tokens():
    tracker = CostTracker()
    tracker.record(FakeUsage(0, 50))
    tracker.record(FakeUsage(0, 75))
    assert tracker.totals().output_tokens == 125


def test_record_accumulates_both_independently():
    tracker = CostTracker()
    tracker.record(FakeUsage(100, 50))
    tracker.record(FakeUsage(200, 75))
    totals = tracker.totals()
    assert totals.input_tokens == 300
    assert totals.output_tokens == 125


def test_record_running_totals_are_cumulative_not_replaced():
    """Multiple record() calls must sum, not overwrite."""
    tracker = CostTracker()
    for _ in range(5):
        tracker.record(FakeUsage(10, 5))
    totals = tracker.totals()
    assert totals.input_tokens == 50
    assert totals.output_tokens == 25


# ---------------------------------------------------------------------------
# record() — return value and invalid usage objects
# ---------------------------------------------------------------------------

def test_record_returns_true_on_valid_usage():
    tracker = CostTracker()
    result = tracker.record(FakeUsage(10, 5))
    assert result is True


def test_record_returns_false_when_attributes_missing():
    tracker = CostTracker()
    result = tracker.record(BadUsage())
    assert result is False


def test_record_does_not_update_counters_on_bad_usage():
    tracker = CostTracker()
    tracker.record(BadUsage())
    totals = tracker.totals()
    assert totals.input_tokens == 0
    assert totals.output_tokens == 0


def test_record_returns_false_for_plain_object_missing_attrs():
    tracker = CostTracker()
    assert tracker.record(object()) is False


def test_record_returns_false_for_partial_usage_only_input():
    """Object with only input_tokens (no output_tokens) should be rejected."""

    class PartialUsage:
        input_tokens = 10

    tracker = CostTracker()
    assert tracker.record(PartialUsage()) is False


def test_record_returns_false_for_partial_usage_only_output():
    """Object with only output_tokens (no input_tokens) should be rejected."""

    class PartialUsage:
        output_tokens = 10

    tracker = CostTracker()
    assert tracker.record(PartialUsage()) is False


# ---------------------------------------------------------------------------
# call_count
# ---------------------------------------------------------------------------

def test_call_count_starts_at_zero():
    tracker = CostTracker()
    assert tracker.call_count == 0


def test_call_count_increments_on_successful_record():
    tracker = CostTracker()
    tracker.record(FakeUsage(10, 5))
    assert tracker.call_count == 1


def test_call_count_increments_once_per_call():
    tracker = CostTracker()
    for i in range(7):
        tracker.record(FakeUsage(1, 1))
    assert tracker.call_count == 7


def test_call_count_does_not_increment_on_bad_usage():
    tracker = CostTracker()
    tracker.record(BadUsage())
    assert tracker.call_count == 0


def test_call_count_only_counts_successful_records():
    tracker = CostTracker()
    tracker.record(FakeUsage(10, 5))   # success
    tracker.record(BadUsage())          # failure — should not count
    tracker.record(FakeUsage(20, 10))  # success
    assert tracker.call_count == 2


# ---------------------------------------------------------------------------
# budget_exceeded()
# ---------------------------------------------------------------------------

def test_budget_not_exceeded_when_below_cap():
    tracker = CostTracker(max_tokens=1_000)
    tracker.record(FakeUsage(400, 400))  # total 800 < 1000
    assert tracker.budget_exceeded() is False


def test_budget_exceeded_when_above_cap():
    tracker = CostTracker(max_tokens=1_000)
    tracker.record(FakeUsage(600, 500))  # total 1100 > 1000
    assert tracker.budget_exceeded() is True


def test_budget_not_exceeded_at_exactly_cap():
    """At exactly the cap (not strictly greater) budget should NOT be exceeded."""
    tracker = CostTracker(max_tokens=1_000)
    tracker.record(FakeUsage(500, 500))  # total == 1000, not > 1000
    assert tracker.budget_exceeded() is False


def test_budget_exceeded_one_token_over_cap():
    tracker = CostTracker(max_tokens=1_000)
    tracker.record(FakeUsage(500, 501))  # total 1001 > 1000
    assert tracker.budget_exceeded() is True


def test_budget_not_exceeded_with_zero_max_tokens_unlimited():
    """max_tokens=0 means unlimited; budget_exceeded() must always return False."""
    tracker = CostTracker(max_tokens=0)
    tracker.record(FakeUsage(999_999, 999_999))
    assert tracker.budget_exceeded() is False


def test_budget_starts_not_exceeded_with_no_records():
    tracker = CostTracker(max_tokens=100)
    assert tracker.budget_exceeded() is False


# ---------------------------------------------------------------------------
# totals() — TokenUsage snapshot
# ---------------------------------------------------------------------------

def test_totals_returns_token_usage_instance():
    tracker = CostTracker()
    assert isinstance(tracker.totals(), TokenUsage)


def test_totals_correct_counts():
    tracker = CostTracker()
    tracker.record(FakeUsage(300, 150))
    totals = tracker.totals()
    assert totals.input_tokens == 300
    assert totals.output_tokens == 150


def test_totals_zero_when_no_records():
    tracker = CostTracker()
    totals = tracker.totals()
    assert totals.input_tokens == 0
    assert totals.output_tokens == 0
    assert totals.estimated_usd == 0.0


def test_totals_usd_estimate_formula():
    """USD = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE, rounded to 6dp."""
    tracker = CostTracker()
    tracker.record(FakeUsage(1_000_000, 1_000_000))
    totals = tracker.totals()
    expected = round(
        1_000_000 * INPUT_PRICE_PER_TOKEN + 1_000_000 * OUTPUT_PRICE_PER_TOKEN, 6
    )
    assert totals.estimated_usd == expected


def test_totals_usd_uses_custom_prices():
    custom_input_price = 1.0 / 1_000_000
    custom_output_price = 2.0 / 1_000_000
    tracker = CostTracker(
        input_price=custom_input_price,
        output_price=custom_output_price,
    )
    tracker.record(FakeUsage(500_000, 500_000))
    totals = tracker.totals()
    expected = round(
        500_000 * custom_input_price + 500_000 * custom_output_price, 6
    )
    assert totals.estimated_usd == expected


def test_totals_usd_output_heavier_than_input():
    """Output tokens are priced 5× higher; verify the ratio is reflected."""
    tracker = CostTracker()
    # Use 1M tokens each to make the math straightforward
    tracker.record(FakeUsage(1_000_000, 0))
    input_only_usd = tracker.totals().estimated_usd

    tracker2 = CostTracker()
    tracker2.record(FakeUsage(0, 1_000_000))
    output_only_usd = tracker2.totals().estimated_usd

    # OUTPUT_PRICE_PER_TOKEN / INPUT_PRICE_PER_TOKEN == 4.00 / 0.80 == 5.0
    assert pytest.approx(output_only_usd / input_only_usd, rel=1e-6) == 5.0


def test_totals_is_snapshot_not_live_reference():
    """Calling totals() twice should return independent snapshots."""
    tracker = CostTracker()
    tracker.record(FakeUsage(100, 50))
    snap1 = tracker.totals()
    tracker.record(FakeUsage(100, 50))
    snap2 = tracker.totals()
    assert snap1.input_tokens == 100
    assert snap2.input_tokens == 200
