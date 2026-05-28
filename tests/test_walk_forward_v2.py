"""Tests for the strategy-agnostic walk-forward runner.

These tests use synthetic deterministic candles so the assertions are
about *behavioural* properties (no look-ahead, purge gap respected,
warmup feed enabled, deterministic selection) rather than about
specific numeric outputs from the real-data run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.walk_forward_v2 import (
    OBJECTIVE_RETURN_DIV_DRAWDOWN,
    OBJECTIVE_SHARPE,
    OBJECTIVE_TOTAL_RETURN,
    ParamGridSpec,
    aggregate_walk_forward,
    generate_month_windows,
    run_strategy_walk_forward,
)
from trade_lab.strategies.sma_cross import SMACrossStrategy


def _candles(n: int, start: str = "2018-01-01", seed: int = 0):
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    closes = 100 + np.linspace(0, 80, n) + rng.normal(0, 1.5, n)
    closes = np.clip(closes, 1.0, None)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )


def _trivial_grid() -> list:
    """Two SMA variants with different lookbacks — enough for selection."""
    return [
        ParamGridSpec(
            label="sma_10_30",
            factory=lambda: SMACrossStrategy(fast_period=10, slow_period=30),
            warmup_days=30,
        ),
        ParamGridSpec(
            label="sma_20_50",
            factory=lambda: SMACrossStrategy(fast_period=20, slow_period=50),
            warmup_days=50,
        ),
    ]


# ---------------------------------------------------------------------------
# Window generator
# ---------------------------------------------------------------------------


def test_generate_windows_inclusive_day_boundaries():
    candles = _candles(800)
    windows = generate_month_windows(
        candles, train_months=12, test_months=3, step_months=3,
    )
    assert len(windows) > 0
    # First window: 12 months train + 3 months test = ~15 months total
    first = windows[0]
    assert first.train_start == candles.index[0]
    # train_end is one day before train_start + 12 months
    expected_train_end = first.train_start + pd.DateOffset(months=12) - pd.DateOffset(days=1)
    assert first.train_end == expected_train_end
    assert first.test_start == first.train_end + pd.DateOffset(days=1)


def test_purge_days_creates_gap_between_train_and_test():
    candles = _candles(800)
    a = generate_month_windows(
        candles, train_months=12, test_months=3, step_months=3, purge_days=0,
    )
    b = generate_month_windows(
        candles, train_months=12, test_months=3, step_months=3, purge_days=30,
    )
    # First window: same train, test_start shifted by purge_days.
    assert a[0].train_end == b[0].train_end
    assert (b[0].test_start - a[0].test_start) == pd.Timedelta(days=30)


def test_generate_windows_rejects_zero_durations():
    candles = _candles(800)
    with pytest.raises(ValueError):
        generate_month_windows(candles, train_months=0, test_months=3, step_months=3)
    with pytest.raises(ValueError):
        generate_month_windows(candles, train_months=12, test_months=0, step_months=3)


def test_no_windows_when_train_longer_than_history():
    candles = _candles(60)  # 2 months
    windows = generate_month_windows(
        candles, train_months=24, test_months=3, step_months=3,
    )
    assert windows == []


# ---------------------------------------------------------------------------
# Runner: no look-ahead and warmup feeding
# ---------------------------------------------------------------------------


def test_appending_future_garbage_does_not_change_any_fold():
    """Adding garbage candles *after* the last test fold must not
    change any selection or any test metric. This is the strict
    no-lookahead check."""
    # Sized so every WF fold fits cleanly inside base before any
    # truncation against the last candle. Extension is stitched on
    # afterwards so its garbage cannot leak backwards.
    base = _candles(1095, seed=1)  # 3 calendar years
    next_day = (base.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    extended = pd.concat([base, _candles(180, start=next_day, seed=999)])

    grid = _trivial_grid()
    res_base = run_strategy_walk_forward(
        base, grid, train_months=12, test_months=3, step_months=3,
        objective=OBJECTIVE_SHARPE,
    )
    res_ext = run_strategy_walk_forward(
        extended, grid, train_months=12, test_months=3, step_months=3,
        objective=OBJECTIVE_SHARPE,
    )
    # The last fold of `res_base` may be truncated against base's end
    # while the same fold in `res_ext` is full-length. Drop it and
    # compare the remaining N-1 folds head-to-head — they must agree
    # down to the float.
    common = max(len(res_base) - 1, 0)
    assert common > 0
    for col in ("selected_label", "train_sharpe", "test_sharpe",
                "train_return_pct", "test_return_pct"):
        pd.testing.assert_series_equal(
            res_base[col].iloc[:common].reset_index(drop=True),
            res_ext[col].iloc[:common].reset_index(drop=True),
            check_names=False,
        )


def test_warmup_feed_lets_strategy_produce_signals_on_short_test_window():
    """A 3-month test window is shorter than a 50-day SMA's warmup
    period. Without warmup feeding, the strategy would never trade.
    With warmup-from-before-test enabled, it should make at least
    some trades."""
    candles = _candles(800, seed=2)
    grid = [
        ParamGridSpec(
            label="sma_20_50",
            factory=lambda: SMACrossStrategy(fast_period=20, slow_period=50),
            warmup_days=50,
        ),
    ]
    res = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3,
        objective=OBJECTIVE_SHARPE,
    )
    # In an uptrend, a 20/50 crossover should produce non-flat returns
    # at least on some folds — i.e. test_sharpe != 0 OR test_return_pct != 0.
    active = (res["test_return_pct"].abs() > 1e-9) | (res["test_sharpe"].abs() > 1e-9)
    assert active.any(), "warmup feed appears broken — strategy never traded"


# ---------------------------------------------------------------------------
# Selection: variants and objectives
# ---------------------------------------------------------------------------


def test_selection_deterministic_for_same_input():
    candles = _candles(700, seed=3)
    grid = _trivial_grid()
    res1 = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3,
    )
    res2 = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3,
    )
    pd.testing.assert_frame_equal(res1, res2)


def test_objective_sharpe_picks_highest_sharpe_variant_when_unique():
    """If one variant clearly dominates Sharpe on train, it must be
    picked. We engineer a one-element grid + a deterministic SMA to
    pin the selection."""
    candles = _candles(700, seed=4)
    grid = [
        ParamGridSpec(
            label="sma_5_15",
            factory=lambda: SMACrossStrategy(fast_period=5, slow_period=15),
            warmup_days=15,
        ),
    ]
    res = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3,
        objective=OBJECTIVE_SHARPE,
    )
    assert (res["selected_label"] == "sma_5_15").all()


def test_invalid_objective_raises():
    candles = _candles(400)
    with pytest.raises(ValueError):
        run_strategy_walk_forward(
            candles, _trivial_grid(),
            train_months=6, test_months=3, step_months=3,
            objective="nonsense",
        )


def test_empty_grid_returns_empty_frame():
    candles = _candles(400)
    res = run_strategy_walk_forward(
        candles, [], train_months=6, test_months=3, step_months=3,
    )
    assert res.empty


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_summary_is_consistent_with_detail():
    candles = _candles(700, seed=5)
    res = run_strategy_walk_forward(
        candles, _trivial_grid(),
        train_months=12, test_months=3, step_months=3,
        objective=OBJECTIVE_SHARPE,
    )
    summary = aggregate_walk_forward(res)
    assert summary["n_folds"] == len(res)
    assert summary["mean_test_return"] == pytest.approx(
        float(res["test_return_pct"].mean())
    )
    assert summary["hit_rate"] == pytest.approx(
        float((res["test_return_pct"] > 0).mean())
    )


def test_aggregate_summary_handles_empty_frame():
    summary = aggregate_walk_forward(pd.DataFrame())
    assert summary["n_folds"] == 0
    assert summary["mean_test_sharpe"] == 0.0


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def test_purge_days_does_not_change_train_metrics():
    """Train ends on the same date with or without purge — the gap is
    inserted between train_end and test_start, so train metrics must
    match exactly."""
    candles = _candles(800, seed=6)
    grid = _trivial_grid()
    res_no_purge = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3, purge_days=0,
    )
    res_purge = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3, purge_days=14,
    )
    common = min(len(res_no_purge), len(res_purge))
    assert common > 0
    pd.testing.assert_series_equal(
        res_no_purge["train_sharpe"].iloc[:common],
        res_purge["train_sharpe"].iloc[:common],
        check_names=False,
    )
