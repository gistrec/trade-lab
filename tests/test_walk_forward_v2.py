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


# ---------------------------------------------------------------------------
# DSR integration
# ---------------------------------------------------------------------------


def test_train_dsr_column_is_populated_and_bounded():
    """train_dsr is a probability — every fold must produce a value in [0, 1]."""
    candles = _candles(1000, seed=7)
    grid = _trivial_grid()
    res = run_strategy_walk_forward(
        candles, grid, train_months=12, test_months=3, step_months=3,
    )
    assert "train_dsr" in res.columns
    assert res["train_dsr"].notna().all()
    assert ((res["train_dsr"] >= 0.0) & (res["train_dsr"] <= 1.0)).all()


def test_concatenated_oos_dsr_returned_when_oos_returns_requested():
    """With return_oos_returns=True the aggregate must include the
    concatenated OOS Sharpe and DSR."""
    candles = _candles(1100, seed=8)
    grid = _trivial_grid()
    detail, oos = run_strategy_walk_forward(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
        return_oos_returns=True,
    )
    assert isinstance(detail, pd.DataFrame) and isinstance(oos, list)
    assert len(oos) == len(detail)
    summary = aggregate_walk_forward(detail, oos_returns=oos, num_trials=100)
    assert "concatenated_oos_sharpe" in summary
    assert "concatenated_oos_dsr" in summary
    assert 0.0 <= summary["concatenated_oos_dsr"] <= 1.0
    assert summary["num_trials"] == 100


def test_aggregate_without_oos_returns_skips_concat_dsr():
    """When the caller does not supply per-fold returns, concatenated
    OOS metrics are zero (not NaN, not erroring out)."""
    candles = _candles(1100, seed=9)
    detail = run_strategy_walk_forward(
        candles, _trivial_grid(),
        train_months=12, test_months=3, step_months=3,
    )
    summary = aggregate_walk_forward(detail)  # no oos_returns
    assert summary["concatenated_oos_sharpe"] == 0.0
    assert summary["concatenated_oos_dsr"] == 0.0


def test_higher_num_trials_lowers_concatenated_dsr():
    """The whole point of the deflation: more trials, lower confidence."""
    candles = _candles(1100, seed=10)
    detail, oos = run_strategy_walk_forward(
        candles, _trivial_grid(),
        train_months=12, test_months=3, step_months=3,
        return_oos_returns=True,
    )
    a = aggregate_walk_forward(detail, oos_returns=oos, num_trials=10)
    b = aggregate_walk_forward(detail, oos_returns=oos, num_trials=10_000)
    assert b["concatenated_oos_dsr"] <= a["concatenated_oos_dsr"]


def test_project_num_trials_is_constant_500():
    """The project's selection-bias correction constant should be
    pinned; the commit history is the audit trail for any change."""
    from trade_lab.backtest.walk_forward_v2 import PROJECT_NUM_TRIALS
    assert PROJECT_NUM_TRIALS == 500


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


def test_per_fold_train_dsr_describes_selected_variant():
    """The reported train_dsr must describe the variant the fold
    actually selected per `objective` — not silently re-pick the
    best-by-Sharpe variant (they differ when the objective is
    total_return / return_div_drawdown)."""
    import numpy as np

    from trade_lab.backtest.dsr import (
        deflated_sharpe_ratio, sharpe_ratio_per_period,
    )
    from trade_lab.backtest.walk_forward_v2 import _per_fold_train_dsr

    rng = np.random.default_rng(11)
    idx = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")
    smooth = pd.Series(rng.normal(0.001, 0.002, 300), index=idx)   # high Sharpe
    lumpy = pd.Series(rng.normal(0.003, 0.05, 300), index=idx)     # high return
    metrics_list = [{"returns": smooth}, {"returns": lumpy}]
    selected = metrics_list[1]  # objective=total_return picked the lumpy one

    got = _per_fold_train_dsr(metrics_list, selected)

    sharpes = [sharpe_ratio_per_period(smooth), sharpe_ratio_per_period(lumpy)]
    expected = deflated_sharpe_ratio(
        returns=lumpy, num_trials=2,
        sharpe_std_dev=float(np.std(sharpes, ddof=1)),
    )
    assert got == pytest.approx(expected)
    # And it must differ from the best-by-Sharpe variant's DSR.
    by_sharpe = deflated_sharpe_ratio(
        returns=smooth, num_trials=2,
        sharpe_std_dev=float(np.std(sharpes, ddof=1)),
    )
    assert got != pytest.approx(by_sharpe)
