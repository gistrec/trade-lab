"""Tests for the MVRV-ratio overlay.

Coverage focus is the look-ahead and threshold invariants — see the
module docstring for the asymmetry of interpretation that frames any
verdict from this proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.mvrv_overlay import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    mvrv_target_position,
    run_mvrv_overlay,
)


def _btc(days: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, days)))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 1.0},
        index=idx,
    )


def _mvrv_series(idx, values):
    return pd.Series(values, index=idx, name="mvrv")


# ---------------------------------------------------------------------------
# mvrv_target_position
# ---------------------------------------------------------------------------


def test_target_full_long_at_or_below_low():
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    out = mvrv_target_position(_mvrv_series(idx, [0.5, 0.8, 1.0]),
                                low_threshold=1.0, high_threshold=3.5)
    assert (out == 1.0).all()


def test_target_cash_at_or_above_high():
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    out = mvrv_target_position(_mvrv_series(idx, [3.5, 4.0, 4.7]),
                                low_threshold=1.0, high_threshold=3.5)
    assert (out == 0.0).all()


def test_target_linear_interp_between():
    idx = pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC")
    # midpoint: ratio = (low + high) / 2 → target 0.5
    mid = (DEFAULT_LOW_THRESHOLD + DEFAULT_HIGH_THRESHOLD) / 2.0
    out = mvrv_target_position(_mvrv_series(idx, [mid]),
                                low_threshold=DEFAULT_LOW_THRESHOLD,
                                high_threshold=DEFAULT_HIGH_THRESHOLD)
    assert out.iloc[0] == pytest.approx(0.5)


def test_invalid_thresholds_raise():
    idx = pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="must be"):
        mvrv_target_position(_mvrv_series(idx, [2.0]),
                              low_threshold=3.5, high_threshold=1.0)


# ---------------------------------------------------------------------------
# run_mvrv_overlay
# ---------------------------------------------------------------------------


def test_full_long_when_mvrv_always_low():
    btc = _btc(400)
    mvrv = pd.Series(0.5, index=btc.index, name="mvrv")
    # rebalance_days=1 so realized hits 1.0 the day after the lag-1
    # MVRV becomes available — equity then tracks BTC bar by bar.
    res = run_mvrv_overlay(
        btc, mvrv, fee_rate=0.0, slippage_rate=0.0, rebalance_days=1,
    )
    assert res.realized_position.iloc[-1] == pytest.approx(1.0)
    btc_total = float(btc["close"].iloc[-1] / btc["close"].iloc[2])
    eq_total = float(res.equity.iloc[-1] / res.equity.iloc[2])
    assert eq_total == pytest.approx(btc_total, rel=0.01)


def test_cash_when_mvrv_always_high():
    btc = _btc(400)
    mvrv = pd.Series(5.0, index=btc.index, name="mvrv")
    res = run_mvrv_overlay(btc, mvrv, fee_rate=0.0, slippage_rate=0.0)
    # Always cash → equity flat at initial.
    assert res.realized_position.iloc[-1] == 0.0
    assert res.equity.iloc[-1] == pytest.approx(res.initial_capital)


def test_publication_lag_excludes_current_day_mvrv():
    """Day t uses MVRV from day t - lag_days. If we set MVRV high only
    on the LAST day, that change must not affect position before it."""
    btc = _btc(400)
    mvrv = pd.Series(0.5, index=btc.index, name="mvrv")  # always full-long
    mvrv.iloc[-1] = 5.0  # very high on the last day only
    res = run_mvrv_overlay(
        btc, mvrv, publication_lag_days=1,
        fee_rate=0.0, slippage_rate=0.0,
    )
    # Last day's MVRV is lagged by 1 → still using day -2 (0.5) on the
    # last bar. So realized at the very last bar should still be 1.0,
    # NOT 0.0.
    assert res.realized_position.iloc[-1] == pytest.approx(1.0)


def test_cost_charged_only_on_position_change():
    """If MVRV is constant for the whole window, exactly ONE turnover
    event happens (the first rebalance from 0 → target)."""
    btc = _btc(400)
    mvrv = pd.Series(0.5, index=btc.index, name="mvrv")  # always full-long
    res = run_mvrv_overlay(
        btc, mvrv, fee_rate=0.001, slippage_rate=0.0005,
    )
    # Expected: one cost event at the first rebalance.
    # Capital at that point ~= initial (haven't earned BTC return yet).
    expected_cost_floor = res.initial_capital * 1.0 * (0.001 + 0.0005) * 0.9
    expected_cost_ceil = res.initial_capital * 1.0 * (0.001 + 0.0005) * 1.1
    actual = res.total_fees + res.total_slippage
    assert expected_cost_floor <= actual <= expected_cost_ceil


def test_no_lookahead_changing_future_mvrv_does_not_alter_past():
    """Corrupt MVRV strictly after a rebalance; equity up to and
    including that rebalance must be byte-for-byte identical."""
    btc = _btc(400)
    rng = np.random.default_rng(1)
    mvrv_a = pd.Series(
        rng.uniform(0.5, 3.5, 400), index=btc.index, name="mvrv",
    )
    res_a = run_mvrv_overlay(btc, mvrv_a)

    # Find a rebalance date well before the end.
    pivot = res_a.rebalance_dates[-10]
    pivot_idx = btc.index.get_loc(pivot)
    mvrv_b = mvrv_a.copy()
    mvrv_b.iloc[pivot_idx + 2:] = 99.0  # corrupt future (past lag window)
    res_b = run_mvrv_overlay(btc, mvrv_b)

    pd.testing.assert_series_equal(
        res_a.equity.iloc[:pivot_idx + 1],
        res_b.equity.iloc[:pivot_idx + 1],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Empty / degenerate
# ---------------------------------------------------------------------------


def test_empty_intersection_returns_empty_result():
    btc = _btc(50)
    # MVRV with non-overlapping index.
    other_idx = pd.date_range("2000-01-01", periods=10, freq="D", tz="UTC")
    mvrv = pd.Series(2.0, index=other_idx, name="mvrv")
    res = run_mvrv_overlay(btc, mvrv)
    assert res.equity.empty
    assert res.num_rebalances == 0


def test_missing_close_column_raises():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="close"):
        run_mvrv_overlay(df, pd.Series([1, 2, 3]))
