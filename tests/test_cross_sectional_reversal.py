"""Tests for run_cross_sectional_reversal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.cross_sectional import (
    CrossSectionalResult, run_cross_sectional_reversal,
)


def _candles(closes, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


def _universe(n_assets=4, n_bars=300, seed=0):
    rng = np.random.default_rng(seed)
    candles = {}
    for i in range(n_assets):
        path = 100 + 0.2 * np.arange(n_bars) + rng.normal(0, 2.0, n_bars)
        candles[f"A{i}/USDT"] = _candles(path.clip(min=1).tolist())
    return candles


def test_empty_input_returns_empty_result():
    res = run_cross_sectional_reversal({})
    assert isinstance(res, CrossSectionalResult)
    assert res.equity.empty


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        run_cross_sectional_reversal(_universe(), lookback_days=0)
    with pytest.raises(ValueError):
        run_cross_sectional_reversal(_universe(), rebalance_days=0)
    with pytest.raises(ValueError):
        run_cross_sectional_reversal(_universe(), bottom_k=0)
    with pytest.raises(ValueError):
        run_cross_sectional_reversal(_universe(), weighting="oops")
    with pytest.raises(ValueError):
        run_cross_sectional_reversal(_universe(), vol_lookback=1)


def test_reversal_picks_bottom_k():
    """Three assets with distinct day-1 returns. The TWO lowest must
    be the picks at bar 2 (after the engine's one-bar shift)."""
    # bar 0 = 100; bar 1 sets clear ranking; bar 2 holds the picks.
    a_path = [100.0, 99.0, 99.0]    # ret bar 1 = -1.0%  (worst)
    b_path = [100.0, 100.0, 100.0]  # ret bar 1 = 0.0%
    c_path = [100.0, 102.0, 102.0]  # ret bar 1 = +2.0%  (best)
    candles = {
        "A/USDT": _candles(a_path),
        "B/USDT": _candles(b_path),
        "C/USDT": _candles(c_path),
    }
    res = run_cross_sectional_reversal(
        candles, lookback_days=1, rebalance_days=1, bottom_k=2,
        weighting="equal",
    )
    # res.weights is the *shifted* positions panel: row at bar t holds
    # the weights set at the close of bar t-1. So bar 2's weights
    # reflect the ranking computed at the close of bar 1 — A (-1%)
    # and B (0%) are the two losers, C (+2%) is the largest, excluded.
    weights_at_bar2 = res.weights.iloc[2]
    assert weights_at_bar2["A/USDT"] == pytest.approx(0.5)
    assert weights_at_bar2["B/USDT"] == pytest.approx(0.5)
    assert weights_at_bar2["C/USDT"] == pytest.approx(0.0)


def test_no_lookahead_in_reversal_weights():
    rng = np.random.default_rng(0)
    n = 300
    base = {f"A{i}/USDT": _candles(
        (100 + rng.normal(0, 1, n).cumsum()).clip(min=1).tolist()
    ) for i in range(4)}
    next_start = (base["A0/USDT"].index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    extended = {k: pd.concat([df, _candles([1e6, 1e-6] * 10, start=next_start)])
                for k, df in base.items()}
    res_base = run_cross_sectional_reversal(base, lookback_days=1, rebalance_days=1, bottom_k=2)
    res_ext = run_cross_sectional_reversal(extended, lookback_days=1, rebalance_days=1, bottom_k=2)
    common = min(len(res_base.weights), len(res_ext.weights))
    np.testing.assert_array_equal(
        res_base.weights.iloc[:common].to_numpy(),
        res_ext.weights.iloc[:common].to_numpy(),
    )


def test_weights_sum_at_most_one():
    res = run_cross_sectional_reversal(
        _universe(n_assets=5, n_bars=200, seed=1),
        bottom_k=3, weighting="equal",
    )
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (res.weights.to_numpy() >= 0.0).all()


def test_daily_rebalance_produces_high_turnover():
    """Daily rebalance with random returns should produce LOTS of
    turnover — meaningfully more than a weekly rebalance."""
    universe = _universe(n_assets=5, n_bars=300, seed=2)
    res_daily = run_cross_sectional_reversal(
        universe, lookback_days=1, rebalance_days=1, bottom_k=2,
    )
    res_weekly = run_cross_sectional_reversal(
        universe, lookback_days=1, rebalance_days=7, bottom_k=2,
    )
    daily_turnover = res_daily.weights.diff().abs().sum().sum()
    weekly_turnover = res_weekly.weights.diff().abs().sum().sum()
    assert daily_turnover > weekly_turnover


def test_eligibility_excludes_specified_asset():
    universe = _universe(n_assets=4, n_bars=200, seed=3)
    common_idx = list(universe.values())[0].index
    eligibility = pd.DataFrame(True, index=common_idx, columns=list(universe.keys()))
    eligibility["A0/USDT"] = False
    res = run_cross_sectional_reversal(
        universe, lookback_days=1, rebalance_days=1, bottom_k=2,
        eligibility=eligibility,
    )
    # A0 must never be in the basket.
    assert (res.weights["A0/USDT"] == 0.0).all()
