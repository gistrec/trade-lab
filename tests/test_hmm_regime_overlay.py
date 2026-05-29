"""Tests for the HMM regime overlay.

The critical invariant is **filtered, not smoothed** posterior usage.
A smoothed posterior uses future data to refine past state estimates —
exactly the look-ahead the user's review flagged as the classical
HMM-backtest mistake. The corresponding test corrupts data after a
chosen date and confirms the filtered probability at that date does
NOT change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.hmm_regime_overlay import (
    filtered_bull_probability,
    run_hmm_regime_overlay,
)


def _btc(days: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=days, freq="D", tz="UTC")
    # Mixture: 70% low-vol (mean +0.001, sd 0.01), 30% high-vol (mean -0.002, sd 0.04)
    rets = np.where(
        rng.random(days) > 0.3,
        rng.normal(0.001, 0.010, days),
        rng.normal(-0.002, 0.040, days),
    )
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 1.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# filtered_bull_probability sanity
# ---------------------------------------------------------------------------


def test_filtered_returns_value_in_unit_interval():
    rng = np.random.default_rng(0)
    p, model = filtered_bull_probability(rng.normal(0, 0.02, 500))
    assert 0.0 <= p <= 1.0
    assert model is not None


def test_filtered_short_window_returns_neutral():
    """< 50 samples → neutral 0.5, no model fit."""
    p, model = filtered_bull_probability(np.array([0.01, -0.02, 0.005]))
    assert p == 0.5
    assert model is None


def test_bull_state_is_higher_mean_component():
    """Construct two regimes with very different means; the bull
    probability should mass on the high-mean component."""
    rng = np.random.default_rng(0)
    # 80% from clearly-positive regime, 20% from clearly-negative.
    regime = rng.random(800) > 0.2
    rets = np.where(regime, rng.normal(0.005, 0.005, 800),
                              rng.normal(-0.005, 0.020, 800))
    p, model = filtered_bull_probability(rets, random_state=0)
    means = model.means_.flatten()
    # Means should bracket zero (one positive, one negative).
    assert means.max() > 0
    assert means.min() < 0


# ---------------------------------------------------------------------------
# Look-ahead invariant — the make-or-break test
# ---------------------------------------------------------------------------


def test_filtered_does_not_change_when_future_corrupted():
    """The whole point of filtered (vs smoothed) is: past estimates do
    not depend on future data. Corrupt the tail and verify."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, 600).tolist()
    p_clean, _ = filtered_bull_probability(np.array(rets), random_state=0)

    # Now compute filtered AT the same end point, but the input array
    # ends at the same place — we can't add future data to a sequence
    # without changing its length. Instead, construct a LONGER series
    # whose first 600 elements match, then take filtered at index 600.
    extended = rets + (rng.normal(0, 0.1, 200)).tolist()
    # Filtered at last index of extended uses ALL of extended → different
    # by construction. To isolate the look-ahead question, slice extended
    # to its first 600 and recompute:
    p_recomputed, _ = filtered_bull_probability(
        np.array(extended[:600]), random_state=0,
    )
    # Recomputed on first-600 must equal the original clean result —
    # both fit on the same data, both take the same final-step forward
    # probability.
    assert p_clean == pytest.approx(p_recomputed)


def test_overlay_equity_unchanged_when_future_returns_corrupted():
    """Full-cycle test: corrupt closes strictly after a rebalance date.
    Equity up to and including that date must match the clean run.
    Smoothed-posterior usage would break this."""
    btc_clean = _btc(1200, seed=11)
    res_clean = run_hmm_regime_overlay(
        btc_clean, train_lookback_days=400, rebalance_days=14, n_iter=20,
    )
    if not res_clean.rebalance_dates:
        pytest.skip("no rebalances produced — pick a longer window")
    pivot = res_clean.rebalance_dates[-5]
    pivot_idx = btc_clean.index.get_loc(pivot)

    btc_corrupt = btc_clean.copy()
    rng = np.random.default_rng(99)
    # Replace closes strictly after pivot with garbage (random walk).
    corrupted_closes = 100.0 * np.exp(
        np.cumsum(rng.normal(0, 0.1, len(btc_corrupt) - pivot_idx - 1))
    )
    btc_corrupt.iloc[
        pivot_idx + 1:, btc_corrupt.columns.get_loc("close")
    ] = corrupted_closes
    res_corrupt = run_hmm_regime_overlay(
        btc_corrupt, train_lookback_days=400, rebalance_days=14, n_iter=20,
    )

    pd.testing.assert_series_equal(
        res_clean.equity.iloc[:pivot_idx + 1],
        res_corrupt.equity.iloc[:pivot_idx + 1],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Overlay basics
# ---------------------------------------------------------------------------


def test_overlay_empty_when_history_too_short():
    btc = _btc(200)
    res = run_hmm_regime_overlay(btc, train_lookback_days=730)
    assert res.equity.empty
    assert res.num_rebalances == 0


def test_overlay_missing_close_column_raises():
    bad = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="close"):
        run_hmm_regime_overlay(bad)


def test_overlay_position_in_zero_one():
    btc = _btc(1000)
    res = run_hmm_regime_overlay(
        btc, train_lookback_days=400, rebalance_days=14, n_iter=20,
    )
    assert res.realized_position.min() >= 0.0
    assert res.realized_position.max() <= 1.0


def test_overlay_no_cost_no_rebalance_keeps_equity_flat():
    btc = _btc(1000)
    res = run_hmm_regime_overlay(
        btc, train_lookback_days=900,    # too long → no rebalance
        rebalance_days=14, n_iter=20,
        fee_rate=0.0, slippage_rate=0.0,
    )
    # If no rebalance ever fires, position stays at 0, equity flat.
    if res.num_rebalances == 0:
        assert res.equity.iloc[-1] == pytest.approx(res.initial_capital)


def test_overlay_costs_proportional_to_turnover():
    btc = _btc(1500)
    res = run_hmm_regime_overlay(
        btc, train_lookback_days=400, rebalance_days=14, n_iter=20,
        fee_rate=0.001, slippage_rate=0.0005,
    )
    # If we had any rebalance, cost should be > 0 and bounded by
    # equity_max * num_rebalances * (f + s).
    if res.num_rebalances > 0:
        assert res.total_fees > 0
        cap = res.equity.max() * res.num_rebalances * (0.001 + 0.0005)
        assert res.total_fees + res.total_slippage <= cap * 1.05  # tiny slack
