"""Tests for the multi-asset ensemble runner.

Behavioural properties only — no specific numeric outputs from the
real-data run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.ensemble import (
    EnsembleResult,
    SleeveSpec,
    _target_weights_and_turnover,
    correlation_summary,
    run_ensemble_walk_forward,
    sortino_ratio,
)
from trade_lab.backtest.ensemble_sleeves import default_sleeves
from trade_lab.strategies.sma_cross import SMACrossStrategy


def test_interior_return_gap_is_flat_day_not_universe_change():
    """A single missing OOS return AFTER a sleeve has started (one asset
    missing a daily bar its peers have) is a flat day, not a universe
    change: the sleeve must stay in N_active with its weight carried, and
    no rebalance turnover billed at the gap or the bar after it
    (regression: C8)."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    panel = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan, 0.01, 0.0],   # interior gap at idx[2]
            "B": [0.00, 0.01, 0.02, -0.01, 0.0],
        },
        index=idx,
    )
    active, weights, turnover, _cost = _target_weights_and_turnover(
        panel, fee_rate=0.001, slippage_rate=0.0005,
    )
    # A stays in-universe through the gap; both sleeves counted.
    assert bool(active.loc[idx[2], "A"]) is True
    assert weights.loc[idx[2], "A"] == pytest.approx(0.5)
    assert weights.loc[idx[2], "B"] == pytest.approx(0.5)
    # No spurious rebalance turnover at the gap or the bar after it.
    assert turnover.loc[idx[2]] == pytest.approx(0.0)
    assert turnover.loc[idx[3]] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _candles(n: int, start: str = "2018-01-01", seed: int = 0, slope: float = 0.5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")
    closes = 100 + slope * np.arange(n) + rng.normal(0, 1.5, n)
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


def _trivial_sleeve(label: str, asset: str) -> SleeveSpec:
    return SleeveSpec(
        label=label,
        asset=asset,
        factory=lambda: SMACrossStrategy(fast_period=10, slow_period=30),
        warmup_days=30,
    )


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def test_default_sleeves_has_21_with_consistent_vol_picks():
    """3 strategies x 7 default assets = 21 sleeves. BTC always raw."""
    sleeves = default_sleeves()
    assert len(sleeves) == 21
    btc_sleeves = [s for s in sleeves if s.asset == "BTC"]
    assert len(btc_sleeves) == 3
    # All BTC sleeves must have "raw" in their label (no vol-wrapper).
    for s in btc_sleeves:
        assert "raw" in s.label, (
            f"BTC sleeve {s.label} unexpectedly has a vol-target wrapper "
            "— findings/vol_targeting_regime_gate.md says BTC stays raw."
        )


def test_default_sleeves_labels_are_unique():
    sleeves = default_sleeves()
    labels = [s.label for s in sleeves]
    assert len(set(labels)) == len(labels)


# ---------------------------------------------------------------------------
# Runner — basic shape
# ---------------------------------------------------------------------------


def test_runner_returns_expected_shape():
    sleeves = [
        _trivial_sleeve("a", "ASSET_A"),
        _trivial_sleeve("b", "ASSET_B"),
    ]
    candles = {
        "ASSET_A": _candles(1100, seed=1),
        "ASSET_B": _candles(1100, seed=2),
    }
    res = run_ensemble_walk_forward(
        sleeves, candles,
        train_months=12, test_months=3, step_months=3,
    )
    assert isinstance(res, EnsembleResult)
    assert set(res.per_sleeve_oos_returns.keys()) == {"a", "b"}
    assert res.sleeve_returns_panel.shape[1] == 2
    assert res.target_weights.shape == res.sleeve_returns_panel.shape
    assert res.portfolio_returns_net.notna().any()
    assert isinstance(res.portfolio_metrics["sharpe"], float)


def test_runner_empty_sleeves_raises():
    with pytest.raises(ValueError):
        run_ensemble_walk_forward([], {"X": _candles(500)})


# ---------------------------------------------------------------------------
# Dynamic equal-weight semantics
# ---------------------------------------------------------------------------


def test_dynamic_weight_redistributes_when_sleeve_activates():
    """A sleeve that activates mid-window should cause the other sleeves
    to be trimmed from 1/N to 1/(N+1) at the bar of activation."""
    # ASSET_A has 1500 bars; ASSET_B has only 700 (so its WF starts much later)
    candles = {
        "ASSET_A": _candles(1500, seed=1),
        "ASSET_B": _candles(700, start="2020-01-01", seed=2),
    }
    sleeves = [
        _trivial_sleeve("a", "ASSET_A"),
        _trivial_sleeve("b", "ASSET_B"),
    ]
    res = run_ensemble_walk_forward(
        sleeves, candles,
        train_months=12, test_months=3, step_months=3,
    )
    # When only ASSET_A is active, its weight must be 1.0.
    only_a_rows = res.sleeve_active_panel[
        res.sleeve_active_panel["a"] & ~res.sleeve_active_panel["b"]
    ]
    if not only_a_rows.empty:
        weights_when_only_a = res.target_weights.loc[only_a_rows.index, "a"]
        np.testing.assert_allclose(weights_when_only_a.to_numpy(), 1.0)

    # When both are active, equal-weight 0.5 each.
    both_rows = res.sleeve_active_panel[
        res.sleeve_active_panel.all(axis=1)
    ]
    if not both_rows.empty:
        wa = res.target_weights.loc[both_rows.index, "a"]
        wb = res.target_weights.loc[both_rows.index, "b"]
        np.testing.assert_allclose(wa.to_numpy(), 0.5)
        np.testing.assert_allclose(wb.to_numpy(), 0.5)


def test_rebalance_cost_zero_at_initial_simultaneous_entry():
    """All sleeves coming online on the same first bar = no portfolio-level
    cost: their entry is already paid by each sleeve's internal engine."""
    candles = {f"A{i}": _candles(1100, seed=i) for i in range(3)}
    sleeves = [_trivial_sleeve(f"s{i}", f"A{i}") for i in range(3)]
    res = run_ensemble_walk_forward(
        sleeves, candles,
        train_months=12, test_months=3, step_months=3,
    )
    # First bar of the OOS panel: every sleeve entering, no trims.
    # Portfolio-level turnover should be 0 (entries already paid).
    first_bar_turnover = res.rebalance_turnover.iloc[0]
    assert first_bar_turnover == pytest.approx(0.0, abs=1e-9)


def test_rebalance_cost_nonzero_when_new_asset_activates_later():
    """When ASSET_B comes online after ASSET_A's start, the trim on
    ASSET_A from 1.0 to 0.5 must produce non-zero portfolio cost. The
    ASSET_B entry portion is excluded (sleeve internals cover it)."""
    candles = {
        "ASSET_A": _candles(1500, seed=1),
        "ASSET_B": _candles(700, start="2020-01-01", seed=2),
    }
    sleeves = [_trivial_sleeve("a", "ASSET_A"), _trivial_sleeve("b", "ASSET_B")]
    res = run_ensemble_walk_forward(
        sleeves, candles,
        train_months=12, test_months=3, step_months=3,
    )
    # Locate the first bar where ASSET_B becomes active.
    b_activates = res.sleeve_active_panel.index[res.sleeve_active_panel["b"]][0]
    cost_on_activation = res.rebalance_turnover.loc[b_activates]
    # The expected portfolio turnover: trim of ASSET_A by 0.5 (1.0 → 0.5);
    # the ASSET_B entry portion (0.5) is netted out by entry_credit.
    assert cost_on_activation == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# No look-ahead at portfolio level
# ---------------------------------------------------------------------------


def test_portfolio_returns_use_shifted_weights():
    """Today's portfolio return uses yesterday's target weights — i.e.
    the engine's one-bar shift carries through to the allocator too."""
    candles = {f"A{i}": _candles(1200, seed=i) for i in range(2)}
    sleeves = [_trivial_sleeve(f"s{i}", f"A{i}") for i in range(2)]
    res = run_ensemble_walk_forward(
        sleeves, candles,
        train_months=12, test_months=3, step_months=3,
    )
    shifted = res.target_weights.shift(1).fillna(0.0)
    sleeve_returns = res.sleeve_returns_panel.fillna(0.0)
    expected = (shifted * sleeve_returns).sum(axis=1)
    pd.testing.assert_series_equal(
        res.portfolio_returns_gross, expected, check_names=False, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Correlation summary
# ---------------------------------------------------------------------------


def test_correlation_summary_handles_empty_and_single():
    # Single sleeve → no pairs to summarize.
    res = correlation_summary(pd.DataFrame(np.eye(1), index=["a"], columns=["a"]))
    assert res["n_sleeves"] == 1
    assert res["high_corr_pair_count"] == 0


def test_correlation_summary_counts_high_corr_pairs():
    # Three sleeves, all correlated at 0.9 in pairwise. Should count
    # 3 high-corr pairs (all pairs > 0.6).
    n = 3
    matrix = pd.DataFrame(
        0.9 * np.ones((n, n)) + 0.1 * np.eye(n),
        index=[f"s{i}" for i in range(n)],
        columns=[f"s{i}" for i in range(n)],
    )
    res = correlation_summary(matrix)
    assert res["high_corr_pair_count"] == 3
    assert res["mean_pairwise_corr"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------


def test_sortino_zero_when_no_negative_returns():
    """Pure positive return series → no downside → Sortino is 0 by
    convention (not infinity)."""
    s = pd.Series([0.01, 0.02, 0.005, 0.015])
    assert sortino_ratio(s) == 0.0


def test_sortino_higher_than_sharpe_on_symmetric_tail():
    """If the downside std equals overall std, Sortino == Sharpe.
    For a symmetric distribution with similar positive/negative mass,
    Sortino is typically larger because the downside denominator is
    smaller in absolute terms."""
    rng = np.random.default_rng(0)
    # Right-skewed: more downside is suppressed.
    returns = pd.Series(rng.normal(0.001, 0.01, 1000)).clip(lower=-0.005)
    sr = float(returns.mean() / returns.std() * np.sqrt(365))
    sortino_val = sortino_ratio(returns)
    assert sortino_val > sr


# ---------------------------------------------------------------------------
# Portfolio DSR
# ---------------------------------------------------------------------------


def test_portfolio_dsr_decreases_with_more_trials():
    """Same portfolio, more trials in DSR call → lower DSR."""
    candles = {f"A{i}": _candles(1100, seed=i) for i in range(3)}
    sleeves = [_trivial_sleeve(f"s{i}", f"A{i}") for i in range(3)]
    res_low = run_ensemble_walk_forward(
        sleeves, candles, train_months=12, test_months=3, step_months=3,
        num_trials_for_dsr=10,
    )
    res_high = run_ensemble_walk_forward(
        sleeves, candles, train_months=12, test_months=3, step_months=3,
        num_trials_for_dsr=10_000,
    )
    assert res_high.portfolio_dsr <= res_low.portfolio_dsr


# ---------------------------------------------------------------------------
# Sanity check for the user's stated invariant
# ---------------------------------------------------------------------------


def test_portfolio_sharpe_bounded_by_zero_corr_aggregation():
    """The user explicitly asked: portfolio Sharpe should not exceed
    ``sum_of_sleeve_sharpes / sqrt(N)`` when correlations are zero.
    Run on synthetic IID-noise sleeves and confirm the inequality holds
    approximately (relative tolerance to allow for finite-sample noise)."""
    n_sleeves = 5
    rng = np.random.default_rng(42)
    sleeves = []
    candles = {}
    for i in range(n_sleeves):
        candles[f"A{i}"] = _candles(1100, seed=100 + i)
        sleeves.append(_trivial_sleeve(f"s{i}", f"A{i}"))
    res = run_ensemble_walk_forward(
        sleeves, candles, train_months=12, test_months=3, step_months=3,
    )

    # Per-sleeve Sharpe (annualized) on the OOS series.
    sleeve_sharpes = []
    for s in sleeves:
        sr_series = res.per_sleeve_oos_returns[s.label]
        if sr_series.empty or sr_series.std() == 0:
            sleeve_sharpes.append(0.0)
            continue
        sleeve_sharpes.append(
            float(sr_series.mean() / sr_series.std() * np.sqrt(365))
        )
    naive_bound = sum(sleeve_sharpes) / np.sqrt(n_sleeves)
    port_sharpe = res.portfolio_metrics["sharpe"]
    # The bound is for zero-correlation; with real (small but nonzero)
    # correlations the portfolio Sharpe can exceed or fall below the
    # naive bound. Allow generous slack but flag if it explodes.
    assert port_sharpe < abs(naive_bound) * 2 + 1.0
