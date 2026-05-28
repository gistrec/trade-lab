"""Tests for VolatilityTargetWrapper.

Behavioural properties only — no specific numeric outputs, just
invariants that must hold for any inner strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.sma_cross import SMACrossStrategy
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy
from trade_lab.strategies.pma_ratio import PriceMaRatioStrategy
from trade_lab.strategies.vol_target_wrapper import VolatilityTargetWrapper


def _candles(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Construction / param validation
# ---------------------------------------------------------------------------


def test_invalid_parameters_raise():
    inner = SMACrossStrategy(20, 100)
    with pytest.raises(ValueError):
        VolatilityTargetWrapper(inner, annual_vol_target=0)
    with pytest.raises(ValueError):
        VolatilityTargetWrapper(inner, vol_lookback=1)
    with pytest.raises(ValueError):
        VolatilityTargetWrapper(inner, max_position_size=0)
    with pytest.raises(ValueError):
        VolatilityTargetWrapper(inner, max_position_size=2)


def test_name_includes_target_and_inner():
    inner = SMACrossStrategy(20, 100)
    w = VolatilityTargetWrapper(inner, annual_vol_target=0.30)
    assert "vol30" in w.name
    assert "sma_cross" in w.name


# ---------------------------------------------------------------------------
# No-lookahead and warmup
# ---------------------------------------------------------------------------


def test_no_lookahead_in_vol_scaling():
    """Appending arbitrary future candles must not change signals on
    the overlapping prefix."""
    rng = np.random.default_rng(0)
    base = (100 + np.linspace(0, 60, 400) + rng.normal(0, 2.0, 400)).tolist()
    future = [1e6, 1e-6, 1e6, 1e-6, 1e6]

    inner = SMACrossStrategy(20, 100)
    w = VolatilityTargetWrapper(inner, annual_vol_target=0.30, vol_lookback=20)
    sig_base = w.generate_signals(_candles(base))
    sig_ext = w.generate_signals(_candles(base + future)).iloc[: len(base)]
    np.testing.assert_array_equal(sig_base.values, sig_ext.values)


def test_warmup_bars_have_zero_exposure():
    """Within the rolling-vol warmup the realized-vol is NaN; the
    wrapper must default the weight to 0 (never lever up)."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 60, 300) + rng.normal(0, 2.0, 300)).tolist()
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    w = VolatilityTargetWrapper(inner, annual_vol_target=0.30, vol_lookback=50)
    sig = w.generate_signals(candles)
    # First 50 bars cannot have a valid vol estimate → exposure must be 0.
    assert (sig.iloc[:50] == 0.0).all()


# ---------------------------------------------------------------------------
# Cap and clipping
# ---------------------------------------------------------------------------


def test_exposure_never_exceeds_max_position_size():
    """A near-zero-vol path would otherwise produce a huge vol_weight;
    the cap must keep exposure in [0, max]."""
    n = 400
    closes = (100 + np.arange(n) * 0.01).tolist()  # vanishing vol
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    w = VolatilityTargetWrapper(
        inner, annual_vol_target=0.50, vol_lookback=20, max_position_size=1.0,
    )
    sig = w.generate_signals(candles)
    assert (sig >= 0.0).all()
    assert (sig <= 1.0).all()


def test_high_vol_shrinks_position_relative_to_low_vol():
    """Two series with the same trend but different daily noise. The
    high-vol path should yield a smaller average wrapper exposure once
    the inner SMA filter is satisfied — that's the convexity story."""
    n = 400
    rng = np.random.default_rng(0)
    trend = np.linspace(100, 400, n)
    low_vol = (trend + rng.normal(0, 0.5, n)).tolist()
    high_vol = (trend + rng.normal(0, 8.0, n)).tolist()

    inner = SMACrossStrategy(10, 30)
    w = VolatilityTargetWrapper(inner, annual_vol_target=0.30, vol_lookback=20)
    sig_low = w.generate_signals(_candles(low_vol))
    sig_high = w.generate_signals(_candles(high_vol))

    avg_low = sig_low.iloc[100:].mean()
    avg_high = sig_high.iloc[100:].mean()
    assert avg_low > 0
    assert avg_high >= 0
    assert avg_low > avg_high


def test_lower_target_gives_lower_average_exposure():
    """target=0.50 should produce ~5x the position of target=0.10 in
    the same vol regime, up to the cap."""
    rng = np.random.default_rng(0)
    n = 400
    closes = (100 + np.linspace(0, 100, n) + rng.normal(0, 4.0, n)).tolist()
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)

    w_low = VolatilityTargetWrapper(inner, annual_vol_target=0.10, vol_lookback=20)
    w_high = VolatilityTargetWrapper(inner, annual_vol_target=0.50, vol_lookback=20)

    sig_low = w_low.generate_signals(candles)
    sig_high = w_high.generate_signals(candles)

    assert sig_low.iloc[100:].mean() < sig_high.iloc[100:].mean()


# ---------------------------------------------------------------------------
# Integration: wraps any of the priority-5 strategies
# ---------------------------------------------------------------------------


def test_wraps_tsmom_with_disabled_internal_vol_target():
    """Wrap a TSMOM whose internal vol target has been disabled, so the
    only vol-scaling is from the wrapper. The wrapper produces a
    *different* exposure profile from the raw ensemble — the inner
    signal can be on the ``{0, 1/2, 1}`` ladder while the wrapped
    output is continuous-valued in ``[0, 1]``."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 80, 500) + rng.normal(0, 2.0, 500)).tolist()
    candles = _candles(closes)
    raw = TimeSeriesMomentumStrategy(
        lookbacks=(30, 90),
        sma_filter_periods=(100,),
        use_vol_target=False,
    )
    wrapped = VolatilityTargetWrapper(raw, annual_vol_target=0.30, vol_lookback=20)

    raw_sig = raw.generate_signals(candles)
    wrapped_sig = wrapped.generate_signals(candles)

    # Both are non-negative and capped at 1 (long-only spot).
    assert (wrapped_sig >= 0.0).all()
    assert (wrapped_sig <= 1.0 + 1e-9).all()
    # When raw is zero, wrapped must also be zero — the wrapper cannot
    # synthesize exposure out of nothing.
    flat = raw_sig == 0.0
    assert (wrapped_sig[flat] == 0.0).all()
    # In a steady up-trending regime both should be positive on average.
    assert wrapped_sig.iloc[200:].mean() > 0
    assert raw_sig.iloc[200:].mean() > 0


def test_wraps_pma_with_disabled_internal_vol_target():
    rng = np.random.default_rng(1)
    closes = (100 + np.linspace(0, 80, 500) + rng.normal(0, 2.0, 500)).tolist()
    candles = _candles(closes)
    raw = PriceMaRatioStrategy(
        ma_periods=(5, 10, 20),
        use_vol_target=False,
    )
    wrapped = VolatilityTargetWrapper(raw, annual_vol_target=0.30, vol_lookback=20)

    raw_sig = raw.generate_signals(candles)
    wrapped_sig = wrapped.generate_signals(candles)
    # Wrapped exposure stays in [0, 1] and is zero wherever raw is zero.
    assert (wrapped_sig >= 0.0).all()
    assert (wrapped_sig <= 1.0 + 1e-9).all()
    flat = raw_sig == 0.0
    assert (wrapped_sig[flat] == 0.0).all()


def test_engine_shift_applies_to_wrapper_signal():
    """Engine still shifts the wrapper's signal by one bar — the
    wrapper does not introduce a different execution lag."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 50, 400) + rng.normal(0, 1.5, 400)).tolist()
    candles = _candles(closes)
    inner = SMACrossStrategy(10, 30)
    w = VolatilityTargetWrapper(inner, annual_vol_target=0.30, vol_lookback=20)
    raw_signals = w.generate_signals(candles)

    result = run_backtest(candles, w, fee_rate=0.001, slippage_rate=0.0005)
    expected = raw_signals.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.positions, expected, check_names=False)


# ---------------------------------------------------------------------------
# Internal vol-target disable parity
# ---------------------------------------------------------------------------


def test_tsmom_use_vol_target_false_is_pure_ensemble_ladder():
    """tsmom(use_vol_target=False) returns the raw ensemble ladder
    clipped to [0, max_position_size]. With three lookbacks the ladder
    levels are {0, 1/3, 2/3, 1}."""
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 60, 500) + rng.normal(0, 3.0, 500)).tolist()
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30, 60, 90),
        sma_filter_periods=(),
        use_vol_target=False,
    )
    sig = strat.generate_signals(candles)
    nonzero = sig[sig > 0]
    if not nonzero.empty:
        levels = {round(v, 6) for v in nonzero.unique()}
        ladder = {round(1 / 3, 6), round(2 / 3, 6), 1.0}
        assert levels.issubset(ladder)


def test_pma_use_vol_target_false_is_pure_vote_ladder():
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 60, 500) + rng.normal(0, 3.0, 500)).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 10, 20, 50),
        use_vol_target=False,
    )
    sig = strat.generate_signals(candles)
    nonzero = sig[sig > 0]
    if not nonzero.empty:
        levels = {round(v, 6) for v in nonzero.unique()}
        ladder = {round(k / 4, 6) for k in (1, 2, 3, 4)}
        assert levels.issubset(ladder)
