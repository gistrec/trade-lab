import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.regime_only import RegimeOnlyStrategy


def _candles(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1] * len(closes),
        },
        index=idx,
    )


def test_signal_is_one_when_close_above_regime_sma():
    # Steady uptrend: close stays above its own trailing average.
    candles = _candles(list(range(1, 50)))
    strat = RegimeOnlyStrategy(regime_period=10)
    signals = strat.generate_signals(candles)
    assert signals.iloc[15:].eq(1).all()


def test_signal_is_zero_when_close_below_regime_sma():
    candles = _candles(list(range(50, 0, -1)))
    strat = RegimeOnlyStrategy(regime_period=10)
    signals = strat.generate_signals(candles)
    assert signals.iloc[15:].eq(0).all()


def test_signal_is_flat_during_warmup():
    candles = _candles(list(range(1, 50)))
    strat = RegimeOnlyStrategy(regime_period=10)
    signals = strat.generate_signals(candles)
    # The first (regime_period - 1) bars cannot have a valid SMA.
    assert signals.iloc[:9].eq(0).all()


def test_signal_index_matches_candles():
    candles = _candles(list(range(1, 30)))
    strat = RegimeOnlyStrategy(regime_period=10)
    signals = strat.generate_signals(candles)
    assert signals.index.equals(candles.index)


def test_invalid_regime_period_raises():
    with pytest.raises(ValueError):
        RegimeOnlyStrategy(regime_period=0)


def test_signal_at_bar_n_does_not_use_close_after_n():
    """Per-bar causality: changing a future bar must not change any
    signal at or before it."""
    closes_a = [100.0 + 0.5 * i for i in range(40)]
    closes_b = closes_a.copy()
    closes_b[30] = 1e6  # absurd spike, well into the future relative to bar 25

    strat = RegimeOnlyStrategy(regime_period=10)
    sig_a = strat.generate_signals(_candles(closes_a))
    sig_b = strat.generate_signals(_candles(closes_b))

    pd.testing.assert_series_equal(sig_a.iloc[:30], sig_b.iloc[:30])


def test_extending_series_with_future_data_does_not_change_past():
    rng = np.random.default_rng(0)
    base = (np.linspace(0, 30, 60) + rng.normal(0, 1.5, 60) + 100).tolist()
    extension = [10.0, 1000.0, 5.0, 5000.0]

    strat = RegimeOnlyStrategy(regime_period=10)
    sig_base = strat.generate_signals(_candles(base))
    sig_extended_prefix = strat.generate_signals(
        _candles(base + extension)
    ).iloc[: len(base)]

    np.testing.assert_array_equal(sig_base.values, sig_extended_prefix.values)


def test_engine_executes_regime_only_signal_one_bar_after_decision():
    """Signal at the close of bar N becomes a position at bar N+1."""
    # 11 bars; price flips to "above" only at bar 5.
    # With regime_period=3, SMAs by bar index 2 onward:
    # close: 90,90,90,90,90,120,120,120,120,120,120
    # sma(3): nan, nan, 90, 90, 90, 100, 110, 120, 120, 120, 120
    # close > sma: -, -, F, F, F, T, T, F (eq), F, F, F
    closes = [90.0, 90, 90, 90, 90, 120, 120, 120, 120, 120, 120]
    candles = _candles(closes)
    strat = RegimeOnlyStrategy(regime_period=3)
    sig = strat.generate_signals(candles)
    # Signal flips to 1 at bar 5 (the first bar where close > rolling avg).
    assert sig.iloc[4] == 0
    assert sig.iloc[5] == 1

    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    # Engine shifts by 1: positions[6] = signal[5] = 1 (not positions[5]).
    assert result.positions.iloc[5] == 0
    assert result.positions.iloc[6] == 1
