import pandas as pd
import pytest

from trade_lab.strategies.sma_cross import SMACrossStrategy


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


def test_signals_are_zero_before_indicators_warm_up():
    candles = _candles(list(range(1, 20)))
    strat = SMACrossStrategy(fast_period=5, slow_period=10)
    signals = strat.generate_signals(candles)
    # The first (slow_period - 1) bars cannot have a valid slow SMA.
    assert (signals.iloc[:9] == 0).all()


def test_signals_are_one_in_steady_uptrend():
    candles = _candles(list(range(1, 100)))
    strat = SMACrossStrategy(fast_period=5, slow_period=20)
    signals = strat.generate_signals(candles)
    assert signals.iloc[30:].eq(1).all()


def test_signals_are_zero_in_steady_downtrend():
    candles = _candles(list(range(100, 0, -1)))
    strat = SMACrossStrategy(fast_period=5, slow_period=20)
    signals = strat.generate_signals(candles)
    assert signals.iloc[30:].eq(0).all()


def test_signals_index_matches_candles():
    candles = _candles(list(range(1, 50)))
    strat = SMACrossStrategy(fast_period=5, slow_period=10)
    signals = strat.generate_signals(candles)
    assert signals.index.equals(candles.index)


def test_sma_signals_are_causal_appending_future_does_not_change_past():
    """Look-ahead audit: extending the series with arbitrary future bars
    must not change signals over the original prefix."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = (rng.normal(0, 1, 50).cumsum() + 100).tolist()
    extension = [1e6, 1e-6, 1e6, 1e-6]

    strat = SMACrossStrategy(fast_period=5, slow_period=10)
    sig_base = strat.generate_signals(_candles(base))
    sig_extended_prefix = strat.generate_signals(_candles(base + extension)).iloc[: len(base)]

    np.testing.assert_array_equal(sig_base.values, sig_extended_prefix.values)


def test_invalid_periods_raise():
    with pytest.raises(ValueError):
        SMACrossStrategy(fast_period=20, slow_period=20)
    with pytest.raises(ValueError):
        SMACrossStrategy(fast_period=30, slow_period=20)
    with pytest.raises(ValueError):
        SMACrossStrategy(fast_period=0, slow_period=20)
