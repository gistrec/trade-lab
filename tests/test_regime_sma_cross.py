import numpy as np
import pandas as pd
import pytest

from trade_lab.strategies.regime_sma_cross import RegimeSMACrossStrategy


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


def test_signals_are_one_only_when_both_crossover_and_regime_align():
    # Steady uptrend: fast > slow eventually, close > regime eventually too.
    candles = _candles(list(range(1, 60)))
    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    signals = strat.generate_signals(candles)
    # After all three SMAs warm up, the strategy should be long in this
    # monotonic uptrend.
    assert signals.iloc[30:].eq(1).all()


def test_regime_filter_forces_flat_in_downtrend():
    candles = _candles(list(range(100, 0, -1)))
    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    signals = strat.generate_signals(candles)
    # Pure downtrend: never long.
    assert signals.eq(0).all()


def test_regime_filter_blocks_long_when_below_regime_sma():
    # First half flat at 200, second half flat at 50 — well below the long
    # regime SMA. Even if a fast/slow cross fires in the dip, the regime
    # filter should keep the strategy flat.
    closes = [200] * 30 + [50] * 30
    candles = _candles(closes)
    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    signals = strat.generate_signals(candles)
    # After the step down, the regime SMA stays above the closes for a long
    # time -> all signals 0 once we're in the dip and indicators have caught up.
    assert (signals.iloc[40:] == 0).all()


def test_warmup_period_is_flat():
    candles = _candles(list(range(1, 30)))
    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    signals = strat.generate_signals(candles)
    # First (regime_period - 1) bars cannot have a valid regime SMA.
    assert (signals.iloc[:19] == 0).all()


def test_signals_index_matches_candles():
    candles = _candles(list(range(1, 50)))
    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    signals = strat.generate_signals(candles)
    assert signals.index.equals(candles.index)


def test_invalid_periods_raise():
    with pytest.raises(ValueError):
        RegimeSMACrossStrategy(fast_period=20, slow_period=20, regime_period=200)
    with pytest.raises(ValueError):
        RegimeSMACrossStrategy(fast_period=20, slow_period=100, regime_period=100)
    with pytest.raises(ValueError):
        RegimeSMACrossStrategy(fast_period=0, slow_period=20, regime_period=50)
    with pytest.raises(ValueError):
        RegimeSMACrossStrategy(fast_period=200, slow_period=100, regime_period=300)


def test_regime_filter_strictly_subset_of_plain_sma_cross():
    # The regime variant can never be long where the plain SMA cross isn't —
    # it only ever filters signals out, never adds them.
    from trade_lab.strategies.sma_cross import SMACrossStrategy

    # Mixed series: uptrend, dip, recovery.
    rng = np.random.default_rng(0)
    closes = np.concatenate(
        [
            np.linspace(100, 150, 200),
            np.linspace(150, 80, 100),
            np.linspace(80, 130, 200),
        ]
    ) + rng.normal(0, 0.5, 500)
    candles = _candles(closes.tolist())

    plain = SMACrossStrategy(fast_period=10, slow_period=50).generate_signals(candles)
    filtered = RegimeSMACrossStrategy(
        fast_period=10, slow_period=50, regime_period=200,
    ).generate_signals(candles)

    # Wherever the regime strategy is long, the plain crossover is also long.
    assert (filtered <= plain).all()
    # And the regime strategy should be strictly less active than the plain
    # one over this mixed series.
    assert filtered.sum() < plain.sum()
