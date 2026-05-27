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


def test_regime_signals_are_causal_appending_future_does_not_change_past():
    """Look-ahead audit: if the regime strategy peeked at future bars, then
    appending arbitrary future data would change signals over the original
    prefix. Asserting byte-identical signals on the overlap proves it
    doesn't."""
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0, 1, 60)) + 100.0
    extension = np.array([10.0, 1000.0, 5.0, 5000.0, 1.0, 9999.0])
    candles_base = _candles(base.tolist())
    candles_extended = _candles(np.concatenate([base, extension]).tolist())

    strat = RegimeSMACrossStrategy(fast_period=5, slow_period=10, regime_period=20)
    sig_base = strat.generate_signals(candles_base)
    sig_extended_prefix = strat.generate_signals(candles_extended).iloc[: len(base)]

    np.testing.assert_array_equal(sig_base.values, sig_extended_prefix.values)


def test_regime_signal_at_bar_n_doesnt_use_close_after_n():
    """Per-bar causality: changing one future bar's close must not affect
    any signal at or before that bar."""
    closes_a = [100.0 + 0.5 * i for i in range(40)]
    closes_b = closes_a.copy()
    closes_b[30] = 1e6  # absurd spike, well into the future relative to bar 25

    strat = RegimeSMACrossStrategy(fast_period=3, slow_period=6, regime_period=10)
    sig_a = strat.generate_signals(_candles(closes_a))
    sig_b = strat.generate_signals(_candles(closes_b))

    # Bars 0..29 are at-or-before the future change at bar 30; they must match.
    pd.testing.assert_series_equal(sig_a.iloc[:30], sig_b.iloc[:30])


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
