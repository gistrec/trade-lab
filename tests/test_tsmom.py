import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy


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


def test_no_lookahead_in_tsmom_signal():
    rng = np.random.default_rng(0)
    base = (100 + np.linspace(0, 60, 500) + rng.normal(0, 2.0, 500)).tolist()
    future = [1e6, 1e-6, 1e6, 1e-6, 1e6]

    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30, 90),
        sma_filter_periods=(),
        vol_lookback=20,
    )
    sig_base = strat.generate_signals(_candles(base))
    sig_extended = strat.generate_signals(_candles(base + future)).iloc[: len(base)]
    np.testing.assert_array_equal(sig_base.values, sig_extended.values)


def test_negative_trailing_return_produces_zero_contribution():
    """A clean downtrend over the lookback window must produce a 0 signal
    once the warmup is over (no vol scaling can make a 0 raw signal positive)."""
    n = 500
    closes = np.linspace(200, 50, n).tolist()
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30,),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,
    )
    sig = strat.generate_signals(candles)
    assert (sig.iloc[40:] == 0).all()


def test_positive_trailing_return_produces_positive_signal():
    """A clean uptrend must give a positive average position after warmup."""
    n = 500
    closes = np.linspace(50, 500, n).tolist()
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30, 90),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,  # huge -> capped at 1
    )
    sig = strat.generate_signals(candles)
    assert sig.iloc[200:].mean() > 0.5


def test_ensemble_ladder_with_two_lookbacks():
    """With two lookbacks and the vol weight capped, raw signal must
    sit on the {0, 0.5, 1.0} ladder."""
    rng = np.random.default_rng(1)
    closes = (100 + np.linspace(0, 80, 600) + rng.normal(0, 3.0, 600)).tolist()
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30, 90),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,
        rebalance_threshold=0.0,
    )
    sig = strat.generate_signals(candles)
    nonzero = sig[sig > 0]
    if not nonzero.empty:
        levels = {round(v, 6) for v in nonzero.unique()}
        assert levels.issubset({0.5, 1.0})


def test_sma_filter_zeroes_out_signal_below_long_sma():
    """A long uptrend followed by a crash that puts close well under the
    SMA must produce zero exposure even if some short lookbacks still
    show positive trailing return."""
    uptrend = list(range(100, 600))
    crash_floor = [180.0] * 80
    candles = _candles(uptrend + crash_floor)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30,),
        sma_filter_periods=(200,),
        vol_lookback=20,
    )
    sig = strat.generate_signals(candles)
    assert (sig.iloc[-40:] == 0).all()


def test_vol_targeting_reduces_position_when_realized_vol_rises():
    n = 500
    rng = np.random.default_rng(0)
    trend = np.linspace(100, 400, n)
    low_vol = (trend + rng.normal(0, 0.5, n)).tolist()
    high_vol = (trend + rng.normal(0, 8.0, n)).tolist()
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30,),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=0.25,
    )
    sig_low = strat.generate_signals(_candles(low_vol))
    sig_high = strat.generate_signals(_candles(high_vol))
    assert sig_low.iloc[100:].mean() > sig_high.iloc[100:].mean()


def test_exposure_capped_at_max_position_size():
    n = 500
    closes = (100 + np.arange(n) * 0.01).tolist()  # tiny drift, vanishing vol
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy(
        lookbacks=(30,),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=0.25,
        max_position_size=1.0,
    )
    sig = strat.generate_signals(candles)
    assert (sig >= 0.0).all()
    assert (sig <= 1.0).all()


def test_engine_shifts_tsmom_signal_by_one_bar():
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 50, 500) + rng.normal(0, 1.5, 500)).tolist()
    candles = _candles(closes)
    strat = TimeSeriesMomentumStrategy()
    result = run_backtest(
        candles, strat, initial_capital=10_000.0, fee_rate=0.001, slippage_rate=0.0005,
    )
    raw_signals = strat.generate_signals(candles)
    expected_positions = raw_signals.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        result.positions, expected_positions, check_names=False
    )


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(lookbacks=())
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(vol_lookback=1)
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(annual_vol_target=0)
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(max_position_size=2)
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(rebalance_threshold=-0.01)


def test_string_lookbacks_accepted_for_cli_use():
    strat = TimeSeriesMomentumStrategy(lookbacks="30,90,180")
    assert strat.lookbacks == (30, 90, 180)


def test_use_vol_target_boolean_literals_coerced():
    """The CLI passes booleans as strings; recognized literals and 0/1
    coerce to the right bool (not bool('false') == True)."""
    assert TimeSeriesMomentumStrategy(use_vol_target="false").use_vol_target is False
    assert TimeSeriesMomentumStrategy(use_vol_target="true").use_vol_target is True
    assert TimeSeriesMomentumStrategy(use_vol_target="off").use_vol_target is False
    assert TimeSeriesMomentumStrategy(use_vol_target=0).use_vol_target is False
    assert TimeSeriesMomentumStrategy(use_vol_target=1).use_vol_target is True
    assert TimeSeriesMomentumStrategy(use_vol_target=False).use_vol_target is False
    assert TimeSeriesMomentumStrategy().use_vol_target is True  # default


def test_use_vol_target_rejects_ambiguous_value():
    """A typo'd / unrecognized boolean must fail loud, not silently enable
    the flag via bool('maybe') == True (verify finding)."""
    with pytest.raises(ValueError, match="use_vol_target"):
        TimeSeriesMomentumStrategy(use_vol_target="maybe")


def test_pma_ratio_use_vol_target_rejects_ambiguous_value():
    from trade_lab.strategies.pma_ratio import PriceMaRatioStrategy

    assert PriceMaRatioStrategy(use_vol_target="no").use_vol_target is False
    with pytest.raises(ValueError, match="use_vol_target"):
        PriceMaRatioStrategy(use_vol_target="maybe")
