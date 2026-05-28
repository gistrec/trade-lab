import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.pma_ratio import PriceMaRatioStrategy


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


def test_no_lookahead_in_pma_signal():
    rng = np.random.default_rng(0)
    base = (100 + np.linspace(0, 60, 400) + rng.normal(0, 2.0, 400)).tolist()
    future = [1e6, 1e-6, 1e6, 1e-6, 1e6]
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20, 50),
        sma_filter_periods=(),
        vol_lookback=20,
    )
    sig_base = strat.generate_signals(_candles(base))
    sig_extended = strat.generate_signals(_candles(base + future)).iloc[: len(base)]
    np.testing.assert_array_equal(sig_base.values, sig_extended.values)


def test_uptrend_keeps_all_pma_votes_above_one():
    """A clean uptrend keeps close above every MA, so the ensemble
    average reaches 1.0 after the slowest MA finishes warming up."""
    n = 400
    closes = np.linspace(50, 500, n).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20, 50, 100),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,
        rebalance_threshold=0.0,
    )
    sig = strat.generate_signals(candles)
    # After all MAs are warmed up the raw signal is 1.0; capped vol weight
    # brings the final position to 1.0 too.
    assert sig.iloc[150:].mean() > 0.9


def test_downtrend_drives_signal_to_zero():
    n = 400
    closes = np.linspace(500, 50, n).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20, 50),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,
    )
    sig = strat.generate_signals(candles)
    assert (sig.iloc[100:] == 0).all()


def test_ladder_with_three_ma_periods():
    """Mixed regime: short MAs say yes, long MAs say no -> raw signal
    must land on the {0, 1/3, 2/3, 1} ladder."""
    # Long downtrend then a short rebound. Short MAs reflect the rebound
    # quickly, longer ones lag.
    closes = np.linspace(500, 100, 200).tolist() + np.linspace(100, 200, 60).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20, 100),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=10.0,
        rebalance_threshold=0.0,
    )
    sig = strat.generate_signals(candles)
    nonzero = sig[sig > 0]
    if not nonzero.empty:
        levels = {round(v, 6) for v in nonzero.unique()}
        ladder = {round(1 / 3, 6), round(2 / 3, 6), 1.0}
        assert levels.issubset(ladder)


def test_optional_sma_filter_zeroes_out_signal():
    uptrend = list(range(100, 400))
    crash_floor = [120.0] * 80
    candles = _candles(uptrend + crash_floor)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20),
        sma_filter_periods=(200,),
        vol_lookback=20,
    )
    sig = strat.generate_signals(candles)
    assert (sig.iloc[-40:] == 0).all()


def test_exposure_capped_at_max_position_size():
    n = 400
    closes = (100 + np.arange(n) * 0.01).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy(
        ma_periods=(5, 20, 50),
        sma_filter_periods=(),
        vol_lookback=20,
        annual_vol_target=0.25,
        max_position_size=1.0,
    )
    sig = strat.generate_signals(candles)
    assert (sig >= 0.0).all()
    assert (sig <= 1.0).all()


def test_engine_shifts_pma_signal_by_one_bar():
    rng = np.random.default_rng(0)
    closes = (100 + np.linspace(0, 50, 400) + rng.normal(0, 1.5, 400)).tolist()
    candles = _candles(closes)
    strat = PriceMaRatioStrategy()
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
        PriceMaRatioStrategy(ma_periods=())
    with pytest.raises(ValueError):
        PriceMaRatioStrategy(vol_lookback=1)
    with pytest.raises(ValueError):
        PriceMaRatioStrategy(annual_vol_target=0)
    with pytest.raises(ValueError):
        PriceMaRatioStrategy(max_position_size=2)
    with pytest.raises(ValueError):
        PriceMaRatioStrategy(rebalance_threshold=-0.01)


def test_string_periods_accepted_for_cli_use():
    strat = PriceMaRatioStrategy(ma_periods="5,10,20")
    assert strat.ma_periods == (5, 10, 20)
