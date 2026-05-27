import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.backtest.reports import (
    DEBUG_TRADE_COLUMNS,
    debug_trades_dataframe,
    write_debug_trades_csv,
)
from trade_lab.strategies.base import Strategy
from trade_lab.strategies.regime_sma_cross import RegimeSMACrossStrategy
from trade_lab.strategies.sma_cross import SMACrossStrategy


class _SignalStrategy(Strategy):
    name = "_test"

    def __init__(self, signals):
        self._signals = list(signals)

    def generate_signals(self, candles):
        return pd.Series(self._signals, index=candles.index, dtype=int)


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


def test_debug_trades_columns_match_spec():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    df = debug_trades_dataframe(result, candles)
    assert list(df.columns) == DEBUG_TRADE_COLUMNS


def test_debug_trades_execution_time_is_one_bar_after_signal_time():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    df = debug_trades_dataframe(result, candles)
    # signal at idx 1, execution at idx 2 (one bar after).
    assert df.iloc[0]["signal_time"] == candles.index[1]
    assert df.iloc[0]["execution_time"] == candles.index[2]
    assert df.iloc[0]["execution_time"] > df.iloc[0]["signal_time"]


def test_debug_trades_signal_close_is_close_at_signal_time():
    candles = _candles([100, 105, 110, 115])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    df = debug_trades_dataframe(result, candles)
    # signal_time = idx 1 -> close = 105; execution_time = idx 2 -> close = 110.
    assert df.iloc[0]["signal_close"] == pytest.approx(105.0)
    assert df.iloc[0]["execution_open_or_close"] == pytest.approx(110.0)


def test_debug_trades_entry_price_applies_slippage():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0.01,
    )
    df = debug_trades_dataframe(result, candles)
    # execution close = 100; buy pays 1% more.
    assert df.iloc[0]["entry_price_after_slippage"] == pytest.approx(101.0)
    # exit close = 100; sell receives 1% less.
    assert df.iloc[0]["exit_price_after_slippage"] == pytest.approx(99.0)


def _trending_then_falling_closes():
    # Up, then down, then up — guarantees at least one closed trade for
    # any reasonable trend follower.
    up = list(range(100, 130))
    down = list(range(130, 100, -1))
    up2 = list(range(100, 130))
    return up + down + up2


def test_debug_trades_reason_mentions_sma_values_for_sma_cross():
    closes = _trending_then_falling_closes()
    candles = _candles(closes)
    strat = SMACrossStrategy(fast_period=3, slow_period=6)
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    df = debug_trades_dataframe(result, candles, strategy=strat)
    assert not df.empty
    reason = df.iloc[0]["reason"]
    assert "fast" in reason and "slow" in reason


def test_debug_trades_reason_mentions_all_three_indicators_for_regime():
    closes = _trending_then_falling_closes()
    candles = _candles(closes)
    strat = RegimeSMACrossStrategy(fast_period=3, slow_period=6, regime_period=10)
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    df = debug_trades_dataframe(result, candles, strategy=strat)
    assert not df.empty
    reason = df.iloc[0]["reason"]
    for tag in ("fast", "slow", "regime", "close"):
        assert tag in reason


def test_debug_trades_respects_limit():
    # 3 closed trades on this series.
    candles = _candles([100, 100, 100, 100, 100, 100, 100, 100, 100])
    strat = _SignalStrategy([0, 1, 0, 1, 0, 1, 0, 1, 0])
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    # 4 entries, 4 exits all paired -> 4 completed trades.
    df = debug_trades_dataframe(result, candles, limit=2)
    assert len(df) == 2


def test_debug_trades_csv_round_trips_to_disk(tmp_path):
    candles = _candles([100, 100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    out = write_debug_trades_csv(result, candles, tmp_path / "debug.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == DEBUG_TRADE_COLUMNS
