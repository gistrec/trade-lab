import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.backtest.reports import (
    TRADE_COLUMNS,
    trades_to_dataframe,
    write_trades_csv,
)
from trade_lab.strategies.base import Strategy


class _SignalStrategy(Strategy):
    name = "test_signal"

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


def _basic_result(signals, closes, **kwargs):
    candles = _candles(closes)
    result = run_backtest(
        candles,
        _SignalStrategy(signals),
        initial_capital=kwargs.pop("initial_capital", 10_000),
        fee_rate=kwargs.pop("fee_rate", 0.0),
        slippage_rate=kwargs.pop("slippage_rate", 0.0),
        **kwargs,
    )
    return candles, result


def test_csv_columns_match_spec():
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 100, 100],
        fee_rate=0.001, slippage_rate=0.0005,
    )
    df = trades_to_dataframe(result, candles)
    assert list(df.columns) == TRADE_COLUMNS


def test_default_excludes_open_trades_and_omits_is_open_column():
    # Position open at the end: positions = [0, 0, 1, 1]
    candles, result = _basic_result([0, 1, 1, 1], [100, 100, 110, 120])
    df = trades_to_dataframe(result, candles)
    assert "is_open" not in df.columns
    assert df.empty  # the sole trade is open and therefore excluded


def test_include_open_marks_open_trades():
    candles, result = _basic_result([0, 1, 1, 1], [100, 100, 110, 120])
    df = trades_to_dataframe(result, candles, include_open=True)
    assert "is_open" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["is_open"]


def test_entry_time_is_execution_bar_not_signal_bar():
    candles, result = _basic_result([0, 1, 0, 0], [100, 100, 110, 110])
    # signal at idx 1 -> positions = [0, 0, 1, 0]; execution bar is idx 2
    df = trades_to_dataframe(result, candles)
    assert df.iloc[0]["entry_time"] == candles.index[2]
    assert df.iloc[0]["entry_time"] != candles.index[1]  # signal bar
    assert df.iloc[0]["exit_time"] == candles.index[3]


def test_entry_price_applies_slippage_above_close():
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 100, 100],
        fee_rate=0.0, slippage_rate=0.01,
    )
    df = trades_to_dataframe(result, candles)
    row = df.iloc[0]
    # Execution close at entry = 100; with 1% slippage, buy pays 101.
    assert row["entry_price"] == pytest.approx(101.0)
    # Execution close at exit = 100; with 1% slippage, sell receives 99.
    assert row["exit_price"] == pytest.approx(99.0)


def test_gross_return_uses_raw_close_to_close():
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 100, 110],
        fee_rate=0.0, slippage_rate=0.005,
    )
    # positions = [0, 0, 1, 0]; raw close at entry idx 2 = 100, exit idx 3 = 110.
    df = trades_to_dataframe(result, candles)
    assert df.iloc[0]["gross_return_pct"] == pytest.approx(0.10)


def test_net_return_matches_equity_change():
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 110, 110],
        fee_rate=0.001, slippage_rate=0.0005,
    )
    df = trades_to_dataframe(result, candles)
    row = df.iloc[0]
    # positions = [0, 0, 1, 0]; prior equity = equity[1], final = equity[3]
    prior_equity = result.equity.iloc[1]
    final_equity = result.equity.iloc[3]
    expected_pnl = final_equity - prior_equity
    expected_net = expected_pnl / prior_equity
    assert row["pnl"] == pytest.approx(expected_pnl)
    assert row["net_return_pct"] == pytest.approx(expected_net)


def test_fees_paid_sums_entry_and_exit():
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 100, 100],
        fee_rate=0.01, slippage_rate=0.0,
    )
    df = trades_to_dataframe(result, candles)
    # Entry fee: 1.0 * 0.01 * equity[1] (= 10_000) = 100
    # Exit fee:  1.0 * 0.01 * equity[2] (=  9_900) =  99
    assert df.iloc[0]["fees_paid"] == pytest.approx(199.0)


def test_holding_period_counts_bars_in_market():
    candles, result = _basic_result(
        [0, 1, 1, 1, 0, 0], [100] * 6,
    )
    # positions = [0, 0, 1, 1, 1, 0]; held during bars 2, 3, 4 = 3 bars
    df = trades_to_dataframe(result, candles)
    assert df.iloc[0]["holding_period"] == 3


def test_open_trade_pnl_matches_mark_to_market():
    candles, result = _basic_result(
        [0, 1, 1, 1, 1], [100, 100, 110, 120, 130],
        fee_rate=0.0, slippage_rate=0.0,
    )
    # positions = [0, 0, 1, 1, 1] — open at end (entry idx = 2)
    df = trades_to_dataframe(result, candles, include_open=True)
    row = df.iloc[0]
    assert row["is_open"]
    assert row["pnl"] == pytest.approx(
        result.equity.iloc[-1] - result.equity.iloc[1]
    )
    assert row["exit_time"] == candles.index[-1]


def test_csv_round_trips_to_disk(tmp_path):
    candles, result = _basic_result(
        [0, 1, 0, 0], [100, 100, 110, 100],
        fee_rate=0.001, slippage_rate=0.0005,
    )
    out = write_trades_csv(result, candles, tmp_path / "outputs" / "trades.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == TRADE_COLUMNS
    assert len(loaded) == 1
    # Dates should round-trip as ISO strings
    assert str(loaded["entry_time"].iloc[0]).startswith("2024-01-01")


def test_csv_empty_when_no_trades(tmp_path):
    candles, result = _basic_result([0, 0, 0], [100, 100, 100])
    out = write_trades_csv(result, candles, tmp_path / "trades.csv")
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == TRADE_COLUMNS
    assert len(loaded) == 0


def test_empty_candles_returns_empty_dataframe():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    result = run_backtest(empty, _SignalStrategy([]))
    df = trades_to_dataframe(result, empty)
    assert list(df.columns) == TRADE_COLUMNS
    assert df.empty
