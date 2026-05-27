import pandas as pd
import pytest

from trade_lab.backtest.engine import run_backtest
from trade_lab.strategies.base import Strategy


class _SignalStrategy(Strategy):
    """Test helper: return whatever signal series we were initialized with."""

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


def test_flat_strategy_produces_no_trades():
    candles = _candles([100, 101, 102, 103, 104])
    result = run_backtest(
        candles,
        _SignalStrategy([0, 0, 0, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert result.trades == []
    assert result.equity.iloc[-1] == pytest.approx(10_000)


def test_constant_long_compounds_returns():
    closes = [100, 110, 121, 133.1, 146.41]
    candles = _candles(closes)
    result = run_backtest(
        candles,
        _SignalStrategy([1] * 5),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    # Position is shifted by one bar, so it earns the returns at bars 1..4
    # — exactly 1.1 ** 4 over 10_000.
    assert result.equity.iloc[-1] == pytest.approx(10_000 * 1.1 ** 4)


def test_look_ahead_bias_is_prevented():
    # A big jump happens between bar 0 and bar 1. If the strategy only
    # decides to be long once it has already SEEN bar 1 (signal at index 1),
    # the shift means the position only becomes active at bar 2 and we miss
    # the move.
    closes = [100, 200, 200, 200]
    candles = _candles(closes)
    result = run_backtest(
        candles,
        _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert result.equity.iloc[-1] == pytest.approx(10_000)

    # If the signal was committed BEFORE bar 1 (signal at index 0), the
    # position is active during bar 1 and we participate in the move — this
    # is legitimate, not look-ahead.
    legit = run_backtest(
        candles,
        _SignalStrategy([1, 0, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert legit.equity.iloc[-1] == pytest.approx(20_000)


def test_total_fees_match_per_bar_calculation():
    # Round-trip trade with a known fee rate. With closes flat at 100, the
    # only equity change comes from the fees themselves.
    candles = _candles([100, 100, 100, 100])
    # signals  = [0, 1, 0, 0]
    # positions= [0, 0, 1, 0]
    # turnover = [0, 0, 1, 1]   (entry fee at bar 2, exit fee at bar 3)
    result = run_backtest(
        candles,
        _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000,
        fee_rate=0.01,
        slippage_rate=0,
    )
    # Bar 2 fee: 10_000 * 0.01 = 100 (equity at end of bar 1 is still 10_000)
    # Bar 3 fee: 9_900  * 0.01 = 99  (equity at end of bar 2 is 9_900)
    assert result.total_fees == pytest.approx(199.0)
    assert result.equity.iloc[-1] == pytest.approx(10_000 - 199.0, rel=1e-6)


def test_buy_and_hold_return_is_close_to_close():
    candles = _candles([100, 110, 120, 150])
    result = run_backtest(
        candles,
        _SignalStrategy([0, 0, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert result.buy_and_hold_return == pytest.approx(0.5)


def test_buy_and_hold_equity_tracks_price():
    closes = [100, 110, 121, 120]
    candles = _candles(closes)
    result = run_backtest(
        candles,
        _SignalStrategy([0] * 4),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    bh = result.buy_and_hold_equity
    assert bh.index.equals(candles.index)
    assert bh.iloc[0] == pytest.approx(10_000)
    assert bh.iloc[1] == pytest.approx(11_000)
    assert bh.iloc[2] == pytest.approx(12_100)
    assert bh.iloc[3] == pytest.approx(12_000)
    # scalar buy_and_hold_return should agree with the curve
    assert bh.iloc[-1] / bh.iloc[0] - 1 == pytest.approx(result.buy_and_hold_return)


def test_buy_and_hold_scales_with_initial_capital():
    candles = _candles([100, 200])
    small = run_backtest(
        candles, _SignalStrategy([0, 0]),
        initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    big = run_backtest(
        candles, _SignalStrategy([0, 0]),
        initial_capital=50_000, fee_rate=0, slippage_rate=0,
    )
    # First bar parks the initial cash 1:1 into the asset.
    assert small.buy_and_hold_equity.iloc[0] == pytest.approx(10_000)
    assert big.buy_and_hold_equity.iloc[0] == pytest.approx(50_000)
    # Both should end up with the same asset-return ratio.
    small_ratio = small.buy_and_hold_equity.iloc[-1] / small.buy_and_hold_equity.iloc[0]
    big_ratio = big.buy_and_hold_equity.iloc[-1] / big.buy_and_hold_equity.iloc[0]
    assert small_ratio == pytest.approx(big_ratio)
    assert small_ratio == pytest.approx(2.0)


def test_strategy_is_independent_of_buy_and_hold_curve():
    # A strategy that's flat the whole time should produce a constant equity
    # curve while the buy & hold curve tracks the price.
    candles = _candles([100, 105, 110, 95])
    result = run_backtest(
        candles,
        _SignalStrategy([0, 0, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert (result.equity == 10_000).all()
    assert not (result.buy_and_hold_equity == 10_000).all()


def test_fees_and_slippage_reduce_returns():
    closes = [100, 100, 110, 110, 110]
    signals = [0, 1, 1, 0, 0]
    candles = _candles(closes)
    no_cost = run_backtest(
        candles,
        _SignalStrategy(signals),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    with_cost = run_backtest(
        candles,
        _SignalStrategy(signals),
        initial_capital=10_000,
        fee_rate=0.01,
        slippage_rate=0.005,
    )
    assert with_cost.equity.iloc[-1] < no_cost.equity.iloc[-1]
    assert with_cost.trades[0].return_pct < no_cost.trades[0].return_pct


def test_trade_extraction_matches_position_changes():
    candles = _candles([100, 100, 100, 110, 120, 130, 130, 140, 150])
    # signals  = [0,0,1,1,0,0,1,1,0]
    # positions= [0,0,0,1,1,0,0,1,1]   (shifted by one)
    # Two trades: one fully realized, one still open at the end.
    strat = _SignalStrategy([0, 0, 1, 1, 0, 0, 1, 1, 0])
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0
    )
    assert len(result.trades) == 2
    assert result.trades[0].bars_held == 3
    assert result.trades[1].bars_held == 2


def test_position_size_scales_exposure():
    closes = [100, 110, 121]
    candles = _candles(closes)
    full = run_backtest(
        candles,
        _SignalStrategy([1, 1, 1]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
        position_size=1.0,
    )
    half = run_backtest(
        candles,
        _SignalStrategy([1, 1, 1]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
        position_size=0.5,
    )
    full_return = full.equity.iloc[-1] / 10_000 - 1
    half_return = half.equity.iloc[-1] / 10_000 - 1
    assert 0 < half_return < full_return


def test_invalid_position_size_raises():
    candles = _candles([100, 101, 102])
    with pytest.raises(ValueError):
        run_backtest(candles, _SignalStrategy([0, 0, 0]), position_size=0)
    with pytest.raises(ValueError):
        run_backtest(candles, _SignalStrategy([0, 0, 0]), position_size=1.5)


def test_empty_candles_returns_empty_result():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    result = run_backtest(empty, _SignalStrategy([]))
    assert result.equity.empty
    assert result.trades == []
