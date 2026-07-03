import pandas as pd
import pytest

from trade_lab.backtest.engine import execution_bars, run_backtest
from trade_lab.strategies.base import Strategy


class _SignalStrategy(Strategy):
    """Test helper: return whatever signal series we were initialized with."""

    name = "test_signal"

    def __init__(self, signals):
        self._signals = list(signals)

    def generate_signals(self, candles):
        return pd.Series(self._signals, index=candles.index, dtype=int)


class _FloatSignalStrategy(Strategy):
    """Test helper: emit a float (laddered) signal series verbatim."""

    name = "float_signal"

    def __init__(self, signals):
        self._signals = list(signals)

    def generate_signals(self, candles):
        return pd.Series(self._signals, index=candles.index, dtype=float)


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


def test_execution_bars_finds_position_transitions():
    idx = pd.date_range("2024-01-01", periods=9, freq="1h")
    positions = pd.Series(
        [0, 0, 1, 1, 0, 0, 1, 1, 0], index=idx, dtype=float
    )
    entries, exits = execution_bars(positions)
    assert entries == [2, 6]
    assert exits == [4, 8]


def test_execution_bars_with_open_position_at_end():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    positions = pd.Series([0, 1, 1, 1], index=idx, dtype=float)
    entries, exits = execution_bars(positions)
    assert entries == [1]
    # Still open at the end -> no exit transition; markers correctly miss this.
    assert exits == []


def test_execution_bar_is_one_after_signal_bar():
    # Only one signal at index 3. After shift(1), the position becomes 1 at
    # index 4 — that's the execution bar, the one the marker should land on.
    candles = _candles([100] * 8)
    result = run_backtest(
        candles,
        _SignalStrategy([0, 0, 0, 1, 0, 0, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    # Confirm look-ahead invariant in positions space.
    assert result.positions.iloc[3] == 0
    assert result.positions.iloc[4] == 1
    assert result.positions.iloc[5] == 0

    entries, exits = execution_bars(result.positions)
    assert entries == [4]
    assert exits == [5]


def test_trade_entry_time_is_execution_bar_signal_time_is_one_before():
    # Trade.entry_time records the *execution* bar (where the position is
    # actually held); entry_signal_time records the bar before, where the
    # decision was made. The two are one bar apart by construction of the
    # look-ahead-protecting shift.
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles,
        _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000,
        fee_rate=0,
        slippage_rate=0,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    entries, _ = execution_bars(result.positions)

    # entry_time matches the execution bar from execution_bars().
    assert trade.entry_time == candles.index[entries[0]]
    assert trade.entry_time == candles.index[2]
    # entry_signal_time is the bar before — where the strategy decided.
    assert trade.entry_signal_time == candles.index[1]
    assert trade.entry_signal_time != trade.entry_time


def test_execution_bars_count_matches_trade_count_for_closed_trades():
    candles = _candles([100, 101, 102, 103, 104, 105, 106, 107, 108])
    strat = _SignalStrategy([0, 1, 1, 0, 0, 1, 0, 0, 0])
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0
    )
    entries, exits = execution_bars(result.positions)
    # positions = [0, 0, 1, 1, 0, 0, 1, 0, 0] — two cleanly closed trades.
    assert len(entries) == len(exits) == len(result.trades) == 2


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
    assert with_cost.trades[0].net_return_pct < no_cost.trades[0].net_return_pct


def test_trade_extraction_matches_position_changes():
    candles = _candles([100, 100, 100, 110, 120, 130, 130, 140, 150])
    # signals  = [0,0,1,1,0,0,1,1,0]
    # positions= [0,0,0,1,1,0,0,1,1]   (shifted by one)
    # Two trades: one fully realized, one still open at the end.
    strat = _SignalStrategy([0, 0, 1, 1, 0, 0, 1, 1, 0])
    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0
    )
    # positions = [0,0,0,1,1,0,0,1,1]
    # Trade 1: held at bars 3, 4 -> 2 bars
    # Trade 2 (still open at end): held at bars 7, 8 -> 2 bars
    assert len(result.trades) == 2
    assert result.trades[0].bars_held == 2
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


def test_gross_return_equals_net_at_zero_cost_for_ladder():
    """Documented invariant: at zero cost gross_return_pct == net_return_pct.
    For a pro-rata ladder the exposure varies within one trade, so the raw
    close-to-close ratio (which assumes 100% exposure every bar) overstates
    gross and breaks the invariant (regression: C5)."""
    closes = [100, 110, 121, 133.1, 146.41, 161.05, 177.155]  # +10%/bar
    candles = _candles(closes)
    result = run_backtest(
        candles, _FloatSignalStrategy([0, 0.5, 1.0, 1.0, 0.5, 0, 0]),
        initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0,
        position_size=1.0,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.gross_return_pct == pytest.approx(trade.net_return_pct)
    # And it is the exposure-weighted return, not the full price move.
    assert trade.gross_return_pct == pytest.approx(0.334021, abs=1e-5)


def test_invalid_position_size_raises():
    candles = _candles([100, 101, 102])
    with pytest.raises(ValueError):
        run_backtest(candles, _SignalStrategy([0, 0, 0]), position_size=0)
    with pytest.raises(ValueError):
        run_backtest(candles, _SignalStrategy([0, 0, 0]), position_size=1.5)


def test_fee_is_charged_on_buy():
    candles = _candles([100, 100, 100, 100])
    # signals[1]=1 -> positions[2]=1; never exits before end.
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 1, 1]),
        initial_capital=10_000, fee_rate=0.01, slippage_rate=0.0,
    )
    # Entry fee at bar 2 = 1.0 * 0.01 * 10_000 = 100
    assert result.total_fees == pytest.approx(100.0)
    # And the entry shows up on Trade.fees_paid.
    assert result.trades[0].fees_paid == pytest.approx(100.0)


def test_fee_is_charged_on_sell():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.01, slippage_rate=0.0,
    )
    # Buy at bar 2 (fee 100), sell at bar 3 (fee = 0.01 * equity after buy = 99).
    # Both legs roll into the same trade.
    trade = result.trades[0]
    assert trade.fees_paid == pytest.approx(199.0)


def test_buy_execution_price_includes_positive_slippage():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.0, slippage_rate=0.01,
    )
    # close at execution bar 2 = 100; +1% slippage -> 101.
    assert result.trades[0].entry_execution_price == pytest.approx(101.0)


def test_sell_execution_price_includes_negative_slippage():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.0, slippage_rate=0.01,
    )
    # close at execution bar 3 = 100; -1% slippage -> 99.
    assert result.trades[0].exit_execution_price == pytest.approx(99.0)


def test_net_return_is_lower_than_gross_when_costs_enabled():
    candles = _candles([100, 100, 110, 110])
    no_cost = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0,
    )
    with_cost = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005,
    )
    # Without costs: gross == net.
    assert no_cost.trades[0].gross_return_pct == pytest.approx(
        no_cost.trades[0].net_return_pct
    )
    # With costs: gross is unchanged, net is strictly lower.
    assert with_cost.trades[0].gross_return_pct == pytest.approx(
        no_cost.trades[0].gross_return_pct
    )
    assert with_cost.trades[0].net_return_pct < with_cost.trades[0].gross_return_pct


def test_total_slippage_separately_tracked():
    candles = _candles([100, 100, 100, 100])
    result = run_backtest(
        candles, _SignalStrategy([0, 1, 0, 0]),
        initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005,
    )
    # total_fees and total_slippage should be in their respective ratio:
    # slippage / fee == 0.5
    assert result.total_fees > 0
    assert result.total_slippage > 0
    assert result.total_slippage / result.total_fees == pytest.approx(0.5)


def test_activity_diagnostics_via_compute_metrics():
    from trade_lab.backtest.metrics import compute_metrics

    # signals  = [0, 1, 1, 0, 0, 1, 1, 1, 0]
    # positions= [0, 0, 1, 1, 0, 0, 1, 1, 1]  (shift by 1)
    # Trade 1 closed: held bars 2, 3 -> 2 bars
    # Trade 2 open at end: held bars 6, 7, 8 -> 3 bars
    candles = _candles([100, 100, 100, 110, 120, 130, 130, 140, 150])
    strat = _SignalStrategy([0, 1, 1, 0, 0, 1, 1, 1, 0])

    result = run_backtest(
        candles, strat,
        initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005,
    )
    m = compute_metrics(result)

    assert m.num_trades == 1
    assert m.num_open_trades == 1

    # Only the closed trade contributes; it held 2 bars.
    assert m.avg_holding_period == pytest.approx(2.0)
    assert m.median_holding_period == pytest.approx(2.0)

    # Positions are long at bars 2, 3, 6, 7, 8 -> 5 / 9.
    assert m.exposure_pct == pytest.approx(5 / 9)

    # Single completed trade -> best == worst == average.
    assert m.best_trade_return == m.worst_trade_return == pytest.approx(
        m.avg_net_trade_return
    )

    assert m.total_fees > 0
    assert m.fees_pct_of_initial_cash == pytest.approx(
        m.total_fees / m.initial_capital
    )


def test_best_and_worst_trade_identify_extremes():
    from trade_lab.backtest.metrics import compute_metrics

    # Want one winning trade and one losing trade so best != worst.
    # closes:    100 100 110 100 100 80 100 100
    # signals:   [0,  1,  0,  0,  1, 0,  0,  0]
    # positions: [0,  0,  1,  0,  0, 1,  0,  0]
    # Trade 1: held bar 2 (close 100 -> 110) = +10%
    # Trade 2: held bar 5 (close 100 -> 80)  = -20%
    candles = _candles([100, 100, 110, 100, 100, 80, 100, 100])
    strat = _SignalStrategy([0, 1, 0, 0, 1, 0, 0, 0])

    result = run_backtest(
        candles, strat, initial_capital=10_000, fee_rate=0, slippage_rate=0,
    )
    m = compute_metrics(result)

    assert m.num_trades == 2
    assert m.best_trade_return == pytest.approx(0.10)
    assert m.worst_trade_return == pytest.approx(-0.20)
    assert m.best_trade_return > m.worst_trade_return


def test_empty_candles_returns_empty_result():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    result = run_backtest(empty, _SignalStrategy([]))
    assert result.equity.empty
    assert result.trades == []
