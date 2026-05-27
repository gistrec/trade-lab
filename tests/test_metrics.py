import pandas as pd
import pytest

from trade_lab.backtest.engine import BacktestResult, Trade
from trade_lab.backtest.metrics import compute_metrics


def _result(
    equity_values,
    trades=None,
    initial=10_000,
    buy_and_hold_return=0.0,
    total_fees=0.0,
):
    idx = pd.date_range("2024-01-01", periods=len(equity_values), freq="1D")
    equity = pd.Series(equity_values, index=idx, dtype=float)
    returns = equity.pct_change().fillna(0)
    positions = pd.Series([0.0] * len(equity_values), index=idx, dtype=float)
    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades=list(trades or []),
        initial_capital=initial,
        fee_rate=0.0,
        slippage_rate=0.0,
        total_fees=total_fees,
        buy_and_hold_return=buy_and_hold_return,
    )


def test_no_trades_returns_zero_metrics():
    result = _result([10_000] * 10)
    m = compute_metrics(result)
    assert m.total_return == pytest.approx(0.0)
    assert m.max_drawdown == pytest.approx(0.0)
    assert m.num_trades == 0
    assert m.win_rate == 0.0
    assert m.avg_trade_return == 0.0


def test_total_return_matches_equity_curve():
    result = _result([10_000, 11_000, 12_000])
    m = compute_metrics(result)
    assert m.total_return == pytest.approx(0.2)


def test_max_drawdown_finds_worst_peak_to_trough():
    # Peak at 11_000, trough at 9_500 -> drawdown = 1500 / 11_000 ~= 0.1364
    result = _result([10_000, 11_000, 9_500, 10_000, 10_500])
    m = compute_metrics(result)
    assert m.max_drawdown == pytest.approx(1500 / 11_000, rel=1e-3)


def test_max_drawdown_uses_global_peak_not_local():
    # Two distinct dips; the second is deeper relative to the all-time high
    # at index 1 (12_000), even though the equity later recovers.
    result = _result([10_000, 12_000, 9_000, 11_000, 8_000, 10_000])
    m = compute_metrics(result)
    # (8_000 - 12_000) / 12_000 = -1/3 ~ 0.3333
    assert m.max_drawdown == pytest.approx(1 / 3, rel=1e-3)


def test_buy_and_hold_and_total_fees_pass_through():
    result = _result(
        [10_000, 10_500],
        buy_and_hold_return=0.07,
        total_fees=42.5,
    )
    m = compute_metrics(result)
    assert m.buy_and_hold_return == pytest.approx(0.07)
    assert m.total_fees == pytest.approx(42.5)
    assert m.final_equity == pytest.approx(10_500)
    assert m.initial_capital == pytest.approx(10_000)


def test_win_rate_and_averages():
    ts = pd.Timestamp("2024-01-01")
    trades = [
        Trade(entry_time=ts, exit_time=ts, entry_price=100, exit_price=110,
              return_pct=0.10, bars_held=1),
        Trade(entry_time=ts, exit_time=ts, entry_price=100, exit_price=90,
              return_pct=-0.10, bars_held=1),
        Trade(entry_time=ts, exit_time=ts, entry_price=100, exit_price=105,
              return_pct=0.05, bars_held=1),
    ]
    result = _result([10_000, 10_500], trades=trades)
    m = compute_metrics(result)
    assert m.num_trades == 3
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.avg_trade_return == pytest.approx((0.10 - 0.10 + 0.05) / 3)
    assert m.avg_win == pytest.approx(0.075)
    assert m.avg_loss == pytest.approx(-0.10)


def test_metrics_with_empty_equity_are_zero():
    empty = pd.Series(dtype=float)
    result = BacktestResult(
        equity=empty,
        returns=empty,
        positions=empty,
        trades=[],
        initial_capital=10_000,
    )
    m = compute_metrics(result)
    assert m.total_return == 0.0
    assert m.max_drawdown == 0.0
    assert m.num_trades == 0
