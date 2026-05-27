"""Backtest performance metrics with explicit gross/net cost split."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult


@dataclass
class Metrics:
    """Summary of backtest performance with cost transparency."""

    # Capital
    initial_capital: float
    final_equity: float

    # Returns
    gross_return: float                 # before any costs
    total_return: float                 # = net return after costs
    buy_and_hold_return: float

    # Drawdowns
    max_drawdown: float
    buy_and_hold_final_equity: float
    buy_and_hold_max_drawdown: float

    # Trade stats
    num_trades: int
    win_rate: float
    avg_trade_return: float             # alias for avg_net_trade_return
    avg_gross_trade_return: float
    avg_net_trade_return: float
    avg_cost_per_trade: float           # (fees + slippage) / entry capital
    avg_win: float
    avg_loss: float

    # Costs (dollars)
    total_fees: float
    total_slippage: float

    # Round-trip cost parameters (computed from the rates themselves)
    buy_cost_pct: float
    sell_cost_pct: float
    round_trip_cost_pct: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    if drawdown.empty:
        return 0.0
    worst = drawdown.min()
    return float(abs(worst)) if pd.notna(worst) else 0.0


def compute_metrics(result: BacktestResult) -> Metrics:
    """Compute summary metrics from a :class:`BacktestResult`."""
    equity = result.equity
    trades = result.trades

    bh_equity = result.buy_and_hold_equity
    bh_final = float(bh_equity.iloc[-1]) if not bh_equity.empty else 0.0
    bh_dd = _max_drawdown(bh_equity)

    buy_cost = result.fee_rate + result.slippage_rate
    sell_cost = result.fee_rate + result.slippage_rate
    round_trip = buy_cost + sell_cost

    def _empty_metrics(final_equity: float, total_return: float, max_dd: float):
        return Metrics(
            initial_capital=result.initial_capital,
            final_equity=final_equity,
            gross_return=0.0,
            total_return=total_return,
            buy_and_hold_return=result.buy_and_hold_return,
            max_drawdown=max_dd,
            buy_and_hold_final_equity=bh_final,
            buy_and_hold_max_drawdown=bh_dd,
            num_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_gross_trade_return=0.0,
            avg_net_trade_return=0.0,
            avg_cost_per_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_fees=result.total_fees,
            total_slippage=result.total_slippage,
            buy_cost_pct=buy_cost,
            sell_cost_pct=sell_cost,
            round_trip_cost_pct=round_trip,
        )

    if equity.empty or result.initial_capital <= 0:
        return _empty_metrics(0.0, 0.0, 0.0)

    final_equity = float(equity.iloc[-1])
    total_return = float(final_equity / result.initial_capital - 1)
    max_dd = _max_drawdown(equity)

    gross_equity = result.gross_equity
    gross_return = (
        float(gross_equity.iloc[-1] / result.initial_capital - 1)
        if not gross_equity.empty
        else 0.0
    )

    if not trades:
        return _empty_metrics(final_equity, total_return, max_dd)

    gross_returns = pd.Series([t.gross_return_pct for t in trades])
    net_returns = pd.Series([t.net_return_pct for t in trades])
    cost_per_trade = pd.Series(
        [
            (t.fees_paid + t.slippage_cost_estimate) / t.entry_capital
            if t.entry_capital > 0
            else 0.0
            for t in trades
        ]
    )
    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]

    return Metrics(
        initial_capital=result.initial_capital,
        final_equity=final_equity,
        gross_return=gross_return,
        total_return=total_return,
        buy_and_hold_return=result.buy_and_hold_return,
        max_drawdown=max_dd,
        buy_and_hold_final_equity=bh_final,
        buy_and_hold_max_drawdown=bh_dd,
        num_trades=len(trades),
        win_rate=float(len(wins) / len(trades)),
        avg_trade_return=float(net_returns.mean()),
        avg_gross_trade_return=float(gross_returns.mean()),
        avg_net_trade_return=float(net_returns.mean()),
        avg_cost_per_trade=float(cost_per_trade.mean()),
        avg_win=float(wins.mean()) if not wins.empty else 0.0,
        avg_loss=float(losses.mean()) if not losses.empty else 0.0,
        total_fees=result.total_fees,
        total_slippage=result.total_slippage,
        buy_cost_pct=buy_cost,
        sell_cost_pct=sell_cost,
        round_trip_cost_pct=round_trip,
    )
