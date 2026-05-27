"""Backtest performance metrics."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult


@dataclass
class Metrics:
    """Summary of backtest performance."""

    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    total_fees: float
    buy_and_hold_final_equity: float
    buy_and_hold_return: float
    buy_and_hold_max_drawdown: float


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of ``equity`` as a positive fraction."""
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

    if equity.empty or result.initial_capital <= 0:
        return Metrics(
            initial_capital=result.initial_capital,
            final_equity=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_fees=result.total_fees,
            buy_and_hold_final_equity=bh_final,
            buy_and_hold_return=result.buy_and_hold_return,
            buy_and_hold_max_drawdown=bh_dd,
        )

    final_equity = float(equity.iloc[-1])
    total_return = float(final_equity / result.initial_capital - 1)
    max_dd = _max_drawdown(equity)

    if not trades:
        return Metrics(
            initial_capital=result.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_dd,
            num_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_fees=result.total_fees,
            buy_and_hold_final_equity=bh_final,
            buy_and_hold_return=result.buy_and_hold_return,
            buy_and_hold_max_drawdown=bh_dd,
        )

    trade_returns = pd.Series([t.return_pct for t in trades])
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    return Metrics(
        initial_capital=result.initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        max_drawdown=max_dd,
        num_trades=len(trades),
        win_rate=float(len(wins) / len(trades)),
        avg_trade_return=float(trade_returns.mean()),
        avg_win=float(wins.mean()) if not wins.empty else 0.0,
        avg_loss=float(losses.mean()) if not losses.empty else 0.0,
        total_fees=result.total_fees,
        buy_and_hold_final_equity=bh_final,
        buy_and_hold_return=result.buy_and_hold_return,
        buy_and_hold_max_drawdown=bh_dd,
    )
