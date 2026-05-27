"""Backtest performance metrics with explicit gross/net cost split."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult


VERDICT_OUTPERFORMS_BH = "OUTPERFORMS_BH"
VERDICT_LOWER_RETURN_LOWER_DD = "LOWER_RETURN_LOWER_DD"
VERDICT_UNDERPERFORMS_BH = "UNDERPERFORMS_BH"

# Threshold for "meaningfully lower" drawdown in the LOWER_RETURN_LOWER_DD
# verdict. 2 percentage points absolute keeps things conservative: a 1pp
# improvement on DD isn't worth giving up return for.
_MEANINGFUL_DD_DIFFERENCE = 0.02


def benchmark_verdict(metrics: "Metrics") -> str:
    """Classify the strategy against buy & hold.

    Returns one of:

    * ``OUTPERFORMS_BH`` — strategy return is higher AND max drawdown is
      not worse than buy & hold.
    * ``LOWER_RETURN_LOWER_DD`` — strategy return is lower but max drawdown
      is meaningfully lower (>= 2pp). Defensible if you prefer lower risk.
    * ``UNDERPERFORMS_BH`` — everything else. Includes the mixed case
      ("higher return but worse drawdown") because trading more risk for
      more return without a clear edge isn't an unambiguous win.
    """
    if (
        metrics.total_return > metrics.buy_and_hold_return
        and metrics.max_drawdown <= metrics.buy_and_hold_max_drawdown
    ):
        return VERDICT_OUTPERFORMS_BH
    if (
        metrics.total_return < metrics.buy_and_hold_return
        and metrics.max_drawdown
        < metrics.buy_and_hold_max_drawdown - _MEANINGFUL_DD_DIFFERENCE
    ):
        return VERDICT_LOWER_RETURN_LOWER_DD
    return VERDICT_UNDERPERFORMS_BH


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

    # Trade stats (over completed trades only)
    num_trades: int                     # completed trades
    num_open_trades: int                # still open at end of window
    win_rate: float
    avg_trade_return: float             # alias for avg_net_trade_return
    avg_gross_trade_return: float
    avg_net_trade_return: float
    avg_cost_per_trade: float           # (fees + slippage) / entry capital
    avg_win: float
    avg_loss: float
    best_trade_return: float            # max net_return_pct among completed
    worst_trade_return: float           # min net_return_pct among completed

    # Activity diagnostics
    avg_holding_period: float           # mean bars in position (completed)
    median_holding_period: float        # median bars in position (completed)
    exposure_pct: float                 # share of bars with non-zero position
    fees_pct_of_initial_cash: float     # total_fees / initial_capital

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
    """Compute summary metrics from a :class:`BacktestResult`.

    Trade-level aggregates (win rate, averages, holding period, best /
    worst) cover *completed* trades only. Open positions at the end of
    the window are mark-to-market but not counted in those statistics —
    their unrealized P&L still shows up in equity, total_return, and
    drawdown.
    """
    equity = result.equity
    trades = result.trades

    completed_trades = [t for t in trades if t.exit_signal_time is not None]
    num_open_trades = len(trades) - len(completed_trades)

    bh_equity = result.buy_and_hold_equity
    bh_final = float(bh_equity.iloc[-1]) if not bh_equity.empty else 0.0
    bh_dd = _max_drawdown(bh_equity)

    buy_cost = result.fee_rate + result.slippage_rate
    sell_cost = result.fee_rate + result.slippage_rate
    round_trip = buy_cost + sell_cost

    positions = result.positions
    exposure_pct = (
        float((positions > 0).mean()) if not positions.empty else 0.0
    )
    fees_pct = (
        float(result.total_fees / result.initial_capital)
        if result.initial_capital > 0
        else 0.0
    )

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
            num_open_trades=num_open_trades,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_gross_trade_return=0.0,
            avg_net_trade_return=0.0,
            avg_cost_per_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade_return=0.0,
            worst_trade_return=0.0,
            avg_holding_period=0.0,
            median_holding_period=0.0,
            exposure_pct=exposure_pct,
            fees_pct_of_initial_cash=fees_pct,
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

    if not completed_trades:
        return _empty_metrics(final_equity, total_return, max_dd)

    gross_returns = pd.Series([t.gross_return_pct for t in completed_trades])
    net_returns = pd.Series([t.net_return_pct for t in completed_trades])
    holding_periods = pd.Series([t.bars_held for t in completed_trades])
    cost_per_trade = pd.Series(
        [
            (t.fees_paid + t.slippage_cost_estimate) / t.entry_capital
            if t.entry_capital > 0
            else 0.0
            for t in completed_trades
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
        num_trades=len(completed_trades),
        num_open_trades=num_open_trades,
        win_rate=float(len(wins) / len(completed_trades)),
        avg_trade_return=float(net_returns.mean()),
        avg_gross_trade_return=float(gross_returns.mean()),
        avg_net_trade_return=float(net_returns.mean()),
        avg_cost_per_trade=float(cost_per_trade.mean()),
        avg_win=float(wins.mean()) if not wins.empty else 0.0,
        avg_loss=float(losses.mean()) if not losses.empty else 0.0,
        best_trade_return=float(net_returns.max()),
        worst_trade_return=float(net_returns.min()),
        avg_holding_period=float(holding_periods.mean()),
        median_holding_period=float(holding_periods.median()),
        exposure_pct=exposure_pct,
        fees_pct_of_initial_cash=fees_pct,
        total_fees=result.total_fees,
        total_slippage=result.total_slippage,
        buy_cost_pct=buy_cost,
        sell_cost_pct=sell_cost,
        round_trip_cost_pct=round_trip,
    )
