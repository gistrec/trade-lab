"""Vectorized long-only backtest engine with explicit cost accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..strategies.base import Strategy


@dataclass
class Trade:
    """A round-trip long trade with detailed cost breakdown.

    Time fields are split between the *signal* candle (where the strategy
    decided) and the *execution* candle (the next bar, where the engine
    actually held the position). Prices are slippage-adjusted to reflect
    what the strategy effectively paid / received.
    """

    # Execution timing (when the position was actually held).
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    # Signal timing (one bar before execution; ``exit_signal_time`` is
    # ``None`` for trades still open at the end of the window).
    entry_signal_time: pd.Timestamp
    exit_signal_time: Optional[pd.Timestamp]

    # Slippage-adjusted execution prices.
    entry_execution_price: float
    exit_execution_price: float

    # Returns.
    gross_return_pct: float          # raw close-to-close at execution bars
    net_return_pct: float            # from equity, after fees + slippage

    # Costs (dollars).
    fees_paid: float                 # exchange fees only
    slippage_cost_estimate: float    # slippage in dollars
    pnl: float                       # final_equity - entry_capital

    # Other.
    entry_capital: float             # equity at the moment of entry
    bars_held: int                   # bars the position was actually held


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 0.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    buy_and_hold_return: float = 0.0
    buy_and_hold_equity: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    gross_equity: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )


def run_backtest(
    candles: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> BacktestResult:
    """Run a long-only backtest of ``strategy`` against ``candles``.

    Cost model is split explicitly:

    * Fees: ``fee_rate`` charged on every change in exposure (entry + exit).
    * Slippage: ``slippage_rate`` charged on every change in exposure.
      Conceptually a buy fills at ``close * (1 + slippage_rate)`` and a sell
      at ``close * (1 - slippage_rate)`` — vectorized into the same
      turnover-times-rate formulation as fees because it is equivalent at
      the equity-curve level.

    The engine returns separate gross and net return tracks so the
    cost-free path can be inspected alongside the realistic one.
    """
    if not 0 < position_size <= 1:
        raise ValueError("position_size must be in (0, 1]")
    if candles.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            equity=empty,
            returns=empty,
            positions=empty,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

    signals = strategy.generate_signals(candles).reindex(candles.index).fillna(0)
    positions = signals.shift(1).fillna(0).astype(float) * float(position_size)

    close = candles["close"].astype(float)
    bar_returns = close.pct_change(fill_method=None).fillna(0.0)
    gross_returns = positions * bar_returns                    # before costs

    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])
    fee_costs = turnover * fee_rate
    slippage_costs = turnover * slippage_rate
    net_returns = gross_returns - fee_costs - slippage_costs

    equity = initial_capital * (1 + net_returns).cumprod()
    gross_equity = initial_capital * (1 + gross_returns).cumprod()

    # Dollar costs are turnover * rate applied to the equity at the end of
    # the prior bar (i.e. the capital that was rebalanced).
    prior_equity = equity.shift(1).fillna(initial_capital)
    total_fees = float((turnover * fee_rate * prior_equity).sum())
    total_slippage = float((turnover * slippage_rate * prior_equity).sum())

    buy_and_hold_equity, buy_and_hold_return = buy_and_hold_with_costs(
        close,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )

    trades = _extract_trades(
        positions=positions,
        close=close,
        equity=equity,
        turnover=turnover,
        gross_returns=gross_returns,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        initial_capital=initial_capital,
    )

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        positions=positions,
        trades=trades,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        total_fees=total_fees,
        total_slippage=total_slippage,
        buy_and_hold_return=buy_and_hold_return,
        buy_and_hold_equity=buy_and_hold_equity,
        gross_equity=gross_equity,
    )


def buy_and_hold_with_costs(
    close: pd.Series,
    *,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> tuple[pd.Series, float]:
    """Buy-and-hold equity series and return AFTER one round of entry costs.

    A buy-and-hold trader incurs the same fee + slippage on entry as a
    strategy that opens an equal-sized long on bar 1. They do NOT pay
    an exit fee at the window's end — same convention the engine uses
    for strategies that finish the window still long (mark-to-market,
    no closing turnover charged).

    Before this helper, every B&H computation in the repo silently used
    a pre-cost ``close / close.iloc[0]`` curve, which gave B&H a free
    ~0.15% head-start over every strategy entering on the first bar.
    Symmetric costing is the honest comparison.

    ``initial_capital``, ``fee_rate`` and ``slippage_rate`` should
    match what was passed to the strategy run; otherwise the B&H
    benchmark again becomes asymmetric.
    """
    if close.empty or len(close) < 2:
        return close.copy() * 0.0, 0.0
    entry_cost = float(fee_rate) + float(slippage_rate)
    effective_capital = initial_capital * (1.0 - entry_cost)
    equity = effective_capital * (close / close.iloc[0])
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    return equity, total_return


def execution_bars(positions: pd.Series) -> tuple[List[int], List[int]]:
    """Return integer indices where ``positions`` transitions to/from long.

    Entries / exits are the *execution candles* — the bars where the engine
    actually holds (or first stops holding) a position. A signal at bar N
    becomes a position at bar N+1; this helper returns N+1, not N.
    """
    entries: List[int] = []
    exits: List[int] = []
    in_position = False
    for i, pos in enumerate(positions):
        is_long = pos > 0
        if is_long and not in_position:
            entries.append(i)
            in_position = True
        elif not is_long and in_position:
            exits.append(i)
            in_position = False
    return entries, exits


def _extract_trades(
    positions: pd.Series,
    close: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    gross_returns: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    initial_capital: float,
) -> List[Trade]:
    trades: List[Trade] = []
    entries, exits = execution_bars(positions)
    n_bars = len(positions)

    for i, entry_idx in enumerate(entries):
        is_open = i >= len(exits)
        exit_idx = exits[i] if not is_open else n_bars - 1
        trades.append(
            _build_trade(
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                positions=positions,
                close=close,
                equity=equity,
                turnover=turnover,
                gross_returns=gross_returns,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                initial_capital=initial_capital,
                open_at_end=is_open,
            )
        )

    return trades


def _build_trade(
    entry_idx: int,
    exit_idx: int,
    positions: pd.Series,
    close: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    gross_returns: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    initial_capital: float,
    open_at_end: bool = False,
) -> Trade:
    n_bars = len(positions)

    # entry_idx is guaranteed >= 1 because positions[0] is always 0
    # (signal.shift(1).fillna(0) starts flat).
    entry_signal_time = positions.index[entry_idx - 1]
    exit_signal_time: Optional[pd.Timestamp] = (
        None if open_at_end else positions.index[exit_idx - 1]
    )

    raw_entry_close = float(close.iloc[entry_idx])
    raw_exit_close = float(close.iloc[exit_idx])

    entry_execution_price = raw_entry_close * (1 + slippage_rate)
    exit_execution_price = raw_exit_close * (1 - slippage_rate)

    # gross_return is the exposure-weighted return the engine actually
    # accrued over the trade — the product of (1 + position × bar_return)
    # across the held bars. A raw close-to-close ratio would assume 100%
    # exposure every bar and overstate any trade whose exposure varies
    # within it (the pro-rata ladder {0, 0.5, 1.0}, the vol-target
    # wrapper). Slicing [entry_idx : exit_idx + 1] mirrors net_return's
    # window, so at zero costs gross_return_pct == net_return_pct exactly.
    gross_slice = gross_returns.iloc[entry_idx : exit_idx + 1]
    gross_return_pct = float((1.0 + gross_slice).prod() - 1.0)

    entry_capital = float(equity.iloc[entry_idx - 1])
    final_equity = float(equity.iloc[exit_idx])
    pnl = final_equity - entry_capital
    net_return_pct = pnl / entry_capital if entry_capital > 0 else 0.0

    entry_fee = float(turnover.iloc[entry_idx] * fee_rate * entry_capital)
    entry_slippage = float(turnover.iloc[entry_idx] * slippage_rate * entry_capital)
    if open_at_end:
        exit_fee = 0.0
        exit_slippage = 0.0
    else:
        exit_prior_equity = float(equity.iloc[exit_idx - 1])
        exit_fee = float(turnover.iloc[exit_idx] * fee_rate * exit_prior_equity)
        exit_slippage = float(
            turnover.iloc[exit_idx] * slippage_rate * exit_prior_equity
        )
    fees_paid = entry_fee + exit_fee
    slippage_cost = entry_slippage + exit_slippage

    bars_held = (n_bars - entry_idx) if open_at_end else (exit_idx - entry_idx)

    return Trade(
        entry_time=positions.index[entry_idx],
        exit_time=positions.index[exit_idx],
        entry_signal_time=entry_signal_time,
        exit_signal_time=exit_signal_time,
        entry_execution_price=entry_execution_price,
        exit_execution_price=exit_execution_price,
        gross_return_pct=gross_return_pct,
        net_return_pct=net_return_pct,
        fees_paid=fees_paid,
        slippage_cost_estimate=slippage_cost,
        pnl=pnl,
        entry_capital=entry_capital,
        bars_held=bars_held,
    )
