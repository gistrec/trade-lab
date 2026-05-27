"""Vectorized long-only backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from ..strategies.base import Strategy


@dataclass
class Trade:
    """A round-trip long trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    bars_held: int


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
    buy_and_hold_return: float = 0.0
    buy_and_hold_equity: pd.Series = field(
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

    Signals are shifted by one bar before application so a signal generated
    on bar ``N`` executes against bar ``N+1`` (no look-ahead). Fees and
    slippage are charged on every change in exposure, in proportion to the
    size of that change.

    Parameters
    ----------
    candles
        OHLCV DataFrame indexed by timestamp.
    strategy
        Strategy producing 0/1 target-position signals.
    initial_capital
        Starting equity used as the base of the equity curve.
    fee_rate
        Per-side fee charged as a fraction of position notional.
    slippage_rate
        Per-side slippage charged as a fraction of position notional.
    position_size
        Fraction of equity to allocate when long, in ``(0, 1]``.
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
    bar_returns = close.pct_change().fillna(0.0)
    strat_returns = positions * bar_returns

    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])
    costs = turnover * (fee_rate + slippage_rate)
    net_returns = strat_returns - costs

    equity = initial_capital * (1 + net_returns).cumprod()

    # Dollar fees are turnover * fee_rate applied to the equity held just
    # before the rebalance (i.e. equity at the end of the prior bar).
    prior_equity = equity.shift(1).fillna(initial_capital)
    total_fees = float((turnover * fee_rate * prior_equity).sum())

    # Buy-and-hold benchmark: park the same starting cash in the asset at the
    # first bar and mark-to-market every subsequent bar. No fees.
    buy_and_hold_equity = initial_capital * (close / close.iloc[0])
    buy_and_hold_return = (
        float(close.iloc[-1] / close.iloc[0] - 1) if len(close) >= 2 else 0.0
    )

    trades = _extract_trades(positions, close, net_returns)

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        positions=positions,
        trades=trades,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        total_fees=total_fees,
        buy_and_hold_return=buy_and_hold_return,
        buy_and_hold_equity=buy_and_hold_equity,
    )


def execution_bars(positions: pd.Series) -> tuple[List[int], List[int]]:
    """Return integer indices where ``positions`` transitions to/from long.

    These are the *execution candles* — the bars during which the engine
    actually holds (or first stops holding) a position after the look-ahead
    shift. A signal generated at bar ``N`` (close of bar N) becomes the
    position at bar ``N+1``; that bar ``N+1`` is what this helper returns
    for the entry, not the signal bar.

    Exits are the first bar where ``positions`` drops back to zero. If a
    position is still open at the end of the series there is no
    corresponding exit index — visualizing that gap is the point.
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
    net_returns: pd.Series,
) -> List[Trade]:
    trades: List[Trade] = []
    in_position = False
    entry_idx: int | None = None

    n = len(positions)
    for i in range(n):
        long_now = positions.iloc[i] > 0
        if long_now and not in_position:
            entry_idx = i
            in_position = True
        elif not long_now and in_position:
            assert entry_idx is not None
            trades.append(_build_trade(entry_idx, i, positions, close, net_returns))
            in_position = False
            entry_idx = None

    if in_position and entry_idx is not None:
        trades.append(
            _build_trade(
                entry_idx, n - 1, positions, close, net_returns, open_at_end=True
            )
        )

    return trades


def _build_trade(
    entry_idx: int,
    exit_idx: int,
    positions: pd.Series,
    close: pd.Series,
    net_returns: pd.Series,
    open_at_end: bool = False,
) -> Trade:
    # Entry is executed at the prior bar's close (where the signal was made).
    entry_ref = max(entry_idx - 1, 0)
    # On a normal exit, that same convention applies; if the trade is still
    # open at the end of the series we mark-to-market at the final bar.
    exit_ref = exit_idx if open_at_end else max(exit_idx - 1, 0)

    trade_net = net_returns.iloc[entry_idx : exit_idx + 1]
    trade_return = float((1 + trade_net).prod() - 1)

    return Trade(
        entry_time=positions.index[entry_ref],
        exit_time=positions.index[exit_ref],
        entry_price=float(close.iloc[entry_ref]),
        exit_price=float(close.iloc[exit_ref]),
        return_pct=trade_return,
        bars_held=exit_idx - entry_idx + 1,
    )
