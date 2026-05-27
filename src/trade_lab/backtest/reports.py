"""Tabular exports of backtest results (trade lists, CSV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import BacktestResult, execution_bars


TRADE_COLUMNS = [
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "gross_return_pct",
    "net_return_pct",
    "fees_paid",
    "holding_period",
    "pnl",
]


def trades_to_dataframe(
    result: BacktestResult,
    candles: pd.DataFrame,
    include_open: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame describing every trade in ``result``.

    Entries / exits are placed on the *execution* candles — the bars where
    positions actually transition (one bar after the signal bar). Prices
    include slippage:

        entry_price = close[entry_bar] * (1 + slippage_rate)   # buys pay more
        exit_price  = close[exit_bar]  * (1 - slippage_rate)   # sells get less

    ``gross_return_pct`` is the raw close-to-close return between execution
    bars (no fees, no slippage). ``net_return_pct`` and ``pnl`` come from
    the strategy equity curve and reflect everything the engine deducts.
    ``fees_paid`` only includes exchange fees — slippage is implicit in the
    entry / exit prices and shows up in the gross-vs-net spread.

    Open positions at the end of the series are excluded by default. With
    ``include_open=True`` they are returned with ``is_open=True`` (an extra
    column) and marked-to-market at the last bar.
    """
    positions = result.positions
    if positions.empty:
        cols = TRADE_COLUMNS + (["is_open"] if include_open else [])
        return pd.DataFrame(columns=cols)

    close = candles["close"]
    equity = result.equity
    fee_rate = result.fee_rate
    slippage_rate = result.slippage_rate
    initial_capital = result.initial_capital

    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])

    entries, exits = execution_bars(positions)
    n_bars = len(positions)

    rows = []
    for i, entry_idx in enumerate(entries):
        is_open = i >= len(exits)
        if is_open and not include_open:
            continue

        exit_idx = exits[i] if not is_open else n_bars - 1

        raw_entry_close = float(close.iloc[entry_idx])
        raw_exit_close = float(close.iloc[exit_idx])

        exec_entry_price = raw_entry_close * (1 + slippage_rate)
        exec_exit_price = raw_exit_close * (1 - slippage_rate)

        gross_return = raw_exit_close / raw_entry_close - 1

        prior_equity = (
            float(equity.iloc[entry_idx - 1])
            if entry_idx > 0
            else initial_capital
        )
        final_equity = float(equity.iloc[exit_idx])
        pnl = final_equity - prior_equity
        net_return = pnl / prior_equity if prior_equity > 0 else 0.0

        entry_fee = float(turnover.iloc[entry_idx] * fee_rate * prior_equity)
        if is_open:
            exit_fee = 0.0
        else:
            exit_prior_equity = (
                float(equity.iloc[exit_idx - 1])
                if exit_idx > 0
                else initial_capital
            )
            exit_fee = float(turnover.iloc[exit_idx] * fee_rate * exit_prior_equity)
        fees_paid = entry_fee + exit_fee

        # The position is actually held during bars [entry_idx, exit_idx-1]
        # for a closed trade; for an open trade it carries through to the
        # final bar (inclusive). Either way the count is the number of bars
        # we sat in the market.
        holding_period = (
            (n_bars - entry_idx) if is_open else (exit_idx - entry_idx)
        )

        row = {
            "entry_time": positions.index[entry_idx],
            "entry_price": exec_entry_price,
            "exit_time": positions.index[exit_idx],
            "exit_price": exec_exit_price,
            "gross_return_pct": gross_return,
            "net_return_pct": net_return,
            "fees_paid": fees_paid,
            "holding_period": holding_period,
            "pnl": pnl,
        }
        if include_open:
            row["is_open"] = is_open
        rows.append(row)

    cols = TRADE_COLUMNS + (["is_open"] if include_open else [])
    return pd.DataFrame(rows, columns=cols)


def write_trades_csv(
    result: BacktestResult,
    candles: pd.DataFrame,
    path: Path | str,
    include_open: bool = False,
) -> Path:
    """Write trades to ``path`` as CSV, returning the resolved path."""
    df = trades_to_dataframe(result, candles, include_open=include_open)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
