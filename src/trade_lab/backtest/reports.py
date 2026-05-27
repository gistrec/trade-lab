"""Tabular exports of backtest results (trade lists, CSV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import BacktestResult


TRADE_COLUMNS = [
    "entry_time",
    "entry_signal_time",
    "entry_execution_price",
    "exit_time",
    "exit_signal_time",
    "exit_execution_price",
    "gross_return_pct",
    "net_return_pct",
    "fees_paid",
    "slippage_cost_estimate",
    "holding_period",
    "pnl",
]


def trades_to_dataframe(
    result: BacktestResult,
    candles: pd.DataFrame | None = None,
    include_open: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame describing every trade in ``result``.

    Columns map directly onto :class:`~trade_lab.backtest.engine.Trade`
    fields. ``entry_time`` / ``exit_time`` are execution candles;
    ``entry_signal_time`` / ``exit_signal_time`` are the bars where the
    decision was made (one bar earlier). ``entry_execution_price`` and
    ``exit_execution_price`` are slippage-adjusted (``close * (1 ± rate)``).

    The ``candles`` parameter is kept for API compatibility but no longer
    needed — all the data comes from ``result.trades``.

    Open positions at the end of the series are excluded by default. With
    ``include_open=True`` they are returned with ``is_open=True`` (an extra
    column).
    """
    del candles  # kept for signature compatibility; no longer needed

    cols = TRADE_COLUMNS + (["is_open"] if include_open else [])
    rows = []
    for trade in result.trades:
        is_open = trade.exit_signal_time is None
        if is_open and not include_open:
            continue
        row = {
            "entry_time": trade.entry_time,
            "entry_signal_time": trade.entry_signal_time,
            "entry_execution_price": trade.entry_execution_price,
            "exit_time": trade.exit_time,
            "exit_signal_time": trade.exit_signal_time,
            "exit_execution_price": trade.exit_execution_price,
            "gross_return_pct": trade.gross_return_pct,
            "net_return_pct": trade.net_return_pct,
            "fees_paid": trade.fees_paid,
            "slippage_cost_estimate": trade.slippage_cost_estimate,
            "holding_period": trade.bars_held,
            "pnl": trade.pnl,
        }
        if include_open:
            row["is_open"] = is_open
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def write_trades_csv(
    result: BacktestResult,
    candles: pd.DataFrame | None,
    path: Path | str,
    include_open: bool = False,
) -> Path:
    """Write trades to ``path`` as CSV, returning the resolved path."""
    df = trades_to_dataframe(result, candles, include_open=include_open)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
