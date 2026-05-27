"""Tabular exports of backtest results (trade lists, CSV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..strategies.base import Strategy
from ..strategies.regime_sma_cross import RegimeSMACrossStrategy
from ..strategies.sma_cross import SMACrossStrategy
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


DEBUG_TRADE_COLUMNS = [
    "signal_time",
    "execution_time",
    "signal_close",
    "execution_open_or_close",
    "entry_price_after_slippage",
    "exit_price_after_slippage",
    "reason",
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


def debug_trades_dataframe(
    result: BacktestResult,
    candles: pd.DataFrame,
    strategy: Strategy | None = None,
    limit: int = 10,
    include_open: bool = False,
) -> pd.DataFrame:
    """Export the first ``limit`` trades with signal-vs-execution timing.

    Designed for manual look-ahead audits: every row makes the gap between
    *when the strategy decided* (``signal_time``, ``signal_close``) and
    *when the trade actually filled* (``execution_time``,
    ``execution_open_or_close``, ``entry_price_after_slippage``) explicit.
    The ``reason`` column dumps the strategy's relevant indicator values at
    the signal bar so you can hand-verify that the decision used only that
    bar's data.
    """
    close = candles["close"]
    indicators = _strategy_indicators(strategy, candles)

    trades_iter = (
        result.trades
        if include_open
        else [t for t in result.trades if t.exit_signal_time is not None]
    )

    rows: list[dict] = []
    for trade in trades_iter[:limit]:
        signal_idx = candles.index.get_loc(trade.entry_signal_time)
        exec_idx = candles.index.get_loc(trade.entry_time)
        rows.append(
            {
                "signal_time": trade.entry_signal_time,
                "execution_time": trade.entry_time,
                "signal_close": float(close.iloc[signal_idx]),
                "execution_open_or_close": float(close.iloc[exec_idx]),
                "entry_price_after_slippage": trade.entry_execution_price,
                "exit_price_after_slippage": trade.exit_execution_price,
                "reason": _reason_at(indicators, signal_idx),
            }
        )

    return pd.DataFrame(rows, columns=DEBUG_TRADE_COLUMNS)


def _strategy_indicators(
    strategy: Strategy | None, candles: pd.DataFrame
) -> dict | None:
    """Pre-compute indicators a strategy uses, for the audit's ``reason``
    column. Returns ``None`` for unrecognised strategies — the reason then
    falls back to a generic "signal flipped long" string."""
    if isinstance(strategy, RegimeSMACrossStrategy):
        close = candles["close"]
        return {
            "kind": "regime_sma_cross",
            "fast": close.rolling(strategy.fast_period).mean(),
            "slow": close.rolling(strategy.slow_period).mean(),
            "regime": close.rolling(strategy.regime_period).mean(),
            "close": close,
        }
    if isinstance(strategy, SMACrossStrategy):
        close = candles["close"]
        return {
            "kind": "sma_cross",
            "fast": close.rolling(strategy.fast_period).mean(),
            "slow": close.rolling(strategy.slow_period).mean(),
        }
    return None


def _reason_at(indicators: dict | None, signal_idx: int) -> str:
    if indicators is None:
        return "signal flipped long"
    if indicators["kind"] == "sma_cross":
        f = float(indicators["fast"].iloc[signal_idx])
        s = float(indicators["slow"].iloc[signal_idx])
        return f"fast({f:.2f}) > slow({s:.2f})"
    if indicators["kind"] == "regime_sma_cross":
        f = float(indicators["fast"].iloc[signal_idx])
        s = float(indicators["slow"].iloc[signal_idx])
        r = float(indicators["regime"].iloc[signal_idx])
        c = float(indicators["close"].iloc[signal_idx])
        return (
            f"fast({f:.2f})>slow({s:.2f}) & close({c:.2f})>regime({r:.2f})"
        )
    return "signal flipped long"


def write_debug_trades_csv(
    result: BacktestResult,
    candles: pd.DataFrame,
    path: Path | str,
    strategy: Strategy | None = None,
    limit: int = 10,
) -> Path:
    """Write the audit-friendly first-N-trades CSV to ``path``."""
    df = debug_trades_dataframe(result, candles, strategy=strategy, limit=limit)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
