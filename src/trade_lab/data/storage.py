"""Local OHLCV storage as Parquet files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _filename(exchange: str, symbol: str, timeframe: str) -> str:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return f"{exchange}_{safe_symbol}_{timeframe}.parquet"


def candles_path(data_dir: Path | str, exchange: str, symbol: str, timeframe: str) -> Path:
    """Return the on-disk path for a (exchange, symbol, timeframe) tuple."""
    return Path(data_dir) / _filename(exchange, symbol, timeframe)


def save_candles(
    df: pd.DataFrame,
    data_dir: Path | str,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """Persist a candles DataFrame to Parquet under ``data_dir``."""
    path = candles_path(data_dir, exchange, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_candles(
    data_dir: Path | str,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Load a previously persisted candles DataFrame."""
    path = candles_path(data_dir, exchange, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No candles file at {path}. Run `trade-lab fetch` first."
        )
    return pd.read_parquet(path)


def filter_candles_by_date(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Slice ``df`` to bars whose timestamp falls inside the given range.

    Both bounds are inclusive at the day level: ``end_date="2024-06-30"``
    keeps every bar through 2024-06-30 23:59:59 (i.e. the full day). Dates
    are parsed with :func:`pandas.Timestamp` and localized to match the
    index's timezone if needed, so a tz-naive bound works against a UTC
    index and vice versa.

    Passing both bounds as ``None`` returns ``df`` unchanged.
    """
    if not start_date and not end_date:
        return df

    idx_tz = df.index.tz

    def _coerce(value: str) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if idx_tz is not None and ts.tz is None:
            ts = ts.tz_localize(idx_tz)
        return ts

    out = df
    if start_date:
        out = out[out.index >= _coerce(start_date)]
    if end_date:
        # Inclusive end-of-day: shift the bound forward by one day and use <.
        out = out[out.index < _coerce(end_date) + pd.Timedelta(days=1)]
    return out
