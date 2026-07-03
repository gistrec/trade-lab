"""Fetch historical OHLCV candles from a ccxt exchange."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` doesn't have the canonical OHLCV shape.

    Required: index named ``timestamp`` of UTC datetimes, columns
    ``open``, ``high``, ``low``, ``close``, ``volume`` in that order.
    """
    if list(df.columns) != _OHLCV_COLUMNS:
        raise ValueError(
            f"Expected columns {_OHLCV_COLUMNS}, got {list(df.columns)}"
        )
    if df.index.name != "timestamp":
        raise ValueError(
            f"Expected index name 'timestamp', got {df.index.name!r}"
        )


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV candles, paginating until ``until`` (or the exchange runs out).

    Parameters
    ----------
    exchange_id
        ccxt exchange id (e.g. ``"binance"``).
    symbol
        Symbol in ccxt format (e.g. ``"BTC/USDT"``).
    timeframe
        Candle timeframe (e.g. ``"1m"``, ``"1h"``, ``"1d"``).
    since
        Optional start datetime; naive values are treated as UTC.
    until
        Optional end datetime; naive values are treated as UTC.
    limit
        Page size used in each ccxt call.

    Returns
    -------
    DataFrame indexed by UTC timestamp with columns ``open``, ``high``,
    ``low``, ``close``, ``volume``.
    """
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown ccxt exchange id: {exchange_id!r}")
    exchange = exchange_cls({"enableRateLimit": True})

    since_ms = _to_ms(since) if since else None
    until_ms = _to_ms(until) if until else None

    rows: list[list] = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        # Stop when the page didn't advance or when we passed ``until``.
        # A short page is NOT a stop condition: exchanges cap the page
        # size server-side, so every page can be shorter than ``limit``
        # — breaking on it silently truncated history to one page. End
        # of data shows up as an empty next page instead (one extra
        # request, never missing candles).
        if since_ms is not None and last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        if until_ms is not None and last_ts >= until_ms:
            break
        time.sleep(getattr(exchange, "rateLimit", 1000) / 1000.0)

    df = pd.DataFrame(rows, columns=["timestamp", *_OHLCV_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if until_ms is not None and not df.empty:
        # Reuse the already tz-correct until_ms: pd.Timestamp(until, tz="UTC")
        # raises ValueError when `until` is tz-aware (e.g. CLI --until with an
        # offset, or datetime.now(timezone.utc)).
        df = df[df.index <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    validate_ohlcv(df)
    return df


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
