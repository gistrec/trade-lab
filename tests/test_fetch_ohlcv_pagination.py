"""Pagination behaviour of the historical OHLCV fetcher.

An exchange's server-side page cap can be smaller than the requested
``limit``; every page then comes back "short". Short pages must NOT
stop pagination — that silently truncated history to a single page.
"""
from __future__ import annotations

import ccxt

from trade_lab.data.fetch_ohlcv import fetch_ohlcv


_DAY_MS = 86_400_000


class _PagedExchange:
    """Serves 6 daily candles in pages of at most 2, regardless of limit."""

    rateLimit = 0
    PAGE_CAP = 2
    N_CANDLES = 6
    START_MS = 1_600_000_000_000

    def __init__(self, params=None):
        self.fetch_calls = 0

    def _all_rows(self):
        return [
            [self.START_MS + i * _DAY_MS, 1.0, 1.0, 1.0, 1.0, 1.0]
            for i in range(self.N_CANDLES)
        ]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.fetch_calls += 1
        rows = self._all_rows()
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[: min(limit, self.PAGE_CAP)]


def test_short_pages_do_not_truncate_history(monkeypatch):
    instances = []

    def factory(params=None):
        exch = _PagedExchange(params)
        instances.append(exch)
        return exch

    monkeypatch.setattr(ccxt, "pagedexchange", factory, raising=False)

    from datetime import datetime, timezone

    df = fetch_ohlcv(
        "pagedexchange", "BTC/USDT", timeframe="1d",
        since=datetime(2020, 9, 13, tzinfo=timezone.utc),
        limit=1000,
    )
    # All 6 candles arrive even though every page was "short" (2 < 1000).
    assert len(df) == _PagedExchange.N_CANDLES
    assert instances[0].fetch_calls >= 3


def test_tz_aware_until_does_not_crash_and_trims(monkeypatch):
    """A timezone-aware `until` (produced by CLI --until with an offset,
    or datetime.now(timezone.utc)) must trim, not crash:
    pd.Timestamp(until, tz="UTC") raises ValueError on a tz-aware datetime
    (regression: C2). Trimming reuses the already-computed until_ms."""
    def factory(params=None):
        return _PagedExchange(params)

    monkeypatch.setattr(ccxt, "pagedexchange", factory, raising=False)

    from datetime import datetime, timezone

    # until at the 3rd candle (START + 2 days), tz-AWARE.
    until_dt = datetime.fromtimestamp(
        (_PagedExchange.START_MS + 2 * _DAY_MS) / 1000, tz=timezone.utc,
    )
    df = fetch_ohlcv(
        "pagedexchange", "BTC/USDT", timeframe="1d",
        since=datetime(2020, 9, 13, tzinfo=timezone.utc),
        until=until_dt,
        limit=1000,
    )
    # Must not raise; trimmed to candles at or before `until` (indices 0,1,2).
    assert len(df) == 3
