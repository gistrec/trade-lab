"""Rate-limited CoinGecko free-tier fetcher with parquet caching.

CoinGecko's `coins/{id}/market_chart` endpoint returns three aligned
time series — daily price, market cap, and 24h volume in USD — going
back to the coin's listing on CoinGecko. We use it to build the
point-in-time universe data we cannot get from Binance directly
(historical market cap + survivability info for delisted pairs).

The free tier rate limit is officially 30 req/min for the public API
but in practice we see throttling at ~10-15 req/min. ``DEFAULT_PAUSE_SECONDS``
keeps us safely below that with no API key required.

This module is read-only — it never modifies CoinGecko state — so all
the risk is in (a) rate limiting and (b) data integrity, both of which
we handle below.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFAULT_PAUSE_SECONDS = 12.0   # ~5 req/min — well under the free-tier ceiling.
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4


class CoinGeckoError(RuntimeError):
    """Raised for unrecoverable CoinGecko failures (404, persistent 5xx)."""


def fetch_market_chart(
    coin_id: str,
    *,
    vs_currency: str = "usd",
    days: int | str = "max",
    pause: float = DEFAULT_PAUSE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch CoinGecko market chart for ``coin_id``.

    Returns a DataFrame indexed by UTC midnight timestamps with columns
    ``price``, ``market_cap`` and ``volume_usd``. The three CoinGecko
    series are aligned to the same timestamp grid by the API, so no
    join is needed.

    The function sleeps ``pause`` seconds after the request as a courtesy
    to the API (so a caller doing many fetches stays under the rate
    limit). On HTTP 429 (rate-limited) the function uses exponential
    backoff up to ``max_retries``.
    """
    url = (
        f"{COINGECKO_BASE}/coins/{urllib.parse.quote(coin_id)}/market_chart"
        f"?vs_currency={urllib.parse.quote(vs_currency)}&days={days}"
    )
    payload = _http_get_json(url, timeout=timeout, max_retries=max_retries)

    prices = payload.get("prices") or []
    market_caps = payload.get("market_caps") or []
    volumes = payload.get("total_volumes") or []
    if not prices:
        raise CoinGeckoError(f"CoinGecko returned no prices for {coin_id!r}")

    df = pd.DataFrame(
        {
            "price": [p[1] for p in prices],
            "market_cap": [m[1] for m in market_caps] if market_caps else [None] * len(prices),
            "volume_usd": [v[1] for v in volumes] if volumes else [None] * len(prices),
        },
        index=pd.to_datetime([p[0] for p in prices], unit="ms", utc=True),
    )
    df.index.name = "timestamp"
    # CoinGecko returns daily snapshots stamped at midnight UTC for long
    # histories. Drop any subdaily duplicates from boundary periods.
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Daily resampling: keep the latest observation per UTC day so the
    # output index aligns one-to-one with OHLCV bars.
    df = df.resample("1D").last().dropna(how="all")
    df.index.name = "timestamp"

    if pause > 0:
        time.sleep(pause)
    return df


def fetch_market_chart_cached(
    coin_id: str,
    cache_dir: Path | str,
    *,
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Return :func:`fetch_market_chart` results from a parquet cache.

    The cache key is ``{cache_dir}/coingecko_{coin_id}.parquet``. Pass
    ``refresh=True`` to force a re-fetch (e.g. for incremental updates
    near the end of the series).
    """
    cache_path = Path(cache_dir) / f"coingecko_{coin_id}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    df = fetch_market_chart(coin_id, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _http_get_json(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """GET ``url`` and decode JSON, with exponential backoff on 429 / 5xx.

    Raises :class:`CoinGeckoError` on persistent failure. Non-HTTP errors
    (network down, DNS failure) bubble up as-is; the caller can catch
    them with ``urllib.error.URLError``.
    """
    backoff = 2.0
    last_status: Optional[int] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, headers={"User-Agent": "trade-lab/0.1 (research)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            if exc.code == 429 or 500 <= exc.code < 600:
                time.sleep(backoff)
                backoff *= 2
                continue
            if exc.code == 404:
                raise CoinGeckoError(f"CoinGecko 404 for {url}") from exc
            raise
    raise CoinGeckoError(
        f"CoinGecko request failed after {max_retries} retries "
        f"(last HTTP status: {last_status}): {url}"
    )
