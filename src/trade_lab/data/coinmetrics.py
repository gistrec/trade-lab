"""Coin Metrics community API fetcher with parquet caching.

The Coin Metrics community endpoint (https://community-api.coinmetrics.io)
serves a curated subset of metrics with no authentication required and
no per-day cap on the date range. The three metrics we need for the
PIT universe builder all sit in that subset:

* ``PriceUSD`` — daily closing price in USD.
* ``CapMrktCurUSD`` — current circulating market cap in USD.
* ``volume_reported_spot_usd_1d`` — daily reported spot volume in USD.

Compared to CoinGecko free, the upside is that ``days`` is unlimited
(history goes back to each asset's inception). The downside is that
the asset namespace is narrower than CoinGecko's. We rely on
``CoinMeta.coin_metrics_id`` in the registry to bridge.

The community API has no documented rate limit but is shared
infrastructure — we still sleep a courtesy interval between calls.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

COIN_METRICS_BASE = "https://community-api.coinmetrics.io/v4"
DEFAULT_PAUSE_SECONDS = 1.5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4

DEFAULT_METRICS = (
    "PriceUSD",
    "CapMrktCurUSD",
    "volume_reported_spot_usd_1d",
)
# Fallback metric set for assets whose "Cur" market-cap and "PriceUSD"
# are gated behind the paid tier. CapMrktEstUSD is community-accessible
# for nearly every asset; multi-day comparison on BTC shows the two
# cap metrics agree to ~0.005%, so swapping them is harmless for
# ranking.
FALLBACK_METRICS = ("CapMrktEstUSD", "volume_reported_spot_usd_1d")


class CoinMetricsError(RuntimeError):
    """Raised on unrecoverable Coin Metrics failures (404, persistent 5xx)."""


def fetch_asset_metrics(
    asset_id: str,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    start_time: str = "2017-01-01",
    end_time: Optional[str] = None,
    pause: float = DEFAULT_PAUSE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch ``metrics`` for ``asset_id`` at daily frequency.

    Returns a DataFrame indexed by UTC daily timestamps with one column
    per requested metric. Renames the columns to ``price``,
    ``market_cap`` and ``volume_usd`` so the result is a drop-in for
    the previous CoinGecko-shaped panel.
    """
    metric_list = list(metrics)
    base_params = {
        "assets": asset_id,
        "metrics": ",".join(metric_list),
        "start_time": start_time,
        "frequency": "1d",
        "page_size": 10000,
    }
    if end_time:
        base_params["end_time"] = end_time

    rows: list[dict] = []
    next_page_token: Optional[str] = None
    while True:
        params = dict(base_params)
        if next_page_token:
            params["next_page_token"] = next_page_token
        url = f"{COIN_METRICS_BASE}/timeseries/asset-metrics?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, timeout=timeout, max_retries=max_retries)
        rows.extend(payload.get("data") or [])
        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break
        if pause > 0:
            time.sleep(pause)
    if not rows:
        raise CoinMetricsError(f"Coin Metrics returned no rows for {asset_id!r}")

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        raise CoinMetricsError(f"Coin Metrics response missing 'time' for {asset_id!r}")
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("timestamp").drop(columns=["asset", "time"], errors="ignore")
    # Coin Metrics returns metric values as strings (preserving precision);
    # we cast back to float — the loss of precision is irrelevant for
    # daily ranking work.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rename_map = {
        "PriceUSD": "price",
        "CapMrktCurUSD": "market_cap",
        "CapMrktEstUSD": "market_cap",   # fallback collapses to the same field
        "volume_reported_spot_usd_1d": "volume_usd",
    }
    df = df.rename(columns=rename_map)
    # If both CapMrktCurUSD and CapMrktEstUSD were present (a single call
    # carrying both), the second rename would create duplicate columns —
    # collapse them by taking the first non-NaN.
    if df.columns.duplicated().any():
        # groupby(axis=1) was removed in pandas 3.0; transpose-groupby
        # is the portable spelling of "first non-NaN per column name".
        df = df.T.groupby(level=0).first().T
    # Keep only the three expected columns where present.
    keep = [c for c in ("price", "market_cap", "volume_usd") if c in df.columns]
    df = df[keep].sort_index()

    if pause > 0:
        time.sleep(pause)
    return df


def fetch_asset_metrics_with_fallback(
    asset_id: str,
    **kwargs,
) -> pd.DataFrame:
    """Wrap :func:`fetch_asset_metrics` with the documented metric fallback.

    Tries the default (Cur + Price + Volume) metric set first; if Coin
    Metrics returns 403 for any of them (community-tier gating on
    post-2020 alts), falls back to (Est + Volume). The fallback drops
    PriceUSD because every coin where it was gated also lacks it under
    a different name in community.
    """
    try:
        return fetch_asset_metrics(asset_id, metrics=DEFAULT_METRICS, **kwargs)
    except CoinMetricsError:
        return fetch_asset_metrics(asset_id, metrics=FALLBACK_METRICS, **kwargs)


def fetch_asset_metrics_cached(
    asset_id: str,
    cache_dir: Path | str,
    *,
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Cache wrapper for :func:`fetch_asset_metrics_with_fallback`."""
    cache_path = Path(cache_dir) / f"coinmetrics_{asset_id}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    df = fetch_asset_metrics_with_fallback(asset_id, **kwargs)
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
    """GET ``url`` and decode JSON, with backoff on 429 / 5xx."""
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
            if exc.code in (400, 403, 404):
                # 400 == asset id not recognised at this frequency.
                # 403 == community-tier doesn't carry one of the
                # requested metrics. 404 == not in catalog at all.
                # All three are "skip this coin" outcomes upstream.
                raise CoinMetricsError(
                    f"Coin Metrics HTTP {exc.code} for {url}"
                ) from exc
            raise
    raise CoinMetricsError(
        f"Coin Metrics request failed after {max_retries} retries "
        f"(last HTTP status: {last_status}): {url}"
    )
