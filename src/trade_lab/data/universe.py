"""Point-in-time universe construction for cross-sectional strategies.

Given the curated :mod:`.coin_registry` plus CoinGecko historical data,
``build_pit_universe`` returns an eligibility mask: a DataFrame whose
``True`` cells mark "coin X was in the universe on date D".

A coin is in the universe on date D iff **all** of the following hold:

1. ``tradable_at(D, meta)`` — Binance had this pair listed at D.
2. Market cap rank at D <= ``top_n`` (lower rank = bigger market cap).
3. Trailing ``volume_lookback_days`` median USD volume rank <= ``top_n``.
4. (Optional) Not in the stablecoin denylist — stablecoins distort
   momentum rankings because they have no real momentum signal.

The composite (1 AND 2 AND 3) follows the Research-Claude recipe
("top 20 by market cap and 90-day median volume") exactly.

**Important caveat the function cannot fix:** CoinGecko market cap is
global, and CoinGecko USD volume is summed across all exchanges. For
backtests that simulate Binance fills, that means we are ranking by
broader liquidity than what was actually on Binance. The Binance-only
volume series is available for *currently-listed* pairs (via existing
OHLCV parquets), but missing for delisted pairs — using two different
metrics for living and dead coins would itself be a leak. We accept
the global proxy and flag this in ``docs/results/pit_universe.md``.
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .coin_registry import COIN_REGISTRY, CoinMeta, stablecoins, tradable_at
from .coinmetrics import CoinMetricsError, fetch_asset_metrics_cached


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------


def load_panel(
    candidates: Optional[Mapping[str, CoinMeta]] = None,
    cache_dir: Path | str = "data/coinmetrics",
    *,
    refresh: bool = False,
    fetch: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (price, market_cap, volume_usd) panels for ``candidates``.

    Each returned DataFrame is indexed by UTC daily timestamps and has
    one column per coin symbol (the registry key, not the Coin Metrics
    id). Missing dates are simply absent in the index for that coin —
    the caller is expected to align on a common index.

    Set ``fetch=False`` to only use the on-disk cache; missing coins
    are silently dropped. This is the safe option for tests and CI.
    """
    pool = candidates or COIN_REGISTRY
    price_frames: list[pd.Series] = []
    mc_frames: list[pd.Series] = []
    vol_frames: list[pd.Series] = []
    for symbol, meta in pool.items():
        try:
            if fetch:
                df = fetch_asset_metrics_cached(
                    meta.cm_id, cache_dir=cache_dir, refresh=refresh
                )
            else:
                cache_path = Path(cache_dir) / f"coinmetrics_{meta.cm_id}.parquet"
                if not cache_path.exists():
                    continue
                df = pd.read_parquet(cache_path)
        except CoinMetricsError:
            # Asset id not in Coin Metrics community catalog. Skip — the
            # caller will see the missing column.
            continue
        # Some assets miss a metric (e.g. brand-new listings have no
        # 90-day volume yet); fill with NaN columns so the concat shapes
        # remain consistent.
        for col in ("price", "market_cap", "volume_usd"):
            if col not in df.columns:
                df = df.assign(**{col: pd.NA})
        price_frames.append(df["price"].rename(symbol))
        mc_frames.append(df["market_cap"].rename(symbol))
        vol_frames.append(df["volume_usd"].rename(symbol))

    if not price_frames:
        empty = pd.DataFrame()
        return empty, empty, empty

    prices = pd.concat(price_frames, axis=1).sort_index()
    market_caps = pd.concat(mc_frames, axis=1).sort_index()
    volumes = pd.concat(vol_frames, axis=1).sort_index()
    return prices, market_caps, volumes


def build_pit_universe(
    market_caps: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    candidates: Optional[Mapping[str, CoinMeta]] = None,
    top_n: int = 20,
    volume_lookback_days: int = 90,
    exclude_stablecoins: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Compute the eligibility mask.

    ``market_caps`` and ``volumes`` come from :func:`load_panel`. The
    function aligns them on a common index and returns one boolean
    column per coin.
    """
    pool = candidates or COIN_REGISTRY
    if market_caps.empty:
        return pd.DataFrame()

    # Restrict to the panel coins.
    columns = [c for c in market_caps.columns if c in pool]
    market_caps = market_caps[columns]
    volumes = volumes.reindex(market_caps.index)[columns]

    # Trailing N-day median volume. Median (not mean) so a single
    # exchange-wide spike doesn't make a coin look more liquid than it
    # really is. Skip-NaN means a coin that just listed gets ranked
    # against its first few observations.
    rolling_vol = volumes.rolling(volume_lookback_days, min_periods=1).median()

    # Tradability mask: True where the Binance pair was listed at the date.
    tradable = pd.DataFrame(False, index=market_caps.index, columns=columns)
    for symbol in columns:
        meta = pool[symbol]
        for ts in market_caps.index:
            iso_date = ts.strftime("%Y-%m-%d")
            tradable.at[ts, symbol] = tradable_at(iso_date, meta)

    if exclude_stablecoins:
        stables = stablecoins()
        for symbol in columns:
            base = pool[symbol].base
            if base in stables:
                tradable[symbol] = False

    # Zero-out market cap / volume for non-tradable cells so the rank
    # can never pick them. This is the *correctness* step — we must
    # never pretend a delisted pair was tradable.
    mc_for_rank = market_caps.where(tradable, other=float("nan"))
    vol_for_rank = rolling_vol.where(tradable, other=float("nan"))

    # Rank descending (1 = biggest). ``na_option="bottom"`` keeps NaN
    # coins out of the top-N selection. Ranking is done per row.
    mc_rank = mc_for_rank.rank(axis=1, method="min", ascending=False, na_option="bottom")
    vol_rank = vol_for_rank.rank(axis=1, method="min", ascending=False, na_option="bottom")

    eligible = (mc_rank <= top_n) & (vol_rank <= top_n) & tradable

    if start_date:
        eligible = eligible[eligible.index >= pd.Timestamp(start_date, tz=eligible.index.tz)]
    if end_date:
        eligible = eligible[eligible.index <= pd.Timestamp(end_date, tz=eligible.index.tz)]

    return eligible


def closes_for_universe(
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    fallback: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return per-coin close prices, masked to dates the coin was eligible.

    Outside the eligibility windows the value is left as NaN — the
    cross-sectional runner must treat NaN as "not in universe" rather
    than "price is zero".

    ``fallback`` is an optional alternative panel (typically the
    market-cap series) used for coins where ``prices`` is missing
    entirely. The Coin Metrics community tier gates ``PriceUSD`` on
    several alts but exposes ``CapMrktEstUSD``; an empirical check on
    BTC shows ``pct_change`` of the two series agrees to ~6 bps per
    day, far below the 30-day momentum signal. The substitution affects
    only return computations — never absolute price levels — so it's
    safe for momentum but not for, e.g., dollar-volume slippage models.
    """
    aligned = prices.reindex(eligibility.index)[eligibility.columns]
    if fallback is not None:
        fb = fallback.reindex(eligibility.index)[eligibility.columns]
        # Fill column-by-column where the price series is entirely empty.
        for col in aligned.columns:
            if aligned[col].isna().all():
                aligned[col] = fb[col]
    return aligned.where(eligibility, other=float("nan"))


# ---------------------------------------------------------------------------
# Convenience: fetch the whole panel and build the universe in one call
# ---------------------------------------------------------------------------


def build_universe_from_registry(
    cache_dir: Path | str = "data/coinmetrics",
    *,
    top_n: int = 20,
    volume_lookback_days: int = 90,
    exclude_stablecoins: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    candidates: Optional[Mapping[str, CoinMeta]] = None,
    refresh: bool = False,
    fetch: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-shot helper: returns ``(eligibility, closes)``.

    Both DataFrames share the same index and column layout. Use
    ``fetch=False`` after the first run to avoid hitting CoinGecko.
    """
    prices, market_caps, volumes = load_panel(
        candidates=candidates,
        cache_dir=cache_dir,
        refresh=refresh,
        fetch=fetch,
    )
    eligibility = build_pit_universe(
        market_caps,
        volumes,
        candidates=candidates,
        top_n=top_n,
        volume_lookback_days=volume_lookback_days,
        exclude_stablecoins=exclude_stablecoins,
        start_date=start_date,
        end_date=end_date,
    )
    closes = closes_for_universe(prices, eligibility, fallback=market_caps)
    return eligibility, closes
