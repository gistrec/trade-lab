"""Synthetic crypto-market index built from an asset basket.

Han, Kang, Ryu (2024) — the most cost-realistic crypto trend paper in
the literature — argue that the strongest long-only crypto signal is
TSMOM on a **market basket**, not on individual assets. The basket
captures the macro trend; per-asset noise averages out; and (under
the paper's parameters) the signal is then transmitted via a single
long/cash decision rather than dozens of per-asset entries that each
pay their own slippage.

This module builds that basket. It is **deliberately not** a strict
point-in-time universe — it uses the hand-picked top-7 majors we
already have OHLCV for, so the index inherits the same survivor bias
as the rest of the project's single-timeframe results. The honest
upgrade path is to derive the basket from ``data/universe.py``'s PIT
mask; that is left for a follow-up.

The index series produced here is a synthetic ``close`` time series
that any single-asset Strategy can consume. The rebalance cost is
charged inside the index construction so callers see a "clean" series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketIndex:
    """The basket index series plus the per-asset weights that built it.

    ``index`` is the OHLCV-shaped synthetic close series a single-asset
    Strategy consumes. ``weights`` holds the *actual* per-asset weight
    held at each bar — flat ``1/N_active`` immediately after a rebalance,
    drifting with returns between rebalances, renormalised to sum to 1
    over active assets. Its index matches ``index``; its columns are the
    asset keys of ``asset_candles``.

    The live executor sizes to ``weights.loc[asof]`` so it replicates the
    backtest's *drifted* holdings, rather than forcing a flat-weight
    rebalance every daily cycle (which would add per-day turnover the
    monthly-rebalanced backtest never paid). See
    :func:`trade_lab.execution.allocator.compute_target_allocation`.
    """

    index: pd.DataFrame
    weights: pd.DataFrame


def build_crypto_market_index(
    asset_candles: Mapping[str, pd.DataFrame],
    *,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    rebalance_freq: str = "MS",
) -> pd.DataFrame:
    """Equal-weight market-basket index (OHLCV only).

    Thin wrapper over :func:`build_crypto_market_index_with_weights` for
    the many callers (backtest engine, harness, lookahead detector) that
    only consume the synthetic close series. Callers that must size to
    the basket's drifted per-asset weights — i.e. the live executor —
    use :func:`build_crypto_market_index_with_weights` directly.
    """
    return build_crypto_market_index_with_weights(
        asset_candles,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        rebalance_freq=rebalance_freq,
    ).index


def build_crypto_market_index_with_weights(
    asset_candles: Mapping[str, pd.DataFrame],
    *,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    rebalance_freq: str = "MS",
) -> MarketIndex:
    """Equal-weight market-basket index with periodic rebalancing.

    Algorithm:

    1. Outer-join the per-asset close series on a common UTC daily index.
    2. At every bar, the *target* weight per active asset is
       ``1 / N_active`` (dynamic equal-weight; assets pre-listing have
       NaN closes and are simply absent from N_active).
    3. Actual weights drift between rebalance dates from the target
       (because asset returns differ); on each rebalance date
       (`rebalance_freq`, default monthly), positions are reset to
       target.
    4. Rebalance turnover is charged ``fee_rate + slippage_rate`` per
       unit of |weight change|, mirroring the engine's standard cost
       model.
    5. The output is OHLCV-shaped (all four price columns hold the
       same index value) so any single-asset Strategy can consume it
       unchanged.

    Known cost-model residual (documented, not fixed here): on a rebalance
    bar the turnover in step 4 is measured as ``|new_target - current|``
    where ``current`` is the *pre-drift* weight carried from the previous
    bar — it is not marked to market through the rebalance bar's own
    return first. Live execution (which sizes to these weights and trades
    the constituents) marks to market before rebalancing, so its realised
    rebalance turnover differs from the index's charged turnover by one
    bar's drift. The gap is tiny and, more importantly, fixing it would
    alter historical index equity and therefore the published DSR — an
    ask-first change to the validated backtest, deliberately out of scope
    for the live-execution weight work.

    Returns a :class:`MarketIndex` whose ``index`` is a DataFrame indexed
    by ``timestamp`` with columns ``open, high, low, close, volume`` (all
    open/high/low = close; volume is 1.0 as a placeholder — no real
    volume aggregation is attempted here), and whose ``weights`` holds
    the per-asset drifted weight at every bar (see :class:`MarketIndex`).
    """
    if not asset_candles:
        return MarketIndex(_empty_index(), _empty_weights())

    closes = pd.concat(
        {k: v["close"].astype(float) for k, v in asset_candles.items()},
        axis=1,
    ).sort_index()
    if closes.empty:
        return MarketIndex(_empty_index(), _empty_weights())

    # Fail loud on data gaps. Leading NaN = asset not yet listed
    # (dynamic universe entry, by design). NaN *after* an asset's first
    # valid close — an interior gap, or a series that ends before the
    # others — would silently shrink N_active, force an unscheduled
    # rebalance, re-grant the first-active cost credit on reappearance,
    # and zero the price move across the gap. Hard rule: missing
    # candles raise, the basket never shrinks silently.
    seen = closes.notna().cummax()
    gaps = closes.isna() & seen
    if gaps.to_numpy().any():
        details = []
        for col in closes.columns[gaps.any(axis=0)]:
            ts = closes.index[gaps[col]]
            details.append(
                f"{col}: {len(ts)} missing bar(s) between {ts[0]} and {ts[-1]}"
            )
        raise ValueError(
            "Missing candles after listing — refusing to build the index "
            "on a silently shrunken basket: " + "; ".join(details)
        )

    asset_returns = closes.pct_change(fill_method=None)
    active_panel = closes.notna()
    n_active = active_panel.sum(axis=1)

    # Target weights = 1/N_active if active, else 0.
    target_weights = active_panel.div(
        n_active.where(n_active > 0, 1.0), axis=0
    )
    target_weights = target_weights.where(active_panel, 0.0)

    # Rebalance schedule: at every period start (e.g. month start) AND
    # every bar where N_active changes (new asset listing forces a
    # rebalance regardless of the schedule).
    rebalance_dates = pd.date_range(
        closes.index[0], closes.index[-1], freq=rebalance_freq, tz=closes.index.tz,
    )
    rebalance_mask = pd.Series(False, index=closes.index)
    for date in rebalance_dates:
        # Snap to the nearest available bar at or after the schedule date.
        candidates = closes.index[closes.index >= date]
        if not candidates.empty:
            rebalance_mask.at[candidates[0]] = True
    # Always rebalance when N_active changes (new asset comes online).
    rebalance_mask = rebalance_mask | n_active.diff().fillna(0).ne(0)
    rebalance_mask.iloc[0] = True   # initial allocation

    # Walk the bars: between rebalances, weights drift with returns.
    bars = list(closes.index)
    n_assets = closes.shape[1]
    weights = np.zeros((len(bars), n_assets), dtype=float)
    cost_per_bar = np.zeros(len(bars), dtype=float)
    target_arr = target_weights.to_numpy()
    returns_arr = asset_returns.fillna(0.0).to_numpy()

    current = np.zeros(n_assets, dtype=float)
    for i, ts in enumerate(bars):
        if rebalance_mask.iloc[i]:
            new_target = target_arr[i]
            # Rebalance turnover: |new - current| summed.
            turnover = np.abs(new_target - current).sum()
            # First-active credit: on the first non-zero bar of any
            # asset, that asset's full target weight was paid by
            # whoever initially deployed capital into it — don't
            # double-charge at the index level. (Matches
            # ensemble.py's logic.)
            if i == 0:
                first_active_credit = new_target.sum()
            else:
                # Subtract only the assets newly coming online at THIS bar
                newly_active = (current == 0) & (new_target > 0)
                first_active_credit = new_target[newly_active].sum()
            charged_turnover = max(turnover - first_active_credit, 0.0)
            cost_per_bar[i] = charged_turnover * (fee_rate + slippage_rate)
            current = new_target
        else:
            # No rebalance: weights drift with this bar's returns then
            # are renormalised to keep the basket fully invested.
            # (Standard "let it ride" between rebalances.)
            grown = current * (1.0 + returns_arr[i])
            total = grown.sum()
            current = grown / total if total > 0 else current
        weights[i] = current

    # Portfolio per-bar return = weights at t-1 applied to returns at t,
    # minus the rebalance cost at t.
    weights_df = pd.DataFrame(weights, index=closes.index, columns=closes.columns)
    shifted = weights_df.shift(1).fillna(0.0)
    portfolio_returns = (shifted * asset_returns.fillna(0.0)).sum(axis=1) - cost_per_bar
    portfolio_equity = initial_capital * (1.0 + portfolio_returns).cumprod()

    # Rescale to start at 100 (standard index convention).
    index_close = 100.0 * portfolio_equity / portfolio_equity.iloc[0]

    out = pd.DataFrame(
        {
            "open": index_close,
            "high": index_close,
            "low": index_close,
            "close": index_close,
            "volume": 1.0,
        },
        index=closes.index,
    )
    out.index.name = "timestamp"
    return MarketIndex(index=out, weights=weights_df)


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLUMNS).rename_axis("timestamp")


def _empty_weights() -> pd.DataFrame:
    return pd.DataFrame().rename_axis("timestamp")
