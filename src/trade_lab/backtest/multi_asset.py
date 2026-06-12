"""Multi-asset fixed-strategy yearly validation.

Runs the per-calendar-year validator from :mod:`.yearly` across several
assets so the same fixed strategies can be compared side-by-side without
any optimization. The goal is to test whether behaviour observed on one
symbol generalizes to others.
"""
from __future__ import annotations

import logging

from typing import Mapping, Sequence

import pandas as pd

from .yearly import (
    DEFAULT_FIXED_STRATEGIES,
    FixedStrategySpec,
    YEARLY_COLUMNS,
    run_yearly_validation,
)


MULTI_ASSET_DETAIL_COLUMNS = ["asset", *YEARLY_COLUMNS]

MULTI_ASSET_AGGREGATE_COLUMNS = [
    "asset",
    "strategy",
    "total_years",
    "avg_annual_return",
    "median_annual_return",
    "best_year_return",
    "worst_year_return",
    "years_outperforming_bh",
    "years_lower_dd_than_bh",
    "avg_exposure",
]

MULTI_ASSET_SUMMARY_COLUMNS = [
    "strategy",
    "n_assets",
    "avg_return_across_assets",
    "avg_worst_year",
    "total_years_outperforming_bh",
    "total_years_lower_dd",
    "avg_exposure_across_assets",
]

logger = logging.getLogger(__name__)



def run_multi_asset_yearly_validation(
    asset_candles: Mapping[str, pd.DataFrame],
    strategies: Sequence[FixedStrategySpec] = DEFAULT_FIXED_STRATEGIES,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Run the per-year fixed-strategy validation across every asset.

    ``asset_candles`` maps a symbol label (e.g. ``"BTC/USDT"``) to a
    candles DataFrame. Each asset is evaluated independently; the result
    is a single long-format DataFrame with an ``asset`` column prepended.
    Assets with empty candle frames are skipped with a warning.
    """
    frames: list[pd.DataFrame] = []
    for asset, candles in asset_candles.items():
        if candles.empty:
            logger.warning(
                "multi-asset validation: skipping %r — empty candle frame; "
                "results cover fewer assets than requested.", asset,
            )
            continue
        per_asset = run_yearly_validation(
            candles,
            strategies=strategies,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
        )
        if per_asset.empty:
            continue
        per_asset = per_asset.copy()
        per_asset.insert(0, "asset", asset)
        frames.append(per_asset)

    if not frames:
        return pd.DataFrame(columns=MULTI_ASSET_DETAIL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def aggregate_multi_asset(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(asset, strategy) summary: avg / median / best / worst, B&H counts."""
    if detail_df.empty:
        return pd.DataFrame(columns=MULTI_ASSET_AGGREGATE_COLUMNS)

    rows: list[dict] = []
    for (asset, strategy), group in detail_df.groupby(
        ["asset", "strategy"], sort=False
    ):
        bh_wins = int(
            (group["return_pct"] > group["buy_and_hold_return_pct"]).sum()
        )
        lower_dd = int(
            (group["max_drawdown_pct"] < group["buy_and_hold_max_drawdown_pct"]).sum()
        )
        rows.append(
            {
                "asset": asset,
                "strategy": strategy,
                "total_years": len(group),
                "avg_annual_return": float(group["return_pct"].mean()),
                "median_annual_return": float(group["return_pct"].median()),
                "best_year_return": float(group["return_pct"].max()),
                "worst_year_return": float(group["return_pct"].min()),
                "years_outperforming_bh": bh_wins,
                "years_lower_dd_than_bh": lower_dd,
                "avg_exposure": float(group["exposure_pct"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=MULTI_ASSET_AGGREGATE_COLUMNS)


def summarize_across_assets(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy summary collapsing the asset dimension.

    Tells you, at a glance, which fixed strategies generalize: did this
    rule beat buy-and-hold across many assets, or only on the one we
    happened to optimise for?
    """
    if aggregate_df.empty:
        return pd.DataFrame(columns=MULTI_ASSET_SUMMARY_COLUMNS)

    rows: list[dict] = []
    for strategy, group in aggregate_df.groupby("strategy", sort=False):
        rows.append(
            {
                "strategy": strategy,
                "n_assets": len(group),
                "avg_return_across_assets": float(group["avg_annual_return"].mean()),
                "avg_worst_year": float(group["worst_year_return"].mean()),
                "total_years_outperforming_bh": int(
                    group["years_outperforming_bh"].sum()
                ),
                "total_years_lower_dd": int(group["years_lower_dd_than_bh"].sum()),
                "avg_exposure_across_assets": float(group["avg_exposure"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=MULTI_ASSET_SUMMARY_COLUMNS)
