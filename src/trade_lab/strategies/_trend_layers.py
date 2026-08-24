"""Shared layers of the long-only trend stack.

These are deliberately identical across the trend strategies so that
cross-strategy comparisons isolate the *signal* while sizing and
filtering stay fixed.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def validate_trend_params(
    vol_lookback: int,
    annual_vol_target: float,
    annualization_factor: int,
    max_position_size: float,
    rebalance_threshold: float,
) -> None:
    if vol_lookback < 2:
        raise ValueError("vol_lookback must be >= 2")
    if annual_vol_target <= 0:
        raise ValueError("annual_vol_target must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if not 0 < max_position_size <= 1:
        raise ValueError(
            "max_position_size must be in (0, 1] for spot-only mode"
        )
    if rebalance_threshold < 0:
        raise ValueError("rebalance_threshold must be >= 0")


def sma_filter(close: pd.Series, periods: Iterable[int]) -> pd.Series:
    """True only where close exceeds every SMA period; warm-up fails shut."""
    ok = pd.Series(True, index=close.index)
    for period in periods:
        sma = close.rolling(period).mean()
        cond = close > sma
        cond[sma.isna()] = False
        ok = ok & cond
    return ok


def vol_weight(
    close: pd.Series,
    vol_lookback: int,
    annual_vol_target: float,
    annualization_factor: int,
) -> pd.Series:
    """``target_vol / realized_vol``; NaN and inf map to 0 — never lever up."""
    daily_returns = close.pct_change(fill_method=None)
    realized_vol_daily = daily_returns.rolling(vol_lookback).std()
    realized_vol_annual = realized_vol_daily * np.sqrt(annualization_factor)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = annual_vol_target / realized_vol_annual
    return weight.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_rebalance_band(
    target_position: pd.Series, rebalance_threshold: float
) -> pd.Series:
    """Suppress sub-threshold size changes; entries and exits always pass.

    ``rebalance_threshold == 0`` returns the input unchanged.
    """
    if rebalance_threshold == 0.0:
        return target_position
    held = pd.Series(0.0, index=target_position.index, dtype=float)
    current = 0.0
    for i, target in enumerate(target_position.to_numpy()):
        target = float(target)
        if target == 0.0 or current == 0.0:
            current = target
        elif abs(target - current) >= rebalance_threshold:
            current = target
        held.iloc[i] = current
    return held
