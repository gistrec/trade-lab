"""MVRV-ratio overlay for single-asset BTC spot — slow weekly tilt.

This is an INTENTIONALLY simplified proxy. Three deliberate deviations
from the canonical compass-artifact pseudocode:

1. **MVRV ratio, not MVRV-Z score.** The Z-score requires the realized
   cap (``CapRealUSD``) and its rolling stddev. Both are paid-tier-only
   on Coin Metrics. The community-tier exposes ``CapMVRVCur`` (the
   raw ratio MarketCap/RealizedCap). We use the ratio with mapped
   thresholds (high≈3.5, low≈1.0) approximately equivalent to the
   compass's Z>6 / Z<0 regions on historical BTC. The Z is more
   precise; the ratio is what's free.
2. **Linear interpolation between thresholds**, not the binary
   "cash above / full long below" the compass pseudo. Linear absorbs
   small threshold-tuning sensitivity (Q8-class concern).
3. **One-day publication lag.** Coin Metrics community data publishes
   t-1 at midnight UTC of day t; the signal at day t uses MVRV from
   day t-1 to avoid look-ahead through real-world latency.

Asymmetry of interpretation
===========================
* Compass artifact explicitly predicted INCONCLUSIVE due to only
  2-3 BTC market cycles in the sample. DSR @ N=500 on this scale
  WILL be negligible by construction.
* A pass would require a multi-cycle independent dataset (impossible
  on BTC) or a strong economic story for why this overlay should
  outperform out-of-sample on the NEXT cycle. Neither is on offer.
* A failure is also weak: with 2-3 cycles, we can't distinguish "no
  edge" from "edge buried in cycle noise."

Look-ahead guarantees
=====================
* MVRV input lagged by ``publication_lag_days`` days before use.
* Position target at day t depends only on MVRV up to ``t - lag``.
* No threshold tuning in this module — thresholds are caller-supplied
  and the finding records them as literature-derived (no in-sample
  tuning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .cross_sectional import _max_drawdown, _sharpe


DEFAULT_LOW_THRESHOLD = 1.0      # ratio at/below → full long
DEFAULT_HIGH_THRESHOLD = 3.5     # ratio at/above → cash
DEFAULT_REBALANCE_DAYS = 7       # weekly tilt — slow by design
DEFAULT_PUBLICATION_LAG_DAYS = 1


@dataclass
class MvrvOverlayResult:
    equity: pd.Series
    returns: pd.Series
    target_position: pd.Series
    realized_position: pd.Series
    rebalance_dates: list = field(default_factory=list)
    initial_capital: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    num_rebalances: int = 0
    mean_position: float = 0.0


def mvrv_target_position(
    mvrv: pd.Series,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.Series:
    """Map MVRV to a long-only target in [0, 1] via linear interpolation.

    * ``mvrv <= low`` → 1.0 (full long)
    * ``mvrv >= high`` → 0.0 (cash)
    * between → linear: ``(high - mvrv) / (high - low)`` clamped.
    """
    if not (low_threshold < high_threshold):
        raise ValueError(
            f"low_threshold ({low_threshold}) must be < high_threshold "
            f"({high_threshold})"
        )
    raw = (high_threshold - mvrv) / (high_threshold - low_threshold)
    return raw.clip(lower=0.0, upper=1.0)


def run_mvrv_overlay(
    btc_candles: pd.DataFrame,
    mvrv: pd.Series,
    *,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    rebalance_days: int = DEFAULT_REBALANCE_DAYS,
    publication_lag_days: int = DEFAULT_PUBLICATION_LAG_DAYS,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    annualization_factor: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> MvrvOverlayResult:
    """Weekly MVRV-ratio overlay on a long-only BTC spot position."""
    if "close" not in btc_candles.columns:
        raise ValueError("btc_candles must contain a 'close' column")

    closes = btc_candles["close"].copy()
    closes.index = closes.index if closes.index.tz else closes.index.tz_localize("UTC")
    mvrv = mvrv.copy()
    mvrv.index = mvrv.index if mvrv.index.tz else mvrv.index.tz_localize("UTC")

    # Align on common index.
    common = closes.index.intersection(mvrv.index)
    closes = closes.loc[common]
    mvrv = mvrv.loc[common]

    if start_date is not None:
        cutoff = pd.Timestamp(start_date, tz="UTC")
        closes = closes[closes.index >= cutoff]
        mvrv = mvrv[mvrv.index >= cutoff]
    if end_date is not None:
        cutoff = pd.Timestamp(end_date, tz="UTC")
        closes = closes[closes.index <= cutoff]
        mvrv = mvrv[mvrv.index <= cutoff]

    if len(closes) < 2:
        return MvrvOverlayResult(
            equity=pd.Series(dtype=float),
            returns=pd.Series(dtype=float),
            target_position=pd.Series(dtype=float),
            realized_position=pd.Series(dtype=float),
            initial_capital=initial_capital,
        )

    # Lag MVRV — we trade off the value PUBLISHED 1 day ago.
    mvrv_avail = mvrv.shift(publication_lag_days)

    target = mvrv_target_position(mvrv_avail, low_threshold, high_threshold)
    target = target.reindex(closes.index).ffill().fillna(0.0)

    # Apply target only on rebalance days; hold between.
    realized = pd.Series(0.0, index=closes.index)
    rebalance_dates: list[pd.Timestamp] = []
    current = 0.0
    last_rebal_idx = -10**9
    for i, d in enumerate(closes.index):
        if (i - last_rebal_idx) >= rebalance_days:
            new = float(target.iloc[i])
            if new != current:
                current = new
                last_rebal_idx = i
                rebalance_dates.append(d)
            elif i == 0:
                # First bar: even if target=current=0, mark as rebalance to
                # anchor the loop without spurious turnover.
                last_rebal_idx = i
        realized.iloc[i] = current

    # Simulate equity with symmetric costs on |Δposition|.
    daily_returns = closes.pct_change().fillna(0.0)
    eq = pd.Series(0.0, index=closes.index)
    eq.iloc[0] = initial_capital
    total_fees = 0.0
    total_slippage = 0.0
    for i in range(1, len(closes)):
        prev_w = float(realized.iloc[i - 1])
        port_ret = prev_w * float(daily_returns.iloc[i])
        new_eq = float(eq.iloc[i - 1]) * (1.0 + port_ret)
        delta_w = abs(float(realized.iloc[i]) - prev_w)
        if delta_w > 1e-12:
            fee = new_eq * delta_w * fee_rate
            slip = new_eq * delta_w * slippage_rate
            new_eq -= (fee + slip)
            total_fees += fee
            total_slippage += slip
        eq.iloc[i] = new_eq

    ret = eq.pct_change().fillna(0.0)
    return MvrvOverlayResult(
        equity=eq,
        returns=ret,
        target_position=target,
        realized_position=realized,
        rebalance_dates=rebalance_dates,
        initial_capital=initial_capital,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_return=float(eq.iloc[-1] / initial_capital - 1.0),
        max_drawdown=_max_drawdown(eq),
        sharpe=_sharpe(ret, annualization_factor),
        num_rebalances=len(rebalance_dates),
        mean_position=float(realized.mean()),
    )
