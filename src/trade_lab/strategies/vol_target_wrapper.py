"""Volatility-targeting decorator for any :class:`Strategy`.

The wrapper takes an inner strategy's raw long-position signal and
scales it by ``annual_vol_target / realized_vol``, capped at
``max_position_size``. The implementation deliberately matches the
inline vol-targeting inside :class:`TimeSeriesMomentumStrategy` and
:class:`PriceMaRatioStrategy` so that "tsmom + wrapper(target=0.30)"
matches "tsmom(use_vol_target=False) wrapped at target=0.30" exactly
(modulo the rebalance band).

Why a wrapper instead of an option on every strategy:

* The conventional reference (Moreira & Muir 2017, *Journal of Finance*)
  treats vol-managed portfolios as a *layer* on top of any base
  factor, not as a property of the factor itself.
* It lets us A/B compare a strategy with and without vol-targeting
  while holding every other detail fixed — the wrapper is the only
  thing that changes.

Convexity caveat (also called out in `docs/results/vol_targeting.md`):
when realized vol spikes — typically during crashes — the position
*shrinks*. Sharpe and max drawdown tend to improve; hit-rate (fraction
of positive-return periods) and raw return often degrade because the
shrunken position misses part of the rebound.
"""
from __future__ import annotations

import pandas as pd

from ._trend_layers import vol_weight
from .base import Strategy


class VolatilityTargetWrapper(Strategy):
    """Wrap any long-only :class:`Strategy` with a vol-targeting layer.

    ``inner.generate_signals(candles)`` is expected to return a series
    of long-position targets in ``[0, 1]``. We multiply by
    ``annual_vol_target / realized_vol(vol_lookback) * sqrt(annualization_factor)``,
    clip to ``[0, max_position_size]``, and forward NaNs as 0.
    """

    name = "vol_target_wrapper"

    def __init__(
        self,
        inner: Strategy,
        *,
        annual_vol_target: float = 0.30,
        vol_lookback: int = 30,
        annualization_factor: int = 365,
        max_position_size: float = 1.0,
    ) -> None:
        if annual_vol_target <= 0:
            raise ValueError("annual_vol_target must be positive")
        if vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")
        if not 0 < max_position_size <= 1:
            raise ValueError(
                "max_position_size must be in (0, 1] for spot-only mode"
            )
        self.inner = inner
        self.annual_vol_target = float(annual_vol_target)
        self.vol_lookback = int(vol_lookback)
        self.annualization_factor = int(annualization_factor)
        self.max_position_size = float(max_position_size)
        # Expose a descriptive name for logging and CSV columns.
        target_pct = int(round(annual_vol_target * 100))
        inner_name = getattr(inner, "name", inner.__class__.__name__)
        self.name = f"vol{target_pct}({inner_name})"

    @property
    def required_warmup(self) -> int:
        # A wrapper needs whatever its inner strategy needs PLUS its own
        # vol window — reporting only its own would starve the inner
        # indicators, which is the very bias this property guards.
        return max(self.inner.required_warmup, self.vol_lookback)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        inner_signal = self.inner.generate_signals(candles)
        inner_signal = inner_signal.reindex(candles.index).fillna(0.0).astype(float)
        weight = vol_weight(
            candles["close"].astype(float),
            self.vol_lookback,
            self.annual_vol_target,
            self.annualization_factor,
        )
        scaled = (inner_signal * weight).clip(
            lower=0.0, upper=self.max_position_size
        ).fillna(0.0)
        return scaled
