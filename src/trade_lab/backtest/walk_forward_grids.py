"""Parameter grids for walk-forward validation of the priority-5 strategies.

Each ``build_*_grid`` returns a list of :class:`ParamGridSpec` ready to
feed to :func:`run_strategy_walk_forward`. The grids are deliberately
small (3-12 variants per strategy) to limit selection bias — a wider
grid would invite parameter mining and require a heavier DSR
correction downstream.

Design choices:

* For TSMOM and PMA-ratio (both *ensemble* strategies in this repo) we
  walk-forward across *ensemble compositions*, not across individual
  rolling-window lengths. That tests the design decision that matters
  ("which ladder of lookbacks did we pick?") without devolving into a
  single-window grid that would be a different strategy.
* For SMA crossover the grid is the conventional (fast, slow) product
  used elsewhere in the repo's sweep code.

``warmup_days`` comes from ``Strategy.required_warmup`` — the longest
rolling window in the variant, which for a filtered ensemble is usually
the regime SMA and not the headline lookback. It was hand-typed here
until a (30, 60, 90) ensemble behind SMA(200) was given 90. The
walk-forward runner doubles the value internally as a safety factor.
"""
from __future__ import annotations

from typing import List

from ..strategies.pma_ratio import PriceMaRatioStrategy
from ..strategies.sma_cross import SMACrossStrategy
from ..strategies.tsmom import TimeSeriesMomentumStrategy
from .walk_forward_v2 import ParamGridSpec


def _tsmom(lookbacks):
    """Factory for one TSMOM composition — named so the spec can build a
    probe instance and read ``required_warmup`` off it."""
    return lambda: TimeSeriesMomentumStrategy(
        lookbacks=lookbacks, sma_filter_periods=(200,),
    )


def _pma(ma_periods):
    return lambda: PriceMaRatioStrategy(ma_periods=ma_periods)


def build_tsmom_grid() -> List[ParamGridSpec]:
    """Three TSMOM ensemble compositions: short / medium / long."""
    return [
        ParamGridSpec(
            label="tsmom_short_30_60_90",
            factory=_tsmom((30, 60, 90)),
            warmup_days=_tsmom((30, 60, 90))().required_warmup,
        ),
        ParamGridSpec(
            label="tsmom_medium_30_90_180_365",
            factory=_tsmom((30, 90, 180, 365)),
            warmup_days=_tsmom((30, 90, 180, 365))().required_warmup,
        ),
        ParamGridSpec(
            label="tsmom_long_90_180_365",
            factory=_tsmom((90, 180, 365)),
            warmup_days=_tsmom((90, 180, 365))().required_warmup,
        ),
    ]


def build_pma_grid() -> List[ParamGridSpec]:
    """Three P/MA-ratio ladders: short / medium (Detzel et al. default) / long."""
    return [
        ParamGridSpec(
            label="pma_short_5_10_20",
            factory=_pma((5, 10, 20)),
            warmup_days=_pma((5, 10, 20))().required_warmup,
        ),
        ParamGridSpec(
            label="pma_medium_5_10_20_50_100",
            factory=_pma((5, 10, 20, 50, 100)),
            warmup_days=_pma((5, 10, 20, 50, 100))().required_warmup,
        ),
        ParamGridSpec(
            label="pma_long_10_20_50_100_200",
            factory=_pma((10, 20, 50, 100, 200)),
            warmup_days=_pma((10, 20, 50, 100, 200))().required_warmup,
        ),
    ]


def build_sma_grid(
    fast_periods=(10, 20, 30, 50),
    slow_periods=(50, 100, 150, 200, 300),
) -> List[ParamGridSpec]:
    """Cartesian product of (fast, slow) SMA pairs, skipping invalid ones."""
    grid: list[ParamGridSpec] = []
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            grid.append(
                ParamGridSpec(
                    label=f"sma_{fast}_{slow}",
                    factory=lambda fast=fast, slow=slow: SMACrossStrategy(
                        fast_period=fast, slow_period=slow
                    ),
                    warmup_days=SMACrossStrategy(
                        fast_period=fast, slow_period=slow
                    ).required_warmup,
                )
            )
    return grid
