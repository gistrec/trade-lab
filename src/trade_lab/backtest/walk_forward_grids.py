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

``warmup_days`` is set conservatively at the longest rolling lookback
inside the variant. The walk-forward runner doubles that internally
as a safety factor.
"""
from __future__ import annotations

from typing import List

from ..strategies.pma_ratio import PriceMaRatioStrategy
from ..strategies.sma_cross import SMACrossStrategy
from ..strategies.tsmom import TimeSeriesMomentumStrategy
from .walk_forward_v2 import ParamGridSpec


def build_tsmom_grid() -> List[ParamGridSpec]:
    """Three TSMOM ensemble compositions: short / medium / long."""
    return [
        ParamGridSpec(
            label="tsmom_short_30_60_90",
            factory=lambda: TimeSeriesMomentumStrategy(
                lookbacks=(30, 60, 90),
                sma_filter_periods=(200,),
            ),
            warmup_days=90,
        ),
        ParamGridSpec(
            label="tsmom_medium_30_90_180_365",
            factory=lambda: TimeSeriesMomentumStrategy(
                lookbacks=(30, 90, 180, 365),
                sma_filter_periods=(200,),
            ),
            warmup_days=365,
        ),
        ParamGridSpec(
            label="tsmom_long_90_180_365",
            factory=lambda: TimeSeriesMomentumStrategy(
                lookbacks=(90, 180, 365),
                sma_filter_periods=(200,),
            ),
            warmup_days=365,
        ),
    ]


def build_pma_grid() -> List[ParamGridSpec]:
    """Three P/MA-ratio ladders: short / medium (Detzel et al. default) / long."""
    return [
        ParamGridSpec(
            label="pma_short_5_10_20",
            factory=lambda: PriceMaRatioStrategy(ma_periods=(5, 10, 20)),
            warmup_days=20,
        ),
        ParamGridSpec(
            label="pma_medium_5_10_20_50_100",
            factory=lambda: PriceMaRatioStrategy(ma_periods=(5, 10, 20, 50, 100)),
            warmup_days=100,
        ),
        ParamGridSpec(
            label="pma_long_10_20_50_100_200",
            factory=lambda: PriceMaRatioStrategy(ma_periods=(10, 20, 50, 100, 200)),
            warmup_days=200,
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
                    warmup_days=slow,
                )
            )
    return grid
