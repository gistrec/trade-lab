"""Default sleeve registry — per-asset vol-target picks per the findings.

The vol-target variant for each (strategy, asset) combination is taken
from the Sharpe-winner column of the 7-asset matrix in
``findings/vol_targeting_regime_gate.md``. The deliberate rule is:

* BTC: always ``raw``. Findings showed BTC loses on Calmar at every
  vol-target level and Sharpe is essentially a tie — the regime gate
  already does the risk management, the wrapper just dilutes upside.
* Every other asset: pick the variant that won OOS Sharpe in the
  matrix. Negative-Sharpe sleeves are kept (a 1/N portfolio is
  diversification, not stock-picking; excluding negatives mid-run is
  itself a forward-looking selection).

These picks are fixed configurations, not walk-forward selections.
They are part of the project's selection budget — see the bump in
``walk_forward_v2.PROJECT_NUM_TRIALS`` for the corresponding
acknowledgement.
"""
from __future__ import annotations

from typing import Iterable, List

from ..strategies.pma_ratio import PriceMaRatioStrategy
from ..strategies.sma_cross import SMACrossStrategy
from ..strategies.tsmom import TimeSeriesMomentumStrategy
from ..strategies.vol_target_wrapper import VolatilityTargetWrapper
from .ensemble import SleeveSpec


# ---------------------------------------------------------------------------
# Per-(strategy, asset) vol-target picks — from the 7-asset matrix
# ---------------------------------------------------------------------------

# Values are the annual vol target (e.g. 0.30 = 30%). ``None`` means
# "no wrapper" (raw strategy as-is).
_VOL_TARGETS: dict[tuple[str, str], float | None] = {
    # sma_cross — Sharpe winners across the 7-asset matrix
    ("sma_20_100", "BTC"):  None,    # raw
    ("sma_20_100", "ETH"):  0.50,
    ("sma_20_100", "BNB"):  0.30,
    ("sma_20_100", "SOL"):  None,    # raw (all variants negative, raw is least bad)
    ("sma_20_100", "ADA"):  0.50,
    ("sma_20_100", "XRP"):  0.30,
    ("sma_20_100", "DOGE"): None,    # raw (all variants negative)

    # tsmom_medium — Sharpe winners
    ("tsmom_medium", "BTC"):  None,
    ("tsmom_medium", "ETH"):  0.50,
    ("tsmom_medium", "BNB"):  0.50,
    ("tsmom_medium", "SOL"):  0.50,
    ("tsmom_medium", "ADA"):  0.50,
    ("tsmom_medium", "XRP"):  0.50,
    ("tsmom_medium", "DOGE"): 0.30,

    # pma_medium — Sharpe winners (vol30 sweeps 6/7)
    ("pma_medium", "BTC"):  None,
    ("pma_medium", "ETH"):  0.30,
    ("pma_medium", "BNB"):  0.30,
    ("pma_medium", "SOL"):  0.30,
    ("pma_medium", "ADA"):  0.30,
    ("pma_medium", "XRP"):  0.30,
    ("pma_medium", "DOGE"): 0.30,
}


_BASE_STRATEGIES = {
    "sma_20_100": (
        lambda: SMACrossStrategy(fast_period=20, slow_period=100),
        100,
    ),
    "tsmom_medium": (
        lambda: TimeSeriesMomentumStrategy(
            lookbacks=(30, 90, 180, 365),
            sma_filter_periods=(200,),
            use_vol_target=False,
        ),
        365,
    ),
    "pma_medium": (
        lambda: PriceMaRatioStrategy(
            ma_periods=(5, 10, 20, 50, 100),
            use_vol_target=False,
        ),
        100,
    ),
}


DEFAULT_ASSETS: tuple[str, ...] = (
    "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
)


def _make_factory(base_factory, target: float | None):
    """Wrap base factory in VolatilityTargetWrapper if target is set."""
    if target is None:
        return base_factory
    return lambda bf=base_factory, t=target: VolatilityTargetWrapper(
        bf(), annual_vol_target=t, vol_lookback=30,
    )


def default_sleeves(
    assets: Iterable[str] = DEFAULT_ASSETS,
    strategies: Iterable[str] = ("sma_20_100", "tsmom_medium", "pma_medium"),
) -> List[SleeveSpec]:
    """Build the 21-sleeve default ensemble (3 strategies × 7 assets).

    The per-(strategy, asset) vol-target variant is hard-coded from
    the findings in ``findings/vol_targeting_regime_gate.md``. Pass a
    subset of ``assets`` or ``strategies`` to test smaller ensembles.
    """
    sleeves: List[SleeveSpec] = []
    for strat_name in strategies:
        if strat_name not in _BASE_STRATEGIES:
            raise ValueError(f"Unknown strategy {strat_name!r}")
        base_factory, base_warmup = _BASE_STRATEGIES[strat_name]
        for asset in assets:
            key = (strat_name, asset)
            if key not in _VOL_TARGETS:
                raise ValueError(
                    f"No vol-target picked for {key!r}; update _VOL_TARGETS "
                    "or extend the findings matrix before running."
                )
            target = _VOL_TARGETS[key]
            wrapped_warmup = base_warmup + (30 if target is not None else 0)
            vt_label = "raw" if target is None else f"vol{int(target * 100)}"
            label = f"{strat_name}__{asset}__{vt_label}"
            sleeves.append(SleeveSpec(
                label=label,
                asset=asset,
                factory=_make_factory(base_factory, target),
                warmup_days=wrapped_warmup,
            ))
    return sleeves
