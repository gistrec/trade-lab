"""Translate the basket signal into target per-asset quantities.

Given:

* The live signal value ``s`` (in the ladder ``{0, 0.5, 1.0}`` for the
  deployable TSMOM(28, 60) configuration).
* Current total equity in quote currency.
* Latest ticker prices for every asset in the basket.

The target dollar allocation per asset is

    target_quote[i] = s × (1 / N) × total_equity

where ``N`` is the basket size. ``s = 0`` → all-cash, ``s = 1`` → fully
invested at equal weight, ``s = 0.5`` → half-position per asset (the
backtest's pro-rata semantics).

Target quantity per asset:

    target_qty[i] = target_quote[i] / price[i]

The function is a **pure calculation** — no exchange calls. It is
fed prices the caller has already fetched (typically once per cycle,
from the broker). Keeping this layer pure makes it trivially testable
and ensures the deployment math matches the backtest math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TargetAllocation:
    """Per-asset target allocation snapshot."""

    signal: float                           # ladder value used
    total_equity: float                     # quote-currency reference
    target_quote_per_asset: dict[str, float]   # asset -> quote allocation
    target_qty_per_asset: dict[str, float]     # asset -> base quantity
    prices_used: dict[str, float]              # asset -> price the math used


def compute_target_allocation(
    signal: float,
    *,
    total_equity: float,
    prices: Mapping[str, float],
    basket: Sequence[str],
) -> TargetAllocation:
    """Compute target quantities from the signal + equity + prices.

    ``basket`` is the ordered list of assets that define ``N``. Assets
    in ``basket`` but missing from ``prices`` cause a :class:`KeyError`
    — we never silently treat a missing price as zero, because that
    would shrink the basket without the operator noticing.
    """
    if signal < 0.0 or signal > 1.0:
        raise ValueError(f"signal must be in [0, 1], got {signal}")
    if total_equity < 0.0:
        raise ValueError(f"total_equity must be >= 0, got {total_equity}")
    if not basket:
        raise ValueError("basket must be non-empty")

    n = len(basket)
    per_asset_fraction = signal / n
    target_quote = {sym: per_asset_fraction * total_equity for sym in basket}

    target_qty: dict[str, float] = {}
    prices_used: dict[str, float] = {}
    for sym in basket:
        if sym not in prices:
            raise KeyError(
                f"Missing price for {sym!r}; basket size is {n} and the "
                "allocator refuses to silently shrink the universe."
            )
        price = float(prices[sym])
        if price <= 0.0:
            raise ValueError(f"Non-positive price for {sym!r}: {price}")
        target_qty[sym] = target_quote[sym] / price
        prices_used[sym] = price

    return TargetAllocation(
        signal=signal,
        total_equity=total_equity,
        target_quote_per_asset=target_quote,
        target_qty_per_asset=target_qty,
        prices_used=prices_used,
    )
