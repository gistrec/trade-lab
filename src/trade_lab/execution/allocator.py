"""Translate the basket signal into target per-asset quantities.

Given:

* The live signal value ``s`` (in the ladder ``{0, 0.5, 1.0}`` for the
  deployable TSMOM(28, 60) configuration).
* Current total equity in quote currency.
* The basket index's per-asset **drifted weights** ``w_i`` at asof.
* Latest ticker prices for every asset in the basket.

The target dollar allocation per asset is

    target_quote[i] = s × w_i × total_equity

where ``w_i`` is the weight asset ``i`` actually carries in the
monthly-rebalanced market-basket index at asof — flat ``1 / N_active``
immediately after a rebalance, drifting with returns between rebalances
(and summing to 1 over active assets). ``s = 0`` → all-cash, ``s = 1`` →
fully invested at the index's current weights, ``s = 0.5`` → half of each
(the backtest's pro-rata semantics).

Sizing to ``w_i`` — not flat ``1/N`` — is what makes live execution
replicate the backtest. Between monthly rebalances the drifted-weight
target tracks the drifted holdings, so the per-asset deltas fall below
the exchange minimums and no orders fire; real orders only fire at the
month-start weight reset and on signal changes — the same turnover
profile the backtest measured. Flat ``1/N`` would instead force a
full rebalance every daily cycle, adding turnover the backtest never
paid (C3 / Option B).

Target quantity per asset:

    target_qty[i] = target_quote[i] / price[i]

The function is a **pure calculation** — no exchange calls. It is
fed prices and weights the caller has already computed (once per cycle),
which keeps this layer trivially testable and the deployment math
identical to the backtest math.
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
    weights_used: dict[str, float]             # asset -> drifted weight the math used


def compute_target_allocation(
    signal: float,
    *,
    total_equity: float,
    prices: Mapping[str, float],
    basket: Sequence[str],
    weights: Mapping[str, float],
) -> TargetAllocation:
    """Compute target quantities from the signal + equity + weights + prices.

    ``basket`` is the ordered list of assets. ``weights`` maps each asset
    to its drifted weight in the basket index at asof (see the module
    docstring). Assets in ``basket`` but missing from ``weights`` or
    ``prices`` raise :class:`KeyError` — we never guess a weight or treat
    a missing price as zero, because either would silently reshape the
    basket without the operator noticing.
    """
    if signal < 0.0 or signal > 1.0:
        raise ValueError(f"signal must be in [0, 1], got {signal}")
    if total_equity < 0.0:
        raise ValueError(f"total_equity must be >= 0, got {total_equity}")
    if not basket:
        raise ValueError("basket must be non-empty")

    # Resolve + validate the per-asset weights first. A missing weight is
    # a KeyError (symmetric with a missing price); a negative or NaN
    # weight is garbage; and the sum may not exceed a fully-invested book.
    weights_used: dict[str, float] = {}
    total_weight = 0.0
    for sym in basket:
        if sym not in weights:
            raise KeyError(
                f"Missing basket weight for {sym!r}; the allocator refuses "
                "to guess a weight (that would silently reshape the basket)."
            )
        w = float(weights[sym])
        if not (w >= 0.0):   # also rejects NaN (NaN >= 0.0 is False)
            raise ValueError(f"Weight for {sym!r} must be >= 0, got {w}")
        weights_used[sym] = w
        total_weight += w
    if total_weight > 1.0 + 1e-6:
        raise ValueError(
            f"Basket weights sum to {total_weight:.6f} > 1; refusing to "
            "over-invest beyond the fully-invested book."
        )

    target_quote = {
        sym: signal * weights_used[sym] * total_equity for sym in basket
    }

    target_qty: dict[str, float] = {}
    prices_used: dict[str, float] = {}
    for sym in basket:
        if sym not in prices:
            raise KeyError(
                f"Missing price for {sym!r}; basket size is {len(basket)} and "
                "the allocator refuses to silently shrink the universe."
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
        weights_used=weights_used,
    )
