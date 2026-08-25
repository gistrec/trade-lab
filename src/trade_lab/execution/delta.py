"""Translate (current holdings, target allocation) into concrete orders.

Two outputs:

1. **Sendable orders** — ``OrderIntent`` records that meet the per-pair
   minimum notional / amount constraints reported by CCXT.
2. **Skipped sub-min divergences** — sub-minimum order requests
   silently absorbed. These are logged separately because **accumulating
   skipped tiny rebalances is the main mechanism through which the
   live portfolio drifts from the backtest**. The operator needs to
   see them.

This module never sends an order. It produces a plan; the executor
module (step #2b) wires the plan into real CCXT calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .allocator import TargetAllocation
from .broker import MarketConstraints


logger = logging.getLogger(__name__)

# Owner-sanctioned microstructure deviation from the backtest (#28): hold
# back 10 bp of the quote available to buys. Sizes come from a ticker
# snapshot but market BUYs fill at the ask, so a plan spending 100% of the
# available quote loses its tail order to InsufficientFunds when prices
# tick up between snapshot and send. The backtest models neither fees nor
# lot steps; this buffer is the same class of gap.
RESERVE_BPS = 10


@dataclass(frozen=True)
class OrderIntent:
    """An order we intend to place to move toward the target."""

    symbol: str                  # CCXT pair, e.g. "BTC/USDT"
    side: str                    # "buy" or "sell"
    base_amount: float           # qty in base asset (positive)
    notional_quote: float        # base_amount × price
    price_used: float            # the price the math used (for divergence log)
    reason: str                  # e.g. "delta from target", short text


@dataclass(frozen=True)
class SkippedDelta:
    """A target-vs-current gap we chose NOT to send (below min notional)."""

    symbol: str
    desired_side: str            # what we *would* have sent
    desired_amount: float        # qty we *would* have moved
    desired_notional: float      # notional that's below the minimum
    constraint_min_amount: Optional[float]
    constraint_min_cost: Optional[float]
    reason: str


@dataclass(frozen=True)
class DeltaPlan:
    """Output of :func:`compute_delta_plan` — what to do this cycle."""

    orders: list[OrderIntent]
    skipped: list[SkippedDelta]


def compute_delta_plan(
    *,
    allocation: TargetAllocation,
    current_holdings: Mapping[str, float],
    constraints: Mapping[str, MarketConstraints],
    quote_currency: str,
    quote_free: Optional[float] = None,
) -> DeltaPlan:
    """Build the order plan from a target allocation and live holdings.

    ``current_holdings`` is the broker's ``asset_totals`` (free + used,
    in base units). ``constraints`` maps the CCXT pair (e.g.
    ``"BTC/USDT"``) to a :class:`MarketConstraints` describing the
    exchange's minimum-size rules; pass an empty dict to disable
    filtering (useful in tests).

    ``quote_free`` (the broker's free quote balance) enables the
    :data:`RESERVE_BPS` buy-spend cap; ``None`` disables it — the live
    cycle passes ``None`` and calls :func:`apply_reserve_cap` itself
    AFTER pending-order filtering, so a filtered pair's share of the
    free quote is not burned on intents that never go out.

    Sub-minimum deltas are recorded in ``skipped``. The total
    fractional drift carried by ``skipped`` should be reported in a
    reconciliation log so the operator sees what we couldn't move.
    """
    orders: list[OrderIntent] = []
    skipped: list[SkippedDelta] = []

    for sym in allocation.target_qty_per_asset.keys():
        pair = f"{sym}/{quote_currency}"
        target_qty = allocation.target_qty_per_asset[sym]
        current_qty = float(current_holdings.get(sym, 0.0) or 0.0)
        delta_qty = target_qty - current_qty
        price = allocation.prices_used[sym]

        if delta_qty == 0.0:
            continue

        side = "buy" if delta_qty > 0 else "sell"
        desired_qty = abs(delta_qty)
        desired_notional = desired_qty * price

        c = constraints.get(pair)
        # Truncate to the exchange lot step BEFORE the min gates and the
        # intent. ccxt truncates the amount inside ``create_order`` anyway
        # (``amount_to_precision``, TRUNCATE mode), so an unquantized
        # intent would make ``intended_amount`` unreachable by design and
        # a fully filled order would be journaled as a false ``partial``.
        # The intent must carry exactly the quantity that will be sent.
        abs_qty = c.quantize_amount(desired_qty) if c is not None else desired_qty
        notional = abs_qty * price

        if abs_qty <= 0.0:
            # The whole delta is below one lot step — same first-class
            # skip treatment as the sub-minimum cases below. The desired_*
            # fields keep the raw (pre-truncation) values so the skipped
            # drift metric measures the true gap we could not move.
            skipped.append(SkippedDelta(
                symbol=pair,
                desired_side=side,
                desired_amount=desired_qty,
                desired_notional=desired_notional,
                constraint_min_amount=c.min_amount,
                constraint_min_cost=c.min_cost,
                reason=(
                    f"amount {desired_qty:.8f} truncates to 0 at the "
                    "exchange lot step"
                ),
            ))
            continue

        below_min_amount = c is not None and c.min_amount is not None and abs_qty < c.min_amount
        below_min_cost = c is not None and c.min_cost is not None and notional < c.min_cost

        if below_min_amount or below_min_cost:
            reason_parts = []
            if below_min_amount:
                reason_parts.append(
                    f"amount {abs_qty:.8f} < min_amount {c.min_amount}"
                )
            if below_min_cost:
                reason_parts.append(
                    f"notional {notional:.4f} < min_cost {c.min_cost}"
                )
            skipped.append(SkippedDelta(
                symbol=pair,
                desired_side=side,
                desired_amount=desired_qty,
                desired_notional=desired_notional,
                constraint_min_amount=c.min_amount,
                constraint_min_cost=c.min_cost,
                reason="; ".join(reason_parts),
            ))
            continue

        orders.append(OrderIntent(
            symbol=pair,
            side=side,
            base_amount=abs_qty,
            notional_quote=notional,
            price_used=price,
            reason="delta from target",
        ))

    if quote_free is None:
        return DeltaPlan(orders=orders, skipped=skipped)
    capped, cap_skips = apply_reserve_cap(
        orders, quote_free=quote_free, constraints=constraints,
    )
    return DeltaPlan(orders=capped, skipped=skipped + cap_skips)


def apply_reserve_cap(
    orders: Sequence[OrderIntent],
    *,
    quote_free: float,
    constraints: Mapping[str, MarketConstraints],
) -> tuple[list[OrderIntent], list[SkippedDelta]]:
    """Cap total buy spend at ``RESERVE_BPS`` under the available quote.

    Sells place first (sort_orders_for_placement) and their proceeds fund
    the buys, so the quote available to buys is the free balance plus the
    sendable sell notional — capping at bare quote_free would zero out the
    month-start cross-rebalance. On the full-entry path (no sells) this
    reduces to exactly quote_free.

    Every capped-away buy portion comes back as a ``funding_cap``
    :class:`SkippedDelta` carrying the UNSCALED gap — the cap is a real
    divergence from the backtest and must stay first-class, not vanish
    into zero-valued lot-step skips.
    """
    buy_spend = sum(o.notional_quote for o in orders if o.side == "buy")
    if buy_spend <= 0.0:
        return list(orders), []
    available = float(quote_free) + sum(
        o.notional_quote for o in orders if o.side == "sell"
    )
    cap = max(available, 0.0) * (1.0 - RESERVE_BPS / 10_000)
    if buy_spend <= cap:
        return list(orders), []   # slack exists — orders unchanged

    scale = cap / buy_spend
    buys = {o.symbol: o for o in orders if o.side == "buy"}
    amounts: dict[str, float] = {}
    for sym, o in buys.items():
        c = constraints.get(sym)
        a = o.base_amount * scale
        amounts[sym] = c.quantize_amount(a) if c is not None else a

    # Re-check the POST-quantization total: scaling already-quantized
    # amounts and truncating keeps the sum ≤ cap in exact arithmetic,
    # but float dust can leak over. Shave the largest buy until the cap
    # holds — an on-grid amount minus any positive quantity re-truncates
    # at least one full lot step lower, so every pass strictly shrinks
    # the total and the loop terminates.
    def _spend() -> float:
        return sum(amounts[s] * buys[s].price_used for s in amounts)

    while _spend() - cap > 1e-9:
        sym = max(
            (s for s in amounts if amounts[s] > 0.0),
            key=lambda s: amounts[s] * buys[s].price_used,
            default=None,
        )
        if sym is None:
            break
        c = constraints.get(sym)
        reduced = max(
            amounts[sym] - (_spend() - cap) / buys[sym].price_used, 0.0,
        )
        amounts[sym] = c.quantize_amount(reduced) if c is not None else reduced

    capped: list[OrderIntent] = []
    skips: list[SkippedDelta] = []
    for o in orders:
        if o.side != "buy":
            capped.append(o)
            continue
        a = amounts[o.symbol]
        c = constraints.get(o.symbol)
        below_min = c is not None and (
            (c.min_amount is not None and a < c.min_amount)
            or (c.min_cost is not None and a * o.price_used < c.min_cost)
        )
        if a <= 0.0 or below_min:
            # Whole buy suppressed by the cap — record the full gap.
            skips.append(SkippedDelta(
                symbol=o.symbol,
                desired_side="buy",
                desired_amount=o.base_amount,
                desired_notional=o.notional_quote,
                constraint_min_amount=c.min_amount if c else None,
                constraint_min_cost=c.min_cost if c else None,
                reason="funding_cap",
            ))
            continue
        capped.append(OrderIntent(
            symbol=o.symbol,
            side="buy",
            base_amount=a,
            notional_quote=a * o.price_used,
            price_used=o.price_used,
            reason=o.reason,
        ))
        if a < o.base_amount:
            skips.append(SkippedDelta(
                symbol=o.symbol,
                desired_side="buy",
                desired_amount=o.base_amount - a,
                desired_notional=(o.base_amount - a) * o.price_used,
                constraint_min_amount=c.min_amount if c else None,
                constraint_min_cost=c.min_cost if c else None,
                reason="funding_cap",
            ))
    return capped, skips


def total_skipped_quote_drift(plan: DeltaPlan) -> float:
    """Sum the quote-currency notional of all skipped sub-min deltas.

    Reported in the dry-run log as the cumulative tracking error this
    cycle. Persistent non-zero values across cycles indicate the
    portfolio is drifting from the backtest by more than the order
    minimums allow.
    """
    return float(sum(s.desired_notional for s in plan.skipped))
