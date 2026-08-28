"""Translate (current holdings, target allocation) into concrete orders.

Two outputs:

1. **Sendable orders** — ``OrderIntent`` records that meet the per-pair
   minimum notional / amount constraints reported by CCXT.
2. **Skipped divergences** — order requests absorbed rather than sent.
   These are logged separately because **accumulating skipped tiny
   rebalances is the main mechanism through which the live portfolio
   drifts from the backtest**. The operator needs to see them.
   Sub-minimum gaps and ``funding_cap`` shaves share the list but are
   metered apart (:func:`total_skipped_quote_drift` vs
   :func:`total_funding_cap_quote`): one is work the exchange refuses,
   the other a deliberate reserve on valid orders.

This module never sends an order. It produces a plan; the executor
module (step #2b) wires the plan into real CCXT calls.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Collection, Mapping, Optional, Sequence

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

# Reason string for the portion of a buy the reserve cap shaved off. It is
# an intentional buffer, not work that failed an exchange minimum, so it is
# reported apart from the sub-min drift metric (see
# :func:`total_funding_cap_quote`).
FUNDING_CAP_REASON = "funding_cap"


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
    fee_rate: Optional[float] = None,
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
    free quote is not burned on intents that never go out. Capping needs
    the cycle's ``fee_rate`` (sells fund the buys net of it); omitting it
    raises rather than silently sizing against a zero-fee exchange.

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
    if fee_rate is None:
        raise ValueError(
            "quote_free enables the reserve cap, which needs the cycle's "
            "fee_rate: sell proceeds fund the buys NET of the taker fee."
        )
    capped, cap_skips = apply_reserve_cap(
        orders, quote_free=quote_free, constraints=constraints,
        fee_rate=fee_rate,
    )
    return DeltaPlan(orders=capped, skipped=skipped + cap_skips)


def apply_reserve_cap(
    orders: Sequence[OrderIntent],
    *,
    quote_free: float,
    constraints: Mapping[str, MarketConstraints],
    fee_rate: float,
    no_flow_symbols: Collection[str] = (),
) -> tuple[list[OrderIntent], list[SkippedDelta]]:
    """Cap total buy spend at ``RESERVE_BPS`` under the available quote.

    Sells place first (sort_orders_for_placement) and their proceeds fund
    the buys, so the quote available to buys is the free balance plus the
    sendable sell notional — capping at bare quote_free would zero out the
    month-start cross-rebalance. On the full-entry path (no sells) this
    reduces to exactly quote_free.

    Sell proceeds count NET of ``fee_rate``: the exchange credits a sell
    minus the taker fee, so sizing buys off the gross ticker notional
    hands the whole reserve to the fee (at the default 10 bp it exactly
    cancels; above 10 bp the plan is underfunded even at unchanged
    prices).

    ``no_flow_symbols`` names orders that move no quote this cycle — a
    buy that draws nothing from ``quote_free``, or a sell that adds
    nothing to it (its clientOrderId is already resolved: filled, so the
    proceeds are inside ``quote_free`` already, or canceled/rejected, so
    they never arrive). The caller establishes which; this module stays
    out of clientOrderId/state semantics. Such buys pass through
    unscaled and out of the spend total; such sells are excluded from
    the proceeds that fund the buys.

    Every capped-away buy portion comes back as a ``funding_cap``
    :class:`SkippedDelta` carrying the UNSCALED gap — the cap is a real
    divergence from the backtest and must stay first-class, not vanish
    into zero-valued lot-step skips. It is reported apart from the sub-min
    drift metric (:func:`total_funding_cap_quote`).
    """
    no_flow = frozenset(no_flow_symbols)
    scalable = [o for o in orders if o.side == "buy" and o.symbol not in no_flow]
    buy_spend = sum(o.notional_quote for o in scalable)
    if buy_spend <= 0.0:
        return list(orders), []
    # A sell whose today-coid is already resolved brings no NEW quote:
    # filled, its proceeds are already inside quote_free; canceled or
    # rejected, they are never coming. Counting it either way sizes the
    # buys against money that does not exist.
    gross_sells = sum(
        o.notional_quote for o in orders
        if o.side == "sell" and o.symbol not in no_flow
    )
    available = float(quote_free) + gross_sells * (1.0 - float(fee_rate))
    cap = max(available, 0.0) * (1.0 - RESERVE_BPS / 10_000)
    if buy_spend <= cap:
        return list(orders), []   # slack exists — orders unchanged

    scale = cap / buy_spend
    buys = {o.symbol: o for o in scalable}
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

    _restore_min_sized_buys(amounts, buys, constraints)

    capped: list[OrderIntent] = []
    skips: list[SkippedDelta] = []
    for o in orders:
        if o.side != "buy" or o.symbol in no_flow:
            capped.append(o)
            continue
        a = amounts[o.symbol]
        c = constraints.get(o.symbol)
        if a <= 0.0 or _below_min(a, o.price_used, c):
            # Whole buy suppressed by the cap — record the full gap.
            skips.append(SkippedDelta(
                symbol=o.symbol,
                desired_side="buy",
                desired_amount=o.base_amount,
                desired_notional=o.notional_quote,
                constraint_min_amount=c.min_amount if c else None,
                constraint_min_cost=c.min_cost if c else None,
                reason=FUNDING_CAP_REASON,
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
                reason=FUNDING_CAP_REASON,
            ))
    return capped, skips


def _below_min(
    amount: float, price: float, c: Optional[MarketConstraints],
) -> bool:
    """True when ``amount`` violates either exchange minimum."""
    if c is None:
        return False
    return bool(
        (c.min_amount is not None and amount < c.min_amount)
        or (c.min_cost is not None and amount * price < c.min_cost)
    )


def _quantize_up(c: MarketConstraints, amount: float) -> Optional[float]:
    """Smallest on-grid amount >= ``amount`` (quantize_amount truncates).

    The grid may be a tick size rather than a decimal place, so the step
    is probed by doubling instead of assumed. ``None`` when no grid point
    above ``amount`` shows up within the probe range.
    """
    down = c.quantize_amount(amount)
    if down >= amount:
        return down
    probe = max(abs(amount), 1e-12) * 1e-9
    for _ in range(100):
        up = c.quantize_amount(down + probe)
        if up >= amount:
            return up
        probe *= 2.0
    return None


def _smallest_valid_amount(
    o: OrderIntent, c: Optional[MarketConstraints],
) -> Optional[float]:
    """Smallest on-grid amount for ``o`` that clears both minima.

    ``None`` when it cannot be reached without exceeding the intent
    itself (or when the pair has no constraints — nothing was below a
    minimum then).
    """
    if c is None or o.price_used <= 0.0:
        return None
    need = max(c.min_amount or 0.0, (c.min_cost or 0.0) / o.price_used)
    a = _quantize_up(c, need)
    # Up to two bumps: min_cost / price is float-dusty, so the first grid
    # point above it can still price a hair under the minimum notional.
    for _ in range(3):
        if a is None or a > o.base_amount:
            return None
        if not _below_min(a, o.price_used, c):
            return a
        a = _quantize_up(c, math.nextafter(a, math.inf))
    return None


def _restore_min_sized_buys(
    amounts: dict[str, float],
    buys: Mapping[str, OrderIntent],
    constraints: Mapping[str, MarketConstraints],
) -> None:
    """Shift the cap's reduction off buys it pushed under a minimum.

    Proportional scaling can drop a buy that was sendable below
    min_amount/min_cost; suppressing it strands its quote and can leave a
    basket asset out of the entry entirely — far more tracking error than
    the reserve. Restore such a buy to its smallest valid size and take
    the difference from the largest buy that stays valid after donating.
    The donor is quantized DOWN, so it gives up at least what the
    recipient gains and the total never climbs back over the cap. One
    pass in symbol order: deterministic and terminating.
    """
    for sym in sorted(amounts):
        o = buys[sym]
        c = constraints.get(sym)
        if amounts[sym] > 0.0 and not _below_min(amounts[sym], o.price_used, c):
            continue
        if _below_min(o.base_amount, o.price_used, c):
            continue                 # not sendable before the cap either
        restored = _smallest_valid_amount(o, c)
        if restored is None or restored <= amounts[sym]:
            continue
        donor = _pick_donor(
            sym, (restored - amounts[sym]) * o.price_used,
            amounts, buys, constraints,
        )
        if donor is None:
            continue                 # nobody can absorb it — stays suppressed
        amounts[sym] = restored
        amounts[donor[0]] = donor[1]


def _pick_donor(
    recipient: str,
    need_quote: float,
    amounts: Mapping[str, float],
    buys: Mapping[str, OrderIntent],
    constraints: Mapping[str, MarketConstraints],
) -> Optional[tuple[str, float]]:
    """Largest buy that can hand over ``need_quote`` and stay valid."""
    candidates = sorted(
        (s for s in amounts if s != recipient and amounts[s] > 0.0),
        key=lambda s: (-amounts[s] * buys[s].price_used, s),
    )
    for sym in candidates:
        o = buys[sym]
        c = constraints.get(sym)
        reduced = amounts[sym] - need_quote / o.price_used
        if reduced <= 0.0:
            continue
        reduced = c.quantize_amount(reduced) if c is not None else reduced
        if reduced > 0.0 and not _below_min(reduced, o.price_used, c):
            return sym, reduced
    return None


def total_skipped_quote_drift(plan: DeltaPlan) -> float:
    """Sum the quote-currency notional of skipped sub-min deltas.

    Reported in the dry-run log as the cumulative tracking error this
    cycle. Persistent non-zero values across cycles indicate the
    portfolio is drifting from the backtest by more than the order
    minimums allow.

    Funding-cap shaves are NOT here: this metric's contract (monitoring's
    "unfillable rebalance drift") is work that failed an exchange
    minimum, while the reserve is a deliberate buffer on valid orders.
    They live in :func:`total_funding_cap_quote`.
    """
    return float(sum(
        s.desired_notional for s in plan.skipped
        if s.reason != FUNDING_CAP_REASON
    ))


def total_funding_cap_quote(plan: DeltaPlan) -> float:
    """Sum the quote the :data:`RESERVE_BPS` cap held back this cycle."""
    return float(sum(
        s.desired_notional for s in plan.skipped
        if s.reason == FUNDING_CAP_REASON
    ))
