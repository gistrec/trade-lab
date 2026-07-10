"""Production live-paper-trading cycle orchestration.

End-to-end pipeline for the daily cron:

1. **Reconstruction.** For every non-terminal entry in the order-state
   store, ask the exchange what really happened. This MUST run before
   :meth:`Broker.fetch_balance_snapshot` — otherwise the balance lies
   about the current state, because partial fills from a prior cycle
   haven't been integrated into our local view yet.
2. **Live read.** Signal, balance, ticker prices, equity, target
   allocation, delta plan. Reuses the same primitives that
   :mod:`dry_run` uses — no strategy logic drifts between modes.
3. **Sort.** :func:`sort_orders_for_placement` — sells first, then buys.
   A cross-direction rebalance that runs buys first hits
   ``InsufficientFunds`` even when total accounting balances.
4. **Place sequentially.** 200ms inter-order sleep so we never exceed
   the per-second rate limit on a burst of 7 orders. Each placement
   waits for terminal status with bounded exponential backoff.
5. **Journal.** One :class:`Cycle` entry per orchestrator run, schema
   v2, with ``orders_executed`` populated. A reconstruction run writes
   a *separate* Cycle entry first so provenance is preserved — orders
   reconstructed from a prior cycle are not the current cycle's work.

Failure handling
================
An exception anywhere inside phase 2-5 still produces a journal entry
(``outcome='failed'``, ``error={...}``) with any partially-placed
orders recorded under ``orders_executed``. The exception then
propagates so the cron's stderr captures the actual traceback —
silently swallowing it would hide a real incident from the operator.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from .allocator import compute_target_allocation
from .broker import Broker, BrokerError, MarketConstraints
from .clientorder import make_client_order_id
from .delta import compute_delta_plan, total_skipped_quote_drift
from .journal import (
    Cycle,
    JournalWriter,
    get_git_commit_short,
    get_python_version,
    new_cycle_id,
)
from .order_state import OrderStateStore, utcnow_iso
from .orders import (
    TOTAL_TIMEOUT_S,
    OrderResult,
    fees_from_order,
    place_order,
    reconstruct_status,
    sort_orders_for_placement,
)
from .signal import SignalSnapshot, compute_live_signal
from ..logging_setup import set_cycle_id


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveCycleResult:
    """Summary returned to the CLI for printing.

    The full audit lives in the journal — this struct is just the
    handful of fields the CLI needs to print a one-screen summary.
    """

    cycle_id: str
    outcome: str
    order_results: list                  # list[OrderResult]
    reconstructed_count: int
    error: Optional[dict]
    lost_track_count: int = 0            # orders in lost_track state (any age)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_live_cycle(
    broker: Broker,
    *,
    lookbacks: Sequence[int] = (28, 60),
    sma_filter_period: int = 200,
    candles_per_asset: int = 400,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    journal: JournalWriter,
    state: OrderStateStore,
    inter_order_sleep_s: float = 0.2,
    total_timeout_s: float = TOTAL_TIMEOUT_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> LiveCycleResult:
    """Execute one live cycle: reconstruct, plan, place, journal."""

    started_at = datetime.now(timezone.utc)
    context = _build_context(broker)
    main_cycle_id = new_cycle_id()
    set_cycle_id(main_cycle_id)  # tag every log line in this run with the id

    # Phase 1: Reconstruction. Always first — main phase's
    # fetch_balance depends on reconciled state.
    reconstructed = _reconstruct_open_orders(broker, state)
    if reconstructed:
        _write_reconstruction_cycle(
            journal=journal,
            cycle_id=new_cycle_id(),
            started_at=started_at,
            context=context,
            reconstructed=reconstructed,
        )

    # A lost_track order is an unresolved incident flagged for manual
    # review. Count every lost_track entry still in state — newly
    # transitioned this cycle OR persisting from a prior one — so the CLI
    # can escalate the exit code for cron alerting even when the main
    # cycle places no orders and returns 'success'. Persistent entries are
    # deliberately NOT re-journaled (see _reconstruct_open_orders); this
    # counter only feeds the exit code, keeping alerting red until an
    # operator resolves the order.
    lost_track_count = sum(
        1 for e in state.open_entries().values() if e.status == "lost_track"
    )

    # Phase 2-5: Main cycle. Wrapped so any exception still gets
    # journaled, then re-raised so the cron stderr sees the traceback.
    rebal_date = datetime.now(timezone.utc).date()
    order_results: list[OrderResult] = []
    try:
        snap = compute_live_signal(
            broker,
            lookbacks=lookbacks,
            sma_filter_period=sma_filter_period,
            candles_per_asset=candles_per_asset,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        balance = broker.fetch_balance_snapshot()
        equity = broker.estimate_total_equity_usd(snapshot=balance)
        ticker_prices = _gather_ticker_prices(broker, snap)
        allocation = compute_target_allocation(
            signal=snap.signal,
            total_equity=equity,
            prices=ticker_prices,
            basket=broker.config.basket,
            weights=snap.basket_weights,
        )
        quote = broker.config.quote_currency
        constraints = _gather_constraints(broker, broker.config.basket, quote)
        plan = compute_delta_plan(
            allocation=allocation,
            current_holdings=balance.asset_totals,
            constraints=constraints,
            quote_currency=quote,
        )
        current_holdings_quote = {
            sym: float(balance.asset_totals.get(sym, 0.0))
                 * ticker_prices[sym]
            for sym in broker.config.basket
        }

        sorted_intents = sort_orders_for_placement(plan.orders)
        for i, intent in enumerate(sorted_intents):
            if i > 0:
                sleep_fn(inter_order_sleep_s)
            coid = make_client_order_id(rebal_date, intent.symbol, intent.side)
            result = place_order(
                broker, intent,
                client_order_id=coid,
                state=state,
                total_timeout_s=total_timeout_s,
                sleep_fn=sleep_fn, time_fn=time_fn,
            )
            order_results.append(result)

        outcome = _determine_outcome(order_results)
        # Snapshot this cycle's exchange round-trip latency into the journal
        # (read-only telemetry; the /metrics exporter surfaces it). Metadata
        # only — no effect on the orders just placed.
        context["exchange_latency"] = broker.exchange_call_stats()
        _write_main_cycle(
            journal=journal,
            cycle_id=main_cycle_id,
            started_at=started_at,
            context=context,
            outcome=outcome,
            snap=snap,
            balance=balance,
            equity=equity,
            allocation=allocation,
            current_holdings_quote=current_holdings_quote,
            plan=plan,
            order_results=order_results,
        )
        return LiveCycleResult(
            cycle_id=main_cycle_id,
            outcome=outcome,
            order_results=order_results,
            reconstructed_count=len(reconstructed),
            error=None,
            lost_track_count=lost_track_count,
        )

    except Exception as exc:
        context["exchange_latency"] = broker.exchange_call_stats()
        _write_failed_cycle(
            journal=journal,
            cycle_id=main_cycle_id,
            started_at=started_at,
            context=context,
            exc=exc,
            partial_orders=order_results,
        )
        raise


# ---------------------------------------------------------------------------
# Phase 1: Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_open_orders(
    broker: Broker, state: OrderStateStore,
) -> list[dict]:
    """Query the exchange for every non-terminal entry; reconcile state.

    Returns the list of *resolved* reconstructions (closed, canceled,
    or lost_track) for the reconstruction-cycle journal entry. Orders
    that the exchange still reports as open are left in state for the
    next cycle to retry — we do not block here waiting for them.

    ``lost_track`` is journaled only on the *transition* into that
    state. Entries already marked lost_track are still re-checked
    against the exchange every cycle (recovery stays possible if the
    record appears later), but a still-missing order is not the same
    incident again — re-journaling it daily would bury real events.
    """
    open_entries = state.open_entries()
    if not open_entries:
        return []

    resolved: list[dict] = []
    for coid, entry in open_entries.items():
        order = reconstruct_status(
            broker, coid, entry.symbol,
            exchange_order_id=entry.exchange_order_id,
            since_ms=_placed_at_ms(entry.placed_at),
        )

        if order is None:
            if entry.status == "lost_track":
                logger.info(
                    "Order %s still lost_track — exchange has no record; "
                    "will re-check next cycle.", coid,
                )
                continue
            logger.warning(
                "LOST TRACK: order %s (intended %s %s %s) not found on "
                "exchange. Marking lost_track for manual review.",
                coid, entry.side, entry.intended_amount, entry.symbol,
            )
            new_entry = replace(
                entry, status="lost_track", last_seen_at=utcnow_iso(),
            )
            state.put(new_entry)
            resolved.append({
                "client_order_id": coid,
                "exchange_order_id": entry.exchange_order_id,
                "symbol": entry.symbol,
                "side": entry.side,
                "intended_amount": entry.intended_amount,
                "terminal_status": "lost_track",
                "filled_amount": 0.0,
                "filled_notional_quote": 0.0,
                "average_price": None,
                "fees_paid_quote": None,
                "placed_at": entry.placed_at,
                "terminal_at": utcnow_iso(),
                "error": {
                    "type": "LostTrack",
                    "message": (
                        "Exchange has no record of this client order ID."
                    ),
                },
            })
            continue

        status_str = order.get("status")
        filled = float(order.get("filled") or 0.0)
        cost = float(order.get("cost") or 0.0)
        avg = order.get("average")
        quote = entry.symbol.split("/")[1] if "/" in entry.symbol else ""
        fees_quote, fees_reported = fees_from_order(order, quote)
        exchange_id = order.get("id")

        if status_str == "closed":
            terminal = "closed" if filled >= entry.intended_amount * 0.9999 else "partial"
        elif status_str == "canceled":
            terminal = "canceled"
        elif status_str in ("expired", "rejected"):
            # Exchange-terminal without a (full) fill: ccxt maps Binance
            # EXPIRED / EXPIRED_IN_MATCH → 'expired' and REJECTED →
            # 'rejected'. A non-zero fill is accounted as 'partial',
            # same as an exchange-closed partial.
            terminal = "partial" if filled > 0 else status_str
        else:
            # Still non-terminal — leave entry for next cycle to retry.
            logger.info(
                "Order %s still non-terminal at reconstruction (status=%s); "
                "leaving in state.", coid, status_str,
            )
            continue

        # Every exchange-terminal status goes into state as terminal so
        # the next cycle does not re-attempt reconstruction. 'partial'
        # is stored as 'closed' — the exchange will never fill more, so
        # there is nothing left to reconcile (mirrors orders._persist).
        persisted_status = "closed" if terminal == "partial" else terminal
        new_entry = replace(
            entry,
            status=persisted_status,
            exchange_order_id=str(exchange_id) if exchange_id is not None else entry.exchange_order_id,
            last_seen_at=utcnow_iso(),
        )
        state.put(new_entry)

        resolved.append({
            "client_order_id": coid,
            "exchange_order_id": str(exchange_id) if exchange_id is not None else None,
            "symbol": entry.symbol,
            "side": entry.side,
            "intended_amount": entry.intended_amount,
            "terminal_status": terminal,
            "filled_amount": filled,
            "filled_notional_quote": cost,
            "average_price": float(avg) if avg is not None else None,
            "fees_paid_quote": fees_quote,
            "placed_at": entry.placed_at,
            "terminal_at": utcnow_iso(),
            "error": None,
            "fees_reported": fees_reported,
        })

    return resolved


# ---------------------------------------------------------------------------
# Outcome determination
# ---------------------------------------------------------------------------


def _determine_outcome(order_results: list[OrderResult]) -> str:
    """Map the multiset of per-order statuses to one cycle-level word.

    Priority order (most urgent first): timeout/lost_track →
    partial/canceled/rejected/expired → success. Empty (no orders to
    place, e.g. signal=0 with current=0) is a clean success too.
    """
    if not order_results:
        return "success"
    statuses = {r.terminal_status for r in order_results}
    if statuses & {"timeout", "lost_track"}:
        return "unknown_orders"
    if statuses & {"partial", "canceled", "rejected", "expired"}:
        return "partial"
    return "success"


# ---------------------------------------------------------------------------
# Lifted helpers (duplicated from dry_run.py — refactor if a third
# cycle implementation appears)
# ---------------------------------------------------------------------------


def _placed_at_ms(placed_at: str) -> Optional[int]:
    """Epoch-ms of an ISO-8601 ``placed_at``, or ``None`` if unparseable.

    Bounds the reconstruction trade query to the order's age so Binance's
    default ~24h ``fetch_my_trades`` window does not hide an older fill.
    An unparseable stamp falls back to ``None`` (exchange default window)
    rather than raising — reconstruction is a recovery path.
    """
    try:
        dt = datetime.fromisoformat(placed_at)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_context(broker: Broker) -> dict:
    return {
        # Durable live/dry marker for read-only monitoring (the health
        # server). Present even when a cycle fails before placing an order,
        # so a failed live attempt is never misread as a dry-run. Metadata
        # only — changes no trading behaviour.
        "mode": "live",
        "exchange": broker.config.exchange_id,
        "sandbox": broker.config.sandbox,
        "quote_currency": broker.config.quote_currency,
        "basket": list(broker.config.basket),
    }


def _gather_ticker_prices(broker: Broker, snap: SignalSnapshot) -> dict[str, float]:
    """Live ticker prices, falling back to candle close on per-pair failure."""
    ticker_prices: dict[str, float] = {}
    quote = broker.config.quote_currency
    for sym in broker.config.basket:
        try:
            ticker_prices[sym] = broker.fetch_ticker_price(f"{sym}/{quote}")
        except BrokerError as exc:
            logger.warning(
                "Ticker for %s failed: %s — using candle close.", sym, exc,
            )
            # Direct indexing: a basket symbol missing from the signal's
            # closes is an invariant violation that must raise HERE, not
            # surface as a 0.0 price deep inside the allocator.
            ticker_prices[sym] = snap.asset_closes[sym]
    return ticker_prices


def _gather_constraints(
    broker: Broker, basket: Sequence[str], quote: str,
) -> dict[str, MarketConstraints]:
    constraints: dict[str, MarketConstraints] = {}
    for sym in basket:
        pair = f"{sym}/{quote}"
        try:
            constraints[pair] = broker.fetch_market_constraints(pair)
        except BrokerError as exc:
            logger.warning(
                "Constraints for %s unavailable: %s — sub-min filter "
                "disabled for this pair.", pair, exc,
            )
    return constraints


def _ts_iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _basket_close_series_dict(tail) -> Optional[dict]:
    if tail is None or len(tail) == 0:
        return None
    # 6 decimals on an index normalized to start at 100 — full-precision
    # floats here roughly double the serialized cycle size for nothing.
    return {
        "start_ts": _ts_iso(tail.index[0]),
        "values": [round(float(v), 6) for v in tail.tolist()],
    }


# ---------------------------------------------------------------------------
# Journal write helpers
# ---------------------------------------------------------------------------


def _write_reconstruction_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    reconstructed: list[dict],
) -> None:
    """One Cycle entry per orchestrator run that had something to recover.

    Kept separate from the main cycle's entry: reconstructed orders are
    a *prior* cycle's work, not the current run's. Mixing them muddies
    the audit trail when reconciling backtest vs reality later.
    """
    ended_at = datetime.now(timezone.utc)
    cycle = Cycle(
        cycle_id=cycle_id,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_ms=int((ended_at - started_at).total_seconds() * 1000),
        outcome="reconstructed",
        error=None,
        git_commit=get_git_commit_short(),
        python_version=get_python_version(),
        context=context,
        signal=None,
        basket_close_series=None,
        balance=None,
        equity_usd=None,
        target_allocation=None,
        current_holdings_quote=None,
        orders_planned=None,
        orders_skipped=None,
        total_skipped_quote_drift=None,
        orders_executed=list(reconstructed),
    )
    try:
        journal.append(cycle)
    except Exception as exc:
        logger.error(
            "Could not write reconstruction journal entry %s: %s",
            cycle_id, exc,
        )


def _write_main_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    outcome: str,
    snap: SignalSnapshot,
    balance,
    equity: float,
    allocation,
    current_holdings_quote: dict,
    plan,
    order_results: list[OrderResult],
) -> None:
    ended_at = datetime.now(timezone.utc)
    cycle = Cycle(
        cycle_id=cycle_id,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_ms=int((ended_at - started_at).total_seconds() * 1000),
        outcome=outcome,
        error=None,
        git_commit=get_git_commit_short(),
        python_version=get_python_version(),
        context=context,
        signal={
            "asof": _ts_iso(snap.asof),
            "ladder_value": snap.signal,
            "sma_gate_open": snap.sma_gate_open,
            "sma_value": snap.sma_value,
            "per_lookback_states": {
                str(k): int(v) for k, v in snap.per_lookback_states.items()
            },
            "per_lookback_returns": {
                str(k): float(v) for k, v in snap.per_lookback_returns.items()
            },
            "basket_close": snap.basket_close,
            "asset_closes": snap.asset_closes,
            "basket_weights": dict(snap.basket_weights),
        },
        basket_close_series=_basket_close_series_dict(snap.basket_close_tail),
        balance={
            "quote_currency": balance.quote_currency,
            "quote_total": balance.quote_total,
            "quote_free": balance.quote_free,
            "quote_used": balance.quote_used,
            "asset_totals": balance.asset_totals,
        },
        equity_usd=equity,
        target_allocation=dict(allocation.target_quote_per_asset),
        current_holdings_quote=current_holdings_quote,
        orders_planned=[_intent_dict(o) for o in plan.orders],
        orders_skipped=[_skipped_dict(s) for s in plan.skipped],
        total_skipped_quote_drift=total_skipped_quote_drift(plan),
        orders_executed=[r.to_dict() for r in order_results],
    )
    try:
        journal.append(cycle)
    except Exception as exc:
        logger.error(
            "Could not write main cycle journal entry %s: %s", cycle_id, exc,
        )


def _write_failed_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    exc: BaseException,
    partial_orders: list[OrderResult],
) -> None:
    """Failed cycle still gets a journal entry — silently dropping is
    the failure mode the journal exists to prevent. Partially-placed
    orders go in ``orders_executed`` so they are not lost to history.
    """
    ended_at = datetime.now(timezone.utc)
    cycle = Cycle(
        cycle_id=cycle_id,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_ms=int((ended_at - started_at).total_seconds() * 1000),
        outcome="failed",
        error={"type": type(exc).__name__, "message": str(exc)[:512]},
        git_commit=get_git_commit_short(),
        python_version=get_python_version(),
        context=context,
        signal=None,
        basket_close_series=None,
        balance=None,
        equity_usd=None,
        target_allocation=None,
        current_holdings_quote=None,
        orders_planned=None,
        orders_skipped=None,
        total_skipped_quote_drift=None,
        orders_executed=(
            [r.to_dict() for r in partial_orders] if partial_orders else None
        ),
    )
    try:
        journal.append(cycle)
    except Exception as journal_exc:
        logger.error(
            "Could not write failed-cycle journal entry %s: %s",
            cycle_id, journal_exc,
        )


def _intent_dict(intent) -> dict:
    return {
        "symbol": intent.symbol,
        "side": intent.side,
        "base_amount": intent.base_amount,
        "notional_quote": intent.notional_quote,
        "price_used": intent.price_used,
    }


def _skipped_dict(skipped) -> dict:
    return {
        "symbol": skipped.symbol,
        "desired_side": skipped.desired_side,
        "desired_amount": skipped.desired_amount,
        "desired_notional": skipped.desired_notional,
        "reason": skipped.reason,
    }
