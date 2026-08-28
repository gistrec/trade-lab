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
orders recorded under ``orders_executed``. This covers *any*
``BaseException`` — including ``KeyboardInterrupt``, because Ctrl-C
during the up-to-35-minute wait-for-ack of a manual run is the most
likely mid-cycle abort. The exception then propagates so the cron's
stderr captures the actual traceback — silently swallowing it would
hide a real incident from the operator.

One deliberate exception to the rule: ``InsufficientWarmupError`` on a
*sandbox* config does not fail the cycle. Binance testnet wipes its
candle history ~monthly, so the SMA(200) warm-up is structurally
impossible there — the cycle is journaled as a first-class
``outcome='skipped_warmup'`` with an explicit ``skip_reason`` block and
returned normally (no orders, no plan). On mainnet the same error keeps
the failed-entry + re-raise posture: truncated kline history on the
real-money path is an incident, full stop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from .broker import Broker
from .clientorder import make_client_order_id
from .cycle_common import (
    balance_dict,
    basket_close_series_dict,
    build_context,
    failed_cycle,
    intent_dict,
    run_read_phase,
    signal_dict,
    skipped_dict,
    skipped_warmup_cycle,
)
from .cycle_common import (  # noqa: F401  (re-export: tests hit the ticker fallback through this name)
    gather_ticker_prices as _gather_ticker_prices,
)
from .delta import (
    SkippedDelta, apply_reserve_cap, total_funding_cap_quote,
    total_skipped_quote_drift,
)
from .journal import (
    Cycle,
    JournalWriter,
    _encode_cycle,
    get_git_commit_short,
    get_python_version,
    new_cycle_id,
)
from .order_state import TERMINAL_STATUSES, OrderStateStore, utcnow_iso
from .orders import (
    TOTAL_TIMEOUT_S,
    OrderResult,
    fees_from_order,
    place_order,
    reconstruct_status,
    sort_orders_for_placement,
)
from .signal import (
    InsufficientWarmupError,
    SignalComputationError,
    SignalSnapshot,
    compute_live_signal,
    decision_age_seconds,
)
from ..logging_setup import set_cycle_id


logger = logging.getLogger(__name__)

# Ceiling on how long after the decision bar's close a LIVE cycle may
# still place orders. The backtest trades each signal right after its
# bar closes; a cycle hours later is a different, unvalidated timing
# (the MSK-scheduled-cron incident shape). 6h keeps room for a same-
# morning manual recovery re-run while catching any wrong-timezone
# schedule (the nearest such error lands at ≥21h).
MAX_DECISION_AGE_S = 6 * 3600.0


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
    skip_reason: Optional[dict] = None
    # Set only for outcome='skipped_warmup' (testnet-only structural
    # SMA warm-up skip): the same reason dict that went into the
    # journal, so the CLI can print bars_available/bars_required.
    journal_write_failed: bool = False
    # A journal.append in this run failed AFTER orders were placed —
    # raising then would be worse than the missing entry, so the CLI
    # escalates the exit code instead.


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
    max_decision_age_s: Optional[float] = MAX_DECISION_AGE_S,
) -> LiveCycleResult:
    """Execute one live cycle: reconstruct, plan, place, journal.

    ``max_decision_age_s`` — refuse to place orders when the decision
    bar closed more than this many seconds ago (``None`` disables; the
    CLI exposes it as ``--max-signal-age-h``)."""

    started_at = datetime.now(timezone.utc)
    context = build_context(broker, mode="live")
    main_cycle_id = new_cycle_id()
    set_cycle_id(main_cycle_id)  # tag every log line in this run with the id

    # Phase 1: Reconstruction. Always first — main phase's
    # fetch_balance depends on reconciled state.
    reconstructed = _reconstruct_open_orders(broker, state)
    journal_write_failed = False
    if reconstructed:
        journal_write_failed |= _write_reconstruction_cycle(
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
    open_after_recon = state.open_entries()
    lost_track_count = sum(
        1 for e in open_after_recon.values() if e.status == "lost_track"
    )
    # Entries still live ON THE EXCHANGE after reconstruction (open /
    # partial / timeout): their expected fill is not in the balance yet,
    # so a fresh delta for the same pair doubles the position once both
    # fill. lost_track is deliberately NOT here — the exchange has no
    # record, the balance is truthful, and the incident already
    # escalates via lost_track_count.
    pending_by_pair: dict[str, list] = {}
    for e in open_after_recon.values():
        if e.status != "lost_track":
            pending_by_pair.setdefault(e.symbol, []).append(e)
    # clientOrderIds already terminal AFTER reconstruction: place_order's
    # state fast-path returns those without touching the exchange, so an
    # intent carrying one spends nothing this cycle.
    terminal_coids = {
        coid for coid, e in state.all_entries().items()
        if e.status in TERMINAL_STATUSES
    }

    # Phase 2-5: Main cycle. Wrapped so any exception still gets
    # journaled, then re-raised so the cron stderr sees the traceback.
    rebal_date = datetime.now(timezone.utc).date()
    order_results: list[OrderResult] = []
    # Pre-bound so the failure handler can thread read.price_fallbacks
    # into the failed-cycle entry even when the read phase never ran.
    read = None
    try:
        snap = compute_live_signal(
            broker,
            lookbacks=lookbacks,
            sma_filter_period=sma_filter_period,
            candles_per_asset=candles_per_asset,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        decision_age_s = decision_age_seconds(snap.asof)
        if max_decision_age_s is not None and decision_age_s > max_decision_age_s:
            raise SignalComputationError(
                f"Decision bar closed {decision_age_s / 3600:.1f}h ago "
                f"(limit {max_decision_age_s / 3600:.1f}h). The backtest "
                f"trades each signal right after its bar closes; placing "
                f"orders this late is an unvalidated timing. Check the "
                f"cron schedule / host UTC clock; for a deliberate late "
                f"manual entry raise --max-signal-age-h."
            )
        # reserve_cap=False: the cap is applied below, AFTER pending
        # filtering — capping the full plan would scale every buy down
        # for quote that a filtered pending pair never releases.
        read = run_read_phase(broker, snap, log=logger, reserve_cap=False)
        balance = read.balance
        equity = read.equity
        allocation = read.allocation
        plan = read.plan
        current_holdings_quote = read.current_holdings_quote
        quote = broker.config.quote_currency

        # A pair with a live foreign-coid order sits this cycle out: the
        # pending fill is not in the balance, so today's delta for it is
        # computed against a stale position. Same-coid entries pass
        # through — place_order's query-before-place finds that exact
        # order and waits on it, which cannot duplicate.
        pending_skips: list[SkippedDelta] = []
        sendable: list = []
        # Buys that take nothing out of balance.quote_free under THIS
        # cycle's coid, so the reserve cap must leave them alone (see
        # apply_reserve_cap): either the same-coid order is still live —
        # its quote sits in `used`, and place_order resolves that order
        # rather than sending a resized one — or it is already terminal,
        # and place_order's state fast-path places nothing at all.
        # Scaling either only shrinks the buys that DO need funding.
        no_flow_symbols: set[str] = set()
        for intent in plan.orders:
            coid = make_client_order_id(rebal_date, intent.symbol, intent.side)
            pending = pending_by_pair.get(intent.symbol, [])
            blockers = [e for e in pending if e.client_order_id != coid]
            if not blockers:
                sendable.append(intent)
                # A buy takes nothing from quote_free whether its coid is
                # terminal (fast-path places nothing) or still live (the
                # exchange already holds the quote). A SELL is different:
                # only a TERMINAL one is known to add nothing — proceeds
                # already inside quote_free, or never coming. A live
                # same-coid sell is still expected to fund the buys; the
                # executor waits for it before placing them, so excluding
                # it would cap the buys away and underallocate the cycle.
                same_coid_live = any(e.client_order_id == coid for e in pending)
                if intent.side == "buy":
                    no_flow = coid in terminal_coids or same_coid_live
                else:
                    no_flow = coid in terminal_coids
                if no_flow:
                    no_flow_symbols.add(intent.symbol)
                continue
            logger.warning(
                "SKIP %s %s (%.2f %s): order %s from a prior cycle is "
                "still non-terminal on the exchange — planning against "
                "the current balance would double the position once "
                "both fill. Will retry next cycle.",
                intent.side, intent.symbol, intent.notional_quote, quote,
                blockers[0].client_order_id,
            )
            pending_skips.append(SkippedDelta(
                symbol=intent.symbol,
                desired_side=intent.side,
                desired_amount=intent.base_amount,
                desired_notional=intent.notional_quote,
                constraint_min_amount=None,
                constraint_min_cost=None,
                reason="pending_order",
            ))

        # A blocked SELL breaks the sells-fund-buys ordering invariant
        # (sort_orders_for_placement): the buys were sized against equity
        # that counts the blocked pair's base, but its proceeds are not
        # in free quote — the exchange would reject them. Defer them the
        # same transient way instead of collecting the rejection.
        # Unblocked sells still go: they need no funding and reduce risk.
        if any(s.desired_side == "sell" for s in pending_skips):
            deferred_buys = [i for i in sendable if i.side == "buy"]
            sendable = [i for i in sendable if i.side != "buy"]
            for intent in deferred_buys:
                logger.warning(
                    "DEFER buy %s (%.2f %s): a sell this cycle is blocked "
                    "by a pending order, so its proceeds are not in free "
                    "quote. Will retry next cycle.",
                    intent.symbol, intent.notional_quote, quote,
                )
                pending_skips.append(SkippedDelta(
                    symbol=intent.symbol,
                    desired_side=intent.side,
                    desired_amount=intent.base_amount,
                    desired_notional=intent.notional_quote,
                    constraint_min_amount=None,
                    constraint_min_cost=None,
                    reason="pending_funding_sell",
                ))

        # plan.skipped bypasses the loop above, but the same staleness
        # applies: a sub-min delta on a pending pair was computed against
        # a balance the pending fill will change — transient, not
        # unfillable drift.
        submin_skips: list[SkippedDelta] = []
        for s in plan.skipped:
            if pending_by_pair.get(s.symbol):
                pending_skips.append(replace(s, reason="pending_order"))
            else:
                submin_skips.append(s)

        # RESERVE_BPS buy cap (#28) on the post-filter sendable set only:
        # sizing against what actually goes out, not against intents the
        # pending filter just removed.
        sendable, funding_skips = apply_reserve_cap(
            sendable,
            quote_free=balance.quote_free,
            constraints=read.constraints,
            fee_rate=fee_rate,
            no_flow_symbols=no_flow_symbols,
        )

        sorted_intents = sort_orders_for_placement(sendable)
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
        journal_write_failed |= _write_main_cycle(
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
            submin_skips=submin_skips,
            pending_skips=pending_skips,
            funding_skips=funding_skips,
            order_results=order_results,
            decision_age_s=decision_age_s,
            price_fallbacks=read.price_fallbacks or None,
        )
        return LiveCycleResult(
            cycle_id=main_cycle_id,
            outcome=outcome,
            order_results=order_results,
            reconstructed_count=len(reconstructed),
            error=None,
            lost_track_count=lost_track_count,
            journal_write_failed=journal_write_failed,
        )

    except InsufficientWarmupError as exc:
        # Environment branch lives HERE (the orchestrator), never inside
        # signal computation. Sandbox (Binance testnet): the basket
        # structurally cannot warm SMA(200) — the exchange wipes candles
        # ~monthly — so this is a first-class 'skipped_warmup' cycle
        # with an explicit reason block, not an incident: no orders, no
        # plan, exit 0 at the CLI. Mainnet: the identical depth means
        # truncated kline history — keep the H3 posture (failed entry +
        # re-raise) with zero softening.
        context["exchange_latency"] = broker.exchange_call_stats()
        if broker.config.sandbox is not True:
            _write_failed_cycle(
                journal=journal,
                cycle_id=main_cycle_id,
                started_at=started_at,
                context=context,
                exc=exc,
                partial_orders=order_results,
            )
            raise
        skip_reason = exc.reason_dict()
        journal_write_failed |= _write_skipped_warmup_cycle(
            journal=journal,
            cycle_id=main_cycle_id,
            started_at=started_at,
            context=context,
            skip_reason=skip_reason,
        )
        return LiveCycleResult(
            cycle_id=main_cycle_id,
            outcome="skipped_warmup",
            order_results=[],
            reconstructed_count=len(reconstructed),
            error=None,
            # lost_track stays surfaced: a perpetual warm-up skip on
            # testnet must not mute an unresolved order incident.
            lost_track_count=lost_track_count,
            skip_reason=skip_reason,
            journal_write_failed=journal_write_failed,
        )

    except BaseException as exc:
        # BaseException, not Exception: Ctrl-C (KeyboardInterrupt) during
        # the up-to-35-minute wait-for-ack is the single most likely way a
        # manual run dies mid-cycle, and it must not leave already-placed
        # orders without a journal record. Everything below is local and
        # bounded (in-memory stats, git subprocess with timeout, file
        # append) — nothing here can hang a shutdown. The unconditional
        # re-raise preserves the interrupt/exit semantics for the caller.
        context["exchange_latency"] = broker.exchange_call_stats()
        _write_failed_cycle(
            journal=journal,
            cycle_id=main_cycle_id,
            started_at=started_at,
            context=context,
            exc=exc,
            partial_orders=order_results,
            # Partial orders sized on a candle-close fallback must stay
            # attributable on the incident path, not just on success.
            price_fallbacks=(
                (read.price_fallbacks or None) if read is not None else None
            ),
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
            if entry.status == "pending_create":
                # The create request never reached the exchange: nothing
                # was placed, nothing to track. NOT a lost_track incident.
                # The entry is removed (not marked terminal) so a
                # same-day retry's state fast-path cannot skip the real
                # placement.
                logger.info(
                    "Order %s was never created on the exchange "
                    "(pending_create intent, no record found); resolving "
                    "as not_created.", coid,
                )
                state.remove(coid)
                resolved.append({
                    "client_order_id": coid,
                    "exchange_order_id": None,
                    "symbol": entry.symbol,
                    "side": entry.side,
                    "intended_amount": entry.intended_amount,
                    "terminal_status": "not_created",
                    "filled_amount": 0.0,
                    "filled_notional_quote": 0.0,
                    "average_price": None,
                    "fees_paid_quote": None,
                    "placed_at": entry.placed_at,
                    "terminal_at": utcnow_iso(),
                    "error": None,
                })
                continue
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
            if entry.status == "pending_create":
                # Now confirmed on the exchange: upgrade the intent so
                # state reflects what is known.
                state.put(replace(
                    entry,
                    status="open",
                    exchange_order_id=(
                        str(exchange_id) if exchange_id is not None
                        else entry.exchange_order_id
                    ),
                    last_seen_at=utcnow_iso(),
                ))
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
# Reconstruction helpers
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


# ---------------------------------------------------------------------------
# Journal write helpers
# ---------------------------------------------------------------------------


def _log_unwritten(cycle: Cycle) -> None:
    # A failed append discards the payload — the process log is the
    # recovery path. The text after the first '{' is the exact journal
    # line; the default JSON log formatter wraps it (escaped) in the
    # record's msg field, so recovery extracts, never copies the raw
    # log line.
    try:
        logger.error(
            "Unwritten journal entry payload %s: %s",
            cycle.cycle_id,
            _encode_cycle(cycle).decode("utf-8").rstrip("\n"),
        )
    except Exception:
        logger.error(
            "Could not serialize unwritten journal entry %s", cycle.cycle_id,
        )


def _write_reconstruction_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    reconstructed: list[dict],
) -> bool:
    """One Cycle entry per orchestrator run that had something to recover.

    Kept separate from the main cycle's entry: reconstructed orders are
    a *prior* cycle's work, not the current run's. Mixing them muddies
    the audit trail when reconciling backtest vs reality later.

    Returns True when the append failed (logged, not raised — the run
    must go on; the flag reaches the CLI via LiveCycleResult).
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
        _log_unwritten(cycle)
        return True
    return False


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
    submin_skips: list[SkippedDelta],
    pending_skips: list[SkippedDelta],
    funding_skips: list[SkippedDelta],
    order_results: list[OrderResult],
    decision_age_s: float,
    price_fallbacks: Optional[dict] = None,
) -> bool:
    """Returns True when the append failed — orders are already placed,
    so the failure is surfaced via LiveCycleResult, not raised."""
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
        signal=signal_dict(snap, decision_age_s=decision_age_s),
        basket_close_series=basket_close_series_dict(snap.basket_close_tail),
        balance=balance_dict(balance),
        equity_usd=equity,
        target_allocation=dict(allocation.target_quote_per_asset),
        current_holdings_quote=current_holdings_quote,
        orders_planned=[intent_dict(o) for o in plan.orders],
        orders_skipped=[
            skipped_dict(s)
            for s in submin_skips + pending_skips + funding_skips
        ],
        # No pending_skips here: a pending_order skip is transient
        # (retried next cycle), while this metric tracks quote the
        # exchange refused this cycle. Funding-cap shaves are metered
        # apart — they are a deliberate reserve on valid orders, and the
        # monitoring layer reads this field as "unfillable".
        total_skipped_quote_drift=total_skipped_quote_drift(
            replace(plan, skipped=submin_skips)
        ),
        total_funding_cap_quote=total_funding_cap_quote(
            replace(plan, skipped=funding_skips)
        ),
        orders_executed=[r.to_dict() for r in order_results],
        price_fallbacks=price_fallbacks,
    )
    try:
        journal.append(cycle)
    except Exception as exc:
        logger.error(
            "Could not write main cycle journal entry %s: %s", cycle_id, exc,
        )
        _log_unwritten(cycle)
        return True
    return False


def _write_failed_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    exc: BaseException,
    partial_orders: list[OrderResult],
    price_fallbacks: Optional[dict] = None,
) -> None:
    cycle = failed_cycle(
        cycle_id, started_at, datetime.now(timezone.utc), context, exc,
        orders_executed=(
            [r.to_dict() for r in partial_orders] if partial_orders else None
        ),
        price_fallbacks=price_fallbacks,
    )
    try:
        journal.append(cycle)
    except Exception as journal_exc:
        # No journal_write_failed flag to set: the caller re-raises the
        # original error, so no LiveCycleResult exists on this path and
        # the CLI already exits non-zero.
        logger.error(
            "Could not write failed-cycle journal entry %s: %s",
            cycle_id, journal_exc,
        )
        _log_unwritten(cycle)


def _write_skipped_warmup_cycle(
    *,
    journal: JournalWriter,
    cycle_id: str,
    started_at: datetime,
    context: dict,
    skip_reason: dict,
) -> bool:
    """Returns True when the append failed (surfaced via LiveCycleResult)."""
    cycle = skipped_warmup_cycle(
        cycle_id, started_at, datetime.now(timezone.utc), context,
        skip_reason=skip_reason,
        orders_executed=[],
    )
    try:
        journal.append(cycle)
    except Exception as journal_exc:
        logger.error(
            "Could not write skipped_warmup journal entry %s: %s",
            cycle_id, journal_exc,
        )
        _log_unwritten(cycle)
        return True
    return False
