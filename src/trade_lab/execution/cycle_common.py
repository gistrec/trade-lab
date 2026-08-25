"""Read-phase and journal-cycle helpers shared by the two cycle
orchestrators (:mod:`dry_run`, :mod:`live_cycle`).

Broker reads and pure serialization only: no order placement, no journal
writes, no exception handling — each orchestrator keeps its own control
flow, failure posture, and journal-write error messages. Helpers that
log take the caller's logger so records keep the orchestrator's logger
name; builders take ``ended_at`` so ``datetime.now`` stays resolvable
(and patchable) in the calling module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from .allocator import TargetAllocation, compute_target_allocation
from .broker import BalanceSnapshot, Broker, BrokerError, MarketConstraints
from .delta import DeltaPlan, compute_delta_plan
from .journal import Cycle, get_git_commit_short, get_python_version
from .signal import SignalSnapshot, decision_age_seconds


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read phase
# ---------------------------------------------------------------------------


def build_context(broker: Broker, *, mode: str) -> dict:
    return {
        # Durable live/dry marker for read-only monitoring (the health
        # server). Present even when a cycle fails before placing an
        # order, so a failed live attempt is never misread as a dry-run.
        # Metadata only.
        "mode": mode,
        "exchange": broker.config.exchange_id,
        "sandbox": broker.config.sandbox,
        "quote_currency": broker.config.quote_currency,
        "basket": list(broker.config.basket),
    }


def gather_ticker_prices(
    broker: Broker,
    snap: SignalSnapshot,
    *,
    log: logging.Logger = logger,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Live ticker prices, falling back to candle close on per-pair failure.

    Returns ``(prices, fallbacks)``: ``fallbacks`` marks every symbol
    priced from the candle close (source + price age in seconds) so the
    journal can attribute a stale-priced sizing decision post-hoc.
    """
    ticker_prices: dict[str, float] = {}
    fallbacks: dict[str, dict] = {}
    quote = broker.config.quote_currency
    for sym in broker.config.basket:
        try:
            ticker_prices[sym] = broker.fetch_ticker_price(f"{sym}/{quote}")
        except BrokerError as exc:
            log.warning(
                "Ticker for %s failed: %s — using candle close.", sym, exc,
            )
            # Direct indexing: a basket symbol missing from the signal's
            # closes is an invariant violation that must raise HERE, not
            # surface as a 0.0 price deep inside the allocator.
            ticker_prices[sym] = snap.asset_closes[sym]
            fallbacks[sym] = {
                "source": "candle_close_fallback",
                # asof is the bar's OPEN; its close (the price we just
                # substituted) is a day later — decision_age_seconds
                # measures from that close.
                "age_s": round(decision_age_seconds(snap.asof), 1),
            }
    return ticker_prices, fallbacks


def gather_constraints(
    broker: Broker,
    basket: Sequence[str],
    quote: str,
    *,
    log: logging.Logger = logger,
) -> dict[str, MarketConstraints]:
    """Min-amount / min-cost constraints for every basket pair.

    A pair that fails to load is excluded from the map so the delta
    planner treats it as "trust the allocator" rather than blocking;
    the warning tells the operator which pairs lacked metadata.
    """
    constraints: dict[str, MarketConstraints] = {}
    for sym in basket:
        pair = f"{sym}/{quote}"
        try:
            constraints[pair] = broker.fetch_market_constraints(pair)
        except BrokerError as exc:
            log.warning(
                "Constraints for %s unavailable: %s — sub-min filter "
                "disabled for this pair.", pair, exc,
            )
    return constraints


@dataclass(frozen=True)
class ReadPhase:
    """Everything the read phase produces; inputs to planning/journal."""

    balance: BalanceSnapshot
    equity: float
    ticker_prices: dict[str, float]
    price_fallbacks: dict[str, dict]   # empty when every ticker succeeded
    allocation: TargetAllocation
    constraints: dict[str, MarketConstraints]
    plan: DeltaPlan
    current_holdings_quote: dict[str, float]


def run_read_phase(
    broker: Broker,
    snap: SignalSnapshot,
    *,
    log: logging.Logger = logger,
) -> ReadPhase:
    """Balance → equity → tickers → allocation → constraints → delta plan
    → holdings-in-quote. Read-only; any step may raise (fail loud)."""
    balance = broker.fetch_balance_snapshot()
    equity = broker.estimate_total_equity_usd(snapshot=balance)

    # The broker's ticker prices, not the candle closes from the signal
    # step — they are the freshest and reflect what the order will
    # actually fill against.
    ticker_prices, price_fallbacks = gather_ticker_prices(broker, snap, log=log)

    allocation = compute_target_allocation(
        signal=snap.signal,
        total_equity=equity,
        prices=ticker_prices,
        basket=broker.config.basket,
        weights=snap.basket_weights,
    )

    quote = broker.config.quote_currency
    constraints = gather_constraints(broker, broker.config.basket, quote, log=log)

    plan = compute_delta_plan(
        allocation=allocation,
        current_holdings=balance.asset_totals,
        constraints=constraints,
        quote_currency=quote,
    )

    current_holdings_quote = {
        sym: float(balance.asset_totals.get(sym, 0.0)) * ticker_prices[sym]
        for sym in broker.config.basket
    }

    return ReadPhase(
        balance=balance,
        equity=equity,
        ticker_prices=ticker_prices,
        price_fallbacks=price_fallbacks,
        allocation=allocation,
        constraints=constraints,
        plan=plan,
        current_holdings_quote=current_holdings_quote,
    )


# ---------------------------------------------------------------------------
# Journal serialization
# ---------------------------------------------------------------------------


def ts_iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def basket_close_series_dict(tail) -> Optional[dict]:
    if tail is None or len(tail) == 0:
        return None
    # 6 decimals on an index normalized to start at 100 — full-precision
    # floats here roughly double the serialized cycle size for nothing.
    return {
        "start_ts": ts_iso(tail.index[0]),
        "values": [round(float(v), 6) for v in tail.tolist()],
    }


def signal_dict(snap: SignalSnapshot, *, decision_age_s: float) -> dict:
    return {
        "asof": ts_iso(snap.asof),
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
        "decision_age_s": round(decision_age_s, 1),
    }


def balance_dict(balance: BalanceSnapshot) -> dict:
    return {
        "quote_currency": balance.quote_currency,
        "quote_total": balance.quote_total,
        "quote_free": balance.quote_free,
        "quote_used": balance.quote_used,
        "asset_totals": balance.asset_totals,
    }


def intent_dict(intent) -> dict:
    return {
        "symbol": intent.symbol,
        "side": intent.side,
        "base_amount": intent.base_amount,
        "notional_quote": intent.notional_quote,
        "price_used": intent.price_used,
    }


def skipped_dict(skipped) -> dict:
    return {
        "symbol": skipped.symbol,
        "desired_side": skipped.desired_side,
        "desired_amount": skipped.desired_amount,
        "desired_notional": skipped.desired_notional,
        "reason": skipped.reason,
    }


# ---------------------------------------------------------------------------
# Journal cycle builders (no writes — callers own append + its errors)
# ---------------------------------------------------------------------------


def failed_cycle(
    cycle_id: str,
    started_at: datetime,
    ended_at: datetime,
    context: dict,
    exc: BaseException,
    *,
    orders_executed: Optional[list] = None,
) -> Cycle:
    """Failed cycle still gets a journal entry — silently dropping is
    the failure mode the journal exists to prevent. ``orders_executed``
    carries any partially-placed orders (live only) so they are not
    lost to history.
    """
    return Cycle(
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
        orders_executed=orders_executed,
    )


def skipped_warmup_cycle(
    cycle_id: str,
    started_at: datetime,
    ended_at: datetime,
    context: dict,
    *,
    skip_reason: dict,
    orders_executed: Optional[list] = None,
) -> Cycle:
    """Sandbox-only first-class skip (SMA warm-up structurally impossible
    on testnet — ~monthly candle wipe). ``error`` stays None — monitoring
    and health checks key incidents off ``outcome``/``error`` — and the
    depth facts live in the explicit ``skip_reason`` block.

    ``orders_executed`` is ``[]`` for the live cycle (a list, like every
    main live cycle, so the live-cron freshness clock and /healthz/daily
    do not misread a perpetually-skipping testnet as a dead cron) and
    ``None`` for a dry run.
    """
    return Cycle(
        cycle_id=cycle_id,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_ms=int((ended_at - started_at).total_seconds() * 1000),
        outcome="skipped_warmup",
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
        orders_executed=orders_executed,
        skip_reason=skip_reason,
    )
