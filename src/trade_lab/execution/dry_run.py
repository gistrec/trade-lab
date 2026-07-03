"""Dry-run orchestration: fetch, compute, plan, print. NO orders sent.

This is the last step before live order placement. It wires:

1. :func:`compute_live_signal` — pulls fresh candles, runs the
   deployable strategy, returns the ladder signal.
2. :class:`Broker` — pulls live balance and ticker prices.
3. :func:`compute_target_allocation` — turns signal into target qty.
4. :func:`compute_delta_plan` — produces sendable orders + skipped
   sub-minimum deltas.

It prints what it WOULD do but does not call ``broker.exchange.create_order``.
Running this against the testnet during paper-trading week 1 is the
recommended sanity check before flipping the order switch.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import pandas as pd

from .allocator import compute_target_allocation
from .broker import BalanceSnapshot, Broker, BrokerError, MarketConstraints
from .delta import compute_delta_plan, total_skipped_quote_drift
from .journal import (
    Cycle, JournalWriter, get_git_commit_short, get_python_version,
    new_cycle_id,
)
from .signal import SignalSnapshot, compute_live_signal


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DryRunResult:
    """One-cycle output of :func:`run_dry_cycle`. Easy to dump to log
    or to JSON for the reconciliation logger."""

    asof: pd.Timestamp
    signal: float
    sma_gate_open: bool
    total_equity: float
    target_allocation: dict[str, float]   # asset -> target quote
    current_holdings_quote: dict[str, float]  # asset -> current quote
    orders_planned: list[dict]            # serialized OrderIntent
    orders_skipped: list[dict]            # serialized SkippedDelta
    total_skipped_quote_drift: float      # cumulative sub-min divergence


def run_dry_cycle(
    broker: Broker,
    *,
    lookbacks: Sequence[int] = (28, 60),
    sma_filter_period: int = 200,
    candles_per_asset: int = 400,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    journal: Optional[JournalWriter] = None,
) -> DryRunResult:
    """Execute one full dry-run cycle and return the structured plan.

    If ``journal`` is provided, exactly one :class:`Cycle` is appended
    per call — success or failure. A failed cycle still gets a journal
    entry so monitoring can surface the incident; the exception is then
    re-raised so the caller is not left thinking everything went fine.
    A journal-write failure is logged and swallowed: it must not mask
    or replace the operational outcome of the cycle itself.
    """
    cycle_id = new_cycle_id()
    started_at = datetime.now(timezone.utc)
    context = _build_context(broker)

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

        # Use the broker's ticker prices, not the candle closes from the
        # signal step — the broker's prices are the freshest and reflect
        # what the order will actually fill against.
        ticker_prices: dict[str, float] = {}
        quote = broker.config.quote_currency
        for sym in broker.config.basket:
            try:
                ticker_prices[sym] = broker.fetch_ticker_price(f"{sym}/{quote}")
            except BrokerError as exc:
                logger.warning("Ticker for %s failed: %s — using candle close.", sym, exc)
                # Direct indexing: a missing basket symbol must raise
                # here, not become a 0.0 price inside the allocator.
                ticker_prices[sym] = snap.asset_closes[sym]

        allocation = compute_target_allocation(
            signal=snap.signal,
            total_equity=equity,
            prices=ticker_prices,
            basket=broker.config.basket,
            weights=snap.basket_weights,
        )

        constraints = _gather_constraints(broker, broker.config.basket, quote)

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

        result = DryRunResult(
            asof=snap.asof,
            signal=snap.signal,
            sma_gate_open=snap.sma_gate_open,
            total_equity=equity,
            target_allocation=allocation.target_quote_per_asset,
            current_holdings_quote=current_holdings_quote,
            orders_planned=[
                {"symbol": o.symbol, "side": o.side, "base_amount": o.base_amount,
                 "notional_quote": o.notional_quote, "price_used": o.price_used}
                for o in plan.orders
            ],
            orders_skipped=[
                {"symbol": s.symbol, "desired_side": s.desired_side,
                 "desired_amount": s.desired_amount,
                 "desired_notional": s.desired_notional,
                 "reason": s.reason}
                for s in plan.skipped
            ],
            total_skipped_quote_drift=total_skipped_quote_drift(plan),
        )
    except Exception as exc:
        if journal is not None:
            _try_write(journal, _failed_cycle(cycle_id, started_at, context, exc))
        raise

    if journal is not None:
        _try_write(
            journal,
            _success_cycle(
                cycle_id, started_at, context, snap, balance, equity, result,
            ),
        )
    return result


def _build_context(broker: Broker) -> dict:
    return {
        # Durable live/dry marker for read-only monitoring (the health
        # server), mirroring live_cycle._build_context. Metadata only.
        "mode": "dry_run",
        "exchange": broker.config.exchange_id,
        "sandbox": broker.config.sandbox,
        "quote_currency": broker.config.quote_currency,
        "basket": list(broker.config.basket),
    }


def _try_write(journal: JournalWriter, cycle: Cycle) -> None:
    """Append a cycle; log and swallow any error.

    A journal-write failure must not hide or alter the operational
    outcome of the cycle. If we can't journal, we still want to
    surface the actual cycle exception (or successful return).
    """
    try:
        journal.append(cycle)
    except Exception as exc:
        logger.error("Could not write journal entry %s: %s", cycle.cycle_id, exc)


def _failed_cycle(
    cycle_id: str, started_at: datetime, context: dict, exc: BaseException,
) -> Cycle:
    ended_at = datetime.now(timezone.utc)
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
    )


def _success_cycle(
    cycle_id: str,
    started_at: datetime,
    context: dict,
    snap: SignalSnapshot,
    balance: BalanceSnapshot,
    equity: float,
    result: DryRunResult,
) -> Cycle:
    ended_at = datetime.now(timezone.utc)
    return Cycle(
        cycle_id=cycle_id,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        duration_ms=int((ended_at - started_at).total_seconds() * 1000),
        outcome="success",
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
        basket_close_series=_basket_close_series(snap.basket_close_tail),
        balance={
            "quote_currency": balance.quote_currency,
            "quote_total": balance.quote_total,
            "quote_free": balance.quote_free,
            "quote_used": balance.quote_used,
            "asset_totals": balance.asset_totals,
        },
        equity_usd=equity,
        target_allocation=dict(result.target_allocation),
        current_holdings_quote=dict(result.current_holdings_quote),
        orders_planned=list(result.orders_planned),
        orders_skipped=list(result.orders_skipped),
        total_skipped_quote_drift=result.total_skipped_quote_drift,
    )


def _ts_iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _basket_close_series(tail) -> Optional[dict]:
    if tail is None or len(tail) == 0:
        return None
    # 6 decimals on an index normalized to start at 100 — full-precision
    # floats here roughly double the serialized cycle size for nothing.
    return {
        "start_ts": _ts_iso(tail.index[0]),
        "values": [round(float(v), 6) for v in tail.tolist()],
    }


def _gather_constraints(
    broker: Broker, basket: Sequence[str], quote: str,
) -> dict[str, MarketConstraints]:
    """Pull min-amount / min-cost constraints for every basket pair.

    A pair that fails to load constraints is **excluded from the
    constraint map** so the delta planner treats it as "trust the
    allocator" rather than blocking. Logged as a warning so the
    operator sees which pairs lacked exchange-side metadata.
    """
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


def print_dry_run(result: DryRunResult, *, quote: str) -> None:
    """Human-readable summary for the CLI."""
    print(f"As-of:                {result.asof}")
    print(f"Signal:               {result.signal:.2f} (ladder {{0, 0.5, 1.0}})")
    print(f"SMA(200) gate:        {'open' if result.sma_gate_open else 'closed'}")
    print(f"Total equity:         {result.total_equity:,.2f} {quote}")
    print()
    print(f"  {'asset':6s}{'target ' + quote:>15s}{'current ' + quote:>15s}{'delta':>15s}")
    for sym in result.target_allocation:
        tgt = result.target_allocation[sym]
        cur = result.current_holdings_quote.get(sym, 0.0)
        print(f"  {sym:6s}{tgt:>15,.2f}{cur:>15,.2f}{(tgt - cur):>+15,.2f}")
    print()
    if result.orders_planned:
        print(f"Orders planned ({len(result.orders_planned)}):")
        for o in result.orders_planned:
            print(f"  {o['side'].upper():4s} {o['symbol']:12s} "
                  f"{o['base_amount']:.8f}  "
                  f"({o['notional_quote']:.2f} {quote})")
    else:
        print("Orders planned: (none — target matches current within minima)")
    print()
    if result.orders_skipped:
        print(f"Sub-min divergence ({len(result.orders_skipped)}, "
              f"cumulative {result.total_skipped_quote_drift:.2f} {quote}):")
        for s in result.orders_skipped:
            print(f"  SKIP {s['desired_side'].upper():4s} {s['symbol']:12s} "
                  f"{s['desired_amount']:.8f}  ({s['desired_notional']:.2f} "
                  f"{quote})  reason: {s['reason']}")
    else:
        print("Sub-min divergence: 0.00 — no tracking drift this cycle.")
