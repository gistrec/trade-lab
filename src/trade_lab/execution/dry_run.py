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

from .broker import BalanceSnapshot, Broker
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
from .delta import total_skipped_quote_drift
from .journal import (
    Cycle, JournalWriter, get_git_commit_short, get_python_version,
    new_cycle_id,
)
from .signal import (
    InsufficientWarmupError,
    SignalSnapshot,
    compute_live_signal,
    decision_age_seconds,
)
from ..logging_setup import set_cycle_id


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

    Special case: :class:`InsufficientWarmupError` on a sandbox config
    journals ``outcome='skipped_warmup'`` (testnet's ~monthly candle
    wipe makes the warm-up structurally impossible — a first-class
    skip, not an incident) but is still re-raised: no plan exists, and
    the CLI turns it into a friendly exit-0 message. On mainnet the
    same error keeps the failed-entry posture unchanged.
    """
    cycle_id = new_cycle_id()
    set_cycle_id(cycle_id)  # tag every log line in this cycle with the id
    started_at = datetime.now(timezone.utc)
    context = build_context(broker, mode="dry_run")

    try:
        snap = compute_live_signal(
            broker,
            lookbacks=lookbacks,
            sma_filter_period=sma_filter_period,
            candles_per_asset=candles_per_asset,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        # Journaled only: a 6-hourly dry-run watch legitimately runs up
        # to ~18h after the decision bar's close, so no threshold here —
        # the live cycle's gate owns the hard limit. Stale DATA (a
        # missing newest bar) already raised inside compute_live_signal.
        decision_age_s = decision_age_seconds(snap.asof)
        logger.info(
            "signal decision age %.0fs", decision_age_s,
            extra={"decision_age_s": round(decision_age_s, 1)},
        )

        read = run_read_phase(broker, snap, log=logger)
        balance = read.balance
        equity = read.equity

        result = DryRunResult(
            asof=snap.asof,
            signal=snap.signal,
            sma_gate_open=snap.sma_gate_open,
            total_equity=equity,
            target_allocation=read.allocation.target_quote_per_asset,
            current_holdings_quote=read.current_holdings_quote,
            orders_planned=[intent_dict(o) for o in read.plan.orders],
            orders_skipped=[skipped_dict(s) for s in read.plan.skipped],
            total_skipped_quote_drift=total_skipped_quote_drift(read.plan),
        )
    except InsufficientWarmupError as exc:
        # Environment branch lives HERE (the orchestrator), never inside
        # signal computation. On the sandbox (Binance testnet) an
        # unwarmed SMA is structural — the exchange wipes candles
        # ~monthly, so the basket can never reach 200 bars — and the
        # cycle is journaled as a first-class 'skipped_warmup' with an
        # explicit reason block, not as an incident. On mainnet the
        # same depth means truncated kline history: keep the H3 posture
        # (outcome='failed') without any softening. Re-raised either
        # way — no plan exists, and the caller must not think one does.
        if journal is not None:
            context["exchange_latency"] = broker.exchange_call_stats()
            ended_at = datetime.now(timezone.utc)
            if broker.config.sandbox is True:
                cycle = skipped_warmup_cycle(
                    cycle_id, started_at, ended_at, context,
                    skip_reason=exc.reason_dict(),
                )
            else:
                cycle = failed_cycle(cycle_id, started_at, ended_at, context, exc)
            _try_write(journal, cycle)
        raise
    except BaseException as exc:
        # BaseException, not Exception: Ctrl-C (KeyboardInterrupt) mid-cycle
        # must still leave a failed-cycle journal entry (mirrors
        # live_cycle.run_live_cycle). The write path is local and bounded;
        # the unconditional re-raise preserves interrupt/exit semantics.
        if journal is not None:
            context["exchange_latency"] = broker.exchange_call_stats()
            _try_write(
                journal,
                failed_cycle(
                    cycle_id, started_at, datetime.now(timezone.utc),
                    context, exc,
                ),
            )
        raise

    if journal is not None:
        # Read-only exchange round-trip telemetry for the /metrics exporter.
        context["exchange_latency"] = broker.exchange_call_stats()
        _try_write(
            journal,
            _success_cycle(
                cycle_id, started_at, context, snap, balance, equity, result,
                decision_age_s=decision_age_s,
            ),
        )
    return result


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


def _success_cycle(
    cycle_id: str,
    started_at: datetime,
    context: dict,
    snap: SignalSnapshot,
    balance: BalanceSnapshot,
    equity: float,
    result: DryRunResult,
    *,
    decision_age_s: float,
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
        signal=signal_dict(snap, decision_age_s=decision_age_s),
        basket_close_series=basket_close_series_dict(snap.basket_close_tail),
        balance=balance_dict(balance),
        equity_usd=equity,
        target_allocation=dict(result.target_allocation),
        current_holdings_quote=dict(result.current_holdings_quote),
        orders_planned=list(result.orders_planned),
        orders_skipped=list(result.orders_skipped),
        total_skipped_quote_drift=result.total_skipped_quote_drift,
    )


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
