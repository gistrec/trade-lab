"""Journal-cycle helpers shared by the two cycle orchestrators
(:mod:`dry_run`, :mod:`live_cycle`).

Pure serialization only: no journal writes, no exception handling —
each orchestrator keeps its own control flow, failure posture, and
journal-write error messages. Builders take ``ended_at`` so
``datetime.now`` stays resolvable (and patchable) in the calling module.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .broker import BalanceSnapshot, Broker
from .journal import Cycle, get_git_commit_short, get_python_version
from .signal import SignalSnapshot


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
