"""Order placement primitives: idempotency, wait-for-ack, reconstruction.

Principles
==========
* The exchange is the single source of truth. Local state is a hint
  for "should I skip placement?" — never for the order's actual fill
  details. Final OrderResult fields come from a fresh ``fetch_order``.
* Idempotency via clientOrderId. Two calls to :func:`place_order` with
  the same client order ID never produce two distinct exchange orders:
  the second one observes the first and integrates it.
* Business outcomes (rejected, partial, timeout) come back as
  :class:`OrderResult` — never raised. Network and unexpected exchange
  errors propagate; the caller decides how to journal them.

Timing
======
Wait-for-ack budget is ``TOTAL_TIMEOUT_S`` (5 minutes) with exponential
backoff ``POLL_INITIAL_S`` → ``POLL_MAX_S``. Fits comfortably inside a
once-per-day cycle even with all 7 basket symbols and a transient
network hiccup or two.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import ccxt

from .broker import Broker
from .delta import OrderIntent
from .order_state import (
    OrderStateEntry,
    OrderStateStore,
    TERMINAL_STATUSES,
    utcnow_iso,
)


logger = logging.getLogger(__name__)


POLL_INITIAL_S = 1.0
POLL_MAX_S = 30.0
TOTAL_TIMEOUT_S = 300.0

# CCXT status strings that represent a finished order on the exchange.
# We map these onto our own terminal taxonomy in :func:`_result_from_order`.
# 'expired' and 'rejected' are terminal too: ccxt maps Binance EXPIRED /
# EXPIRED_IN_MATCH → 'expired' (a market order against an empty book —
# routine on testnet — or a self-trade-prevention cancel on mainnet) and
# REJECTED → 'rejected'. Treating them as non-terminal burns the whole
# wait budget on an order that will never change and journals a false
# 'timeout' whose state entry no later cycle can resolve.
_CCXT_TERMINAL = frozenset({"closed", "canceled", "expired", "rejected"})


@dataclass(frozen=True)
class OrderResult:
    """Outcome of one order placement. Goes into journal v2 verbatim.

    Failures (rejected, timeout, lost_track) are first-class entries —
    silently dropping them would hide exactly the incidents the journal
    exists to surface.
    """

    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    side: str
    intended_amount: float
    terminal_status: str
    filled_amount: float
    filled_notional_quote: float
    average_price: Optional[float]
    fees_paid_quote: Optional[float]
    # None = the exchange reported no fee info (Binance spot
    # fetch_order never does — fees only surface via trades). 0.0 is
    # reserved for "reported and actually zero".
    placed_at: str
    terminal_at: Optional[str]
    error: Optional[dict]
    fees_reported: Optional[list] = None
    # Verbatim [{cost, currency}, ...] as reported. A market BUY pays
    # its fee in the BASE asset, which must not be summed into a
    # quote-denominated number.

    def to_dict(self) -> dict:
        """JSON-serializable form for the journal."""
        return {
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "intended_amount": self.intended_amount,
            "terminal_status": self.terminal_status,
            "filled_amount": self.filled_amount,
            "filled_notional_quote": self.filled_notional_quote,
            "average_price": self.average_price,
            "fees_paid_quote": self.fees_paid_quote,
            "placed_at": self.placed_at,
            "terminal_at": self.terminal_at,
            "error": self.error,
            "fees_reported": self.fees_reported,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def place_order(
    broker: Broker,
    intent: OrderIntent,
    *,
    client_order_id: str,
    state: OrderStateStore,
    poll_initial_s: float = POLL_INITIAL_S,
    poll_max_s: float = POLL_MAX_S,
    total_timeout_s: float = TOTAL_TIMEOUT_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> OrderResult:
    """Place one order with exchange-driven idempotency + wait-for-ack.

    Algorithm:

    1. **State fast-path**: if state says ``client_order_id`` is already
       terminal, return without touching the exchange. Caller is on the
       hook for understanding that ``filled_amount`` from this fast
       path may be 0 — the state file does not cache fill details.
    2. **Query-before-place**: ``fetch_order_by_coid``. If the exchange
       knows the ID, wait for it to terminalize; never call
       ``create_order``.
    3. **Create**: ``create_order_safe`` with the deterministic ID.
       Business rejections (``InvalidOrder`` / ``InsufficientFunds`` /
       ``BadRequest``) come back as ``terminal_status='rejected'``.
       Network and unexpected exchange errors propagate.
    4. **Wait for terminal**: bounded exponential backoff.
    5. **Persist**: update state.

    The state-cache shortcut is here because a re-run of the daily cron
    on the same UTC date must complete fast — without it, an idle cycle
    fires 7 ``fetch_order`` calls just to confirm nothing changed.
    """
    cached = state.get(client_order_id)
    if cached is not None and cached.status in TERMINAL_STATUSES:
        logger.info(
            "Order %s already terminal per state cache (status=%s); "
            "skipping exchange roundtrip.",
            client_order_id, cached.status,
        )
        return _result_from_cached(cached)

    try:
        existing = broker.fetch_order_by_coid(client_order_id, intent.symbol)
    except ccxt.OrderNotFound:
        existing = None
    # NetworkError / ExchangeError propagate by design.

    if existing is not None:
        logger.info(
            "Order %s already exists on exchange (id=%s, status=%s); "
            "waiting for terminal.",
            client_order_id, existing.get("id"), existing.get("status"),
        )
        placed_at = _placed_at_from_order(existing)
        final_order = _wait_or_use_terminal(
            broker, client_order_id, intent.symbol, existing,
            poll_initial_s=poll_initial_s, poll_max_s=poll_max_s,
            total_timeout_s=total_timeout_s,
            sleep_fn=sleep_fn, time_fn=time_fn,
        )
        result = _result_from_order(intent, client_order_id, final_order, placed_at)
        _persist(state, intent, client_order_id, result)
        return result

    placed_at = utcnow_iso()
    try:
        new_order = broker.create_order_safe(
            intent.symbol, intent.side, intent.base_amount, client_order_id,
        )
    except (ccxt.InvalidOrder, ccxt.InsufficientFunds, ccxt.BadRequest) as exc:
        logger.warning(
            "Order %s rejected by exchange: %s: %s",
            client_order_id, type(exc).__name__, exc,
        )
        result = OrderResult(
            client_order_id=client_order_id,
            exchange_order_id=None,
            symbol=intent.symbol,
            side=intent.side,
            intended_amount=intent.base_amount,
            terminal_status="rejected",
            filled_amount=0.0,
            filled_notional_quote=0.0,
            average_price=None,
            fees_paid_quote=0.0,
            placed_at=placed_at,
            terminal_at=utcnow_iso(),
            error={"type": type(exc).__name__, "message": str(exc)[:512]},
        )
        _persist(state, intent, client_order_id, result)
        return result

    final_order = _wait_or_use_terminal(
        broker, client_order_id, intent.symbol, new_order,
        poll_initial_s=poll_initial_s, poll_max_s=poll_max_s,
        total_timeout_s=total_timeout_s,
        sleep_fn=sleep_fn, time_fn=time_fn,
    )
    result = _result_from_order(intent, client_order_id, final_order, placed_at)
    _persist(state, intent, client_order_id, result)
    return result


def wait_for_terminal(
    broker: Broker,
    client_order_id: str,
    symbol: str,
    *,
    poll_initial_s: float = POLL_INITIAL_S,
    poll_max_s: float = POLL_MAX_S,
    total_timeout_s: float = TOTAL_TIMEOUT_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> dict:
    """Poll ``fetch_order`` until terminal or until the budget elapses.

    Backoff schedule: ``poll_initial_s`` doubling each tick, capped at
    ``poll_max_s``. Total elapsed time is bounded by ``total_timeout_s``.

    Returns the final ccxt order dict. On timeout returns the *last
    observed* dict — its status will be non-terminal and the caller is
    responsible for marking the result accordingly.
    """
    delay = poll_initial_s
    start = time_fn()
    last_order: dict = {}
    while True:
        last_order = broker.fetch_order_by_coid(client_order_id, symbol)
        status = last_order.get("status")
        if status in _CCXT_TERMINAL:
            return last_order
        elapsed = time_fn() - start
        if elapsed >= total_timeout_s:
            logger.warning(
                "Order %s did not terminalize within %.0fs "
                "(last status=%s, elapsed=%.1fs).",
                client_order_id, total_timeout_s, status, elapsed,
            )
            return last_order
        sleep_fn(delay)
        delay = min(delay * 2, poll_max_s)


def reconstruct_status(
    broker: Broker,
    client_order_id: str,
    symbol: str,
    *,
    exchange_order_id: Optional[str] = None,
    since_ms: Optional[int] = None,
) -> Optional[dict]:
    """Recover the terminal state of an order we may have lost track of.

    Two-step discovery:

    1. ``fetch_order_by_coid`` — if Binance still has the order record
       (90+ day retention), we get the canonical state.
    2. Fallback: ``fetch_my_trades`` filtered by symbol since
       ``since_ms``. We match trades by the exchange order id
       (``exchange_order_id``) primarily — Binance ``myTrades`` carry the
       exchange orderId (ccxt ``order``) but NOT a clientOrderId, so a
       clientOrderId-only match never fires there — falling back to
       clientOrderId for exchanges that do populate it. Matching trades
       are aggregated into an order-like dict.

    Returns ``None`` only when the exchange has no record of the ID at
    all — the caller marks the state ``lost_track`` and surfaces it
    loudly.
    """
    try:
        return broker.fetch_order_by_coid(client_order_id, symbol)
    except ccxt.OrderNotFound:
        pass  # fall through to trade-based discovery

    trades = broker.fetch_my_trades_since(symbol, since_ms)
    matching = [
        t for t in trades
        if (
            exchange_order_id is not None
            and t.get("order") is not None
            and str(t.get("order")) == str(exchange_order_id)
        )
        or (t.get("info") or {}).get("clientOrderId") == client_order_id
        or t.get("clientOrderId") == client_order_id
    ]
    if not matching:
        return None

    total_amount = sum(float(t.get("amount") or 0) for t in matching)
    total_cost = sum(float(t.get("cost") or 0) for t in matching)
    total_fee = sum(
        float((t.get("fee") or {}).get("cost") or 0) for t in matching
    )
    avg_price = total_cost / total_amount if total_amount > 0 else None
    last_trade = max(matching, key=lambda t: t.get("timestamp") or 0)

    return {
        "id": last_trade.get("order"),
        "clientOrderId": client_order_id,
        "symbol": symbol,
        "side": last_trade.get("side"),
        # Trades exist → the order partially or fully filled → it's
        # closed from the exchange's perspective.
        "status": "closed",
        "filled": total_amount,
        "cost": total_cost,
        "average": avg_price,
        "fee": {
            "cost": total_fee,
            "currency": (last_trade.get("fee") or {}).get("currency"),
        },
        "timestamp": last_trade.get("timestamp"),
    }


def fees_from_order(order: dict, quote_currency: str) -> tuple[Optional[float], Optional[list]]:
    """Extract reported fees from a ccxt order dict.

    Returns ``(fees_paid_quote, fees_reported)``:

    * ``fees_reported`` — verbatim ``[{cost, currency}, ...]``, or
      ``None`` when the exchange reported nothing. Binance spot
      ``fetch_order`` reports no fee at all; fee info only arrives via
      trade-based reconstruction.
    * ``fees_paid_quote`` — sum of the entries denominated in
      ``quote_currency``, or ``None`` when nothing was reported.
      0.0 means "reported and zero", never "unknown". A market BUY's
      fee is charged in the BASE asset and stays out of this sum —
      it is visible in ``fees_reported``.

    ccxt populates ``fees`` (list) from ``fee`` (dict) when both exist,
    so the list is preferred to avoid double counting.
    """
    raw = order.get("fees")
    if not raw:
        fee = order.get("fee")
        raw = [fee] if isinstance(fee, dict) else []
    reported = [
        {"cost": float(f["cost"]), "currency": f.get("currency")}
        for f in raw
        if isinstance(f, dict) and f.get("cost") is not None
    ]
    if not reported:
        return None, None
    quote_sum = sum(
        f["cost"] for f in reported if f["currency"] == quote_currency
    )
    return float(quote_sum), reported


def sort_orders_for_placement(intents: list[OrderIntent]) -> list[OrderIntent]:
    """Sells first, buys second.

    On a cross-direction rebalance, executing sells first frees the
    quote currency that subsequent buys need. Placing buys first risks
    ``InsufficientFunds`` even when total accounting is balanced — the
    exchange settles each order against the current *free* balance,
    not the planned end-state.
    """
    sells = [i for i in intents if i.side == "sell"]
    buys = [i for i in intents if i.side == "buy"]
    return sells + buys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wait_or_use_terminal(
    broker: Broker,
    client_order_id: str,
    symbol: str,
    current_order: dict,
    *,
    poll_initial_s: float,
    poll_max_s: float,
    total_timeout_s: float,
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
) -> dict:
    """If ``current_order`` is already terminal, return it; else poll."""
    if current_order.get("status") in _CCXT_TERMINAL:
        return current_order
    return wait_for_terminal(
        broker, client_order_id, symbol,
        poll_initial_s=poll_initial_s, poll_max_s=poll_max_s,
        total_timeout_s=total_timeout_s,
        sleep_fn=sleep_fn, time_fn=time_fn,
    )


def _result_from_order(
    intent: OrderIntent,
    client_order_id: str,
    order: dict,
    placed_at: str,
) -> OrderResult:
    """Map a ccxt order dict into an :class:`OrderResult`.

    Terminal-status mapping rules:

    * ``status='closed'`` + ``filled ≈ intended`` → ``closed``
    * ``status='closed'`` + ``filled < intended``  → ``partial``
      (Binance closes an order when no more fills are possible — for
      market orders that can mean fee-eaten size or instant-cancel.)
    * ``status='canceled'`` → ``canceled``
    * ``status='expired'`` / ``'rejected'`` → the same status when
      unfilled, ``partial`` when the exchange filled some of it first
      (an IOC-style expiry). Either way exchange-terminal
      (``terminal_at`` set): no further fills will ever come.
    * non-terminal (we hit the wait-for-ack timeout) → ``timeout`` if
      no fill yet, ``partial`` if there's a non-zero ``filled``.
    """
    status = order.get("status")
    filled = float(order.get("filled") or 0.0)
    cost = float(order.get("cost") or 0.0)
    avg_price_raw = order.get("average")
    quote = intent.symbol.split("/")[1]
    fees_quote, fees_reported = fees_from_order(order, quote)
    exchange_id = order.get("id")

    intended = float(intent.base_amount)
    fully_filled = intended > 0 and filled >= intended * 0.9999

    if status == "closed" and fully_filled:
        terminal_status, terminal_at = "closed", utcnow_iso()
    elif status == "closed":
        terminal_status, terminal_at = "partial", utcnow_iso()
    elif status == "canceled":
        terminal_status, terminal_at = "canceled", utcnow_iso()
    elif status in ("expired", "rejected"):
        logger.warning(
            "Order %s is %s on the exchange (filled=%s of intended %s) — "
            "exchange-terminal, no further fills will come.",
            client_order_id, status, filled, intended,
        )
        terminal_status = "partial" if filled > 0 else status
        terminal_at = utcnow_iso()
    elif filled > 0:
        terminal_status, terminal_at = "partial", None
    else:
        terminal_status, terminal_at = "timeout", None

    return OrderResult(
        client_order_id=client_order_id,
        exchange_order_id=str(exchange_id) if exchange_id is not None else None,
        symbol=intent.symbol,
        side=intent.side,
        intended_amount=intended,
        terminal_status=terminal_status,
        filled_amount=filled,
        filled_notional_quote=cost,
        average_price=float(avg_price_raw) if avg_price_raw is not None else None,
        fees_paid_quote=fees_quote,
        placed_at=placed_at,
        terminal_at=terminal_at,
        error=None,
        fees_reported=fees_reported,
    )


def _result_from_cached(cached: OrderStateEntry) -> OrderResult:
    """Build an :class:`OrderResult` from a state-cache hit.

    Fill-detail fields are 0/None because the state store does not
    persist them. Caller (live_cycle) interprets a cache hit as
    "previously completed; no new placement made this run".
    """
    return OrderResult(
        client_order_id=cached.client_order_id,
        exchange_order_id=cached.exchange_order_id,
        symbol=cached.symbol,
        side=cached.side,
        intended_amount=cached.intended_amount,
        terminal_status=cached.status,
        filled_amount=0.0,
        filled_notional_quote=0.0,
        average_price=None,
        fees_paid_quote=None,
        placed_at=cached.placed_at,
        terminal_at=cached.last_seen_at,
        error=None,
    )


def _placed_at_from_order(order: dict) -> str:
    ts = order.get("timestamp")
    if ts:
        return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
    return utcnow_iso()


def _persist(
    state: OrderStateStore,
    intent: OrderIntent,
    client_order_id: str,
    result: OrderResult,
) -> None:
    """Update the state store after a placement attempt.

    The stored status answers one question for future cycles: "is any
    further reconciliation needed?" — it is NOT fill accounting; the
    journal carries the fill detail. An exchange-CLOSED partial fill
    (``terminal_status='partial'`` with a ``terminal_at``) is stored
    as ``closed``: the exchange will never fill more, so there is
    nothing left to reconstruct. This mirrors the mapping in
    ``live_cycle._reconstruct_open_orders``; storing ``partial`` here
    made the next cycle reconstruct and re-journal the same incident.
    A timeout-with-partial-fill (``terminal_at is None``) stays
    ``partial`` — that one is still live on the exchange.
    """
    status = result.terminal_status
    if status == "partial" and result.terminal_at is not None:
        status = "closed"
    entry = OrderStateEntry(
        client_order_id=client_order_id,
        symbol=intent.symbol,
        side=intent.side,
        intended_amount=intent.base_amount,
        status=status,
        exchange_order_id=result.exchange_order_id,
        placed_at=result.placed_at,
        last_seen_at=utcnow_iso(),
    )
    state.put(entry)
