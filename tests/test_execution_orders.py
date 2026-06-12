"""Tests for the order placement primitives.

Coverage focus:

* Idempotency: state-cached terminal skips both fetch and create.
* Query-before-place: existing exchange-side order is observed and
  waited on, never re-created.
* Wait-for-ack: exponential backoff terminates on closed/canceled
  status; on budget exhaustion returns the last observed dict.
* Business rejections map to ``terminal_status='rejected'`` and do
  not raise; network errors propagate.
* Reconstruction: fetch_order success; fetch_my_trades fallback;
  truly-lost-track returns None.
* Sort order: sells precede buys on cross-direction rebalances.
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.delta import OrderIntent
from trade_lab.execution.order_state import (
    OrderStateEntry,
    OrderStateStore,
    TERMINAL_STATUSES,
)
from trade_lab.execution.orders import (
    POLL_INITIAL_S,
    POLL_MAX_S,
    TOTAL_TIMEOUT_S,
    OrderResult,
    place_order,
    reconstruct_status,
    sort_orders_for_placement,
    wait_for_terminal,
)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _MockClock:
    """Deterministic clock so tests don't actually sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _MockExchange:
    """Programmable subset of CCXT exchange for order tests."""

    id = "mock"

    def __init__(
        self,
        fetch_order_sequence: list = None,
        create_order_response: dict = None,
        create_order_raises: Exception = None,
        my_trades: list = None,
    ) -> None:
        self._fetch_seq = list(fetch_order_sequence or [])
        self._fetch_index = 0
        self.create_order_response = create_order_response
        self.create_order_raises = create_order_raises
        self.my_trades = my_trades or []

        self.create_order_calls: list[dict] = []
        self.fetch_order_calls: list[dict] = []
        self.fetch_my_trades_calls: list[dict] = []

    # --- methods used by Broker -------------------------------------------

    def set_sandbox_mode(self, enabled): pass

    def fetch_balance(self):
        return {"USDT": {"free": 10_000, "used": 0, "total": 10_000}}

    def fetch_ticker(self, symbol):
        return {"last": 50_000.0, "close": 50_000.0}

    def fetch_status(self):
        return {"status": "ok"}

    def load_markets(self, reload=False):
        return {"BTC/USDT": {}}

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.create_order_calls.append({
            "symbol": symbol, "type": type, "side": side,
            "amount": amount, "price": price, "params": params,
        })
        if self.create_order_raises is not None:
            raise self.create_order_raises
        return dict(self.create_order_response or {})

    def fetch_order(self, id, symbol=None, params=None):
        self.fetch_order_calls.append({
            "id": id, "symbol": symbol, "params": params,
        })
        if self._fetch_index >= len(self._fetch_seq):
            raise ccxt.OrderNotFound(f"no more responses programmed for {id}")
        item = self._fetch_seq[self._fetch_index]
        self._fetch_index += 1
        if isinstance(item, Exception):
            raise item
        return dict(item)

    def fetch_open_orders(self, symbol=None):
        return []

    def fetch_my_trades(self, symbol=None, since=None, limit=None):
        self.fetch_my_trades_calls.append({
            "symbol": symbol, "since": since, "limit": limit,
        })
        return list(self.my_trades)


def _broker(exch: _MockExchange) -> Broker:
    cfg = PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT",
        basket=("BTC", "ETH"), request_timeout_ms=5000,
    )
    return Broker(cfg, exch)


def _intent(side: str = "buy", amount: float = 0.001) -> OrderIntent:
    return OrderIntent(
        symbol="BTC/USDT", side=side, base_amount=amount,
        notional_quote=amount * 50_000,
        price_used=50_000.0,
        reason="test",
    )


def _store(tmp_path) -> OrderStateStore:
    return OrderStateStore(tmp_path / "orders.json")


def _ccxt_order(
    status: str = "closed",
    filled: float = 0.001,
    cost: float = 49.95,
    avg: float = 49950.0,
    fee_cost: float = 0.05,
    exchange_id: str = "12345",
    client_order_id: str = "tsmom_20260530_BTCUSDT_buy",
) -> dict:
    return {
        "id": exchange_id,
        "clientOrderId": client_order_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": status,
        "filled": filled,
        "cost": cost,
        "average": avg,
        "fee": {"cost": fee_cost, "currency": "USDT"},
        "timestamp": 1717000000000,
    }


# ---------------------------------------------------------------------------
# place_order — state fast-path
# ---------------------------------------------------------------------------


def test_state_cached_terminal_skips_exchange_roundtrip(tmp_path):
    """Cycle re-run on the same day: cached terminal entry → no fetch,
    no create. The orders.py module avoids 7 idle fetch_order calls
    per cycle this way."""
    exch = _MockExchange()
    broker = _broker(exch)
    store = _store(tmp_path)
    coid = "tsmom_20260530_BTCUSDT_buy"
    store.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="closed",
        exchange_order_id="prior-12345",
        placed_at="2026-05-30T00:05:00+00:00",
        last_seen_at="2026-05-30T00:05:03+00:00",
    ))
    result = place_order(broker, _intent(), client_order_id=coid, state=store)
    assert exch.fetch_order_calls == []
    assert exch.create_order_calls == []
    assert result.terminal_status == "closed"
    assert result.exchange_order_id == "prior-12345"


def test_state_cached_open_does_not_short_circuit(tmp_path):
    """A non-terminal cache entry must NOT skip the exchange round-trip."""
    exch = _MockExchange(fetch_order_sequence=[_ccxt_order(status="closed")])
    broker = _broker(exch)
    store = _store(tmp_path)
    coid = "tsmom_20260530_BTCUSDT_buy"
    store.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="12345",
        placed_at="2026-05-30T00:05:00+00:00",
        last_seen_at="2026-05-30T00:05:00+00:00",
    ))
    place_order(broker, _intent(), client_order_id=coid, state=store)
    # Found existing on exchange → no create_order.
    assert len(exch.fetch_order_calls) >= 1
    assert exch.create_order_calls == []


# ---------------------------------------------------------------------------
# place_order — happy paths
# ---------------------------------------------------------------------------


def test_immediate_fill(tmp_path):
    """create_order then fetch_order returns closed immediately."""
    exch = _MockExchange(
        create_order_response=_ccxt_order(status="open", filled=0.0),
        fetch_order_sequence=[
            ccxt.OrderNotFound("first lookup before placement"),
            _ccxt_order(status="closed"),
        ],
    )
    broker = _broker(exch)
    store = _store(tmp_path)
    clock = _MockClock()

    result = place_order(
        broker, _intent(),
        client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.terminal_status == "closed"
    assert result.filled_amount == 0.001
    assert result.average_price == 49950.0
    assert len(exch.create_order_calls) == 1
    assert store.get("tsmom_20260530_BTCUSDT_buy").status == "closed"


def test_partial_fill_eventually_closed(tmp_path):
    """open → open → closed but filled < intended → terminal_status=partial."""
    exch = _MockExchange(
        create_order_response=_ccxt_order(status="open", filled=0.0),
        fetch_order_sequence=[
            ccxt.OrderNotFound("before placement"),
            {"id": "12345", "status": "open", "filled": 0.0005,
             "cost": 24.97, "average": 49940.0, "fee": {"cost": 0.025}},
            {"id": "12345", "status": "closed", "filled": 0.0007,
             "cost": 34.95, "average": 49930.0, "fee": {"cost": 0.035}},
        ],
    )
    broker = _broker(exch)
    store = _store(tmp_path)
    clock = _MockClock()

    result = place_order(
        broker, _intent(amount=0.001),
        client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    # closed but filled (0.0007) < intended (0.001) → partial.
    assert result.terminal_status == "partial"
    assert result.filled_amount == 0.0007
    assert result.terminal_at is not None
    # Exchange-terminal → state stores "closed" (nothing left to
    # reconstruct); the partial fill detail lives in the journal.
    assert store.get("tsmom_20260530_BTCUSDT_buy").status == "closed"


def test_query_before_place_finds_existing(tmp_path):
    """A second cycle (state wiped) re-discovers an existing exchange
    order via fetch_order and does NOT call create_order."""
    exch = _MockExchange(
        fetch_order_sequence=[
            _ccxt_order(status="closed"),  # initial query finds existing
        ],
    )
    broker = _broker(exch)
    store = _store(tmp_path)

    result = place_order(
        broker, _intent(),
        client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
    )
    assert exch.create_order_calls == []
    assert result.terminal_status == "closed"


# ---------------------------------------------------------------------------
# place_order — rejection paths (no raise, OrderResult)
# ---------------------------------------------------------------------------


def test_invalid_order_returns_rejected(tmp_path):
    exch = _MockExchange(
        create_order_raises=ccxt.InvalidOrder("min notional 10 USDT not met"),
        fetch_order_sequence=[ccxt.OrderNotFound("not placed yet")],
    )
    broker = _broker(exch)
    store = _store(tmp_path)

    result = place_order(
        broker, _intent(),
        client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
    )
    assert result.terminal_status == "rejected"
    assert result.error["type"] == "InvalidOrder"
    assert "min notional" in result.error["message"]
    assert store.get("tsmom_20260530_BTCUSDT_buy").status == "rejected"


def test_insufficient_funds_returns_rejected(tmp_path):
    exch = _MockExchange(
        create_order_raises=ccxt.InsufficientFunds("not enough USDT"),
        fetch_order_sequence=[ccxt.OrderNotFound("not placed")],
    )
    broker = _broker(exch)
    store = _store(tmp_path)

    result = place_order(
        broker, _intent(),
        client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
    )
    assert result.terminal_status == "rejected"
    assert result.error["type"] == "InsufficientFunds"


def test_network_error_propagates(tmp_path):
    exch = _MockExchange(
        create_order_raises=ccxt.NetworkError("timeout"),
        fetch_order_sequence=[ccxt.OrderNotFound("not placed")],
    )
    broker = _broker(exch)
    store = _store(tmp_path)

    with pytest.raises(ccxt.NetworkError):
        place_order(
            broker, _intent(),
            client_order_id="tsmom_20260530_BTCUSDT_buy", state=store,
        )


# ---------------------------------------------------------------------------
# wait_for_terminal — backoff and timeout
# ---------------------------------------------------------------------------


def test_wait_for_terminal_returns_on_first_closed():
    exch = _MockExchange(fetch_order_sequence=[_ccxt_order(status="closed")])
    broker = _broker(exch)
    clock = _MockClock()

    order = wait_for_terminal(
        broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT",
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert order["status"] == "closed"
    assert clock.sleeps == []  # never slept


def test_wait_for_terminal_backoff_schedule():
    """Open then closed: expect one sleep at the initial delay."""
    exch = _MockExchange(fetch_order_sequence=[
        {"status": "open"},
        _ccxt_order(status="closed"),
    ])
    broker = _broker(exch)
    clock = _MockClock()

    wait_for_terminal(
        broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT",
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert clock.sleeps == [POLL_INITIAL_S]


def test_wait_for_terminal_exponential_growth_capped():
    """Verify the doubling stops at POLL_MAX_S."""
    open_responses = [{"status": "open"} for _ in range(12)]
    exch = _MockExchange(fetch_order_sequence=open_responses + [_ccxt_order(status="closed")])
    broker = _broker(exch)
    clock = _MockClock()

    wait_for_terminal(
        broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT",
        sleep_fn=clock.sleep, time_fn=clock.time,
        total_timeout_s=1_000_000,  # don't trigger timeout, just observe growth
    )
    # First several sleeps follow 1, 2, 4, 8, 16, 30, 30, ...
    assert clock.sleeps[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_wait_for_terminal_returns_last_observed_on_timeout():
    exch = _MockExchange(
        fetch_order_sequence=[{"status": "open"}] * 100,
    )
    broker = _broker(exch)
    clock = _MockClock()

    order = wait_for_terminal(
        broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT",
        total_timeout_s=10.0,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert order["status"] == "open"  # last observed, non-terminal
    assert clock.now >= 10.0


# ---------------------------------------------------------------------------
# reconstruct_status
# ---------------------------------------------------------------------------


def test_reconstruct_via_fetch_order():
    exch = _MockExchange(fetch_order_sequence=[_ccxt_order(status="closed")])
    broker = _broker(exch)
    result = reconstruct_status(broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT")
    assert result is not None
    assert result["status"] == "closed"


def test_reconstruct_falls_back_to_trades_when_order_unknown():
    """fetch_order raises OrderNotFound, but fetch_my_trades has a
    trade tagged with the clientOrderId — synthesize a closed order."""
    coid = "tsmom_20260530_BTCUSDT_buy"
    exch = _MockExchange(
        fetch_order_sequence=[ccxt.OrderNotFound("gone")],
        my_trades=[
            {
                "order": "exch-99",
                "info": {"clientOrderId": coid},
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.0008,
                "cost": 39.95,
                "price": 49937.5,
                "fee": {"cost": 0.04, "currency": "USDT"},
                "timestamp": 1717000000000,
            },
        ],
    )
    broker = _broker(exch)
    result = reconstruct_status(broker, coid, "BTC/USDT")
    assert result is not None
    assert result["status"] == "closed"
    assert result["filled"] == 0.0008
    assert result["clientOrderId"] == coid


def test_reconstruct_returns_none_when_truly_lost():
    """OrderNotFound and no matching trades → None.

    The caller turns this into ``status='lost_track'`` and surfaces a
    loud alert: an order we believe we placed has no trace anywhere.
    """
    exch = _MockExchange(
        fetch_order_sequence=[ccxt.OrderNotFound("gone")],
        my_trades=[
            {"info": {"clientOrderId": "some-other-id"},
             "symbol": "BTC/USDT", "amount": 0.001},
        ],
    )
    broker = _broker(exch)
    result = reconstruct_status(broker, "tsmom_20260530_BTCUSDT_buy", "BTC/USDT")
    assert result is None


# ---------------------------------------------------------------------------
# sort_orders_for_placement
# ---------------------------------------------------------------------------


def test_sort_orders_sells_first():
    orders = [
        _intent(side="buy"),
        _intent(side="sell"),
        _intent(side="buy"),
        _intent(side="sell"),
    ]
    sorted_orders = sort_orders_for_placement(orders)
    assert [o.side for o in sorted_orders] == ["sell", "sell", "buy", "buy"]


def test_sort_orders_preserves_within_group_order():
    a = OrderIntent(symbol="A/USDT", side="sell", base_amount=1.0, notional_quote=1.0, price_used=1.0, reason="")
    b = OrderIntent(symbol="B/USDT", side="sell", base_amount=2.0, notional_quote=2.0, price_used=1.0, reason="")
    c = OrderIntent(symbol="C/USDT", side="buy", base_amount=3.0, notional_quote=3.0, price_used=1.0, reason="")
    d = OrderIntent(symbol="D/USDT", side="buy", base_amount=4.0, notional_quote=4.0, price_used=1.0, reason="")
    sorted_orders = sort_orders_for_placement([a, c, b, d])
    assert [o.symbol for o in sorted_orders] == ["A/USDT", "B/USDT", "C/USDT", "D/USDT"]


def test_sort_orders_only_buys_unchanged():
    orders = [_intent(side="buy"), _intent(side="buy")]
    assert sort_orders_for_placement(orders) == orders


def test_sort_orders_empty():
    assert sort_orders_for_placement([]) == []


# ---------------------------------------------------------------------------
# State persistence after placement
# ---------------------------------------------------------------------------


def test_state_persisted_after_successful_placement(tmp_path):
    """The state-store entry exists with the correct exchange_order_id
    and terminal status after place_order completes."""
    exch = _MockExchange(
        create_order_response=_ccxt_order(status="open", filled=0.0),
        fetch_order_sequence=[
            ccxt.OrderNotFound("before placement"),
            _ccxt_order(status="closed"),
        ],
    )
    broker = _broker(exch)
    store = _store(tmp_path)
    clock = _MockClock()
    coid = "tsmom_20260530_BTCUSDT_buy"

    place_order(
        broker, _intent(),
        client_order_id=coid, state=store,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    entry = store.get(coid)
    assert entry is not None
    assert entry.status == "closed"
    assert entry.exchange_order_id == "12345"
    assert entry.intended_amount == 0.001


def test_state_persisted_after_rejection(tmp_path):
    exch = _MockExchange(
        create_order_raises=ccxt.InvalidOrder("min notional"),
        fetch_order_sequence=[ccxt.OrderNotFound("not placed")],
    )
    broker = _broker(exch)
    store = _store(tmp_path)
    coid = "tsmom_20260530_BTCUSDT_buy"

    place_order(broker, _intent(), client_order_id=coid, state=store)
    entry = store.get(coid)
    assert entry is not None
    assert entry.status == "rejected"


# ---------------------------------------------------------------------------
# Fee extraction semantics
# ---------------------------------------------------------------------------


def test_fees_none_when_exchange_reports_nothing():
    """Binance spot fetch_order carries no fee info. 0.0 would claim
    'zero fees paid'; None says 'not reported'."""
    from trade_lab.execution.orders import fees_from_order

    quote_sum, reported = fees_from_order(
        {"id": "1", "status": "closed", "filled": 1.0, "fee": None}, "USDT",
    )
    assert quote_sum is None
    assert reported is None


def test_fees_quote_currency_summed():
    from trade_lab.execution.orders import fees_from_order

    quote_sum, reported = fees_from_order(
        {"fee": {"cost": 0.05, "currency": "USDT"}}, "USDT",
    )
    assert quote_sum == pytest.approx(0.05)
    assert reported == [{"cost": 0.05, "currency": "USDT"}]


def test_fees_base_currency_not_summed_into_quote():
    """A market BUY pays its fee in BASE units — shoving 0.00001 BTC
    into a USDT-denominated field corrupts the audit. It must stay
    visible verbatim instead."""
    from trade_lab.execution.orders import fees_from_order

    quote_sum, reported = fees_from_order(
        {"fee": {"cost": 0.00001, "currency": "BTC"}}, "USDT",
    )
    assert quote_sum == 0.0
    assert reported == [{"cost": 0.00001, "currency": "BTC"}]


def test_fees_list_preferred_over_fee_to_avoid_double_count():
    """ccxt mirrors fee into fees; counting both doubles the number."""
    from trade_lab.execution.orders import fees_from_order

    quote_sum, _ = fees_from_order(
        {
            "fee": {"cost": 0.05, "currency": "USDT"},
            "fees": [{"cost": 0.05, "currency": "USDT"}],
        },
        "USDT",
    )
    assert quote_sum == pytest.approx(0.05)
