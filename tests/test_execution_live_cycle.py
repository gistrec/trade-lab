"""End-to-end tests for run_live_cycle.

Each test wires a full mock exchange (OHLCV + balance + tickers +
create_order + fetch_order + fetch_my_trades) and validates one
cycle outcome. The mock skips the actual CCXT layer entirely — no
network, no API keys.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.journal import JournalWriter
from trade_lab.execution.live_cycle import run_live_cycle
from trade_lab.execution.order_state import OrderStateEntry, OrderStateStore


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class _MockClock:
    def __init__(self): self.now = 0.0
    def time(self): return self.now
    def sleep(self, s): self.now += s


def _config(basket=("BTC", "ETH")):
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


class _LiveStub:
    """Full mock exchange: data + orders."""

    id = "stub"

    def __init__(
        self,
        *,
        basket=("BTC", "ETH"),
        balance_usdt: float = 10_000.0,
        asset_holdings: dict | None = None,
        closes: list | None = None,
        ticker_price: float = 50_000.0,
        # Order behavior
        create_raises: Exception | None = None,
        fetch_order_responses: dict | None = None,  # coid -> list of responses
        my_trades: list | None = None,
    ):
        self.basket = basket
        self.balance = {
            "USDT": {"free": balance_usdt, "used": 0.0, "total": balance_usdt},
        }
        for a in basket:
            amt = (asset_holdings or {}).get(a, 0.0)
            self.balance[a] = {"free": amt, "used": 0.0, "total": amt}

        self.tickers = {
            f"{a}/USDT": {"last": ticker_price, "close": ticker_price}
            for a in basket
        }
        # Default to clean uptrend so signal=1.0 unless overridden.
        if closes is None:
            closes = (100 + np.linspace(0, 200, 500)).tolist()
        self._closes = closes

        self.create_raises = create_raises
        self.fetch_order_responses = fetch_order_responses or {}
        self.my_trades = my_trades or []

        self.create_order_calls: list[dict] = []
        self.fetch_order_calls: list[dict] = []
        self.placed_coids: set[str] = set()

    # ------------------ data side ------------------

    def set_sandbox_mode(self, enabled): pass

    def fetch_balance(self): return self.balance

    def fetch_ticker(self, symbol): return self.tickers[symbol]

    def fetch_status(self): return {"status": "ok"}

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=400):
        ts = pd.date_range(
            "2023-01-01", periods=len(self._closes), freq="1D", tz="UTC",
        ).astype("int64") // 10**6
        rows = [[int(t), c, c, c, c, 1.0] for t, c in zip(ts, self._closes)]
        return rows[-limit:]

    def load_markets(self, reload=False):
        return {
            f"{a}/USDT": {
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10.0}},
                "precision": {"amount": 8},
            }
            for a in self.basket
        }

    # ------------------ order side ------------------

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.create_order_calls.append({
            "symbol": symbol, "side": side, "amount": amount, "params": params,
        })
        if self.create_raises is not None:
            raise self.create_raises
        coid = (params or {}).get("newClientOrderId")
        self.placed_coids.add(coid)
        return {
            "id": f"exch-{len(self.create_order_calls)}",
            "clientOrderId": coid,
            "symbol": symbol, "side": side,
            "status": "open", "filled": 0.0, "cost": 0.0,
            "average": None, "fee": {"cost": 0.0}, "timestamp": 0,
        }

    def fetch_order(self, id, symbol=None, params=None):
        self.fetch_order_calls.append({
            "id": id, "symbol": symbol, "params": params,
        })
        coid = (params or {}).get("origClientOrderId") or id
        # Programmed responses always take precedence (used by tests that
        # simulate a pre-existing exchange order or a specific status).
        seq = self.fetch_order_responses.get(coid)
        if seq:
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return dict(item)
        # No programmed response: query-before-place must return NotFound
        # unless this stub has actually placed the order.
        if coid not in self.placed_coids:
            raise ccxt.OrderNotFound(f"not placed: {coid}")
        # Placed and no programmed response: default to a closed full-fill.
        matching = [
            c for c in self.create_order_calls
            if (c.get("params") or {}).get("newClientOrderId") == coid
        ]
        intended = matching[0]["amount"] if matching else 0.001
        return _closed_order(coid, symbol, filled=intended)

    def fetch_open_orders(self, symbol=None): return []

    def fetch_my_trades(self, symbol=None, since=None, limit=None):
        return list(self.my_trades)


def _closed_order(coid: str, symbol: str, filled: float = 0.001) -> dict:
    return {
        "id": f"exch-final-{coid}",
        "clientOrderId": coid,
        "symbol": symbol,
        "status": "closed",
        "filled": filled,
        "cost": filled * 50_000,
        "average": 50_000.0,
        "fee": {"cost": filled * 50_000 * 0.001, "currency": "USDT"},
        "timestamp": 1717000000000,
    }


def _broker(stub: _LiveStub, basket=("BTC", "ETH")) -> Broker:
    return Broker(_config(basket=basket), stub)


def _journal(tmp_path) -> JournalWriter:
    return JournalWriter(tmp_path / "cycles.jsonl")


def _state(tmp_path) -> OrderStateStore:
    return OrderStateStore(tmp_path / "orders.json")


def _read_cycles(tmp_path) -> list[dict]:
    path = tmp_path / "cycles.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_success_path_clean_uptrend(tmp_path):
    """signal=1.0 + zero current holdings → buys for each basket asset
    → all fill → outcome=success, schema v2 in journal."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "success"
    assert len(result.order_results) == 2  # one per basket asset
    cycles = _read_cycles(tmp_path)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle["schema_version"] == 2
    assert cycle["outcome"] == "success"
    assert len(cycle["orders_executed"]) == 2


def test_journal_records_basket_weights(tmp_path):
    """The whole C3 chain must reach the journal: the signal snapshot's
    drifted per-asset weights are recorded so a reviewer can reconcile
    that execution sized to the basket's drifted weights, not flat 1/N.
    With the stub's identical closes the weights come out equal-weight
    and sum to 1."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()
    run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    cycle = _read_cycles(tmp_path)[0]
    weights = cycle["signal"]["basket_weights"]
    assert set(weights.keys()) == {"BTC", "ETH"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["BTC"] == pytest.approx(0.5)


def test_nan_weight_fails_loud_through_cycle(tmp_path, monkeypatch):
    """Fail-loud, end to end: a NaN in the signal's basket_weights must not
    silently mis-size the book. run_live_cycle raises and journals
    outcome='failed' — the guarantee enforced through the production
    pipeline, not just in an allocator unit test."""
    import math
    from trade_lab.execution import live_cycle as lc
    from trade_lab.execution.signal import SignalSnapshot

    bad_snap = SignalSnapshot(
        asof=pd.Timestamp("2026-06-11", tz="UTC"), signal=1.0,
        basket_close=150.0, asset_closes={"BTC": 50_000.0, "ETH": 3_000.0},
        sma_gate_open=True, n_assets_in_basket=2,
        basket_weights={"BTC": math.nan, "ETH": 0.5},
    )
    monkeypatch.setattr(lc, "compute_live_signal", lambda *a, **k: bad_snap)

    broker = _broker(_LiveStub(basket=("BTC", "ETH")))
    clock = _MockClock()
    with pytest.raises(ValueError, match="BTC"):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "ValueError"


def test_signal_zero_no_orders(tmp_path):
    """Downtrend → signal=0 → no orders planned → outcome=success,
    orders_executed=[]."""
    stub = _LiveStub(
        basket=("BTC", "ETH"),
        closes=np.linspace(200, 100, 500).tolist(),
    )
    broker = _broker(stub)
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
    )
    assert result.outcome == "success"
    assert result.order_results == []
    cycle = _read_cycles(tmp_path)[0]
    assert cycle["orders_executed"] == []
    assert stub.create_order_calls == []


# ---------------------------------------------------------------------------
# Non-success outcomes
# ---------------------------------------------------------------------------


def test_partial_fill_outcome(tmp_path):
    """One order returns closed with filled<intended → outcome=partial."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    # Override the response for whichever coid hits first — we'll override both
    # to be safe.
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    stub.fetch_order_responses[f"tsmom_{today}_BTCUSDT_buy"] = [
        {
            "id": "exch-1", "status": "closed",
            "filled": 0.00005,   # way below intended
            "cost": 2.5, "average": 50000.0,
            "fee": {"cost": 0.002}, "timestamp": 0,
        },
    ]
    broker = _broker(stub)
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "partial"
    cycle = _read_cycles(tmp_path)[0]
    assert cycle["outcome"] == "partial"


def test_timeout_outcome(tmp_path):
    """One order never reaches terminal → outcome=unknown_orders."""
    stub = _LiveStub(basket=("BTC",))
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    # First call (query-before-place) → OrderNotFound; placement → open.
    stub.fetch_order_responses[f"tsmom_{today}_BTCUSDT_buy"] = (
        [ccxt.OrderNotFound("before placement")]
        + [{"id": "exch-1", "status": "open", "filled": 0.0,
            "cost": 0.0, "average": None, "fee": {"cost": 0.0}, "timestamp": 0}
           for _ in range(50)]
    )
    broker = _broker(stub, basket=("BTC",))
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        total_timeout_s=5.0,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "unknown_orders"


def test_rejected_outcome_when_create_raises_invalid(tmp_path):
    """create_order raises InvalidOrder → terminal_status=rejected →
    cycle outcome=partial."""
    stub = _LiveStub(
        basket=("BTC",),
        create_raises=ccxt.InvalidOrder("min notional 10 USDT"),
    )
    broker = _broker(stub, basket=("BTC",))
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "partial"
    cycle = _read_cycles(tmp_path)[0]
    assert cycle["orders_executed"][0]["terminal_status"] == "rejected"


# ---------------------------------------------------------------------------
# Failed cycle (exception in pipeline)
# ---------------------------------------------------------------------------


def test_exception_in_pipeline_writes_failed_cycle(tmp_path):
    """A network error during signal/balance/etc still produces a
    journal entry with outcome=failed, error captured."""
    class _NetErrStub(_LiveStub):
        def fetch_balance(self):
            raise ccxt.NetworkError("balance call dropped")

    stub = _NetErrStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()

    with pytest.raises(ccxt.NetworkError):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )

    cycle = _read_cycles(tmp_path)[0]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "NetworkError"
    assert "balance call dropped" in cycle["error"]["message"]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_reconstruction_writes_separate_entry(tmp_path):
    """Pre-existing open state entry → reconstruction phase produces a
    separate cycle entry first, then the normal cycle entry."""
    state = _state(tmp_path)
    pending_coid = "tsmom_20260528_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=pending_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-prior",
        placed_at="2026-05-28T00:05:00+00:00",
        last_seen_at="2026-05-28T00:05:00+00:00",
    ))
    # The reconstruction fetch_order for this pending coid returns closed.
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[pending_coid] = [_closed_order(pending_coid, "BTC/USDT")]
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 1
    cycles = _read_cycles(tmp_path)
    assert len(cycles) == 2
    assert cycles[0]["outcome"] == "reconstructed"
    assert cycles[1]["outcome"] in ("success", "partial")
    assert cycles[0]["orders_executed"][0]["client_order_id"] == pending_coid
    # State updated to closed.
    assert state.get(pending_coid).status == "closed"


def test_reconstruction_lost_track(tmp_path):
    """Pre-existing open state + OrderNotFound + no matching trades →
    status='lost_track' with loud warning."""
    state = _state(tmp_path)
    pending_coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=pending_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-vanished",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-20T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    # Both reconstruction lookups return OrderNotFound.
    stub.fetch_order_responses[pending_coid] = [
        ccxt.OrderNotFound("gone"),
    ]
    # No trades match the coid either.
    stub.my_trades = []
    broker = _broker(stub)
    clock = _MockClock()

    run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert state.get(pending_coid).status == "lost_track"
    cycles = _read_cycles(tmp_path)
    recon = cycles[0]
    assert recon["outcome"] == "reconstructed"
    assert recon["orders_executed"][0]["terminal_status"] == "lost_track"


def test_new_lost_track_escalates_result(tmp_path):
    """A lost_track discovered during reconstruction must be surfaced on
    the result even when the MAIN cycle is a clean success, so the CLI can
    escalate the exit code for cron alerting (regression: R1). Without
    this, a vanished order was journaled but the process still exited 0."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-vanished",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-20T00:05:00+00:00",
    ))
    # Main cycle places nothing: downtrend → signal=0, no holdings.
    stub = _LiveStub(
        basket=("BTC", "ETH"),
        closes=np.linspace(200, 100, 500).tolist(),
    )
    stub.fetch_order_responses[coid] = [ccxt.OrderNotFound("gone")]
    stub.my_trades = []
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    # The main cycle itself is a clean success (no orders to place)...
    assert result.outcome == "success"
    assert result.order_results == []
    # ...but a lost_track WAS discovered and must be surfaced for alerting.
    assert result.lost_track_count == 1
    assert state.get(coid).status == "lost_track"


def test_persistent_lost_track_still_escalates_result(tmp_path):
    """A lost_track that was already recorded and is STILL missing keeps
    the result's lost_track_count > 0 (unresolved incident), even though
    it is not re-journaled — so exit-code alerting stays red until an
    operator resolves it (regression: R1)."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="lost_track",
        exchange_order_id="exch-vanished",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-21T00:05:00+00:00",
    ))
    stub = _LiveStub(
        basket=("BTC", "ETH"),
        closes=np.linspace(200, 100, 500).tolist(),
    )
    stub.fetch_order_responses[coid] = [ccxt.OrderNotFound("still gone")]
    stub.my_trades = []
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    # Not re-journaled (that decision is preserved)...
    assert result.reconstructed_count == 0
    # ...but the exit code must still escalate: the incident is unresolved.
    assert result.lost_track_count == 1


def test_reconstruction_recovers_via_trade_by_exchange_order_id(tmp_path):
    """Order record expired (OrderNotFound) but the fill's trade is still
    queryable: reconstruction recovers it by matching the exchange order
    id threaded from state, instead of flagging lost_track (regression:
    C13). Binance trades carry no clientOrderId, so the old
    clientOrderId-only match made this fallback dead code."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-recover",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-20T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[coid] = [ccxt.OrderNotFound("record expired")]
    stub.my_trades = [{
        "order": "exch-recover",
        "info": {"orderId": "exch-recover"},
        "symbol": "BTC/USDT", "side": "buy",
        "amount": 0.001, "cost": 50.0, "price": 50000.0,
        "fee": {"cost": 0.05, "currency": "USDT"},
        "timestamp": 1717000000000,
    }]
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    # Recovered, not lost.
    assert state.get(coid).status == "closed"
    assert result.lost_track_count == 0
    recon = _read_cycles(tmp_path)[0]
    assert recon["outcome"] == "reconstructed"
    assert recon["orders_executed"][0]["terminal_status"] in ("closed", "partial")


def test_no_reconstruction_when_state_empty(tmp_path):
    """Fresh start (no open state entries) → no reconstruction cycle entry."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 0
    cycles = _read_cycles(tmp_path)
    # Only the main cycle entry was written.
    assert len(cycles) == 1
    assert cycles[0]["outcome"] in ("success", "partial")


# ---------------------------------------------------------------------------
# Order sorting (sells first on cross-direction)
# ---------------------------------------------------------------------------


def test_cross_direction_sells_first(tmp_path):
    """Hold BTC + ETH while signal=0 → sells for both. With single
    direction the order is consistent. We verify the first orders
    are sells, not buys."""
    stub = _LiveStub(
        basket=("BTC", "ETH"),
        balance_usdt=0.0,
        asset_holdings={"BTC": 0.1, "ETH": 0.5},
        closes=np.linspace(200, 100, 500).tolist(),  # signal=0
    )
    broker = _broker(stub)
    clock = _MockClock()
    run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    sides_in_order = [c["side"] for c in stub.create_order_calls]
    assert all(s == "sell" for s in sides_in_order), sides_in_order


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotency_same_day_state_cache_hit(tmp_path):
    """Second run on the same day: place_order's state-cache fast-path
    skips both fetch_order and create_order for terminal entries."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    state = _state(tmp_path)
    journal = _journal(tmp_path)
    clock = _MockClock()

    # First run: places orders.
    run_live_cycle(
        broker=_broker(stub), journal=journal, state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    first_call_count = len(stub.create_order_calls)
    first_fetch_count = len(stub.fetch_order_calls)
    assert first_call_count >= 2  # at least one per basket asset

    # Second run with the SAME state: no new create_order or fetch_order.
    run_live_cycle(
        broker=_broker(stub), journal=journal, state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert len(stub.create_order_calls) == first_call_count
    assert len(stub.fetch_order_calls) == first_fetch_count


# ---------------------------------------------------------------------------
# Schema v2 validation
# ---------------------------------------------------------------------------


def test_journal_schema_v2_fields_present(tmp_path):
    """Every cycle written by run_live_cycle declares schema_version=2 and
    includes orders_executed (may be empty list)."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()
    run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    cycle = _read_cycles(tmp_path)[0]
    assert cycle["schema_version"] == 2
    assert "orders_executed" in cycle
    assert isinstance(cycle["orders_executed"], list)


def test_lost_track_not_rejournaled_every_cycle(tmp_path):
    """An entry already marked lost_track and still missing from the
    exchange must NOT produce a new reconstruction journal entry on
    every subsequent cycle — only the transition into lost_track is an
    incident. The exchange is still queried (recovery stays possible)."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="lost_track",
        exchange_order_id="exch-vanished",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-21T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[coid] = [ccxt.OrderNotFound("still gone")]
    stub.my_trades = []
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.reconstructed_count == 0
    cycles = _read_cycles(tmp_path)
    assert all(c["outcome"] != "reconstructed" for c in cycles)
    assert state.get(coid).status == "lost_track"


def test_lost_track_recovers_when_exchange_record_appears(tmp_path):
    """If the exchange record shows up later (lag, restored history),
    a lost_track entry transitions out normally and IS journaled."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="lost_track",
        exchange_order_id=None,
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-21T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[coid] = [_closed_order(coid, "BTC/USDT")]
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.reconstructed_count == 1
    cycles = _read_cycles(tmp_path)
    assert cycles[0]["outcome"] == "reconstructed"
    assert state.get(coid).status == "closed"


def test_closed_partial_not_rereconstructed_next_cycle(tmp_path):
    """An order the exchange CLOSED with a partial fill is terminal on
    the exchange — nothing more will fill. The main cycle already
    journaled the partial result; the next cycle must not reconstruct
    and re-journal the same incident."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    coid = f"tsmom_{today}_BTCUSDT_buy"
    closed_partial = {
        "id": "exch-1", "status": "closed",
        "filled": 0.00005, "cost": 2.5, "average": 50000.0,
        "fee": {"cost": 0.002}, "timestamp": 0,
    }
    stub.fetch_order_responses[coid] = [dict(closed_partial), dict(closed_partial)]
    broker = _broker(stub)
    clock = _MockClock()
    state = _state(tmp_path)

    run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    n_first = len(_read_cycles(tmp_path))
    assert state.get(coid).status == "closed"   # exchange-terminal

    run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    new_cycles = _read_cycles(tmp_path)[n_first:]
    assert all(c["outcome"] != "reconstructed" for c in new_cycles)


def test_ticker_fallback_raises_on_missing_close():
    """If a ticker fails AND the signal snapshot somehow lacks that
    asset's close (invariant violation), the fallback must raise at the
    cause — not feed a 0.0 price into the allocator."""
    from trade_lab.execution.live_cycle import _gather_ticker_prices
    from trade_lab.execution.signal import SignalSnapshot

    class _NoTickerStub(_LiveStub):
        def fetch_ticker(self, symbol):
            return {}  # no last/close → BrokerError in fetch_ticker_price

    broker = _broker(_NoTickerStub(basket=("BTC", "ETH")))
    snap = SignalSnapshot(
        asof=pd.Timestamp("2026-06-11", tz="UTC"), signal=1.0,
        basket_close=150.0, asset_closes={"BTC": 50_000.0},  # ETH missing
        sma_gate_open=True, n_assets_in_basket=2,
    )
    with pytest.raises(KeyError, match="ETH"):
        _gather_ticker_prices(broker, snap)
