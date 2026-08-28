"""End-to-end tests for run_live_cycle.

Each test wires a full mock exchange (OHLCV + balance + tickers +
create_order + fetch_order + fetch_my_trades) and validates one
cycle outcome. The mock skips the actual CCXT layer entirely — no
network, no API keys.
"""
from __future__ import annotations

import functools
import json
from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.journal import JournalWriter
from trade_lab.execution.live_cycle import run_live_cycle as _real_run_live_cycle
from trade_lab.execution.order_state import OrderStateEntry, OrderStateStore

# The stub's candles end at yesterday's UTC midnight, so inside these e2e
# tests the decision bar's age equals the wall-clock hour of the run — the
# 6h decision-age gate would flake by time of day. Disabled here; the gate
# has its own deterministic tests (decision-age section below), which call
# _real_run_live_cycle via the lc module.
run_live_cycle = functools.partial(_real_run_live_cycle, max_decision_age_s=None)


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
        # End at yesterday's UTC midnight — the most recently closed daily
        # bar — so the signal's asof-freshness guard passes at any hour.
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        ts = pd.date_range(
            end=end, periods=len(self._closes), freq="1D", tz="UTC",
        ).as_unit("ns").astype("int64") // 10**6
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
# Decision-age gate: a live cycle must run shortly after the decision
# bar's close — anything later is a timing the backtest never validated
# ---------------------------------------------------------------------------


def _hand_snap(asof):
    from trade_lab.execution.signal import SignalSnapshot
    return SignalSnapshot(
        asof=asof, signal=1.0, basket_close=150.0,
        asset_closes={"BTC": 50_000.0, "ETH": 50_000.0},
        sma_gate_open=True, n_assets_in_basket=2,
        basket_weights={"BTC": 0.5, "ETH": 0.5},
    )


def test_live_cycle_refuses_stale_decision_bar(tmp_path, monkeypatch):
    """The MSK-cron scenario: the signal itself is data-fresh, but the
    decision bar closed ~21h before the cycle runs (cron scheduled in
    the wrong timezone). Placing orders then is a different, unvalidated
    strategy timing — the cycle must fail loud with zero orders."""
    from trade_lab.execution import live_cycle as lc
    from trade_lab.execution.signal import SignalComputationError

    stale_asof = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=45)
    monkeypatch.setattr(
        lc, "compute_live_signal", lambda *a, **k: _hand_snap(stale_asof),
    )
    stub = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()
    with pytest.raises(SignalComputationError, match=r"[Dd]ecision bar"):
        lc.run_live_cycle(              # real default gate, not the partial
            _broker(stub), journal=_journal(tmp_path),
            state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    assert stub.create_order_calls == []
    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "SignalComputationError"


def test_live_cycle_accepts_fresh_decision_bar_and_journals_age(
        tmp_path, monkeypatch):
    """A decision bar that closed 45 minutes ago (the 00:45 UTC cron
    shape) passes the gate, and the journaled signal block carries the
    measured decision age for schedule-drift monitoring."""
    from trade_lab.execution import live_cycle as lc

    fresh_asof = (
        pd.Timestamp.now(tz="UTC")
        - pd.Timedelta(days=1) - pd.Timedelta(minutes=45)
    )
    monkeypatch.setattr(
        lc, "compute_live_signal", lambda *a, **k: _hand_snap(fresh_asof),
    )
    stub = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()
    result = lc.run_live_cycle(         # real default gate, not the partial
        _broker(stub), journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "success"
    cycle = _read_cycles(tmp_path)[-1]
    age = cycle["signal"]["decision_age_s"]
    assert age == pytest.approx(45 * 60, abs=120)


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


def test_expired_order_outcome_partial_and_next_cycle_skips(tmp_path):
    """Regression (M1): an order the exchange reports as 'expired' (ccxt
    unified status for Binance EXPIRED / EXPIRED_IN_MATCH) must resolve
    immediately: cycle outcome 'partial' (non-zero exit), journal entry
    with terminal_status='expired', terminal state entry — and the NEXT
    cycle must neither reconstruct nor re-poll nor re-place it."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    coid = f"tsmom_{today}_BTCUSDT_buy"
    state = _state(tmp_path)
    journal = _journal(tmp_path)
    clock = _MockClock()

    stub = _LiveStub(basket=("BTC",))
    stub.fetch_order_responses[coid] = [
        ccxt.OrderNotFound("before placement"),
        {"id": "exch-1", "status": "expired", "filled": 0.0,
         "cost": 0.0, "average": None, "timestamp": 0},
    ]
    broker = _broker(stub, basket=("BTC",))
    result = run_live_cycle(
        broker, journal=journal, state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "partial"           # incident, not success
    assert result.outcome != "unknown_orders"    # and not a false timeout
    cycle = _read_cycles(tmp_path)[0]
    assert cycle["orders_executed"][0]["terminal_status"] == "expired"
    assert state.get(coid).status == "expired"
    assert state.open_entries() == {}            # nothing left to reconcile

    # Next cycle (same day re-run): the state fast-path answers from the
    # terminal entry — zero fetch_order / create_order round-trips and no
    # reconstruction journal entry.
    stub2 = _LiveStub(basket=("BTC",))
    broker2 = _broker(stub2, basket=("BTC",))
    result2 = run_live_cycle(
        broker2, journal=journal, state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert stub2.fetch_order_calls == []
    assert stub2.create_order_calls == []
    assert result2.reconstructed_count == 0
    outcomes = [c["outcome"] for c in _read_cycles(tmp_path)]
    assert "reconstructed" not in outcomes
    # The day's rebalance still did not execute — re-run stays loud.
    assert result2.outcome == "partial"


# ---------------------------------------------------------------------------
# Insufficient warm-up: first-class skip on testnet, hard failure on mainnet
# ---------------------------------------------------------------------------


def _mainnet_config(basket=("BTC", "ETH")):
    return PaperConfig(
        exchange_id="binance", sandbox=False, api_key="k", api_secret="s",
        allow_mainnet=True, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


def _short_history_stub(basket=("BTC", "ETH")) -> _LiveStub:
    """Binance-testnet shape: ~monthly candle wipe leaves 36 daily bars."""
    return _LiveStub(
        basket=basket,
        closes=(100 + np.linspace(0, 20, 36)).tolist(),  # clean uptrend
    )


def test_short_history_on_testnet_is_first_class_skipped_warmup(tmp_path):
    """Sandbox config + 36-bar history: NOT a failed cycle. run_live_cycle
    returns outcome='skipped_warmup' with the structured reason, journals
    a first-class entry (error=None, skip_reason populated,
    orders_executed=[] so the live-cron clocks still see the daily run),
    and never touches create_order."""
    stub = _short_history_stub()
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "skipped_warmup"
    assert result.order_results == []
    assert result.error is None
    assert result.skip_reason["type"] == "insufficient_warmup"
    assert result.skip_reason["bars_available"] == 36
    assert result.skip_reason["bars_required"] == 200
    assert stub.create_order_calls == []

    cycles = _read_cycles(tmp_path)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle["outcome"] == "skipped_warmup"
    assert cycle["error"] is None
    assert cycle["skip_reason"]["bars_available"] == 36
    assert cycle["skip_reason"]["bars_required"] == 200
    # A list (empty) — the daily live cron DID run; monitoring's
    # live-cycle discriminator and /healthz/daily key on this.
    assert cycle["orders_executed"] == []
    assert cycle["context"]["mode"] == "live"
    assert cycle["context"]["sandbox"] is True


def test_short_history_on_mainnet_stays_failed_and_raises(tmp_path):
    """PIN of mainnet strictness (H3): sandbox=False + the identical short
    history = truncated kline history on the real-money path. Must journal
    outcome='failed' AND re-raise — zero skipped_warmup softening. This
    test exists to break any future dilution of the mainnet posture."""
    from trade_lab.execution.signal import InsufficientWarmupError

    stub = _short_history_stub()
    broker = Broker(_mainnet_config(), stub)
    clock = _MockClock()

    with pytest.raises(InsufficientWarmupError, match="36 completed bars"):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )

    assert stub.create_order_calls == []
    cycles = _read_cycles(tmp_path)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "InsufficientWarmupError"
    assert "36 completed bars" in cycle["error"]["message"]
    assert cycle["skip_reason"] is None
    assert cycle["context"]["sandbox"] is False


def test_other_signal_errors_still_fail_on_testnet(tmp_path, monkeypatch):
    """Only the depth guard's InsufficientWarmupError is a first-class
    skip. Any other SignalComputationError (uneven history, empty
    candles, fetch failure) keeps outcome='failed' + raise on the
    sandbox too — real data incidents never hide behind the skip."""
    from trade_lab.execution import live_cycle as lc
    from trade_lab.execution.signal import SignalComputationError

    def _raise(*a, **k):
        raise SignalComputationError(
            "Uneven basket history: DOGE has 150 bars starting 2025-08-05"
        )

    monkeypatch.setattr(lc, "compute_live_signal", _raise)
    broker = _broker(_LiveStub(basket=("BTC", "ETH")))
    clock = _MockClock()

    with pytest.raises(SignalComputationError, match="Uneven"):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )

    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "SignalComputationError"
    assert cycle["skip_reason"] is None


def test_skipped_warmup_still_surfaces_lost_track(tmp_path):
    """A perpetually-skipping testnet must not mute an unresolved order
    incident: reconstruction still runs first, and a lost_track discovered
    there keeps lost_track_count > 0 on the skipped_warmup result so the
    CLI exit code stays red."""
    state = _state(tmp_path)
    coid = "tsmom_20260520_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-vanished",
        placed_at="2026-05-20T00:05:00+00:00",
        last_seen_at="2026-05-20T00:05:00+00:00",
    ))
    stub = _short_history_stub()
    stub.fetch_order_responses[coid] = [ccxt.OrderNotFound("gone")]
    stub.my_trades = []
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "skipped_warmup"
    assert result.lost_track_count == 1
    assert result.reconstructed_count == 1
    outcomes = [c["outcome"] for c in _read_cycles(tmp_path)]
    assert outcomes == ["reconstructed", "skipped_warmup"]


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


def test_keyboard_interrupt_mid_cycle_journals_failed_and_reraises(tmp_path):
    """Regression (M4): Ctrl-C (KeyboardInterrupt is a BaseException, not
    an Exception) during the sleep between order placements — the shape of
    an operator interrupting a manual run mid-wait — must still write the
    failed-cycle journal entry, record the order already placed on the
    exchange under orders_executed, and re-raise so the process dies as
    the operator asked. Before the fix `except Exception` let the
    interrupt bypass the journal entirely: a real placed order with zero
    journal record."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)

    def _ctrl_c_sleep(_s: float) -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=_ctrl_c_sleep, time_fn=lambda: 0.0,
        )

    # The first order completed before the inter-order sleep raised.
    assert len(stub.create_order_calls) == 1
    cycles = _read_cycles(tmp_path)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "KeyboardInterrupt"
    # The already-placed order is not lost to history.
    executed = cycle["orders_executed"]
    assert executed is not None and len(executed) == 1
    assert executed[0]["terminal_status"] == "closed"


# ---------------------------------------------------------------------------
# Journal append failure AFTER orders are placed: surface, never crash
# ---------------------------------------------------------------------------


class _AppendRaisesJournal(JournalWriter):
    """Every append raises — the disk-full / permission-lost shape."""

    def append(self, cycle):
        raise OSError("No space left on device")


class _FirstAppendRaisesJournal(JournalWriter):
    """Only the first append raises; later entries land normally."""

    def __init__(self, path):
        super().__init__(path)
        self._raised = False

    def append(self, cycle):
        if not self._raised:
            self._raised = True
            raise OSError("No space left on device")
        super().append(cycle)


def test_journal_append_failure_sets_flag_not_crash(tmp_path):
    """Real orders are already on the exchange when the main-cycle append
    fails — raising then is worse than the missing entry. The cycle must
    return normally with outcome unchanged and journal_write_failed=True
    so the CLI can escalate the exit code (issue #45)."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_AppendRaisesJournal(tmp_path / "cycles.jsonl"),
        state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert len(stub.create_order_calls) == 2   # placement unaffected
    assert result.outcome == "success"          # outcome unchanged
    assert result.journal_write_failed is True
    assert _read_cycles(tmp_path) == []         # the entry really is missing


def test_journal_append_failure_logs_recoverable_payload(tmp_path, caplog):
    """The discarded entry must be recoverable: the error log carries the
    exact serialized journal line (appendable verbatim), not just the
    cycle id and exception."""
    import json as _json
    import logging as _logging

    stub = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()
    with caplog.at_level(_logging.ERROR, logger="trade_lab.execution.live_cycle"):
        run_live_cycle(
            _broker(stub), journal=_AppendRaisesJournal(tmp_path / "cycles.jsonl"),
            state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    payload_lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("Unwritten journal entry payload")
    ]
    assert len(payload_lines) == 1, caplog.text
    serialized = payload_lines[0].split(": ", 1)[1]
    entry = _json.loads(serialized)
    assert entry["outcome"] == "success"
    assert len(entry["orders_executed"]) == 2


def test_journal_write_failed_false_on_healthy_cycle(tmp_path):
    """A cycle whose journal write succeeds must not raise a false alarm."""
    stub = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()

    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    assert result.journal_write_failed is False


def test_reconstruction_append_failure_propagates_flag(tmp_path):
    """A lost reconstruction entry is the same audit hole: the flag must
    survive into the final result even though the MAIN entry landed."""
    state = _state(tmp_path)
    pending_coid = "tsmom_20260528_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=pending_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="open",
        exchange_order_id="exch-prior",
        placed_at="2026-05-28T00:05:00+00:00",
        last_seen_at="2026-05-28T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[pending_coid] = [
        _closed_order(pending_coid, "BTC/USDT"),
    ]
    clock = _MockClock()

    result = run_live_cycle(
        _broker(stub),
        journal=_FirstAppendRaisesJournal(tmp_path / "cycles.jsonl"),
        state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.reconstructed_count == 1
    assert result.journal_write_failed is True
    # Main entry landed; the reconstruction entry is the missing one.
    assert [c["outcome"] for c in _read_cycles(tmp_path)] == ["success"]


def test_skipped_warmup_append_failure_sets_flag(tmp_path):
    """The skipped_warmup writer is a swallow site too: a lost entry
    breaks the daily-run heartbeat, so the flag must escalate."""
    stub = _short_history_stub()
    clock = _MockClock()

    result = run_live_cycle(
        _broker(stub), journal=_AppendRaisesJournal(tmp_path / "cycles.jsonl"),
        state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "skipped_warmup"
    assert result.journal_write_failed is True


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


def test_crash_between_create_and_persist_recovered_next_cycle(tmp_path):
    """End-to-end regression (M3): cycle 1 creates the order on the
    exchange, then the connection dies on the first wait-for-ack poll
    (same shape as a SIGKILL between create and persist). The cycle
    fails loud — but the order must land in state as 'open' so cycle 2's
    reconstruction finds the fill and journals it. Before the fix the
    order was invisible forever: no state entry, no journal record, the
    fill silently dissolved into the balance, and no startup open-order
    discovery exists."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )

    # --- Cycle 1: create acked, then every wait poll dies ---------------
    stub1 = _LiveStub(basket=("BTC",))
    stub1.fetch_order_responses[coid] = [
        ccxt.OrderNotFound("query-before-place: not there yet"),
        # Broker retries transient reads (retry_max_attempts=3).
        ccxt.NetworkError("dropped right after create"),
        ccxt.NetworkError("still down"),
        ccxt.NetworkError("still down"),
    ]
    clock = _MockClock()
    with pytest.raises(ccxt.NetworkError):
        run_live_cycle(
            broker=_broker(stub1, basket=("BTC",)),
            journal=_journal(tmp_path), state=state,
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    assert len(stub1.create_order_calls) == 1     # order IS on the exchange
    entry = state.get(coid)
    assert entry is not None, "crashed placement must be visible in state"
    assert entry.status == "open"

    # --- Cycle 2: reconstruction resolves the orphan --------------------
    stub2 = _LiveStub(basket=("BTC",))
    stub2.fetch_order_responses[coid] = [
        _closed_order(coid, "BTC/USDT", filled=entry.intended_amount),
    ]
    result = run_live_cycle(
        broker=_broker(stub2, basket=("BTC",)),
        journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 1
    assert stub2.create_order_calls == []          # recovered, not re-placed
    assert state.get(coid).status == "closed"

    cycles = _read_cycles(tmp_path)
    # cycle 1 failed loud; cycle 2 = reconstruction entry + main entry.
    assert [c["outcome"] for c in cycles[:2]] == ["failed", "reconstructed"]
    recon = cycles[1]["orders_executed"][0]
    assert recon["client_order_id"] == coid
    assert recon["terminal_status"] == "closed"
    assert recon["filled_amount"] == pytest.approx(entry.intended_amount)


def _still_open_order(coid: str, symbol: str, exchange_id: str) -> dict:
    return {
        "id": exchange_id, "clientOrderId": coid, "symbol": symbol,
        "side": "buy", "status": "open", "filled": 0.0, "cost": 0.0,
        "average": None, "fee": {"cost": 0.0}, "timestamp": 1717000000000,
    }


def test_pending_order_symbol_excluded_from_next_plan(tmp_path):
    """Day-boundary half of the idempotency hole: yesterday's BUY is
    still live on the exchange (its expected fill is not in the balance)
    and today's coid is a different date, so neither the state fast-path
    nor query-before-place stops a fresh BUY for the same pair — the
    position doubles once both fill. The pair must sit out this cycle as
    a first-class 'pending_order' skip; other pairs trade normally."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    yesterday_dt = datetime.now(timezone.utc) - timedelta(days=1)
    stale_coid = make_client_order_id(yesterday_dt.date(), "BTC/USDT", "buy")
    state.put(OrderStateEntry(
        client_order_id=stale_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.05, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=yesterday_dt.isoformat(),
        last_seen_at=yesterday_dt.isoformat(),
    ))

    stub = _LiveStub(basket=("BTC", "ETH"))
    # Reconstruction finds the order STILL open on the exchange.
    stub.fetch_order_responses[stale_coid] = [
        _still_open_order(stale_coid, "BTC/USDT", "exch-live-1"),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    placed_pairs = [c["symbol"] for c in stub.create_order_calls]
    assert placed_pairs == ["ETH/USDT"], (
        "BTC has a live foreign-coid order — a fresh BTC buy doubles "
        "the position once both fill"
    )
    assert result.outcome == "success"
    cycle = _read_cycles(tmp_path)[-1]
    skips = [s for s in cycle["orders_skipped"]
             if s["reason"] == "pending_order"]
    assert len(skips) == 1 and skips[0]["symbol"] == "BTC/USDT"
    # The transient pending skip must NOT inflate the sub-min drift
    # metric (it resolves next cycle; the metric tracks unfillable
    # divergence).
    assert cycle["total_skipped_quote_drift"] == 0.0


def test_buys_deferred_when_funding_sell_is_pending_blocked(tmp_path):
    """Sells place before buys because their proceeds fund the buys.
    When a pending foreign-coid order blocks a SELL, the buys were sized
    against equity that counts the blocked base — its proceeds are not
    in free quote, so sending them collects an InsufficientFunds
    rejection. They must sit out as first-class transient skips;
    unblocked sells still place (they reduce risk and need no funding)."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    yesterday_dt = datetime.now(timezone.utc) - timedelta(days=1)
    stale_coid = make_client_order_id(yesterday_dt.date(), "BTC/USDT", "sell")
    state.put(OrderStateEntry(
        client_order_id=stale_coid, symbol="BTC/USDT", side="sell",
        intended_amount=0.05, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=yesterday_dt.isoformat(),
        last_seen_at=yesterday_dt.isoformat(),
    ))

    # Equity 15000 → target 5000/asset. BTC sell 2500 (blocked by the
    # pending order), ADA sell 1000, ETH buy 5000 — but free quote is
    # only 1500: the ETH buy needs the blocked BTC sell's proceeds.
    stub = _LiveStub(
        basket=("BTC", "ETH", "ADA"),
        balance_usdt=1500.0,
        asset_holdings={"BTC": 0.15, "ADA": 0.12},
    )
    stub.fetch_order_responses[stale_coid] = [
        _still_open_order(stale_coid, "BTC/USDT", "exch-live-1"),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub, basket=("BTC", "ETH", "ADA")),
        journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    placed = [(c["symbol"], c["side"]) for c in stub.create_order_calls]
    assert placed == [("ADA/USDT", "sell")], (
        "the buy must not reach the exchange without its funding sell"
    )
    cycle = _read_cycles(tmp_path)[-1]
    reasons = {s["symbol"]: s["reason"] for s in cycle["orders_skipped"]}
    assert reasons["BTC/USDT"] == "pending_order"
    assert reasons["ETH/USDT"] == "pending_funding_sell"
    # Both skips are transient — neither belongs in the drift metric.
    assert cycle["total_skipped_quote_drift"] == 0.0


def test_submin_skip_on_pending_pair_reclassified_as_pending(tmp_path):
    """A pending pair whose stale-balance delta falls below the exchange
    minimum lands in plan.skipped, bypassing the pending filter over
    plan.orders — the journal would label it sub-min drift and count it
    in total_skipped_quote_drift, though the pending fill invalidates
    the delta. It must be reclassified 'pending_order' and excluded from
    the drift metric; a genuine sub-min pair (ADA) keeps both."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    yesterday_dt = datetime.now(timezone.utc) - timedelta(days=1)
    stale_coid = make_client_order_id(yesterday_dt.date(), "BTC/USDT", "buy")
    state.put(OrderStateEntry(
        client_order_id=stale_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.05, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=yesterday_dt.isoformat(),
        last_seen_at=yesterday_dt.isoformat(),
    ))

    # Equity 15000 → target 5000/asset. BTC delta = 5 USDT (sub-min,
    # pending order live), ADA delta = 2 USDT (sub-min, no pending),
    # ETH delta = 5000 USDT (trades normally).
    stub = _LiveStub(
        basket=("BTC", "ETH", "ADA"),
        balance_usdt=5007.0,
        asset_holdings={"BTC": 0.0999, "ADA": 0.09996},
    )
    stub.fetch_order_responses[stale_coid] = [
        _still_open_order(stale_coid, "BTC/USDT", "exch-live-1"),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub, basket=("BTC", "ETH", "ADA")),
        journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    assert [c["symbol"] for c in stub.create_order_calls] == ["ETH/USDT"]
    cycle = _read_cycles(tmp_path)[-1]
    reasons = {s["symbol"]: s["reason"] for s in cycle["orders_skipped"]}
    assert reasons["BTC/USDT"] == "pending_order", (
        "sub-min delta on a pending pair is computed against a stale "
        "balance — it is transient, not unfillable drift"
    )
    assert "min_cost" in reasons["ADA/USDT"]
    assert cycle["total_skipped_quote_drift"] == pytest.approx(2.0)


def test_same_day_retry_with_same_coid_is_not_blocked(tmp_path):
    """A same-day retry plans the SAME coid the pending entry carries —
    query-before-place finds that exact order and waits on it (no
    duplicate is possible), so the pending_order skip must not fire:
    blocking would defer the resolution a full day for nothing."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    today_coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    state.put(OrderStateEntry(
        client_order_id=today_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.1, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=now_iso, last_seen_at=now_iso,
    ))

    stub = _LiveStub(basket=("BTC", "ETH"))
    # An open buy locks its quote: $10k equity, $5k of it held by BTC.
    stub.balance["USDT"] = {"free": 5_000.0, "used": 5_000.0, "total": 10_000.0}
    stub.fetch_order_responses[today_coid] = [
        # reconstruction: still open → left in state
        _still_open_order(today_coid, "BTC/USDT", "exch-live-1"),
        # query-before-place inside place_order: found, then terminal
        _still_open_order(today_coid, "BTC/USDT", "exch-live-1"),
        # $10k / 2 assets / $50k: the already-funded BTC intent keeps its
        # full 0.1 — the reserve cap (#28) only shaves what quote_free
        # still has to fund.
        _closed_order(today_coid, "BTC/USDT", filled=0.1),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "success"
    # BTC resolved via query-before-place (no create), ETH created.
    assert [c["symbol"] for c in stub.create_order_calls] == ["ETH/USDT"]
    cycle = _read_cycles(tmp_path)[-1]
    assert [s for s in cycle["orders_skipped"]
            if s["reason"] == "pending_order"] == []
    assert state.get(today_coid).status == "closed"


def test_reserve_cap_sized_after_pending_filter(tmp_path):
    """Codex review worked example: $5k locked by a pending BTC buy +
    $5k free; targets $5k BTC / $5k ETH. The BTC intent sits out as
    pending, so the cap must size the ETH buy off the full free $5k —
    capping the PRE-filter plan scales both buys to ~$2.5k and strands
    the filtered pair's share of the quote."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    yesterday_dt = datetime.now(timezone.utc) - timedelta(days=1)
    stale_coid = make_client_order_id(yesterday_dt.date(), "BTC/USDT", "buy")
    state.put(OrderStateEntry(
        client_order_id=stale_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.1, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=yesterday_dt.isoformat(),
        last_seen_at=yesterday_dt.isoformat(),
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    # The pending BTC buy holds $5k of the quote in `used`.
    stub.balance["USDT"] = {"free": 5_000.0, "used": 5_000.0, "total": 10_000.0}
    stub.fetch_order_responses[stale_coid] = [
        _still_open_order(stale_coid, "BTC/USDT", "exch-live-1"),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    [call] = stub.create_order_calls
    assert call["symbol"] == "ETH/USDT"
    # $5k free × (1 − 10 bp) — NOT ≈$2.5k from a pre-filter cap.
    assert call["amount"] * 50_000.0 == pytest.approx(5_000.0 * 0.999)
    cycle = _read_cycles(tmp_path)[-1]
    reasons = {s["symbol"]: s["reason"] for s in cycle["orders_skipped"]}
    assert reasons == {"BTC/USDT": "pending_order", "ETH/USDT": "funding_cap"}
    # The pending skip carries the UNSCALED gap, not a capped remnant.
    btc = next(s for s in cycle["orders_skipped"]
               if s["symbol"] == "BTC/USDT")
    assert btc["desired_notional"] == pytest.approx(5_000.0)
    # The 10 bp shave is metered as funding cap, NOT as sub-min drift
    # (monitoring reads that field as "unfillable"); the transient
    # pending skip stays out of both.
    assert cycle["total_skipped_quote_drift"] == 0.0
    assert cycle["total_funding_cap_quote"] == pytest.approx(5_000.0 * 0.001)


def test_reserve_cap_ignores_already_funded_same_coid_buy(tmp_path):
    """Second-round Codex example: the same-coid BTC buy from an earlier
    run today is still open and holds $5k of the quote, leaving $5k free.
    That intent deliberately survives the pending filter (place_order
    resolves the existing order), but its quote is NOT in quote_free —
    capping it again scales BOTH buys to ~$2.5k while the untouched $5k
    BTC order resolves anyway, so half the free quote never leaves. The
    funded buy must stay out of the cap's spend total and its scaling."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    today_coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    state.put(OrderStateEntry(
        client_order_id=today_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.1, status="timeout",
        exchange_order_id="exch-live-1",
        placed_at=now_iso, last_seen_at=now_iso,
    ))

    stub = _LiveStub(basket=("BTC", "ETH"))
    # The open BTC buy locks its $5k: equity is still $10k, free is $5k.
    stub.balance["USDT"] = {"free": 5_000.0, "used": 5_000.0, "total": 10_000.0}
    stub.fetch_order_responses[today_coid] = [
        _still_open_order(today_coid, "BTC/USDT", "exch-live-1"),  # recon
        _still_open_order(today_coid, "BTC/USDT", "exch-live-1"),  # pre-place
        _closed_order(today_coid, "BTC/USDT", filled=0.1),
    ]
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    [call] = stub.create_order_calls
    assert call["symbol"] == "ETH/USDT"
    # The whole free $5k less the 10 bp reserve — not ~$2.5k.
    assert call["amount"] * 50_000.0 == pytest.approx(5_000.0 * 0.999)
    cycle = _read_cycles(tmp_path)[-1]
    btc = next(o for o in cycle["orders_executed"]
               if o["symbol"] == "BTC/USDT")
    assert btc["intended_amount"] == pytest.approx(0.1), (
        "scaling a buy the exchange already funded is lost work: "
        "place_order resolves the original order, not a smaller one"
    )
    reasons = {s["symbol"]: s["reason"] for s in cycle["orders_skipped"]}
    assert reasons == {"ETH/USDT": "funding_cap"}
    assert cycle["total_skipped_quote_drift"] == 0.0
    assert cycle["total_funding_cap_quote"] == pytest.approx(5_000.0 * 0.001)
    assert state.get(today_coid).status == "closed"


def test_reserve_cap_ignores_buy_whose_today_coid_is_already_terminal(tmp_path):
    """Issue #70 worked example: today's BTC coid resolved 'canceled' in
    an earlier run, so place_order's state fast-path sends nothing for
    it — it consumes no quote. quote_free $3k, two $5k intents: the free
    quote must fund the ETH buy in full (less the 10 bp reserve), NOT be
    halved by a BTC buy that never leaves."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    today_coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    state.put(OrderStateEntry(
        client_order_id=today_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.1, status="canceled",
        exchange_order_id="exch-live-1",
        placed_at=now_iso, last_seen_at=now_iso,
    ))

    stub = _LiveStub(basket=("BTC", "ETH"))
    # Equity $10k → $5k target per asset; only $3k of the quote is free.
    stub.balance["USDT"] = {"free": 3_000.0, "used": 7_000.0, "total": 10_000.0}
    clock = _MockClock()
    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    # A canceled order today is a real divergence — still surfaced.
    assert result.outcome == "partial"
    [call] = stub.create_order_calls
    assert call["symbol"] == "ETH/USDT"
    # The whole free $3k less the 10 bp reserve — not ~$1.5k from a
    # scale of 2997/10000 that counts the dead BTC intent.
    assert call["amount"] * 50_000.0 == pytest.approx(3_000.0 * 0.999)
    # The terminal coid is resolved from state alone: no roundtrip.
    assert all(c["symbol"] != "BTC/USDT" for c in stub.fetch_order_calls)
    cycle = _read_cycles(tmp_path)[-1]
    btc = next(o for o in cycle["orders_executed"]
               if o["symbol"] == "BTC/USDT")
    assert btc["terminal_status"] == "canceled"
    assert btc["filled_amount"] == 0.0
    reasons = {s["symbol"]: s["reason"] for s in cycle["orders_skipped"]}
    assert reasons == {"ETH/USDT": "funding_cap"}
    assert cycle["total_skipped_quote_drift"] == 0.0
    assert cycle["total_funding_cap_quote"] == pytest.approx(5_000.0 - 2_997.0)


def test_day_two_rerun_after_filled_day_one_places_nothing(tmp_path, monkeypatch):
    """Day-boundary pin of the recompute-from-balance property: day 1's
    buys FILLED and the balance reflects them → a day-2 run (new coid,
    so no idempotency layer applies) must compute delta ≈ 0 and place
    nothing. This property was emergent and unpinned — the cron has
    already mis-fired across a date boundary once (MSK/UTC)."""
    from trade_lab.execution import live_cycle as lc

    state = _state(tmp_path)
    stub1 = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()
    result1 = run_live_cycle(
        _broker(stub1), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result1.outcome == "success"
    bought = {
        c["symbol"].split("/")[0]: c["amount"]
        for c in stub1.create_order_calls
    }
    assert set(bought) == {"BTC", "ETH"}

    # Day 2: same market, balance now holds day 1's fills.
    quote_spent = sum(amt * 50_000.0 for amt in bought.values())
    stub2 = _LiveStub(
        basket=("BTC", "ETH"),
        balance_usdt=10_000.0 - quote_spent,
        asset_holdings=bought,
    )

    real_datetime = datetime

    class _DayTwo:
        @staticmethod
        def now(tz=None):
            return real_datetime.now(tz) + timedelta(days=1)

        @staticmethod
        def fromtimestamp(*a, **k):
            return real_datetime.fromtimestamp(*a, **k)

    monkeypatch.setattr(lc, "datetime", _DayTwo)
    result2 = run_live_cycle(
        _broker(stub2), journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result2.outcome == "success"
    assert stub2.create_order_calls == [], (
        "with fills reflected in the balance, the day-2 delta must be "
        "below exchange minimums — a placement here doubles the position"
    )


def test_timeout_on_create_itself_recovered_next_cycle(tmp_path):
    """The other half of M3: the create *request* dies in flight after
    reaching Binance. Cycle 1 fails loud leaving a 'pending_create'
    intent; the next cycle's reconstruction finds the order by coid and
    journals its fill. Before the fix there was no state entry at all —
    the fill silently dissolved into the balance."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )

    # --- Cycle 1: create raises after the request reached the exchange --
    stub1 = _LiveStub(
        basket=("BTC",), create_raises=ccxt.RequestTimeout("response lost"),
    )
    clock = _MockClock()
    with pytest.raises(ccxt.RequestTimeout):
        run_live_cycle(
            broker=_broker(stub1, basket=("BTC",)),
            journal=_journal(tmp_path), state=state,
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    assert len(stub1.create_order_calls) == 1
    entry = state.get(coid)
    assert entry is not None, "lost create must be visible in state"
    assert entry.status == "pending_create"

    # --- Cycle 2: the order DID land on the exchange ---------------------
    stub2 = _LiveStub(basket=("BTC",))
    stub2.fetch_order_responses[coid] = [
        _closed_order(coid, "BTC/USDT", filled=entry.intended_amount),
    ]
    result = run_live_cycle(
        broker=_broker(stub2, basket=("BTC",)),
        journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 1
    assert stub2.create_order_calls == []          # recovered, not re-placed
    assert state.get(coid).status == "closed"
    recon = [c for c in _read_cycles(tmp_path) if c["outcome"] == "reconstructed"][-1]
    assert recon["orders_executed"][0]["client_order_id"] == coid
    assert recon["orders_executed"][0]["terminal_status"] == "closed"


def test_never_created_pending_resolves_clean_and_same_day_retry_places(tmp_path):
    """Create timed out and the request never reached the exchange.
    Reconstruction must resolve the pending intent as 'not_created' —
    a clean resolution, NOT a lost_track incident — and remove it from
    state so a same-day retry's state fast-path cannot skip the real
    placement."""
    from trade_lab.execution.clientorder import make_client_order_id

    state = _state(tmp_path)
    coid = make_client_order_id(
        datetime.now(timezone.utc).date(), "BTC/USDT", "buy",
    )

    stub1 = _LiveStub(
        basket=("BTC",), create_raises=ccxt.RequestTimeout("never arrived"),
    )
    clock = _MockClock()
    with pytest.raises(ccxt.RequestTimeout):
        run_live_cycle(
            broker=_broker(stub1, basket=("BTC",)),
            journal=_journal(tmp_path), state=state,
            sleep_fn=clock.sleep, time_fn=clock.time,
        )
    assert state.get(coid).status == "pending_create"

    # --- Same-day retry: exchange has no record of the coid --------------
    stub2 = _LiveStub(basket=("BTC",))
    result = run_live_cycle(
        broker=_broker(stub2, basket=("BTC",)),
        journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 1
    assert result.lost_track_count == 0, "not an incident: nothing was placed"
    recon = [c for c in _read_cycles(tmp_path) if c["outcome"] == "reconstructed"][-1]
    assert recon["orders_executed"][0]["terminal_status"] == "not_created"
    assert recon["orders_executed"][0]["error"] is None
    # The retry re-placed the order for real — the resolved intent must
    # not satisfy the state fast-path.
    assert len(stub2.create_order_calls) == 1
    assert state.get(coid).status == "closed"


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


@pytest.mark.parametrize("ccxt_status", ["expired", "rejected"])
def test_reconstruction_resolves_expired_rejected(tmp_path, ccxt_status):
    """Regression (M1): a stale non-terminal state entry (e.g. journaled
    as 'timeout' before expired/rejected were terminal) whose exchange
    status is now expired/rejected must resolve to a terminal state entry
    on the next cycle instead of being re-polled forever."""
    state = _state(tmp_path)
    pending_coid = "tsmom_20260528_BTCUSDT_buy"
    state.put(OrderStateEntry(
        client_order_id=pending_coid, symbol="BTC/USDT", side="buy",
        intended_amount=0.001, status="timeout",
        exchange_order_id="exch-prior",
        placed_at="2026-05-28T00:05:00+00:00",
        last_seen_at="2026-05-28T00:05:00+00:00",
    ))
    stub = _LiveStub(basket=("BTC", "ETH"))
    stub.fetch_order_responses[pending_coid] = [
        {"id": "exch-prior", "status": ccxt_status, "filled": 0.0,
         "cost": 0.0, "average": None, "timestamp": 0},
    ]
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=state,
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.reconstructed_count == 1
    recon = _read_cycles(tmp_path)[0]
    assert recon["outcome"] == "reconstructed"
    assert recon["orders_executed"][0]["client_order_id"] == pending_coid
    assert recon["orders_executed"][0]["terminal_status"] == ccxt_status
    # Terminal in state → the cycle after this one has nothing to redo.
    assert state.get(pending_coid).status == ccxt_status
    assert pending_coid not in state.open_entries()


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
    """Hold BTC + ETH while signal=0 → exactly one sell per held asset,
    no buys, all fills journaled. Exact counts, not just all(...) over
    the observed sides — that predicate is vacuously true on an empty
    call list, so a regression placing zero orders would pass."""
    stub = _LiveStub(
        basket=("BTC", "ETH"),
        balance_usdt=0.0,
        asset_holdings={"BTC": 0.1, "ETH": 0.5},
        closes=np.linspace(200, 100, 500).tolist(),  # signal=0
    )
    broker = _broker(stub)
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "success"
    assert len(stub.create_order_calls) == 2    # one per held asset
    for call in stub.create_order_calls:
        assert call["side"] == "sell"
        assert call["amount"] > 0
    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "success"
    executed = cycle["orders_executed"]
    assert len(executed) == 2
    assert all(o["side"] == "sell" for o in executed)
    assert all(o["terminal_status"] == "closed" for o in executed)


def test_full_liquidation_sells_every_held_asset(tmp_path):
    """Ladder→0 across a fully-held 4-asset basket → exactly one sell
    per held asset with a positive amount, zero buys, and a journaled
    closed fill for each. Pins the liquidation path end to end: an
    empty plan or a silently dropped sell breaks the exact counts."""
    holdings = {"BTC": 0.1, "ETH": 0.5, "ADA": 0.2, "SOL": 0.3}
    basket = tuple(holdings)
    stub = _LiveStub(
        basket=basket,
        balance_usdt=0.0,
        asset_holdings=holdings,
        closes=np.linspace(200, 100, 500).tolist(),  # signal=0
    )
    broker = _broker(stub, basket=basket)
    clock = _MockClock()
    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )
    assert result.outcome == "success"
    assert len(result.order_results) == len(holdings)
    assert len(stub.create_order_calls) == len(holdings)
    assert {c["symbol"] for c in stub.create_order_calls} == {
        f"{a}/USDT" for a in holdings
    }
    for call in stub.create_order_calls:
        assert call["side"] == "sell"
        assert call["amount"] > 0
    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "success"
    executed = cycle["orders_executed"]
    assert len(executed) == len(holdings)
    assert all(o["side"] == "sell" for o in executed)
    assert all(o["terminal_status"] == "closed" for o in executed)
    assert all(o["filled_amount"] > 0 for o in executed)


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


class _EthTickerDownStub(_LiveStub):
    def fetch_ticker(self, symbol):
        if symbol == "ETH/USDT":
            return {}  # no last/close → BrokerError in fetch_ticker_price
        return super().fetch_ticker(symbol)


def test_live_cycle_journals_candle_close_price_fallback(tmp_path):
    """A failed ticker falls back to the candle close for order sizing
    (documented posture) — the main-cycle journal entry must mark the
    symbol with the price's source and age so the sizing miss is
    attributable when reconciling execution against the backtest."""
    from trade_lab.execution.signal import decision_age_seconds

    stub = _EthTickerDownStub(basket=("BTC", "ETH"))
    broker = _broker(stub)
    clock = _MockClock()

    result = run_live_cycle(
        broker, journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    cycle = _read_cycles(tmp_path)[-1]
    fb = cycle["price_fallbacks"]
    assert set(fb) == {"ETH"}
    assert fb["ETH"]["source"] == "candle_close_fallback"
    expected_age = decision_age_seconds(pd.Timestamp(cycle["signal"]["asof"]))
    assert fb["ETH"]["age_s"] == pytest.approx(expected_age, abs=60.0)
    # Sizing used the candle close (stub closes end at 300.0), not the
    # healthy 50k ticker and not 0.0.
    eth = next(o for o in cycle["orders_planned"] if o["symbol"] == "ETH/USDT")
    assert eth["price_used"] == pytest.approx(300.0)


def test_live_cycle_all_tickers_ok_journals_no_price_fallbacks(tmp_path):
    stub = _LiveStub(basket=("BTC", "ETH"))
    clock = _MockClock()

    result = run_live_cycle(
        _broker(stub), journal=_journal(tmp_path), state=_state(tmp_path),
        sleep_fn=clock.sleep, time_fn=clock.time,
    )

    assert result.outcome == "success"
    assert _read_cycles(tmp_path)[-1]["price_fallbacks"] is None


def test_failed_cycle_after_read_phase_keeps_price_fallbacks(tmp_path):
    """A cycle that dies AFTER the read phase (Ctrl-C between order
    placements) must journal the fallback markers — the incident path
    with partially-placed fallback-priced orders is exactly what the
    field exists to audit."""
    stub = _EthTickerDownStub(basket=("BTC", "ETH"))
    broker = _broker(stub)

    def _ctrl_c_sleep(_s: float) -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_live_cycle(
            broker, journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=_ctrl_c_sleep, time_fn=lambda: 0.0,
        )

    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "failed"
    fb = cycle["price_fallbacks"]
    assert set(fb) == {"ETH"}
    assert fb["ETH"]["source"] == "candle_close_fallback"
    # The order placed before the interrupt stays recorded alongside.
    executed = cycle["orders_executed"]
    assert executed is not None and len(executed) == 1


def test_failed_cycle_before_read_phase_price_fallbacks_none(tmp_path):
    """A failure before the read phase produced any prices journals
    price_fallbacks=None (and must not NameError on the pre-bound
    read variable)."""
    class _BalanceDownStub(_LiveStub):
        def fetch_balance(self):
            raise RuntimeError("balance down")

    stub = _BalanceDownStub(basket=("BTC", "ETH"))
    clock = _MockClock()

    with pytest.raises(RuntimeError, match="balance down"):
        run_live_cycle(
            _broker(stub), journal=_journal(tmp_path), state=_state(tmp_path),
            sleep_fn=clock.sleep, time_fn=clock.time,
        )

    cycle = _read_cycles(tmp_path)[-1]
    assert cycle["outcome"] == "failed"
    assert cycle["price_fallbacks"] is None
    # No orders were placed: None, never [] (failed-vs-empty invariant).
    assert cycle["orders_executed"] is None


def test_live_same_coid_sell_still_funds_the_buys(monkeypatch, tmp_path):
    """A same-coid SELL left open by reconstruction is waited for by the
    executor before the buys go out, so its proceeds still fund them.
    Excluding it (as a TERMINAL sell is excluded) would cap the buys away
    and end the cycle underallocated."""
    from trade_lab.execution.delta import OrderIntent, apply_reserve_cap

    sell = OrderIntent(
        symbol="BTC/USDT", side="sell", base_amount=0.2,
        notional_quote=10_000.0, price_used=50_000.0, reason="delta",
    )
    buy = OrderIntent(
        symbol="ETH/USDT", side="buy", base_amount=3.0,
        notional_quote=9_000.0, price_used=3_000.0, reason="delta",
    )
    # Sell counted (live, still expected to settle) → the buy survives.
    capped, _ = apply_reserve_cap(
        [sell, buy], quote_free=0.0, constraints={}, fee_rate=0.001,
    )
    assert any(o.side == "buy" for o in capped)
    # Sell excluded (terminal) → nothing funds the buy, it is capped away.
    capped_terminal, _ = apply_reserve_cap(
        [sell, buy], quote_free=0.0, constraints={}, fee_rate=0.001,
        no_flow_symbols={"BTC/USDT"},
    )
    assert not any(o.side == "buy" for o in capped_terminal)
