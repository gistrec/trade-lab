"""Tests for the broker abstraction with a mocked CCXT exchange.

No live API, no network. The mock exchange satisfies the
``_CcxtExchange`` Protocol structurally and lets us assert that the
broker calls it correctly and surfaces errors clearly.
"""
from __future__ import annotations

import time

import pytest

import ccxt

from trade_lab.execution.broker import (
    BalanceSnapshot, Broker, BrokerError, ConnectionRefused,
)
from trade_lab.execution.config import PaperConfig


# ---------------------------------------------------------------------------
# Mock exchange satisfying the Protocol
# ---------------------------------------------------------------------------


class _MockExchange:
    """Implements just enough of the CCXT exchange surface for the broker."""

    id = "mock"

    def __init__(self):
        self.sandbox = False
        self.balance = {
            "USDT": {"free": 1000.0, "used": 50.0, "total": 1050.0},
            "BTC":  {"free": 0.1,    "used": 0.0,  "total": 0.1},
            "ETH":  {"free": 0.0,    "used": 0.0,  "total": 0.0},
        }
        self.tickers = {
            "BTC/USDT": {"last": 50000.0, "close": 50000.0},
            "ETH/USDT": {"last": 3000.0, "close": 3000.0},
        }
        # For failure-mode tests:
        self.fetch_balance_raises = None
        self.set_sandbox_mode_calls = 0

    def set_sandbox_mode(self, enabled):
        self.set_sandbox_mode_calls += 1
        self.sandbox = enabled

    def fetch_balance(self):
        if self.fetch_balance_raises:
            raise self.fetch_balance_raises
        return self.balance

    def fetch_ticker(self, symbol):
        if symbol not in self.tickers:
            raise ccxt.BadSymbol(f"unknown symbol {symbol}")
        return self.tickers[symbol]

    def fetch_status(self):
        return {"status": "ok"}

    def load_markets(self, reload=False):
        return {sym: {} for sym in self.tickers}


def _config(sandbox=True, allow_mainnet=False, clock_skew_max_ms=0) -> PaperConfig:
    return PaperConfig(
        exchange_id="binance",
        sandbox=sandbox,
        api_key="k",
        api_secret="s",
        allow_mainnet=allow_mainnet,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
        clock_skew_max_ms=clock_skew_max_ms,
    )


# ---------------------------------------------------------------------------
# Constructor + connection verification
# ---------------------------------------------------------------------------


def test_broker_constructed_with_exchange_keeps_reference():
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    assert broker.exchange is exch


def test_verify_connection_passes_on_ok_balance():
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    broker._verify_connection()  # must not raise


def test_verify_connection_raises_on_auth_error():
    exch = _MockExchange()
    exch.fetch_balance_raises = ccxt.AuthenticationError("bad key")
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="Authentication failed"):
        broker._verify_connection()


def test_verify_connection_raises_on_network_error():
    exch = _MockExchange()
    exch.fetch_balance_raises = ccxt.NetworkError("timeout")
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="Network error"):
        broker._verify_connection()


def test_verify_connection_raises_on_unexpected_exception():
    exch = _MockExchange()
    exch.fetch_balance_raises = RuntimeError("weird")
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="Unexpected error"):
        broker._verify_connection()


def test_verify_connection_raises_when_balance_not_dict():
    exch = _MockExchange()
    exch.balance = "not a dict"
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="non-dict"):
        broker._verify_connection()


# ---------------------------------------------------------------------------
# Mainnet safety gate at connect() level
# ---------------------------------------------------------------------------


def test_connect_refuses_mainnet_without_allow_flag(monkeypatch):
    """Even if a malformed config slips through, the connect classmethod
    refuses to point at mainnet without explicit allow_mainnet."""
    cfg = _config(sandbox=False, allow_mainnet=False)
    with pytest.raises(ConnectionRefused, match="mainnet"):
        Broker.connect(cfg)


def test_connect_uses_real_ccxt_for_unknown_exchange_id():
    """An exchange_id not present in CCXT must raise BrokerError, not
    silently fall back to anything."""
    cfg = PaperConfig(
        exchange_id="not_a_real_exchange",
        sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT",
        basket=("BTC",), request_timeout_ms=5000,
    )
    with pytest.raises(BrokerError, match="Unknown CCXT exchange"):
        Broker.connect(cfg)


# ---------------------------------------------------------------------------
# Balance snapshot
# ---------------------------------------------------------------------------


def test_fetch_balance_snapshot_pulls_quote_and_asset_totals():
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    snap = broker.fetch_balance_snapshot()
    assert isinstance(snap, BalanceSnapshot)
    assert snap.quote_currency == "USDT"
    assert snap.quote_free == 1000.0
    assert snap.quote_used == 50.0
    assert snap.quote_total == 1050.0
    assert snap.asset_totals == {"BTC": 0.1, "ETH": 0.0}


def test_fetch_balance_snapshot_handles_missing_currencies():
    """An asset in the basket with no entry on the exchange returns 0,
    not a KeyError. Real testnets often have empty balances on listed
    pairs."""
    exch = _MockExchange()
    exch.balance = {"USDT": {"free": 100.0, "used": 0.0, "total": 100.0}}
    broker = Broker(_config(), exch)
    snap = broker.fetch_balance_snapshot()
    assert snap.asset_totals == {"BTC": 0.0, "ETH": 0.0}


def test_fetch_balance_snapshot_handles_none_balance_fields():
    """CCXT sometimes returns None for unused fields; we coerce to 0."""
    exch = _MockExchange()
    exch.balance = {"USDT": {"free": None, "used": None, "total": None}}
    broker = Broker(_config(), exch)
    snap = broker.fetch_balance_snapshot()
    assert snap.quote_free == 0.0
    assert snap.quote_total == 0.0


def test_fetch_balance_snapshot_does_not_cache():
    """Two consecutive calls must each round-trip — the broker must NOT
    return stale state. The mock counts calls implicitly via the
    fetch_balance method; we change the balance between calls and
    confirm the second result reflects the change."""
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    first = broker.fetch_balance_snapshot()
    # Simulate the testnet wiping or topping up our balance mid-session.
    exch.balance["USDT"]["total"] = 9999.0
    exch.balance["USDT"]["free"] = 9999.0
    second = broker.fetch_balance_snapshot()
    assert first.quote_total == 1050.0
    assert second.quote_total == 9999.0


# ---------------------------------------------------------------------------
# Ticker pricing
# ---------------------------------------------------------------------------


def test_fetch_ticker_price_uses_last_then_close():
    exch = _MockExchange()
    exch.tickers["BTC/USDT"] = {"last": 51000.0, "close": 50000.0}
    broker = Broker(_config(), exch)
    assert broker.fetch_ticker_price("BTC/USDT") == 51000.0


def test_fetch_ticker_price_falls_back_to_close_when_last_missing():
    exch = _MockExchange()
    exch.tickers["BTC/USDT"] = {"last": None, "close": 50000.0}
    broker = Broker(_config(), exch)
    assert broker.fetch_ticker_price("BTC/USDT") == 50000.0


def test_fetch_ticker_price_raises_when_both_missing():
    exch = _MockExchange()
    exch.tickers["BTC/USDT"] = {}
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="no last/close"):
        broker.fetch_ticker_price("BTC/USDT")


# ---------------------------------------------------------------------------
# Equity estimate
# ---------------------------------------------------------------------------


def test_estimate_total_equity_marks_assets_to_market():
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    # USDT total = 1050; BTC 0.1 × $50k = $5000; ETH 0 → 0.
    equity = broker.estimate_total_equity_usd()
    assert equity == pytest.approx(1050.0 + 5000.0)


def test_estimate_total_equity_raises_on_failing_ticker():
    """A held position that cannot be marked must raise, not count as
    zero — understated equity shrinks every target and turns one
    missing price into spurious sells across the whole basket."""
    exch = _MockExchange()
    # Empty the BTC ticker so its mark fails (BTC total is non-zero).
    exch.tickers["BTC/USDT"] = {}
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="no last/close"):
        broker.estimate_total_equity_usd()


def test_fetch_market_constraints_normalizes_limits():
    exch = _MockExchange()
    # Override load_markets to expose CCXT-style limits.
    exch.load_markets = lambda reload=False: {
        "BTC/USDT": {
            "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10.0}},
            "precision": {"amount": 8},
        }
    }
    broker = Broker(_config(), exch)
    c = broker.fetch_market_constraints("BTC/USDT")
    assert c.min_amount == 0.0001
    assert c.min_cost == 10.0
    assert c.amount_precision == 8


def test_fetch_market_constraints_handles_missing_fields():
    exch = _MockExchange()
    exch.load_markets = lambda reload=False: {
        "BTC/USDT": {"limits": {}, "precision": {}}
    }
    broker = Broker(_config(), exch)
    c = broker.fetch_market_constraints("BTC/USDT")
    assert c.min_amount is None
    assert c.min_cost is None
    assert c.amount_precision is None


def test_fetch_market_constraints_raises_for_unknown_pair():
    exch = _MockExchange()
    exch.load_markets = lambda reload=False: {}
    broker = Broker(_config(), exch)
    with pytest.raises(BrokerError, match="not found"):
        broker.fetch_market_constraints("WAT/USDT")


def test_estimate_total_equity_accepts_explicit_snapshot():
    """Passing a snapshot avoids the duplicate fetch_balance call."""
    exch = _MockExchange()
    broker = Broker(_config(), exch)
    snap = broker.fetch_balance_snapshot()
    exch.balance["USDT"]["total"] = 99999.0  # changes after the snapshot
    equity = broker.estimate_total_equity_usd(snapshot=snap)
    # We computed from the OLD snapshot, not the new exchange state.
    assert equity == pytest.approx(1050.0 + 5000.0)


# ---------------------------------------------------------------------------
# Precision normalization — TICK_SIZE vs DECIMAL_PLACES
# ---------------------------------------------------------------------------


def test_constraints_tick_size_step_converted_to_decimals():
    """Binance (ccxt default) reports precision as a step: 1e-05 means
    5 decimals. int(1e-05) used to store 0 — "whole units only"."""
    exch = _MockExchange()
    exch.precisionMode = ccxt.TICK_SIZE
    exch.load_markets = lambda reload=False: {
        "BTC/USDT": {"limits": {}, "precision": {"amount": 1e-05}},
    }
    c = Broker(_config(), exch).fetch_market_constraints("BTC/USDT")
    assert c.amount_precision == 5


def test_constraints_tick_size_whole_units():
    exch = _MockExchange()
    exch.precisionMode = ccxt.TICK_SIZE
    exch.load_markets = lambda reload=False: {
        "DOGE/USDT": {"limits": {}, "precision": {"amount": 1.0}},
    }
    c = Broker(_config(), exch).fetch_market_constraints("DOGE/USDT")
    assert c.amount_precision == 0


def test_constraints_non_power_of_ten_step_maps_to_none():
    """A 0.5 step has no decimal-count equivalent; raw dict keeps it."""
    exch = _MockExchange()
    exch.precisionMode = ccxt.TICK_SIZE
    exch.load_markets = lambda reload=False: {
        "X/USDT": {"limits": {}, "precision": {"amount": 0.5}},
    }
    c = Broker(_config(), exch).fetch_market_constraints("X/USDT")
    assert c.amount_precision is None
    assert c.raw["precision"]["amount"] == 0.5


# ---------------------------------------------------------------------------
# Exchange round-trip instrumentation (#2a latency)
# ---------------------------------------------------------------------------


def test_exchange_call_stats_starts_empty():
    broker = Broker(_config(), _MockExchange())
    assert broker.exchange_call_stats() == {
        "count": 0, "errors": 0, "max_ms": 0.0, "p95_ms": 0.0,
        "total_ms": 0.0, "by_endpoint": {}}


def test_timed_calls_accumulate_stats():
    broker = Broker(_config(), _MockExchange())
    broker.fetch_balance_snapshot()
    broker.fetch_ticker_price("BTC/USDT")
    stats = broker.exchange_call_stats()
    assert stats["count"] == 2
    assert stats["errors"] == 0
    assert set(stats["by_endpoint"]) == {"fetch_balance", "fetch_ticker"}
    assert stats["max_ms"] >= 0.0
    assert stats["by_endpoint"]["fetch_ticker"]["count"] == 1


def test_timed_call_records_failure_and_reraises():
    # A failing call is still timed (ok=False) and the exception passes through
    # unchanged — the wrapper adds no control flow.
    broker = Broker(_config(), _MockExchange())
    with pytest.raises(ccxt.BadSymbol):
        broker.fetch_ticker_price("NOPE/USDT")
    stats = broker.exchange_call_stats()
    assert stats["errors"] == 1
    assert stats["by_endpoint"]["fetch_ticker"]["errors"] == 1


# ---------------------------------------------------------------------------
# Read-only retry on transient network errors (#2b). _config() leaves the
# dataclass default retry_base_delay_s=0.0, so retries are instant in tests.
# ---------------------------------------------------------------------------


def test_read_call_retries_transient_then_succeeds():
    exch = _MockExchange()
    n = {"c": 0}

    def flaky():
        n["c"] += 1
        if n["c"] < 3:
            raise ccxt.RequestTimeout("blip")  # a NetworkError subclass
        return exch.balance

    exch.fetch_balance = flaky
    broker = Broker(_config(), exch)
    snap = broker.fetch_balance_snapshot()
    assert snap.quote_total == 1050.0
    assert n["c"] == 3  # 2 transient failures + 1 success
    assert broker.exchange_call_stats()["by_endpoint"]["fetch_balance"]["count"] == 3


def test_read_call_gives_up_after_max_attempts():
    exch = _MockExchange()
    n = {"c": 0}

    def always_fail():
        n["c"] += 1
        raise ccxt.NetworkError("down")

    exch.fetch_balance = always_fail
    broker = Broker(_config(), exch)
    with pytest.raises(ccxt.NetworkError):
        broker.fetch_balance_snapshot()
    assert n["c"] == 3  # default retry_max_attempts


def test_read_call_does_not_retry_non_transient():
    exch = _MockExchange()
    n = {"c": 0}

    def auth_fail():
        n["c"] += 1
        raise ccxt.AuthenticationError("bad key")

    exch.fetch_balance = auth_fail
    broker = Broker(_config(), exch)
    with pytest.raises(ccxt.AuthenticationError):
        broker.fetch_balance_snapshot()
    assert n["c"] == 1  # non-transient: no retry


def test_create_order_is_not_retried():
    # Placement must never be retried at the broker (idempotency is via the
    # reconstruction path). A transient error surfaces after a single attempt.
    exch = _MockExchange()
    n = {"c": 0}

    def co(*a, **k):
        n["c"] += 1
        raise ccxt.NetworkError("timeout")

    exch.create_order = co
    broker = Broker(_config(), exch)
    with pytest.raises(ccxt.NetworkError):
        broker.create_order_safe("BTC/USDT", "buy", 0.001, "cid-1")
    assert n["c"] == 1  # timed once, NOT retried


# ---------------------------------------------------------------------------
# Clock-skew guard (NTP)
# ---------------------------------------------------------------------------


def test_clock_skew_within_limit_passes():
    exch = _MockExchange()
    exch.fetch_time = lambda: int(time.time() * 1000)  # in sync
    broker = Broker(_config(clock_skew_max_ms=1000), exch)
    broker._verify_connection()  # must not raise


def test_clock_skew_exceeds_limit_raises():
    exch = _MockExchange()
    exch.fetch_time = lambda: int(time.time() * 1000) + 5000  # 5s ahead
    broker = Broker(_config(clock_skew_max_ms=1000), exch)
    with pytest.raises(BrokerError, match="lock skew"):
        broker._verify_connection()


def test_clock_skew_disabled_skips_fetch_time():
    # clock_skew_max_ms=0 disables the guard, so a mock without fetch_time
    # still verifies cleanly.
    exch = _MockExchange()  # no fetch_time attribute
    broker = Broker(_config(clock_skew_max_ms=0), exch)
    broker._verify_connection()  # must not raise


def test_clock_skew_fetch_time_failure_raises_brokererror():
    exch = _MockExchange()

    def boom():
        raise ccxt.NetworkError("no time endpoint")

    exch.fetch_time = boom
    broker = Broker(_config(clock_skew_max_ms=1000), exch)
    with pytest.raises(BrokerError, match="server time"):
        broker._verify_connection()
