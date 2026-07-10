"""Tests for the dry-run orchestrator (end-to-end without orders)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.dry_run import DryRunResult, run_dry_cycle


def _config(basket=("BTC", "ETH")):
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


class _StubExchange:
    """A more featured stub that supports the dry-run pipeline."""
    id = "stub"

    def __init__(self, balance_usdt=10_000.0, btc_holdings=0.0):
        self.balance = {
            "USDT": {"free": balance_usdt, "used": 0.0, "total": balance_usdt},
            "BTC":  {"free": btc_holdings, "used": 0.0, "total": btc_holdings},
            "ETH":  {"free": 0.0,           "used": 0.0, "total": 0.0},
        }
        self.tickers = {
            "BTC/USDT": {"last": 50_000.0, "close": 50_000.0},
            "ETH/USDT": {"last": 3_000.0,  "close": 3_000.0},
        }
        # OHLCV: clean uptrend so signal=1.0, gate open.
        self._closes = (100 + np.linspace(0, 200, 500)).tolist()

    def set_sandbox_mode(self, enabled): pass

    def fetch_balance(self):
        return self.balance

    def fetch_ticker(self, symbol):
        return self.tickers[symbol]

    def fetch_status(self):
        return {"status": "ok"}

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=400):
        timestamps = pd.date_range(
            "2023-01-01", periods=len(self._closes), freq="1D", tz="UTC",
        ).astype("int64") // 10**6
        rows = [
            [int(ts), c, c, c, c, 1.0]
            for ts, c in zip(timestamps, self._closes)
        ]
        return rows[-limit:]

    def load_markets(self, reload=False):
        return {
            "BTC/USDT": {
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10.0}},
                "precision": {"amount": 8},
            },
            "ETH/USDT": {
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 10.0}},
                "precision": {"amount": 8},
            },
        }


def test_dry_run_emits_buys_when_fully_long_signal_and_no_holdings():
    """Clean uptrend → signal=1.0 → buy each asset at 1/N × equity."""
    exch = _StubExchange(balance_usdt=10_000.0, btc_holdings=0.0)
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert isinstance(result, DryRunResult)
    assert result.signal == 1.0
    assert result.sma_gate_open is True

    # 2-asset basket × $10k equity → $5k target each.
    sides = {o["symbol"]: o["side"] for o in result.orders_planned}
    assert sides == {"BTC/USDT": "buy", "ETH/USDT": "buy"}
    btc_order = next(o for o in result.orders_planned if o["symbol"] == "BTC/USDT")
    assert btc_order["notional_quote"] == pytest.approx(5_000.0, rel=0.001)


def test_dry_run_does_not_call_create_order():
    """Defensive: confirm the stub's create_order is never invoked.

    The stub doesn't define create_order at all; if the dry-run pipeline
    ever calls it, AttributeError surfaces immediately. The fact that
    this test runs cleanly without that attribute IS the assertion."""
    exch = _StubExchange()
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert isinstance(result, DryRunResult)


def test_dry_run_skips_tiny_delta_below_min_cost():
    """current holdings ~= target → sub-$10 delta should be SKIPPED,
    not sent. The result records it under orders_skipped."""
    # Math: equity = balance_usdt + btc_qty × $50k. For a 2-asset
    # basket at signal=1.0, target per asset = equity / 2. We want
    # BTC delta to be ~$5 below min_cost=$10. Choose
    # btc_holdings=0.02 (= $1000), balance_usdt=$1010 → equity=$2010,
    # target_btc_value=$1005, current=$1000, delta=$5 ⇒ below min_cost.
    exch = _StubExchange(balance_usdt=1_010.0, btc_holdings=0.02)
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    btc_orders = [o for o in result.orders_planned if o["symbol"] == "BTC/USDT"]
    btc_skipped = [s for s in result.orders_skipped if s["symbol"] == "BTC/USDT"]
    # The BTC delta should be in skipped, not in orders.
    assert btc_orders == [], f"BTC delta should have been skipped, got {btc_orders}"
    assert len(btc_skipped) == 1
    assert result.total_skipped_quote_drift > 0


def test_dry_run_with_signal_zero_plans_full_sell():
    """Build a downtrend → signal=0 → if BTC is held, plan a full sell."""
    exch = _StubExchange(balance_usdt=0.0, btc_holdings=0.1)
    # Override candles with a clean downtrend.
    exch._closes = np.linspace(200, 100, 500).tolist()
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert result.signal == 0.0
    # BTC must be sold (target qty is 0; current is 0.1).
    btc_order = next(
        o for o in result.orders_planned if o["symbol"] == "BTC/USDT"
    )
    assert btc_order["side"] == "sell"
    assert btc_order["base_amount"] == pytest.approx(0.1)


def test_dry_run_returns_structured_data_for_logging():
    """All result fields populated — useful for the JSON
    reconciliation logger in step #2b."""
    exch = _StubExchange()
    broker = Broker(_config(), exch)
    r = run_dry_cycle(broker, candles_per_asset=400)
    assert r.asof is not None
    assert isinstance(r.signal, float)
    assert isinstance(r.total_equity, float)
    assert isinstance(r.target_allocation, dict)
    assert isinstance(r.current_holdings_quote, dict)
    assert isinstance(r.orders_planned, list)
    assert isinstance(r.orders_skipped, list)
    assert isinstance(r.total_skipped_quote_drift, float)


def test_nan_weight_fails_loud_through_dry_cycle(tmp_path, monkeypatch):
    """Mirror of the live-cycle fail-loud guard on the dry-run path: a NaN
    in basket_weights raises and journals outcome='failed' rather than
    silently mis-sizing the printed plan."""
    import json
    import math

    from trade_lab.execution import dry_run as dr
    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import SignalSnapshot

    bad_snap = SignalSnapshot(
        asof=pd.Timestamp("2026-06-11", tz="UTC"), signal=1.0,
        basket_close=150.0, asset_closes={"BTC": 50_000.0, "ETH": 3_000.0},
        sma_gate_open=True, n_assets_in_basket=2,
        basket_weights={"BTC": math.nan, "ETH": 0.5},
    )
    monkeypatch.setattr(dr, "compute_live_signal", lambda *a, **k: bad_snap)

    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    with pytest.raises(ValueError, match="BTC"):
        run_dry_cycle(broker, journal=journal)

    lines = (tmp_path / "cycles.jsonl").read_text().splitlines()
    cycle = json.loads(lines[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "ValueError"


def test_dry_run_short_history_journals_failed_cycle_and_raises(tmp_path):
    """Binance-testnet shape: the exchange wiped candles and returns only
    ~36 daily bars, so SMA(200) can never warm. The cycle must NOT
    'succeed' with signal=0 (that plans a full liquidation of any open
    book); it raises SignalComputationError AND journals a structured
    outcome='failed' entry so monitoring surfaces the incident."""
    import json

    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import SignalComputationError

    exch = _StubExchange(balance_usdt=0.0, btc_holdings=0.1)
    exch._closes = (100 + np.linspace(0, 20, 36)).tolist()  # uptrend, 36 bars
    broker = Broker(_config(), exch)
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(SignalComputationError, match="36 completed bars"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "SignalComputationError"
    assert "36 completed bars" in cycle["error"]["message"]
    # No plan was produced — nothing that could be mistaken for orders.
    assert cycle["orders_planned"] is None


def test_dry_run_records_exchange_latency_in_journal(tmp_path):
    """A successful dry cycle stamps context.exchange_latency — read-only
    telemetry the /metrics exporter surfaces. Metadata only."""
    import json

    from trade_lab.execution.journal import JournalWriter

    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    lat = cycle["context"]["exchange_latency"]
    assert lat["count"] > 0  # fetch_balance / ticker / ohlcv / markets were timed
    assert lat["errors"] == 0
    assert set(lat) >= {"count", "errors", "max_ms", "p95_ms", "by_endpoint"}
