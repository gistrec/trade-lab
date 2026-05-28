"""Tests for the ``paper-place-test-order`` smoke-test CLI.

Coverage focus:

* Mainnet refusal — ``sandbox=false`` exits with a clear message.
* Sub-minimum preflight skips placement before the exchange is touched.
* ``smoke_`` clientOrderId namespace separates smoke tests from
  production ``tsmom_`` orders.
* Happy path produces an ``OrderResult`` with the expected shape and
  ``--journal`` writes one JSON line of the agreed schema.

Tested via direct call to ``cmd_paper_place_test_order`` with
``argparse.Namespace`` and monkey-patched ``load_paper_config`` and
``Broker.connect`` — no subprocess overhead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ccxt
import pytest

from trade_lab.cli import cmd_paper_place_test_order
from trade_lab.execution.broker import Broker, MarketConstraints
from trade_lab.execution.config import PaperConfig


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _config(sandbox: bool = True) -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=sandbox, api_key="k", api_secret="s",
        allow_mainnet=False if sandbox else True,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
    )


class _FakeExchange:
    """Records create_order calls for assertions."""

    id = "binance"

    def __init__(
        self,
        price: float = 50_000.0,
        min_cost: float | None = 10.0,
        min_amount: float | None = 0.0001,
        create_response: dict | None = None,
        create_raises: Exception | None = None,
        fetch_terminal: dict | None = None,
    ) -> None:
        self.price = price
        self.min_cost = min_cost
        self.min_amount = min_amount
        self.create_response = create_response or {
            "id": "exch-1", "status": "open", "filled": 0.0,
            "cost": 0.0, "average": None, "fee": {"cost": 0.0}, "timestamp": 0,
        }
        self.create_raises = create_raises
        self.fetch_terminal = fetch_terminal or {
            "id": "exch-1", "status": "closed", "filled": 0.0004,
            "cost": 20.0, "average": 50_000.0,
            "fee": {"cost": 0.02, "currency": "USDT"}, "timestamp": 0,
        }
        self.create_order_calls: list[dict] = []
        self.fetch_order_calls: list[dict] = []
        self._first_fetch = True

    def set_sandbox_mode(self, enabled): pass
    def fetch_balance(self): return {"USDT": {"free": 1000, "used": 0, "total": 1000}}

    def fetch_ticker(self, symbol):
        return {"last": self.price, "close": self.price}

    def fetch_status(self): return {"status": "ok"}

    def load_markets(self, reload=False):
        return {
            "BTC/USDT": {
                "limits": {
                    "amount": {"min": self.min_amount},
                    "cost": {"min": self.min_cost},
                },
                "precision": {"amount": 8},
            },
        }

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.create_order_calls.append({
            "symbol": symbol, "side": side, "amount": amount, "params": params,
        })
        if self.create_raises is not None:
            raise self.create_raises
        return dict(self.create_response)

    def fetch_order(self, id, symbol=None, params=None):
        self.fetch_order_calls.append({"id": id, "params": params})
        # First call is the query-before-place — return OrderNotFound so
        # the CLI proceeds to create. Subsequent calls return the
        # terminal state.
        if self._first_fetch:
            self._first_fetch = False
            raise ccxt.OrderNotFound("not yet")
        return dict(self.fetch_terminal)

    def fetch_open_orders(self, symbol=None): return []
    def fetch_my_trades(self, symbol=None, since=None, limit=None): return []


def _patch_config_and_broker(
    monkeypatch, *, config: PaperConfig, exchange: _FakeExchange,
):
    """Replace load_paper_config and Broker.connect on the CLI module."""
    import trade_lab.cli

    def fake_load_paper_config():
        return config

    def fake_connect(cls, cfg):
        return Broker(cfg, exchange)

    # The CLI imports these inside the function via `from .execution import ...`
    # so we patch the source modules.
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", fake_load_paper_config,
    )
    monkeypatch.setattr(Broker, "connect", classmethod(fake_connect))


def _args(
    tmp_path: Path,
    *,
    symbol: str = "BTC/USDT",
    side: str = "buy",
    notional: float = 20.0,
    journal: str | None = None,
    state: str | None = None,
    timeout_s: float = 5.0,
) -> argparse.Namespace:
    return argparse.Namespace(
        symbol=symbol,
        side=side,
        notional=notional,
        state=state if state is not None else str(tmp_path / "state.json"),
        journal=journal,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# Mainnet refusal
# ---------------------------------------------------------------------------


def test_refuses_mainnet(monkeypatch, tmp_path):
    _patch_config_and_broker(
        monkeypatch,
        config=_config(sandbox=False),
        exchange=_FakeExchange(),
    )
    with pytest.raises(SystemExit, match="mainnet"):
        cmd_paper_place_test_order(_args(tmp_path))


# ---------------------------------------------------------------------------
# Sub-minimum preflight
# ---------------------------------------------------------------------------


def test_sub_min_cost_skips_placement(monkeypatch, tmp_path):
    exch = _FakeExchange(min_cost=10.0)
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    with pytest.raises(SystemExit, match="SKIPPED.*min_cost"):
        cmd_paper_place_test_order(_args(tmp_path, notional=5.0))
    # Exchange was NEVER asked to create the order.
    assert exch.create_order_calls == []


def test_sub_min_amount_skips_placement(monkeypatch, tmp_path):
    """Notional clears min_cost but amount is below min_amount."""
    exch = _FakeExchange(
        price=1_000_000.0,    # very high → amount tiny for same notional
        min_cost=10.0,
        min_amount=0.001,
    )
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    with pytest.raises(SystemExit, match="SKIPPED.*min_amount"):
        cmd_paper_place_test_order(_args(tmp_path, notional=20.0))
    assert exch.create_order_calls == []


def test_network_error_during_constraints_propagates(monkeypatch, tmp_path):
    """Without constraints the sub-min preflight is impossible. Proceeding
    blind is a worse failure mode than aborting loudly — the network
    error propagates so the operator sees the actual problem."""
    exch = _FakeExchange()

    def boom(reload=False):
        raise ccxt.NetworkError("markets temporarily unavailable")
    monkeypatch.setattr(exch, "load_markets", boom)

    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    with pytest.raises(ccxt.NetworkError):
        cmd_paper_place_test_order(_args(tmp_path))
    assert exch.create_order_calls == []


def test_unknown_symbol_exits_cleanly(monkeypatch, tmp_path):
    """fetch_market_constraints raises BrokerError on unknown symbol;
    the CLI converts that to a SystemExit with a clear message."""
    exch = _FakeExchange()

    def empty_markets(reload=False):
        return {}  # symbol not present
    monkeypatch.setattr(exch, "load_markets", empty_markets)

    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    with pytest.raises(SystemExit, match="market constraints"):
        cmd_paper_place_test_order(_args(tmp_path, symbol="XYZ/USDT"))
    assert exch.create_order_calls == []


# ---------------------------------------------------------------------------
# Happy path + clientOrderId namespace
# ---------------------------------------------------------------------------


def test_calls_create_order_with_smoke_coid(monkeypatch, tmp_path):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    cmd_paper_place_test_order(_args(tmp_path))

    assert len(exch.create_order_calls) == 1
    call = exch.create_order_calls[0]
    assert call["symbol"] == "BTC/USDT"
    assert call["side"] == "buy"
    # base_amount = notional / price = 20 / 50000
    assert abs(call["amount"] - (20.0 / 50_000.0)) < 1e-12
    coid = call["params"]["newClientOrderId"]
    assert coid.startswith("smoke_")
    # NEVER tsmom_ — production namespace must stay clean.
    assert not coid.startswith("tsmom_")
    assert "BTCUSDT" in coid
    assert coid.endswith("_buy")


def test_smoke_coid_persisted_to_state(monkeypatch, tmp_path):
    """After a successful placement, the smoke-prefixed entry is in
    the state file. Same-day re-run will be idempotent off this."""
    from trade_lab.execution.order_state import OrderStateStore

    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    state_path = tmp_path / "state.json"
    cmd_paper_place_test_order(_args(tmp_path, state=str(state_path)))

    store = OrderStateStore(state_path)
    keys = list(store.all_entries().keys())
    assert len(keys) == 1
    assert keys[0].startswith("smoke_")


# ---------------------------------------------------------------------------
# --journal smoke-test log
# ---------------------------------------------------------------------------


def test_journal_writes_jsonl_record(monkeypatch, tmp_path):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    journal_path = tmp_path / "smoke_log.jsonl"
    cmd_paper_place_test_order(_args(tmp_path, journal=str(journal_path)))

    assert journal_path.exists()
    lines = journal_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "smoke_test"
    assert record["exchange"] == "binance"
    assert record["sandbox"] is True
    assert "asof" in record
    assert record["result"]["client_order_id"].startswith("smoke_")
    assert record["result"]["terminal_status"] in ("closed", "partial")


def test_no_journal_no_file(monkeypatch, tmp_path):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    cmd_paper_place_test_order(_args(tmp_path, journal=None))
    # state file exists but no smoke log file was created anywhere.
    assert not (tmp_path / "smoke_log.jsonl").exists()


def test_journal_creates_parent_directory(monkeypatch, tmp_path):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    nested = tmp_path / "deep" / "nested" / "smoke.jsonl"
    cmd_paper_place_test_order(_args(tmp_path, journal=str(nested)))
    assert nested.exists()


# ---------------------------------------------------------------------------
# Side argument
# ---------------------------------------------------------------------------


def test_sell_side_passed_through(monkeypatch, tmp_path):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    cmd_paper_place_test_order(_args(tmp_path, side="sell"))
    assert exch.create_order_calls[0]["side"] == "sell"
    coid = exch.create_order_calls[0]["params"]["newClientOrderId"]
    assert coid.endswith("_sell")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_prints_human_readable_result(monkeypatch, tmp_path, capsys):
    exch = _FakeExchange()
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    cmd_paper_place_test_order(_args(tmp_path))
    out = capsys.readouterr().out
    assert "Smoke test:" in out
    assert "client_order_id" in out
    assert "ticker price" in out
    assert "Result: terminal_status=" in out
    assert "exchange_order_id" in out
    assert "filled:" in out


def test_prints_error_on_rejection(monkeypatch, tmp_path, capsys):
    exch = _FakeExchange(
        create_raises=ccxt.InvalidOrder("min notional 10 USDT not met"),
    )
    _patch_config_and_broker(monkeypatch, config=_config(), exchange=exch)
    cmd_paper_place_test_order(_args(tmp_path))
    out = capsys.readouterr().out
    assert "terminal_status=rejected" in out
    assert "InvalidOrder" in out
