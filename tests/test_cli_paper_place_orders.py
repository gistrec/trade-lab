"""Mainnet gating for the production ``paper-place-orders`` CLI.

CLAUDE.md hard rule: mainnet order placement requires THREE explicit
flags — SANDBOX=false + ALLOW_MAINNET=true (read paths) +
MAINNET_LIVE_ORDERS=true (this command). The command runs under cron,
so a printed warning protects nobody — a missing flag must exit before
the broker is even constructed. Journal and state files are
environment-checked so testnet and mainnet can never share files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from trade_lab.cli import cmd_paper_place_orders
from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig


def _mainnet_config(mainnet_live_orders: bool = False) -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=False, api_key="k", api_secret="s",
        allow_mainnet=True,
        mainnet_live_orders=mainnet_live_orders,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        journal=str(tmp_path / "cycles.jsonl"),
        state=str(tmp_path / "orders.json"),
        candles=400,
        timeout_s=5.0,
    )


def test_refuses_mainnet_without_live_orders_flag(monkeypatch, tmp_path):
    """Two flags are NOT enough: without MAINNET_LIVE_ORDERS=true the
    command must exit before the broker is constructed."""
    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _mainnet_config,
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    with pytest.raises(SystemExit, match="MAINNET_LIVE_ORDERS"):
        cmd_paper_place_orders(_args(tmp_path))

    assert connect_calls == [], (
        "broker must never be constructed without the third flag"
    )


def test_mainnet_runs_with_three_flags(monkeypatch, tmp_path):
    """The full three-flag config reaches the live cycle."""
    from trade_lab.execution.live_cycle import LiveCycleResult

    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config",
        lambda: _mainnet_config(mainnet_live_orders=True),
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(
            lambda cls, config: connect_calls.append(config) or object()
        ),
    )
    result = LiveCycleResult(
        cycle_id="0" * 32, outcome="success", order_results=[],
        reconstructed_count=0, error=None,
    )
    monkeypatch.setattr(
        "trade_lab.execution.run_live_cycle",
        lambda broker, **kwargs: result,
    )

    assert cmd_paper_place_orders(_args(tmp_path)) is None  # no SystemExit
    assert len(connect_calls) == 1


def test_mainnet_refuses_testnet_journal(monkeypatch, tmp_path):
    """A journal already holding testnet cycles must never accept
    mainnet cycles — exit before any exchange call."""
    journal = tmp_path / "cycles.jsonl"
    testnet_cycle = {
        "cycle_id": "x", "outcome": "success",
        "context": {"mode": "live", "exchange": "binance", "sandbox": True},
    }
    journal.write_text(json.dumps(testnet_cycle) + "\n")

    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config",
        lambda: _mainnet_config(mainnet_live_orders=True),
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    with pytest.raises(SystemExit, match="never share a journal"):
        cmd_paper_place_orders(_args(tmp_path))

    assert connect_calls == []


def test_mainnet_refuses_unstamped_state_file(monkeypatch, tmp_path):
    """A pre-existing unstamped state file is presumed testnet — a
    mainnet run must not adopt it (clientOrderIds would collide)."""
    state = tmp_path / "orders.json"
    state.write_text(json.dumps({
        "tsmom_20260709_BTCUSDT_buy": {
            "client_order_id": "tsmom_20260709_BTCUSDT_buy",
            "symbol": "BTC/USDT", "side": "buy", "intended_amount": 1.0,
            "status": "closed", "exchange_order_id": "1",
            "placed_at": "2026-07-09T00:05:00+00:00",
            "last_seen_at": "2026-07-09T00:05:10+00:00",
        },
    }))

    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config",
        lambda: _mainnet_config(mainnet_live_orders=True),
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    with pytest.raises(SystemExit, match="environment stamp"):
        cmd_paper_place_orders(_args(tmp_path))

    assert connect_calls == []


# ---------------------------------------------------------------------------
# Exit codes — cron alerts on anything but a clean success
# ---------------------------------------------------------------------------


def _sandbox_config() -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
    )


def _patch_pipeline(monkeypatch, outcome: str):
    from trade_lab.execution.live_cycle import LiveCycleResult

    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )
    result = LiveCycleResult(
        cycle_id="0" * 32, outcome=outcome, order_results=[],
        reconstructed_count=0, error=None,
    )
    monkeypatch.setattr(
        "trade_lab.execution.run_live_cycle",
        lambda broker, **kwargs: result,
    )


def test_success_outcome_exits_zero(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, outcome="success")
    assert cmd_paper_place_orders(_args(tmp_path)) is None  # no SystemExit


@pytest.mark.parametrize("outcome", ["unknown_orders", "partial"])
def test_bad_outcome_exits_nonzero(monkeypatch, tmp_path, outcome):
    """Cycles ending with stuck or partially-executed orders must exit
    non-zero so cron/alerting catches them — exit 0 hides the incident."""
    _patch_pipeline(monkeypatch, outcome=outcome)
    with pytest.raises(SystemExit) as exc_info:
        cmd_paper_place_orders(_args(tmp_path))
    assert exc_info.value.code not in (0, None)


def test_refuses_when_another_instance_holds_the_lock(
    monkeypatch, tmp_path, capsys,
):
    """cron 00:05 + a manual run (H2): check-then-act idempotency
    cannot stop a concurrent duplicate create_order, so the second
    process must refuse loudly — exit 3, distinct from refusals (1)
    and bad cycle outcomes (2) — BEFORE the broker is constructed."""
    from trade_lab.execution.instance_lock import acquire_instance_lock

    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    args = _args(tmp_path)
    lock = acquire_instance_lock(args.state)  # the "cron" run
    try:
        with pytest.raises(SystemExit) as exc_info:
            cmd_paper_place_orders(args)  # the "manual" run
    finally:
        lock.release()

    assert exc_info.value.code == 3
    assert "REFUSED" in capsys.readouterr().err
    assert connect_calls == [], (
        "the second instance must never even construct the broker"
    )


def test_lost_track_exits_nonzero_even_on_success(monkeypatch, tmp_path):
    """A lost_track surfaced by reconstruction must exit non-zero even
    when the MAIN cycle outcome is 'success' — cron alerting keys on the
    exit code, and a vanished order is an unresolved incident (regression:
    R1)."""
    from trade_lab.execution.live_cycle import LiveCycleResult

    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )
    result = LiveCycleResult(
        cycle_id="0" * 32, outcome="success", order_results=[],
        reconstructed_count=1, error=None, lost_track_count=1,
    )
    monkeypatch.setattr(
        "trade_lab.execution.run_live_cycle",
        lambda broker, **kwargs: result,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_paper_place_orders(_args(tmp_path))
    assert exc_info.value.code not in (0, None)
