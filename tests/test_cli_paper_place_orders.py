"""Mainnet refusal for the production ``paper-place-orders`` CLI.

CLAUDE.md hard rule: mainnet order placement is unsupported even when
the two-flag gate (SANDBOX=false + ALLOW_MAINNET=true) is satisfied.
The command runs under cron, so a printed warning protects nobody —
it must exit before the broker is even constructed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from trade_lab.cli import cmd_paper_place_orders
from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig


def _mainnet_config() -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=False, api_key="k", api_secret="s",
        allow_mainnet=True,
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


def test_refuses_mainnet_even_with_both_flags(monkeypatch, tmp_path):
    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _mainnet_config,
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    with pytest.raises(SystemExit, match="mainnet"):
        cmd_paper_place_orders(_args(tmp_path))

    assert connect_calls == [], "broker must never be constructed on mainnet"


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
