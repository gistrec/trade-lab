"""Signal warm-up guards for the ``paper-dry-run`` CLI.

The dry-run is the mainnet read-only observation vehicle, so its
failure surface matters operationally: a ``--candles`` window that
cannot warm SMA(200) is refused up front, and a
``SignalComputationError`` from the cycle exits non-zero with a
one-line structured message — the journal entry is written inside
``run_dry_cycle`` before the re-raise, so monitoring still sees the
incident. One deliberate exception: ``InsufficientWarmupError`` on a
*sandbox* config (Binance testnet's ~monthly candle wipes leave ~36
bars — structural, unfixable by waiting) is a healthy first-class skip
and exits 0; on mainnet the same error keeps the non-zero posture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from trade_lab.cli import cmd_paper_dry_run
from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig


def _sandbox_config() -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
    )


def _mainnet_config() -> PaperConfig:
    return PaperConfig(
        exchange_id="binance", sandbox=False, api_key="k", api_secret="s",
        allow_mainnet=True,
        quote_currency="USDT",
        basket=("BTC", "ETH"),
        request_timeout_ms=5000,
    )


def _args(tmp_path: Path, candles: int = 400) -> argparse.Namespace:
    return argparse.Namespace(
        journal=str(tmp_path / "cycles.jsonl"),
        candles=candles,
    )


def test_refuses_candles_window_below_signal_warmup(monkeypatch, tmp_path):
    """--candles 150 < 201 (SMA(200)/lookback warm-up + the dropped
    in-progress candle) must exit before the broker is constructed."""
    connect_calls: list = []
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect",
        classmethod(lambda cls, config: connect_calls.append(config)),
    )

    with pytest.raises(SystemExit, match=r"--candles 150 is below"):
        cmd_paper_dry_run(_args(tmp_path, candles=150))

    assert connect_calls == []


def test_accepts_candles_at_exact_minimum(monkeypatch, tmp_path):
    """--candles 201 passes the up-front window check (the runtime basket
    depth guard inside compute_live_signal still applies)."""
    from trade_lab.execution.dry_run import DryRunResult

    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )
    result = DryRunResult(
        asof=None, signal=1.0, sma_gate_open=True, total_equity=0.0,
        target_allocation={}, current_holdings_quote={},
        orders_planned=[], orders_skipped=[], total_skipped_quote_drift=0.0,
    )
    monkeypatch.setattr(
        "trade_lab.execution.run_dry_cycle",
        lambda broker, **kwargs: result,
    )

    assert cmd_paper_dry_run(_args(tmp_path, candles=201)) is None


def test_signal_computation_error_exits_structured_nonzero(
    monkeypatch, tmp_path,
):
    """A non-warm-up SignalComputationError (uneven history, empty
    candles, fetch failure) from run_dry_cycle → one-line SystemExit,
    non-zero, no raw traceback — on ANY environment, testnet included.
    Only the structured InsufficientWarmupError gets the exit-0
    treatment, and only on a sandbox config."""
    from trade_lab.execution.signal import SignalComputationError

    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )

    def _raise(broker, **kwargs):
        raise SignalComputationError(
            "Uneven basket history: DOGE has 150 bars starting 2025-08-05"
        )

    monkeypatch.setattr("trade_lab.execution.run_dry_cycle", _raise)

    with pytest.raises(
        SystemExit, match=r"Signal computation failed: .*Uneven basket",
    ) as exc_info:
        cmd_paper_dry_run(_args(tmp_path))
    assert exc_info.value.code not in (0, None)


def _raise_warmup(broker, **kwargs):
    from trade_lab.execution.signal import InsufficientWarmupError

    raise InsufficientWarmupError(
        "Basket history too short to warm the signal: 36 completed "
        "bars, need >= 200",
        bars_available=36, bars_required=200,
    )


def test_insufficient_warmup_on_testnet_exits_zero(
    monkeypatch, tmp_path, capsys,
):
    """Testnet's structurally-unwarmable SMA is a healthy first-class
    skip: the command prints an explanatory line (with the bar counts)
    and returns normally — exit 0, so the cron alert is not permanently
    red for a state that can never change on this venue."""
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _sandbox_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )
    monkeypatch.setattr("trade_lab.execution.run_dry_cycle", _raise_warmup)

    assert cmd_paper_dry_run(_args(tmp_path)) is None  # no SystemExit
    out = capsys.readouterr().out
    assert "SKIPPED" in out
    assert "36" in out and "200" in out
    assert "skipped_warmup" in out


def test_insufficient_warmup_on_mainnet_exits_nonzero(monkeypatch, tmp_path):
    """PIN of mainnet strictness (H3): the same InsufficientWarmupError on
    a sandbox=False config means truncated kline history on the
    real-money path — one-line SystemExit, non-zero, no exit-0 softening.
    This test exists to break any future dilution."""
    monkeypatch.setattr(
        "trade_lab.execution.load_paper_config", _mainnet_config,
    )
    monkeypatch.setattr(
        Broker, "connect", classmethod(lambda cls, config: object()),
    )
    monkeypatch.setattr("trade_lab.execution.run_dry_cycle", _raise_warmup)

    with pytest.raises(
        SystemExit, match=r"Signal computation failed: .*36 completed bars",
    ) as exc_info:
        cmd_paper_dry_run(_args(tmp_path))
    assert exc_info.value.code not in (0, None)
