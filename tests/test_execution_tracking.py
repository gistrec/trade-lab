"""Execution-tracking layer (issue #11): real mainnet vs simulation."""
from __future__ import annotations

import json

import pytest

from trade_lab.paper_trading.execution_tracking import (
    DEFAULT_GAP_THRESHOLD_PCT,
    check_execution_tracking,
)
from trade_lab.paper_trading.execution_tracking_cli import main as tracking_cli_main
from trade_lab.paper_trading.journal import HarnessLogRow, append_row


# ---------------------------------------------------------------------------
# Synthetic journal builders (no network, no exchange)
# ---------------------------------------------------------------------------

def _sim_row(date_iso: str, equity: float, ladder: float = 0.5) -> HarnessLogRow:
    return HarnessLogRow(
        date=date_iso, config_hash="x" * 64, vintage_content_hash="y" * 64,
        basket_close=100.0, sma_value=99.0, sma_gate_open=True,
        ladder_state=ladder, prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.01, "60": 0.01},
        target_weights={"BTC": ladder / 7},
        current_weights={"BTC": 0.0},
        intended_trades={"BTC": ladder / 7},
        portfolio_equity=equity,
        daily_return=0.0, gross_position_return=0.0, net_position_return=0.0,
    )


def _cycle(
    date_iso: str,
    equity: float,
    ladder: float,
    *,
    outcome: str = "success",
    orders_executed=None,
    orders_skipped=None,
    cycle_id: str = "aabbccdd-0000",
) -> dict:
    return {
        "schema_version": 2,
        "cycle_id": cycle_id,
        "started_at": f"{date_iso}T06:05:00+00:00",
        "ended_at": f"{date_iso}T06:05:30+00:00",
        "duration_ms": 30000,
        "outcome": outcome,
        "error": None,
        "context": {"exchange": "binance", "sandbox": False, "mode": "live"},
        "signal": {
            "asof": f"{date_iso}T00:00:00+00:00",
            "ladder_value": ladder,
            "sma_gate_open": True,
        },
        "equity_usd": equity,
        "orders_executed": orders_executed,
        "orders_skipped": orders_skipped,
        "total_skipped_quote_drift": 0.0,
    }


def _filled_order(symbol: str = "BTC/USDT", side: str = "buy") -> dict:
    return {
        "client_order_id": "tsmom_x",
        "symbol": symbol,
        "side": side,
        "terminal_status": "closed",
        "filled_amount": 0.001,
        "filled_notional_quote": 25.0,
    }


def _min_notional_skip(symbol: str = "DOGE/USDT") -> dict:
    return {
        "symbol": symbol,
        "desired_side": "buy",
        "desired_amount": 10.0,
        "desired_notional": 3.0,
        "reason": "notional 3.0000 < min_cost 5",
    }


def _write_real(path, cycles: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cycles:
            f.write(json.dumps(c) + "\n")


def _write_sim(path, rows: list[HarnessLogRow]) -> None:
    for r in rows:
        append_row(r, path)


def _cli(real_p, sim_p, *extra) -> list[str]:
    return ["--real-journal", str(real_p), "--sim-journal", str(sim_p), *extra]


DATES = ["2026-08-01", "2026-08-02", "2026-08-03"]


# ---------------------------------------------------------------------------
# Equity tracking
# ---------------------------------------------------------------------------

def test_equity_tracking_below_threshold_exit_0(tmp_path):
    """Identical daily returns → zero gap → no breach even with the flag."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5) for d, eq in zip(DATES, [100.0, 101.0, 102.0])
    ])
    _write_sim(sim_p, [
        _sim_row(d, eq) for d, eq in zip(DATES, [10_000.0, 10_100.0, 10_200.0])
    ])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 3
    assert report.equity.cum_abs_return_diff == pytest.approx(0.0, abs=1e-12)
    assert report.equity.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not report.equity.breached
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0


def test_equity_tracking_above_threshold_exit_1_with_flag(tmp_path, capsys):
    """Real drops 10% while sim stays flat → gap −10% > default 5%."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5) for d, eq in zip(DATES, [100.0, 100.0, 90.0])
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.level_gap_pct == pytest.approx(-10.0)
    assert report.equity.cum_abs_return_diff == pytest.approx(0.10)
    assert report.equity.breached
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 1
    assert "BREACH" in capsys.readouterr().out


def test_equity_breach_without_flag_stays_exit_0(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5) for d, eq in zip(DATES, [100.0, 100.0, 90.0])
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "BREACH" in capsys.readouterr().out


def test_equity_breach_json_payload(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5) for d, eq in zip(DATES, [100.0, 100.0, 90.0])
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    rc = tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach", "--json"))
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["equity"]["breached"] is True
    assert payload["equity"]["level_gap_pct"] == pytest.approx(-10.0)


def test_custom_threshold_widens_the_band(tmp_path):
    """−10% gap is inside a 15% threshold — owner-adjustable by design."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5) for d, eq in zip(DATES, [100.0, 100.0, 90.0])
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    rc = tracking_cli_main(_cli(
        real_p, sim_p, "--fail-on-breach", "--gap-threshold-pct", "15",
    ))
    assert rc == 0
    assert DEFAULT_GAP_THRESHOLD_PCT == 5.0  # documented starting point


def test_last_cycle_of_the_day_wins(tmp_path):
    """6-hourly dry-runs re-observe the same signal date; the latest
    equity reading is the one compared."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[0], 105.0, 0.5),   # later observation, same asof date
        _cycle(DATES[1], 105.0, 0.5),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    report = check_execution_tracking(real_p, sim_p)
    # 105 → 105: flat, matching the flat sim; 100 → 105 would be +5%.
    assert report.equity.cum_abs_return_diff == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Position-transition check
# ---------------------------------------------------------------------------

def test_real_order_without_ladder_transition_flagged(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 0.5, orders_executed=[_filled_order()]),
        _cycle(DATES[2], 100.0, 0.5),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.n_ladder_transitions == 0
    assert tr.n_real_order_events == 1
    assert len(tr.orders_without_transition) == 1
    assert tr.orders_without_transition[0]["symbol"] == "BTC/USDT"
    assert "mismatch" in report.advisory.lower()
    # Descriptive only: mismatches never drive the exit code.
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    assert "ORDER WITHOUT TRANSITION" in capsys.readouterr().out


def test_transition_with_matching_order_not_flagged(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[_filled_order()]),
        _cycle(DATES[2], 100.0, 1.0),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_ladder_transitions == 1
    assert tr.orders_without_transition == []
    assert tr.transitions_without_order == []


def test_transition_without_order_flagged(tmp_path):
    """Ladder moved but nothing was placed and nothing was journaled as
    skipped → real execution silently missed a transition."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 1.0),
        _cycle(DATES[2], 100.0, 1.0),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.transitions_without_order == [
        {"date": DATES[1], "prior": 0.5, "new": 1.0}
    ]


def test_min_notional_skip_covers_transition_and_is_not_flagged(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 1.0,
               orders_executed=[], orders_skipped=[_min_notional_skip()]),
        _cycle(DATES[2], 100.0, 1.0),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.transitions_without_order == []
    assert tr.n_transitions_skip_covered == 1
    assert tr.n_min_notional_skips == 1
    assert "mismatch" not in report.advisory.lower()
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0


def test_repeated_skips_on_one_date_count_once(tmp_path):
    """The 6-hourly dry-run re-records the same skip — distinct
    (date, symbol) must not inflate with observation time."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_skipped=[_min_notional_skip()]),
        _cycle(DATES[0], 100.0, 0.5, orders_skipped=[_min_notional_skip()]),
    ])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_min_notional_skips == 1


# ---------------------------------------------------------------------------
# Exit-code contract (mirrors fingerprint_cli, incl. the #62 OSError fix)
# ---------------------------------------------------------------------------

def test_missing_real_journal_exit_2(tmp_path, capsys):
    sim_p = tmp_path / "sim.jsonl"
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    rc = tracking_cli_main(_cli(tmp_path / "missing.jsonl", sim_p))
    assert rc == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


def test_missing_sim_journal_exit_2(tmp_path, capsys):
    real_p = tmp_path / "real.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5)])
    rc = tracking_cli_main(_cli(real_p, tmp_path / "missing.jsonl"))
    assert rc == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


def test_real_journal_is_directory_exit_2_not_1(tmp_path, capsys):
    """IsADirectoryError (OSError) must be a tool error (2), not
    indistinguishable from a --fail-on-breach exit 1."""
    sim_p = tmp_path / "sim.jsonl"
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    real_dir = tmp_path / "realdir"
    real_dir.mkdir()
    rc = tracking_cli_main(_cli(real_dir, sim_p, "--fail-on-breach"))
    assert rc == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


def test_empty_overlap_is_descriptive_exit_0(tmp_path, capsys):
    """Both journals exist but share no dates → note, exit 0, no breach."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle("2026-07-01", 100.0, 0.5)])
    _write_sim(sim_p, [_sim_row("2026-08-01", 10_000.0)])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 0
    assert not report.equity.breached
    assert "no aligned dates" in report.advisory.lower()
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    assert "No aligned dates" in capsys.readouterr().out


def test_non_positive_equity_on_aligned_date_exit_2(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 0.0, 0.5)])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    rc = tracking_cli_main(_cli(real_p, sim_p))
    assert rc == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_threshold_rejected_at_parse_time(tmp_path, capsys, value):
    """Threshold <= 0 makes the breach check vacuous — argparse must
    reject with its native exit 2."""
    with pytest.raises(SystemExit) as excinfo:
        tracking_cli_main(_cli(
            tmp_path / "r.jsonl", tmp_path / "s.jsonl",
            "--gap-threshold-pct", value,
        ))
    assert excinfo.value.code == 2
    assert "must be > 0" in capsys.readouterr().err


def test_failed_cycles_do_not_enter_equity_series(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    failed = _cycle(DATES[1], 50.0, 0.5, outcome="failed")
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5), failed, _cycle(DATES[2], 100.0, 0.5),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    # Failed cycle's equity is excluded: only 2 aligned days, flat → no gap.
    assert report.equity.n_aligned_days == 2
    assert report.equity.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not report.equity.breached
