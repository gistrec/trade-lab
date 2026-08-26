"""Execution-tracking layer (issue #11): real mainnet vs simulation."""
from __future__ import annotations

import dataclasses
import itertools
import json

import pytest

from trade_lab.paper_trading.execution_tracking import (
    DEFAULT_GAP_THRESHOLD_PCT,
    check_execution_tracking,
    live_min_notional_skips,
    real_equity_by_date,
    real_order_events,
    sim_pre_trade_equity_by_date,
)
from trade_lab.paper_trading.execution_tracking_cli import main as tracking_cli_main
from trade_lab.paper_trading.journal import HarnessLogRow, append_row


# ---------------------------------------------------------------------------
# Synthetic journal builders (no network, no exchange)
# ---------------------------------------------------------------------------

def _sim_row(
    date_iso: str,
    equity: float,
    ladder: float = 0.5,
    prior: float | None = None,
    intended: dict | None = None,
) -> HarnessLogRow:
    # Default: no transition (prior == ladder) and no intended trades, so
    # equity-only tests carry no trade expectations.
    prior = ladder if prior is None else prior
    intended = {} if intended is None else intended
    return HarnessLogRow(
        date=date_iso, config_hash="x" * 64, vintage_content_hash="y" * 64,
        basket_close=100.0, sma_value=99.0, sma_gate_open=True,
        ladder_state=ladder, prior_ladder_state=prior,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.01, "60": 0.01},
        target_weights={"BTC": ladder / 7},
        current_weights={"BTC": prior / 7},
        intended_trades=intended,
        portfolio_equity=equity,
        daily_return=0.0, gross_position_return=0.0, net_position_return=0.0,
    )


def _cycle(
    date_iso: str,
    equity: float,
    ladder: float,
    *,
    outcome: str = "success",
    mode: str = "live",
    signal: bool = True,
    orders_executed=None,
    orders_skipped=None,
    cycle_id: str = "aabbccdd-0000",
    ended_at: str | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "cycle_id": cycle_id,
        "started_at": f"{date_iso}T06:05:00+00:00",
        "ended_at": ended_at or f"{date_iso}T06:05:30+00:00",
        "duration_ms": 30000,
        "outcome": outcome,
        "error": None,
        "context": {"exchange": "binance", "sandbox": False, "mode": mode},
        "signal": {
            "asof": f"{date_iso}T00:00:00+00:00",
            "ladder_value": ladder,
            "sma_gate_open": True,
        } if signal else None,
        "equity_usd": equity,
        "orders_executed": orders_executed,
        "orders_skipped": orders_skipped,
        "total_skipped_quote_drift": 0.0,
    }


_COID_SEQ = itertools.count()


def _filled_order(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    *,
    coid: str | None = None,
    terminal_status: str = "closed",
    intended_amount: float = 0.001,
    filled_amount: float = 0.001,
) -> dict:
    return {
        # Distinct default ids: events dedupe by clientOrderId.
        "client_order_id": coid or f"tsmom_x_{next(_COID_SEQ)}",
        "symbol": symbol,
        "side": side,
        "terminal_status": terminal_status,
        "intended_amount": intended_amount,
        "filled_amount": filled_amount,
        "filled_notional_quote": 25.0,
    }


def _min_notional_skip(
    symbol: str = "DOGE/USDT",
    side: str = "buy",
    reason: str = "notional 3.0000 < min_cost 5",
) -> dict:
    return {
        "symbol": symbol,
        "desired_side": side,
        "desired_amount": 10.0,
        "desired_notional": 3.0,
        "reason": reason,
    }


def _write_real(path, cycles: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cycles:
            f.write(json.dumps(c) + "\n")


_SIM_COST_RATE = 0.001  # harness fee + slippage, charged on turnover


def _sim_row_phased(
    date_iso: str,
    pre_trade_equity: float,
    *,
    gross: float,
    ladder: float = 0.5,
    prior: float | None = None,
    intended: dict | None = None,
) -> HarnessLogRow:
    """A row shaped exactly as ``harness.py`` writes it: the journaled
    ``portfolio_equity`` is POST-trade and gross/net straddle the same
    turnover cost, so the pre-trade phase is recoverable from the row."""
    intended = {} if intended is None else intended
    cost_fraction = sum(abs(v) for v in intended.values()) * _SIM_COST_RATE
    row = _sim_row(
        date_iso, pre_trade_equity * (1.0 - cost_fraction),
        ladder=ladder, prior=prior, intended=intended,
    )
    return dataclasses.replace(
        row,
        gross_position_return=gross,
        net_position_return=gross - cost_fraction,
    )


def _write_sim(path, rows: list[HarnessLogRow]) -> None:
    for r in rows:
        append_row(r, path)


def _write_sim_raw(path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def _sim_line(row: HarnessLogRow) -> str:
    return json.dumps(dataclasses.asdict(row)) + "\n"


def _cli(real_p, sim_p, *extra) -> list[str]:
    return ["--real-journal", str(real_p), "--sim-journal", str(sim_p), *extra]


DATES = ["2026-08-01", "2026-08-02", "2026-08-03"]

BASKET = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE"]

# Sim intent for a 0.5 -> 1.0 ladder step on one asset.
BTC_BUY = {"BTC": 0.5 / 7}
BTC_SELL = {"BTC": -0.5 / 7}
# ...and the same step across the whole basket: turnover 0.5.
FULL_STEP_BUY = {a: 0.5 / 7 for a in BASKET}


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


# ---------------------------------------------------------------------------
# Trade-phase alignment of the two equity curves (Codex 3858504494)
# ---------------------------------------------------------------------------

def test_transition_date_equity_compared_at_same_trade_phase(tmp_path):
    """Execution replicates the simulation exactly and the transition
    falls on the LAST aligned date. Real ``equity_usd`` is read pre-trade
    while the harness journals equity post-cost, so comparing them raw
    charges the sim's turnover cost one observation early — a false
    level gap and a false return difference on the transition date."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    # +0.5% per day at ladder 0.5 on both curves, pre-trade phase.
    real_pre = [100.0, 100.5, 101.0025]
    sim_pre = [10_000.0, 10_050.0, 10_100.25]
    _write_real(real_p, [
        _cycle(DATES[0], real_pre[0], 0.5, orders_executed=[]),
        _cycle(DATES[1], real_pre[1], 0.5, orders_executed=[]),
        _cycle(DATES[2], real_pre[2], 1.0, orders_executed=[
            _filled_order(f"{a}/USDT", "buy") for a in BASKET
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row_phased(DATES[0], sim_pre[0], gross=0.0),
        _sim_row_phased(DATES[1], sim_pre[1], gross=0.005),
        _sim_row_phased(DATES[2], sim_pre[2], gross=0.005,
                        ladder=1.0, prior=0.5, intended=FULL_STEP_BUY),
    ])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 3
    assert report.equity.cum_abs_return_diff == pytest.approx(0.0, abs=1e-12)
    assert report.equity.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not report.equity.breached
    tr = report.transitions
    assert tr.missing_trades == []
    assert tr.unexpected_orders == []


def test_sim_equity_series_is_the_pre_trade_phase():
    """The series the comparison consumes backs the journaled turnover
    cost out, using committed fields only (gross − net)."""
    row = _sim_row_phased(
        DATES[1], 10_000.0, gross=0.005,
        ladder=1.0, prior=0.5, intended=FULL_STEP_BUY,
    )
    assert row.portfolio_equity == pytest.approx(10_000.0 * (1.0 - 0.0005))
    series = sim_pre_trade_equity_by_date([row])
    assert series[DATES[1]] == pytest.approx(10_000.0)


@pytest.mark.parametrize("gross,net", [
    (0.0, 0.5),   # net above gross → negative "cost"
    (1.0, 0.0),   # cost fraction 1.0 → division by zero
])
def test_impossible_cost_fraction_is_tool_error(tmp_path, capsys, gross, net):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5)])
    drifted = dataclasses.replace(
        _sim_row(DATES[0], 10_000.0),
        gross_position_return=gross, net_position_return=net,
    )
    _write_sim(sim_p, [drifted])
    with pytest.raises(ValueError, match="pre-trade equity phase"):
        check_execution_tracking(real_p, sim_p)
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Consistent per-date equity sampling (Codex 3858329138)
# ---------------------------------------------------------------------------

def test_live_cycle_wins_over_later_dry_run(tmp_path):
    """An 18:00 dry-run valuation must not replace the 00:05 live
    observation — a different market window."""
    real_p = tmp_path / "real.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, mode="live", orders_executed=[]),
        _cycle(DATES[0], 105.0, 0.5, mode="dry_run",
               ended_at=f"{DATES[0]}T18:00:30+00:00"),
    ])
    from trade_lab.monitoring.data_source import JournalReader
    cycles = JournalReader(real_p).cycles(n=100)
    assert real_equity_by_date(cycles) == {DATES[0]: 100.0}


def test_live_cycle_wins_over_earlier_dry_run(tmp_path):
    real_p = tmp_path / "real.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 105.0, 0.5, mode="dry_run"),
        _cycle(DATES[0], 100.0, 0.5, mode="live", orders_executed=[]),
        _cycle(DATES[0], 107.0, 0.5, mode="dry_run"),
    ])
    from trade_lab.monitoring.data_source import JournalReader
    cycles = JournalReader(real_p).cycles(n=100)
    assert real_equity_by_date(cycles) == {DATES[0]: 100.0}


def test_first_dry_run_wins_when_no_live_cycle(tmp_path):
    """Observation-only journal: the FIRST dry-run of the date is the
    sample; later 6-hourly re-observations never overwrite it."""
    real_p = tmp_path / "real.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, mode="dry_run"),
        _cycle(DATES[0], 105.0, 0.5, mode="dry_run"),
    ])
    from trade_lab.monitoring.data_source import JournalReader
    cycles = JournalReader(real_p).cycles(n=100)
    assert real_equity_by_date(cycles) == {DATES[0]: 100.0}


# ---------------------------------------------------------------------------
# Per-symbol trade check: expectations come from the SIM (Codex 3858329142)
# ---------------------------------------------------------------------------

def test_expected_trades_come_from_sim_not_mainnet(tmp_path):
    """A mainnet-journal ladder move with its own matching order must NOT
    self-legitimize: with no sim-intended trade the fill is unexpected."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        # Production signal (erroneously) steps to 1.0 and trades on it.
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[_filled_order()]),
        _cycle(DATES[2], 100.0, 1.0, orders_executed=[]),
    ])
    # The simulation never intended a trade.
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_sim_trade_dates == 0
    assert len(tr.unexpected_orders) == 1
    assert tr.unexpected_orders[0]["symbol"] == "BTC/USDT"
    assert tr.missing_trades == []
    # The date HAS a harness row, so its "no trade" expectation is real:
    # this is a mismatch, not a coverage gap.
    assert tr.out_of_coverage_fills == []


def test_sim_intended_trade_with_matching_fill_not_flagged(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[_filled_order()]),
        _cycle(DATES[2], 100.0, 1.0, orders_executed=[]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
        _sim_row(DATES[2], 10_000.0, ladder=1.0),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.n_sim_trade_dates == 1
    assert tr.missing_trades == []
    assert tr.wrong_direction_trades == []
    assert tr.partial_fills == []
    assert tr.unexpected_orders == []
    assert "mismatch" not in report.advisory.lower()


def test_sim_intended_trade_without_fill_flagged_missing(tmp_path, capsys):
    """Sim intends a trade; real journal shows neither fill nor skip →
    real execution silently missed it. Descriptive: exit stays 0."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, 100.0, 0.5, orders_executed=[]) for d in DATES
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
        _sim_row(DATES[2], 10_000.0, ladder=1.0),
    ])
    report = check_execution_tracking(real_p, sim_p)
    assert report.transitions.missing_trades == [
        {"date": DATES[1], "symbol": "BTC", "expected_side": "buy"}
    ]
    assert "mismatch" in report.advisory.lower()
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    assert "MISSING TRADE" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Per-symbol granularity: partial / wrong-direction (Codex 3858329145)
# ---------------------------------------------------------------------------

def test_per_symbol_partial_basket_execution_surfaces(tmp_path):
    """Sim intends BTC and ETH buys; only BTC filled → ETH missing.
    A date-level 'any fill' check would have passed this."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0,
               orders_executed=[_filled_order("BTC/USDT")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"BTC": 0.5 / 7, "ETH": 0.5 / 7}),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.missing_trades == [
        {"date": DATES[1], "symbol": "ETH", "expected_side": "buy"}
    ]
    assert tr.unexpected_orders == []


def test_wrong_direction_fill_flagged(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0,
               orders_executed=[_filled_order("BTC/USDT", "sell")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.wrong_direction_trades == [{
        "date": DATES[1], "symbol": "BTC",
        "expected_side": "buy", "actual_side": "sell",
    }]
    assert tr.missing_trades == []
    tracking_cli_main(_cli(real_p, sim_p))
    assert "WRONG DIRECTION" in capsys.readouterr().out


@pytest.mark.parametrize("status,filled", [
    ("partial", 0.001),   # exchange-terminal partial
    ("closed", 0.0004),   # under-filled vs intent
])
def test_partial_fill_surfaces(tmp_path, capsys, status, filled):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    order = _filled_order(
        terminal_status=status, intended_amount=0.001, filled_amount=filled,
    )
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[order]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert len(tr.partial_fills) == 1
    assert tr.partial_fills[0]["date"] == DATES[1]
    assert tr.partial_fills[0]["symbol"] == "BTC/USDT"
    assert tr.missing_trades == []
    tracking_cli_main(_cli(real_p, sim_p))
    assert "PARTIAL FILL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Skip coverage: live + min-notional-class reasons only (Codex 3858329150)
# ---------------------------------------------------------------------------

def test_live_min_notional_skip_covers_missing_trade(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[],
               orders_skipped=[_min_notional_skip("DOGE/USDT", "buy")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"DOGE": 0.5 / 7}),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.missing_trades == []
    assert tr.n_trades_skip_covered == 1
    assert tr.n_min_notional_skips == 1
    assert "mismatch" not in report.advisory.lower()
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0


@pytest.mark.parametrize("reason", [
    "amount 0.00000100 truncates to 0 at the exchange lot step",
    "amount 0.00000100 < min_amount 0.001; notional 3.0000 < min_cost 5",
])
def test_all_delta_submin_reason_shapes_cover(tmp_path, reason):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[],
               orders_skipped=[_min_notional_skip("DOGE/USDT", "buy", reason)]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"DOGE": 0.5 / 7}),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.missing_trades == []
    assert tr.n_trades_skip_covered == 1


def test_dry_run_skip_does_not_cover(tmp_path):
    """Planning skips of the 6-hourly dry-run never blocked a real order."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, mode="dry_run",
               orders_skipped=[_min_notional_skip("DOGE/USDT", "buy")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"DOGE": 0.5 / 7}),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_min_notional_skips == 0
    assert tr.n_trades_skip_covered == 0
    assert tr.missing_trades == [
        {"date": DATES[1], "symbol": "DOGE", "expected_side": "buy"}
    ]


@pytest.mark.parametrize("reason", ["pending_order", "pending_funding_sell"])
def test_pending_reason_skip_does_not_cover(tmp_path, reason):
    """Transient pending_* skips are retried next cycle — they must not
    legitimize a missed transition."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[],
               orders_skipped=[_min_notional_skip("DOGE/USDT", "buy", reason)]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"DOGE": 0.5 / 7}),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_min_notional_skips == 0
    assert tr.missing_trades == [
        {"date": DATES[1], "symbol": "DOGE", "expected_side": "buy"}
    ]


def test_wrong_side_skip_does_not_cover(tmp_path):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[],
               orders_skipped=[_min_notional_skip("DOGE/USDT", "sell")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"DOGE": 0.5 / 7}),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_trades_skip_covered == 0
    assert tr.missing_trades == [
        {"date": DATES[1], "symbol": "DOGE", "expected_side": "buy"}
    ]


def test_repeated_live_skips_on_one_date_count_once(tmp_path):
    """A re-run live cycle re-records the same skip — distinct
    (date, symbol) must not inflate with observation count."""
    real_p = tmp_path / "real.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[],
               orders_skipped=[_min_notional_skip()]),
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[],
               orders_skipped=[_min_notional_skip()]),
    ])
    from trade_lab.monitoring.data_source import JournalReader
    cycles = JournalReader(real_p).cycles(n=100)
    assert len(live_min_notional_skips(cycles)) == 1


# ---------------------------------------------------------------------------
# Signal-less records dated by clientOrderId (Codex 3858329155)
# ---------------------------------------------------------------------------

def test_signalless_partial_fill_dated_by_coid_not_ended_at(tmp_path):
    """A failed live cycle journals fills without a signal; a
    reconstruction recovery lands days later. Both must land on the
    decision's signal date (coid YYYYMMDD − 1), not ended_at."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    # Decision made 2026-08-03 (the day after signal date 2026-08-02).
    order = _filled_order(coid="tsmom_20260803_BTCUSDT_buy")
    _write_real(real_p, [
        _cycle(DATES[2], 100.0, 1.0, outcome="failed", signal=False,
               orders_executed=[order],
               ended_at=f"{DATES[2]}T00:06:00+00:00"),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    # Attributed to DATES[1] (signal date), matching the sim intent;
    # ended_at attribution would flag missing at [1] + unexpected at [2].
    assert tr.missing_trades == []
    assert tr.unexpected_orders == []
    assert tr.n_real_fill_events == 1


def test_reconstruction_final_state_supersedes_failed_record(tmp_path):
    """The same coid journaled twice (timeout partial, then reconstructed
    closed) is ONE event with the final state — no stale false partial."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    coid = "tsmom_20260803_BTCUSDT_buy"
    stale = _filled_order(
        coid=coid, terminal_status="timeout",
        intended_amount=0.001, filled_amount=0.0005,
    )
    final = _filled_order(
        coid=coid, terminal_status="closed",
        intended_amount=0.001, filled_amount=0.001,
    )
    _write_real(real_p, [
        _cycle(DATES[2], 100.0, 1.0, outcome="failed", signal=False,
               orders_executed=[stale]),
        _cycle("2026-08-05", 100.0, 1.0, outcome="reconstructed", signal=False,
               orders_executed=[final],
               ended_at="2026-08-05T06:05:30+00:00"),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_real_fill_events == 1
    assert tr.partial_fills == []
    assert tr.missing_trades == []
    assert tr.unexpected_orders == []


def test_unparseable_coid_falls_back_to_ended_at(tmp_path):
    real_p = tmp_path / "real.jsonl"
    order = _filled_order(coid="manual_fix_123")
    _write_real(real_p, [
        _cycle(DATES[2], 100.0, 1.0, outcome="failed", signal=False,
               orders_executed=[order]),
    ])
    from trade_lab.monitoring.data_source import JournalReader
    events = real_order_events(JournalReader(real_p).cycles(n=100))
    assert [e["date"] for e in events] == [DATES[2]]


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


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_threshold_rejected_at_parse_time(tmp_path, capsys, value):
    """Threshold <= 0 makes the breach check vacuous; NaN compares False
    everywhere (never breaches, non-standard JSON); inf never breaches.
    argparse must reject all with its native exit 2 (Codex 3858329160)."""
    with pytest.raises(SystemExit) as excinfo:
        tracking_cli_main(_cli(
            tmp_path / "r.jsonl", tmp_path / "s.jsonl",
            # = form: argparse reads a bare "-inf" as an option flag.
            f"--gap-threshold-pct={value}",
        ))
    assert excinfo.value.code == 2
    assert "must be a finite number > 0" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# Reader data quality (Codex 3858329165) and sim schema drift (3858329169)
# ---------------------------------------------------------------------------

def test_corrupt_mainnet_journal_line_exit_2(tmp_path, capsys):
    """A malformed line can hold the day's ladder change — reconciliation
    over a journal with holes is a tool error, not a clean report."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5)])
    with open(real_p, "a", encoding="utf-8") as f:
        f.write("{corrupt json\n")
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    with pytest.raises(ValueError, match="corrupt"):
        check_execution_tracking(real_p, sim_p)
    rc = tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach"))
    assert rc == 2
    assert "corrupt" in capsys.readouterr().err


def test_unknown_schema_version_lines_warn_incomplete(tmp_path, capsys):
    """Unknown-version lines degrade to an explicit incomplete-data
    warning: the report is produced (a version skew is actionable), but
    'no mismatches' must not read as verified."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    unknown = _cycle(DATES[1], 100.0, 0.5)
    unknown["schema_version"] = 99
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5), unknown])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    report = check_execution_tracking(real_p, sim_p)
    assert report.real_unknown_version_lines == 1
    assert "INCOMPLETE DATA" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    assert "INCOMPLETE DATA" in capsys.readouterr().out


def test_sim_schema_drift_exit_2_names_the_row(tmp_path, capsys):
    """An unknown field in a harness row (TypeError in HarnessLogRow) is
    a tool error naming the row — never exit 1, the breach code."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5)])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    drifted = dataclasses.asdict(_sim_row(DATES[1], 10_000.0))
    drifted["surprise_field"] = 1
    with open(sim_p, "a", encoding="utf-8") as f:
        f.write(json.dumps(drifted) + "\n")
    with pytest.raises(ValueError, match=DATES[1]):
        check_execution_tracking(real_p, sim_p)
    rc = tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach"))
    assert rc == 2
    err = capsys.readouterr().err
    assert "TRACKING ERROR" in err
    assert DATES[1] in err  # message names the offending row


# ---------------------------------------------------------------------------
# Harness journal: only a malformed FINAL line is tolerable (Codex 3858504483)
# ---------------------------------------------------------------------------

def test_corrupt_middle_harness_row_exit_2_names_the_line(tmp_path, capsys):
    """A malformed mid-journal row can carry the intended transition; if
    the matching real trade is missing too, skipping it would report
    'no mismatch'. Tool error naming the line, like the mainnet path."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, 100.0, 0.5, orders_executed=[]) for d in DATES
    ])
    _write_sim_raw(sim_p, [
        _sim_line(_sim_row(DATES[0], 10_000.0)),
        "{corrupt json\n",  # held the DATES[1] transition
        _sim_line(_sim_row(DATES[2], 10_000.0, ladder=1.0)),
    ])
    with pytest.raises(ValueError, match="line 2"):
        check_execution_tracking(real_p, sim_p)
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    err = capsys.readouterr().err
    assert "TRACKING ERROR" in err
    assert "line 2" in err


@pytest.mark.parametrize("tail", ["{trunc", "{trunc\n", "{trunc\n\n"])
def test_corrupt_tail_harness_row_tolerated(tmp_path, tail):
    """The documented crash-truncated append: last row with content,
    skipped, report still produced."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(d, 100.0, 0.5) for d in DATES[:2]])
    _write_sim_raw(sim_p, [
        _sim_line(_sim_row(DATES[0], 10_000.0)),
        _sim_line(_sim_row(DATES[1], 10_000.0)),
        tail,
    ])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 2
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0


# ---------------------------------------------------------------------------
# Simulation coverage vs "unexpected" (Codex 3858504486)
# ---------------------------------------------------------------------------

def test_fill_before_sim_coverage_is_not_unexpected(tmp_path, capsys):
    """Mainnet journal starts before the harness journal (staggered
    deployment / partial retention): those dates hold no simulated
    expectation at all, so their fills are a coverage note — labelling
    them 'unexpected' would be a permanent false mismatch."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[_filled_order()]),
        _cycle(DATES[1], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[2], 100.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[1:]])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.unexpected_orders == []
    assert [e["date"] for e in tr.out_of_coverage_fills] == [DATES[0]]
    assert tr.coverage == (DATES[1], DATES[2])
    assert "COVERAGE NOTE" in report.advisory
    assert DATES[0] in report.advisory
    assert "mismatch" not in report.advisory.lower().split("coverage note")[0]
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    assert "OUTSIDE SIM COVERAGE" in capsys.readouterr().out


def test_fill_after_sim_coverage_is_not_unexpected(tmp_path):
    """Same on the other end: the harness cron stopped, mainnet kept
    trading — later fills are outside coverage, not mismatches."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[2], 100.0, 0.5, orders_executed=[_filled_order()]),
    ])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.unexpected_orders == []
    assert [e["date"] for e in tr.out_of_coverage_fills] == [DATES[2]]
    assert tr.coverage == (DATES[0], DATES[0])
