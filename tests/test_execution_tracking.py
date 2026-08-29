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
    # Default book in these fixtures is 100.0 and the default sim intent
    # is 0.5/7 of it, so the fill must be ~7.14 quote for the size check
    # to see a faithful execution.
    filled_notional_quote: float = 100.0 * 0.5 / 7,
) -> dict:
    return {
        # Distinct default ids: events dedupe by clientOrderId.
        "client_order_id": coid or f"tsmom_x_{next(_COID_SEQ)}",
        "symbol": symbol,
        "side": side,
        "terminal_status": terminal_status,
        "intended_amount": intended_amount,
        "filled_amount": filled_amount,
        "filled_notional_quote": filled_notional_quote,
    }


def _min_notional_skip(
    symbol: str = "DOGE/USDT",
    side: str = "buy",
    reason: str = "notional 7.1429 < min_cost 10",
    desired_notional: float = 100.0 * 0.5 / 7,
) -> dict:
    # Size-consistent with the default fixtures: the sim intends 0.5/7 of
    # a 100 book (~7.14 quote) and the venue minimum sits above it, so the
    # skip legitimately IS that intent. A skip far under the intent is a
    # sizing bug and must NOT grant coverage.
    return {
        "symbol": symbol,
        "desired_side": side,
        "desired_amount": 10.0,
        "desired_notional": desired_notional,
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
        # A live cycle must exist, else the date sits before live coverage
        # and carries no expectation at all.
        _cycle(DATES[1], 100.0, 1.0),
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


def test_failed_live_cycle_with_read_phase_equity_is_used(tmp_path):
    """A live attempt owns its date whatever the outcome (Codex 3858753685).

    'partial' / 'unknown_orders' cycles still read equity PRE-trade and may
    have placed fills; substituting a later dry-run would value the book
    AFTER those fills, at a different market window."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 0.5, outcome="partial"),
        _cycle(DATES[2], 100.0, 0.5),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 3
    assert report.equity.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not report.equity.breached


def test_live_cycle_without_equity_drops_the_date(tmp_path):
    """A live attempt that never reached the read phase must NOT be
    back-filled by that day's dry-run: whether it traded is exactly what
    is unknown, so a post-execution valuation could be silently compared
    against the sim's pre-trade one."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    dead = _cycle(DATES[1], 100.0, 0.5, outcome="failed")
    dead["equity_usd"] = None
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        dead,
        # Same date, later dry-run — must not become the date's sample.
        _cycle(DATES[1], 250.0, 0.5, mode="dry_run"),
        _cycle(DATES[2], 100.0, 0.5),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.n_aligned_days == 2
    assert report.equity.overlap == (DATES[0], DATES[2])


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


# ---------------------------------------------------------------------------
# Third review round: size, pre-live coverage, cache re-journal, non-finite
# ---------------------------------------------------------------------------

def test_undersized_fill_is_flagged(tmp_path, capsys):
    """Codex 3858753674: presence + direction is not a match. A fully
    filled dust order where the sim intended a real weight change must
    surface, not read as clean."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        # Sim intends 0.5/7 ≈ 7.1% of a 100 book (~7.14 quote); a sizing
        # bug sends 1.00 quote and fills it completely.
        _cycle(DATES[1], 100.0, 1.0,
               orders_executed=[_filled_order(filled_notional_quote=1.0)]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.missing_trades == []      # the order exists and fills fully
    assert tr.partial_fills == []       # ... and is not partial
    assert len(tr.size_mismatches) == 1
    m = tr.size_mismatches[0]
    assert m["symbol"] == "BTC" and m["side"] == "buy"
    assert m["ratio"] == pytest.approx(0.14, abs=0.01)
    assert "mis-sized" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "SIZE MISMATCH" in capsys.readouterr().out


def test_faithful_fill_size_is_not_flagged(tmp_path):
    """The band must tolerate lot steps, the funding reserve and price
    drift — only order-of-magnitude errors trip it."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[
            # 3% under the intent: lot step + 10 bp reserve territory.
            _filled_order(filled_notional_quote=100.0 * 0.5 / 7 * 0.97),
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.size_mismatches == []


def test_sim_intents_before_first_live_attempt_are_not_missing(tmp_path):
    """Codex 3858753683: the harness journal predates live trading — those
    intents never had a real cycle to execute them."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        # Observation phase: dry-runs only on DATES[0].
        _cycle(DATES[0], 100.0, 0.5, mode="dry_run"),
        _cycle(DATES[1], 100.0, 1.0),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
        _sim_row(DATES[1], 10_000.0, ladder=1.0),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.missing_trades == []
    assert [e["date"] for e in tr.pre_live_sim_trades] == [DATES[0]]
    assert "before the first live attempt" in report.advisory


def test_sim_intent_after_first_live_attempt_still_flagged(tmp_path):
    """The pre-live exemption must not swallow genuine gaps afterwards."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.pre_live_sim_trades == []
    assert [m["symbol"] for m in tr.missing_trades] == ["BTC"]


def test_cached_rerun_record_does_not_erase_the_fill(tmp_path):
    """Codex 3858753688: a same-day rerun re-journals the state-cache
    result with zeroed fill detail; letting it win would report the
    executed trade as missing."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    # The coid carries the DECISION date; the bar it acted on is the day
    # before, i.e. signal date DATES[1].
    coid = "tsmom_20260803_BTCUSDT_buy"
    filled = _filled_order(coid=coid)
    cached = _filled_order(
        coid=coid, intended_amount=0.0, filled_amount=0.0,
        filled_notional_quote=0.0,
    )
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[filled]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[cached],
               cycle_id="aabbccdd-0001"),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.missing_trades == []
    assert tr.size_mismatches == []
    assert tr.n_real_fill_events == 1


@pytest.mark.parametrize("bad", ["NaN", '"nan"'])
def test_non_finite_equity_is_a_tool_error(tmp_path, capsys, bad):
    """Codex 3858753678: NaN fails every comparison, so a bare positivity
    check would pass it through and the breach test would silently read
    'within threshold'. Both shapes reach the equity check — bare NaN is
    a json module extension, "nan" is a string float() accepts (bare
    lowercase nan is invalid JSON and dies earlier, as a corrupt line)."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5), _cycle(DATES[1], 100.0, 0.5),
    ])
    lines = real_p.read_text().splitlines()
    lines[1] = lines[1].replace('"equity_usd": 100.0', f'"equity_usd": {bad}')
    real_p.write_text("\n".join(lines) + "\n")
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    err = capsys.readouterr().err
    assert "non-finite equity" in err


def test_dust_skip_does_not_cover_a_material_intent(tmp_path):
    """Codex 3859624351: a sizing bug computes a dust order, the venue
    refuses it as sub-minimum, and the legitimate-looking reason would
    excuse a materially larger simulated intent."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        # Sim intends ~7.14 quote of a 100 book; the real planner sized
        # 0.30 and the venue refused it.
        _cycle(DATES[1], 100.0, 1.0, orders_skipped=[
            _min_notional_skip("BTC/USDT", "buy", desired_notional=0.30),
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_trades_skip_covered == 0
    assert [m["symbol"] for m in tr.missing_trades] == ["BTC"]


def test_same_sized_skip_still_covers(tmp_path):
    """The genuine case must keep working: the skip IS the sim's intent,
    refused by the venue minimum."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5),
        _cycle(DATES[1], 100.0, 1.0,
               orders_skipped=[_min_notional_skip("BTC/USDT", "buy")]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.n_trades_skip_covered == 1
    assert tr.missing_trades == []


def test_fill_on_another_quote_market_is_not_a_match(tmp_path, capsys):
    """Codex 3859624374: the sim trades {asset}/USDT. Reducing symbols to
    the base asset would let BTC/FDUSD satisfy a BTC/USDT expectation."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[
            _filled_order(symbol="BTC/FDUSD"),
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert [e["symbol"] for e in tr.wrong_market_fills] == ["BTC/FDUSD"]
    # ...and the USDT trade is still reported as never executed.
    assert [m["symbol"] for m in tr.missing_trades] == ["BTC"]
    assert "WRONG MARKET" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "WRONG MARKET" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Capital flows: deposits and withdrawals are not returns (first real run)
# ---------------------------------------------------------------------------

def _capital_events_file(tmp_path, entries):
    path = tmp_path / "capital_events.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


# The production journals as of the first real run: the mainnet account
# was opened for read-only observation with dust on it, funded to ~99 the
# day the live cron started, and topped up to ~150 later.
OBS_DATE = "2026-07-08"
LIVE_DATES = ["2026-08-24", "2026-08-25"]


def test_equity_comparison_starts_at_live_coverage(tmp_path):
    """Normalizing to the first date ANY cycle exists anchored the curve
    on the observation phase, where the account held 0.048 USDT — the
    funding that followed then read as a +285194% level gap."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(OBS_DATE, 0.048, 0.5, mode="dry_run"),
        _cycle(LIVE_DATES[0], 99.0, 0.5, orders_executed=[]),
        _cycle(LIVE_DATES[1], 99.5, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [
        _sim_row(OBS_DATE, 10_000.0),
        _sim_row(LIVE_DATES[0], 10_000.0),
        _sim_row(LIVE_DATES[1], 10_050.0),
    ])
    report = check_execution_tracking(real_p, sim_p)
    eq = report.equity
    assert eq.tracking_start == LIVE_DATES[0]
    assert eq.n_aligned_days == 2
    assert eq.overlap == (LIVE_DATES[0], LIVE_DATES[1])
    assert eq.unexplained_moves == []
    assert eq.level_gap_pct == pytest.approx(0.0, abs=0.05)
    assert not eq.breached


def test_observation_only_journal_tracks_nothing(tmp_path):
    """No live attempt and no retained fill: dry-run equity is not a
    tracked book, so there is no gap to report at all."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(d, eq, 0.5, mode="dry_run")
        for d, eq in zip(DATES, [0.048, 99.0, 150.0])
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    report = check_execution_tracking(real_p, sim_p)
    assert report.equity.tracking_start is None
    assert report.equity.n_aligned_days == 0
    assert report.equity.level_gap_pct is None
    assert not report.equity.breached
    assert "No live-execution coverage yet" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0


def test_undeclared_deposit_suppresses_the_gap_number(tmp_path, capsys):
    """A +50% overnight step in a book holding the same basket as the sim
    is a transfer, not performance — and the layer cannot tell. It must
    refuse the number, loudly, instead of publishing a conflated one."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 150.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    report = check_execution_tracking(real_p, sim_p)
    eq = report.equity
    assert len(eq.unexplained_moves) == 1
    move = eq.unexplained_moves[0]
    assert (move["from_date"], move["date"]) == (DATES[0], DATES[1])
    assert move["real_return"] == pytest.approx(0.5)
    assert move["declared_flow_usd"] == 0.0
    assert eq.level_gap_pct is None
    assert eq.cum_abs_return_diff is None
    assert not eq.breached          # no number, hence no verdict
    assert "UNEXPLAINED EQUITY MOVE" in report.advisory
    # A refusal to report is not a pass.
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 1
    out = capsys.readouterr().out
    assert "UNEXPLAINED EQUITY MOVE" in out
    assert "level gap = NOT REPORTED" in out


def test_declared_deposit_restores_the_gap_number(tmp_path):
    """Declared capital is removed from the return, not from the book."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 150.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 50.0, "note": "top-up"},
    ])
    report = check_execution_tracking(real_p, sim_p, capital_events_path=events)
    eq = report.equity
    assert eq.unexplained_moves == []
    assert eq.capital_flows_applied == [
        {"from_date": DATES[0], "date": DATES[1], "amount_usd": 50.0}
    ]
    assert eq.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not eq.breached
    assert tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(events), "--fail-on-breach",
    )) == 0


def test_declared_flow_inside_a_skipped_step_is_removed(tmp_path):
    """The transfer landed on a date the harness never logged; the jump
    still sits inside the enclosing aligned step and must be removed —
    settled at its own date, which is where the real book records it."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 150.0, 0.5, orders_executed=[]),
        _cycle(DATES[2], 150.0, 0.5, orders_executed=[]),
    ])
    # No harness row for DATES[1]: the aligned step is DATES[0] -> DATES[2].
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0), _sim_row(DATES[2], 10_000.0),
    ])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 50.0},
    ])
    eq = check_execution_tracking(
        real_p, sim_p, capital_events_path=events,
    ).equity
    assert eq.unexplained_moves == []
    assert eq.capital_flows_applied == [
        {"from_date": DATES[0], "date": DATES[1], "amount_usd": 50.0}
    ]
    assert eq.level_gap_pct == pytest.approx(0.0, abs=1e-9)


def test_flow_earns_the_market_move_that_follows_it(tmp_path):
    """Codex 3865143806: 100 USD book, +100 deposit mid-span, the enlarged
    book then gains 10% -> 220. Netting the nominal 100 at the span end
    reads (220 - 100)/100 - 1 = +20% and manufactures a 9.1% gap against
    a sim that also gained 10%. The transfer earns the move that follows
    it: (200 - 100)/100 x 220/200 - 1 = +10%, gap 0."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 200.0, 0.5, orders_executed=[]),   # deposit landed
        _cycle(DATES[2], 220.0, 0.5, orders_executed=[]),   # +10% on 200
    ])
    # No harness row for DATES[1]: the aligned step is DATES[0] -> DATES[2].
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0), _sim_row(DATES[2], 11_000.0),
    ])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 100.0},
    ])
    eq = check_execution_tracking(
        real_p, sim_p, capital_events_path=events,
    ).equity
    assert eq.unexplained_moves == []
    assert eq.capital_flows_applied == [
        {"from_date": DATES[0], "date": DATES[1], "amount_usd": 100.0}
    ]
    assert eq.cum_abs_return_diff == pytest.approx(0.0, abs=1e-12)
    assert eq.level_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not eq.breached
    assert tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(events), "--fail-on-breach",
    )) == 0


def test_two_flows_in_one_step_chain_at_their_own_dates(tmp_path):
    """100 -> +100 deposit -> +10% -> +20 top-up -> +10%. Each leg is
    measured against the book the previous transfer left behind, so the
    real return is 1.1 x 1.1 - 1 = +21%, matching the sim exactly."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]
    _write_real(real_p, [
        _cycle(days[0], 100.0, 0.5, orders_executed=[]),
        _cycle(days[1], 200.0, 0.5, orders_executed=[]),   # +100 deposit
        _cycle(days[2], 240.0, 0.5, orders_executed=[]),   # 220 + 20 top-up
        _cycle(days[3], 264.0, 0.5, orders_executed=[]),   # +10% on 240
    ])
    _write_sim(sim_p, [
        _sim_row(days[0], 10_000.0), _sim_row(days[3], 12_100.0),
    ])
    events = _capital_events_file(tmp_path, [
        {"date": days[1], "amount_usd": 100.0},
        {"date": days[2], "amount_usd": 20.0},
    ])
    eq = check_execution_tracking(
        real_p, sim_p, capital_events_path=events,
    ).equity
    assert eq.capital_flows_applied == [
        {"from_date": days[0], "date": days[1], "amount_usd": 100.0},
        {"from_date": days[1], "date": days[2], "amount_usd": 20.0},
    ]
    assert eq.unexplained_moves == []
    assert eq.level_gap_pct == pytest.approx(0.0, abs=1e-9)


def test_flow_on_a_date_without_a_real_equity_reading_is_a_tool_error(
    tmp_path, capsys,
):
    """The split needs the book AT the transfer. Without a reading there
    the flow could only be netted at one end of the span — the arithmetic
    this file exists to avoid — so the declaration is rejected."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[2], 220.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0), _sim_row(DATES[2], 11_000.0),
    ])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 100.0},
    ])
    with pytest.raises(ValueError, match="no usable real equity reading"):
        check_execution_tracking(real_p, sim_p, capital_events_path=events)
    assert tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(events), "--fail-on-breach",
    )) == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


def test_mis_sized_declaration_still_surfaces(tmp_path):
    """The detector runs on the flow-ADJUSTED return, so declaring 10 for
    a 50 transfer cannot silence it."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 150.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 10.0},
    ])
    eq = check_execution_tracking(
        real_p, sim_p, capital_events_path=events,
    ).equity
    assert len(eq.unexplained_moves) == 1
    assert eq.unexplained_moves[0]["declared_flow_usd"] == 10.0
    assert eq.level_gap_pct is None


def test_market_crash_is_not_read_as_a_capital_event(tmp_path):
    """The real book sits in cash while the sim is fully invested through
    a −30% basket day. That divergence is execution reality, not a
    transfer — the basket's own move is the plausibility yardstick."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 1.0, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0, ladder=1.0),
        dataclasses.replace(
            _sim_row(DATES[1], 7_000.0, ladder=1.0), daily_return=-0.30,
        ),
    ])
    eq = check_execution_tracking(real_p, sim_p).equity
    assert eq.unexplained_moves == []
    assert eq.level_gap_pct == pytest.approx(42.857, abs=0.01)
    assert eq.breached          # a real, reportable tracking gap


def test_crash_across_a_skipped_span_is_not_read_as_a_capital_event(tmp_path):
    """Codex 3865143794: the same crash spread over days the two journals
    never intersect on. The compared returns span the whole gap, so the
    yardstick must too: judged against the ending day's -1% the -30.7%
    divergence clears the floor and mislabels a genuine tracking gap as
    an undeclared transfer, suppressing the number entirely."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]
    # The real book sits in cash; only the endpoints have a real cycle.
    _write_real(real_p, [
        _cycle(days[0], 100.0, 1.0, orders_executed=[]),
        _cycle(days[3], 100.0, 1.0, orders_executed=[]),
    ])
    # Sim fully invested through -20%, -12.5%, -1%: 10_000 -> 6_930.
    _write_sim(sim_p, [
        dataclasses.replace(
            _sim_row(d, eq, ladder=1.0), daily_return=ret,
        )
        for d, eq, ret in zip(
            days, [10_000.0, 8_000.0, 7_000.0, 6_930.0],
            [0.0, -0.20, -0.125, -0.01],
        )
    ])
    report = check_execution_tracking(real_p, sim_p)
    eq = report.equity
    assert eq.n_aligned_days == 2
    assert eq.unexplained_moves == []
    assert eq.cum_abs_return_diff == pytest.approx(0.307, abs=1e-9)
    assert eq.level_gap_pct == pytest.approx(44.300, abs=0.01)
    assert eq.breached
    assert "TRACKING BREACH" in report.advisory
    assert "UNEXPLAINED" not in report.advisory


def test_deposit_across_a_skipped_span_still_surfaces(tmp_path):
    """The widened yardstick is the basket's own move, not the span's
    length: over a quiet basket a +50% step across the same gap is still
    an undeclared transfer."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]
    _write_real(real_p, [
        _cycle(days[0], 100.0, 0.5, orders_executed=[]),
        _cycle(days[3], 150.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in days])
    eq = check_execution_tracking(real_p, sim_p).equity
    assert len(eq.unexplained_moves) == 1
    assert eq.unexplained_moves[0]["real_return"] == pytest.approx(0.5)
    assert eq.level_gap_pct is None


def test_withdrawal_larger_than_the_book_is_a_tool_error(tmp_path, capsys):
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 0.5, orders_executed=[]),
    ])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    events = _capital_events_file(tmp_path, [
        {"date": DATES[1], "amount_usd": 500.0},
    ])
    with pytest.raises(ValueError, match="non-positive equity"):
        check_execution_tracking(real_p, sim_p, capital_events_path=events)
    assert tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(events),
    )) == 2
    assert "TRACKING ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("entries,needle", [
    ({"2026-08-01": 50.0}, "expected a JSON list"),
    ([["2026-08-01", 50.0]], "expected an object"),
    ([{"date": "01/08/2026", "amount_usd": 50.0}], "YYYY-MM-DD"),
    ([{"amount_usd": 50.0}], "YYYY-MM-DD"),
    ([{"date": "2026-08-01", "amount_usd": "fifty"}], "must be a number"),
    ([{"date": "2026-08-01"}], "must be a number"),
    ([{"date": "2026-08-01", "amount_usd": 0.0}], "finite non-zero"),
    (
        [{"date": "2026-08-01", "amount_usd": 50.0},
         {"date": "2026-08-01", "amount_usd": 50.0}],
        "duplicate date",
    ),
])
def test_malformed_capital_events_file_is_a_tool_error(
    tmp_path, capsys, entries, needle,
):
    """A silently ignored declaration leaves the transfer inside the
    return series — the exact number the file exists to remove."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5, orders_executed=[])])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    events = _capital_events_file(tmp_path, entries)
    assert tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(events), "--fail-on-breach",
    )) == 2
    err = capsys.readouterr().err
    assert "TRACKING ERROR" in err
    assert needle in err


def test_missing_capital_events_file_is_a_tool_error(tmp_path, capsys):
    """A typo in the path must not read as 'no transfers ever happened'."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[0], 100.0, 0.5, orders_executed=[])])
    _write_sim(sim_p, [_sim_row(DATES[0], 10_000.0)])
    rc = tracking_cli_main(_cli(
        real_p, sim_p, "--capital-events", str(tmp_path / "nope.json"),
    ))
    assert rc == 2
    assert "capital-events file not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Residual #71 items on the execution_tracking cluster
# ---------------------------------------------------------------------------

def test_extra_opposite_side_fill_surfaces(tmp_path, capsys):
    """A sell that rode along with the expected buy is invisible to every
    other check: the intent reads as satisfied and the date+asset is not
    'unexpected' either — the leg was round-tripped for two spreads."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5, orders_executed=[]),
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[
            _filled_order("BTC/USDT", "buy"),
            _filled_order("BTC/USDT", "sell"),
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[0], 10_000.0),
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert len(tr.extra_opposite_fills) == 1
    extra = tr.extra_opposite_fills[0]
    assert extra["date"] == DATES[1]
    assert extra["expected_side"] == "buy" and extra["actual_side"] == "sell"
    assert tr.missing_trades == []
    assert tr.wrong_direction_trades == []
    assert tr.unexpected_orders == []
    assert "extra opposite-side" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "EXTRA OPPOSITE FILL" in capsys.readouterr().out


def test_second_opposite_fill_surfaces_beside_wrong_direction(tmp_path):
    """With no correct fill at all the first opposite fill is already the
    wrong-direction verdict; the rest must not vanish with it."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, orders_executed=[
            _filled_order("BTC/USDT", "sell"),
            _filled_order("BTC/USDT", "sell"),
        ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert len(tr.wrong_direction_trades) == 1
    assert len(tr.extra_opposite_fills) == 1


@pytest.mark.parametrize("bad", ['"abc"', "null"])
def test_non_numeric_real_equity_is_a_tool_error(tmp_path, capsys, bad):
    """Dropping the date silently shortens the compared span and hides
    whatever divergence it carried — the same posture as non-finite."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[0], 100.0, 0.5), _cycle(DATES[1], 100.0, 0.5),
    ])
    lines = real_p.read_text().splitlines()
    lines[1] = lines[1].replace(
        '"equity_usd": 100.0', f'"equity_usd": {{"amount": {bad}}}',
    )
    real_p.write_text("\n".join(lines) + "\n")
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES[:2]])
    with pytest.raises(ValueError, match=f"non-numeric equity_usd on {DATES[1]}"):
        check_execution_tracking(real_p, sim_p)
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    assert "non-numeric equity_usd" in capsys.readouterr().err


@pytest.mark.parametrize("bad,needle", [
    (float("nan"), "non-finite intended trade"),
    (float("inf"), "non-finite intended trade"),
    ("n/a", "non-numeric intended trade"),
])
def test_non_finite_intended_trade_is_a_tool_error(
    tmp_path, capsys, bad, needle,
):
    """NaN fails ``abs(dw) > eps``, so the intent would drop out of the
    expectations and a genuinely missed trade would read as clean."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(DATES[1], 100.0, 1.0, orders_executed=[])])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5,
                 intended={"BTC": bad}),
    ])
    with pytest.raises(ValueError, match=needle):
        check_execution_tracking(real_p, sim_p)
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    err = capsys.readouterr().err
    assert needle in err and DATES[1] in err


def test_retained_fill_date_counts_as_live_coverage(tmp_path):
    """A signal-less recovery record carries no cycle signal, but its
    retained fill proves live execution ran for that date. Treating the
    date as pre-live would exempt every intent on it from the check."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[2], 100.0, 1.0, outcome="failed", signal=False,
               orders_executed=[
                   _filled_order(coid="tsmom_20260803_BTCUSDT_buy"),
               ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.live_coverage_start == DATES[1]
    assert tr.pre_live_sim_trades == []      # the intent IS under coverage
    assert tr.missing_trades == []           # ...and it was executed
    assert tr.unexpected_orders == []


def test_late_execution_keeps_its_original_signal_date(tmp_path):
    """A recovery cycle days later carries its OWN fresh signal. Dating
    the recovered fill by that signal splits it from the intent it
    belongs to: missing on the decision date, unexpected on the recovery
    date. The coid's embedded decision date is the authority."""
    from trade_lab.monitoring.data_source import JournalReader
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [
        _cycle(DATES[1], 100.0, 1.0, outcome="failed", signal=False),
        # Live cycle on 2026-08-05 with a signal of its own, re-journaling
        # the order decided on 2026-08-03 (signal date 2026-08-02).
        _cycle("2026-08-05", 100.0, 1.0, outcome="reconstructed",
               orders_executed=[
                   _filled_order(coid="tsmom_20260803_BTCUSDT_buy"),
               ]),
    ])
    _write_sim(sim_p, [
        _sim_row(DATES[1], 10_000.0, ladder=1.0, prior=0.5, intended=BTC_BUY),
        _sim_row("2026-08-05", 10_000.0, ladder=1.0),
    ])
    events = real_order_events(JournalReader(real_p).cycles(n=100))
    assert [e["date"] for e in events] == [DATES[1]]
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.missing_trades == []
    assert tr.unexpected_orders == []
    assert tr.out_of_coverage_fills == []


# ---------------------------------------------------------------------------
# First-live catch-up orders are bootstrap, not "unexpected" (production
# shape: the sim stepped 0 -> 1 days before the live cron existed)
# ---------------------------------------------------------------------------

def _standing_row(date_iso: str, equity: float) -> HarnessLogRow:
    """Harness row holding the full basket, with no intent of its own."""
    return dataclasses.replace(
        _sim_row(date_iso, equity, ladder=1.0),
        target_weights={a: 1.0 / 7 for a in BASKET},
        current_weights={a: 1.0 / 7 for a in BASKET},
    )


# A faithful catch-up buys the sim's whole standing leg: 1/7 of a 100 book.
STANDING_NOTIONAL = 100.0 / 7


def test_first_live_catchup_orders_are_bootstrap(tmp_path, capsys):
    """Production shape: the sim stepped into the basket on 2026-08-20,
    the live cron started on 2026-08-24 without trading, and the seven
    catch-up buys landed on 08-25. They execute an intent the simulation
    really made — a category of their own, not seven UNEXPECTED ORDERs."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    sim_step, cron_day, live_day = "2026-08-20", "2026-08-24", "2026-08-25"
    _write_real(real_p, [
        _cycle(cron_day, 100.0, 1.0, orders_executed=[]),
        _cycle(live_day, 100.0, 1.0, orders_executed=[
            _filled_order(f"{a}/USDT", "buy",
                          filled_notional_quote=STANDING_NOTIONAL)
            for a in BASKET
        ]),
    ])
    _write_sim(sim_p, [
        dataclasses.replace(
            _sim_row(sim_step, 10_000.0, ladder=1.0, prior=0.0,
                     intended={a: 1.0 / 7 for a in BASKET}),
            target_weights={a: 1.0 / 7 for a in BASKET},
            current_weights={},
        ),
        _standing_row(cron_day, 10_000.0),
        _standing_row(live_day, 10_000.0),
    ])
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert tr.unexpected_orders == []
    assert len(tr.bootstrap_orders) == 7
    # Complete and correctly sized — the exemption applies to the date only.
    assert tr.partial_fills == []
    assert tr.size_mismatches == []
    assert {e["symbol"] for e in tr.bootstrap_orders} == {
        f"{a}/USDT" for a in BASKET
    }
    assert {e["sim_intent_date"] for e in tr.bootstrap_orders} == {sim_step}
    assert {e["date"] for e in tr.bootstrap_orders} == {live_day}
    assert tr.live_coverage_start == cron_day
    assert "BOOTSTRAP" in report.advisory
    assert "Trade mismatches" not in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
    out = capsys.readouterr().out
    assert "BOOTSTRAP ORDER" in out
    assert "UNEXPECTED ORDER" not in out


def test_buy_after_the_bootstrap_date_stays_unexpected(tmp_path):
    """The exemption is one date wide: once the book is established, a
    fill the simulation never intended is a genuine surprise."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    sim_step, live_day, next_day = "2026-08-20", "2026-08-25", "2026-08-26"
    _write_real(real_p, [
        _cycle(live_day, 100.0, 1.0, orders_executed=[
            _filled_order(f"{a}/USDT", "buy",
                          filled_notional_quote=STANDING_NOTIONAL)
            for a in BASKET
        ]),
        _cycle(next_day, 100.0, 1.0, orders_executed=[
            _filled_order("BTC/USDT", "buy"),
        ]),
    ])
    _write_sim(sim_p, [
        dataclasses.replace(
            _sim_row(sim_step, 10_000.0, ladder=1.0, prior=0.0,
                     intended={a: 1.0 / 7 for a in BASKET}),
            target_weights={a: 1.0 / 7 for a in BASKET},
        ),
        _standing_row(live_day, 10_000.0),
        _standing_row(next_day, 10_000.0),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert len(tr.bootstrap_orders) == 7
    assert [e["date"] for e in tr.unexpected_orders] == [next_day]


def test_first_fill_sell_without_intent_stays_unexpected(tmp_path):
    """Same journal as the bootstrap case except the side: nothing was
    bought before the bootstrap date for a sell to unwind."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    sim_step, live_day = "2026-08-20", "2026-08-25"
    _write_real(real_p, [
        _cycle(live_day, 100.0, 1.0, orders_executed=[
            _filled_order("BTC/USDT", "sell"),
        ]),
    ])
    _write_sim(sim_p, [
        dataclasses.replace(
            _sim_row(sim_step, 10_000.0, ladder=1.0, prior=0.0,
                     intended={a: 1.0 / 7 for a in BASKET}),
            target_weights={a: 1.0 / 7 for a in BASKET},
        ),
        _standing_row(live_day, 10_000.0),
    ])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.bootstrap_orders == []
    assert [e["symbol"] for e in tr.unexpected_orders] == ["BTC/USDT"]


def test_first_fill_without_an_earlier_intent_stays_unexpected(tmp_path):
    """A standing weight alone is not evidence of a delayed intent: with
    no earlier sim row that wanted to buy the asset, the first real fill
    is an order the simulation never asked for."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    live_day = "2026-08-25"
    _write_real(real_p, [
        _cycle(live_day, 100.0, 1.0, orders_executed=[
            _filled_order("BTC/USDT", "buy"),
        ]),
    ])
    _write_sim(sim_p, [_standing_row(live_day, 10_000.0)])
    tr = check_execution_tracking(real_p, sim_p).transitions
    assert tr.bootstrap_orders == []
    assert [e["symbol"] for e in tr.unexpected_orders] == ["BTC/USDT"]


# ---------------------------------------------------------------------------
# ...but bootstrap excuses the DATE only, never the fill quality
# (Codex 3865143784)
# ---------------------------------------------------------------------------

def _bootstrap_journals(tmp_path, btc_order: dict):
    """The production bootstrap shape with BTC's catch-up fill swapped out."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    sim_step, live_day = "2026-08-20", "2026-08-25"
    _write_real(real_p, [
        _cycle(live_day, 100.0, 1.0, orders_executed=[btc_order] + [
            _filled_order(f"{a}/USDT", "buy",
                          filled_notional_quote=STANDING_NOTIONAL)
            for a in BASKET if a != "BTC"
        ]),
    ])
    _write_sim(sim_p, [
        dataclasses.replace(
            _sim_row(sim_step, 10_000.0, ladder=1.0, prior=0.0,
                     intended={a: 1.0 / 7 for a in BASKET}),
            target_weights={a: 1.0 / 7 for a in BASKET},
            current_weights={},
        ),
        _standing_row(live_day, 10_000.0),
    ])
    return real_p, sim_p


def test_partially_filled_bootstrap_order_surfaces(tmp_path, capsys):
    """A catch-up that filled 90% of its order is still classified as
    bootstrap — and must still be reported as partial: the real book
    holds less than the position the simulation stands in."""
    real_p, sim_p = _bootstrap_journals(tmp_path, _filled_order(
        "BTC/USDT", "buy", terminal_status="partial",
        intended_amount=0.001, filled_amount=0.0009,
        filled_notional_quote=STANDING_NOTIONAL * 0.9,
    ))
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert len(tr.bootstrap_orders) == 7      # still not "unexpected"
    assert tr.unexpected_orders == []
    assert len(tr.partial_fills) == 1
    assert tr.partial_fills[0]["symbol"] == "BTC/USDT"
    assert tr.partial_fills[0]["date"] == "2026-08-25"
    assert tr.size_mismatches == []           # 90% is inside the size band
    assert "1 partial" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "PARTIAL FILL" in capsys.readouterr().out


@pytest.mark.parametrize("notional,ratio", [
    (1.0, 0.07),                        # dust where a seventh was due
    (100.0, 7.0),                       # the whole book into one leg
])
def test_mis_sized_bootstrap_order_surfaces(tmp_path, capsys, notional, ratio):
    """A complete catch-up fill of the wrong SIZE reads as clean on every
    other axis: it is not missing, not wrong-direction, not partial, and
    the bootstrap category exempted it from 'unexpected'."""
    real_p, sim_p = _bootstrap_journals(tmp_path, _filled_order(
        "BTC/USDT", "buy", filled_notional_quote=notional,
    ))
    report = check_execution_tracking(real_p, sim_p)
    tr = report.transitions
    assert len(tr.bootstrap_orders) == 7
    assert tr.unexpected_orders == []
    assert tr.partial_fills == []
    assert len(tr.size_mismatches) == 1
    m = tr.size_mismatches[0]
    assert m["symbol"] == "BTC" and m["side"] == "buy"
    assert m["expected_weight_delta"] == pytest.approx(1.0 / 7)
    assert m["ratio"] == pytest.approx(ratio, abs=0.01)
    assert "1 mis-sized" in report.advisory
    assert tracking_cli_main(_cli(real_p, sim_p)) == 0
    assert "SIZE MISMATCH" in capsys.readouterr().out


def test_non_numeric_basket_return_is_a_tool_error(tmp_path, capsys):
    """Codex 3877928981: as_float turned a null daily_return into 0.0 — a
    flat yardstick. A real book in cash through a basket crash would then
    look like an unexplained transfer and its genuine gap be suppressed."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(d, 100.0, 0.5) for d in DATES])
    rows = [_sim_row(d, 10_000.0) for d in DATES]
    lines = [_sim_line(r) for r in rows]
    lines[1] = lines[1].replace('"daily_return": 0.0', '"daily_return": null')
    _write_sim_raw(sim_p, lines)
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    assert "daily_return" in capsys.readouterr().err


def test_backfilled_harness_row_is_a_tool_error(tmp_path, capsys):
    """Codex 3877928979: a row appended out of date order was written
    against a different predecessor, so its return overlaps its
    successor's span — compounding both double-counts the overlap."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(d, 100.0, 0.5) for d in DATES])
    # DATES[2] journaled BEFORE DATES[1]: a --asof backfill.
    _write_sim_raw(sim_p, [
        _sim_line(_sim_row(DATES[0], 10_000.0)),
        _sim_line(_sim_row(DATES[2], 10_200.0)),
        _sim_line(_sim_row(DATES[1], 10_100.0)),
    ])
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 2
    err = capsys.readouterr().err
    assert "not in date order" in err and DATES[1] in err


def test_in_order_journal_still_reconciles(tmp_path):
    """The guard must not fire on a normal ascending journal."""
    real_p, sim_p = tmp_path / "real.jsonl", tmp_path / "sim.jsonl"
    _write_real(real_p, [_cycle(d, 100.0, 0.5) for d in DATES])
    _write_sim(sim_p, [_sim_row(d, 10_000.0) for d in DATES])
    assert tracking_cli_main(_cli(real_p, sim_p, "--fail-on-breach")) == 0
