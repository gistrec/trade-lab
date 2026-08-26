"""CLI for the execution-tracking layer (issue #11).

Typical operator invocation (manual or cron)::

    .venv/bin/python -m trade_lab.paper_trading.execution_tracking_cli

Exit codes — same contract as ``fingerprint_cli``: 0 — report produced
(default even on breach: descriptive, not normative); 1 — tracking
threshold breached OR an unexplained equity move suppressed the gap
number, AND ``--fail-on-breach`` was passed (a refusal to report is not
a pass); 2 — tool error (missing journal, filesystem failure, corrupt
mainnet journal lines, a malformed harness row before the end of its
journal, harness-row schema drift, a malformed capital-events file;
argparse exits 2 natively on invalid flag values).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from .execution_tracking import (
    DEFAULT_GAP_THRESHOLD_PCT,
    check_execution_tracking,
)


def _positive_float(value: str) -> float:
    # A nonpositive threshold flags every journal — vacuous. NaN compares
    # False against everything, so it would slip past "<= 0" and disarm
    # the breach check forever (and emit non-standard NaN in --json);
    # inf never breaches either. Reject all of them at parse.
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number > 0, got {value}"
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trade_lab.paper_trading.execution_tracking_cli",
        description=(
            "Compare real mainnet execution (equity + orders) against the "
            "simulated forward-test journal. Reports divergence; does NOT "
            "take action."
        ),
    )
    parser.add_argument(
        "--real-journal",
        type=Path,
        default=Path("data/journal/cycles_mainnet.jsonl"),
        help="Mainnet execution journal (cycles_mainnet.jsonl).",
    )
    parser.add_argument(
        "--sim-journal",
        type=Path,
        default=Path("paper_trading/logs/journal.jsonl"),
        help="Harness (simulation) journal.",
    )
    parser.add_argument(
        "--gap-threshold-pct",
        type=_positive_float,
        default=DEFAULT_GAP_THRESHOLD_PCT,
        help=(
            "Alert when |real-vs-sim level gap| exceeds this many percent "
            f"(default {DEFAULT_GAP_THRESHOLD_PCT} — an owner-adjustable "
            "starting point, not a calibrated bound)."
        ),
    )
    parser.add_argument(
        "--capital-events",
        type=Path,
        default=None,
        help=(
            "JSON file declaring deposits / withdrawals: a list of "
            '{"date": "YYYY-MM-DD", "amount_usd": <signed>, "note": ...}. '
            "The date is the first signal date whose equity reading "
            "already includes the transfer. Capital flows are operator "
            "knowledge — undeclared ones suppress the gap number instead "
            "of being reported as performance."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full TrackingReport as JSON instead of the human summary.",
    )
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help=(
            "Exit 1 when the tracking threshold is breached, or when an "
            "unexplained equity move suppressed the gap number. Default "
            "stays exit 0 — descriptive, not normative."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = check_execution_tracking(
            real_journal=args.real_journal,
            sim_journal=args.sim_journal,
            gap_threshold_pct=args.gap_threshold_pct,
            capital_events_path=args.capital_events,
        )
    except OSError as exc:
        # Not just FileNotFoundError: IsADirectoryError / PermissionError must
        # exit 2 too, or they'd surface as 1 — the --fail-on-breach code.
        print(f"TRACKING ERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        # ValueError: non-positive or non-numeric equity, corrupt mainnet
        # journal lines, a malformed mid-journal harness row, harness-row
        # schema drift (incl. an unrecoverable equity phase), a malformed
        # capital-events declaration. TypeError: a HarnessLogRow that
        # slipped past the per-row wrap — still a tool error, never exit 1
        # (the breach code).
        print(f"TRACKING ERROR: {exc}", file=sys.stderr)
        return 2

    # A suppressed gap number is not a pass: the layer could not do its
    # job, and a cron that only watches the exit code must see it.
    flagged = report.equity.breached or bool(report.equity.unexplained_moves)
    rc = 1 if (args.fail_on_breach and flagged) else 0

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
        return rc

    eq = report.equity
    print(f"Real journal: {report.real_journal}  ({eq.n_real_days} equity days)")
    print(f"Sim journal:  {report.sim_journal}  ({eq.n_sim_days} equity days)")
    if report.capital_events_file:
        print(f"Capital events: {report.capital_events_file}")
    print(f"Aligned days: {eq.n_aligned_days}  overlap={eq.overlap}")
    print(f"Tracking starts at live coverage: {eq.tracking_start}\n")

    # Phase, not just date: real equity_usd is read before the date's
    # orders, so the harness value is used before its turnover cost too.
    print("  equity tracking (both curves PRE-trade: real read-phase")
    print("  equity_usd vs harness equity before its simulated cost):")
    for f in eq.capital_flows_applied:
        print(
            f"    capital flow {f['amount_usd']:+.2f} USD declared on "
            f"{f['date']} — removed from the {f['from_date']}..{f['date']} "
            "return"
        )
    for m in eq.unexplained_moves:
        print(
            f"    UNEXPLAINED EQUITY MOVE: {m['from_date']}..{m['date']} "
            f"real {m['real_equity_from']:.2f} → {m['real_equity']:.2f} "
            f"({m['real_return'] * 100:+.1f}% after declared flows) while "
            f"the sim returned {m['sim_return'] * 100:+.1f}% — a deposit or "
            "withdrawal is not a return; declare it with --capital-events"
        )
    if eq.cum_abs_return_diff is not None:
        print(f"    cum |Δdaily-return| = {eq.cum_abs_return_diff:.4f}")
    if eq.level_gap_pct is not None:
        print(f"    level gap = {eq.level_gap_pct:+.2f}%")
    elif eq.unexplained_moves:
        print("    level gap = NOT REPORTED (unexplained equity move)")
    print(f"    threshold = ±{eq.threshold_pct:.2f}%")
    print(f"    breached = {eq.breached}")
    print()

    tr = report.transitions
    print("  per-symbol trades (sim-expected vs real):")
    print(f"    sim coverage = {tr.coverage}")
    print(f"    real fill events = {tr.n_real_fill_events}")
    print(
        f"    sim trade dates = {tr.n_sim_trade_dates} "
        f"(intended trades = {tr.n_sim_intended_trades})"
    )
    print(f"    live min-notional skips (legitimate) = {tr.n_min_notional_skips}")
    print(f"    trades covered by skips = {tr.n_trades_skip_covered}")
    for m in tr.missing_trades:
        print(
            f"    MISSING TRADE: {m['date']} {m['symbol']} "
            f"expected {m['expected_side']}, no fill and no covering skip"
        )
    for m in tr.wrong_direction_trades:
        print(
            f"    WRONG DIRECTION: {m['date']} {m['symbol']} "
            f"expected {m['expected_side']}, got {m['actual_side']}"
        )
    for m in tr.partial_fills:
        print(
            f"    PARTIAL FILL: {m['date']} {m['symbol']} {m['side']} "
            f"filled {m['filled_amount']} of {m['intended_amount']} "
            f"({m['client_order_id']})"
        )
    for m in tr.size_mismatches:
        print(
            f"    SIZE MISMATCH: {m['date']} {m['symbol']} {m['side']} "
            f"filled {m['filled_notional_quote']:.2f} quote = "
            f"{m['actual_weight_delta']:.4f} of the book, sim intended "
            f"{m['expected_weight_delta']:.4f} (x{m['ratio']:.2f})"
        )
    for m in tr.extra_opposite_fills:
        print(
            f"    EXTRA OPPOSITE FILL: {m['date']} {m['symbol']} "
            f"{m['actual_side']} {m['filled_notional_quote']:.2f} quote "
            f"({m['client_order_id']}) beside the expected "
            f"{m['expected_side']} — the leg was round-tripped"
        )
    for e in tr.unexpected_orders:
        print(
            f"    UNEXPECTED ORDER: {e['date']} {e['symbol']} "
            f"{e['side']} {e['filled_notional_quote']:.2f} quote "
            f"(cycle {e['cycle_id']}) — no sim-intended trade"
        )
    for e in tr.bootstrap_orders:
        # Not a mismatch: real execution's first trades catching up to a
        # position the simulation had already stepped into.
        print(
            f"    BOOTSTRAP ORDER: {e['date']} {e['symbol']} {e['side']} "
            f"{e['filled_notional_quote']:.2f} quote (cycle {e['cycle_id']}) "
            f"— first real fills, catching up to the sim intent from "
            f"{e['sim_intent_date']}"
        )
    for e in tr.wrong_market_fills:
        print(
            f"    WRONG MARKET: {e['date']} {e['symbol']} {e['side']} "
            f"{e['filled_notional_quote']:.2f} quote (cycle {e['cycle_id']}) "
            f"— simulation trades only /{e['expected_quote']}"
        )
    for m in tr.pre_live_sim_trades:
        # Not a mismatch: no live cycle existed on that date yet.
        print(
            f"    BEFORE LIVE COVERAGE: {m['date']} {m['symbol']} "
            f"sim intended {m['expected_side']} — no real cycle to execute it"
        )
    for e in tr.out_of_coverage_fills:
        # Not a mismatch: the harness never logged this date, so it holds
        # no expectation at all (staggered start / partial retention).
        print(
            f"    OUTSIDE SIM COVERAGE: {e['date']} {e['symbol']} "
            f"{e['side']} {e['filled_notional_quote']:.2f} quote "
            f"(cycle {e['cycle_id']}) — no harness row for this date"
        )
    if report.real_unknown_version_lines:
        print(
            f"\n  WARNING: {report.real_unknown_version_lines} mainnet "
            "journal line(s) with unknown schema_version skipped — "
            "incomplete data."
        )
    print(f"\nADVISORY: {report.advisory}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
