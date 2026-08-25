"""CLI for the execution-tracking layer (issue #11).

Typical operator invocation (manual or cron)::

    .venv/bin/python -m trade_lab.paper_trading.execution_tracking_cli

Exit codes — same contract as ``fingerprint_cli``: 0 — report produced
(default even on breach: descriptive, not normative); 1 — tracking
threshold breached AND ``--fail-on-breach`` was passed; 2 — tool error
(missing journal, filesystem failure; argparse exits 2 natively on
invalid flag values).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .execution_tracking import (
    DEFAULT_GAP_THRESHOLD_PCT,
    check_execution_tracking,
)


def _positive_float(value: str) -> float:
    # A nonpositive threshold flags every journal — vacuous, reject at parse.
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
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
        "--json",
        action="store_true",
        help="Emit the full TrackingReport as JSON instead of the human summary.",
    )
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help=(
            "Exit 1 when the tracking threshold is breached. Default stays "
            "exit 0 — descriptive, not normative."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = check_execution_tracking(
            real_journal=args.real_journal,
            sim_journal=args.sim_journal,
            gap_threshold_pct=args.gap_threshold_pct,
        )
    except OSError as exc:
        # Not just FileNotFoundError: IsADirectoryError / PermissionError must
        # exit 2 too, or they'd surface as 1 — the --fail-on-breach code.
        print(f"TRACKING ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # non-positive equity on an aligned date, etc.
        print(f"TRACKING ERROR: {exc}", file=sys.stderr)
        return 2

    rc = 1 if (args.fail_on_breach and report.equity.breached) else 0

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
        return rc

    eq = report.equity
    print(f"Real journal: {report.real_journal}  ({eq.n_real_days} equity days)")
    print(f"Sim journal:  {report.sim_journal}  ({eq.n_sim_days} equity days)")
    print(f"Aligned days: {eq.n_aligned_days}  overlap={eq.overlap}\n")

    print("  equity tracking:")
    if eq.cum_abs_return_diff is not None:
        print(f"    cum |Δdaily-return| = {eq.cum_abs_return_diff:.4f}")
    if eq.level_gap_pct is not None:
        print(f"    level gap = {eq.level_gap_pct:+.2f}%")
    print(f"    threshold = ±{eq.threshold_pct:.2f}%")
    print(f"    breached = {eq.breached}")
    print()

    tr = report.transitions
    print("  position transitions:")
    print(f"    real order events = {tr.n_real_order_events}")
    print(f"    ladder transitions = {tr.n_ladder_transitions}")
    print(f"    min-notional skips (legitimate) = {tr.n_min_notional_skips}")
    print(f"    transitions covered by skips = {tr.n_transitions_skip_covered}")
    for e in tr.orders_without_transition:
        print(
            f"    ORDER WITHOUT TRANSITION: {e['date']} {e['symbol']} "
            f"{e['side']} {e['filled_notional_quote']:.2f} quote "
            f"(cycle {e['cycle_id']})"
        )
    for t in tr.transitions_without_order:
        print(
            f"    TRANSITION WITHOUT ORDER: {t['date']} "
            f"{t['prior']} -> {t['new']}"
        )
    print(f"\nADVISORY: {report.advisory}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
