"""CLI for the behavioral monitor.

Typical operator invocation (manual or cron)::

    .venv/bin/python -m trade_lab.paper_trading.fingerprint_cli

Reports breach status of the live journal against the frozen
reference fingerprint. Exit code is **always 0** unless a real
error (missing files, hash mismatch, etc.) occurred — the monitor
is descriptive, not normative, and is not allowed to auto-fail the
strategy.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .fingerprint_monitor import check_journal_against_reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trade_lab.paper_trading.fingerprint_cli",
        description=(
            "Compare the live forward-test journal against the frozen "
            "reference behavioral fingerprint. Reports breaches; does "
            "NOT take action."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("paper_trading/logs/journal.jsonl"),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("paper_trading/fingerprint/reference_fingerprint.json"),
    )
    parser.add_argument(
        "--sustained-days",
        type=int,
        default=7,
        help="Minimum consecutive days of single-metric breach to flag as 'sustained'.",
    )
    parser.add_argument(
        "--multi-metric-threshold",
        type=int,
        default=3,
        help="Minimum number of metrics breached on the same day to flag as 'multi-metric'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full BreachReport as JSON instead of the human summary.",
    )
    args = parser.parse_args(argv)

    try:
        report = check_journal_against_reference(
            log_path=args.log_path,
            reference_path=args.reference_path,
            sustained_days_threshold=args.sustained_days,
            multi_metric_threshold=args.multi_metric_threshold,
        )
    except FileNotFoundError as exc:
        print(f"MONITOR ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # content-hash mismatch on reference, etc.
        print(f"MONITOR ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
        return 0

    # Human-readable summary
    print(f"Journal: {report.journal_n_rows} rows  ({report.journal_window})")
    print(f"Reference window: {report.reference_window}  hash={report.reference_content_hash[:12]}...")
    print(f"Rolling window: {report.rolling_window_days} days\n")

    for status in (report.exposure_flip, report.regime_gate_flip):
        latest = (
            f"{status.latest_value:.2f}" if status.latest_value is not None else "n/a"
        )
        print(f"  {status.name}:")
        print(f"    bands [p05, p95] = [{status.p05:.2f}, {status.p95:.2f}]")
        print(f"    latest = {latest}")
        print(f"    currently_breached = {status.currently_breached}")
        print(f"    consecutive_breach_days_now = {status.currently_consecutive_breach}")
        print(f"    longest_run_observed = {status.longest_consecutive_breach}")
        print()

    dd = report.drawdown
    print("  drawdown:")
    print(f"    latest_dd = {dd.latest_drawdown*100:+.2f}%")
    print(f"    max_live_dd = {dd.max_live_drawdown*100:+.2f}%")
    print(f"    reference max_historical_dd = {dd.reference_max_historical_dd*100:+.2f}%")
    print(f"    headroom = {dd.headroom_pp:+.2f} pp")
    print(f"    breached = {dd.breached}")
    print()

    rb = report.rebalance_turnover_per_event
    print("  rebalance_turnover_per_event:")
    print(f"    n_live_events = {rb['n_live_events']}")
    print(f"    events_outside_p05_p95 = {rb['events_outside_reference_p05_p95']}")
    print()

    print(f"Multi-metric days: {report.multi_metric_days}")
    print(f"Sustained breach: {report.overall_sustained_breach}")
    print(f"Multi-metric breach: {report.overall_multi_metric_breach}")
    print(f"\nADVISORY: {report.advisory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
