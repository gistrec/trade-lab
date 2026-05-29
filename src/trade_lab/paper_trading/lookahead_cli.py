"""CLI for the live look-ahead detector (Part B).

Replays every journal row's vintage and compares against the
harness-logged signal. Reports match / offset / random-disagreement
classification.

This is descriptive — it never auto-kills. Even a "LOOK-AHEAD SUSPECT"
advisory is forwarded to the operator + Part A re-run, not acted on
automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .lookahead_detector import check_journal_for_lookahead


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trade_lab.paper_trading.lookahead_cli",
        description=(
            "Replay the backtest signal on each journal row's vintage "
            "bytes and classify any disagreement as match / offset / "
            "random. Descriptive; never auto-kills the strategy."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("paper_trading/logs/journal.jsonl"),
    )
    parser.add_argument(
        "--vintage-root",
        type=Path,
        default=Path("paper_trading/vintages"),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full report as JSON instead of the human summary.",
    )
    args = parser.parse_args(argv)

    try:
        report = check_journal_for_lookahead(
            log_path=args.log_path,
            vintage_root=args.vintage_root,
        )
    except Exception as exc:
        print(f"DETECTOR ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
        return 0

    print(f"Journal rows checked: {report.n_checked} / {report.journal_n_rows}")
    print(f"  match              : {report.n_match}")
    print(f"  offset_1_match     : {report.n_offset_1_match}")
    print(f"  random_disagreement: {report.n_random_disagreement}")
    print(f"  vintage_missing    : {report.n_vintage_missing}")
    print(f"  constant_offset_pattern    : {report.constant_offset_pattern}")
    print(f"  random_disagreement_present: {report.random_disagreement_present}")
    print(f"\nADVISORY: {report.advisory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
