"""CLI entry point for the validation forward-test harness.

Typical cron invocation::

    cd /path/to/trade-lab
    .venv/bin/python -m trade_lab.paper_trading.cli \\
        --log-path paper_trading/logs/journal.jsonl \\
        --vintage-root paper_trading/vintages

Idempotent: re-running within the same UTC day returns the previously
recorded row without writing a duplicate.

Exit codes:
* 0 — cycle written (or no-op idempotent return).
* 2 — ``HarnessError`` (config hash drift, fetch failure, empty
  basket, etc.). Cron should surface this to the operator.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

from .harness import HarnessError, run_paper_trading_cycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trade_lab.paper_trading.cli",
        description=(
            "Validation forward-test harness for the frozen TSMOM(28,60) + "
            "SMA(200) basket strategy. Runs one daily cycle: fetches "
            "candles, snapshots them immutably, computes the signal, and "
            "writes a structured journal row."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("paper_trading/logs/journal.jsonl"),
        help="Append-only JSONL journal path.",
    )
    parser.add_argument(
        "--vintage-root",
        type=Path,
        default=Path("paper_trading/vintages"),
        help="Root directory for content-hashed OHLCV snapshots.",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="ISO date (YYYY-MM-DD) of the cycle. Default: today UTC.",
    )
    parser.add_argument(
        "--candles-per-asset",
        type=int,
        default=400,
        help="Trailing bars to fetch per asset (default 400 — covers "
             "SMA(200) warmup with margin).",
    )
    args = parser.parse_args(argv)

    asof = _date.fromisoformat(args.asof) if args.asof else None
    try:
        row = run_paper_trading_cycle(
            log_path=args.log_path,
            vintage_root=args.vintage_root,
            asof=asof,
            candles_per_asset=args.candles_per_asset,
        )
    except HarnessError as exc:
        print(f"HARNESS ERROR: {exc}", file=sys.stderr)
        return 2

    gate = "OPEN" if row.sma_gate_open else "CLOSED"
    print(
        f"OK date={row.date} ladder={row.ladder_state:.2f} sma_gate={gate} "
        f"vintage={row.vintage_content_hash[:12]}... "
        f"equity=${row.portfolio_equity:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
