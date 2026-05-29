"""Append-only structured journal for the validation forward-test.

A journal row is one cycle of the harness — what the strategy saw,
what it decided, what it would have traded, and the content-hash of
the data it saw. Rows are written one per UTC date in JSON-Lines
(JSONL) so they are streamable and easy to diff.

Idempotency contract
====================
The harness loop is designed so that re-running on the same UTC date
is a no-op at every layer:
* Vintage store: hash-addressed, write skipped if file exists.
* Journal: ``is_already_logged(date)`` returns True; the harness
  returns the previously-written row without appending.

The 'append-only' guarantee means rows are never edited in place
once written. A look-ahead detector (Test 4) reads them as immutable
history. A schema migration would write rows in a new shape going
forward; old rows stay as they were.

Row schema (v1)
===============
``date``                       ISO YYYY-MM-DD (UTC date of the cycle)
``config_hash``                ``CANONICAL_HASH`` at write time
``vintage_content_hash``       SHA-256 of the OHLCV bytes used
``basket_close``               float — basket index close at as-of
``sma_value``                  float | None — SMA(sma_period) value
``sma_gate_open``              bool — close > SMA(period)
``ladder_state``               float in {0.0, 0.5, 1.0}
``prior_ladder_state``         float (yesterday's, 0.0 on bootstrap)
``per_lookback_states``        {"28": 0|1, "60": 0|1}
``per_lookback_returns``       {"28": pct, "60": pct}
``target_weights``             {asset: 1/N × ladder}
``current_weights``            {asset: prior held weight}
``intended_trades``            {asset: target_weight - current_weight}
``portfolio_equity``           float — virtual USD equity start of cycle
``daily_return``               float — basket pct_change since prior cycle
``gross_position_return``      float — prior_ladder × daily_return
``net_position_return``        float — gross minus simulated turnover cost
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HarnessLogRow:
    date: str
    config_hash: str
    vintage_content_hash: str
    basket_close: float
    sma_value: Optional[float]
    sma_gate_open: bool
    ladder_state: float
    prior_ladder_state: float
    per_lookback_states: dict     # {"28": int, "60": int}
    per_lookback_returns: dict    # {"28": float, "60": float}
    target_weights: dict          # {asset: float}
    current_weights: dict         # {asset: float}
    intended_trades: dict         # {asset: float}
    portfolio_equity: float
    daily_return: float
    gross_position_return: float
    net_position_return: float
    notes: str = ""


def append_row(row: HarnessLogRow, log_path: Path) -> None:
    """Atomic append-only JSONL write.

    On POSIX, opening with ``"a"`` + writing a single ``write`` call
    and ``fsync`` is sufficient for crash-safe append-only behaviour.
    The journal is line-buffered by design — partial-write corruption
    on power loss would leave at most one truncated row that
    ``read_log`` will skip via JSON-decode error handling.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(row), separators=(",", ":")) + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_log(log_path: Path) -> list[HarnessLogRow]:
    """Read the journal and return rows. Partial / corrupted final
    line is skipped silently (see crash-safety note in
    :func:`append_row`).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    rows: list[HarnessLogRow] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Last line truncated by a crash mid-write; drop it.
                continue
            rows.append(HarnessLogRow(**data))
    return rows


def is_already_logged(date_str: str, log_path: Path) -> bool:
    """Idempotency check: True iff the journal contains a row for this date."""
    for row in read_log(log_path):
        if row.date == date_str:
            return True
    return False


def get_row_for_date(date_str: str, log_path: Path) -> Optional[HarnessLogRow]:
    """Return the row for ``date_str`` if present (last one wins on duplicates)."""
    found: Optional[HarnessLogRow] = None
    for row in read_log(log_path):
        if row.date == date_str:
            found = row
    return found
