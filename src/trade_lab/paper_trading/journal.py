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
``date``                       ISO YYYY-MM-DD (UTC *signal date* — the
                               last completed daily bar the signal was
                               computed on: ``asof`` for a backfill,
                               yesterday for a same-day run, since the
                               bar stamped today is still forming)
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
from dataclasses import asdict, dataclass
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
    """Read the journal and return rows.

    Only a corrupt FINAL line is tolerated — that is the crash-mid-write
    case :func:`append_row` documents, and the row was never completed.
    A corrupt line anywhere else means bytes were lost from history
    (disk fault, a merge conflict, an editor), and dropping it silently
    is worse than failing: the fingerprint monitor would compute flip
    and drawdown streaks over a series with an invisible hole, and the
    look-ahead detector would simply never replay the missing days.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    raw = [
        (n, s) for n, s in
        ((n, line.strip()) for n, line in enumerate(
            log_path.read_text(encoding="utf-8").splitlines(), start=1))
        if s
    ]
    rows: list[HarnessLogRow] = []
    for idx, (lineno, line) in enumerate(raw):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == len(raw) - 1:
                continue          # truncated final append; never completed
            raise ValueError(
                f"{log_path}: line {lineno} of {len(raw)} is not valid JSON "
                f"({exc}). This is NOT the crash-mid-write case — a corrupt "
                f"line in the middle means history was lost. Repair the "
                f"journal; the monitors must not run on a holed series."
            ) from exc
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
