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

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


class JournalCorruptionError(RuntimeError):
    """A journal line that is neither valid nor the tolerated torn tail.

    Its own class, not a bare ValueError: the harness CLI maps declared
    failures to exit code 2, and cron integrations distinguish that from
    an uncaught traceback (exit 1). A corrupt journal is a declared
    failure — the operator must repair it — so it has to arrive as one.
    """


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


def _repair_unterminated_tail(log_path: Path) -> None:
    """Make the file end on a line boundary before an append.

    A crash between the write and its newline leaves a partial object
    with no terminator. Appending straight onto it fuses two records
    into ONE invalid line: ``read_log`` tolerates a broken final line,
    so the NEW row vanishes while the cycle reports success — and every
    later cycle repeats it. (Reproduced before fixing: 06-01 written,
    torn 06-02, then 06-03 appended → the journal reads back as 06-01
    alone.)

    Two different tails, two different repairs:

    * the tail parses as JSON — the record is complete and only its
      newline is missing (a hand-edit, an editor stripping the final
      newline). Terminate it and keep the row.
    * the tail does not parse — it is a torn write. Those bytes are
      already invisible to every reader, and keeping them would leave a
      permanently invalid line in the MIDDLE of the file, which
      ``read_log`` now refuses outright. Drop them, restoring the
      invariant that every line is a valid record.
    """
    try:
        size = log_path.stat().st_size
    except FileNotFoundError:
        return
    if size == 0:
        return
    with open(log_path, "rb") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return
    raw = log_path.read_bytes()
    cut = raw.rfind(b"\n") + 1          # 0 when the file has no newline at all
    tail = raw[cut:]
    try:
        json.loads(tail.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        with open(log_path, "r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
        return
    with open(log_path, "ab") as f:     # complete record, just unterminated
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _journal_lock(log_path: Path):
    """Serialize repair+append across processes.

    The repair is DESTRUCTIVE (it can truncate a torn tail), so it must
    not race the append. Without the lock, a cron run and a manual
    ``--asof`` backfill that both observe the same torn tail compute the
    same cut: one truncates and appends its row, the other then executes
    its stale truncate and deletes that row before writing its own.
    Advisory ``flock`` on a sidecar file — a lock on the journal itself
    would have to be taken before deciding whether to open it ``r+b``.
    """
    lock_path = log_path.with_suffix(log_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def append_row(row: HarnessLogRow, log_path: Path) -> None:
    """Atomic append-only JSONL write.

    On POSIX, opening with ``"a"`` + writing a single ``write`` call
    and ``fsync`` is sufficient for crash-safe append-only behaviour.
    An unterminated previous write is repaired first (see
    :func:`_repair_unterminated_tail`); repair and append run under one
    lock, so power loss costs at most the row that was mid-flight and a
    concurrent run cannot delete a row the other just wrote.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(row), separators=(",", ":")) + "\n"
    with _journal_lock(log_path):
        _repair_unterminated_tail(log_path)
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
    text = log_path.read_text(encoding="utf-8")
    # A torn append cannot have written its own terminator, so a malformed
    # last line is only forgivable when the file does NOT end on a line
    # boundary. Malformed AND newline-terminated means the write finished
    # and the bytes rotted afterwards — that is corruption, not a crash.
    unterminated_tail = bool(text) and not text.endswith("\n")
    raw = [
        (n, s) for n, s in
        ((n, line.strip()) for n, line in enumerate(
            text.splitlines(), start=1))
        if s
    ]
    rows: list[HarnessLogRow] = []
    for idx, (lineno, line) in enumerate(raw):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == len(raw) - 1 and unterminated_tail:
                continue          # truncated final append; never completed
            raise JournalCorruptionError(
                f"{log_path}: line {lineno} of {len(raw)} is not valid JSON "
                f"({exc}). This is NOT the crash-mid-write case — a corrupt "
                f"line in the middle means history was lost. Repair the "
                f"journal; the monitors must not run on a holed series."
            ) from exc
        try:
            rows.append(HarnessLogRow(**data))
        except TypeError as exc:
            # Valid JSON, wrong shape: a missing field, an extra one, or a
            # non-object top level. Left as TypeError it escapes both CLIs'
            # declared-corruption handling and exits 1 — which the monitor
            # also uses for --fail-on-breach.
            raise JournalCorruptionError(
                f"{log_path}: line {lineno} does not match HarnessLogRow "
                f"({exc}). Repair the journal."
            ) from exc
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
