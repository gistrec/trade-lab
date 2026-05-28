"""Append-only JSON Lines journal for execution cycles.

Each cycle of the bot's main loop — dry-run today, real orders after
phase #2b — emits exactly one JSON object on its own line. The reader
side lives in :mod:`trade_lab.monitoring.data_source` and never touches
this module; the bot writes, monitoring reads.

Schema versioning
=================
``schema_version=1`` is the dry-run shape: signal + balance + planned
orders, no fills. Phase #2b will introduce ``schema_version=2`` with
order results and fill events. Readers must accept all known versions
and skip unknown ones with a warning — never crash.

Atomicity
=========
Each line is written by a single ``write()`` syscall on an O_APPEND
file, followed by ``fsync``. POSIX guarantees writes of up to
``PIPE_BUF`` bytes are atomic on append; on Linux and macOS that's
4096. The :data:`MAX_LINE_BYTES` cap enforces this. A crash mid-write
can still leave a truncated final line: the reader handles this case
by skipping any line that fails to parse as JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


JOURNAL_SCHEMA_VERSION = 1
MAX_LINE_BYTES = 4096


class JournalEntryTooLarge(RuntimeError):
    """Raised when a serialized cycle exceeds :data:`MAX_LINE_BYTES`.

    Past the 4KB cap a single ``write`` may be split by the kernel,
    breaking the per-line atomicity guarantee. The caller must trim
    the payload (most commonly: shorten ``basket_close_series.values``).
    """


@dataclass
class Cycle:
    """One execution-loop cycle, serializable to a single JSON line.

    Fields that depend on a successful read (``signal``, ``balance``,
    ...) are ``Optional`` so a failed cycle still produces a valid
    journal entry. The journal must be append-only even on partial
    failures — silently skipping failed cycles would hide the very
    incidents monitoring exists to surface.
    """

    cycle_id: str
    started_at: str             # ISO-8601 UTC
    ended_at: str               # ISO-8601 UTC
    duration_ms: int
    outcome: str                # "success" | "failed"
    error: Optional[dict]       # {"type": ..., "message": ...} or None
    git_commit: Optional[str]
    python_version: str
    context: dict
    signal: Optional[dict]
    basket_close_series: Optional[dict]
    balance: Optional[dict]
    equity_usd: Optional[float]
    target_allocation: Optional[dict]
    current_holdings_quote: Optional[dict]
    orders_planned: Optional[list]
    orders_skipped: Optional[list]
    total_skipped_quote_drift: Optional[float]
    schema_version: int = JOURNAL_SCHEMA_VERSION


class JournalWriter:
    """Append-only writer for :class:`Cycle` records.

    Open/close per call to keep the writer crash-safe — a stale file
    handle across a restart can desynchronize from what's on disk.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, cycle: Cycle) -> None:
        """Serialize and append one cycle. Raises if oversized."""
        encoded = _encode_cycle(cycle)
        if len(encoded) > MAX_LINE_BYTES:
            raise JournalEntryTooLarge(
                f"Cycle {cycle.cycle_id} serializes to {len(encoded)} bytes "
                f"(limit {MAX_LINE_BYTES}). Trim the payload "
                "(most commonly: shorten basket_close_series.values)."
            )
        # If a previous writer crashed mid-write and left no trailing
        # newline, prepend one so this entry lands on its own line.
        # Without this, a partial tail would silently eat the next
        # valid entry's leading bytes.
        prefix = b"\n" if self._needs_leading_newline() else b""
        with open(self.path, "ab") as f:
            f.write(prefix + encoded)
            f.flush()
            os.fsync(f.fileno())

    def _needs_leading_newline(self) -> bool:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        if size == 0:
            return False
        with open(self.path, "rb") as f:
            f.seek(-1, 2)
            return f.read(1) != b"\n"


def _encode_cycle(cycle: Cycle) -> bytes:
    """Serialize a cycle to a UTF-8 JSON line ending in newline."""
    data = asdict(cycle)
    line = json.dumps(data, separators=(",", ":"), default=str)
    return (line + "\n").encode("utf-8")


def get_git_commit_short() -> Optional[str]:
    """Return the short git SHA of HEAD, or ``None`` if unavailable.

    Resolved at write time, not import time, so a restart after a
    fast-forward picks up the new commit without restarting the bot.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    sha = out.decode().strip()
    return sha or None


def get_python_version() -> str:
    """Return ``major.minor.micro`` of the current interpreter."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def new_cycle_id() -> str:
    """Generate a new cycle UUID4 as a string."""
    return str(uuid.uuid4())


def utcnow_iso() -> str:
    """Return current UTC time as an ISO-8601 string with timezone."""
    return datetime.now(timezone.utc).isoformat()
