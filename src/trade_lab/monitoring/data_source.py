"""Read-only consumer of the bot's JSON Lines journal.

This module knows the journal's schema by structural agreement, not by
import: it does not pull from ``trade_lab.execution`` because the
monitoring process runs without rights to that layer's environment
(API keys, broker objects).

Robustness rules
================
* A corrupted JSON line is skipped and counted. The reader does not
  raise — a single bad line must not blind the dashboard to everything
  written before or after it.
* Unknown ``schema_version`` is skipped and counted separately. When
  phase #2b ships schema_version=2 and the monitoring is still on
  pre-#2b code, the operator sees a "N unknown-version entries" hint
  in the dashboard rather than a crash.
* The reader caches by file mtime, so repeated Streamlit reruns do not
  re-parse the journal every refresh tick.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


KNOWN_SCHEMA_VERSIONS = frozenset({1, 2})


class Staleness(str, Enum):
    """Bucketed answer to: how recent is the last cycle?"""

    NO_DATA = "no_data"   # journal absent or empty
    FRESH = "fresh"       # last cycle within 1.5× expected interval
    STALE = "stale"       # 1.5× ≤ elapsed < 10× expected interval
    DOWN = "down"         # >10× expected interval


@dataclass
class ReadStats:
    """Cumulative read counters for a single journal scan."""

    total_lines: int = 0
    valid_cycles: int = 0
    corrupt_lines: int = 0
    unknown_version_lines: int = 0


def cycle_orders_executed(cycle: dict) -> list:
    """Return ``orders_executed`` list, or empty for v1 entries.

    Schema v1 cycles predate ``orders_executed``; this helper returns
    ``[]`` so callers do not need to special-case ``schema_version``.
    Also normalizes the explicit ``None`` written for dry-run cycles
    and failed cycles with no partial fills.
    """
    return cycle.get("orders_executed") or []


def parse_iso(s) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp written by the journal; total function.

    Returns ``None`` for anything unparseable — JSON ``null``, non-string
    values, malformed strings. The journal is external input to this
    process; a single bad field must degrade one value, not raise an
    AttributeError that takes the dashboard down. Naive timestamps are
    treated as UTC: the writer always emits an offset, but defensive
    coding keeps a future writer regression from silently producing
    wrong wall-clock comparisons.
    """
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class JournalReader:
    """Cached, fail-soft reader for a journal file.

    Construction does not touch the file system. Each query method
    refreshes from disk if the file's ``mtime`` changed, otherwise
    serves cached data.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._mtime: Optional[float] = None
        self._size: Optional[int] = None
        self._cache: list[dict] = []
        self._stats: ReadStats = ReadStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def latest_cycle(self) -> Optional[dict]:
        """Return the most recently appended valid cycle, or None."""
        self._refresh_if_changed()
        return self._cache[-1] if self._cache else None

    def cycles(self, n: int = 20) -> list[dict]:
        """Return up to ``n`` most recent valid cycles, newest last."""
        if n <= 0:
            return []
        self._refresh_if_changed()
        return list(self._cache[-n:])

    def signal_history(
        self, days: int = 30,
    ) -> list[tuple[datetime, float, bool]]:
        """Return (asof, ladder_value, sma_gate_open) for charts.

        Failed cycles and cycles older than ``days`` are excluded.
        """
        self._refresh_if_changed()
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400.0
        out: list[tuple[datetime, float, bool]] = []
        for c in self._cache:
            sig = c.get("signal")
            if sig is None:
                continue
            asof = parse_iso(sig.get("asof"))
            if asof is None:
                continue
            if asof.timestamp() < cutoff:
                continue
            out.append((
                asof,
                float(sig.get("ladder_value", 0.0)),
                bool(sig.get("sma_gate_open", False)),
            ))
        return out

    def staleness(self, expected_interval_s: float) -> Staleness:
        """Bucket the last cycle's age into NO_DATA/FRESH/STALE/DOWN."""
        self._refresh_if_changed()
        if not self._cache:
            return Staleness.NO_DATA
        last = self._cache[-1]
        ended = parse_iso(last.get("ended_at"))
        if ended is None:
            return Staleness.NO_DATA
        elapsed = datetime.now(timezone.utc).timestamp() - ended.timestamp()
        if elapsed < expected_interval_s * 1.5:
            return Staleness.FRESH
        if elapsed < expected_interval_s * 10:
            return Staleness.STALE
        return Staleness.DOWN

    def cumulative_skipped_drift(self) -> float:
        """Sum of ``total_skipped_quote_drift`` across all cycles."""
        self._refresh_if_changed()
        return sum(
            float(c.get("total_skipped_quote_drift") or 0.0)
            for c in self._cache
        )

    def stats(self) -> ReadStats:
        """Counters from the most recent file scan."""
        self._refresh_if_changed()
        return self._stats

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _refresh_if_changed(self) -> None:
        """Re-read the file if mtime or size changed since last scan."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            if self._mtime is not None:
                # File was deleted between scans; clear caches.
                self._mtime = None
                self._size = None
                self._cache = []
                self._stats = ReadStats()
            return
        if st.st_mtime == self._mtime and st.st_size == self._size:
            return
        self._mtime = st.st_mtime
        self._size = st.st_size
        self._read_all()

    def _read_all(self) -> None:
        """Scan the file end-to-end, populate ``_cache`` and ``_stats``."""
        stats = ReadStats()
        cache: list[dict] = []
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            self._cache = []
            self._stats = stats
            return
        for line_bytes in raw.split(b"\n"):
            if not line_bytes.strip():
                continue
            stats.total_lines += 1
            try:
                obj = json.loads(line_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                stats.corrupt_lines += 1
                continue
            if not isinstance(obj, dict):
                stats.corrupt_lines += 1
                continue
            version = obj.get("schema_version")
            if version not in KNOWN_SCHEMA_VERSIONS:
                stats.unknown_version_lines += 1
                logger.warning(
                    "Skipping journal entry with unknown schema_version=%r",
                    version,
                )
                continue
            stats.valid_cycles += 1
            cache.append(obj)
        self._cache = cache
        self._stats = stats
