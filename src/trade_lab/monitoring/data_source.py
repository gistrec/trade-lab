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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


KNOWN_SCHEMA_VERSIONS = frozenset({1, 2})

# Staleness thresholds, as multiples of the expected cycle interval. Single
# source of truth: staleness() decides with these, and the operator-facing
# messages in app.py import the SAME constants so the number shown always
# equals the number decided on.
STALE_MULTIPLIER = 1.5
DOWN_MULTIPLIER = 10.0


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
    # Set when the journal exists but could not be read (permission/OS error,
    # a directory at the path). The whole deployment hinges on the monitoring
    # user having exactly read access — this is the failure the README's
    # permission steps warn about, so it must be surfaced, not swallowed.
    read_error: Optional[str] = None


def cycle_orders_executed(cycle: dict) -> list:
    """Return ``orders_executed`` list, or empty for v1 entries.

    Schema v1 cycles predate ``orders_executed``; this helper returns
    ``[]`` so callers do not need to special-case ``schema_version``.
    Also normalizes the explicit ``None`` written for dry-run cycles
    and failed cycles with no partial fills.
    """
    return cycle.get("orders_executed") or []


def is_live_cycle(cycle: dict) -> bool:
    """True if this cycle attempted real order placement (not a dry-run).

    ``run_live_cycle`` always writes ``orders_executed`` as a list — even an
    empty one for a signal=0 no-op or a reconstruction. Dry-run (planning-
    only) cycles write ``None`` (or omit the field on schema v1). The hourly
    dry-run and the daily live run share one journal, so this predicate is
    what separates 'the real order cron ran' from 'a dry-run kept the
    journal warm'.

    Caveat: a live cycle that raised *before* placing any order also writes
    ``orders_executed=None`` (live_cycle.py:592), so it reads as non-live
    here — but such a cycle still surfaces through its ``outcome=="failed"``
    in :func:`recent_incidents`, so nothing is lost.
    """
    return cycle.get("orders_executed") is not None


def first_live_cycle_time(cycles: list[dict]) -> Optional[datetime]:
    """``ended_at`` of the EARLIEST live cycle in the window, or ``None``.

    Marks when real order execution first ran. Before it the journal is
    dry-run-only — planning cycles with no trades, because there was no
    ``paper-place-orders`` cron yet — so equity/drift up to this point reflect
    a held balance that was never actually rebalanced. The charts draw a
    reference line here to separate that pre-execution stretch from live
    trading. Order-independent: scans all cycles and keeps the minimum
    timestamp; ``None`` if the window holds only dry-runs.
    """
    best: Optional[datetime] = None
    for c in cycles:
        if not is_live_cycle(c):
            continue
        t = parse_iso(c.get("ended_at"))
        if t is None:
            continue
        if best is None or t < best:
            best = t
    return best


# Cycle-level outcomes that mean something went wrong and wants an operator's
# eye. ``reconstructed`` is a *recovery* (an unknown-order state was resolved),
# not an incident, so it is surfaced separately, not here.
INCIDENT_OUTCOMES = frozenset({"failed", "unknown_orders", "partial"})

# Terminal order states that count as fully resolved. Anything else an
# executed order lands in (partial, rejected, lost_track, timeout, ...) is an
# open incident until a later cycle resolves it.
RESOLVED_ORDER_STATUSES = frozenset({"closed", "canceled", "cancelled"})


def recent_incidents(cycles: list[dict]) -> list[dict]:
    """Non-success cycles in the window, newest-first.

    The Status tab otherwise only reacts to the *latest* cycle's outcome, so
    a failed / unknown_orders / partial LIVE cycle from hours ago is
    overwritten in the UI by a later dry-run 'success' and never seen. This
    scans the whole window instead. ``reconstructed`` is excluded (it is a
    recovery, not a failure).
    """
    out: list[dict] = []
    for c in reversed(cycles):  # newest-first
        if c.get("outcome") in INCIDENT_OUTCOMES:
            err = c.get("error") or {}
            out.append({
                "ended_at": c.get("ended_at"),
                "outcome": str(c.get("outcome") or "?"),
                "mode": "LIVE" if is_live_cycle(c) else "DRY",
                "cycle_id": (c.get("cycle_id") or "?")[:8],
                "error_type": err.get("type") if isinstance(err, dict) else None,
                "error_message": err.get("message") if isinstance(err, dict) else None,
            })
    return out


def open_order_incidents(cycles: list[dict]) -> list[dict]:
    """Executed orders across the window not in a resolved terminal state.

    Surfaces partial / rejected / lost_track / timeout orders that would
    otherwise be reachable only by opening each cycle's detail one at a time.
    Newest-first. A window-level scan (does not attempt cross-cycle
    resolution tracking — a later 'closed' for the same id is not de-duped;
    that is a deliberate, fail-loud simplification).
    """
    out: list[dict] = []
    for c in reversed(cycles):  # newest-first
        for o in cycle_orders_executed(c):
            if not isinstance(o, dict):
                # A corrupt cycle whose orders_executed holds a non-dict must
                # degrade one entry, not raise (same totality as the reader).
                continue
            status = o.get("terminal_status")
            if status in RESOLVED_ORDER_STATUSES:
                continue
            out.append({
                "ended_at": c.get("ended_at"),
                "cycle_id": (c.get("cycle_id") or "?")[:8],
                "client_order_id": (o.get("client_order_id") or "?")[:24],
                "symbol": o.get("symbol"),
                "side": (o.get("side") or "").upper(),
                "status": str(status or "?"),
            })
    return out


def max_inter_cycle_gap_seconds(cycles: list[dict]) -> Optional[float]:
    """Largest gap (seconds) between consecutive cycles' ``ended_at``.

    ``staleness`` only inspects the newest cycle's age, so a pause in the
    *middle* of the window (the daily cron skipped a day but has fired
    since) is invisible to it. This looks at every consecutive pair.
    Returns ``None`` when fewer than two timestamps parse.
    """
    times = sorted(
        t for t in (parse_iso(c.get("ended_at")) for c in cycles) if t is not None
    )
    if len(times) < 2:
        return None
    return max((b - a).total_seconds() for a, b in zip(times, times[1:]))


# ---------------------------------------------------------------------------
# Time-series extraction for charts (Theme 2). Each is a total function over
# external journal input: a null/garbage field drops that point, never raises.
# Cycles arrive chronological (cache order), so the output is chronological.
# ---------------------------------------------------------------------------


def equity_series(cycles: list[dict]) -> list[tuple[datetime, float]]:
    """(ended_at, equity_usd) for successful cycles with a numeric equity.

    Only the latest ``equity_usd`` is shown as a scalar today; this is the
    whole history so paper equity can be charted over the observation window.
    """
    out: list[tuple[datetime, float]] = []
    for c in cycles:
        if c.get("outcome") != "success":
            continue
        dt = parse_iso(c.get("ended_at"))
        eq = c.get("equity_usd")
        if dt is None or eq is None:
            continue
        try:
            out.append((dt, float(eq)))
        except (TypeError, ValueError):
            continue
    return out


def duration_series(cycles: list[dict]) -> list[tuple[datetime, float]]:
    """(ended_at, duration_ms) for every cycle with a numeric duration.

    A trend of slowing cycles (wait-for-ack backoff, degrading network) is a
    leading indicator the single latest scalar cannot show.
    """
    out: list[tuple[datetime, float]] = []
    for c in cycles:
        dt = parse_iso(c.get("ended_at"))
        dur = c.get("duration_ms")
        if dt is None:
            continue
        try:
            out.append((dt, float(dur)))
        except (TypeError, ValueError):
            continue
    return out


def cycle_total_drift(cycle: dict) -> Optional[float]:
    """Sum of |target - current| across assets, or None if unavailable.

    The strategy deliberately lets weights drift between monthly rebalances
    (the C3 profile), so a per-cycle total-drift trend should be a sawtooth
    that resets on rebalance days — a monotonic climb would flag a problem.
    """
    target = cycle.get("target_allocation")
    current = cycle.get("current_holdings_quote")
    if not isinstance(target, dict) or not isinstance(current, dict):
        return None
    total = 0.0
    for asset in set(target) | set(current):
        total += abs(_as_float(target.get(asset)) - _as_float(current.get(asset)))
    return total


def drift_series(cycles: list[dict]) -> list[tuple[datetime, float]]:
    """(ended_at, total_drift) for successful cycles where drift is defined."""
    out: list[tuple[datetime, float]] = []
    for c in cycles:
        if c.get("outcome") != "success":
            continue
        dt = parse_iso(c.get("ended_at"))
        if dt is None:
            continue
        drift = cycle_total_drift(c)
        if drift is None:
            continue
        out.append((dt, drift))
    return out


@dataclass
class TradeEvent:
    """A successful cycle whose executed orders actually moved the book.

    ``kind`` classifies the cycle by the mix of sides among *filled* orders
    (``filled_amount > 0``):

    * ``"entry"``     — only buys: cash deployed into the basket (e.g. the
                        signal rose off zero, coming out of the SMA gate).
    * ``"exit"``      — only sells: the book raised to cash / gone flat
                        (e.g. the signal fell to zero — the risk-off case).
    * ``"rebalance"`` — both sides: the monthly drifted-weight reset,
                        trimming winners and topping up laggards.

    ``per_symbol`` is ``(symbol, signed_notional_quote)`` with buys positive
    and sells negative, largest magnitude first; ``net_quote`` is their sum.
    A cycle with no real fill (all rejected / lost_track / zero-fill, or a
    dry-run planning-only cycle) is not a trade event.
    """

    at: datetime
    kind: str
    per_symbol: list[tuple[str, float]]
    net_quote: float


def trade_events(cycles: list[dict]) -> list[TradeEvent]:
    """Successful cycles with real fills, classified entry/exit/rebalance.

    Total over external journal input: a corrupt order row (non-dict, garbage
    numbers) degrades that one order, never raises. Cycles arrive chronological
    (cache order), so the output is chronological — the same contract as
    ``equity_series`` / ``drift_series`` so markers align with those curves.
    """
    out: list[TradeEvent] = []
    for c in cycles:
        if c.get("outcome") != "success":
            continue
        dt = parse_iso(c.get("ended_at"))
        if dt is None:
            continue
        per_symbol: dict[str, float] = {}
        saw_buy = False
        saw_sell = False
        for o in cycle_orders_executed(c):
            if not isinstance(o, dict):
                # A corrupt cycle whose orders_executed holds a non-dict must
                # degrade one order, not raise (same totality as the reader).
                continue
            # filled_amount, not terminal_status: a 'partial' fill still moved
            # the book, and a 'closed' order that somehow filled zero did not.
            if as_float(o.get("filled_amount")) <= 0.0:
                continue  # rejected / lost_track / zero-fill: no real move
            side = str(o.get("side") or "").lower()
            symbol = str(o.get("symbol") or "?")
            notional = as_float(o.get("filled_notional_quote"))
            if side == "buy":
                saw_buy = True
                per_symbol[symbol] = per_symbol.get(symbol, 0.0) + notional
            elif side == "sell":
                saw_sell = True
                per_symbol[symbol] = per_symbol.get(symbol, 0.0) - notional
            # An unknown side with a fill is not a directional move we can
            # place on a curve — drop it rather than guess a direction.
        if not saw_buy and not saw_sell:
            continue  # planning-only, or nothing actually filled
        if saw_buy and saw_sell:
            kind = "rebalance"
        elif saw_buy:
            kind = "entry"
        else:
            kind = "exit"
        ordered = sorted(per_symbol.items(), key=lambda kv: -abs(kv[1]))
        out.append(TradeEvent(
            at=dt, kind=kind, per_symbol=ordered,
            net_quote=sum(per_symbol.values()),
        ))
    return out


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) over a non-empty list.

    Small and dependency-free (numpy is not imported in the monitoring
    process); exactness is not operationally critical here.
    """
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def duration_stats(cycles: list[dict]) -> Optional[dict]:
    """p50 / p95 / max of cycle ``duration_ms`` over the window, or None.

    A latency summary the single latest scalar cannot give; live cycles poll
    for ack and run longer than dry-runs, so a rising p95 is a leading
    network-trouble indicator.
    """
    vals = [v for _, v in duration_series(cycles)]
    if not vals:
        return None
    return {
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
        "max": max(vals),
    }


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


def as_float(value, default: float = 0.0) -> float:
    """Coerce a journal field to float; total (never raises).

    JSON null, non-numeric strings, and absent keys all fall back to
    ``default``. The journal is external input to this process — a single
    bad field must degrade one value, not raise and blank the dashboard.
    Mirrors :func:`parse_iso`'s totality for numeric fields. Note that
    ``dict.get(key, default)`` does NOT cover a present JSON null (it
    returns None), which is exactly the case that crashed here.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Internal alias retained for the module's own historical call sites.
_as_float = as_float


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

    def latest_live_cycle(self) -> Optional[dict]:
        """Most recent cycle that placed real orders, or None.

        The hourly dry-run and the daily live run share one journal, so
        :meth:`latest_cycle` is almost always a dry-run — a dead daily order
        cron stays invisible while dry-runs keep the journal fresh. This is
        the clock that answers 'is the real order cron still alive?'. See
        :func:`is_live_cycle` for the discriminator and its one caveat.
        """
        self._refresh_if_changed()
        for c in reversed(self._cache):
            if is_live_cycle(c):
                return c
        return None

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
                _as_float(sig.get("ladder_value")),
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
        if elapsed < expected_interval_s * STALE_MULTIPLIER:
            return Staleness.FRESH
        if elapsed < expected_interval_s * DOWN_MULTIPLIER:
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
        except OSError as exc:
            # Path exists but is unstattable (permission on the directory, a
            # broken mount). Fail loud into a read_error, not an exception
            # that blanks the page.
            self._mtime = None
            self._size = None
            self._cache = []
            self._stats = ReadStats(read_error=f"{type(exc).__name__}: {exc}")
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
        except OSError as exc:
            # The journal exists but is unreadable — a PermissionError (wrong
            # group/mode, the exact README misconfiguration) or a directory at
            # the path (IsADirectoryError). Record it loud instead of letting
            # it propagate out of the un-contained initial read in main().
            self._cache = []
            stats.read_error = f"{type(exc).__name__}: {exc}"
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
