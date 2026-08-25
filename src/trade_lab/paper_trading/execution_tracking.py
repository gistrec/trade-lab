"""Execution tracking — reconciles real mainnet execution with the simulation.

Third verification layer (issue #11), next to the behavioral fingerprint
monitor and the look-ahead detector. Reads two journals:

* real:  the mainnet execution journal (``data/journal/cycles_mainnet.jsonl``),
* sim:   the harness journal (``paper_trading/logs/journal.jsonl``),

and answers two questions:

1. Does real equity track the simulated equity, aligned by signal date?
2. Does every real order correspond to a journaled ladder transition
   (and vice versa)? Min-notional skips are journaled with reasons and
   are legitimate — counted separately, never alerted on.

Descriptive, not normative — same contract as the fingerprint monitor.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from trade_lab.monitoring.data_source import (
    JournalReader,
    as_float,
    cycle_orders_executed,
    parse_iso,
)

from .journal import HarnessLogRow, read_log


# Owner-adjustable starting point (issue #11), not a calibrated number:
# revisit once several weeks of live-vs-sim overlap exist.
DEFAULT_GAP_THRESHOLD_PCT = 5.0


@dataclass(frozen=True)
class EquityTracking:
    n_real_days: int
    n_sim_days: int
    n_aligned_days: int
    overlap: Optional[tuple]              # (first, last) aligned ISO dates
    cum_abs_return_diff: Optional[float]  # sum |real daily ret − sim daily ret|
    level_gap_pct: Optional[float]        # normalized level gap at last aligned day
    threshold_pct: float
    breached: bool


@dataclass(frozen=True)
class TransitionCheck:
    n_real_order_events: int
    n_ladder_transitions: int
    n_min_notional_skips: int             # legitimate, counted, never alerted
    n_transitions_skip_covered: int
    orders_without_transition: list       # [{date, symbol, side, ...}]
    transitions_without_order: list       # [{date, prior, new}]


@dataclass(frozen=True)
class TrackingReport:
    real_journal: str
    sim_journal: str
    equity: EquityTracking
    transitions: TransitionCheck
    advisory: str


# ---------------------------------------------------------------------------
# Journal reading
# ---------------------------------------------------------------------------

def _read_real_cycles(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        # Tool error, not "no data": a wrong --real-journal path must not
        # read as a clean report.
        raise FileNotFoundError(f"mainnet journal not found: {path}")
    reader = JournalReader(path)
    cycles = reader.cycles(n=sys.maxsize)
    err = reader.stats().read_error
    if err:
        raise OSError(f"mainnet journal unreadable: {err}")
    return cycles


def _read_sim_rows(path: Path) -> list[HarnessLogRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"harness journal not found: {path}")
    return read_log(path)


def _cycle_signal_date(cycle: dict) -> Optional[str]:
    """Signal date of a cycle — same semantics as the harness row ``date``."""
    sig = cycle.get("signal")
    if not isinstance(sig, dict):
        return None
    t = parse_iso(sig.get("asof"))
    return t.date().isoformat() if t is not None else None


def _cycle_date(cycle: dict) -> Optional[str]:
    """Signal date, falling back to ``ended_at`` (reconstruction cycles
    carry no signal)."""
    d = _cycle_signal_date(cycle)
    if d is not None:
        return d
    t = parse_iso(cycle.get("ended_at"))
    return t.date().isoformat() if t is not None else None


# ---------------------------------------------------------------------------
# Equity comparison
# ---------------------------------------------------------------------------

def real_equity_by_date(cycles: list[dict]) -> dict[str, float]:
    """Successful cycles' ``equity_usd`` keyed by signal date; the last
    cycle of a date wins (6-hourly dry-runs re-observe the same day)."""
    out: dict[str, float] = {}
    for c in cycles:
        if c.get("outcome") != "success":
            continue
        d = _cycle_signal_date(c)
        eq = c.get("equity_usd")
        if d is None or eq is None:
            continue
        try:
            out[d] = float(eq)
        except (TypeError, ValueError):
            continue
    return out


def sim_equity_by_date(rows: list[HarnessLogRow]) -> dict[str, float]:
    return {r.date: float(r.portfolio_equity) for r in rows}


def compare_equity(
    real: dict[str, float],
    sim: dict[str, float],
    threshold_pct: float,
) -> EquityTracking:
    # ISO dates sort chronologically as strings.
    aligned = sorted(set(real) & set(sim))
    if not aligned:
        # Journals exist but do not overlap yet — descriptive, not an error.
        return EquityTracking(
            n_real_days=len(real), n_sim_days=len(sim), n_aligned_days=0,
            overlap=None, cum_abs_return_diff=None, level_gap_pct=None,
            threshold_pct=threshold_pct, breached=False,
        )
    for d in aligned:
        if real[d] <= 0.0 or sim[d] <= 0.0:
            raise ValueError(
                f"non-positive equity on aligned date {d}: "
                f"real={real[d]}, sim={sim[d]} — cannot compute returns"
            )
    first, last = aligned[0], aligned[-1]
    cum = 0.0
    for prev, cur in zip(aligned, aligned[1:]):
        cum += abs(
            (real[cur] / real[prev] - 1.0) - (sim[cur] / sim[prev] - 1.0)
        )
    # Levels are normalized to the first aligned date: real capital and the
    # sim's virtual bankroll differ by orders of magnitude by design.
    gap_pct = (
        (real[last] / real[first]) / (sim[last] / sim[first]) - 1.0
    ) * 100.0
    return EquityTracking(
        n_real_days=len(real), n_sim_days=len(sim),
        n_aligned_days=len(aligned), overlap=(first, last),
        cum_abs_return_diff=cum, level_gap_pct=gap_pct,
        threshold_pct=threshold_pct,
        breached=bool(abs(gap_pct) > threshold_pct),
    )


# ---------------------------------------------------------------------------
# Position-transition check
# ---------------------------------------------------------------------------

def ladder_transitions(cycles: list[dict]) -> list[dict]:
    """Dates where the journaled ladder value changed vs the prior date."""
    by_date: dict[str, float] = {}
    for c in cycles:
        sig = c.get("signal")
        if not isinstance(sig, dict):
            continue
        d = _cycle_signal_date(c)
        if d is None:
            continue
        try:
            by_date[d] = float(sig.get("ladder_value"))
        except (TypeError, ValueError):
            continue
    out: list[dict] = []
    prev: Optional[float] = None
    for d in sorted(by_date):
        v = by_date[d]
        if prev is not None and abs(v - prev) > 1e-12:
            out.append({"date": d, "prior": prev, "new": v})
        prev = v
    return out


def real_order_events(cycles: list[dict]) -> list[dict]:
    """Per-symbol real position moves: executed orders with a positive fill."""
    out: list[dict] = []
    for c in cycles:
        d = _cycle_date(c)
        if d is None:
            continue
        for o in cycle_orders_executed(c):
            if not isinstance(o, dict):
                continue
            if as_float(o.get("filled_amount")) <= 0.0:
                continue  # rejected / lost_track / zero-fill: book did not move
            out.append({
                "date": d,
                "symbol": o.get("symbol"),
                "side": str(o.get("side") or "").lower(),
                "filled_amount": as_float(o.get("filled_amount")),
                "filled_notional_quote": as_float(o.get("filled_notional_quote")),
                "cycle_id": (c.get("cycle_id") or "?")[:8],
            })
    return out


def _skipped_symbols_by_date(cycles: list[dict]) -> dict[str, set]:
    out: dict[str, set] = {}
    for c in cycles:
        d = _cycle_date(c)
        skipped = c.get("orders_skipped")
        if d is None or not isinstance(skipped, list):
            continue
        for s in skipped:
            if not isinstance(s, dict):
                continue
            out.setdefault(d, set()).add(str(s.get("symbol") or "?"))
    return out


def check_transitions(cycles: list[dict]) -> TransitionCheck:
    transitions = ladder_transitions(cycles)
    events = real_order_events(cycles)
    skips = _skipped_symbols_by_date(cycles)
    event_dates = {e["date"] for e in events}
    transition_dates = {t["date"] for t in transitions}

    orders_without_transition = [
        e for e in events if e["date"] not in transition_dates
    ]
    transitions_without_order: list[dict] = []
    skip_covered = 0
    for t in transitions:
        if t["date"] in event_dates:
            continue
        if t["date"] in skips:
            # Journaled min-notional skip explains the missing order —
            # legitimate by design, counted, not flagged.
            skip_covered += 1
            continue
        transitions_without_order.append(t)

    # Distinct (date, symbol): the 6-hourly dry-run re-records the same
    # skip several times a day, and raw entry counts would inflate with
    # observation time, not with trading activity.
    n_skips = sum(len(symbols) for symbols in skips.values())
    return TransitionCheck(
        n_real_order_events=len(events),
        n_ladder_transitions=len(transitions),
        n_min_notional_skips=n_skips,
        n_transitions_skip_covered=skip_covered,
        orders_without_transition=orders_without_transition,
        transitions_without_order=transitions_without_order,
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def check_execution_tracking(
    real_journal: Path,
    sim_journal: Path,
    *,
    gap_threshold_pct: float = DEFAULT_GAP_THRESHOLD_PCT,
) -> TrackingReport:
    cycles = _read_real_cycles(real_journal)
    rows = _read_sim_rows(sim_journal)
    equity = compare_equity(
        real_equity_by_date(cycles), sim_equity_by_date(rows),
        gap_threshold_pct,
    )
    transitions = check_transitions(cycles)

    if equity.n_aligned_days == 0:
        advisory = (
            "No aligned dates yet — real and simulated journals do not "
            "overlap; nothing to compare."
        )
    elif equity.breached:
        advisory = (
            f"TRACKING BREACH — real-vs-sim level gap "
            f"{equity.level_gap_pct:+.2f}% exceeds "
            f"±{equity.threshold_pct:.2f}%. Operator review."
        )
    else:
        advisory = "Real execution tracks the simulation within threshold."
    n_ow = len(transitions.orders_without_transition)
    n_tw = len(transitions.transitions_without_order)
    if n_ow or n_tw:
        advisory += (
            f" Transition mismatches: {n_ow} order(s) without a ladder "
            f"transition, {n_tw} transition(s) without a real order."
        )

    return TrackingReport(
        real_journal=str(real_journal),
        sim_journal=str(sim_journal),
        equity=equity,
        transitions=transitions,
        advisory=advisory,
    )
