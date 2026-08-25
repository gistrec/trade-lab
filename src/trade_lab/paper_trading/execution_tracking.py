"""Execution tracking — reconciles real mainnet execution with the simulation.

Third verification layer (issue #11), next to the behavioral fingerprint
monitor and the look-ahead detector. Reads two journals:

* real:  the mainnet execution journal (``data/journal/cycles_mainnet.jsonl``),
* sim:   the harness journal (``paper_trading/logs/journal.jsonl``),

and answers two questions:

1. Does real equity track the simulated equity, aligned by signal date?
   Per date one CONSISTENT real cycle is sampled: the daily live cycle,
   falling back to the first dry-run only when no live cycle exists.
2. Did real execution carry out the trades the SIMULATION intended?
   Expectations come from the harness rows (``intended_trades``), never
   from the mainnet journal itself — an erroneous production signal and
   its own orders would match each other. Mainnet supplies only the
   actual side: fills and journaled skips. Only LIVE-cycle skips with a
   verified sub-minimum (min-notional class) reason may cover a missing
   trade; they are counted, never alerted on.

Data-quality contract: corrupt lines in the mainnet journal are a tool
error (a malformed line can hold the very cycle under reconciliation);
unknown-schema-version lines degrade to an explicit incomplete-data
warning in the report.

Descriptive, not normative — same contract as the fingerprint monitor.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from trade_lab.monitoring.data_source import (
    JournalReader,
    ReadStats,
    as_float,
    cycle_orders_executed,
    is_live_attempt,
    parse_iso,
)

from .journal import HarnessLogRow


# Owner-adjustable starting point (issue #11), not a calibrated number:
# revisit once several weeks of live-vs-sim overlap exist.
DEFAULT_GAP_THRESHOLD_PCT = 5.0

# Sim weights are exact fractions; anything this small is float noise,
# not an intended trade.
_INTENT_EPS = 1e-9


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
    n_real_fill_events: int
    n_sim_trade_dates: int        # sim dates with a nonzero intended trade
    n_sim_intended_trades: int    # per-symbol intents across those dates
    n_min_notional_skips: int     # live sub-min skips, distinct (date, symbol)
    n_trades_skip_covered: int
    missing_trades: list          # [{date, symbol, expected_side}]
    wrong_direction_trades: list  # [{date, symbol, expected_side, actual_side}]
    partial_fills: list           # [{date, symbol, side, filled, intended, id}]
    unexpected_orders: list       # [{date, symbol, side, notional, cycle_id}]


@dataclass(frozen=True)
class TrackingReport:
    real_journal: str
    sim_journal: str
    equity: EquityTracking
    transitions: TransitionCheck
    real_unknown_version_lines: int
    advisory: str


# ---------------------------------------------------------------------------
# Journal reading
# ---------------------------------------------------------------------------

def _read_real_cycles(path: Path) -> tuple[list[dict], ReadStats]:
    path = Path(path)
    if not path.exists():
        # Tool error, not "no data": a wrong --real-journal path must not
        # read as a clean report.
        raise FileNotFoundError(f"mainnet journal not found: {path}")
    reader = JournalReader(path)
    cycles = reader.cycles(n=sys.maxsize)
    stats = reader.stats()
    if stats.read_error:
        raise OSError(f"mainnet journal unreadable: {stats.read_error}")
    if stats.corrupt_lines:
        # A malformed line can hold the day's ladder change — "no
        # mismatches" over a journal with holes is not a verdict.
        raise ValueError(
            f"mainnet journal {path}: {stats.corrupt_lines} corrupt "
            "line(s) — reconciliation cannot be trusted; repair the "
            "journal first"
        )
    return cycles, stats


def _read_sim_rows(path: Path) -> list[HarnessLogRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"harness journal not found: {path}")
    rows: list[HarnessLogRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Crash-truncated tail — same tolerance as journal.read_log.
                continue
            try:
                rows.append(HarnessLogRow(**data))
            except TypeError as exc:
                # Schema drift is a tool error, not a breach; name the row.
                date = data.get("date", "?") if isinstance(data, dict) else "?"
                raise ValueError(
                    f"harness journal {path} line {lineno} (date={date}) "
                    f"does not match HarnessLogRow: {exc}"
                ) from exc
    return rows


def _cycle_signal_date(cycle: dict) -> Optional[str]:
    """Signal date of a cycle — same semantics as the harness row ``date``."""
    sig = cycle.get("signal")
    if not isinstance(sig, dict):
        return None
    t = parse_iso(sig.get("asof"))
    return t.date().isoformat() if t is not None else None


_COID_RE = re.compile(r"^tsmom_(\d{8})_")


def _coid_signal_date(coid: str) -> Optional[str]:
    m = _COID_RE.match(coid)
    if m is None:
        return None
    try:
        decision = datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None
    # The coid carries the UTC placement-decision date; the bar it acted
    # on closed at that day's 00:00 and is stamped one day earlier — the
    # same axis ``signal.asof`` puts signal-bearing cycles on.
    return (decision - timedelta(days=1)).isoformat()


def _base_asset(symbol) -> str:
    return str(symbol or "?").split("/")[0]


# ---------------------------------------------------------------------------
# Equity comparison
# ---------------------------------------------------------------------------

def real_equity_by_date(cycles: list[dict]) -> dict[str, float]:
    """Successful cycles' ``equity_usd`` keyed by signal date.

    One CONSISTENT cycle per date: the first live cycle wins; the first
    dry-run is a fallback only until a live cycle appears. Last-wins
    would let an 18:00 dry-run valuation retroactively replace the 00:05
    live observation — a different market window.
    """
    out: dict[str, float] = {}
    live_seen: set[str] = set()
    for c in cycles:
        if c.get("outcome") != "success":
            continue
        d = _cycle_signal_date(c)
        eq = c.get("equity_usd")
        if d is None or eq is None:
            continue
        try:
            val = float(eq)
        except (TypeError, ValueError):
            continue
        if is_live_attempt(c):
            if d not in live_seen:
                live_seen.add(d)
                out[d] = val
        elif d not in out:
            out[d] = val
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
# Per-symbol trade check (sim-expected vs real fills/skips)
# ---------------------------------------------------------------------------

def sim_expected_trades(rows: list[HarnessLogRow]) -> dict[str, dict[str, float]]:
    """date -> {asset: intended weight delta}, nonzero intents only.

    The sim intends trades exactly when its ladder moves (its current
    weights equal the prior targets), so this is the harness-side
    transition record — the expectation the real book is judged against.
    """
    by_date: dict[str, HarnessLogRow] = {}
    for r in rows:
        by_date[r.date] = r  # idempotent re-runs: last row per date wins
    out: dict[str, dict[str, float]] = {}
    for d, r in by_date.items():
        trades = {
            a: float(v)
            for a, v in (r.intended_trades or {}).items()
            if abs(float(v)) > _INTENT_EPS
        }
        if trades:
            out[d] = trades
    return out


def real_order_events(cycles: list[dict]) -> list[dict]:
    """Per-symbol real fills, one event per client order.

    The LAST journal record per clientOrderId wins: a reconstruction
    re-journals the same order with its final state, and keeping both
    would double-count the fill and pin a stale partial.
    """
    by_coid: dict[str, dict] = {}
    no_coid: list[dict] = []
    for c in cycles:
        sig_date = _cycle_signal_date(c)
        ended = parse_iso(c.get("ended_at"))
        ended_date = ended.date().isoformat() if ended is not None else None
        for o in cycle_orders_executed(c):
            if not isinstance(o, dict):
                continue
            coid = str(o.get("client_order_id") or "")
            # Signal-less records (failed-cycle partial fills,
            # reconstruction recoveries) are dated by the coid's embedded
            # decision date, not ended_at — a recovery lands days later.
            date = sig_date or _coid_signal_date(coid) or ended_date
            if date is None:
                continue
            event = {
                "date": date,
                "symbol": o.get("symbol"),
                "side": str(o.get("side") or "").lower(),
                "terminal_status": o.get("terminal_status"),
                "intended_amount": as_float(o.get("intended_amount")),
                "filled_amount": as_float(o.get("filled_amount")),
                "filled_notional_quote": as_float(o.get("filled_notional_quote")),
                "client_order_id": coid or "?",
                "cycle_id": (c.get("cycle_id") or "?")[:8],
            }
            if coid:
                by_coid[coid] = event
            else:
                no_coid.append(event)
    events = [
        e for e in list(by_coid.values()) + no_coid
        if e["filled_amount"] > 0.0  # rejected / lost_track / zero-fill
    ]
    return sorted(events, key=lambda e: (e["date"], str(e["symbol"])))


# The only reason shapes delta.py journals for a sub-minimum delta.
# pending_* skips (live_cycle) are transient retries and never legitimize
# a missed trade; dry-run planning skips never blocked a real order.
_MIN_NOTIONAL_MARKERS = ("< min_amount", "< min_cost", "truncates to 0")


def _is_min_notional_reason(reason: str) -> bool:
    return any(marker in reason for marker in _MIN_NOTIONAL_MARKERS)


def live_min_notional_skips(cycles: list[dict]) -> dict[tuple, set]:
    """(signal date, base asset) -> desired sides. LIVE cycles only."""
    out: dict[tuple, set] = {}
    for c in cycles:
        if not is_live_attempt(c):
            continue
        d = _cycle_signal_date(c)
        skipped = c.get("orders_skipped")
        if d is None or not isinstance(skipped, list):
            continue
        for s in skipped:
            if not isinstance(s, dict):
                continue
            if not _is_min_notional_reason(str(s.get("reason") or "")):
                continue
            key = (d, _base_asset(s.get("symbol")))
            out.setdefault(key, set()).add(
                str(s.get("desired_side") or "").lower()
            )
    return out


def _is_partial_fill(event: dict) -> bool:
    if event["terminal_status"] == "partial":
        return True
    intended = event["intended_amount"]
    return intended > 0.0 and event["filled_amount"] < intended * (1.0 - 1e-9)


def check_transitions(
    cycles: list[dict], sim_rows: list[HarnessLogRow],
) -> TransitionCheck:
    expected = sim_expected_trades(sim_rows)
    events = real_order_events(cycles)
    skips = live_min_notional_skips(cycles)

    by_key: dict[tuple, list[dict]] = {}
    for e in events:
        by_key.setdefault((e["date"], _base_asset(e["symbol"])), []).append(e)

    missing: list[dict] = []
    wrong_direction: list[dict] = []
    partials: list[dict] = []
    covered = 0
    for d in sorted(expected):
        for asset in sorted(expected[d]):
            want = "buy" if expected[d][asset] > 0 else "sell"
            evs = by_key.get((d, asset), [])
            same_side = [e for e in evs if e["side"] == want]
            if same_side:
                for e in same_side:
                    if _is_partial_fill(e):
                        partials.append({
                            "date": d,
                            "symbol": e["symbol"],
                            "side": want,
                            "filled_amount": e["filled_amount"],
                            "intended_amount": e["intended_amount"],
                            "client_order_id": e["client_order_id"],
                        })
                continue
            if evs:
                wrong_direction.append({
                    "date": d,
                    "symbol": asset,
                    "expected_side": want,
                    "actual_side": evs[0]["side"],
                })
                continue
            # Coverage requires the SAME side: a journaled sell-skip does
            # not legitimize a missed buy.
            if want in skips.get((d, asset), set()):
                covered += 1
                continue
            missing.append({"date": d, "symbol": asset, "expected_side": want})

    expected_keys = {(d, a) for d, m in expected.items() for a in m}
    unexpected = [
        {
            "date": e["date"],
            "symbol": e["symbol"],
            "side": e["side"],
            "filled_notional_quote": e["filled_notional_quote"],
            "cycle_id": e["cycle_id"],
        }
        for e in events
        if (e["date"], _base_asset(e["symbol"])) not in expected_keys
    ]

    return TransitionCheck(
        n_real_fill_events=len(events),
        n_sim_trade_dates=len(expected),
        n_sim_intended_trades=sum(len(m) for m in expected.values()),
        n_min_notional_skips=len(skips),
        n_trades_skip_covered=covered,
        missing_trades=missing,
        wrong_direction_trades=wrong_direction,
        partial_fills=partials,
        unexpected_orders=unexpected,
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
    cycles, real_stats = _read_real_cycles(real_journal)
    rows = _read_sim_rows(sim_journal)
    equity = compare_equity(
        real_equity_by_date(cycles), sim_equity_by_date(rows),
        gap_threshold_pct,
    )
    transitions = check_transitions(cycles, rows)

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
    n_missing = len(transitions.missing_trades)
    n_wrong = len(transitions.wrong_direction_trades)
    n_partial = len(transitions.partial_fills)
    n_unexpected = len(transitions.unexpected_orders)
    if n_missing or n_wrong or n_partial or n_unexpected:
        advisory += (
            f" Trade mismatches: {n_missing} missing, {n_wrong} "
            f"wrong-direction, {n_partial} partial, {n_unexpected} "
            f"unexpected."
        )
    if real_stats.unknown_version_lines:
        advisory += (
            f" INCOMPLETE DATA: {real_stats.unknown_version_lines} mainnet "
            "journal line(s) with unknown schema_version were skipped — "
            "absence of mismatches is not verified."
        )

    return TrackingReport(
        real_journal=str(real_journal),
        sim_journal=str(sim_journal),
        equity=equity,
        transitions=transitions,
        real_unknown_version_lines=real_stats.unknown_version_lines,
        advisory=advisory,
    )
