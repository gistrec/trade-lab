"""Execution tracking — reconciles real mainnet execution with the simulation.

Third verification layer (issue #11), next to the behavioral fingerprint
monitor and the look-ahead detector. Reads two journals:

* real:  the mainnet execution journal (``data/journal/cycles_mainnet.jsonl``),
* sim:   the harness journal (``paper_trading/logs/journal.jsonl``),

and answers two questions:

1. Does real equity track the simulated equity, aligned by signal date
   AND by trade phase (both curves PRE-trade — see
   :func:`sim_pre_trade_equity_by_date`)? Per date one CONSISTENT real
   cycle is sampled: the daily live cycle, falling back to the first
   dry-run only when no live cycle exists.
2. Did real execution carry out the trades the SIMULATION intended?
   Expectations come from the harness rows (``intended_trades``), never
   from the mainnet journal itself — an erroneous production signal and
   its own orders would match each other. Mainnet supplies only the
   actual side: fills and journaled skips. Only LIVE-cycle skips with a
   verified sub-minimum (min-notional class) reason may cover a missing
   trade; they are counted, never alerted on. A fill is "unexpected"
   only INSIDE simulation coverage (a date with a harness row); fills
   on uncovered dates are a separate coverage note, not a mismatch.

Data-quality contract: corrupt lines in the mainnet journal are a tool
error (a malformed line can hold the very cycle under reconciliation);
in the harness journal only a malformed FINAL line is tolerated (the
crash-truncated append), a malformed row anywhere earlier is a tool
error too; unknown-schema-version lines degrade to an explicit
incomplete-data warning in the report.

Descriptive, not normative — same contract as the fingerprint monitor.
"""
from __future__ import annotations

import json
import math
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

# Size check: a fill is compared as a fraction of the real book against
# the sim's intended weight delta. The band is deliberately wide — lot
# steps, the 10 bp funding reserve and snapshot-to-fill price drift all
# move a leg by percent, not by multiples — so only order-of-magnitude
# sizing errors trip it. Intents below the floor are left unjudged:
# there lot-step quantization dominates the signal.
_SIZE_RATIO_LO = 0.5
_SIZE_RATIO_HI = 2.0
_SIZE_MIN_INTENT_DW = 0.005


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
    size_mismatches: list         # [{date, symbol, expected_dw, actual_dw, ratio}]
    unexpected_orders: list       # [{date, symbol, side, notional, cycle_id}]
    wrong_market_fills: list      # fills on a non-USDT quote market
    out_of_coverage_fills: list   # fills on dates the harness never logged
    pre_live_sim_trades: list     # sim intents before the first live attempt
    coverage: Optional[tuple]     # (first, last) harness-covered ISO dates


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
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Only a crash-truncated append is tolerable, and it can only be the
    # last row with content. A malformed row anywhere earlier drops an
    # intended transition out of the expectations — and if the matching
    # real trade is missing too, the report would read "no mismatch".
    last_content = max(
        (i for i, raw in enumerate(lines, start=1) if raw.strip()), default=0
    )
    rows: list[HarnessLogRow] = []
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            if lineno == last_content:
                continue
            raise ValueError(
                f"harness journal {path} line {lineno}: malformed row "
                "before the end of the journal — an intended trade may be "
                "hidden there; repair the journal first"
            ) from exc
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


# The harness trades {asset}/USDT. A real fill on another quote market is
# config drift, not the intended trade — reducing it to its base asset
# would let BTC/FDUSD (or a bare "BTC") satisfy a BTC/USDT expectation.
SIM_QUOTE = "USDT"


def _quote_asset(symbol) -> Optional[str]:
    parts = str(symbol or "").split("/")
    return parts[1] if len(parts) == 2 and parts[1] else None


# ---------------------------------------------------------------------------
# Equity comparison
# ---------------------------------------------------------------------------

def real_equity_by_date(cycles: list[dict]) -> dict[str, float]:
    """Read-phase ``equity_usd`` keyed by signal date, one cycle per date.

    PRE-TRADE by construction: ``equity_usd`` comes from the cycle's read
    phase, before that date's orders are placed. The sim side is brought
    onto the same phase (see :func:`sim_pre_trade_equity_by_date`).

    A live attempt wins the date whatever its OUTCOME — a 'partial' or
    'unknown_orders' cycle still read its equity pre-trade, and it may
    have placed fills, which would make any later dry-run that day a
    POST-trade valuation at a different market window. When a live
    attempt exists but never reached the read phase, the date is dropped
    rather than back-filled from a dry-run: whether it traded is exactly
    what is unknown. Dry-runs supply a date only when no live attempt
    touched it at all (observation-phase journals).
    """
    live_eq: dict[str, float] = {}
    live_dates: set[str] = set()
    dry_eq: dict[str, float] = {}
    for c in cycles:
        d = _cycle_signal_date(c)
        if d is None:
            continue
        live = is_live_attempt(c)
        if live:
            live_dates.add(d)
        eq = c.get("equity_usd")
        if eq is None:
            continue
        try:
            val = float(eq)
        except (TypeError, ValueError):
            continue
        if live:
            live_eq.setdefault(d, val)
        elif c.get("outcome") == "success":
            dry_eq.setdefault(d, val)
    out = dict(live_eq)
    for d, val in dry_eq.items():
        if d not in live_dates:
            out[d] = val
    return out


def first_live_attempt_date(cycles: list[dict]) -> Optional[str]:
    """Earliest signal date carrying a live attempt, or None.

    Sim intents before it were never given a chance to execute — the
    mirror image of real fills outside harness coverage.
    """
    dates = {
        d for c in cycles if is_live_attempt(c)
        for d in [_cycle_signal_date(c)] if d is not None
    }
    return min(dates) if dates else None


def sim_pre_trade_equity_by_date(rows: list[HarnessLogRow]) -> dict[str, float]:
    """Harness equity per signal date, backed out to the PRE-trade phase.

    ``portfolio_equity`` is journaled AFTER the row's simulated turnover
    cost is deducted, while real ``equity_usd`` is read BEFORE that date's
    orders go out. Comparing them raw charges the cost one observation
    early on the sim curve — a false level gap and a false return
    difference on exactly the transition dates that matter. The cost
    fraction is recoverable from committed fields: the harness computes
    ``net = gross - turnover x cost_rate`` and
    ``equity_post = equity_pre x (1 - turnover x cost_rate)``.
    """
    out: dict[str, float] = {}
    for r in rows:
        cost_fraction = (
            float(r.gross_position_return) - float(r.net_position_return)
        )
        # turnover and cost_rate are both non-negative, so anything
        # outside [0, 1) (NaN included) is schema drift, not a cost.
        if not 0.0 <= cost_fraction < 1.0:
            raise ValueError(
                f"harness row {r.date}: implied turnover cost fraction "
                f"{cost_fraction} outside [0, 1) — cannot restore the "
                "pre-trade equity phase"
            )
        out[r.date] = float(r.portfolio_equity) / (1.0 - cost_fraction)
    return out


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
        # isfinite first: NaN fails every comparison, so a bare <= 0 test
        # would pass it through and turn the whole report into NaN — with
        # `abs(gap) > threshold` false, i.e. a silent "within threshold".
        if not (math.isfinite(real[d]) and math.isfinite(sim[d])):
            raise ValueError(
                f"non-finite equity on aligned date {d}: "
                f"real={real[d]}, sim={sim[d]} — repair the journal"
            )
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
                prev = by_coid.get(coid)
                # Last-wins EXCEPT for a detail-less state-cache record: a
                # same-day rerun re-journals the cached terminal result with
                # zeroed fill fields (the store does not retain them), and
                # letting it win would erase the real fill and report the
                # trade as missing.
                if not (
                    prev is not None
                    and prev["filled_amount"] > 0.0
                    and event["filled_amount"] <= 0.0
                ):
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


def live_min_notional_skips(cycles: list[dict]) -> dict[tuple, dict]:
    """(signal date, base asset) -> {side: largest skipped notional}.

    The notional is retained, not just the side: a skip only legitimizes
    a missed trade when the SIM's own intent was sub-minimum too. A
    production sizing bug that computes a dust order and skips it as
    ``< min_cost`` must not excuse a material intent.
    """
    out: dict[tuple, dict] = {}
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
            side = str(s.get("desired_side") or "").lower()
            notional = as_float(s.get("desired_notional"))
            sides = out.setdefault(key, {})
            sides[side] = max(sides.get(side, 0.0), notional)
    return out


def _skip_covers(
    expected_dw: float, skipped_notional: float, equity: Optional[float],
) -> bool:
    """True when the skip plausibly IS the sim's intent, sub-minimum.

    Without the book we cannot compare the two, so the side match alone
    decides (the pre-existing posture). With it, the skipped notional
    must be within the same size band as the intent — a dust skip beside
    a material intent is a sizing bug wearing a legitimate reason.
    """
    if equity is None or not math.isfinite(equity) or equity <= 0.0:
        return True
    want = abs(float(expected_dw))
    if want < _SIZE_MIN_INTENT_DW:
        return True     # lot-step territory: not judgeable either way
    return (skipped_notional / equity) / want >= _SIZE_RATIO_LO


def _size_mismatch(
    date: str,
    asset: str,
    expected_dw: float,
    events: list[dict],
    real_equity: dict[str, float],
) -> Optional[dict]:
    """Aggregate fill vs the sim's intended weight delta, or None.

    Presence and direction alone cannot catch a sizing bug: a fully
    filled 1-USDT buy where the sim intended a tenth of the book reads
    as a clean match and can sit under the equity threshold for weeks.
    Real notional is normalized by the book so the two sides are
    comparable (the sim bankroll and the real book differ by orders of
    magnitude by design).
    """
    equity = real_equity.get(date)
    want = abs(float(expected_dw))
    if equity is None or not math.isfinite(equity) or equity <= 0.0:
        return None
    if want < _SIZE_MIN_INTENT_DW:
        return None
    filled = sum(e["filled_notional_quote"] for e in events)
    actual = filled / equity
    ratio = actual / want
    if _SIZE_RATIO_LO <= ratio <= _SIZE_RATIO_HI:
        return None
    return {
        "date": date,
        "symbol": asset,
        "side": "buy" if expected_dw > 0 else "sell",
        "expected_weight_delta": want,
        "actual_weight_delta": actual,
        "ratio": ratio,
        "filled_notional_quote": filled,
        "equity_usd": equity,
    }


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
    real_eq = real_equity_by_date(cycles)
    first_live = first_live_attempt_date(cycles)

    # A fill on a market the sim never trades can never satisfy an
    # expectation — it is drift, surfaced on its own.
    wrong_market: list[dict] = []
    by_key: dict[tuple, list[dict]] = {}
    for e in events:
        if _quote_asset(e["symbol"]) != SIM_QUOTE:
            wrong_market.append({
                "date": e["date"],
                "symbol": e["symbol"],
                "side": e["side"],
                "filled_notional_quote": e["filled_notional_quote"],
                "expected_quote": SIM_QUOTE,
                "cycle_id": e["cycle_id"],
            })
            continue
        by_key.setdefault((e["date"], _base_asset(e["symbol"])), []).append(e)

    missing: list[dict] = []
    wrong_direction: list[dict] = []
    partials: list[dict] = []
    size_mismatches: list[dict] = []
    pre_live: list[dict] = []
    covered = 0
    for d in sorted(expected):
        # Sim intents predating the first retained live attempt never had
        # a real counterpart to miss — the mirror of out-of-coverage fills.
        if first_live is None or d < first_live:
            for asset in sorted(expected[d]):
                pre_live.append({
                    "date": d, "symbol": asset,
                    "expected_side": "buy" if expected[d][asset] > 0 else "sell",
                })
            continue
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
                mismatch = _size_mismatch(
                    d, asset, expected[d][asset], same_side, real_eq,
                )
                if mismatch is not None:
                    size_mismatches.append(mismatch)
                continue
            if evs:
                wrong_direction.append({
                    "date": d,
                    "symbol": asset,
                    "expected_side": want,
                    "actual_side": evs[0]["side"],
                })
                continue
            # Coverage requires the SAME side AND a comparable size: a
            # journaled sell-skip does not legitimize a missed buy, and a
            # dust skip does not legitimize a material intent (a sizing
            # bug produces exactly that shape).
            skipped_notional = skips.get((d, asset), {}).get(want)
            if skipped_notional is not None and _skip_covers(
                expected[d][asset], skipped_notional, real_eq.get(d),
            ):
                covered += 1
                continue
            missing.append({"date": d, "symbol": asset, "expected_side": want})

    expected_keys = {(d, a) for d, m in expected.items() for a in m}
    # A date WITH a harness row carries a real "no trade" expectation; a
    # date the harness never logged (staggered start, partial retention)
    # carries none — calling those fills unexpected is a permanent false
    # mismatch, so they are reported as coverage instead.
    covered_dates = {r.date for r in sim_rows}
    unexpected: list[dict] = []
    out_of_coverage: list[dict] = []
    for e in events:
        if _quote_asset(e["symbol"]) != SIM_QUOTE:
            continue      # already surfaced as wrong_market
        if (e["date"], _base_asset(e["symbol"])) in expected_keys:
            continue
        record = {
            "date": e["date"],
            "symbol": e["symbol"],
            "side": e["side"],
            "filled_notional_quote": e["filled_notional_quote"],
            "cycle_id": e["cycle_id"],
        }
        if e["date"] in covered_dates:
            unexpected.append(record)
        else:
            out_of_coverage.append(record)

    coverage = (
        (min(covered_dates), max(covered_dates)) if covered_dates else None
    )
    return TransitionCheck(
        n_real_fill_events=len(events),
        n_sim_trade_dates=len(expected),
        n_sim_intended_trades=sum(len(m) for m in expected.values()),
        n_min_notional_skips=len(skips),
        n_trades_skip_covered=covered,
        missing_trades=missing,
        wrong_direction_trades=wrong_direction,
        partial_fills=partials,
        size_mismatches=size_mismatches,
        unexpected_orders=unexpected,
        wrong_market_fills=wrong_market,
        out_of_coverage_fills=out_of_coverage,
        pre_live_sim_trades=pre_live,
        coverage=coverage,
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
        real_equity_by_date(cycles), sim_pre_trade_equity_by_date(rows),
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
            f"±{equity.threshold_pct:.2f}% (both curves pre-trade). "
            "Operator review."
        )
    else:
        advisory = "Real execution tracks the simulation within threshold."
    n_missing = len(transitions.missing_trades)
    n_wrong = len(transitions.wrong_direction_trades)
    n_partial = len(transitions.partial_fills)
    n_size = len(transitions.size_mismatches)
    n_unexpected = len(transitions.unexpected_orders)
    if n_missing or n_wrong or n_partial or n_size or n_unexpected:
        advisory += (
            f" Trade mismatches: {n_missing} missing, {n_wrong} "
            f"wrong-direction, {n_partial} partial, {n_size} mis-sized, "
            f"{n_unexpected} unexpected."
        )
    if transitions.wrong_market_fills:
        advisory += (
            f" WRONG MARKET: {len(transitions.wrong_market_fills)} real "
            f"fill(s) on a quote market the simulation never trades "
            f"(expected {SIM_QUOTE}) — configuration drift, not a match."
        )
    if transitions.pre_live_sim_trades:
        dates = sorted({e["date"] for e in transitions.pre_live_sim_trades})
        advisory += (
            f" COVERAGE NOTE: {len(transitions.pre_live_sim_trades)} sim "
            f"intent(s) on {len(dates)} date(s) before the first live "
            f"attempt ({dates[0]}..{dates[-1]}) — no real cycle existed "
            "to execute them, so they are not counted as missing."
        )
    if transitions.out_of_coverage_fills:
        dates = sorted({e["date"] for e in transitions.out_of_coverage_fills})
        span = (
            f"{transitions.coverage[0]}..{transitions.coverage[1]}"
            if transitions.coverage else "empty"
        )
        listed = ", ".join(dates)
        advisory += (
            f" COVERAGE NOTE: {len(transitions.out_of_coverage_fills)} real "
            f"fill(s) on date(s) outside harness coverage ({span}): "
            f"{listed} — the simulation expected nothing there, so they are "
            "not counted as mismatches."
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
