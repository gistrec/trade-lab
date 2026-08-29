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
   dry-run only when no live cycle exists. The comparison starts at
   live-execution coverage, and DEPOSITS / WITHDRAWALS are not returns:
   they must be declared by the operator (``--capital-events``), and an
   equity step no declared flow explains suppresses the gap number
   instead of conflating a transfer with performance.
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

# Capital-flow detector. A deposit is indistinguishable from a windfall
# in the journals, so the discriminator is magnitude: both books hold the
# SAME long-only basket at ladder <= 1, so a one-day real-vs-sim
# divergence this wide is not execution noise (lot steps, the 10 bp
# reserve, price drift move a day by percent). The basket multiple keeps
# a genuine crash — real book in cash while the sim is fully invested —
# from reading as a transfer.
_CAPITAL_DIVERGENCE_FLOOR = 0.25
_CAPITAL_BASKET_MULTIPLE = 3.0


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
    tracking_start: Optional[str]         # live-coverage date the comparison starts at
    capital_flows_applied: list           # [{from_date, date, amount_usd}]
    unexplained_moves: list               # [{from_date, date, ...}] — gap suppressed


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
    extra_opposite_fills: list    # opposite-side fills beside an intended trade
    unexpected_orders: list       # [{date, symbol, side, notional, cycle_id}]
    bootstrap_orders: list        # first-live catch-up of a pre-live sim intent
    wrong_market_fills: list      # fills on a non-USDT quote market
    out_of_coverage_fills: list   # fills on dates the harness never logged
    pre_live_sim_trades: list     # sim intents before the first live attempt
    coverage: Optional[tuple]     # (first, last) harness-covered ISO dates
    live_coverage_start: Optional[str]   # first date evidencing live execution


@dataclass(frozen=True)
class TrackingReport:
    real_journal: str
    sim_journal: str
    capital_events_file: Optional[str]
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


def load_capital_events(path: Path) -> dict[str, float]:
    """Operator-declared deposits / withdrawals: ISO date -> signed USD.

    Capital flows are operator knowledge — nothing in either journal
    distinguishes a deposit from a windfall — so they are declared, never
    inferred. The file is a JSON list of
    ``{"date": "YYYY-MM-DD", "amount_usd": <signed float>, "note": ...}``;
    ``date`` is the first signal date whose ``equity_usd`` reading ALREADY
    includes the transfer. That reading must exist in the real journal:
    the span return is SPLIT there, so a flow declared on a date with no
    equity reading is rejected rather than netted at one end of the span.
    Positive = deposit, negative = withdrawal.

    Every malformed entry is a tool error: a silently ignored declaration
    would leave a transfer inside the return series, which is exactly the
    number this file exists to remove.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"capital-events file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"capital-events file {path}: malformed JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(
            f"capital-events file {path}: expected a JSON list of "
            "{date, amount_usd} objects"
        )
    out: dict[str, float] = {}
    for i, entry in enumerate(data):
        where = f"capital-events file {path} entry {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected an object, got {entry!r}")
        raw_date = entry.get("date")
        try:
            date_iso = datetime.strptime(str(raw_date), "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{where}: 'date' must be YYYY-MM-DD, got {raw_date!r}"
            ) from exc
        try:
            amount = float(entry["amount_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{where} (date={date_iso}): 'amount_usd' must be a number, "
                f"got {entry.get('amount_usd')!r}"
            ) from exc
        # Zero declares nothing and cannot silence the detector either (it
        # runs on the flow-ADJUSTED return); non-finite would poison it.
        if not math.isfinite(amount) or amount == 0.0:
            raise ValueError(
                f"{where} (date={date_iso}): 'amount_usd' must be a finite "
                f"non-zero number, got {amount}"
            )
        if date_iso in out:
            raise ValueError(
                f"{where}: duplicate date {date_iso} — combine same-day "
                "transfers into one declared amount"
            )
        out[date_iso] = amount
    return out


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
        except (TypeError, ValueError) as exc:
            # Dropping the date silently would shorten the compared span
            # and hide whichever divergence that date carried.
            raise ValueError(
                f"non-numeric equity_usd on {d}: {eq!r} — repair the journal"
            ) from exc
        if live:
            live_eq.setdefault(d, val)
        elif c.get("outcome") == "success":
            dry_eq.setdefault(d, val)
    out = dict(live_eq)
    for d, val in dry_eq.items():
        if d not in live_dates:
            out[d] = val
    return out


def first_live_coverage_date(cycles: list[dict]) -> Optional[str]:
    """Earliest signal date evidencing live execution, or None.

    Two kinds of evidence: a live attempt's own signal date, and the date
    of a RETAINED FILL — a signal-less recovery record carries no cycle
    signal, but its fill proves real execution ran for that date.

    Sim intents before it were never given a chance to execute (the mirror
    image of real fills outside harness coverage), and the equity
    comparison starts here: observation-phase dry-run equity is not a
    tracked book.
    """
    dates = {
        d for c in cycles if is_live_attempt(c)
        for d in [_cycle_signal_date(c)] if d is not None
    }
    dates.update(e["date"] for e in real_order_events(cycles))
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


def sim_basket_returns(rows: list[HarnessLogRow]) -> dict[str, float]:
    """date -> the basket's own daily return, as the harness journaled it.

    Two data-quality guards, both because this feeds the capital-move
    yardstick and a WRONG yardstick suppresses a real tracking gap:

    * a null / non-numeric / non-finite ``daily_return`` is corruption,
      not a flat day — coercing it to 0.0 would erase a crash from the
      yardstick and reclassify the divergence as a transfer;
    * each row's ``daily_return`` spans from whatever row preceded it
      when it was written, so compounding is only sound while journaled
      order matches date order. A backfilled earlier row (``--asof``)
      overlaps its successor's span and would be double-counted.
    """
    out: dict[str, float] = {}
    highest = ""
    for r in rows:
        try:
            value = float(r.daily_return)
        except (TypeError, ValueError):
            raise ValueError(
                f"harness row {r.date}: daily_return={r.daily_return!r} is "
                "not numeric — the basket yardstick cannot be trusted"
            ) from None
        if not math.isfinite(value):
            raise ValueError(
                f"harness row {r.date}: daily_return={value} is not finite "
                "— the basket yardstick cannot be trusted"
            )
        if r.date < highest:
            raise ValueError(
                f"harness journal is not in date order: {r.date} follows "
                f"{highest}. A backfilled row's return overlaps its "
                "successor's, so spans cannot be compounded; re-sort the "
                "journal before reconciling"
            )
        highest = max(highest, r.date)
        out[r.date] = value
    return out


def _span_basket_return(
    basket_returns: dict[str, float], prev: str, cur: str,
) -> Optional[float]:
    """Basket return COMPOUNDED over ``prev``..``cur``, or None if unusable.

    An aligned step spans every day the two journals failed to intersect,
    and the compared returns span it too. Judging such a step against the
    ending date's one-day move alone shrinks the yardstick to a fraction
    of the move that actually happened — a multi-day drawdown then reads
    as a transfer and suppresses a real tracking gap.
    """
    # sim_basket_returns already rejected non-finite values and
    # out-of-order rows, so every entry here is a sound, non-overlapping
    # link of the chain.
    growth = 1.0
    seen = False
    for d, r in basket_returns.items():
        if not prev < d <= cur:
            continue
        growth *= 1.0 + r
        seen = True
    return growth - 1.0 if seen else None


def _is_capital_move(
    real_ret: float, sim_ret: float, basket_ret: Optional[float],
) -> bool:
    ceiling = _CAPITAL_DIVERGENCE_FLOOR
    # A missing / non-finite basket return leaves the stricter floor in
    # place — the conservative direction is to ask, not to assume.
    if basket_ret is not None and math.isfinite(basket_ret):
        ceiling = max(ceiling, _CAPITAL_BASKET_MULTIPLE * abs(basket_ret))
    return abs(real_ret - sim_ret) > ceiling


def compare_equity(
    real: dict[str, float],
    sim: dict[str, float],
    threshold_pct: float,
    *,
    start_date: Optional[str],
    capital_events: Optional[dict[str, float]] = None,
    basket_returns: Optional[dict[str, float]] = None,
) -> EquityTracking:
    """Real-vs-sim equity tracking from ``start_date`` on.

    ``start_date`` is live-execution coverage; ``None`` means live
    execution never ran and nothing is tracked — observation-phase
    dry-run equity is not a book, and the account was funded during it.
    """
    capital_events = capital_events or {}
    basket_returns = basket_returns or {}
    # ISO dates sort chronologically as strings.
    aligned = (
        [d for d in sorted(set(real) & set(sim)) if d >= start_date]
        if start_date is not None else []
    )
    if not aligned:
        # Journals exist but do not overlap yet — descriptive, not an error.
        return EquityTracking(
            n_real_days=len(real), n_sim_days=len(sim), n_aligned_days=0,
            overlap=None, cum_abs_return_diff=None, level_gap_pct=None,
            threshold_pct=threshold_pct, breached=False,
            tracking_start=start_date, capital_flows_applied=[],
            unexplained_moves=[],
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
    real_growth = 1.0
    sim_growth = 1.0
    applied: list[dict] = []
    unexplained: list[dict] = []
    for prev, cur in zip(aligned, aligned[1:]):
        # Flows are attributed to the STEP, not the date: an aligned span
        # can skip days, and the transfer still sits inside that step.
        # Each one is settled AT its own date, because the enlarged book
        # earns every market move that FOLLOWS the transfer — netting the
        # nominal amount at `cur` would book those moves as performance.
        flow_total = 0.0
        growth = 1.0
        anchor_date, anchor_eq = prev, real[prev]
        for d, amount in sorted(
            (d, a) for d, a in capital_events.items() if prev < d <= cur
        ):
            eq = real.get(d)
            if eq is None or not math.isfinite(eq) or eq <= 0.0:
                raise ValueError(
                    f"declared capital flow {amount:+.2f} USD on {d}: no "
                    f"usable real equity reading there ({eq!r}) — declare "
                    "the flow on the first signal date whose equity_usd "
                    "already includes the transfer"
                )
            funded = eq - amount
            if funded <= 0.0:
                raise ValueError(
                    f"declared capital flow {amount:+.2f} USD on {d} leaves "
                    f"non-positive equity ({eq} → {funded}) — check the "
                    "declaration"
                )
            growth *= funded / anchor_eq
            applied.append(
                {"from_date": anchor_date, "date": d, "amount_usd": amount}
            )
            flow_total += amount
            anchor_date, anchor_eq = d, eq
        real_ret = growth * (real[cur] / anchor_eq) - 1.0
        sim_ret = sim[cur] / sim[prev] - 1.0
        basket_ret = _span_basket_return(basket_returns, prev, cur)
        if _is_capital_move(real_ret, sim_ret, basket_ret):
            unexplained.append({
                "from_date": prev, "date": cur,
                "real_equity_from": real[prev], "real_equity": real[cur],
                "declared_flow_usd": flow_total,
                "real_return": real_ret, "sim_return": sim_ret,
                "basket_return": basket_ret,
            })
        cum += abs(real_ret - sim_ret)
        real_growth *= 1.0 + real_ret
        sim_growth *= 1.0 + sim_ret
    if unexplained:
        # A deposit is not a return and the journals cannot tell them
        # apart: report the refusal, never a conflated number.
        return EquityTracking(
            n_real_days=len(real), n_sim_days=len(sim),
            n_aligned_days=len(aligned), overlap=(first, last),
            cum_abs_return_diff=None, level_gap_pct=None,
            threshold_pct=threshold_pct, breached=False,
            tracking_start=start_date, capital_flows_applied=applied,
            unexplained_moves=unexplained,
        )
    # Levels are chained from the flow-adjusted daily returns rather than
    # read off the endpoints: real capital and the sim's virtual bankroll
    # differ by orders of magnitude by design, and a declared transfer
    # must move neither curve.
    gap_pct = (real_growth / sim_growth - 1.0) * 100.0
    return EquityTracking(
        n_real_days=len(real), n_sim_days=len(sim),
        n_aligned_days=len(aligned), overlap=(first, last),
        cum_abs_return_diff=cum, level_gap_pct=gap_pct,
        threshold_pct=threshold_pct,
        breached=bool(abs(gap_pct) > threshold_pct),
        tracking_start=start_date, capital_flows_applied=applied,
        unexplained_moves=[],
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
        intended = r.intended_trades or {}
        if not isinstance(intended, dict):
            raise ValueError(
                f"harness row {d}: intended_trades is {type(intended).__name__}, "
                "not an object — the expectations cannot be read"
            )
        trades: dict[str, float] = {}
        for a, v in intended.items():
            try:
                dw = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"harness row {d}: non-numeric intended trade "
                    f"{a}={v!r} — repair the journal"
                ) from exc
            # NaN fails `abs(dw) > eps`, so it would drop the intent out of
            # the expectations and a genuinely missed trade would read as
            # "no mismatch".
            if not math.isfinite(dw):
                raise ValueError(
                    f"harness row {d}: non-finite intended trade "
                    f"{a}={v!r} — the expectation would vanish silently; "
                    "repair the journal"
                )
            if abs(dw) > _INTENT_EPS:
                trades[a] = dw
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
            # The coid's embedded decision date is the authority: a cycle
            # that re-journals an OLDER order (reconstruction, state
            # recovery, a failed-cycle partial resolved days later) carries
            # its own fresh signal, which would re-date the late execution
            # and split it from the intent it belongs to. For the placing
            # cycle the two agree — signal.py's freshness guard pins asof
            # to run date − 1.
            date = _coid_signal_date(coid) or sig_date or ended_date
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


def _bootstrap_origin(
    event: dict,
    bootstrap_date: Optional[str],
    row: Optional[HarnessLogRow],
    expected: dict[str, dict[str, float]],
) -> Optional[str]:
    """Sim date whose unexecuted intent this catch-up buy executes, else None.

    The sim can step into the basket days before real execution starts
    trading; the first cycle that fills anything then buys that whole
    standing position at once. Such a fill is the delayed execution of an
    intent the sim really made — a distinct category from "the sim never
    wanted this", which is why all three pieces of evidence are required:
    the fill lands on the first date the real journal shows ANY fill (the
    live cron can run for days without trading, which is what production
    did), the sim still holds the asset, and an earlier sim row actually
    intended to buy it. Buys only — nothing was bought before that date
    for a sell to unwind.
    """
    if bootstrap_date is None or event["date"] != bootstrap_date:
        return None
    if event["side"] != "buy" or row is None:
        return None
    asset = _base_asset(event["symbol"])
    weights = row.target_weights if isinstance(row.target_weights, dict) else {}
    if as_float(weights.get(asset)) <= _INTENT_EPS:
        return None
    origins = [
        d for d, m in expected.items()
        if d < event["date"] and m.get(asset, 0.0) > _INTENT_EPS
    ]
    return max(origins) if origins else None


def check_transitions(
    cycles: list[dict], sim_rows: list[HarnessLogRow],
) -> TransitionCheck:
    expected = sim_expected_trades(sim_rows)
    events = real_order_events(cycles)
    skips = live_min_notional_skips(cycles)
    real_eq = real_equity_by_date(cycles)
    first_live = first_live_coverage_date(cycles)
    rows_by_date = {r.date: r for r in sim_rows}  # idempotent reruns: last wins

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
    extra_opposite: list[dict] = []
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
            other_side = [e for e in evs if e["side"] != want]
            # An opposite-side fill NEXT TO the expected one is invisible
            # to every other check: the intent reads as satisfied and the
            # date+asset is not "unexpected" either. Round-tripping a leg
            # costs two spreads and leaves the book off target. Without a
            # same-side fill the first one is already the wrong-direction
            # verdict below; the rest would still vanish.
            extras = other_side if same_side else other_side[1:]
            for e in extras:
                extra_opposite.append({
                    "date": d,
                    "symbol": e["symbol"],
                    "expected_side": want,
                    "actual_side": e["side"],
                    "filled_notional_quote": e["filled_notional_quote"],
                    "client_order_id": e["client_order_id"],
                    "cycle_id": e["cycle_id"],
                })
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
    # The first date real execution filled anything — the only date a
    # catch-up of the sim's standing position can land on.
    bootstrap_date = min((d for d, _ in by_key), default=None)
    unexpected: list[dict] = []
    bootstrap: list[dict] = []
    bootstrap_fills: dict[tuple, list[dict]] = {}
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
        if e["date"] not in covered_dates:
            out_of_coverage.append(record)
            continue
        origin = _bootstrap_origin(
            e, bootstrap_date, rows_by_date.get(e["date"]), expected,
        )
        if origin is not None:
            record["sim_intent_date"] = origin
            bootstrap.append(record)
            # Bootstrap excuses the DATE only — the sim intended earlier,
            # execution came later. Fill quality is never excused: a
            # catch-up that under-fills leaves the real book short of the
            # position the simulation holds.
            if _is_partial_fill(e):
                partials.append({
                    "date": e["date"],
                    "symbol": e["symbol"],
                    "side": e["side"],
                    "filled_amount": e["filled_amount"],
                    "intended_amount": e["intended_amount"],
                    "client_order_id": e["client_order_id"],
                })
            bootstrap_fills.setdefault(
                (e["date"], _base_asset(e["symbol"])), []
            ).append(e)
        else:
            unexpected.append(record)

    # A catch-up establishes the position the sim already STANDS in, so
    # the standing target weight is its size expectation — there is no
    # same-day intent to measure it against.
    for (d, asset), evs in sorted(bootstrap_fills.items()):
        weights = rows_by_date[d].target_weights
        mismatch = _size_mismatch(
            d, asset, as_float(weights.get(asset)), evs, real_eq,
        )
        if mismatch is not None:
            size_mismatches.append(mismatch)

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
        extra_opposite_fills=extra_opposite,
        unexpected_orders=unexpected,
        bootstrap_orders=bootstrap,
        wrong_market_fills=wrong_market,
        out_of_coverage_fills=out_of_coverage,
        pre_live_sim_trades=pre_live,
        coverage=coverage,
        live_coverage_start=first_live,
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def check_execution_tracking(
    real_journal: Path,
    sim_journal: Path,
    *,
    gap_threshold_pct: float = DEFAULT_GAP_THRESHOLD_PCT,
    capital_events_path: Optional[Path] = None,
) -> TrackingReport:
    cycles, real_stats = _read_real_cycles(real_journal)
    rows = _read_sim_rows(sim_journal)
    capital_events = (
        load_capital_events(capital_events_path)
        if capital_events_path is not None else {}
    )
    transitions = check_transitions(cycles, rows)
    equity = compare_equity(
        real_equity_by_date(cycles), sim_pre_trade_equity_by_date(rows),
        gap_threshold_pct,
        # Observation-phase dry-run equity is not a tracked book: the
        # account was funded during it, and a deposit is not a return.
        start_date=transitions.live_coverage_start,
        capital_events=capital_events,
        basket_returns=sim_basket_returns(rows),
    )

    if transitions.live_coverage_start is None:
        advisory = (
            "No live-execution coverage yet — the mainnet journal holds "
            "no live attempt and no retained fill; equity is not tracked "
            "over the observation phase."
        )
    elif equity.n_aligned_days == 0:
        advisory = (
            "No aligned dates yet — real and simulated journals do not "
            "overlap; nothing to compare."
        )
    elif equity.unexplained_moves:
        moves = ", ".join(
            f"{m['from_date']}..{m['date']} {m['real_return'] * 100:+.1f}% "
            f"vs sim {m['sim_return'] * 100:+.1f}%"
            for m in equity.unexplained_moves
        )
        advisory = (
            f"UNEXPLAINED EQUITY MOVE — {len(equity.unexplained_moves)} "
            f"step(s) no declared capital event explains ({moves}). "
            "A deposit or withdrawal is NOT a return and the journals "
            "cannot tell them apart, so no level gap is reported for this "
            "span. Declare the transfer via --capital-events (or "
            "investigate the book if none happened)."
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
    n_opposite = len(transitions.extra_opposite_fills)
    n_unexpected = len(transitions.unexpected_orders)
    if n_missing or n_wrong or n_partial or n_size or n_opposite or n_unexpected:
        advisory += (
            f" Trade mismatches: {n_missing} missing, {n_wrong} "
            f"wrong-direction, {n_partial} partial, {n_size} mis-sized, "
            f"{n_opposite} extra opposite-side, {n_unexpected} unexpected."
        )
    if transitions.wrong_market_fills:
        advisory += (
            f" WRONG MARKET: {len(transitions.wrong_market_fills)} real "
            f"fill(s) on a quote market the simulation never trades "
            f"(expected {SIM_QUOTE}) — configuration drift, not a match."
        )
    if transitions.bootstrap_orders:
        origins = sorted({
            e["sim_intent_date"] for e in transitions.bootstrap_orders
        })
        dates = sorted({e["date"] for e in transitions.bootstrap_orders})
        advisory += (
            f" BOOTSTRAP: {len(transitions.bootstrap_orders)} catch-up "
            f"fill(s) on {dates[0]} — real execution's first trades, "
            f"establishing the position the simulation stepped into on "
            f"{', '.join(origins)}. The delayed execution of an earlier "
            "intent, expected, not an unexpected order."
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
        capital_events_file=(
            str(capital_events_path) if capital_events_path is not None else None
        ),
        equity=equity,
        transitions=transitions,
        real_unknown_version_lines=real_stats.unknown_version_lines,
        advisory=advisory,
    )
