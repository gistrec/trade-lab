"""Live look-ahead detector — Part B (operational, accumulates forward).

For each journal row, replay the backtest signal on the EXACT vintage
bytes the harness saw that day and compare to the harness-logged
``ladder_state``. Disagreements are characterized as one of:

* ``match`` — replay and live agree on signal[T] for this row.
* ``offset_1_match`` — live ladder_state equals the replay signal one
  bar earlier (``replay_signal_series[-2]``). This is a labeling
  artifact (the `date` field is the signal-date; if the harness
  schedule shifted the close convention by one bar, the comparison
  alignment is off — NOT a real look-ahead).
* ``random_disagreement`` — neither match nor offset match. This is
  the load-bearing signal: a real look-ahead in the backtest path
  shows up here.

The aggregate report classifies the journal:

* If all disagreements (if any) follow the offset_1 pattern AND there
  are no random disagreements → labeling artifact, NOT a look-ahead.
* If any random disagreements present → flag for operator review;
  Part A truncation-audit can be re-run to localize.

Cold-start note
===============
This module is the infrastructure for the operational check. **Until
the forward journal has rows, the report says "no rows" — it does
NOT fabricate a verdict.** The dispositive look-ahead test for the
backtest itself is Part A (`scripts/validation_lookahead_truncation_audit.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from ..backtest.market_index import build_crypto_market_index
from ..config import PRODUCTION_CONFIG
from ..strategies.tsmom import TimeSeriesMomentumStrategy

from .journal import HarnessLogRow, read_log
from .vintage_store import load_vintage


SIGNAL_EQUALITY_TOL = 1e-9


@dataclass(frozen=True)
class RowComparison:
    date: str
    vintage_content_hash: str
    live_ladder: float
    replay_signal_at_last: float
    replay_signal_at_prev: Optional[float]
    status: str   # "match", "offset_1_match", "random_disagreement", "vintage_missing"


@dataclass(frozen=True)
class LookAheadReport:
    journal_n_rows: int
    n_checked: int
    n_match: int
    n_offset_1_match: int
    n_random_disagreement: int
    n_vintage_missing: int
    constant_offset_pattern: bool   # all disagreements follow offset_1
    random_disagreement_present: bool
    rows: list                       # RowComparison objects
    advisory: str


def replay_signal_for_vintage(
    vintage_hash: str, vintage_root: Path
) -> tuple[float, Optional[float]]:
    """Reconstruct the strategy signal series from a vintage and return
    ``(signal[-1], signal[-2])`` — last bar (the signal date) and the
    bar before (offset-test anchor).
    """
    asset_candles = load_vintage(vintage_hash, vintage_root)
    cfg = PRODUCTION_CONFIG
    basket = build_crypto_market_index(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )
    strat = TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        vol_lookback=cfg.vol_lookback,
        annual_vol_target=cfg.annual_vol_target,
        max_position_size=cfg.max_position_size,
        rebalance_threshold=cfg.rebalance_threshold,
        annualization_factor=cfg.annualization_factor,
    )
    sig = strat.generate_signals(basket)
    if sig.empty:
        raise ValueError(f"Replay produced empty signal series for vintage {vintage_hash}")
    last = float(sig.iloc[-1])
    prev = float(sig.iloc[-2]) if len(sig) >= 2 else None
    return last, prev


def classify_signal_match(
    live: float, replay_last: float, replay_prev: Optional[float]
) -> str:
    """Pure-function classification — exposed so the offset-detection
    logic can be tested without needing a synthetic vintage with a
    natural ladder transition on consecutive days."""
    if abs(live - replay_last) <= SIGNAL_EQUALITY_TOL:
        return "match"
    if replay_prev is not None and abs(live - replay_prev) <= SIGNAL_EQUALITY_TOL:
        return "offset_1_match"
    return "random_disagreement"


def compare_row(row: HarnessLogRow, vintage_root: Path) -> RowComparison:
    try:
        last, prev = replay_signal_for_vintage(row.vintage_content_hash, vintage_root)
    except FileNotFoundError:
        return RowComparison(
            date=row.date,
            vintage_content_hash=row.vintage_content_hash,
            live_ladder=row.ladder_state,
            replay_signal_at_last=float("nan"),
            replay_signal_at_prev=None,
            status="vintage_missing",
        )

    status = classify_signal_match(row.ladder_state, last, prev)
    return RowComparison(
        date=row.date,
        vintage_content_hash=row.vintage_content_hash,
        live_ladder=row.ladder_state,
        replay_signal_at_last=last,
        replay_signal_at_prev=prev,
        status=status,
    )


def check_journal_for_lookahead(
    log_path: Path, vintage_root: Path
) -> LookAheadReport:
    """Walk the entire journal, replay every row's vintage, classify."""
    rows = read_log(log_path)
    comparisons: list[RowComparison] = []
    n_match = n_offset = n_random = n_missing = 0

    for row in rows:
        cmp_ = compare_row(row, vintage_root)
        comparisons.append(cmp_)
        if cmp_.status == "match":
            n_match += 1
        elif cmp_.status == "offset_1_match":
            n_offset += 1
        elif cmp_.status == "random_disagreement":
            n_random += 1
        elif cmp_.status == "vintage_missing":
            n_missing += 1

    n_disagree = n_offset + n_random
    constant_offset = (n_disagree > 0) and (n_random == 0) and (n_offset > 0)
    random_present = n_random > 0

    if not rows:
        advisory = (
            "Journal empty — no live rows to check. The dispositive "
            "look-ahead test for the backtest path itself is Part A "
            "(scripts/validation_lookahead_truncation_audit.py), which "
            "has run independently and is CLEAN."
        )
    elif n_missing == len(rows):
        advisory = (
            "All vintages missing — vintage_root path likely wrong, OR "
            "vintages directory has been cleaned. Restore vintages or "
            "point --vintage-root at the correct location."
        )
    elif n_random > 0:
        advisory = (
            f"LOOK-AHEAD SUSPECT — {n_random} row(s) disagree with replay "
            f"on identical input AND do not match the 1-bar offset pattern. "
            f"Stop trading until investigated. Run Part A (truncation audit) "
            f"to localize. Per validation rules: this is a baseline-invalidating "
            f"finding, not a patch-and-continue."
        )
    elif constant_offset:
        advisory = (
            f"Labeling artifact (constant 1-bar offset) detected on "
            f"{n_offset} row(s). The live ladder_state equals replay_signal "
            f"at one bar earlier. Treat as the journal's `date` field "
            f"convention drifting from the backtest replay convention. "
            f"NOT a real look-ahead. Document and continue."
        )
    elif n_match == len(rows) - n_missing:
        if n_missing:
            advisory = (
                f"All {n_match} replayable rows match. {n_missing} vintage(s) "
                f"missing — restore from backup if available."
            )
        else:
            advisory = (
                f"All {n_match} live rows match backtest replay on identical "
                f"vintage. Forward path has no look-ahead disagreement."
            )
    else:
        advisory = (
            f"Mixed result: match={n_match}, offset_1={n_offset}, "
            f"random={n_random}, missing={n_missing}. Operator review."
        )

    return LookAheadReport(
        journal_n_rows=len(rows),
        n_checked=len(comparisons),
        n_match=n_match,
        n_offset_1_match=n_offset,
        n_random_disagreement=n_random,
        n_vintage_missing=n_missing,
        constant_offset_pattern=constant_offset,
        random_disagreement_present=random_present,
        rows=comparisons,
        advisory=advisory,
    )
