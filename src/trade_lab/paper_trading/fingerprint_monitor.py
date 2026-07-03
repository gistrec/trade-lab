"""Live behavioral monitor — compares live journal against frozen reference.

What this is
============
The monitor reads the append-only journal written by the harness
(``logs/journal.jsonl``) and the frozen reference fingerprint
(``paper_trading/fingerprint/reference_fingerprint.json``). It
computes the same behavioral metrics on the live data and reports
where (and for how long) the live values sit outside the reference
percentile bands.

What this is NOT
================
* **Not** an auto-killer. The monitor reports breaches and lets the
  operator + the look-ahead detector (Step 4) decide what to do.
* **Not** a re-fitter of the reference. The reference is frozen and
  loaded as-is. The monitor never adjusts the bands with incoming
  data, regardless of how long it has been running.
* **Not** a profit detector. The journal stores realized return data
  but the monitor does not band-check it — see
  ``fingerprint.py`` for why.

Breach semantics (descriptive, not p-values)
============================================
For each rolling 90d metric:
* A day is "breached" if the live value is outside ``[p05, p95]``.
* A breach is "sustained" if the metric is breached for
  ``sustained_days_threshold`` consecutive days (default 7).
* A "multi-metric day" is a day where 3+ of the tracked metrics are
  simultaneously breached.

For drawdown:
* The breach criterion is ``live_drawdown < max_historical_dd``.
  No bands, just the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .fingerprint import (
    MetricBand,
    load_reference,
)
from .journal import HarnessLogRow, read_log


DEFAULT_SUSTAINED_DAYS = 7
DEFAULT_MULTI_METRIC_THRESHOLD = 3


@dataclass(frozen=True)
class MetricLiveStatus:
    """Per-metric snapshot of live behaviour vs reference."""

    name: str
    n_live_observations: int
    latest_value: Optional[float]
    p05: float
    p95: float
    n_days_breached_total: int
    longest_consecutive_breach: int
    currently_breached: bool
    currently_consecutive_breach: int


@dataclass(frozen=True)
class DrawdownLiveStatus:
    n_live_observations: int
    latest_drawdown: float
    max_live_drawdown: float
    reference_max_historical_dd: float
    headroom_pp: float
    breached: bool


@dataclass(frozen=True)
class BreachReport:
    """Structured monitor output. The operator reads this; no
    automated action is taken."""

    journal_n_rows: int
    journal_window: Optional[tuple]      # (first_date, last_date) ISO
    reference_window: tuple              # (start, end) ISO
    reference_content_hash: str
    rolling_window_days: int

    exposure_flip: MetricLiveStatus
    regime_gate_flip: MetricLiveStatus
    rebalance_turnover_per_event: dict   # event-level summary
    drawdown: DrawdownLiveStatus

    multi_metric_days: int               # days with >=3 metrics simultaneously breached
    overall_sustained_breach: bool       # any single metric breached for >=N days
    overall_multi_metric_breach: bool    # any day with multi-metric breach
    advisory: str                        # short human-readable summary


# ---------------------------------------------------------------------------
# Live metric computation from the journal
# ---------------------------------------------------------------------------

def _rolling_flip_count(series: pd.Series, window_days: int, ann: int) -> pd.Series:
    indicator = (series.diff().abs() > 1e-12).astype(int)
    return (indicator.rolling(window_days).sum() * (ann / window_days)).dropna()


def _consecutive_run_end(breach_mask: pd.Series) -> int:
    """Length of the run of breached days ending at the last bar.
    Returns 0 if the last bar is not breached."""
    if breach_mask.empty or not bool(breach_mask.iloc[-1]):
        return 0
    n = 0
    for v in reversed(breach_mask.to_list()):
        if bool(v):
            n += 1
        else:
            break
    return n


def _longest_consecutive_run(breach_mask: pd.Series) -> int:
    """Longest run of consecutive True values in the series."""
    longest = 0
    current = 0
    for v in breach_mask.to_list():
        if bool(v):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _metric_status_from_series(
    name: str,
    live_series: pd.Series,
    band: MetricBand,
) -> MetricLiveStatus:
    p05 = float(band.percentiles["p05"])
    p95 = float(band.percentiles["p95"])
    breach_mask = (live_series < p05) | (live_series > p95)
    return MetricLiveStatus(
        name=name,
        n_live_observations=int(len(live_series)),
        latest_value=float(live_series.iloc[-1]) if not live_series.empty else None,
        p05=p05,
        p95=p95,
        n_days_breached_total=int(breach_mask.sum()),
        longest_consecutive_breach=_longest_consecutive_run(breach_mask),
        currently_breached=bool(breach_mask.iloc[-1]) if not breach_mask.empty else False,
        currently_consecutive_breach=_consecutive_run_end(breach_mask),
    )


def compute_live_metrics_from_journal(
    rows: list[HarnessLogRow],
    *,
    rolling_window_days: int,
    annualization_factor: int,
) -> dict:
    """Reconstruct the same 4 behavioral series from journal rows."""
    if not rows:
        return {
            "ladder": pd.Series(dtype=float),
            "sma_gate": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "rebalance_events": pd.Series(dtype=float),
            "roll_flip": pd.Series(dtype=float),
            "roll_gate": pd.Series(dtype=float),
        }
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r.date, tz="UTC") for r in rows], name="date",
    )
    ladder = pd.Series([r.ladder_state for r in rows], index=idx, name="ladder")
    gate = pd.Series(
        [1.0 if r.sma_gate_open else 0.0 for r in rows], index=idx, name="sma_gate",
    )
    equity = pd.Series(
        [r.portfolio_equity for r in rows], index=idx, name="equity",
    )
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).rename("drawdown")

    # rebalance events: per-event |Δposition|
    diff = ladder.diff().abs()
    events = diff[diff > 1e-12]

    roll_flip = _rolling_flip_count(ladder, rolling_window_days, annualization_factor)
    roll_gate = _rolling_flip_count(gate, rolling_window_days, annualization_factor)
    return {
        "ladder": ladder,
        "sma_gate": gate,
        "drawdown": drawdown,
        "rebalance_events": events,
        "roll_flip": roll_flip,
        "roll_gate": roll_gate,
    }


# ---------------------------------------------------------------------------
# Top-level monitor
# ---------------------------------------------------------------------------

def check_journal_against_reference(
    log_path: Path,
    reference_path: Path,
    *,
    sustained_days_threshold: int = DEFAULT_SUSTAINED_DAYS,
    multi_metric_threshold: int = DEFAULT_MULTI_METRIC_THRESHOLD,
) -> BreachReport:
    """Read the live journal, load the frozen reference, return a
    structured breach report. Side-effect-free."""

    reference = load_reference(reference_path)
    rows = read_log(log_path)

    journal_window: Optional[tuple] = None
    if rows:
        journal_window = (rows[0].date, rows[-1].date)

    live = compute_live_metrics_from_journal(
        rows,
        rolling_window_days=reference.rolling_window_days,
        annualization_factor=reference.annualization_factor,
    )

    flip_status = _metric_status_from_series(
        reference.exposure_flip_freq_rolling.name,
        live["roll_flip"],
        reference.exposure_flip_freq_rolling,
    )
    gate_status = _metric_status_from_series(
        reference.regime_gate_flip_freq_rolling.name,
        live["roll_gate"],
        reference.regime_gate_flip_freq_rolling,
    )

    # Drawdown
    dd_band = reference.drawdown_profile
    if live["drawdown"].empty:
        dd_status = DrawdownLiveStatus(
            n_live_observations=0,
            latest_drawdown=0.0,
            max_live_drawdown=0.0,
            reference_max_historical_dd=dd_band.max_historical_dd,
            headroom_pp=0.0,
            breached=False,
        )
    else:
        latest_dd = float(live["drawdown"].iloc[-1])
        max_live_dd = float(live["drawdown"].min())
        headroom = float(max_live_dd - dd_band.max_historical_dd)  # negative live − negative ref
        dd_status = DrawdownLiveStatus(
            n_live_observations=int(len(live["drawdown"])),
            latest_drawdown=latest_dd,
            max_live_drawdown=max_live_dd,
            reference_max_historical_dd=dd_band.max_historical_dd,
            headroom_pp=headroom * 100.0,
            breached=bool(max_live_dd < dd_band.max_historical_dd),
        )

    # Event-level turnover summary (no consecutive-day semantics — events are sparse)
    ev = live["rebalance_events"]
    rb_band = reference.rebalance_turnover_per_event
    if ev.empty:
        turnover_summary = {
            "n_live_events": 0,
            "events_outside_reference_p05_p95": 0,
            "reference_p05": float(rb_band.percentiles["p05"]),
            "reference_p95": float(rb_band.percentiles["p95"]),
        }
    else:
        p05 = float(rb_band.percentiles["p05"])
        p95 = float(rb_band.percentiles["p95"])
        outside = int(((ev < p05) | (ev > p95)).sum())
        turnover_summary = {
            "n_live_events": int(len(ev)),
            "events_outside_reference_p05_p95": outside,
            "reference_p05": p05,
            "reference_p95": p95,
        }

    # Multi-metric day count: align both rolling-metric breach masks
    if not live["roll_flip"].empty and not live["roll_gate"].empty:
        flip_breach = (live["roll_flip"] < flip_status.p05) | (live["roll_flip"] > flip_status.p95)
        gate_breach = (live["roll_gate"] < gate_status.p05) | (live["roll_gate"] > gate_status.p95)
        dd_breach_series = live["drawdown"] < dd_band.max_historical_dd
        # Align on intersection
        common = flip_breach.index.intersection(gate_breach.index).intersection(dd_breach_series.index)
        if not common.empty:
            n_breached_per_day = (
                flip_breach.reindex(common).astype(int)
                + gate_breach.reindex(common).astype(int)
                + dd_breach_series.reindex(common).astype(int)
            )
            multi_metric_days = int((n_breached_per_day >= multi_metric_threshold).sum())
        else:
            multi_metric_days = 0
    else:
        multi_metric_days = 0

    overall_sustained = (
        flip_status.longest_consecutive_breach >= sustained_days_threshold
        or gate_status.longest_consecutive_breach >= sustained_days_threshold
    )
    overall_multi = multi_metric_days > 0

    if dd_status.breached:
        advisory = (
            "DRAWDOWN BREACH — live drawdown deeper than max historical DD. "
            "Forward to Step-4 look-ahead detector + operator review."
        )
    elif overall_sustained:
        advisory = (
            f"Sustained breach on a behavioral metric "
            f"(>={sustained_days_threshold} consecutive days outside [p05, p95]). "
            "Forward to operator review."
        )
    elif overall_multi:
        advisory = (
            f"Multi-metric breach detected on {multi_metric_days} day(s) "
            f"(>={multi_metric_threshold} metrics simultaneously outside band). "
            "Forward to operator review."
        )
    elif rows and (flip_status.currently_breached or gate_status.currently_breached):
        advisory = (
            "Single-day single-metric breach. Likely noise; watch but do not act."
        )
    elif not rows:
        advisory = (
            "Journal empty — no live data yet. Bootstrap the harness."
        )
    elif len(rows) < reference.rolling_window_days:
        advisory = (
            f"Bootstrap — {len(rows)} / {reference.rolling_window_days} bars "
            "of journal history. Rolling metrics not yet evaluable."
        )
    else:
        advisory = "Within historical envelope."

    return BreachReport(
        journal_n_rows=len(rows),
        journal_window=journal_window,
        reference_window=(reference.window_start, reference.window_end),
        reference_content_hash=reference.content_hash,
        rolling_window_days=reference.rolling_window_days,
        exposure_flip=flip_status,
        regime_gate_flip=gate_status,
        rebalance_turnover_per_event=turnover_summary,
        drawdown=dd_status,
        multi_metric_days=multi_metric_days,
        overall_sustained_breach=overall_sustained,
        overall_multi_metric_breach=overall_multi,
        advisory=advisory,
    )
