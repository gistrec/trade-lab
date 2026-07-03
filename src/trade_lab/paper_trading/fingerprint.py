"""Reference behavioral fingerprint — computed once, frozen, hash-pinned.

This module produces the **reference** fingerprint used by the live
behavioral monitor (``fingerprint_monitor.py``). The reference is
constructed from the FROZEN config's backtest on the venue-verified
window (2022-01-21 → 2026-05-28). The monitor compares live behavior
against this reference; it MUST NOT refit the reference from
incoming data — that would let slow live degradation silently widen
the bands and kill the detector.

What the fingerprint tracks
---------------------------
**Behavioral invariants, NOT realized return / Sharpe / equity.** A
correctly-behaving strategy can still lose money in an adverse
regime; trying to gate that out via Sharpe in a fingerprint would
re-introduce profit-as-pass through the back door, which is exactly
what the validation phase is built to avoid.

Metrics:

1. ``exposure_flip_freq_rolling_90d_annualized`` — count of ladder
   transitions in trailing 90 bars × (365/90). Distribution of this
   value at each bar.
2. ``rebalance_turnover_per_event`` — magnitude of ``|Δposition|``
   at each non-zero transition.
3. ``regime_gate_flip_freq_rolling_90d_annualized`` — count of
   SMA(200) gate flips (close > SMA crossing) per 90-bar window.
4. ``drawdown_profile`` — distribution of current-DD-from-peak across
   the verified window, plus the **explicit max historical DD** as
   the breach threshold (Headroom-style).
5. ``position_concentration`` — degenerate for equal-weight 7-asset
   (mechanically bounded by ``ladder/N``). Recorded as a structural
   note, NOT a percentile band.

Statistical honesty
-------------------
Rolling windows are autocorrelated: today's 90-day window overlaps
89 days with yesterday's. The percentile bands therefore describe
the **observed range of behaviour** on the historical sample. They
are **descriptive, not inferential**. A live breach is "outside the
historical envelope" — NOT "statistically significant at p < 0.05".
This is recorded in the findings and surfaced in any breach report.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SCHEMA_VERSION = "v1"
DEFAULT_ROLLING_WINDOW_DAYS = 90
DEFAULT_ANNUALIZATION_FACTOR = 365
DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class MetricBand:
    """Percentile bands + extremes for a single behavioral metric."""

    name: str
    description: str
    rolling_window_days: Optional[int]   # None for per-event metrics
    annualization_factor: Optional[int]  # None for per-event metrics
    n_observations: int
    percentiles: dict       # {"p05": ..., "p25": ..., ...}
    extremes: dict          # {"min": ..., "max": ..., "mean": ...}


@dataclass(frozen=True)
class DrawdownBand:
    """Drawdown is reported separately so the breach criterion is
    explicit (current_dd < max_historical_dd)."""

    n_observations: int
    percentiles: dict
    extremes: dict
    max_historical_dd: float
    breach_criterion: str = (
        "live_drawdown_from_peak < max_historical_dd "
        "(i.e., a deeper drawdown than the worst observed in the "
        "verified window 2022-01-21 → 2026-05-28)"
    )


@dataclass(frozen=True)
class PositionConcentrationNote:
    """For equal-weight N-asset basket the per-asset weight is
    mechanically ``ladder/N``, so a percentile band is meaningless.
    Recorded as a structural note rather than dropped silently."""

    status: str = "degenerate_for_equal_weight"
    mechanically_bounded_by: str = "ladder/N where N = len(assets)"


@dataclass(frozen=True)
class ReferenceFingerprint:
    schema_version: str
    frozen_config_hash: str
    window_start: str          # ISO date
    window_end: str            # ISO date
    n_bars: int
    rolling_window_days: int   # parameter shared by rolling-metrics
    annualization_factor: int

    exposure_flip_freq_rolling: MetricBand
    rebalance_turnover_per_event: MetricBand
    regime_gate_flip_freq_rolling: MetricBand
    drawdown_profile: DrawdownBand
    position_concentration: PositionConcentrationNote

    content_hash: str = ""     # SHA-256 of the canonical JSON of all other fields


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _percentiles_dict(values: np.ndarray) -> dict:
    return {f"p{p:02d}": float(np.percentile(values, p)) for p in DEFAULT_PERCENTILES}


def _extremes_dict(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def compute_reference_fingerprint(
    *,
    basket_close: pd.Series,
    positions: pd.Series,
    equity: pd.Series,
    sma_series: pd.Series,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    frozen_config_hash: str,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> ReferenceFingerprint:
    """Build the reference fingerprint from a single backtest run.

    Inputs are the same series the harness uses internally so the
    reference and live monitor measure the *same thing* — drift
    between the two definitions would be a silent calibration bug.
    """
    mask = (basket_close.index >= window_start) & (basket_close.index <= window_end)
    bc_w = basket_close[mask]
    pos_w = positions[mask]
    eq_w = equity[mask]
    sma_w = sma_series[mask]

    # M1: exposure-flip frequency, rolling 90d, annualized
    flip_indicator = (pos_w.diff().abs() > 1e-12).astype(int)
    roll_flips = (
        flip_indicator.rolling(rolling_window_days).sum()
        * (annualization_factor / rolling_window_days)
    ).dropna()
    m1 = MetricBand(
        name="exposure_flip_freq_rolling_90d_annualized",
        description=(
            "Count of ladder transitions in trailing 90 bars × (365/90). "
            "Each daily observation is a 90-day rolling window. "
            "Autocorrelated; bands descriptive only."
        ),
        rolling_window_days=rolling_window_days,
        annualization_factor=annualization_factor,
        n_observations=int(len(roll_flips)),
        percentiles=_percentiles_dict(roll_flips.to_numpy()),
        extremes=_extremes_dict(roll_flips.to_numpy()),
    )

    # M2: per-event rebalance turnover
    turnovers = pos_w.diff().abs()[flip_indicator > 0]
    m2 = MetricBand(
        name="rebalance_turnover_per_event",
        description=(
            "Magnitude of |Δposition| at each non-zero transition. "
            "Per-event distribution (NOT rolling)."
        ),
        rolling_window_days=None,
        annualization_factor=None,
        n_observations=int(len(turnovers)),
        percentiles=_percentiles_dict(turnovers.to_numpy()),
        extremes=_extremes_dict(turnovers.to_numpy()),
    )

    # M3: regime-gate flip frequency
    gate = (bc_w > sma_w).fillna(False).astype(int)
    gate_flips = (gate.diff().abs() > 0).astype(int)
    roll_gate = (
        gate_flips.rolling(rolling_window_days).sum()
        * (annualization_factor / rolling_window_days)
    ).dropna()
    m3 = MetricBand(
        name="regime_gate_flip_freq_rolling_90d_annualized",
        description=(
            "Count of SMA(200) gate flips (close crossing its SMA) per "
            "trailing 90 bars × (365/90). Autocorrelated."
        ),
        rolling_window_days=rolling_window_days,
        annualization_factor=annualization_factor,
        n_observations=int(len(roll_gate)),
        percentiles=_percentiles_dict(roll_gate.to_numpy()),
        extremes=_extremes_dict(roll_gate.to_numpy()),
    )

    # M4: drawdown profile + max historical DD as headroom anchor
    running_max = eq_w.cummax()
    drawdown = (eq_w / running_max - 1.0).to_numpy()
    dd_band = DrawdownBand(
        n_observations=int(len(drawdown)),
        percentiles=_percentiles_dict(drawdown),
        extremes=_extremes_dict(drawdown),
        max_historical_dd=float(np.min(drawdown)),
    )

    fp = ReferenceFingerprint(
        schema_version=SCHEMA_VERSION,
        frozen_config_hash=frozen_config_hash,
        window_start=window_start.strftime("%Y-%m-%d"),
        window_end=window_end.strftime("%Y-%m-%d"),
        n_bars=int(mask.sum()),
        rolling_window_days=rolling_window_days,
        annualization_factor=annualization_factor,
        exposure_flip_freq_rolling=m1,
        rebalance_turnover_per_event=m2,
        regime_gate_flip_freq_rolling=m3,
        drawdown_profile=dd_band,
        position_concentration=PositionConcentrationNote(),
        content_hash="",
    )
    # Recompute with the hash set
    h = fingerprint_content_hash(fp)
    return ReferenceFingerprint(**{**asdict(fp), "content_hash": h,
                                   "exposure_flip_freq_rolling": fp.exposure_flip_freq_rolling,
                                   "rebalance_turnover_per_event": fp.rebalance_turnover_per_event,
                                   "regime_gate_flip_freq_rolling": fp.regime_gate_flip_freq_rolling,
                                   "drawdown_profile": fp.drawdown_profile,
                                   "position_concentration": fp.position_concentration})


def fingerprint_content_hash(fp: ReferenceFingerprint) -> str:
    """SHA-256 of the canonical JSON of the fingerprint (excluding the
    ``content_hash`` field itself)."""
    payload = asdict(fp)
    payload.pop("content_hash", None)
    js = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Persistence — frozen artifact contract
# ---------------------------------------------------------------------------

def save_reference(fp: ReferenceFingerprint, path: Path) -> None:
    """Write the fingerprint atomically. The on-disk file IS the
    versioned frozen artifact — committing it to git is what makes
    the reference reproducible across operators."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    js = json.dumps(asdict(fp), sort_keys=True, indent=2, default=str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(js)
    tmp.rename(path)


def load_reference(path: Path) -> ReferenceFingerprint:
    """Read + verify the content-hash on load.

    A mismatch means the file was edited after writing — either an
    operator hand-edited it (intentionally or not) or git surgery
    left it in an inconsistent state. Either case must fail loud
    rather than silently feeding wrong bands to the monitor."""
    path = Path(path)
    data = json.loads(path.read_text())
    # Rehydrate nested dataclasses
    data["exposure_flip_freq_rolling"] = MetricBand(**data["exposure_flip_freq_rolling"])
    data["rebalance_turnover_per_event"] = MetricBand(**data["rebalance_turnover_per_event"])
    data["regime_gate_flip_freq_rolling"] = MetricBand(**data["regime_gate_flip_freq_rolling"])
    data["drawdown_profile"] = DrawdownBand(**data["drawdown_profile"])
    data["position_concentration"] = PositionConcentrationNote(**data["position_concentration"])
    fp = ReferenceFingerprint(**data)
    expected = fingerprint_content_hash(fp)
    if fp.content_hash != expected:
        raise ValueError(
            f"Reference fingerprint content-hash mismatch: "
            f"on-disk={fp.content_hash} recomputed={expected}. "
            f"The file at {path} was edited after writing — refusing "
            f"to use stale bands."
        )
    return fp
