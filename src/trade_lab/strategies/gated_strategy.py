"""Generic regime-gate wrapper for any single-asset Strategy.

The wrapper takes an inner :class:`Strategy` plus a pre-computed
boolean gate ``Series`` and multiplies the inner signal by the gate.
This decouples *what to gate on* (computed externally — breadth,
macro, custom signal) from *what the strategy does* internally.

`compute_breadth_gate` is the canonical example: given a universe of
asset candles, it returns ``True`` on bars where at least
``threshold`` fraction of the universe is above its own SMA. The
deep-research-report.md (Han et al., AQR) flagged breadth as a
useful complement to a single-instrument regime SMA — a single
market leader (BTC) above its SMA while the rest of the universe is
below it is a weaker risk-on signal than a broad-based above-SMA
condition.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .base import Strategy


def compute_breadth_gate(
    asset_candles: Mapping[str, pd.DataFrame],
    *,
    sma_period: int = 200,
    threshold: float = 0.5,
) -> pd.Series:
    """Boolean gate: True when ``threshold`` fraction of universe > own SMA.

    Implementation details:

    * SMA is computed *per asset*, then we count the number of assets
      whose current close exceeds their own SMA. ``threshold`` is a
      fraction in ``[0, 1]``.
    * Assets in their SMA warmup window (NaN) are excluded from the
      denominator — we never silently count them as "below SMA",
      which would bias breadth downward in early bars.
    * The gate uses only data at the current bar; the engine's
      one-bar shift is what prevents look-ahead at the position
      level.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if sma_period < 1:
        raise ValueError("sma_period must be >= 1")
    if not asset_candles:
        return pd.Series(dtype=bool)

    closes = pd.concat(
        {k: v["close"].astype(float) for k, v in asset_candles.items()},
        axis=1,
    ).sort_index()
    if closes.empty:
        return pd.Series(dtype=bool)

    sma = closes.rolling(sma_period).mean()
    active = sma.notna()
    above_sma = (closes > sma) & active

    n_active = active.sum(axis=1)
    n_above = above_sma.sum(axis=1)
    breadth = pd.Series(0.0, index=closes.index)
    nonzero = n_active > 0
    breadth.loc[nonzero] = n_above.loc[nonzero] / n_active.loc[nonzero]
    return breadth >= threshold


class GatedStrategy(Strategy):
    """Multiply ``inner``'s long-position signal by an external gate Series.

    The gate is supplied at construction time. On every bar, the gate
    value (``True``/``False`` or 0/1) is multiplied against the inner
    signal — when the gate is False the position is forced flat. The
    gate is reindexed onto the candles' index with forward-fill, so
    callers can pass a gate computed on a different but
    same-frequency frame.
    """

    name = "gated"

    def __init__(self, inner: Strategy, gate: pd.Series, *, gate_name: str = "gate") -> None:
        if gate is None:
            raise ValueError("gate must be a pd.Series, got None")
        self.inner = inner
        self.gate = gate.astype(bool)
        # Surface a descriptive label so CSV outputs and logs are clear.
        inner_name = getattr(inner, "name", inner.__class__.__name__)
        self.name = f"{gate_name}({inner_name})"

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        signal = self.inner.generate_signals(candles)
        signal = signal.reindex(candles.index).fillna(0.0).astype(float)
        aligned_gate = self.gate.reindex(candles.index, method="ffill").fillna(False)
        return signal.where(aligned_gate, 0.0)
