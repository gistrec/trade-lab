"""Donchian trend ensemble with SMA filter and volatility targeting.

This strategy combines three well-known building blocks that have decent
out-of-sample evidence on their own, without adding ML, short-term mean
reversion, or stat-arb on top:

1. **Donchian breakout ensemble.** For each of several lookbacks, go long
   on a close that exceeds the prior-N-bar high; go flat on a close that
   falls below the prior-N-bar low; otherwise hold state. The raw signal
   is the *mean* of the per-lookback long states, so partial agreement
   produces partial exposure (0, 0.33, 0.66, 1.0 for three lookbacks).

2. **SMA trend filter.** Only allow long exposure while close > SMA for
   every filter lookback (default 100 and 200). Sits in cash through
   bear regimes by construction.

3. **Volatility targeting.** Scale the raw signal by
   ``annual_vol_target / realized_vol_annual``. Reduces exposure when
   the asset gets risky and bumps it up when it's calm — capped at
   ``max_position_size`` so we stay spot-only.

4. (Optional) **BTC market gate.** For altcoins, only allow exposure
   when BTC close > BTC SMA(200). Pass ``btc_candles`` to enable; it's
   off by default (and obviously redundant when running on BTC itself).

No parameter optimization or ML. All inputs use only data available at
the close of bar N to set the signal for bar N. The engine then shifts
signals by one bar, so a signal at bar N affects positions from bar
N+1 onward.

This is a research candidate, not a profitable claim — see
``docs/`` and the README for the validation workflow.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .base import Strategy


def _coerce_int_sequence(value, name: str) -> tuple[int, ...]:
    """Allow CLI-style ``"20,50,100"`` strings as well as proper sequences."""
    if isinstance(value, str):
        parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        parsed = [int(x) for x in value]
    if not parsed:
        raise ValueError(f"{name} must be a non-empty sequence of positive ints")
    if any(p < 1 for p in parsed):
        raise ValueError(f"{name} must contain only positive integers")
    return tuple(parsed)


class DonchianTrendEnsembleStrategy(Strategy):
    """Donchian breakout ensemble + SMA filter + vol targeting.

    See module docstring for the full rule set.
    """

    name = "donchian_trend"

    def __init__(
        self,
        donchian_lookbacks: Iterable[int] | str = (20, 50, 100),
        sma_filter_periods: Iterable[int] | str = (100, 200),
        vol_lookback: int = 30,
        annual_vol_target: float = 0.25,
        annualization_factor: int = 365,
        max_position_size: float = 1.0,
        btc_candles: Optional[pd.DataFrame] = None,
        btc_gate_sma_period: int = 200,
    ) -> None:
        self.donchian_lookbacks = _coerce_int_sequence(
            donchian_lookbacks, "donchian_lookbacks"
        )
        self.sma_filter_periods = _coerce_int_sequence(
            sma_filter_periods, "sma_filter_periods"
        )
        if vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if annual_vol_target <= 0:
            raise ValueError("annual_vol_target must be positive")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")
        if not 0 < max_position_size <= 1:
            raise ValueError(
                "max_position_size must be in (0, 1] for spot-only mode"
            )
        if btc_gate_sma_period < 1:
            raise ValueError("btc_gate_sma_period must be positive")

        self.vol_lookback = int(vol_lookback)
        self.annual_vol_target = float(annual_vol_target)
        self.annualization_factor = int(annualization_factor)
        self.max_position_size = float(max_position_size)
        self.btc_candles = btc_candles
        self.btc_gate_sma_period = int(btc_gate_sma_period)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"].astype(float)

        donchian = self._donchian_ensemble(close)
        sma_ok = self._sma_filter(close)
        raw_signal = donchian.where(sma_ok, 0.0)

        if self.btc_candles is not None:
            btc_gate = self._btc_gate(close.index)
            raw_signal = raw_signal.where(btc_gate, 0.0)

        vol_weight = self._vol_weight(close)
        position = (raw_signal * vol_weight).clip(
            lower=0.0, upper=self.max_position_size
        )
        return position.fillna(0.0)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _donchian_ensemble(self, close: pd.Series) -> pd.Series:
        """Average long-state across the configured Donchian lookbacks."""
        components: list[pd.Series] = []
        for lookback in self.donchian_lookbacks:
            # Critical: shift(1) so the high/low never includes today's close.
            # That's how "no lookahead in Donchian thresholds" is enforced.
            prev_high = close.rolling(lookback).max().shift(1)
            prev_low = close.rolling(lookback).min().shift(1)

            state = pd.Series(np.nan, index=close.index, dtype=float)
            state[close > prev_high] = 1.0
            state[close < prev_low] = 0.0
            state = state.ffill().fillna(0.0)
            components.append(state)

        # Mean of long-states gives a clean 0..1 ladder; with three lookbacks
        # the levels are 0, 1/3, 2/3, 1 (rounded).
        stacked = pd.concat(components, axis=1)
        return stacked.mean(axis=1)

    def _sma_filter(self, close: pd.Series) -> pd.Series:
        """Return True only where close exceeds every SMA filter lookback."""
        ok = pd.Series(True, index=close.index)
        for period in self.sma_filter_periods:
            sma = close.rolling(period).mean()
            cond = close > sma
            # Treat warm-up bars (SMA still NaN) as filter-fails-shut so the
            # strategy stays flat instead of relying on undefined data.
            cond[sma.isna()] = False
            ok = ok & cond
        return ok

    def _btc_gate(self, target_index: pd.Index) -> pd.Series:
        """Boolean series aligned to ``target_index``: True iff BTC > SMA(N)."""
        btc_close = self.btc_candles["close"].astype(float)
        btc_sma = btc_close.rolling(self.btc_gate_sma_period).mean()
        gate = (btc_close > btc_sma) & btc_sma.notna()
        # Align to the target asset's index; pad-forward for any gaps.
        return gate.reindex(target_index, method="ffill").fillna(False)

    def _vol_weight(self, close: pd.Series) -> pd.Series:
        """``target_vol / realized_vol``, with safe handling of NaN / zero vol."""
        daily_returns = close.pct_change()
        realized_vol_daily = daily_returns.rolling(self.vol_lookback).std()
        realized_vol_annual = realized_vol_daily * np.sqrt(self.annualization_factor)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = self.annual_vol_target / realized_vol_annual
        # inf (zero realized vol) and NaN (insufficient history) both map
        # to zero exposure — never silently lever up.
        return weight.replace([np.inf, -np.inf], np.nan).fillna(0.0)
