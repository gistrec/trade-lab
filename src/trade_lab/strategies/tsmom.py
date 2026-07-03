"""Time-series momentum (TSMOM), long-only, with optional vol targeting.

The signal is the sign of trailing returns over one or more lookback
windows. Two foundational references:

* Moskowitz, Ooi, Pedersen (2012). *Time Series Momentum*. JFE 104(2).
  Documents significant TSMOM in 58 liquid futures across asset classes
  on 1-12 month horizons.
* Liu & Tsyvinski (2021). *Risks and Returns of Cryptocurrency*. RFS 34(6).
  Replicates the effect on BTC, ETH and XRP on 1-4 week horizons.

For a long-only spot mandate we drop the short leg: each lookback's
contribution is ``1`` when the trailing return is positive and ``0``
otherwise. With multiple lookbacks the raw signal is the mean of the
per-lookback contributions, producing a clean ladder
``{0, 1/k, 2/k, ..., 1}`` exactly the way the Donchian ensemble does.

The other two layers are shared with ``DonchianTrendEnsembleStrategy``
on purpose, so cross-strategy comparisons isolate the *signal* and keep
sizing/filtering identical:

* Optional SMA regime filter — ``close > SMA(period)`` for every
  configured period must hold or the signal is zeroed out.
* Optional volatility targeting — multiply by
  ``annual_vol_target / realized_vol_annual`` (capped at
  ``max_position_size``), so the position shrinks in turbulent regimes.

All inputs use only data available at the close of bar N. The engine
shifts the resulting signal by one bar before applying it as the
position for bar N+1.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .donchian_trend import _coerce_bool, _coerce_int_sequence


class TimeSeriesMomentumStrategy(Strategy):
    """Multi-lookback long-only TSMOM with optional regime + vol layers.

    Default lookbacks ``(30, 90, 180, 365)`` correspond to ~1, 3, 6 and
    12 months of daily candles. The literature does not single out a
    "best" lookback; the ensemble averages over them to avoid an
    arbitrary choice.
    """

    name = "tsmom"

    def __init__(
        self,
        lookbacks: Iterable[int] | str = (30, 90, 180, 365),
        sma_filter_periods: Iterable[int] | str = (200,),
        vol_lookback: int = 30,
        annual_vol_target: float = 0.25,
        annualization_factor: int = 365,
        max_position_size: float = 1.0,
        rebalance_threshold: float = 0.05,
        use_vol_target: bool = True,
    ) -> None:
        self.lookbacks = _coerce_int_sequence(lookbacks, "lookbacks")
        self.sma_filter_periods = (
            _coerce_int_sequence(sma_filter_periods, "sma_filter_periods")
            if sma_filter_periods
            else ()
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
        if rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be >= 0")

        self.vol_lookback = int(vol_lookback)
        self.annual_vol_target = float(annual_vol_target)
        self.annualization_factor = int(annualization_factor)
        self.max_position_size = float(max_position_size)
        self.rebalance_threshold = float(rebalance_threshold)
        self.use_vol_target = _coerce_bool(use_vol_target, "use_vol_target")

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"].astype(float)

        raw_signal = self._tsmom_ensemble(close)
        if self.sma_filter_periods:
            raw_signal = raw_signal.where(self._sma_filter(close), 0.0)

        if not self.use_vol_target:
            # Skip the vol-target layer entirely. The ensemble's
            # {0, 1/k, ..., 1} ladder passes through to the engine.
            # The rebalance band is also disabled — there's nothing
            # for it to suppress on a discrete ladder.
            return raw_signal.clip(lower=0.0, upper=self.max_position_size).fillna(0.0)

        vol_weight = self._vol_weight(close)
        target_position = (raw_signal * vol_weight).clip(
            lower=0.0, upper=self.max_position_size
        ).fillna(0.0)
        return self._apply_rebalance_band(target_position)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _tsmom_ensemble(self, close: pd.Series) -> pd.Series:
        """Average ``{0,1}`` per-lookback sign-of-return states."""
        components: list[pd.Series] = []
        for lookback in self.lookbacks:
            # close.pct_change(lookback, fill_method=None) at bar N uses close[N] and
            # close[N-lookback] — both known at the close of bar N.
            past_return = close.pct_change(lookback, fill_method=None)
            state = (past_return > 0).astype(float)
            # Warm-up bars (NaN trailing return) are treated as "flat",
            # never as "long" by default.
            state[past_return.isna()] = 0.0
            components.append(state)
        stacked = pd.concat(components, axis=1)
        return stacked.mean(axis=1)

    def _sma_filter(self, close: pd.Series) -> pd.Series:
        ok = pd.Series(True, index=close.index)
        for period in self.sma_filter_periods:
            sma = close.rolling(period).mean()
            cond = close > sma
            cond[sma.isna()] = False
            ok = ok & cond
        return ok

    def _vol_weight(self, close: pd.Series) -> pd.Series:
        daily_returns = close.pct_change(fill_method=None)
        realized_vol_daily = daily_returns.rolling(self.vol_lookback).std()
        realized_vol_annual = realized_vol_daily * np.sqrt(self.annualization_factor)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = self.annual_vol_target / realized_vol_annual
        return weight.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _apply_rebalance_band(self, target_position: pd.Series) -> pd.Series:
        if self.rebalance_threshold == 0.0:
            return target_position
        held = pd.Series(0.0, index=target_position.index, dtype=float)
        current = 0.0
        for i, target in enumerate(target_position.to_numpy()):
            target = float(target)
            if target == 0.0 or current == 0.0:
                current = target
            elif abs(target - current) >= self.rebalance_threshold:
                current = target
            held.iloc[i] = current
        return held
