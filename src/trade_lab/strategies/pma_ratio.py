"""Price-to-moving-average ratio ensemble (Detzel et al. 2021).

Detzel, Liu, Strauss, Zhou, Zhu (2021). *Learning and predictability via
technical analysis: Evidence from Bitcoin and stocks with hard-to-value
fundamentals*. **Financial Management**.

The paper models the price-to-MA ratio as a rational-learning signal in
assets with hard-to-value fundamentals (a category that includes
Bitcoin) and shows that an ensemble of ``close / SMA(k)`` ratios over
``k in {5, 10, 20, 50, 100}`` predicts daily Bitcoin returns both
in- and out-of-sample, with positive alpha vs HODL.

This long-only implementation operationalizes the ratios as discrete
"is the ratio above 1?" votes and averages them, producing a smooth
ladder on ``{0, 1/n, 2/n, ..., 1}``. We deliberately *do not* read
quantitative magnitudes off the ratios — that would invite an
overfitted scaling and the literature warns that the paper's evidence
is for direction, not for size.

The strategy shares two optional layers with the rest of this
repository's trend stack (regime filter + vol targeting) so its
contribution to a strategy comparison is the *signal*, not a different
sizing scheme.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .base import Strategy
from .donchian_trend import _coerce_int_sequence


class PriceMaRatioStrategy(Strategy):
    """Ensemble of ``close > SMA(k)`` votes over a panel of windows."""

    name = "pma_ratio"

    def __init__(
        self,
        ma_periods: Iterable[int] | str = (5, 10, 20, 50, 100),
        sma_filter_periods: Iterable[int] | str = (),
        vol_lookback: int = 30,
        annual_vol_target: float = 0.25,
        annualization_factor: int = 365,
        max_position_size: float = 1.0,
        rebalance_threshold: float = 0.05,
        use_vol_target: bool = True,
    ) -> None:
        self.ma_periods = _coerce_int_sequence(ma_periods, "ma_periods")
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
        self.use_vol_target = bool(use_vol_target)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        close = candles["close"].astype(float)

        raw_signal = self._pma_ensemble(close)
        if self.sma_filter_periods:
            raw_signal = raw_signal.where(self._sma_filter(close), 0.0)

        if not self.use_vol_target:
            # Pass the {0, 1/n, ..., 1} P/MA-vote ladder straight to
            # the engine; the rebalance band has nothing to suppress
            # on a discrete ladder either.
            return raw_signal.clip(lower=0.0, upper=self.max_position_size).fillna(0.0)

        vol_weight = self._vol_weight(close)
        target_position = (raw_signal * vol_weight).clip(
            lower=0.0, upper=self.max_position_size
        ).fillna(0.0)
        return self._apply_rebalance_band(target_position)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _pma_ensemble(self, close: pd.Series) -> pd.Series:
        components: list[pd.Series] = []
        for period in self.ma_periods:
            sma = close.rolling(period).mean()
            state = (close > sma).astype(float)
            # Treat warm-up (SMA NaN) as "flat", never as "long".
            state[sma.isna()] = 0.0
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
