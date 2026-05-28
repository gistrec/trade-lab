"""Cross-sectional momentum (top-N rotation) across a universe of assets.

Reference: Liu, Tsyvinski, Wu (2022). *Common Risk Factors in
Cryptocurrency*. **Journal of Finance** 77(2):1133-1177. The paper
identifies market / size / momentum as the three factors that explain
the cross-section of cryptocurrency returns and reports ten long-short
strategies that load on them. The long-only version used here is the
direct retail analogue, since shorts on spot are not available.

Mechanics (weekly rebalance, daily execution):

1. On each rebalance date, compute the trailing ``lookback_days`` return
   for every asset in the universe.
2. Optionally apply a BTC regime gate — if BTC close < BTC SMA(N), hold
   cash that week instead of any altcoin basket.
3. Drop assets with negative trailing return; from the rest, pick the
   top ``top_k`` by return.
4. Weight the selected basket either equal-weighted or inverse-vol
   (``1 / realized_vol``, normalized to sum to 1).
5. Hold those weights until the next rebalance.

The runner aligns assets on a common date index (forward-filling gaps
to handle missed candles, but never reaching past the available
history). Costs are turnover * (fee_rate + slippage_rate). Long-only:
all weights are >= 0 and sum to <= 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalResult:
    """Portfolio-level output of :func:`run_cross_sectional_momentum`."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # rows = dates, cols = assets, sums <= 1
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    initial_capital: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    num_rebalances: int = 0
    average_basket_size: float = 0.0
    average_cash_fraction: float = 0.0


def run_cross_sectional_momentum(
    asset_candles: Mapping[str, pd.DataFrame],
    lookback_days: int = 30,
    rebalance_days: int = 7,
    top_k: int = 3,
    weighting: str = "equal",
    vol_lookback: int = 30,
    btc_candles: Optional[pd.DataFrame] = None,
    btc_gate_sma_period: int = 200,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    annualization_factor: int = 365,
) -> CrossSectionalResult:
    """Run the long-only top-N cross-sectional momentum portfolio.

    Parameters mirror the literature defaults: 30-day lookback, weekly
    rebalance, top-3 basket. ``weighting`` is either ``"equal"`` or
    ``"inverse_vol"``.
    """
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if weighting not in ("equal", "inverse_vol"):
        raise ValueError("weighting must be 'equal' or 'inverse_vol'")
    if vol_lookback < 2:
        raise ValueError("vol_lookback must be >= 2")

    if not asset_candles:
        empty = pd.Series(dtype=float)
        return CrossSectionalResult(
            equity=empty,
            returns=empty,
            weights=pd.DataFrame(),
            initial_capital=initial_capital,
        )

    closes = _align_closes(asset_candles)
    if closes.empty:
        empty = pd.Series(dtype=float)
        return CrossSectionalResult(
            equity=empty,
            returns=empty,
            weights=pd.DataFrame(columns=closes.columns),
            initial_capital=initial_capital,
        )

    btc_gate = _build_btc_gate(btc_candles, closes.index, btc_gate_sma_period)
    target_weights = _build_target_weights(
        closes=closes,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        top_k=top_k,
        weighting=weighting,
        vol_lookback=vol_lookback,
        btc_gate=btc_gate,
    )

    # Shift target weights by one bar so we only ever hold weights from
    # tomorrow onward — exactly like the single-asset engine. Decisions
    # at the close of N apply at the close of N+1.
    positions = target_weights.shift(1).fillna(0.0)
    rebalance_dates = [
        positions.index[i]
        for i in range(len(positions))
        if i > 0
        and not np.allclose(positions.iloc[i].to_numpy(), positions.iloc[i - 1].to_numpy())
    ]

    asset_returns = closes.pct_change().fillna(0.0)
    gross_returns = (positions * asset_returns).sum(axis=1)

    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    fee_costs = turnover * fee_rate
    slippage_costs = turnover * slippage_rate
    net_returns = gross_returns - fee_costs - slippage_costs

    equity = initial_capital * (1.0 + net_returns).cumprod()
    prior_equity = equity.shift(1).fillna(initial_capital)
    total_fees = float((turnover * fee_rate * prior_equity).sum())
    total_slippage = float((turnover * slippage_rate * prior_equity).sum())

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) >= 2 else 0.0
    max_dd = _max_drawdown(equity)
    sharpe = _sharpe(net_returns, annualization_factor)
    basket_size = float((positions > 0).sum(axis=1).mean())
    cash_fraction = float((1.0 - positions.sum(axis=1)).clip(lower=0.0, upper=1.0).mean())

    return CrossSectionalResult(
        equity=equity,
        returns=net_returns,
        weights=positions,
        rebalance_dates=rebalance_dates,
        initial_capital=initial_capital,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        num_rebalances=len(rebalance_dates),
        average_basket_size=basket_size,
        average_cash_fraction=cash_fraction,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _align_closes(asset_candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all asset close series onto a common date index.

    Forward-fill within each asset to handle missing candles, but only
    *after* its first observed close — we never invent prices that
    predate listing.
    """
    series = {}
    for asset, candles in asset_candles.items():
        if candles is None or candles.empty:
            continue
        close = candles["close"].astype(float)
        series[asset] = close
    if not series:
        return pd.DataFrame()
    closes = pd.concat(series, axis=1).sort_index()
    return closes.ffill()


def _build_btc_gate(
    btc_candles: Optional[pd.DataFrame],
    target_index: pd.Index,
    sma_period: int,
) -> pd.Series:
    """Return a boolean series aligned to ``target_index``: True iff BTC > SMA."""
    if btc_candles is None or btc_candles.empty:
        return pd.Series(True, index=target_index)
    btc_close = btc_candles["close"].astype(float)
    btc_sma = btc_close.rolling(sma_period).mean()
    gate = (btc_close > btc_sma) & btc_sma.notna()
    return gate.reindex(target_index, method="ffill").fillna(False)


def _build_target_weights(
    closes: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    top_k: int,
    weighting: str,
    vol_lookback: int,
    btc_gate: pd.Series,
) -> pd.DataFrame:
    """Compute the target weights matrix held *from* each date forward.

    The weights matrix is rebuilt on rebalance dates and held flat in
    between. All values are valid as of the close of the row's date,
    i.e. they use no information from later bars.
    """
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    # Trailing-return lookback. ``shift(0)`` is intentional — we want
    # ``close[t] / close[t - lookback]`` available at the close of t.
    trailing_return = closes.pct_change(lookback_days)
    realized_vol = closes.pct_change().rolling(vol_lookback).std()

    last_weights = pd.Series(0.0, index=closes.columns)
    for i, date in enumerate(closes.index):
        if i % rebalance_days == 0:
            last_weights = _rebalance(
                trailing_return.iloc[i],
                realized_vol.iloc[i] if weighting == "inverse_vol" else None,
                top_k=top_k,
                weighting=weighting,
                btc_in_market=bool(btc_gate.iloc[i]) if not btc_gate.empty else True,
            )
        weights.iloc[i] = last_weights
    return weights


def _rebalance(
    returns_now: pd.Series,
    vol_now: Optional[pd.Series],
    top_k: int,
    weighting: str,
    btc_in_market: bool,
) -> pd.Series:
    """Compute new target weights given the latest snapshot.

    Empty selection (no positive-return assets, or BTC gate closed) yields
    an all-zeros vector — the portfolio sits in cash that week.
    """
    out = pd.Series(0.0, index=returns_now.index)
    if not btc_in_market:
        return out
    eligible = returns_now.dropna()
    eligible = eligible[eligible > 0]
    if eligible.empty:
        return out
    chosen = eligible.nlargest(top_k)
    if chosen.empty:
        return out

    if weighting == "equal":
        weight = 1.0 / len(chosen)
        for asset in chosen.index:
            out[asset] = weight
        return out

    # inverse-vol — drop assets whose vol is missing or zero and fall
    # back to equal weights if nothing usable is left.
    if vol_now is None:
        return out
    vols = vol_now.reindex(chosen.index).replace(0.0, np.nan).dropna()
    if vols.empty:
        weight = 1.0 / len(chosen)
        for asset in chosen.index:
            out[asset] = weight
        return out
    inv = 1.0 / vols
    inv = inv / inv.sum()
    for asset, w in inv.items():
        out[asset] = float(w)
    return out


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series, annualization_factor: int) -> float:
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    std = float(cleaned.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(cleaned.mean() / std * np.sqrt(annualization_factor))
