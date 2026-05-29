"""CTREND-proxy (price-only) for cross-section spot crypto.

This is an INTENTIONALLY simplified proxy of the Fieberg, Liedtke,
Poddig, Walker, Zaremba (JFQA 2024) CTREND trend factor:

* Original aggregates ~28 indicators including reported AND trusted
  volume. This proxy uses ONLY the 6 price MA-ratio features (no
  volume). The Coin Metrics community-tier volume on this project
  is the *reported* spot volume, which is contaminated by wash
  trading and methodology drift across venues — not a basis for an
  honest test. The volume half is therefore deliberately omitted.
* Original estimator is a cross-sectional Fama-MacBeth-style
  regression with rolling coefficients (per Han, Zhou, Zhu 2016
  "A Trend Factor"). This proxy is a pooled Ridge regression on a
  2-year rolling panel — a different estimator.

Asymmetry of interpretation
===========================
* If THIS proxy fails net-of-cost OOS → REJECT this proxy. It does
  NOT refute the Fieberg paper: the volume half and the FMB
  estimator are not tested here.
* If THIS proxy passes → conditional positive. A KEEP verdict would
  require V2: trusted volume + FMB estimator faithful to the paper.

Look-ahead guarantees
=====================
* Features at time t use SMA over [t-w+1, t] only.
* Target is forward return [t, t+H] where H = ``rebalance_days``.
* Train sample at time s is included in the model fit for rebalance
  date r only if ``s + H + purge_days <= r`` — strict purge between
  the end of train target windows and the rebalance evaluation.
* Coefficients are refit on every rebalance (weekly by default),
  using only the trailing ``train_lookback_days`` of history.
* Eligibility mask (PIT universe) is honoured; coins with NaN
  features (insufficient history for max(windows) days) are dropped.

Coin Metrics ``price`` is a reference rate (VWA across venues), not
the executable Binance/Kraken close. On daily bars the basis to a
single venue's close is small; the configured ``slippage_rate`` is
intended to absorb it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .cross_sectional import _align_closes, _max_drawdown, _sharpe


logger = logging.getLogger(__name__)


DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
DEFAULT_TRAIN_LOOKBACK_DAYS = 730   # ≈ 2 years
DEFAULT_REBALANCE_DAYS = 7          # weekly
DEFAULT_TOP_K = 8                   # top quintile of a ~40 universe
DEFAULT_PURGE_DAYS = 7              # = forward target horizon
DEFAULT_RIDGE_ALPHA = 1.0


@dataclass
class CtrendProxyResult:
    """Portfolio-level output of :func:`run_ctrend_proxy`."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    rebalance_dates: list = field(default_factory=list)
    initial_capital: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    num_rebalances: int = 0
    average_basket_size: float = 0.0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def compute_price_features(
    closes: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> dict[int, pd.DataFrame]:
    """Per-window ``close / SMA(close, w)``.

    Returns ``{w: DataFrame[date x coin]}``. NaN where there is
    insufficient history (``min_periods=w`` — no shortcut warm-up).
    """
    out: dict[int, pd.DataFrame] = {}
    for w in windows:
        sma = closes.rolling(w, min_periods=w).mean()
        out[w] = closes / sma
    return out


# ---------------------------------------------------------------------------
# Walk-forward + Ridge + rank top-K
# ---------------------------------------------------------------------------


def run_ctrend_proxy(
    asset_candles: Mapping[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    train_lookback_days: int = DEFAULT_TRAIN_LOOKBACK_DAYS,
    rebalance_days: int = DEFAULT_REBALANCE_DAYS,
    top_k: int = DEFAULT_TOP_K,
    purge_days: int = DEFAULT_PURGE_DAYS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    annualization_factor: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> CtrendProxyResult:
    """End-to-end CTREND-proxy backtest with strict look-ahead control."""
    closes = _align_closes(asset_candles)
    if closes.empty:
        return _empty(initial_capital, closes.columns)

    if start_date is not None:
        closes = closes[closes.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        closes = closes[closes.index <= pd.Timestamp(end_date, tz="UTC")]
    if closes.empty:
        return _empty(initial_capital, closes.columns)

    elig = eligibility.reindex(
        index=closes.index, columns=closes.columns, fill_value=False,
    ).astype(bool)

    feature_dict = compute_price_features(closes, windows)
    fwd_returns = closes.shift(-rebalance_days) / closes - 1.0
    daily_returns = closes.pct_change().fillna(0.0)

    earliest = int(max(windows)) + train_lookback_days + purge_days
    dates = list(closes.index)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rebalance_dates: list[pd.Timestamp] = []
    current_w = pd.Series(0.0, index=closes.columns)
    basket_sizes: list[int] = []
    last_rebalance_idx = -10**9

    for i, d in enumerate(dates):
        if i < earliest:
            weights.iloc[i] = current_w
            continue
        if (i - last_rebalance_idx) < rebalance_days:
            weights.iloc[i] = current_w
            continue

        # ----- Train window: trailing 2y up to (d - purge_days) -----
        train_cut_idx = i - purge_days
        train_start_idx = max(0, train_cut_idx - train_lookback_days)
        train_dates = closes.index[train_start_idx:train_cut_idx]

        X_train, y_train = _collect_panel(
            feature_dict, fwd_returns, elig, train_dates, windows,
        )
        if len(y_train) < 100:
            weights.iloc[i] = current_w
            continue

        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        X_train_std = (X_train - mu) / sigma

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_train_std, y_train)

        preds = _predict_at_date(
            d, closes.columns, feature_dict, elig, windows,
            model=model, mu=mu, sigma=sigma,
        )
        new_w = _top_k_equal_weight(preds, top_k, closes.columns)
        current_w = new_w
        last_rebalance_idx = i
        rebalance_dates.append(d)
        basket_sizes.append(int((new_w > 0).sum()))
        weights.iloc[i] = current_w

    equity, fees, slippage = _simulate_equity(
        closes, weights, daily_returns,
        initial_capital=initial_capital,
        fee_rate=fee_rate, slippage_rate=slippage_rate,
    )
    returns = equity.pct_change().fillna(0.0)

    return CtrendProxyResult(
        equity=equity,
        returns=returns,
        weights=weights,
        rebalance_dates=rebalance_dates,
        initial_capital=initial_capital,
        total_fees=fees,
        total_slippage=slippage,
        total_return=float(equity.iloc[-1] / initial_capital - 1.0),
        max_drawdown=_max_drawdown(equity),
        sharpe=_sharpe(returns, annualization_factor),
        num_rebalances=len(rebalance_dates),
        average_basket_size=(
            float(np.mean(basket_sizes)) if basket_sizes else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_panel(
    feature_dict: dict[int, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    eligibility: pd.DataFrame,
    train_dates: pd.Index,
    windows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for Ridge fit over all (date, coin) pairs in the window.

    A row is included only if:
    * eligibility[d, c] is True
    * every feature[w][d, c] is finite (sufficient history)
    * forward target return[d, c] is finite (target window also fits)
    """
    coins = fwd_returns.columns
    elig_slice = eligibility.loc[train_dates, coins].to_numpy()
    feat_arrays = [feature_dict[w].loc[train_dates, coins].to_numpy() for w in windows]
    target_arr = fwd_returns.loc[train_dates, coins].to_numpy()

    feat_stack = np.stack(feat_arrays, axis=-1)  # (T, C, F)
    feat_ok = np.isfinite(feat_stack).all(axis=-1)
    tgt_ok = np.isfinite(target_arr)
    mask = elig_slice & feat_ok & tgt_ok

    X = feat_stack[mask]
    y = target_arr[mask]
    return X, y


def _predict_at_date(
    d: pd.Timestamp,
    coins: pd.Index,
    feature_dict: dict[int, pd.DataFrame],
    eligibility: pd.DataFrame,
    windows: Sequence[int],
    *,
    model: Ridge,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    """Predict the forward-target return for each eligible coin at ``d``."""
    preds: dict[str, float] = {}
    for c in coins:
        if not bool(eligibility.at[d, c]):
            continue
        feat = np.array([feature_dict[w].at[d, c] for w in windows], dtype=float)
        if not np.all(np.isfinite(feat)):
            continue
        x = ((feat - mu) / sigma).reshape(1, -1)
        preds[c] = float(model.predict(x)[0])
    return preds


def _top_k_equal_weight(
    preds: Mapping[str, float], top_k: int, columns: pd.Index,
) -> pd.Series:
    """Rank predictions descending, equal-weight top-K (or fewer)."""
    out = pd.Series(0.0, index=columns)
    if not preds or top_k < 1:
        return out
    ranked = sorted(preds.items(), key=lambda kv: -kv[1])
    chosen = [c for c, _ in ranked[:top_k]]
    if not chosen:
        return out
    w = 1.0 / len(chosen)
    for c in chosen:
        out[c] = w
    return out


def _simulate_equity(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.Series, float, float]:
    """Apply weights with symmetric round-trip costs on turnover."""
    eq = pd.Series(0.0, index=closes.index)
    eq.iloc[0] = initial_capital
    fees = 0.0
    slip = 0.0
    for i in range(1, len(closes.index)):
        prev_w = weights.iloc[i - 1]
        port_ret = float((prev_w * daily_returns.iloc[i]).sum())
        new_eq = float(eq.iloc[i - 1]) * (1.0 + port_ret)

        curr_w = weights.iloc[i]
        turnover = float((curr_w - prev_w).abs().sum())
        if turnover > 1e-9:
            fee_chunk = new_eq * turnover * fee_rate
            slip_chunk = new_eq * turnover * slippage_rate
            new_eq -= (fee_chunk + slip_chunk)
            fees += fee_chunk
            slip += slip_chunk
        eq.iloc[i] = new_eq
    return eq, fees, slip


def _empty(initial_capital: float, columns) -> CtrendProxyResult:
    empty = pd.Series(dtype=float)
    return CtrendProxyResult(
        equity=empty, returns=empty,
        weights=pd.DataFrame(columns=columns),
        initial_capital=initial_capital,
    )
