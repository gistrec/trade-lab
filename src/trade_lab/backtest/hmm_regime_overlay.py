"""2-state Gaussian HMM regime overlay for single-asset BTC spot.

The classical Markov-switching model on daily log-returns: two latent
regimes (bull = high mean, low vol; bear = low mean, high vol). At
each rebalance the model is refit on the trailing window; the
**filtered** posterior P(regime=bull | data 1..t) drives the position.

The single most important design choice — and the one most often gotten
wrong in industry implementations — is **filtered vs smoothed
probabilities**:

* ``hmmlearn.GaussianHMM.predict_proba(X)`` returns the *smoothed*
  posteriors P(state[t] | data 1..T). Those use the entire sequence,
  including data after t. Using them as a trading signal back-fits
  the past with future information and produces a phantom edge.
* What we want is the *filtered* posterior P(state[t] | data 1..t).
  Computed via the forward variables only — no backward pass — by
  exponentiating the last row of ``_do_forward_pass``.

A unit test pins this distinction directly: future-data corruption
must not change the filtered probability at earlier dates.

Other invariants
================
* Walk-forward refit: the HMM is fitted from scratch on every
  rebalance using only the trailing ``train_lookback_days``. No
  parameters from one fold leak into another.
* State identification by mean: ``bull`` = component with higher
  empirical ``means_`` after fitting. This is a structural label,
  not a hyperparameter.
* Publication of the daily return at time t happens at the close of t.
  The signal at time t uses returns up to t-1 only — same one-day
  realism as the MVRV overlay. (Strictly speaking daily log-returns
  are computable at the close, but a 1-bar buffer absorbs end-of-day
  data-quality artifacts.)

Asymmetry of interpretation
===========================
Per the user's review and the compass artifact, the most likely
outcome of this test is a draw with the existing
``VolatilityTargetWrapper`` — they target overlapping signals (the
HMM bull state is the low-vol regime; vol-target also scales down in
high vol). A non-additive result over vol-targeting would be REJECT
as a duplicate. A clean win — additive risk-adjusted return — would
be KEEP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import _hmmc
from hmmlearn.hmm import GaussianHMM

from .cross_sectional import _max_drawdown, _sharpe


DEFAULT_TRAIN_LOOKBACK_DAYS = 730   # 2 years
DEFAULT_REBALANCE_DAYS = 7
DEFAULT_BUFFER_DAYS = 1             # 1-day publication lag on returns
DEFAULT_N_COMPONENTS = 2
DEFAULT_N_ITER = 50
DEFAULT_RANDOM_STATE = 42


@dataclass
class HmmRegimeOverlayResult:
    equity: pd.Series
    returns: pd.Series
    bull_probability: pd.Series       # filtered P(bull | data ≤ t-buffer)
    realized_position: pd.Series
    rebalance_dates: list = field(default_factory=list)
    initial_capital: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    num_rebalances: int = 0
    mean_position: float = 0.0


def filtered_bull_probability(
    log_returns: np.ndarray,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[float, GaussianHMM]:
    """Fit a Gaussian HMM and return P(bull | data) at the LAST step.

    Returns ``(p_bull, fitted_model)``. ``bull`` is the component with
    the higher fitted mean. The probability is computed from the
    forward pass alone — NO backward smoothing — so it represents the
    online filtered estimate at the last observation only.
    """
    X = np.asarray(log_returns, dtype=float).reshape(-1, 1)
    if len(X) < 50:
        return 0.5, None  # not enough data to fit — neutral
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
    )
    model.fit(X)

    # Forward pass ONLY — no backward / smoothing. fwdlattice rows are
    # log P(state[t], data 1..t). Normalizing the last row gives the
    # filtered posterior P(state[T] | data 1..T).
    log_frameprob = model._compute_log_likelihood(X)
    _, fwdlattice = _hmmc.forward_log(
        model.startprob_, model.transmat_, log_frameprob,
    )
    last = fwdlattice[-1]
    log_max = float(np.max(last))
    posterior = np.exp(last - log_max)
    posterior /= posterior.sum()

    bull_state = int(np.argmax(model.means_.flatten()))
    return float(posterior[bull_state]), model


def run_hmm_regime_overlay(
    btc_candles: pd.DataFrame,
    *,
    train_lookback_days: int = DEFAULT_TRAIN_LOOKBACK_DAYS,
    rebalance_days: int = DEFAULT_REBALANCE_DAYS,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    probability_threshold: float = 0.5,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    annualization_factor: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> HmmRegimeOverlayResult:
    """Walk-forward HMM regime overlay on BTC daily closes."""
    if "close" not in btc_candles.columns:
        raise ValueError("btc_candles must contain a 'close' column")

    closes = btc_candles["close"].copy()
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")
    if start_date:
        closes = closes[closes.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        closes = closes[closes.index <= pd.Timestamp(end_date, tz="UTC")]
    if len(closes) < train_lookback_days + buffer_days + 1:
        return HmmRegimeOverlayResult(
            equity=pd.Series(dtype=float),
            returns=pd.Series(dtype=float),
            bull_probability=pd.Series(dtype=float),
            realized_position=pd.Series(dtype=float),
            initial_capital=initial_capital,
        )

    log_returns = np.log(closes).diff().fillna(0.0)
    bull_prob = pd.Series(np.nan, index=closes.index)
    realized = pd.Series(0.0, index=closes.index)
    rebalance_dates: list[pd.Timestamp] = []
    current = 0.0
    last_rebal_idx = -10**9
    earliest = train_lookback_days + buffer_days

    for i in range(len(closes)):
        if i < earliest:
            realized.iloc[i] = current
            continue
        if (i - last_rebal_idx) < rebalance_days:
            realized.iloc[i] = current
            continue
        # Train window strictly trailing: [i - lookback - buffer, i - buffer)
        window_end = i - buffer_days
        window_start = max(0, window_end - train_lookback_days)
        window = log_returns.iloc[window_start:window_end]
        if window.std() == 0:
            target = current
        else:
            p_bull, _ = filtered_bull_probability(
                window.to_numpy(),
                n_components=n_components,
                n_iter=n_iter,
                random_state=random_state,
            )
            bull_prob.iloc[i] = p_bull
            target = 1.0 if p_bull > probability_threshold else 0.0
        if target != current:
            current = target
            last_rebal_idx = i
            rebalance_dates.append(closes.index[i])
        elif i == earliest:
            last_rebal_idx = i
            rebalance_dates.append(closes.index[i])
        realized.iloc[i] = current

    daily_returns = closes.pct_change(fill_method=None).fillna(0.0)
    eq = pd.Series(0.0, index=closes.index)
    eq.iloc[0] = initial_capital
    total_fees = 0.0
    total_slippage = 0.0
    for i in range(1, len(closes)):
        prev_w = float(realized.iloc[i - 1])
        port_ret = prev_w * float(daily_returns.iloc[i])
        new_eq = float(eq.iloc[i - 1]) * (1.0 + port_ret)
        delta = abs(float(realized.iloc[i]) - prev_w)
        if delta > 1e-12:
            fee = new_eq * delta * fee_rate
            slip = new_eq * delta * slippage_rate
            new_eq -= (fee + slip)
            total_fees += fee
            total_slippage += slip
        eq.iloc[i] = new_eq

    ret = eq.pct_change(fill_method=None).fillna(0.0)
    return HmmRegimeOverlayResult(
        equity=eq,
        returns=ret,
        bull_probability=bull_prob,
        realized_position=realized,
        rebalance_dates=rebalance_dates,
        initial_capital=initial_capital,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_return=float(eq.iloc[-1] / initial_capital - 1.0),
        max_drawdown=_max_drawdown(eq),
        sharpe=_sharpe(ret, annualization_factor),
        num_rebalances=len(rebalance_dates),
        mean_position=float(realized.mean()),
    )
