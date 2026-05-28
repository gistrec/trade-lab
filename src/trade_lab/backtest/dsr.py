"""Deflated Sharpe Ratio (Bailey & López de Prado 2014).

Bailey, D.H., López de Prado, M. (2014). *The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
**Journal of Portfolio Management** 40(5):94-107.

The DSR answers: "After accounting for the number of strategies we
tried, the non-normality of returns, and the sample length, what is
the probability that the observed Sharpe ratio is *not* the result of
selection bias?"

Use this whenever you've evaluated many parameter combinations (e.g.
``sweep.py`` produces dozens of grid points) and want to know whether
the best result is plausibly real. The Research-Claude survey calls
this metric mandatory once the parameter grid exceeds ~20 trials.

Formulas (using the original notation):

* Expected maximum Sharpe under the null of zero true skill::

      SR_0 = sigma_SR * ((1 - gamma) * Phi^-1(1 - 1/N)
                         + gamma * Phi^-1(1 - 1/(N * e)))

  where ``gamma`` is the Euler-Mascheroni constant, ``Phi^-1`` is the
  inverse standard normal CDF, and ``sigma_SR`` is the cross-trial
  standard deviation of Sharpe ratios.

* Deflated Sharpe Ratio::

      DSR = Phi((SR_hat - SR_0) * sqrt(T - 1)
                / sqrt(1 - skew * SR_hat + ((kurt - 1) / 4) * SR_hat^2))

  ``SR_hat`` is the *non-annualized* observed Sharpe (per-period),
  ``T`` is the number of returns in the sample, ``skew`` and ``kurt``
  are the third and fourth standardized moments of the returns.

DSR in ``[0, 1]`` is a probability: > 0.95 is the rough threshold for
"statistically convincing"; below 0.5 means the observed Sharpe is
*less* impressive than what a random walk would produce given the
search.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (Abramowitz & Stegun 7.1.26)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF using Wichura's AS 241 rational approx.

    Accurate to ~1e-9 across the whole interval (0, 1); good enough for
    DSR work, where the rest of the formula has more error in its
    asymptotic small-T behavior than this approximation contributes.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (
            ((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r
                 + 6.7265770927008700853e4) * r + 4.5921953931549871457e4) * r
               + 1.3731693765509461125e4) * r + 1.9715909503065514427e3) * r
             + 1.3314166789178437745e2) * r + 3.387132872796366608e0
        )
        den = (
            ((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r
                 + 3.9307895800092710610e4) * r + 2.1213794301586595867e4) * r
               + 5.3941960214247511077e3) * r + 6.8718700749205790830e2) * r
             + 4.2313330701600911252e1) * r + 1.0
        )
        return q * num / den
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (
            ((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r
                 + 2.41780725177450611770e-1) * r + 1.27045825245236838258e0) * r
               + 3.64784832476320460504e0) * r + 5.76949722146069140550e0) * r
             + 4.63033784615654529590e0) * r + 1.42343711074968357734e0
        )
        den = (
            ((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r
                 + 1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r
               + 6.89767334985100004550e-1) * r + 1.67638483018380384940e0) * r
             + 2.05319162663775882187e0) * r + 1.0
        )
    else:
        r -= 5.0
        num = (
            ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
                 + 1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r
               + 2.96560571828504891230e-1) * r + 1.78482653991729133580e0) * r
             + 5.46378491116411436990e0) * r + 6.65790464350110377720e0
        )
        den = (
            ((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r
                 + 1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r
               + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r
             + 5.99832206555887937690e-1) * r + 1.0
        )
    return num / den if q >= 0 else -num / den


def sharpe_ratio_per_period(returns: pd.Series) -> float:
    """Non-annualized Sharpe (mean / std). Annualization happens elsewhere."""
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    std = float(cleaned.std(ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    return float(cleaned.mean() / std)


def expected_max_sharpe(num_trials: int, sharpe_std_dev: float) -> float:
    """Expected max Sharpe over ``num_trials`` IID draws from N(0, sigma^2).

    ``sharpe_std_dev`` is the standard deviation of *trial* Sharpe ratios
    — i.e. ``np.std(trial_sharpes, ddof=1)`` if you have a panel from a
    sweep, or an a-priori estimate of how variable Sharpes are across
    the search space.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if sharpe_std_dev < 0:
        raise ValueError("sharpe_std_dev must be >= 0")
    if num_trials == 1:
        return 0.0
    a = _norm_ppf(1.0 - 1.0 / num_trials)
    b = _norm_ppf(1.0 - 1.0 / (num_trials * math.e))
    return float(sharpe_std_dev * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    sharpe_std_dev: float,
) -> float:
    """Return DSR in ``[0, 1]`` from per-period returns.

    ``num_trials`` is the count of strategy / parameter combinations
    that were evaluated *before* picking this one. ``sharpe_std_dev``
    is the std of trial Sharpes (see ``expected_max_sharpe``).
    """
    cleaned = returns.dropna()
    n = len(cleaned)
    if n < 4:
        return 0.0
    sr = sharpe_ratio_per_period(cleaned)
    skew = float(cleaned.skew())
    kurt = float(cleaned.kurtosis()) + 3.0  # pandas reports excess kurtosis
    sr0 = expected_max_sharpe(num_trials, sharpe_std_dev)

    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        # Heavy fat tails make the asymptotic distribution undefined.
        # Returning 0 is the conservative call: it's the equivalent of
        # "we cannot rule out chance".
        return 0.0
    denom = math.sqrt(denom_sq)
    z = (sr - sr0) * math.sqrt(n - 1) / denom
    return float(_norm_cdf(z))


def deflated_sharpe_from_trial_sharpes(
    returns: pd.Series,
    trial_sharpes: Iterable[float],
) -> float:
    """Convenience: pull ``num_trials`` and ``sharpe_std_dev`` from a panel.

    Pass the full list of non-annualized Sharpe ratios produced by every
    trial in the sweep (including the selected one). The function
    computes ``len(trials)`` and ``np.std(trials, ddof=1)``.
    """
    values = np.asarray(list(trial_sharpes), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return deflated_sharpe_ratio(
        returns=returns,
        num_trials=int(values.size),
        sharpe_std_dev=float(values.std(ddof=1)),
    )
