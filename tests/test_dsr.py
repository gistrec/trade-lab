import math

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.dsr import (
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_from_trial_sharpes,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_ratio_per_period,
)


def test_norm_cdf_matches_known_values():
    """Spot-check a few well-known Phi values."""
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert _norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-9)
    assert _norm_cdf(-1.96) == pytest.approx(0.024997895148, abs=1e-9)
    assert _norm_cdf(1.96) == pytest.approx(0.974998, abs=1e-5)


def test_norm_ppf_inverts_norm_cdf():
    for p in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
        x = _norm_ppf(p)
        assert _norm_cdf(x) == pytest.approx(p, abs=1e-7)


def test_norm_ppf_known_values():
    assert _norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-7)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_norm_ppf_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)
    with pytest.raises(ValueError):
        _norm_ppf(-0.1)


def test_sharpe_ratio_per_period_zero_for_empty():
    assert sharpe_ratio_per_period(pd.Series([], dtype=float)) == 0.0


def test_sharpe_ratio_per_period_zero_for_constant_returns():
    """Zero std should not blow up the ratio."""
    assert sharpe_ratio_per_period(pd.Series([0.01] * 50)) == 0.0


def test_sharpe_ratio_per_period_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 1000))
    sr = sharpe_ratio_per_period(returns)
    assert sr > 0


def test_expected_max_sharpe_grows_with_num_trials():
    """The selection-bias correction must increase with N."""
    sr_low = expected_max_sharpe(10, 0.05)
    sr_high = expected_max_sharpe(1000, 0.05)
    assert sr_high > sr_low > 0


def test_expected_max_sharpe_zero_for_single_trial():
    """With one trial there is no selection bias to correct."""
    assert expected_max_sharpe(1, 0.05) == 0.0


def test_expected_max_sharpe_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.05)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.01)


def test_deflated_sharpe_is_high_for_strong_signal_with_few_trials():
    """A clearly positive Sharpe over a long sample with one trial
    should yield DSR close to 1."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.002, 0.01, 2000))
    dsr = deflated_sharpe_ratio(returns, num_trials=1, sharpe_std_dev=0.0)
    assert dsr > 0.99


def test_deflated_sharpe_drops_when_trial_count_explodes():
    """Same observed return series, but increase the number of trials —
    DSR must drop because the selection-bias threshold rises."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))  # mild edge
    low_n = deflated_sharpe_ratio(returns, num_trials=1, sharpe_std_dev=0.05)
    high_n = deflated_sharpe_ratio(returns, num_trials=10_000, sharpe_std_dev=0.05)
    assert low_n > high_n


def test_deflated_sharpe_below_half_when_observed_sharpe_below_threshold():
    """If the observed Sharpe is *below* the selection-bias threshold,
    DSR must drop below 0.5 — the observed result is no better than
    random selection."""
    rng = np.random.default_rng(0)
    # Near-zero mean return so observed Sharpe is small.
    returns = pd.Series(rng.normal(0.0, 0.01, 500))
    dsr = deflated_sharpe_ratio(returns, num_trials=1_000, sharpe_std_dev=0.10)
    assert dsr < 0.5


def test_deflated_sharpe_returns_zero_for_tiny_samples():
    """Less than four samples is too little to estimate skew/kurt."""
    assert deflated_sharpe_ratio(pd.Series([0.01, 0.02]), 10, 0.05) == 0.0


def test_deflated_sharpe_from_trial_panel():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    trial_panel = rng.normal(0.0, 0.05, 50).tolist()
    dsr = deflated_sharpe_from_trial_sharpes(returns, trial_panel)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_in_unit_interval():
    """Property: result must always be a probability in [0, 1]."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(50, 1000))
        mean = float(rng.normal(0, 0.001))
        std = float(rng.uniform(0.005, 0.05))
        returns = pd.Series(rng.normal(mean, std, n))
        trials = int(rng.integers(1, 5000))
        sigma_sr = float(rng.uniform(0.01, 0.2))
        dsr = deflated_sharpe_ratio(returns, trials, sigma_sr)
        assert 0.0 <= dsr <= 1.0
