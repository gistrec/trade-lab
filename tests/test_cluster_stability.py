"""Tests for cluster stability check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.cluster_stability import (
    ClusterStabilityResult, run_cluster_stability_check,
)
from trade_lab.backtest.walk_forward_v2 import ParamGridSpec
from trade_lab.strategies.sma_cross import SMACrossStrategy


def _candles(n: int, start: str = "2018-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")
    closes = 100 + 0.5 * np.arange(n) + rng.normal(0, 1.5, n)
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5,
         "close": closes.clip(min=1), "volume": 1.0},
        index=idx,
    )


def _sma_grid(pairs):
    return [
        ParamGridSpec(
            label=f"sma_{f}_{s}",
            factory=lambda f=f, s=s: SMACrossStrategy(fast_period=f, slow_period=s),
            warmup_days=s,
        )
        for f, s in pairs
    ]


def test_empty_grid_returns_empty_result():
    res = run_cluster_stability_check(_candles(1200), grid=[])
    assert isinstance(res, ClusterStabilityResult)
    assert res.n_variants == 0
    assert res.cluster_passes is False


def test_per_variant_row_count_matches_grid():
    candles = _candles(1200, seed=1)
    grid = _sma_grid([(10, 30), (20, 50), (30, 100)])
    res = run_cluster_stability_check(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
    )
    assert res.n_variants == 3
    assert len(res.per_variant) == 3
    assert set(res.per_variant["variant"]) == {"sma_10_30", "sma_20_50", "sma_30_100"}


def test_annualization_factor_forwarded_to_concat_oos_sharpe():
    """concat_oos_sharpe must be annualized with the caller's factor, the
    same basis as mean_per_fold_sharpe. The factor was passed to the
    per-fold walk-forward but NOT to aggregate_walk_forward, so
    concat_oos_sharpe silently used the 365 default (regression: C11)."""
    candles = _candles(1400, seed=3)
    grid = _sma_grid([(10, 30), (20, 50)])
    kw = dict(train_months=12, test_months=3, step_months=3)
    res365 = run_cluster_stability_check(
        candles, grid, annualization_factor=365, **kw
    )
    res252 = run_cluster_stability_check(
        candles, grid, annualization_factor=252, **kw
    )
    m365 = res365.per_variant.set_index("variant")
    m252 = res252.per_variant.set_index("variant")
    expected_ratio = np.sqrt(252 / 365)
    checked = 0
    for variant in m365.index:
        c365 = m365.loc[variant, "concat_oos_sharpe"]
        c252 = m252.loc[variant, "concat_oos_sharpe"]
        if abs(c365) < 1e-9:
            continue
        checked += 1
        assert c252 / c365 == pytest.approx(expected_ratio, rel=1e-6)
    assert checked > 0, "no variant produced a non-zero concat_oos_sharpe"


def test_passes_threshold_flag_consistent_with_dsr():
    candles = _candles(1200, seed=2)
    grid = _sma_grid([(10, 30), (20, 50)])
    res = run_cluster_stability_check(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
        threshold_dsr=0.5,
    )
    for _, row in res.per_variant.iterrows():
        assert row["passes_threshold"] == (row["dsr"] >= 0.5)


def test_cluster_pass_requires_fraction():
    """``cluster_passes`` reflects (fraction_passing >= required_fraction).

    Same run, same threshold; vary only the ``required_fraction_pass``
    and confirm the verdict flips when we ask for "everybody must
    pass" vs "any one is enough"."""
    candles = _candles(1200, seed=3)
    grid = _sma_grid([(10, 30), (20, 50), (30, 100), (50, 200)])
    res_relaxed = run_cluster_stability_check(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
        threshold_dsr=0.5,
        required_fraction_pass=0.0,    # zero — always passes
    )
    assert res_relaxed.cluster_passes is True

    res_strict = run_cluster_stability_check(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
        threshold_dsr=0.5,
        required_fraction_pass=1.01,   # > 1 — impossible to satisfy
    )
    assert res_strict.cluster_passes is False


def test_per_variant_dsr_in_unit_interval():
    candles = _candles(1100, seed=4)
    grid = _sma_grid([(10, 30), (20, 50)])
    res = run_cluster_stability_check(
        candles, grid,
        train_months=12, test_months=3, step_months=3,
    )
    assert (res.per_variant["dsr"] >= 0.0).all()
    assert (res.per_variant["dsr"] <= 1.0).all()
