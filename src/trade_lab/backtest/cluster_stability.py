"""Cluster-stability validation — accept a strategy family only when most
of the parameter cluster passes the DSR threshold, not just one cherry-
picked variant.

This is the methodology emphasised in `deep-research-report.md` and
recommended throughout the trend-following literature (Faber, Carver,
Robot Wealth): a single "best" parameter point is too easy to overfit.
The honest test is whether a *cluster* of nearby parameter choices —
e.g. SMA crossover with (fast, slow) in
``{10, 20, 30, 50} × {50, 100, 150, 200, 300}`` — collectively passes
out-of-sample.

The function below runs each variant independently through
:func:`walk_forward_v2.run_strategy_walk_forward` (with a single-
element grid so the WF runner does no train-time selection inside the
variant), computes its concatenated-OOS DSR, and aggregates.

Output reports:

* per-variant DSR
* fraction of variants whose DSR clears ``threshold_dsr``
* mean and median DSR across the cluster
* a binary "cluster_passes" verdict if the passing fraction clears
  ``required_fraction_pass``

The function is intentionally a methodology cross-check, NOT a way to
select strategies. Selection still happens via the picked deployable
config (e.g. Han 28/60 from `findings/han_28d_tsmom.md`); this check
verifies that the *family* the picked variant lives in is robust to
neighbouring parameter choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .walk_forward_v2 import (
    PROJECT_NUM_TRIALS,
    ParamGridSpec,
    aggregate_walk_forward,
    run_strategy_walk_forward,
)


@dataclass(frozen=True)
class ClusterStabilityResult:
    """Output bundle of :func:`run_cluster_stability_check`."""

    per_variant: pd.DataFrame             # one row per variant
    n_variants: int
    n_passing: int
    fraction_passing: float
    mean_dsr: float
    median_dsr: float
    min_dsr: float
    max_dsr: float
    cluster_passes: bool                  # fraction_passing >= required_fraction_pass
    threshold_dsr: float
    required_fraction_pass: float


def run_cluster_stability_check(
    candles: pd.DataFrame,
    grid: Sequence[ParamGridSpec],
    *,
    threshold_dsr: float = 0.5,
    required_fraction_pass: float = 0.5,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    objective: str = "sharpe",
    annualization_factor: int = 365,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    num_trials_for_dsr: int = PROJECT_NUM_TRIALS,
) -> ClusterStabilityResult:
    """Run each variant in ``grid`` independently, then assess the cluster.

    Each variant is wrapped in a single-element grid for
    :func:`run_strategy_walk_forward`, which means the WF runner does
    no train-time variant selection — it just walk-forwards the fixed
    variant. The concatenated-OOS DSR computed against
    ``num_trials_for_dsr`` (default :data:`PROJECT_NUM_TRIALS`) is the
    per-variant verdict.

    Parameters
    ----------
    threshold_dsr
        Per-variant pass threshold for DSR. Default 0.5 matches the
        Bailey-LdP "marginal" line used elsewhere in the repo.
    required_fraction_pass
        Fraction of the cluster that must pass for the cluster as a
        whole to be considered stable. Default 0.5 — at least half of
        neighbouring parameter choices must agree.
    """
    if not grid:
        return ClusterStabilityResult(
            per_variant=pd.DataFrame(),
            n_variants=0, n_passing=0, fraction_passing=0.0,
            mean_dsr=0.0, median_dsr=0.0, min_dsr=0.0, max_dsr=0.0,
            cluster_passes=False,
            threshold_dsr=threshold_dsr,
            required_fraction_pass=required_fraction_pass,
        )

    rows: list[dict] = []
    for spec in grid:
        single_grid = [spec]
        detail, oos = run_strategy_walk_forward(
            candles, single_grid,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            objective=objective,
            annualization_factor=annualization_factor,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            return_oos_returns=True,
        )
        summary = aggregate_walk_forward(
            detail,
            annualization_factor=annualization_factor,
            oos_returns=oos,
            num_trials=num_trials_for_dsr,
        )
        rows.append({
            "variant": spec.label,
            "concat_oos_sharpe": summary["concatenated_oos_sharpe"],
            "dsr": summary["concatenated_oos_dsr"],
            "mean_per_fold_sharpe": summary["mean_test_sharpe"],
            "hit_rate": summary["hit_rate"],
            "passes_threshold": summary["concatenated_oos_dsr"] >= threshold_dsr,
        })

    per_variant = pd.DataFrame(rows)
    dsrs = per_variant["dsr"].to_numpy()
    n_passing = int(per_variant["passes_threshold"].sum())
    fraction_passing = float(n_passing / len(per_variant))
    return ClusterStabilityResult(
        per_variant=per_variant,
        n_variants=len(per_variant),
        n_passing=n_passing,
        fraction_passing=fraction_passing,
        mean_dsr=float(np.mean(dsrs)),
        median_dsr=float(np.median(dsrs)),
        min_dsr=float(np.min(dsrs)),
        max_dsr=float(np.max(dsrs)),
        cluster_passes=fraction_passing >= required_fraction_pass,
        threshold_dsr=threshold_dsr,
        required_fraction_pass=required_fraction_pass,
    )
