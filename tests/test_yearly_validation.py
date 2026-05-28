import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.yearly import (
    DEFAULT_FIXED_STRATEGIES,
    FixedStrategySpec,
    VERDICT_BUY_AND_HOLD,
    YEARLY_AGGREGATE_COLUMNS,
    YEARLY_COLUMNS,
    aggregate_yearly_results,
    run_yearly_validation,
)
from trade_lab.strategies.regime_only import RegimeOnlyStrategy
from trade_lab.strategies.sma_cross import SMACrossStrategy


def _daily_candles_multi_year(start: str = "2020-01-01", periods: int = 365 * 3 + 1):
    rng = np.random.default_rng(0)
    close = 100 + np.linspace(0, 60, periods) + rng.normal(0, 1.5, periods)
    idx = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def test_columns_match_spec():
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    assert list(df.columns) == YEARLY_COLUMNS


def test_one_row_per_year_per_strategy():
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    n_years = candles.index.year.nunique()
    n_strategies = len(DEFAULT_FIXED_STRATEGIES)
    assert len(df) == n_years * n_strategies


def test_buy_and_hold_row_reflects_entry_cost_against_close_ratio():
    """B&H now pays one entry round of fee + slippage so the row is
    symmetric with the strategies. Default rates here are 0.10% fee +
    0.05% slippage from ``run_yearly_validation``."""
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    cost_factor = 1.0 - 0.001 - 0.0005
    for year in df["year"].unique():
        year_close = candles["close"][candles.index.year == year]
        expected = cost_factor * (year_close.iloc[-1] / year_close.iloc[0]) - 1
        bh_row = df[(df["year"] == year) & (df["strategy"] == "buy_and_hold")]
        assert len(bh_row) == 1
        assert bh_row.iloc[0]["return_pct"] == pytest.approx(expected, rel=1e-6)


def test_buy_and_hold_verdict_is_baseline_label():
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    bh_verdicts = df[df["strategy"] == "buy_and_hold"]["verdict"].unique()
    assert list(bh_verdicts) == [VERDICT_BUY_AND_HOLD]


def test_buy_and_hold_metrics_are_consistent_within_each_year():
    """For every strategy in a given year, the buy_and_hold_* reference
    columns should be identical (they depend only on the year's prices)."""
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    for year, group in df.groupby("year"):
        assert group["buy_and_hold_return_pct"].nunique() == 1
        assert group["buy_and_hold_max_drawdown_pct"].nunique() == 1


def test_exposure_in_valid_range():
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    assert (df["exposure_pct"] >= 0).all()
    assert (df["exposure_pct"] <= 1).all()


def test_regime_only_uses_full_history_for_warmup():
    """Year-1 metrics for regime_only_200 should reflect the strategy's
    actual behaviour given the *prior* warmup, not "flat all year because
    no SMA yet". A 365-day year alone would have only 165 valid bars
    for a 200-bar SMA; using full history gives proper signals throughout."""
    # 3 years of slowly rising candles -> the 200-day SMA always behind
    # price after the warmup, so signals should fire in years 1 and 2.
    candles = _daily_candles_multi_year(periods=365 * 3 + 1)
    df = run_yearly_validation(
        candles,
        strategies=(
            FixedStrategySpec("regime_only_200", lambda: RegimeOnlyStrategy(200)),
        ),
    )
    # The strategy is exposed for *some* fraction of every year on this
    # uptrending series. If the year were evaluated in isolation, year 1
    # would have ~0 exposure (still in warmup), so this guards against
    # the regression of slicing candles per year before running.
    years_with_exposure = (df["exposure_pct"] > 0).sum()
    assert years_with_exposure == len(df)


def test_buy_and_hold_returns_year_high_drawdown():
    candles = _daily_candles_multi_year()
    df = run_yearly_validation(candles)
    for _, row in df[df["strategy"] == "buy_and_hold"].iterrows():
        # DD reference column matches the strategy column for buy_and_hold.
        assert row["max_drawdown_pct"] == row["buy_and_hold_max_drawdown_pct"]


def test_empty_candles_returns_empty_frame():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
    )
    df = run_yearly_validation(empty)
    assert df.empty
    assert list(df.columns) == YEARLY_COLUMNS


def test_aggregate_columns_match_spec():
    candles = _daily_candles_multi_year()
    detail = run_yearly_validation(candles)
    agg = aggregate_yearly_results(detail)
    assert list(agg.columns) == YEARLY_AGGREGATE_COLUMNS


def test_aggregate_one_row_per_strategy():
    candles = _daily_candles_multi_year()
    detail = run_yearly_validation(candles)
    agg = aggregate_yearly_results(detail)
    assert len(agg) == detail["strategy"].nunique()


def test_aggregate_best_and_worst_year_bracket_avg():
    candles = _daily_candles_multi_year()
    detail = run_yearly_validation(candles)
    agg = aggregate_yearly_results(detail)
    for _, row in agg.iterrows():
        assert row["worst_year_return"] <= row["avg_annual_return"] <= row["best_year_return"]


def test_aggregate_outperform_count_consistent_with_detail():
    candles = _daily_candles_multi_year()
    detail = run_yearly_validation(candles)
    agg = aggregate_yearly_results(detail)
    for _, row in agg.iterrows():
        strat_rows = detail[detail["strategy"] == row["strategy"]]
        expected = int(
            (strat_rows["return_pct"] > strat_rows["buy_and_hold_return_pct"]).sum()
        )
        assert row["years_outperforming_bh"] == expected


def test_aggregate_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=YEARLY_COLUMNS)
    agg = aggregate_yearly_results(empty)
    assert agg.empty
    assert list(agg.columns) == YEARLY_AGGREGATE_COLUMNS


def test_csv_round_trip(tmp_path):
    candles = _daily_candles_multi_year()
    detail = run_yearly_validation(candles)
    out = tmp_path / "yearly.csv"
    detail.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == YEARLY_COLUMNS
    assert len(loaded) == len(detail)
