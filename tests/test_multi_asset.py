import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.multi_asset import (
    MULTI_ASSET_AGGREGATE_COLUMNS,
    MULTI_ASSET_DETAIL_COLUMNS,
    MULTI_ASSET_SUMMARY_COLUMNS,
    aggregate_multi_asset,
    run_multi_asset_yearly_validation,
    summarize_across_assets,
)
from trade_lab.backtest.yearly import (
    DEFAULT_FIXED_STRATEGIES,
    FixedStrategySpec,
    run_yearly_validation,
)
from trade_lab.strategies.regime_only import RegimeOnlyStrategy
from trade_lab.strategies.sma_cross import SMACrossStrategy


def _daily_candles(start: str, periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


def _two_asset_candles():
    return {
        "ABC/USDT": _daily_candles("2020-01-01", 365 * 3 + 1, seed=0),
        "XYZ/USDT": _daily_candles("2020-01-01", 365 * 3 + 1, seed=1),
    }


def test_detail_columns_include_asset_first():
    df = run_multi_asset_yearly_validation(_two_asset_candles())
    assert list(df.columns) == MULTI_ASSET_DETAIL_COLUMNS
    assert df.columns[0] == "asset"


def test_detail_has_per_asset_rows():
    df = run_multi_asset_yearly_validation(_two_asset_candles())
    assert set(df["asset"].unique()) == {"ABC/USDT", "XYZ/USDT"}


def test_per_asset_rows_match_single_asset_validation():
    """Running multi-asset on one symbol should produce exactly the same
    per-year rows as run_yearly_validation alone (modulo the extra asset
    column)."""
    assets = _two_asset_candles()
    multi = run_multi_asset_yearly_validation(assets)
    for asset, candles in assets.items():
        single = run_yearly_validation(candles)
        sliced = multi[multi["asset"] == asset].drop(columns=["asset"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(sliced, single)


def test_empty_assets_returns_empty_frame():
    df = run_multi_asset_yearly_validation({})
    assert df.empty
    assert list(df.columns) == MULTI_ASSET_DETAIL_COLUMNS


def test_skips_empty_asset_frame():
    assets = _two_asset_candles()
    assets["EMPTY/USDT"] = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
    )
    df = run_multi_asset_yearly_validation(assets)
    assert "EMPTY/USDT" not in set(df["asset"])
    assert set(df["asset"].unique()) == {"ABC/USDT", "XYZ/USDT"}


def test_aggregate_columns_match_spec():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    assert list(agg.columns) == MULTI_ASSET_AGGREGATE_COLUMNS


def test_aggregate_one_row_per_asset_strategy_pair():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    expected = detail.drop_duplicates(["asset", "strategy"]).shape[0]
    assert len(agg) == expected


def test_aggregate_worst_le_avg_le_best_per_pair():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    for _, row in agg.iterrows():
        assert row["worst_year_return"] <= row["avg_annual_return"]
        assert row["avg_annual_return"] <= row["best_year_return"]


def test_summary_columns_match_spec():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    summary = summarize_across_assets(agg)
    assert list(summary.columns) == MULTI_ASSET_SUMMARY_COLUMNS


def test_summary_n_assets_matches_unique_assets_in_aggregate():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    summary = summarize_across_assets(agg)
    for _, row in summary.iterrows():
        per_asset = agg[agg["strategy"] == row["strategy"]]
        assert row["n_assets"] == len(per_asset)


def test_summary_totals_equal_sum_over_assets():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    summary = summarize_across_assets(agg)
    for _, row in summary.iterrows():
        per_asset = agg[agg["strategy"] == row["strategy"]]
        assert row["total_years_outperforming_bh"] == int(per_asset["years_outperforming_bh"].sum())
        assert row["total_years_lower_dd"] == int(per_asset["years_lower_dd_than_bh"].sum())


def test_summary_avg_return_equals_mean_over_assets():
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    agg = aggregate_multi_asset(detail)
    summary = summarize_across_assets(agg)
    for _, row in summary.iterrows():
        per_asset = agg[agg["strategy"] == row["strategy"]]
        assert row["avg_return_across_assets"] == pytest.approx(
            float(per_asset["avg_annual_return"].mean())
        )
        assert row["avg_worst_year"] == pytest.approx(
            float(per_asset["worst_year_return"].mean())
        )


def test_aggregate_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=MULTI_ASSET_DETAIL_COLUMNS)
    agg = aggregate_multi_asset(empty)
    assert agg.empty
    assert list(agg.columns) == MULTI_ASSET_AGGREGATE_COLUMNS


def test_summary_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=MULTI_ASSET_AGGREGATE_COLUMNS)
    summary = summarize_across_assets(empty)
    assert summary.empty
    assert list(summary.columns) == MULTI_ASSET_SUMMARY_COLUMNS


def test_assets_with_different_history_lengths_are_supported():
    # ABC has full 3 years; SOL-like has only 1 year — both should produce
    # rows, with the shorter asset contributing fewer years.
    assets = {
        "FULL/USDT": _daily_candles("2020-01-01", 365 * 3 + 1, seed=0),
        "SHORT/USDT": _daily_candles("2022-01-01", 365 + 1, seed=2),
    }
    detail = run_multi_asset_yearly_validation(assets)
    assert (detail[detail["asset"] == "FULL/USDT"]["year"].nunique()) >= 3
    assert (detail[detail["asset"] == "SHORT/USDT"]["year"].nunique()) >= 1


def test_csv_round_trip(tmp_path):
    detail = run_multi_asset_yearly_validation(_two_asset_candles())
    out = tmp_path / "multi.csv"
    detail.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == MULTI_ASSET_DETAIL_COLUMNS
    assert len(loaded) == len(detail)
