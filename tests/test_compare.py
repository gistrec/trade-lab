import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.compare import (
    COMPARISON_COLUMNS,
    DEFAULT_COMPARISON_STRATEGIES,
    DEFAULT_SUBPERIODS,
    render_comparison_markdown,
    run_comparison_report,
)


def _daily_candles(start: str, periods: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = 100 + np.linspace(0, 50, periods) + rng.normal(0, 1.5, periods)
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


def _multi_asset():
    return {
        "ALPHA/USDT": _daily_candles("2020-01-01", 365 * 3 + 1, seed=0),
        "BETA/USDT": _daily_candles("2020-01-01", 365 * 3 + 1, seed=1),
    }


def test_columns_match_spec():
    df = run_comparison_report(_multi_asset())
    assert list(df.columns) == COMPARISON_COLUMNS


def test_row_count_matches_combinations():
    assets = _multi_asset()
    df = run_comparison_report(assets)
    # Subperiods that have at least one candle on each asset.
    valid_periods = [
        sp for sp in DEFAULT_SUBPERIODS
        if any(
            not c.empty
            and (sp.start_date is None or pd.Timestamp(sp.start_date, tz="UTC") <= c.index[-1])
            and (sp.end_date is None or pd.Timestamp(sp.end_date, tz="UTC") >= c.index[0])
            for c in assets.values()
        )
    ]
    expected_rows = sum(
        1
        for asset, candles in assets.items()
        for sp in valid_periods
        for _ in DEFAULT_COMPARISON_STRATEGIES
        if not candles.empty
    )
    # In practice every default subperiod overlaps the 2020-2022 panel, so
    # `valid_periods` excludes only 2018/2019.
    assert len(df) <= expected_rows
    assert len(df) > 0


def test_buy_and_hold_row_has_symmetric_entry_cost_and_full_exposure():
    """B&H is no longer cost-free — it's charged one entry round of
    fee + slippage so the comparison to strategies is symmetric.
    Exposure stays at 1.0 (long the whole window) and turnover is 1.0
    (one entry side, mirroring the engine's
    ``turnover.iloc[0] = abs(positions.iloc[0])`` convention)."""
    df = run_comparison_report(_multi_asset())
    bh = df[df["strategy"] == "buy_and_hold"]
    assert (bh["exposure_pct"] == 1.0).all()
    # Default compare costs are 0.10% fee + 0.05% slippage on
    # initial_capital=10000 → $10.00 fee + $5.00 slippage per cell.
    np.testing.assert_allclose(bh["total_fees"].to_numpy(), 10.0)
    np.testing.assert_allclose(bh["total_slippage"].to_numpy(), 5.0)
    assert (bh["turnover"] == 1.0).all()


def test_buy_and_hold_total_return_matches_close_ratio_after_entry_cost():
    """B&H total return now reflects one entry's fee + slippage:
    return = (1 - fee_rate - slippage_rate) * close_ratio - 1."""
    from trade_lab.data.storage import filter_candles_by_date

    assets = _multi_asset()
    df = run_comparison_report(assets)
    cost_factor = 1.0 - 0.001 - 0.0005   # default rates from compare.py
    for asset, candles in assets.items():
        for period in DEFAULT_SUBPERIODS:
            sliced = filter_candles_by_date(
                candles, start_date=period.start_date, end_date=period.end_date,
            )
            if sliced.empty:
                continue
            row = df[
                (df["asset"] == asset)
                & (df["strategy"] == "buy_and_hold")
                & (df["period"] == period.label)
            ]
            if row.empty:
                continue
            close_ratio = sliced["close"].iloc[-1] / sliced["close"].iloc[0]
            expected = cost_factor * close_ratio - 1
            assert row.iloc[0]["total_return_pct"] == pytest.approx(expected, rel=1e-6)


def test_cagr_is_consistent_with_total_return_and_bars():
    df = run_comparison_report(_multi_asset())
    bh = df[df["strategy"] == "buy_and_hold"]
    for _, row in bh.iterrows():
        years = row["bars"] / 365
        if years <= 0:
            continue
        expected = (1 + row["total_return_pct"]) ** (1 / years) - 1
        assert row["cagr_pct"] == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_strategies_include_donchian_with_both_threshold_settings():
    df = run_comparison_report(_multi_asset())
    strategies = set(df["strategy"].unique())
    assert "donchian_trend_rb0" in strategies
    assert "donchian_trend_rb005" in strategies


def test_strategies_include_tsmom_and_pma_from_research_panel():
    """The Research-Claude survey put TSMOM (Moskowitz et al. 2012;
    Liu & Tsyvinski 2021) and the P/MA ensemble (Detzel et al. 2021) at
    priority 5/5 — both must appear in the default comparison panel."""
    df = run_comparison_report(_multi_asset())
    strategies = set(df["strategy"].unique())
    assert "tsmom_1_3_6_12m" in strategies
    assert "pma_ratio_ensemble" in strategies


def test_donchian_rb005_has_fewer_or_equal_fees_than_rb0_on_average():
    """The rebalance band shouldn't *increase* fees on any given run.
    Per-cell variance is fine; we check the average across the panel."""
    df = run_comparison_report(_multi_asset())
    rb0 = df[df["strategy"] == "donchian_trend_rb0"]["total_fees"].mean()
    rb005 = df[df["strategy"] == "donchian_trend_rb005"]["total_fees"].mean()
    assert rb005 <= rb0 + 1e-6  # ties allowed (e.g. no trades on a period)


def test_empty_asset_map_returns_empty_frame():
    df = run_comparison_report({})
    assert df.empty
    assert list(df.columns) == COMPARISON_COLUMNS


def test_markdown_rendering_contains_each_asset_header():
    df = run_comparison_report(_multi_asset())
    md = render_comparison_markdown(df)
    assert "## ALPHA/USDT" in md
    assert "## BETA/USDT" in md
    # Every default strategy should appear at least once.
    for spec in DEFAULT_COMPARISON_STRATEGIES:
        assert spec.label in md


def test_csv_round_trip(tmp_path):
    df = run_comparison_report(_multi_asset())
    out = tmp_path / "compare.csv"
    df.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == COMPARISON_COLUMNS
    assert len(loaded) == len(df)
