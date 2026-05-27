import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.sweep import SWEEP_COLUMNS, run_sma_sweep


def _candles(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.linspace(0, 40, n) + rng.normal(0, 1.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
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
    df = run_sma_sweep(_candles(), fast_periods=[5, 10], slow_periods=[50, 100])
    assert list(df.columns) == SWEEP_COLUMNS


def test_skips_invalid_combinations_where_fast_geq_slow():
    df = run_sma_sweep(
        _candles(),
        fast_periods=[10, 20, 50],
        slow_periods=[10, 20, 50],
    )
    # Valid pairs (fast < slow): (10,20), (10,50), (20,50) -> 3 rows.
    assert len(df) == 3
    assert set(zip(df["fast_period"], df["slow_period"])) == {
        (10, 20), (10, 50), (20, 50)
    }


def test_result_is_sorted_by_total_return_descending():
    df = run_sma_sweep(
        _candles(),
        fast_periods=[5, 10, 20],
        slow_periods=[50, 100, 150],
    )
    assert df["total_return_pct"].is_monotonic_decreasing


def test_empty_when_all_combinations_invalid():
    df = run_sma_sweep(
        _candles(),
        fast_periods=[50, 100],
        slow_periods=[10, 20],
    )
    assert df.empty
    assert list(df.columns) == SWEEP_COLUMNS


def test_empty_grid_returns_empty_frame():
    df = run_sma_sweep(_candles(), fast_periods=[], slow_periods=[50, 100])
    assert df.empty
    assert list(df.columns) == SWEEP_COLUMNS


def test_periods_are_integers_in_output():
    df = run_sma_sweep(
        _candles(), fast_periods=[5, 10], slow_periods=[50, 100],
    )
    assert df["fast_period"].dtype.kind in "iu"
    assert df["slow_period"].dtype.kind in "iu"


def test_csv_round_trip_preserves_columns(tmp_path):
    df = run_sma_sweep(
        _candles(), fast_periods=[5, 10], slow_periods=[50, 100],
    )
    out = tmp_path / "sweep.csv"
    df.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == SWEEP_COLUMNS
    assert len(loaded) == len(df)


def test_metrics_match_individual_backtest():
    """The sweep should produce the same numbers as a one-off backtest with
    the same parameters."""
    from trade_lab.backtest.engine import run_backtest
    from trade_lab.backtest.metrics import compute_metrics
    from trade_lab.strategies.sma_cross import SMACrossStrategy

    candles = _candles()
    df = run_sma_sweep(
        candles, fast_periods=[10], slow_periods=[50],
        initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005,
    )
    direct = run_backtest(
        candles,
        SMACrossStrategy(fast_period=10, slow_period=50),
        initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005,
    )
    expected = compute_metrics(direct)
    row = df.iloc[0]
    assert row["final_equity"] == pytest.approx(expected.final_equity)
    assert row["total_return_pct"] == pytest.approx(expected.total_return)
    assert row["max_drawdown_pct"] == pytest.approx(expected.max_drawdown)
    assert row["num_trades"] == expected.num_trades
    assert row["fees_paid"] == pytest.approx(expected.total_fees)
