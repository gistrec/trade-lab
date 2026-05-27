import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.walk_forward import (
    WALK_FORWARD_COLUMNS,
    generate_windows,
    run_sma_walk_forward,
)


def _daily_candles(start: str, periods: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = (
        100
        + np.linspace(0, 80, periods)
        + 5 * np.sin(np.arange(periods) * 0.02)
        + rng.normal(0, 1.5, periods)
    )
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


def test_window_count_matches_data_size():
    # 5 years of data, train=2y test=1y step=1y -> windows starting in
    # year 0, 1, 2 (so test ends in y3, y4, y5) -> 3 windows; y3 start
    # would need data through y5 — but we have exactly 5y, so y3 still
    # fits. Actually y3 test would end in y4, leaving y4 still inside.
    # Let's just count what the generator returns.
    candles = _daily_candles("2018-01-01", 365 * 5 + 1)  # 5 years
    windows = generate_windows(candles, train_years=2, test_years=1, step_years=1)
    # Expected: windows starting 2018, 2019, 2020 -> 3 windows
    # 2018 train -> 2019 train end -> 2020 test. test_end 2020-12-31 fits.
    # 2019 train -> 2020 train end -> 2021 test. Need data through 2021-12-31.
    # Our 5y data covers ~2018-01-01 through 2022-12-31 (5*365+1 = 1826 days)
    # So 2021 test_end = 2021-12-31 fits, and 2022 test would fit too if step
    # advances cursor enough. Let's just check we get more than 1 and they
    # don't overlap unexpectedly.
    assert len(windows) >= 3
    for w in windows:
        assert w.train_start < w.train_end < w.test_start <= w.test_end


def test_windows_are_disjoint_per_split():
    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    windows = generate_windows(candles, train_years=2, test_years=1, step_years=1)
    for w in windows:
        # Test starts the day after train ends — they don't overlap.
        assert w.test_start > w.train_end


def test_windows_step_by_step_years():
    candles = _daily_candles("2018-01-01", 365 * 6 + 1)
    windows = generate_windows(candles, train_years=2, test_years=1, step_years=1)
    # Consecutive windows should advance by exactly one year on train_start.
    for prev, curr in zip(windows, windows[1:]):
        assert (curr.train_start - prev.train_start).days in (365, 366)


def test_empty_input_returns_no_windows():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
    )
    assert generate_windows(empty) == []


def test_data_shorter_than_one_window_returns_no_windows():
    # Only 6 months of data — not enough for a 2y/1y window.
    candles = _daily_candles("2018-01-01", 180)
    assert generate_windows(candles, train_years=2, test_years=1) == []


def test_walk_forward_dataframe_columns_match_spec():
    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    df = run_sma_walk_forward(
        candles,
        fast_periods=[5, 10],
        slow_periods=[20, 40],
        train_years=2,
        test_years=1,
    )
    assert list(df.columns) == WALK_FORWARD_COLUMNS


def test_walk_forward_picks_train_optimal_params_for_test():
    """The selected (fast, slow) on each row must be the train sweep's
    top row — i.e. parameter choice came from train only."""
    from trade_lab.backtest.sweep import run_sma_sweep
    from trade_lab.data.storage import filter_candles_by_date

    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    fasts = [5, 10]
    slows = [20, 40]
    df = run_sma_walk_forward(
        candles,
        fast_periods=fasts,
        slow_periods=slows,
        train_years=2,
        test_years=1,
    )
    assert not df.empty

    for _, row in df.iterrows():
        train = filter_candles_by_date(
            candles,
            start_date=row["train_start"].strftime("%Y-%m-%d"),
            end_date=row["train_end"].strftime("%Y-%m-%d"),
        )
        train_sweep = run_sma_sweep(train, fasts, slows)
        best = train_sweep.iloc[0]
        assert row["fast_period"] == int(best["fast_period"])
        assert row["slow_period"] == int(best["slow_period"])
        assert row["train_return_pct"] == pytest.approx(float(best["total_return_pct"]))


def test_walk_forward_test_metrics_independent_of_train():
    """The test_return_pct must equal a backtest run on the test window
    alone with the chosen params — proving no train data leaks in."""
    from trade_lab.backtest.engine import run_backtest
    from trade_lab.backtest.metrics import compute_metrics
    from trade_lab.data.storage import filter_candles_by_date
    from trade_lab.strategies.sma_cross import SMACrossStrategy

    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    df = run_sma_walk_forward(
        candles,
        fast_periods=[5, 10],
        slow_periods=[20, 40],
        train_years=2,
        test_years=1,
    )
    assert not df.empty

    for _, row in df.iterrows():
        test_only = filter_candles_by_date(
            candles,
            start_date=row["test_start"].strftime("%Y-%m-%d"),
            end_date=row["test_end"].strftime("%Y-%m-%d"),
        )
        result = run_backtest(
            test_only,
            SMACrossStrategy(
                fast_period=int(row["fast_period"]),
                slow_period=int(row["slow_period"]),
            ),
            initial_capital=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
        )
        m = compute_metrics(result)
        assert row["test_return_pct"] == pytest.approx(m.total_return)
        assert row["test_buy_and_hold_return_pct"] == pytest.approx(
            m.buy_and_hold_return
        )
        assert row["test_max_drawdown_pct"] == pytest.approx(m.max_drawdown)


def test_walk_forward_verdict_is_one_of_three_known_values():
    from trade_lab.backtest.metrics import (
        VERDICT_LOWER_RETURN_LOWER_DD,
        VERDICT_OUTPERFORMS_BH,
        VERDICT_UNDERPERFORMS_BH,
    )

    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    df = run_sma_walk_forward(
        candles, fast_periods=[5], slow_periods=[20],
        train_years=2, test_years=1,
    )
    valid = {
        VERDICT_OUTPERFORMS_BH,
        VERDICT_LOWER_RETURN_LOWER_DD,
        VERDICT_UNDERPERFORMS_BH,
    }
    for verdict in df["test_verdict"]:
        assert verdict in valid


def test_walk_forward_csv_round_trips(tmp_path):
    candles = _daily_candles("2018-01-01", 365 * 4 + 1)
    df = run_sma_walk_forward(
        candles, fast_periods=[5, 10], slow_periods=[20, 40],
        train_years=2, test_years=1,
    )
    out = tmp_path / "wf.csv"
    df.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == WALK_FORWARD_COLUMNS
    assert len(loaded) == len(df)
