import pandas as pd
import pytest

from trade_lab.data.storage import filter_candles_by_date


def _candles(start: str, periods: int, freq: str, tz: str | None = "UTC"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=idx,
    )


def test_no_dates_returns_input_unchanged():
    candles = _candles("2024-01-01", 5, "1D")
    out = filter_candles_by_date(candles)
    assert out.equals(candles)


def test_start_date_is_inclusive():
    candles = _candles("2024-01-01", 10, "1D")
    out = filter_candles_by_date(candles, start_date="2024-01-03")
    assert out.index[0] == pd.Timestamp("2024-01-03", tz="UTC")
    assert len(out) == 8


def test_end_date_includes_full_day_for_hourly_bars():
    # 72 hourly bars covering 2024-01-01 through 2024-01-03 inclusive.
    candles = _candles("2024-01-01", 72, "1h")
    out = filter_candles_by_date(candles, end_date="2024-01-02")
    # Should keep Jan 1 (24 bars) + all of Jan 2 (24 bars) = 48 bars.
    assert len(out) == 48
    assert out.index[-1] == pd.Timestamp("2024-01-02 23:00:00", tz="UTC")


def test_both_bounds_inclusive():
    candles = _candles("2024-01-01", 10, "1D")
    out = filter_candles_by_date(
        candles, start_date="2024-01-03", end_date="2024-01-05"
    )
    assert len(out) == 3
    assert out.index[0] == pd.Timestamp("2024-01-03", tz="UTC")
    assert out.index[-1] == pd.Timestamp("2024-01-05", tz="UTC")


def test_range_completely_outside_data_returns_empty():
    candles = _candles("2024-01-01", 5, "1D")
    out = filter_candles_by_date(candles, start_date="2025-01-01")
    assert out.empty
    # Columns should be preserved even when there are zero rows.
    assert list(out.columns) == list(candles.columns)


def test_tz_naive_index_is_supported():
    candles = _candles("2024-01-01", 10, "1D", tz=None)
    out = filter_candles_by_date(
        candles, start_date="2024-01-04", end_date="2024-01-07"
    )
    assert len(out) == 4
    assert out.index[0] == pd.Timestamp("2024-01-04")
    assert out.index[-1] == pd.Timestamp("2024-01-07")


def test_invalid_date_string_raises():
    candles = _candles("2024-01-01", 5, "1D")
    with pytest.raises((ValueError, Exception)):
        filter_candles_by_date(candles, start_date="not-a-date")


def test_only_start_keeps_tail():
    candles = _candles("2024-01-01", 10, "1D")
    out = filter_candles_by_date(candles, start_date="2024-01-08")
    assert len(out) == 3
    assert out.index[0] == pd.Timestamp("2024-01-08", tz="UTC")


def test_only_end_keeps_head():
    candles = _candles("2024-01-01", 10, "1D")
    out = filter_candles_by_date(candles, end_date="2024-01-03")
    assert len(out) == 3
    assert out.index[-1] == pd.Timestamp("2024-01-03", tz="UTC")


def test_does_not_mutate_input_frame():
    candles = _candles("2024-01-01", 5, "1D")
    snapshot = candles.copy()
    filter_candles_by_date(candles, start_date="2024-01-03")
    assert candles.equals(snapshot)
