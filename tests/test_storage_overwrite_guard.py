"""``save_candles`` must not silently shrink stored history (regression: H4).

``save_candles`` overwrites the whole Parquet file (no merge). A truncated
fetch (e.g. only the newest ~1000 candles) saved on top of years of history
used to destroy the file without any error, and downstream backtests then
silently ran on the stub. An existing non-empty file may only be replaced
by data that covers at least its date range; anything smaller raises unless
``force=True``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.data.storage import load_candles, save_candles

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _frame(start: str, periods: int, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({col: 1.0 for col in _COLUMNS}, index=idx)


def test_first_save_needs_no_force(tmp_path):
    df = _frame("2020-01-01", 100)
    save_candles(df, tmp_path, "binance", "BTC/USDT", "1h")
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 100


def test_shrinking_overwrite_raises_and_preserves_file(tmp_path):
    full = _frame("2020-01-01", 5000)
    save_candles(full, tmp_path, "binance", "BTC/USDT", "1h")

    truncated = full.iloc[-1000:]  # newest window only, like since=None
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_candles(truncated, tmp_path, "binance", "BTC/USDT", "1h")

    # The original file must be intact after the refused save.
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 5000


def test_empty_frame_over_existing_raises(tmp_path):
    full = _frame("2020-01-01", 50)
    save_candles(full, tmp_path, "binance", "BTC/USDT", "1h")

    empty = full.iloc[0:0]
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_candles(empty, tmp_path, "binance", "BTC/USDT", "1h")


def test_force_allows_shrinking_overwrite(tmp_path):
    full = _frame("2020-01-01", 5000)
    save_candles(full, tmp_path, "binance", "BTC/USDT", "1h")

    truncated = full.iloc[-1000:]
    save_candles(truncated, tmp_path, "binance", "BTC/USDT", "1h", force=True)
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 1000


def test_covering_refetch_passes_without_force(tmp_path):
    old = _frame("2020-01-01", 3000)
    save_candles(old, tmp_path, "binance", "BTC/USDT", "1h")

    # Legitimate update: same start, extended tail (a full re-fetch).
    extended = _frame("2020-01-01", 4000)
    save_candles(extended, tmp_path, "binance", "BTC/USDT", "1h")
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 4000

    # Identical range is also fine (idempotent re-run).
    save_candles(extended, tmp_path, "binance", "BTC/USDT", "1h")
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 4000


def test_guard_is_per_symbol_and_timeframe(tmp_path):
    save_candles(_frame("2020-01-01", 5000), tmp_path, "binance", "BTC/USDT", "1h")

    # A small file for a *different* symbol/timeframe is a first save,
    # not a shrink of the BTC 1h history.
    save_candles(_frame("2024-01-01", 10), tmp_path, "binance", "ETH/USDT", "1h")
    save_candles(_frame("2024-01-01", 10, freq="1D"), tmp_path, "binance", "BTC/USDT", "1d")
    assert len(load_candles(tmp_path, "binance", "BTC/USDT", "1h")) == 5000
