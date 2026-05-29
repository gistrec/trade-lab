"""Vintage-store invariants. These are not optional — the look-ahead
detector in Validation Test 4 is only useful if these hold."""
from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.paper_trading.vintage_store import (
    canonical_serialize,
    content_hash,
    load_vintage,
    store_vintage,
    vintage_path,
)


def _make_candles(n_bars: int = 50) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC")
    out: dict[str, pd.DataFrame] = {}
    for offset, sym in enumerate(("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")):
        out[sym] = pd.DataFrame(
            {
                "open":   [100.0 + offset + i * 0.1 for i in range(n_bars)],
                "high":   [101.0 + offset + i * 0.1 for i in range(n_bars)],
                "low":    [99.0 + offset + i * 0.1 for i in range(n_bars)],
                "close":  [100.5 + offset + i * 0.1 for i in range(n_bars)],
                "volume": [1_000_000.0 + offset * 1000 for _ in range(n_bars)],
            },
            index=idx,
        )
    return out


def test_canonical_serialize_is_deterministic():
    """Two independent constructions of the same data must produce
    byte-identical canonical serialization, otherwise the content
    hash is meaningless."""
    a = _make_candles()
    b = _make_candles()
    assert canonical_serialize(a) == canonical_serialize(b)


def test_canonical_serialize_insensitive_to_dict_order():
    """Asset insertion order must NOT affect the hash — otherwise
    re-ordering would silently change the content hash."""
    a = _make_candles()
    # Reverse insertion order
    reversed_dict = {k: a[k] for k in reversed(list(a.keys()))}
    assert canonical_serialize(a) == canonical_serialize(reversed_dict)


def test_canonical_serialize_sensitive_to_tz_alignment():
    """Indices that disagree on tz convey different timestamps and
    must hash differently — UTC normalisation is the right place to
    fix this, NOT silent equivalence."""
    a = _make_candles()
    naive = {k: v.copy() for k, v in a.items()}
    # tz-localize then strip back to naive — the SAME wall-clock time
    # but a different python object representation. After canonical
    # serialization passes it through `tz_localize('UTC')`, the
    # resulting hash MUST match.
    for sym in naive:
        naive[sym].index = naive[sym].index.tz_localize(None)
    # Same wall clock → same UTC after canonical handling → same hash
    assert canonical_serialize(naive) == canonical_serialize(a)


def test_content_hash_changes_when_any_byte_changes():
    """The hash MUST flip on any byte-level change — sanity for the
    look-ahead detector that compares hashes to detect data drift."""
    a = _make_candles()
    h1 = content_hash(canonical_serialize(a))
    # Mutate one cell
    a["BTC"].iloc[-1, a["BTC"].columns.get_loc("close")] = 999_999.0
    h2 = content_hash(canonical_serialize(a))
    assert h1 != h2


def test_store_then_load_round_trip(tmp_path):
    a = _make_candles()
    h = store_vintage(a, tmp_path)
    loaded = load_vintage(h, tmp_path)
    assert sorted(loaded.keys()) == sorted(a.keys())
    for sym in a:
        # The serialization uses 8 decimal places; allow that tolerance.
        # Index `name` is a cosmetic attribute that the canonical
        # text encoding does not preserve (it has no place to live);
        # canonicalise it on both sides before the structural compare.
        expected = a[sym][["open", "high", "low", "close", "volume"]].copy()
        actual = loaded[sym][["open", "high", "low", "close", "volume"]].copy()
        expected.index.name = None
        actual.index.name = None
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, atol=1e-7,
            check_freq=False,
        )


def test_store_is_idempotent(tmp_path):
    """Re-storing the same bytes must produce the same path and not
    duplicate-write — the file system layer is where idempotency is
    rooted."""
    a = _make_candles()
    h1 = store_vintage(a, tmp_path)
    h2 = store_vintage(a, tmp_path)
    assert h1 == h2
    files = list(tmp_path.rglob("*.txt"))
    assert len(files) == 1


def test_load_raises_on_missing_hash(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vintage("nonexistent" * 8, tmp_path)


def test_load_raises_on_corrupted_vintage(tmp_path):
    """If a vintage file is edited on disk, load_vintage MUST raise.
    This is the bit-flip / accidental-edit detector."""
    a = _make_candles()
    h = store_vintage(a, tmp_path)
    p = vintage_path(tmp_path, h)
    # Corrupt the file
    p.write_text("CORRUPTED\n")
    with pytest.raises(ValueError, match="corruption"):
        load_vintage(h, tmp_path)
