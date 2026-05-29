"""Immutable content-hashed OHLCV snapshots.

Why this matters
================
The look-ahead detector (Validation Test 4) replays the backtest on
*the exact bytes* the harness saw on each live decision day. If the
day-T snapshot is a pointer into a mutable shared data store that
gets revised on day T+30, the detector would compare today's signal
to today's revised signal — masking exactly the look-ahead it is
supposed to surface. Wrong tool, false confidence.

This module enforces immutability the only way that survives data
revisions:

* Snapshots are stored as **physically separate files**, addressed
  by **the SHA-256 hash of their own bytes** (content-hash).
* The content-hash is recorded in the journal row written that day.
* Loading by hash verifies the file contents still hash to the same
  value — a bit-flip on disk or an editor accidentally rewriting the
  file is loud at read time.
* Writes are atomic (tmpfile + rename), so a crash mid-write cannot
  produce a partially-written snapshot whose hash mismatches its
  filename.

Serialization is canonical text (not parquet) for two reasons:
parquet's byte representation is not stable across pyarrow versions
or compression settings, and a human reviewing a vintage by-hand
needs to be able to read it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd

_FIELDS = ("open", "high", "low", "close", "volume")


def canonical_serialize(asset_candles: Mapping[str, pd.DataFrame]) -> bytes:
    """Deterministic byte representation of a per-asset OHLCV dict.

    Determinism rules:

    * Assets are emitted in **sorted (alphabetical) key order**,
      regardless of the caller's insertion order — so two cycles
      that happen to pass the dict in different orders still hash to
      the same value.
    * Within an asset, rows are sorted by timestamp ascending.
    * Timestamps are written as ISO-8601 with the UTC offset.
    * Floats are formatted at 8-decimal precision (sufficient for
      crypto prices; insensitive to numpy float-to-str rounding
      variations across platforms).
    * Field separator ``|``; row separator ``\\n``.

    Output line schema:
    ``{asset}|{iso_timestamp}|{open}|{high}|{low}|{close}|{volume}\\n``
    """
    lines: list[str] = []
    for sym in sorted(asset_candles.keys()):
        df = asset_candles[sym]
        if df.empty:
            continue
        idx = df.index
        # Force tz-aware UTC to make the textual timestamp deterministic.
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df = df.copy()
        df.index = idx
        df = df.sort_index()
        for ts, row in df[list(_FIELDS)].iterrows():
            o, h, l, c, v = (float(row[f]) for f in _FIELDS)
            lines.append(
                f"{sym}|{ts.isoformat()}|"
                f"{o:.8f}|{h:.8f}|{l:.8f}|{c:.8f}|{v:.8f}\n"
            )
    return "".join(lines).encode("utf-8")


def content_hash(payload: bytes) -> str:
    """SHA-256 hex digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def vintage_path(vintage_root: Path, h: str) -> Path:
    """Two-level dir layout (h[:2]/h.txt) — keeps any single dir at
    most a few hundred files even after years of daily cycles.
    """
    return Path(vintage_root) / h[:2] / f"{h}.txt"


def store_vintage(
    asset_candles: Mapping[str, pd.DataFrame],
    vintage_root: Path,
) -> str:
    """Canonical-serialize, content-hash, and write atomically.

    Returns the content hash. If a file with this hash already exists,
    the function does NOT rewrite it — the bytes are by definition
    identical (the hash IS the bytes), so re-writing is a no-op and
    we want re-runs of the daily harness on the same data to be
    idempotent at the file-system level.
    """
    payload = canonical_serialize(asset_candles)
    h = content_hash(payload)
    p = vintage_path(vintage_root, h)
    if p.exists():
        return h
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.rename(p)
    return h


def load_vintage(h: str, vintage_root: Path) -> dict[str, pd.DataFrame]:
    """Restore the per-asset OHLCV dict for a given content-hash.

    Verifies on read that the file's bytes still hash to ``h`` —
    detects corruption or an editor inadvertently rewriting the
    file. Raises ``FileNotFoundError`` if the hash is unknown.
    """
    p = vintage_path(vintage_root, h)
    if not p.exists():
        raise FileNotFoundError(
            f"Vintage {h} not found at {p}. The look-ahead detector "
            f"will need this byte-for-byte; verify it was not deleted."
        )
    payload = p.read_bytes()
    actual = content_hash(payload)
    if actual != h:
        raise ValueError(
            f"Vintage corruption: file {p} hashes to {actual} "
            f"but is named {h}. Do not use this snapshot."
        )
    return _parse_canonical_serialize(payload)


def _parse_canonical_serialize(payload: bytes) -> dict[str, pd.DataFrame]:
    """Reverse of ``canonical_serialize``."""
    text = payload.decode("utf-8")
    per_asset_rows: dict[str, list[tuple]] = {}
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split("|")
        sym = parts[0]
        ts = pd.Timestamp(parts[1])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        o, h, l, c, v = (float(x) for x in parts[2:7])
        per_asset_rows.setdefault(sym, []).append((ts, o, h, l, c, v))

    out: dict[str, pd.DataFrame] = {}
    for sym, rows in per_asset_rows.items():
        idx = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
        df = pd.DataFrame(
            {
                "open":   [r[1] for r in rows],
                "high":   [r[2] for r in rows],
                "low":    [r[3] for r in rows],
                "close":  [r[4] for r in rows],
                "volume": [r[5] for r in rows],
            },
            index=idx,
        )
        out[sym] = df
    return out
