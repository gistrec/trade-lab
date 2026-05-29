"""Tests for pure-function helpers in ``trade_lab.monitoring.app``.

The Streamlit rendering itself is verified manually (no ScriptRunContext
in pytest), but the helpers that format timestamps for the Status tab
are pure and worth pinning so a future refactor does not silently
regress the narrow-screen layout."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_lab.monitoring.app import _humanize_iso, _humanize_relative


NOW = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)


def test_relative_none_returns_dash():
    assert _humanize_relative(None, now=NOW) == "—"


def test_relative_seconds():
    iso = "2026-05-29T11:59:30+00:00"  # 30 seconds before NOW
    assert _humanize_relative(iso, now=NOW) == "30s ago"


def test_relative_minutes():
    iso = "2026-05-29T11:55:00+00:00"  # 5 minutes before
    assert _humanize_relative(iso, now=NOW) == "5m ago"


def test_relative_hours_only():
    iso = "2026-05-29T09:00:00+00:00"  # 3 hours before, no minute remainder
    assert _humanize_relative(iso, now=NOW) == "3h ago"


def test_relative_hours_and_minutes():
    iso = "2026-05-29T09:30:00+00:00"  # 2h 30m before
    assert _humanize_relative(iso, now=NOW) == "2h 30m ago"


def test_relative_days_only():
    iso = "2026-05-26T12:00:00+00:00"  # exactly 3 days
    assert _humanize_relative(iso, now=NOW) == "3d ago"


def test_relative_days_and_hours():
    iso = "2026-05-26T08:00:00+00:00"  # 3d 4h before
    assert _humanize_relative(iso, now=NOW) == "3d 4h ago"


def test_relative_caps_long_intervals_to_days():
    """Past ~30 days, only days are shown (no day+hour breakdown)
    — beyond that granularity the operator wants days at a glance."""
    iso = "2026-04-01T12:00:00+00:00"   # 58 days before, not 58d 0h
    out = _humanize_relative(iso, now=NOW)
    assert out.endswith("d ago")
    assert "h" not in out


def test_relative_in_future():
    iso = "2026-05-29T12:30:00+00:00"
    assert _humanize_relative(iso, now=NOW) == "in the future"


def test_relative_naive_timestamps_assumed_utc():
    """Naive timestamps must be treated as UTC; the writer should
    always emit an offset, but defensive parsing protects against a
    regression."""
    iso = "2026-05-29T11:55:00"   # no tz
    assert _humanize_relative(iso, now=NOW) == "5m ago"


def test_relative_value_shorter_than_absolute():
    """Width-regression pin: the whole point of the helper is to fit a
    narrow column. Verify that for any plausible cycle interval (≤ a
    few hours stale), the relative form is materially shorter than the
    absolute one."""
    iso = "2026-05-29T09:30:15+00:00"
    rel = _humanize_relative(iso, now=NOW)
    abs_ = _humanize_iso(iso)
    assert len(rel) <= 10
    assert len(rel) < len(abs_)


def test_iso_unchanged_by_helpers():
    """Sanity: _humanize_iso still produces the absolute form for the
    caption (not displaced by the relative form)."""
    iso = "2026-05-29T09:30:15+00:00"
    assert _humanize_iso(iso) == "2026-05-29 09:30:15 UTC"
