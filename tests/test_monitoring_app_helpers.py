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


# ---------------------------------------------------------------------------
# Tab failure containment
# ---------------------------------------------------------------------------


def test_render_tab_safely_contains_exception(monkeypatch):
    """A tab whose renderer raises (ImportError on a renamed research
    module, TypeError from a schema-drifted journal row) must surface
    a visible error instead of killing the whole Streamlit run."""
    import trade_lab.monitoring.app as app

    errors: list[str] = []
    monkeypatch.setattr(app.st, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(app.st, "caption", lambda msg: None)

    def broken_tab():
        raise TypeError("unexpected keyword argument 'new_field_from_v2'")

    app._render_tab_safely("Validation", broken_tab)  # must not raise

    assert len(errors) == 1
    assert "Validation" in errors[0]
    assert "TypeError" in errors[0]


def test_render_tab_safely_passes_through_on_success(monkeypatch):
    import trade_lab.monitoring.app as app

    errors: list[str] = []
    monkeypatch.setattr(app.st, "error", lambda msg: errors.append(msg))
    rendered = []

    app._render_tab_safely("Status", lambda: rendered.append(True))

    assert rendered == [True]
    assert errors == []


# ---------------------------------------------------------------------------
# Days since gate OPEN — counts distinct days, not cycles
# ---------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, cycles):
        self._cycles = cycles

    def cycles(self, n=20):
        return self._cycles[-n:]


def _sig_cycle(asof: str, gate_open: bool) -> dict:
    return {"signal": {"asof": asof, "sma_gate_open": gate_open}}


def test_days_since_gate_counts_days_not_cycles():
    """With the hourly dry-run sharing the journal, one closed day is
    ~24 cycles. The metric says 'Days' — it must dedupe by asof date."""
    from trade_lab.monitoring.app import _days_since_gate_last_open

    cycles = [_sig_cycle("2026-06-10T00:00:00+00:00", True)]
    for hour in range(24):  # one full closed day of hourly dry-runs
        cycles.append(_sig_cycle(f"2026-06-11T{hour:02d}:00:00+00:00", False))
    assert _days_since_gate_last_open(_FakeReader(cycles)) == 1


def test_days_since_gate_zero_when_latest_open():
    from trade_lab.monitoring.app import _days_since_gate_last_open

    cycles = [
        _sig_cycle("2026-06-10T00:00:00+00:00", False),
        _sig_cycle("2026-06-11T00:00:00+00:00", True),
    ]
    assert _days_since_gate_last_open(_FakeReader(cycles)) == 0


def test_days_since_gate_none_when_never_open():
    from trade_lab.monitoring.app import _days_since_gate_last_open

    cycles = [_sig_cycle("2026-06-11T00:00:00+00:00", False)]
    assert _days_since_gate_last_open(_FakeReader(cycles)) is None


def test_days_since_gate_skips_cycles_without_signal():
    """Failed and reconstruction cycles say nothing about the gate."""
    from trade_lab.monitoring.app import _days_since_gate_last_open

    cycles = [
        _sig_cycle("2026-06-09T00:00:00+00:00", True),
        {"signal": None, "outcome": "failed"},
        {"outcome": "reconstructed"},
        _sig_cycle("2026-06-11T00:00:00+00:00", False),
    ]
    assert _days_since_gate_last_open(_FakeReader(cycles)) == 1
