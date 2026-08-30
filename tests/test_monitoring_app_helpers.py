"""Tests for pure-function helpers in ``trade_lab.monitoring.app``.

The Streamlit rendering itself is verified manually (no ScriptRunContext
in pytest), but the helpers that format timestamps for the Status tab
are pure and worth pinning so a future refactor does not silently
regress the narrow-screen layout."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trade_lab.monitoring.app import (
    _collapse_signal_rows, _humanize_iso, _humanize_relative,
    _sim_equity_metric,
)


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
# Unfilled-order count — dry-run (planning-only) cycles must not warn
# ---------------------------------------------------------------------------


def test_unfilled_count_none_for_dry_run_planning_only_cycle():
    """A dry-run cycle writes orders_executed=None with orders_planned
    populated. That is planning-only, not 'orders failed to fill', so the
    partial-fill warning must be suppressed (return None), not fire on the
    hourly dry-run cycles that share the monitored journal (regression:
    R2)."""
    from trade_lab.monitoring.app import _unfilled_order_count

    dry_run_cycle = {
        "outcome": "success",
        "orders_planned": [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}],
        "orders_executed": None,
    }
    assert _unfilled_order_count(dry_run_cycle) is None


def test_unfilled_count_counts_live_cycle_partial():
    """A live cycle (orders_executed populated) with a planned order that
    did not fully close returns the unfilled count."""
    from trade_lab.monitoring.app import _unfilled_order_count

    live_cycle = {
        "outcome": "success",
        "orders_planned": [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}],
        "orders_executed": [
            {"terminal_status": "closed"},
            {"terminal_status": "partial"},
        ],
    }
    assert _unfilled_order_count(live_cycle) == 1


def test_unfilled_count_zero_when_all_closed():
    from trade_lab.monitoring.app import _unfilled_order_count

    live_cycle = {
        "outcome": "success",
        "orders_planned": [{"symbol": "BTC/USDT"}],
        "orders_executed": [{"terminal_status": "closed"}],
    }
    assert _unfilled_order_count(live_cycle) == 0


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


def test_gate_duration_counts_days_not_cycles():
    """With the hourly dry-run sharing the journal, one closed day is
    ~24 cycles. The metric says 'Days' — it must dedupe by asof date."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [_sig_cycle("2026-06-10T00:00:00+00:00", True)]
    for hour in range(24):  # one full closed day of hourly dry-runs
        cycles.append(_sig_cycle(f"2026-06-11T{hour:02d}:00:00+00:00", False))
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("CLOSED", 1)


def test_gate_duration_reports_open_streak_not_zero():
    """The regression this metric was rebuilt for: while the gate is OPEN
    the old 'days since OPEN' collapsed to 0 and hid how long the regime
    had been running. Now it reports the length of the open streak."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [
        _sig_cycle("2026-08-18T00:00:00+00:00", False),
        _sig_cycle("2026-08-19T00:00:00+00:00", False),
        _sig_cycle("2026-08-20T00:00:00+00:00", True),
        _sig_cycle("2026-08-21T00:00:00+00:00", True),
    ]
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("OPEN", 2)


def test_gate_duration_open_streak_dedupes_repeated_cycles():
    """Four 6-hourly dry-runs over one open bar are one open day."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [_sig_cycle("2026-08-20T00:00:00+00:00", False)]
    cycles += [_sig_cycle("2026-08-21T00:00:00+00:00", True)] * 4
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("OPEN", 1)


def test_gate_duration_closed_streak_when_never_open():
    """No OPEN in the window is still a usable reading, not a dash: the
    gate is closed and has been for every bar we can see."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [
        _sig_cycle("2026-06-10T00:00:00+00:00", False),
        _sig_cycle("2026-06-11T00:00:00+00:00", False),
    ]
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("CLOSED", 2)


def test_gate_duration_none_without_any_gate_reading():
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [{"signal": None, "outcome": "failed"}, {"outcome": "recon"}]
    assert _gate_state_duration_days(_FakeReader(cycles)) == (None, None)


def test_gate_duration_skips_cycles_without_signal():
    """Failed and reconstruction cycles say nothing about the gate."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [
        _sig_cycle("2026-06-09T00:00:00+00:00", True),
        {"signal": None, "outcome": "failed"},
        {"outcome": "reconstructed"},
        _sig_cycle("2026-06-11T00:00:00+00:00", False),
    ]
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("CLOSED", 1)


def test_gate_duration_intraday_flip_counts_date_as_open():
    """A date carrying any OPEN reading counts as OPEN — the regime was on
    at some point that day (rule inherited from the previous metric)."""
    from trade_lab.monitoring.app import _gate_state_duration_days

    cycles = [
        _sig_cycle("2026-06-10T00:00:00+00:00", True),
        _sig_cycle("2026-06-11T00:00:00+00:00", False),
        _sig_cycle("2026-06-11T12:00:00+00:00", True),
    ]
    assert _gate_state_duration_days(_FakeReader(cycles)) == ("OPEN", 2)


# ---------------------------------------------------------------------------
# DRY vs LIVE surfacing (Theme 1)
# ---------------------------------------------------------------------------


class _Col:
    def metric(self, *a, **k):
        pass


def _stub_st(monkeypatch, capture):
    """Stub the Streamlit surface used by the Status render helpers, routing
    each call into ``capture`` (a dict of lists) so tests can assert what the
    operator would see."""
    import trade_lab.monitoring.app as app

    for name in ("subheader", "info", "caption", "warning", "error",
                 "success", "dataframe"):
        capture.setdefault(name, [])
        monkeypatch.setattr(
            app.st, name,
            lambda *a, _n=name, **k: capture[_n].append(a[0] if a else None),
        )
    monkeypatch.setattr(app.st, "columns", lambda n: [_Col() for _ in range(n)])
    return app


def test_cycle_mode_live_vs_dry():
    from trade_lab.monitoring.app import _cycle_mode

    # Pre-marker rows (no context.mode): placed-orders fallback.
    assert _cycle_mode({"orders_executed": []}) == "LIVE"
    assert _cycle_mode({"orders_executed": [{"symbol": "BTC"}]}) == "LIVE"
    assert _cycle_mode({"orders_executed": None}) == "DRY"
    assert _cycle_mode({}) == "DRY"
    assert _cycle_mode(None) == "DRY"


def test_cycle_mode_failed_live_attempt_is_live():
    """A live cycle that raised before placing (orders_executed=None) must
    still be labeled LIVE via the durable context.mode marker."""
    from trade_lab.monitoring.app import _cycle_mode

    assert _cycle_mode({"context": {"mode": "live"},
                        "orders_executed": None}) == "LIVE"
    assert _cycle_mode({"context": {"mode": "dry_run"},
                        "orders_executed": None}) == "DRY"


class _LiveReader:
    def __init__(self, live=None, cycles=None):
        self._live = live
        self._cycles = cycles or []

    def latest_live_cycle(self):
        return self._live

    def latest_cycle(self):
        # Mirror the real JournalReader: newest cached cycle, or None.
        return self._cycles[-1] if self._cycles else None

    def cycles(self, n=20):
        return self._cycles[-n:]


def test_live_cron_health_info_when_no_live_cycle(monkeypatch):
    app = _stub_st(monkeypatch, cap := {})
    app._render_live_cron_health(_LiveReader(live=None))
    assert cap["info"]                       # info shown
    assert not cap["error"]                  # nothing overdue when none exists


def test_live_cron_health_errors_when_overdue(monkeypatch):
    from datetime import timedelta

    app = _stub_st(monkeypatch, cap := {})
    old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    app._render_live_cron_health(
        _LiveReader(live={"ended_at": old, "outcome": "success",
                          "cycle_id": "abcdef12"})
    )
    assert cap["error"]                      # overdue → loud error
    assert "OVERDUE" in cap["error"][0]


def test_live_cron_health_no_error_when_fresh(monkeypatch):
    from datetime import timedelta

    app = _stub_st(monkeypatch, cap := {})
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    app._render_live_cron_health(
        _LiveReader(live={"ended_at": recent, "outcome": "success",
                          "cycle_id": "abcdef12"})
    )
    assert not cap["error"]


def test_incidents_success_when_clean(monkeypatch):
    app = _stub_st(monkeypatch, cap := {})
    clean = [{"outcome": "success", "cycle_id": "ok", "ended_at": None,
              "orders_executed": []}]
    app._render_incidents(_LiveReader(cycles=clean))
    assert cap["success"]
    assert not cap["warning"] and not cap["error"]


def test_incidents_warns_on_failed_cycle(monkeypatch):
    app = _stub_st(monkeypatch, cap := {})
    cycles = [
        {"outcome": "success", "cycle_id": "ok", "ended_at": None,
         "orders_executed": []},
        {"outcome": "failed", "cycle_id": "boom", "ended_at": None,
         "orders_executed": None, "error": {"type": "CCXTError", "message": "x"}},
    ]
    app._render_incidents(_LiveReader(cycles=cycles))
    assert cap["warning"]                    # non-success cycle surfaced
    assert not cap["success"]
    # The table rides inside the alert body, not as a sibling dataframe.
    assert "| FAILED |" in cap["warning"][0]
    assert not cap["dataframe"]


def test_md_cell_escapes_pipes_and_newlines_in_journal_text():
    """A ccxt error carrying a pipe or newline must not split the row into
    extra columns — journal text is external input."""
    from trade_lab.monitoring.app import _md_cell

    assert _md_cell("a|b") == "a\\|b"
    assert _md_cell("line1\nline2") == "line1 line2"
    assert _md_cell(None) == ""
    assert _md_cell(r"back\slash") == "back\\\\slash"


def test_md_table_shape():
    from trade_lab.monitoring.app import _md_table

    out = _md_table(["a", "b"], [[1, 2], [3, 4]]).splitlines()
    assert out == ["| a | b |", "| --- | --- |", "| 1 | 2 |", "| 3 | 4 |"]


# ---------------------------------------------------------------------------
# Fail-loud robustness (Theme 3)
# ---------------------------------------------------------------------------


def test_series_return_tolerates_null_and_garbage_elements():
    from trade_lab.monitoring.app import _series_return

    assert _series_return([100.0, None], 1) == "—"        # null element
    assert _series_return([100.0, "x"], 1) == "—"         # garbage element
    assert _series_return([100.0, 110.0], 1) == "+10.00%"  # still works


def test_return_chip_colors_follow_direction():
    from trade_lab.monitoring.app import _return_chip

    assert _return_chip([100.0, 110.0], 1) == \
        ":green-badge[:material/arrow_upward: +10.00% vs 1d ago]"
    assert _return_chip([100.0, 90.0], 1) == \
        ":red-badge[:material/arrow_downward: -10.00% vs 1d ago]"


def test_return_chip_zero_and_missing_are_neutral_gray():
    """Flat and no-data chips stay gray and arrowless — mirrors the Ladder
    delta, which switches delta_color to 'off' when the change is zero."""
    from trade_lab.monitoring.app import _return_chip

    assert _return_chip([100.0, 100.0], 1) == ":gray-badge[+0.00% vs 1d ago]"
    assert _return_chip([], 7) == ":gray-badge[— vs 7d ago]"


def test_render_read_stats_warns_on_corrupt_and_errors_on_read_error(monkeypatch):
    import trade_lab.monitoring.app as app
    from trade_lab.monitoring.data_source import ReadStats

    cap = {}
    _stub_st(monkeypatch, cap)

    app._render_read_stats(ReadStats(total_lines=10, valid_cycles=8,
                                     corrupt_lines=2),
                           "data/journal/cycles.jsonl")
    assert cap["warning"]                       # corrupt → warning, not caption

    cap2 = {}
    _stub_st(monkeypatch, cap2)
    app._render_read_stats(ReadStats(read_error="PermissionError: denied"),
                           "data/journal/cycles.jsonl")
    assert cap2["error"]                        # unreadable journal → loud error


def test_render_read_stats_silent_when_clean(monkeypatch):
    import trade_lab.monitoring.app as app
    from trade_lab.monitoring.data_source import ReadStats

    cap = {}
    _stub_st(monkeypatch, cap)
    app._render_read_stats(ReadStats(total_lines=5, valid_cycles=5),
                           "data/journal/cycles.jsonl")
    assert not cap["warning"] and not cap["error"] and not cap["caption"]


# ---------------------------------------------------------------------------
# At-a-glance UX: health verdict, ladder day-delta, commit span (Theme 4)
# ---------------------------------------------------------------------------


class _HealthReader:
    def __init__(self, *, stats, staleness, cycles=None, live=None):
        self._stats = stats
        self._staleness = staleness
        self._cycles = cycles or []
        self._live = live

    def stats(self):
        return self._stats

    def staleness(self, s):
        return self._staleness

    def cycles(self, n=20):
        return self._cycles[-n:]

    def latest_live_cycle(self):
        return self._live


def _mk_stats(**kw):
    from trade_lab.monitoring.data_source import ReadStats
    return ReadStats(**kw)


def test_health_verdict_healthy():
    from datetime import timedelta
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    fresh_live = {"ended_at": (datetime.now(timezone.utc)
                               - timedelta(hours=2)).isoformat(),
                  "orders_executed": []}
    reader = _HealthReader(
        stats=_mk_stats(valid_cycles=5), staleness=Staleness.FRESH,
        cycles=[{"outcome": "success", "orders_executed": []}], live=fresh_live,
    )
    level, _why = _health_verdict(reader)
    assert level == "HEALTHY"


def test_health_verdict_healthy_without_live_cycle_says_so():
    """A dry-run-only journal (mainnet observation phase) has no live
    cycle — the HEALTHY verdict must not claim 'last live cycle OK'."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    reader = _HealthReader(
        stats=_mk_stats(valid_cycles=5), staleness=Staleness.FRESH,
        cycles=[{"outcome": "success", "orders_executed": None}], live=None,
    )
    level, why = _health_verdict(reader)
    assert level == "HEALTHY"
    assert "no live cycle yet" in why
    assert "live cycle OK" not in why


def _write_journal_rows(path, rows):
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_health_verdict_failed_live_attempt_not_no_live_cycle_yet(tmp_path):
    """A live cycle that raised before placing (orders_executed=None,
    context.mode='live') aged past the incident cut must NOT read as
    'HEALTHY — no live cycle yet': the live cron fired and is now overdue."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import JournalReader

    journal = tmp_path / "j.jsonl"
    _write_journal_rows(journal, [
        {"schema_version": 2, "cycle_id": "boom",
         "ended_at": _iso_ago(days=3), "outcome": "failed",
         "context": {"mode": "live"}, "orders_executed": None,
         "error": {"type": "RuntimeError", "message": "kaput"}},
        {"schema_version": 2, "cycle_id": "dry",
         "ended_at": _iso_ago(hours=1), "outcome": "success",
         "context": {"mode": "dry_run"}, "orders_executed": None},
    ])
    level, why = _health_verdict(JournalReader(journal))
    assert level == "DOWN"
    assert "live order cron overdue" in why
    assert "no live cycle yet" not in why


def test_health_verdict_failed_dry_cycle_still_no_live_cycle_yet(tmp_path):
    """The dry twin of the case above must keep the old reading."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import JournalReader

    journal = tmp_path / "j.jsonl"
    _write_journal_rows(journal, [
        {"schema_version": 2, "cycle_id": "dryboom",
         "ended_at": _iso_ago(days=3), "outcome": "failed",
         "context": {"mode": "dry_run"}, "orders_executed": None,
         "error": {"type": "RuntimeError", "message": "kaput"}},
        {"schema_version": 2, "cycle_id": "dry",
         "ended_at": _iso_ago(hours=1), "outcome": "success",
         "context": {"mode": "dry_run"}, "orders_executed": None},
    ])
    level, why = _health_verdict(JournalReader(journal))
    assert level == "HEALTHY"
    assert "no live cycle yet" in why


def test_health_verdict_down_on_read_error():
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    reader = _HealthReader(stats=_mk_stats(read_error="PermissionError: x"),
                           staleness=Staleness.NO_DATA)
    level, why = _health_verdict(reader)
    assert level == "DOWN"
    assert "unreadable" in why


def test_health_verdict_down_on_unresolved_order():
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    cycles = [{"outcome": "success",
               "orders_executed": [{"terminal_status": "lost_track",
                                    "client_order_id": "x"}]}]
    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.FRESH, cycles=cycles,
                           live=cycles[0])
    level, why = _health_verdict(reader)
    assert level == "DOWN"
    assert "unresolved" in why


def test_health_verdict_degraded_on_stale():
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.STALE,
                           cycles=[{"outcome": "success",
                                    "orders_executed": []}])
    level, _why = _health_verdict(reader)
    assert level == "DEGRADED"


def _iso_ago(**kw):
    return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat()


def test_health_verdict_healthy_after_timeout_resolved():
    """The sticking-banner regression: a timeout resolved by a later
    reconstruction row (same coid) must not keep the verdict DOWN for
    the rest of the 500-cycle window."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    live = {"outcome": "success", "ended_at": _iso_ago(hours=2),
            "orders_executed": []}
    cycles = [
        {"outcome": "unknown_orders", "ended_at": _iso_ago(days=5),
         "orders_executed": [{"terminal_status": "timeout",
                              "client_order_id": "coid-1"}]},
        {"outcome": "reconstructed", "ended_at": _iso_ago(days=4),
         "orders_executed": [{"terminal_status": "closed",
                              "client_order_id": "coid-1"}]},
        live,
    ]
    reader = _HealthReader(stats=_mk_stats(valid_cycles=3),
                           staleness=Staleness.FRESH, cycles=cycles,
                           live=live)
    level, why = _health_verdict(reader)
    assert level == "HEALTHY"
    assert "unresolved" not in why


def test_health_verdict_old_unresolved_order_still_down():
    """A truly-open order gets NO age-cut — DOWN until resolved."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    cycles = [{"outcome": "unknown_orders", "ended_at": _iso_ago(days=30),
               "orders_executed": [{"terminal_status": "timeout",
                                    "client_order_id": "coid-1"}]}]
    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.FRESH, cycles=cycles)
    level, why = _health_verdict(reader)
    assert level == "DOWN"
    assert "unresolved" in why


def test_health_verdict_old_failed_cycle_is_history_not_degraded():
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    cycles = [{"outcome": "failed", "ended_at": _iso_ago(days=10),
               "orders_executed": None},
              {"outcome": "success", "ended_at": _iso_ago(hours=2),
               "orders_executed": None}]
    reader = _HealthReader(stats=_mk_stats(valid_cycles=2),
                           staleness=Staleness.FRESH, cycles=cycles)
    level, _why = _health_verdict(reader)
    assert level == "HEALTHY"


def test_health_verdict_fresh_failed_cycle_is_degraded():
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    cycles = [{"outcome": "failed", "ended_at": _iso_ago(hours=3),
               "orders_executed": None}]
    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.FRESH, cycles=cycles)
    level, why = _health_verdict(reader)
    assert level == "DEGRADED"
    assert "recent incident" in why


def test_health_verdict_partial_fill_is_degraded_not_down():
    """An exchange-terminal partial (terminal_at set) is persisted as closed
    in order-state, so no later row can resolve it — it must not pin DOWN;
    the partial cycle outcome still degrades while recent."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    live = {"outcome": "partial", "ended_at": _iso_ago(hours=2),
            "orders_executed": [{"terminal_status": "partial",
                                 "client_order_id": "coid-1",
                                 "terminal_at": _iso_ago(hours=2)}]}
    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.FRESH, cycles=[live],
                           live=live)
    level, why = _health_verdict(reader)
    assert level == "DEGRADED"
    assert "unresolved" not in why


def test_health_verdict_timeout_partial_is_down():
    """A partial from a wait-for-ack timeout (terminal_at None) is still
    live on the exchange — it must hold DOWN until a later row closes it."""
    from trade_lab.monitoring.app import _health_verdict
    from trade_lab.monitoring.data_source import Staleness

    live = {"outcome": "partial", "ended_at": _iso_ago(hours=2),
            "orders_executed": [{"terminal_status": "partial",
                                 "client_order_id": "coid-1",
                                 "terminal_at": None}]}
    reader = _HealthReader(stats=_mk_stats(valid_cycles=1),
                           staleness=Staleness.FRESH, cycles=[live],
                           live=live)
    level, why = _health_verdict(reader)
    assert level == "DOWN"
    assert "unresolved" in why


def test_incident_is_recent_within_and_beyond_cut():
    from trade_lab.monitoring.app import (
        EXPECTED_INTERVAL_S, GAP_RECENT_MULTIPLIER, _incident_is_recent,
    )

    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    cut = EXPECTED_INTERVAL_S * GAP_RECENT_MULTIPLIER
    fresh = {"ended_at": (now - timedelta(seconds=cut / 2)).isoformat()}
    old = {"ended_at": (now - timedelta(seconds=cut * 2)).isoformat()}
    assert _incident_is_recent(fresh, now=now) is True
    assert _incident_is_recent(old, now=now) is False


def test_incident_is_recent_unparseable_timestamp_counts_as_recent():
    """No timestamp cannot prove age — fail loud, keep it in the verdict."""
    from trade_lab.monitoring.app import _incident_is_recent

    assert _incident_is_recent({"ended_at": None}) is True
    assert _incident_is_recent({"ended_at": "garbage"}) is True


def test_ladder_prev_day_delta_dedups_by_day():
    from trade_lab.monitoring.app import _ladder_prev_day_delta

    cycles = []
    # Day 1: 24 hourly cycles all ladder 1.0
    for h in range(24):
        cycles.append({"signal": {"asof": f"2026-06-10T{h:02d}:00:00+00:00",
                                   "ladder_value": 1.0}})
    # Day 2: flips to 0.5 (last of day)
    for h in range(24):
        cycles.append({"signal": {"asof": f"2026-06-11T{h:02d}:00:00+00:00",
                                   "ladder_value": 0.5}})
    # 1.0 → 0.5 across the day boundary; intraday repeats do not count.
    assert _ladder_prev_day_delta(_LiveReader(cycles=cycles)) == -0.5


def test_ladder_prev_day_delta_none_with_one_day():
    from trade_lab.monitoring.app import _ladder_prev_day_delta

    cycles = [{"signal": {"asof": "2026-06-10T00:00:00+00:00",
                          "ladder_value": 1.0}}]
    assert _ladder_prev_day_delta(_LiveReader(cycles=cycles)) is None


def test_distinct_commits_first_seen_order():
    from trade_lab.monitoring.app import _distinct_commits

    cycles = [{"git_commit": "aaa"}, {"git_commit": "aaa"},
              {"git_commit": "bbb"}, {"git_commit": None}]
    assert _distinct_commits(cycles) == ["aaa", "bbb"]


def test_commit_link_builds_clickable_markdown():
    import trade_lab.monitoring.app as app

    link = app._commit_link("abc1234")
    # markdown link to the commit source (st.warning renders markdown)
    assert link == f"[`abc1234`]({app.REPO_URL}/commit/abc1234)"
    assert "/commit/abc1234" in link


def test_commit_link_plain_span_when_repo_url_disabled(monkeypatch):
    import trade_lab.monitoring.app as app

    monkeypatch.setattr(app, "REPO_URL", "")
    assert app._commit_link("abc1234") == "`abc1234`"


# ---------------------------------------------------------------------------
# Footer — project + author links (env-configurable, hidden when URL empty)
# ---------------------------------------------------------------------------


def _capture_footer(monkeypatch):
    import trade_lab.monitoring.app as app
    md: list[str] = []
    monkeypatch.setattr(
        app.st, "markdown",
        lambda html, unsafe_allow_html=False: md.append(html),
    )
    app._render_footer()
    assert len(md) == 1
    return app, md[0]


def test_footer_hides_linkedin_when_url_empty(monkeypatch):
    import trade_lab.monitoring.app as app
    monkeypatch.setattr(app, "LINKEDIN_URL", "")   # force empty → hidden
    app, html = _capture_footer(monkeypatch)
    assert app.REPO_URL in html and "GitHub" in html
    assert app.TELEGRAM_URL in html and "Telegram" in html
    assert "LinkedIn" not in html
    assert app.AUTHOR_NAME in html


def test_footer_shows_all_three_links_with_defaults(monkeypatch):
    """Defaults now include a LinkedIn URL, so all three render."""
    _app, html = _capture_footer(monkeypatch)
    assert "GitHub" in html and "Telegram" in html and "LinkedIn" in html
    assert "linkedin.com/in/gistrec" in html


def test_footer_shows_linkedin_when_url_set(monkeypatch):
    import trade_lab.monitoring.app as app
    monkeypatch.setattr(app, "LINKEDIN_URL", "https://linkedin.com/in/example")
    _app, html = _capture_footer(monkeypatch)
    assert "linkedin.com/in/example" in html and "LinkedIn" in html


def test_footer_hides_a_link_when_its_url_is_empty(monkeypatch):
    import trade_lab.monitoring.app as app
    monkeypatch.setattr(app, "TELEGRAM_URL", "")
    monkeypatch.setattr(app, "LINKEDIN_URL", "")
    _app, html = _capture_footer(monkeypatch)
    assert "Telegram" not in html and "LinkedIn" not in html
    assert "GitHub" in html                         # REPO_URL still set


def test_banner_not_sticky_to_avoid_health_line_overlap(monkeypatch):
    """The banner is deliberately NOT position:sticky: a sticky banner with a
    non-sticky health line below it makes the health line slide UNDER the
    banner on scroll, and the two same-green plaques read as one merged block
    (user-reported regression). Kept as a plain top-of-page element."""
    html = _captured_banner(
        monkeypatch, {"context": {"sandbox": False, "exchange": "kraken"}}
    )
    assert "MAINNET" in html
    assert "position:sticky" not in html


# ---------------------------------------------------------------------------
# Validation-tab caching: the file signature is the cache key, so it must
# change exactly when the underlying data changes (else a breach is masked).
# ---------------------------------------------------------------------------


def test_file_sig_changes_when_file_changes(tmp_path):
    import time
    from trade_lab.monitoring.app import _file_sig

    p = tmp_path / "journal.jsonl"
    p.write_text("row1\n")
    s1 = _file_sig(p)
    assert s1 is not None
    time.sleep(0.01)
    p.write_text("row1\nrow2 with more bytes\n")   # size changes → key changes
    assert _file_sig(p) != s1


def test_file_sig_none_for_missing():
    from pathlib import Path
    from trade_lab.monitoring.app import _file_sig

    assert _file_sig(Path("/no/such/file.jsonl")) is None


def test_dir_sig_changes_when_vintage_added(tmp_path):
    """Vintages live under a two-level h[:2]/<hash>.txt layout; _dir_sig must
    recurse and count the REAL .txt format (regression: it globbed .parquet,
    which never matches, making the look-ahead cache key a dead constant)."""
    from trade_lab.monitoring.app import _dir_sig

    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / ("a" * 64 + ".txt")).write_text("vintage-1")
    s1 = _dir_sig(tmp_path)
    assert s1[0] == 1                               # nested .txt IS counted
    (tmp_path / "cd").mkdir()
    (tmp_path / "cd" / ("c" * 64 + ".txt")).write_text("vintage-2")
    s2 = _dir_sig(tmp_path)
    assert s2[0] == 2 and s2 != s1


def test_dir_sig_matches_real_vintage_store_format(tmp_path):
    """Couple the signature to the store's actual path builder, so a future
    change to the vintage serialization format re-breaks this loudly instead
    of silently zeroing the cache key again."""
    from trade_lab.monitoring.app import _dir_sig
    from trade_lab.paper_trading.vintage_store import vintage_path

    p = vintage_path(tmp_path, "b" * 64)            # tmp/bb/<hash>.txt
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("real-format vintage bytes")
    assert _dir_sig(tmp_path)[0] == 1               # picked up the real format


def test_dir_sig_empty_for_missing_root(tmp_path):
    from trade_lab.monitoring.app import _dir_sig

    assert _dir_sig(tmp_path / "absent") == (0, 0.0, 0)


# ---------------------------------------------------------------------------
# Ladder chart: gate-closed shading (Theme 2)
# ---------------------------------------------------------------------------


def test_gate_closed_spans_finds_closed_runs():
    from trade_lab.monitoring.app import _gate_closed_spans

    def t(day):
        return datetime(2026, 6, day, tzinfo=timezone.utc)

    history = [
        (t(1), 1.0, True),    # open
        (t(2), 0.0, False),   # closed run starts
        (t(3), 0.0, False),
        (t(4), 1.0, True),    # opens again → run [t2, t3]
        (t(5), 0.0, False),   # trailing closed run to the end
    ]
    spans = _gate_closed_spans(history)
    assert spans == [(t(2), t(3)), (t(5), t(5))]


def test_gate_closed_spans_all_open_is_empty():
    from trade_lab.monitoring.app import _gate_closed_spans

    history = [
        (datetime(2026, 6, d, tzinfo=timezone.utc), 1.0, True)
        for d in range(1, 4)
    ]
    assert _gate_closed_spans(history) == []


# ---------------------------------------------------------------------------
# Safety banner — fail loud on missing/garbage sandbox flag
# ---------------------------------------------------------------------------


def _captured_banner(monkeypatch, latest):
    import trade_lab.monitoring.app as app

    rendered: list[str] = []
    monkeypatch.setattr(
        app.st, "markdown", lambda html, unsafe_allow_html=False: rendered.append(html)
    )
    app._render_top_banner(latest)
    assert len(rendered) == 1
    return rendered[0]


def test_banner_green_only_on_explicit_sandbox_true(monkeypatch):
    html = _captured_banner(
        monkeypatch, {"context": {"sandbox": True, "exchange": "binance"}}
    )
    assert "TESTNET" in html


def test_banner_red_on_mainnet(monkeypatch):
    html = _captured_banner(
        monkeypatch, {"context": {"sandbox": False, "exchange": "kraken"}}
    )
    assert "MAINNET" in html and "REAL MONEY" in html


def test_banner_unknown_when_sandbox_missing(monkeypatch):
    """A cycle whose context lacks the flag must NOT look safe."""
    html = _captured_banner(monkeypatch, {"context": {"exchange": "binance"}})
    assert "UNKNOWN" in html
    assert "TESTNET" not in html


def test_banner_unknown_on_non_bool_garbage(monkeypatch):
    """bool('false') is True — a string flag must not render green."""
    html = _captured_banner(
        monkeypatch, {"context": {"sandbox": "false", "exchange": "binance"}}
    )
    assert "UNKNOWN" in html
    assert "TESTNET" not in html


def test_banner_unknown_on_non_dict_context(monkeypatch):
    """A truthy non-dict context (schema drift / corrupt row) must degrade
    to the UNKNOWN banner, not raise AttributeError — the banner is the
    ONE renderer outside tab-safety, so a crash blanks the whole page
    (regression: R6)."""
    html = _captured_banner(monkeypatch, {"context": "binance-sandbox"})
    assert "UNKNOWN" in html
    assert "TESTNET" not in html


def test_cycle_context_coerces_non_dict_to_empty():
    """_cycle_context returns {} for a missing/non-dict context so callers
    can .get() safely."""
    from trade_lab.monitoring.app import _cycle_context

    assert _cycle_context({"context": {"quote_currency": "USDT"}}) == {
        "quote_currency": "USDT"
    }
    assert _cycle_context({"context": "corrupt"}) == {}
    assert _cycle_context({"context": None}) == {}
    assert _cycle_context({}) == {}
    assert _cycle_context(None) == {}


def test_render_portfolio_survives_non_dict_context(monkeypatch):
    """The Portfolio tab reads latest["context"] too. A truthy non-dict
    context must NOT crash it — `(latest.get("context") or {}).get(...)`
    raised AttributeError on a string context, the same class of bug the
    banner fix (R6) addressed but only in the banner (verify finding)."""
    import trade_lab.monitoring.app as app

    class _Col:
        def metric(self, *a, **k):
            pass

    monkeypatch.setattr(app.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "columns", lambda n: [_Col() for _ in range(n)])
    monkeypatch.setattr(app.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "plotly_chart", lambda *a, **k: None)

    cycle = {
        "outcome": "success",
        "context": "corrupt-non-dict-context",   # truthy non-dict
        "target_allocation": {"BTC": 7500.0},
        "current_holdings_quote": {"BTC": 5000.0},
        "equity_usd": 15000.0,
        "orders_planned": [],
        "orders_executed": [],
    }

    class _Reader:
        def latest_cycle(self):
            return cycle

        def cumulative_skipped_drift(self, live_only=True):
            return 0.0

        def live_cycle_count(self):
            return 0

        def latest_skipped_drift(self):
            return 0.0

        def cycles(self, n=20):
            return [cycle]

    app._render_portfolio(_Reader())   # must not raise


# ---------------------------------------------------------------------------
# _sma_warmup_stall — SMA(200) can-never-warm-up detector
# ---------------------------------------------------------------------------


def _stall_reader(*, sma_value, sandbox=True, exchange="binance",
                  bars=36, start_ts="2026-06-03T00:00:00+00:00"):
    """Minimal reader whose latest cycle carries a signal + context that
    exercise the _sma_warmup_stall branches."""
    cycle = {
        "context": {"exchange": exchange, "sandbox": sandbox},
        "signal": {"sma_value": sma_value, "sma_gate_open": False},
        "basket_close_series": {
            "values": [1.0] * bars,
            "start_ts": start_ts,
        },
    }

    class _R:
        def latest_cycle(self):
            return cycle

    return _R()


def test_sma_stall_fires_on_sandbox_with_no_sma():
    from trade_lab.monitoring.app import _sma_warmup_stall
    stall = _sma_warmup_stall(_stall_reader(sma_value=None, bars=36))
    assert stall is not None
    assert stall["exchange"] == "binance"
    assert stall["bars"] == 36
    assert stall["start_ts"] == "2026-06-03T00:00:00+00:00"


def test_sma_stall_silent_when_gate_warmed():
    """A present sma_value means the 200-bar window is warm — no banner."""
    from trade_lab.monitoring.app import _sma_warmup_stall
    assert _sma_warmup_stall(_stall_reader(sma_value=91.2)) is None


def test_sma_stall_silent_on_mainnet_source():
    """'Never warms up' is only honest where history is reset-capped. A
    full-history mainnet exchange would warm up given time — don't claim
    'never' there even while sma_value is still None early on."""
    from trade_lab.monitoring.app import _sma_warmup_stall
    assert _sma_warmup_stall(
        _stall_reader(sma_value=None, sandbox=False, exchange="kraken")
    ) is None


def test_sma_stall_silent_without_latest_or_signal():
    from trade_lab.monitoring.app import _sma_warmup_stall

    class _Empty:
        def latest_cycle(self):
            return None

    class _NoSignal:
        def latest_cycle(self):
            return {"context": {"sandbox": True}, "signal": None}

    assert _sma_warmup_stall(_Empty()) is None
    assert _sma_warmup_stall(_NoSignal()) is None


def _skipped_warmup_reader(*, bars=36, required=200, sandbox=True):
    """Reader whose latest cycle is the current first-class skip shape:
    outcome='skipped_warmup' + structured skip_reason (no signal at all —
    the executor refused before computing one)."""
    cycle = {
        "outcome": "skipped_warmup",
        "context": {"exchange": "binance", "sandbox": sandbox},
        "signal": None,
        "basket_close_series": None,
        "skip_reason": {
            "type": "insufficient_warmup",
            "bars_available": bars,
            "bars_required": required,
            "message": "Basket history too short to warm the signal",
        },
    }

    class _R:
        def latest_cycle(self):
            return cycle

    return _R()


def test_sma_stall_reads_skipped_warmup_reason():
    """The current journal shape: a skipped_warmup cycle's skip_reason
    (bars_available/bars_required from the executor itself) feeds the
    banner — not the legacy basket_close_series length."""
    from trade_lab.monitoring.app import _sma_warmup_stall

    stall = _sma_warmup_stall(_skipped_warmup_reader(bars=36, required=200))
    assert stall is not None
    assert stall["exchange"] == "binance"
    assert stall["bars"] == 36
    assert stall["bars_required"] == 200


def test_sma_stall_silent_for_skipped_warmup_on_mainnet_content():
    """Defense in depth: a skipped_warmup record whose own context says
    mainnet must NOT produce the 'by design' banner — that outcome is
    impossible on mainnet, and reassuring wording would mask the bug."""
    from trade_lab.monitoring.app import _sma_warmup_stall

    assert _sma_warmup_stall(_skipped_warmup_reader(sandbox=False)) is None


def test_render_warmup_notice_renders_yellow_banner_with_bar_counts(monkeypatch):
    app = _stub_st(monkeypatch, cap := {})
    app._render_warmup_notice(_skipped_warmup_reader(bars=36, required=200))
    assert len(cap["warning"]) == 1
    msg = cap["warning"][0]
    assert "SMA(200)" in msg
    assert "36 of 200" in msg
    assert "skipped_warmup" in msg
    assert "not an incident" in msg
    assert not cap["error"]


def test_render_warmup_notice_silent_when_no_stall(monkeypatch):
    app = _stub_st(monkeypatch, cap := {})
    app._render_warmup_notice(_stall_reader(sma_value=91.2))  # warmed gate
    assert not cap["warning"] and not cap["error"]


def test_render_incidents_no_longer_renders_stall_banner(monkeypatch):
    """The structural notice moved to the single page-bottom banner
    (_render_warmup_notice); the Incidents section renders only the
    operational verdict — one banner on the page, not duplicates."""
    import trade_lab.monitoring.app as app

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(app.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(
        app.st, "success", lambda msg, *a, **k: calls.append(("success", msg)))
    monkeypatch.setattr(
        app.st, "warning", lambda msg, *a, **k: calls.append(("warning", msg)))

    reader = _stall_reader(sma_value=None, bars=36)
    reader.cycles = lambda n=500: []   # clean window: no incidents at all
    app._render_incidents(reader)

    kinds = [k for k, _ in calls]
    assert kinds == ["success"]
    assert "No failed/partial cycles" in calls[0][1]


def test_incidents_clean_when_window_full_of_skipped_warmup(monkeypatch):
    """skipped_warmup cycles are healthy testnet states: a window holding
    nothing else must render the clean-window success line and no
    incident warning."""
    app = _stub_st(monkeypatch, cap := {})
    cycles = [
        {"outcome": "skipped_warmup", "cycle_id": f"s{i}", "ended_at": None,
         "orders_executed": [],
         "skip_reason": {"type": "insufficient_warmup",
                         "bars_available": 36, "bars_required": 200}}
        for i in range(3)
    ]
    app._render_incidents(_LiveReader(cycles=cycles))
    assert cap["success"]
    assert not cap["warning"] and not cap["error"]


# ---------------------------------------------------------------------------
# Testnet/mainnet journal sources (dashboard switcher)
# ---------------------------------------------------------------------------


def test_journal_sources_single_by_default(monkeypatch):
    """Without the mainnet env var the dashboard stays single-source —
    the switcher must not appear on existing deployments."""
    import importlib
    import trade_lab.monitoring.app as app

    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)
    try:
        reloaded = importlib.reload(app)
        assert list(reloaded.JOURNAL_SOURCES) == ["testnet"]
    finally:
        importlib.reload(app)


def test_journal_sources_with_mainnet_env(monkeypatch):
    import importlib
    import trade_lab.monitoring.app as app

    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       "data/journal/cycles_mainnet.jsonl")
    try:
        reloaded = importlib.reload(app)
        assert reloaded.JOURNAL_SOURCES == {
            "testnet": "data/journal/cycles.jsonl",
            "mainnet": "data/journal/cycles_mainnet.jsonl",
        }
    finally:
        monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                           raising=False)
        importlib.reload(app)


# ---------------------------------------------------------------------------
# Runtime verify (AppTest): a skipped_warmup journal renders the page-bottom
# warm-up banner and zero incident/alarm blocks
# ---------------------------------------------------------------------------


def test_app_renders_warmup_banner_not_incident_for_skipped_warmup(
    tmp_path, monkeypatch,
):
    """End-to-end through the real Streamlit script: a fresh testnet
    journal whose only cycle is outcome='skipped_warmup' must render as a
    HEALTHY dashboard — clean incident window, no failure blocks — with
    the single yellow structural notice (bars X of Y from skip_reason) at
    the bottom of the page."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc).isoformat()
    cycle = {
        "cycle_id": "skip1", "started_at": now, "ended_at": now,
        "duration_ms": 5000, "outcome": "skipped_warmup", "error": None,
        "git_commit": None, "python_version": "3.11.0",
        "context": {"mode": "live", "exchange": "binance", "sandbox": True,
                    "quote_currency": "USDT", "basket": ["BTC"]},
        "signal": None, "basket_close_series": None, "balance": None,
        "equity_usd": None, "target_allocation": None,
        "current_holdings_quote": None, "orders_planned": None,
        "orders_skipped": None, "total_skipped_quote_drift": None,
        "orders_executed": [],
        "skip_reason": {"type": "insufficient_warmup",
                        "bars_available": 36, "bars_required": 200,
                        "message": "too short"},
        "schema_version": 2,
    }
    journal = tmp_path / "cycles.jsonl"
    journal.write_text(json.dumps(cycle) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    warnings = [str(w.value) for w in app.warning]
    banner = [w for w in warnings if "SMA(200)" in w]
    assert len(banner) == 1, warnings          # one banner, no duplicates
    assert "36 of 200" in banner[0]
    assert "not an incident" in banner[0]
    # No incident surface anywhere: the skip is a healthy state. Scan titles
    # too — the incident headline lives in the alert title, not the body.
    titles = [str(w.proto.title) for w in app.warning]
    assert not any("non-success cycle" in t for t in warnings + titles), \
        warnings + titles
    errors = [str(e.value) for e in app.error]
    assert not any("FAILED" in e for e in errors), errors
    successes = [str(s.value) for s in app.success]
    assert any("No failed/partial cycles" in s for s in successes), successes


def test_incidents_section_orders_alerts_success_info_warning(
    tmp_path, monkeypatch,
):
    """The Incidents section reads top-down in ascending severity:
    green all-clear, then blue context notes, then yellow warnings.
    Fixture fires all three: healthy cycles (green), a latest dry-run
    with planned sub-min drift (blue), two distinct git commits in the
    window (yellow)."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc)

    def _cycle(cid, ended, commit, drift):
        iso = ended.isoformat()
        return {
            "cycle_id": cid, "started_at": iso, "ended_at": iso,
            "duration_ms": 5000, "outcome": "success", "error": None,
            "git_commit": commit, "python_version": "3.11.0",
            "context": {"mode": "dry_run", "exchange": "binance",
                        "sandbox": True, "quote_currency": "USDT",
                        "basket": ["BTC"]},
            "signal": None, "basket_close_series": None, "balance": None,
            "equity_usd": None, "target_allocation": None,
            "current_holdings_quote": None, "orders_planned": [],
            "orders_skipped": [], "total_skipped_quote_drift": drift,
            # None = dry-run (is_live_cycle keys on orders_executed).
            "orders_executed": None, "schema_version": 2,
        }

    journal = tmp_path / "cycles.jsonl"
    journal.write_text("\n".join(json.dumps(c) for c in [
        _cycle("c1", now - timedelta(hours=6), "aaaa111", 0.0),
        _cycle("c2", now, "bbbb222", 1.16),
    ]) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    from streamlit.testing.v1 import element_tree as et

    def _walk(node):
        children = getattr(node, "children", None)
        if children:
            for child in children.values():
                yield from _walk(child)
        elif isinstance(node, et.AlertBase):
            yield node

    markers = {
        "No failed/partial cycles": "success",
        "could not place": "info",
        "Observation window spans": "warning",
    }
    seq = []
    for el in _walk(app.main):
        text = str(el.value)
        for marker, kind in markers.items():
            if marker in text:
                seq.append(kind)
    assert seq == ["success", "info", "warning"], (
        f"Incidents must read green -> blue -> yellow, got {seq}"
    )


# ---------------------------------------------------------------------------
# Recent-cycles collapse
# ---------------------------------------------------------------------------


def _sig_row(asof: str, ladder: float = 1.0, gate: str = "OPEN") -> dict:
    """One Recent-cycles table row, in the shape _render_signal builds."""
    return {"asof": asof, "basket": 54.03, "ladder": ladder,
            "gate": gate, "28d": 1, "60d": 1}


def test_collapse_merges_repeated_observations_of_one_bar():
    """Four 6-hourly dry-runs over one closed bar are one signal day."""
    rows = [_sig_row("2026-08-21")] * 4
    out = _collapse_signal_rows(rows)
    assert len(out) == 1
    assert out[0]["cycles"] == 4
    assert out[0]["asof"] == "2026-08-21"


def test_collapse_keeps_distinct_bars_separate():
    rows = [_sig_row("2026-08-21")] * 5 + [_sig_row("2026-08-20")] * 2
    out = _collapse_signal_rows(rows)
    assert [(r["asof"], r["cycles"]) for r in out] == [
        ("2026-08-21", 5), ("2026-08-20", 2),
    ]


def test_collapse_never_hides_disagreement_within_a_bar():
    """Same asof, different ladder ⇒ the signal is not deterministic over a
    closed bar. Merging would erase exactly the anomaly worth seeing."""
    rows = [
        _sig_row("2026-08-21", ladder=1.0),
        _sig_row("2026-08-21", ladder=0.5),
        _sig_row("2026-08-21", ladder=1.0),
    ]
    out = _collapse_signal_rows(rows)
    assert len(out) == 3
    assert [r["ladder"] for r in out] == [1.0, 0.5, 1.0]
    assert all(r["cycles"] == 1 for r in out)


def test_collapse_disagreement_on_gate_also_splits():
    rows = [
        _sig_row("2026-08-21", gate="OPEN"),
        _sig_row("2026-08-21", gate="CLOSED"),
    ]
    out = _collapse_signal_rows(rows)
    assert len(out) == 2


def test_collapse_empty_input():
    assert _collapse_signal_rows([]) == []


def test_collapse_does_not_mutate_input_rows():
    rows = [_sig_row("2026-08-21")] * 2
    original = [dict(r) for r in rows]
    _collapse_signal_rows(rows)
    assert rows == original          # no 'cycles' key leaked into the source


# ---------------------------------------------------------------------------
# Source ordering
# ---------------------------------------------------------------------------


def test_mainnet_is_the_first_and_default_source(monkeypatch):
    """Insertion order drives both the switcher layout and the landing
    source (`next(iter(...))`). Real money must be the page you land on."""
    import importlib
    import trade_lab.monitoring.app as app

    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       "data/journal/cycles_mainnet.jsonl")
    try:
        reloaded = importlib.reload(app)
        assert list(reloaded.JOURNAL_SOURCES) == ["mainnet", "testnet"]
        assert next(iter(reloaded.JOURNAL_SOURCES)) == "mainnet"
    finally:
        monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                           raising=False)
        importlib.reload(app)


# ---------------------------------------------------------------------------
# Cadence-gap ageing — a recovered gap must stop shouting
# ---------------------------------------------------------------------------


def test_gap_recent_when_cadence_just_resumed():
    """A long pause that ended an hour ago is still fresh news."""
    from trade_lab.monitoring.app import _gap_is_recent
    from trade_lab.monitoring.data_source import CadenceGap

    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    gap = CadenceGap(
        seconds=15 * 3600,
        started=now - timedelta(hours=16),
        ended=now - timedelta(hours=1),
    )
    assert _gap_is_recent(gap, now=now) is True


def test_gap_stale_after_the_attention_window():
    """The regression: a 15h gap from 19 Aug sat in the 500-cycle window
    as a warning for months after the cron had recovered."""
    from trade_lab.monitoring.app import _gap_is_recent
    from trade_lab.monitoring.data_source import CadenceGap

    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    gap = CadenceGap(
        seconds=15 * 3600,
        started=datetime(2026, 8, 19, 0, 35, tzinfo=timezone.utc),
        ended=datetime(2026, 8, 19, 15, 32, tzinfo=timezone.utc),
    )
    assert _gap_is_recent(gap, now=now) is False


def test_gap_age_is_measured_from_resumption_not_onset():
    """A pause that STARTED long ago but only ended just now is recent —
    measuring from onset would mute an incident that just closed."""
    from trade_lab.monitoring.app import _gap_is_recent
    from trade_lab.monitoring.data_source import CadenceGap

    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    gap = CadenceGap(
        seconds=30 * 86400,
        started=now - timedelta(days=30),
        ended=now - timedelta(minutes=10),
    )
    assert _gap_is_recent(gap, now=now) is True


# ---------------------------------------------------------------------------
# Sim-equity metric (Validation tab): the $ figure is a virtual portfolio,
# the delta pins it to the frozen config's initial capital.
# ---------------------------------------------------------------------------

def test_sim_equity_metric_gain():
    value, delta = _sim_equity_metric(10919.42, 10_000.0)
    assert value == "$10919.42"
    assert delta == "+9.19% since start"


def test_sim_equity_metric_loss_reads_negative():
    """Leading '-' is what makes st.metric render the delta red."""
    _, delta = _sim_equity_metric(9500.0, 10_000.0)
    assert delta == "-5.00% since start"


def test_sim_equity_metric_flat():
    _, delta = _sim_equity_metric(10_000.0, 10_000.0)
    assert delta == "+0.00% since start"


# ---------------------------------------------------------------------------
# Runtime verify (AppTest): Signal tab folds direction/persistence into the
# top metric row — 7d/30d chips under Basket close, days chip under the gate
# ---------------------------------------------------------------------------


def test_signal_tab_renders_chips_not_second_metric_row(tmp_path, monkeypatch):
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc)

    def _cycle(cid, ended, signal):
        iso = ended.isoformat()
        return {
            "cycle_id": cid, "started_at": iso, "ended_at": iso,
            "duration_ms": 5000, "outcome": "success", "error": None,
            "git_commit": "aaaa111", "python_version": "3.11.0",
            "context": {"mode": "dry_run", "exchange": "binance",
                        "sandbox": True, "quote_currency": "USDT",
                        "basket": ["BTC"]},
            "signal": signal, "basket_close_series": None, "balance": None,
            "equity_usd": None, "target_allocation": None,
            "current_holdings_quote": None, "orders_planned": [],
            "orders_skipped": [], "total_skipped_quote_drift": 0.0,
            "orders_executed": None, "schema_version": 2,
        }

    def _signal(asof):
        return {
            "asof": asof.isoformat(), "ladder_value": 1.0,
            "sma_gate_open": True, "basket_close": 110.0, "sma_value": 100.0,
            "per_lookback_states": {"28": 1, "60": 1},
            "per_lookback_returns": {"28": 0.05, "60": 0.10},
        }

    # values[-8] = 100 → 7d chip +10.00%; values[-31] = 88 → 30d chip +25.00%.
    series = [88.0] * 23 + [100.0] * 7 + [110.0]
    c1 = _cycle("c1", now - timedelta(days=1), _signal(now - timedelta(days=1)))
    c2 = _cycle("c2", now, _signal(now))
    c2["basket_close_series"] = {
        "values": series,
        "start_ts": (now - timedelta(days=len(series) - 1)).isoformat(),
    }
    journal = tmp_path / "cycles.jsonl"
    journal.write_text(json.dumps(c1) + "\n" + json.dumps(c2) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    metrics = {m.label: m for m in app.metric}
    gate = metrics["SMA(200) gate"]
    assert gate.value == "OPEN"
    assert gate.delta == "2d OPEN"        # both fixture dates carry OPEN
    # The old second row is gone — its readings live on the tiles now.
    for label in ("vs 7d ago", "vs 30d ago", "Days gate OPEN"):
        assert label not in metrics, sorted(metrics)

    chip_lines = [str(m.value) for m in app.markdown
                  if "-badge[" in str(m.value)]
    assert any(
        ":green-badge[:material/arrow_upward: +10.00% vs 7d ago]  \n"
        ":green-badge[:material/arrow_upward: +25.00% vs 30d ago]" == line
        for line in chip_lines
    ), chip_lines


# ---------------------------------------------------------------------------
# _config_gate_state — the dashboard must ask the question the harness asks
# ---------------------------------------------------------------------------


def test_config_gate_matches_when_runtime_equals_frozen_literal():
    from trade_lab.monitoring.app import _config_gate_state
    from trade_lab.paper_trading.harness import FROZEN_CONFIG_HASH

    will_run, frozen = _config_gate_state(FROZEN_CONFIG_HASH)
    assert will_run is True
    assert frozen == FROZEN_CONFIG_HASH


def test_config_gate_follows_the_harness_literal_not_canonical_hash(
    monkeypatch,
):
    """The discriminating test. Today CANONICAL_HASH == FROZEN_CONFIG_HASH,
    so no fixed pair of values can tell the two references apart — only
    moving one of them can. Repoint the harness literal: the harness
    would now refuse to run, so the tile must go red. A gate still
    comparing against CANONICAL_HASH (recomputed from the same config
    object, hence equal by construction) stays green and fails here."""
    import trade_lab.paper_trading.harness as harness
    from trade_lab.config import CANONICAL_HASH
    from trade_lab.monitoring.app import _config_gate_state

    monkeypatch.setattr(harness, "FROZEN_CONFIG_HASH", "0" * 64)
    will_run, frozen = _config_gate_state(CANONICAL_HASH)
    assert will_run is False
    assert frozen == "0" * 64


def test_config_gate_canonical_hash_is_the_tautology_it_avoids():
    """Documents WHY the reference had to change: CANONICAL_HASH tracks
    PRODUCTION_CONFIG exactly, so a config edit moves both sides of the
    old comparison together and the drift branch is unreachable."""
    from trade_lab.config import (
        CANONICAL_HASH, PRODUCTION_CONFIG, production_config_hash,
    )

    assert production_config_hash(PRODUCTION_CONFIG) == CANONICAL_HASH


# ---------------------------------------------------------------------------
# _gate_label — a missing gate reading is not a closed gate
# ---------------------------------------------------------------------------


def test_gate_label_three_valued():
    from trade_lab.monitoring.app import _gate_label

    assert _gate_label(True) == "OPEN"
    assert _gate_label(False) == "CLOSED"
    assert _gate_label(None) == "—"


def test_gate_label_treats_falsy_non_booleans_as_unknown():
    """Only JSON true/false are gate observations. A schema-drifted 0 /
    "" / [] is an unknown reading — a truthiness test would fabricate
    CLOSED (the flat-position state) for exactly the corrupt rows that
    deserve a dash."""
    from trade_lab.monitoring.app import _gate_label

    for drifted in (0, "", [], {}, 0.0, "false", 1, "true", ["x"]):
        assert _gate_label(drifted) == "—", drifted


def test_recent_cycles_row_does_not_fabricate_closed_for_missing_gate():
    """The Recent-cycles table used to render an absent sma_gate_open as
    CLOSED — a fabricated flat-position reading for a bar whose gate was
    never recorded."""
    from trade_lab.monitoring.app import _gate_label

    csig = {"asof": "2026-08-25", "ladder_value": 1.0}   # no sma_gate_open
    assert _gate_label(csig.get("sma_gate_open")) == "—"


# ---------------------------------------------------------------------------
# _humanize_* — total over external input, and honest about the UTC label
# ---------------------------------------------------------------------------


def test_humanizers_degrade_on_non_string_timestamp():
    """A schema-drifted journal field (dict, list, number) must not take
    the tab down with AttributeError on ``.replace``."""
    for bad in ({"ts": 1}, [1, 2], 1700000000, True):
        assert _humanize_iso(bad) == "—"
        assert _humanize_relative(bad, now=NOW) == "—"


def test_humanizers_show_unparseable_strings_verbatim():
    """A string that failed to parse is still the operator's best clue."""
    assert _humanize_iso("not-a-timestamp") == "not-a-timestamp"
    assert _humanize_relative("not-a-timestamp", now=NOW) == "not-a-timestamp"


def test_humanize_iso_converts_offset_to_utc_before_labeling_it_utc():
    """The label says UTC, so the digits must be UTC. A writer emitting a
    non-zero offset (the cron once ran on MSK) would otherwise get its
    local wall clock stamped 'UTC'."""
    assert _humanize_iso("2026-08-25T15:00:00+03:00") == \
        "2026-08-25 12:00:00 UTC"


def test_humanize_iso_naive_timestamp_treated_as_utc():
    assert _humanize_iso("2026-08-25T12:00:00") == "2026-08-25 12:00:00 UTC"


# ---------------------------------------------------------------------------
# _sma_warmup_stall — route context reads through _cycle_context
# ---------------------------------------------------------------------------


def test_sma_stall_survives_truthy_non_dict_context():
    """A corrupt/schema-drifted context (a string, a list) is truthy, so
    the direct ``.get`` raised AttributeError and killed the warm-up
    notice this detector exists to render."""
    from trade_lab.monitoring.app import _sma_warmup_stall

    for bad_ctx in ("mainnet", ["mainnet"], 7):
        class _R:
            def latest_cycle(self):
                return {
                    "context": bad_ctx,
                    "signal": {"sma_value": None, "sma_gate_open": False},
                    "basket_close_series": {"values": [1.0] * 36},
                }

        assert _sma_warmup_stall(_R()) is None


# ---------------------------------------------------------------------------
# Runtime verify (AppTest): the sub-min notice must not deny live trading
# ---------------------------------------------------------------------------


def test_submin_notice_does_not_claim_no_live_cron_when_live_cycles_exist(
    tmp_path, monkeypatch,
):
    """Journal with real live cycles that skipped nothing (cumulative
    drift 0) plus a latest plan carrying sub-min drift. The notice used
    to gate on the drift SUM, so it announced 'no live order cron has run
    yet' on a page whose own journal shows filled orders."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc)

    def _cycle(cid, ended, *, executed, drift):
        iso = ended.isoformat()
        return {
            "cycle_id": cid, "started_at": iso, "ended_at": iso,
            "duration_ms": 5000, "outcome": "success", "error": None,
            "git_commit": "aaaa111", "python_version": "3.11.0",
            "context": {"mode": "live", "exchange": "binance",
                        "sandbox": False, "quote_currency": "USDT",
                        "basket": ["BTC"]},
            "signal": None, "basket_close_series": None, "balance": None,
            "equity_usd": 147.0, "target_allocation": None,
            "current_holdings_quote": None, "orders_planned": [],
            "orders_skipped": [], "total_skipped_quote_drift": drift,
            "orders_executed": executed, "schema_version": 2,
        }

    filled = [{"symbol": "BTC/USDT", "side": "buy",
               "client_order_id": "tl-2026-08-24-BTC-buy",
               "terminal_status": "closed", "intended_amount": 1.0,
               "filled_amount": 1.0, "filled_notional_quote": 13.18}]
    journal = tmp_path / "cycles.jsonl"
    journal.write_text("\n".join(json.dumps(c) for c in [
        # A live cycle that placed and filled — skipped nothing.
        _cycle("live1", now - timedelta(hours=6), executed=filled, drift=0.0),
        # Latest is a dry-run plan whose deltas fall below min-notional.
        _cycle("plan1", now, executed=None, drift=1.16),
    ]) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    infos = [str(i.value) for i in app.info]
    submin = [i for i in infos if "below the exchange minimum notional" in i]
    assert len(submin) == 1, infos
    assert "no live order cron has run yet" not in submin[0], submin[0]
    assert "cumulative skipped drift across live cycles" in submin[0]


# ---------------------------------------------------------------------------
# _equity_prev_day_delta + the KPI row above the tabs
# ---------------------------------------------------------------------------


def _first_metric(app, label):
    """First metric carrying ``label``, in document order.

    The KPI row renders above the tabs, and the Portfolio tab reuses
    ``Equity ({quote})`` for the same number — a plain label→metric dict
    would silently return the Portfolio tile instead.
    """
    for m in app.metric:
        if m.label == label:
            return m
    raise AssertionError(
        f"no metric labelled {label!r}; saw {[m.label for m in app.metric]}"
    )


def _equity_reader(points):
    """Reader whose cycles carry (ended_at, equity) successful readings."""
    cycles = [
        {"outcome": "success", "ended_at": ts, "equity_usd": eq,
         "context": {"quote_currency": "USDT"}}
        for ts, eq in points
    ]

    class _R:
        def cycles(self, n=20):
            return cycles[-n:]

    return _R()


def test_equity_delta_none_without_two_distinct_days():
    from trade_lab.monitoring.app import _equity_prev_day_delta

    assert _equity_prev_day_delta(_equity_reader([])) == (None, None)
    one_day = _equity_reader([
        ("2026-08-25T06:00:00+00:00", 100.0),
        ("2026-08-25T18:00:00+00:00", 101.0),
    ])
    assert _equity_prev_day_delta(one_day) == (101.0, None)


def test_equity_delta_compares_last_of_day_not_consecutive_cycles():
    """Four dry-runs a day: an intraday repeat must not read as a change."""
    from trade_lab.monitoring.app import _equity_prev_day_delta

    reader = _equity_reader([
        ("2026-08-24T00:00:00+00:00", 100.0),
        ("2026-08-24T06:00:00+00:00", 101.0),
        ("2026-08-24T18:00:00+00:00", 102.0),   # last of 08-24
        ("2026-08-25T00:00:00+00:00", 105.0),
        ("2026-08-25T12:00:00+00:00", 110.0),   # last of 08-25
    ])
    latest, delta = _equity_prev_day_delta(reader)
    assert latest == 110.0
    assert delta == pytest.approx(8.0)          # 110 − 102, not 110 − 105


def test_kpi_row_renders_money_and_state_above_the_tabs(tmp_path, monkeypatch):
    """Runtime verify: the first screen carries equity, ladder, gate and
    last-LIVE without a tap. Also pins the honesty choice — the equity
    delta is NOT painted as performance, because deposits move it too."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc)

    def _cycle(cid, ended, *, equity, ladder, gate, executed):
        iso = ended.isoformat()
        return {
            "cycle_id": cid, "started_at": iso, "ended_at": iso,
            "duration_ms": 5000, "outcome": "success", "error": None,
            "git_commit": "aaaa111", "python_version": "3.11.0",
            "context": {"mode": "live", "exchange": "binance",
                        "sandbox": False, "quote_currency": "USDT",
                        "basket": ["BTC"]},
            "signal": {"asof": ended.date().isoformat(),
                       "ladder_value": ladder, "sma_gate_open": gate,
                       "basket_close": 100.0, "sma_value": 90.0},
            "basket_close_series": None, "balance": None,
            "equity_usd": equity, "target_allocation": None,
            "current_holdings_quote": None, "orders_planned": [],
            "orders_skipped": [], "total_skipped_quote_drift": 0.0,
            "orders_executed": executed, "schema_version": 2,
        }

    filled = [{"symbol": "BTC/USDT", "side": "buy",
               "client_order_id": "tl-2026-08-24-BTC-buy",
               "terminal_status": "closed", "intended_amount": 1.0,
               "filled_amount": 1.0, "filled_notional_quote": 13.18}]
    journal = tmp_path / "cycles.jsonl"
    journal.write_text("\n".join(json.dumps(c) for c in [
        _cycle("d1", now - timedelta(days=1), equity=100.0, ladder=1.0,
               gate=True, executed=filled),
        _cycle("d2", now, equity=110.5, ladder=1.0, gate=True,
               executed=filled),
    ]) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    equity = _first_metric(app, "Equity (USDT)")
    assert equity.value == "110.50"
    assert equity.delta == "+10.50 vs prior day"
    # The honesty pin: a deposit would move this the same way, so the tile
    # must not dress the change as profit. GRAY is what delta_color='off'
    # renders as; GREEN here would mean the dashboard called a top-up a gain.
    from streamlit.proto.Metric_pb2 import Metric as _MetricProto
    assert equity.proto.color == _MetricProto.MetricColor.GRAY, \
        _MetricProto.MetricColor.Name(equity.proto.color)
    assert _first_metric(app, "Ladder").value == "1.00"
    assert _first_metric(app, "SMA(200) gate").value == "OPEN"
    assert _first_metric(app, "Last LIVE").delta == "SUCCESS"


def test_kpi_row_shows_dashes_rather_than_fabricating_a_flat_book(
    tmp_path, monkeypatch,
):
    """An empty-signal journal must render '—', not '0.00' ladder and
    'CLOSED' gate — both of which are real states the strategy can be in."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc).isoformat()
    cycle = {
        "cycle_id": "c1", "started_at": now, "ended_at": now,
        "duration_ms": 5000, "outcome": "success", "error": None,
        "git_commit": None, "python_version": "3.11.0",
        "context": {"mode": "dry_run", "exchange": "binance",
                    "sandbox": True, "quote_currency": "USDT",
                    "basket": ["BTC"]},
        "signal": None, "basket_close_series": None, "balance": None,
        "equity_usd": None, "target_allocation": None,
        "current_holdings_quote": None, "orders_planned": [],
        "orders_skipped": [], "total_skipped_quote_drift": 0.0,
        "orders_executed": None, "schema_version": 2,
    }
    journal = tmp_path / "cycles.jsonl"
    journal.write_text(json.dumps(cycle) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    assert _first_metric(app, "Equity (USDT)").value == "—"
    assert _first_metric(app, "Ladder").value == "—"
    assert _first_metric(app, "SMA(200) gate").value == "—"
    assert _first_metric(app, "Last LIVE").value == "—"
    # Same journal, same claim in the Portfolio tab: an unread equity is
    # unknown, not zero. Both tiles for that label must agree.
    assert [m.value for m in app.metric if m.label == "Equity (USDT)"] == \
        ["—", "—"]


# ---------------------------------------------------------------------------
# Codex round on the KPI row + environment copy
# ---------------------------------------------------------------------------


def test_numeric_rejects_what_as_float_would_coerce_to_a_real_rung():
    """0.00 is a real ladder rung (flat), so a corrupt field must not
    borrow it — and float(True) == 1.0 would conjure full exposure."""
    from trade_lab.monitoring.app import _numeric

    assert _numeric(0.5) == 0.5
    assert _numeric("0.5") == 0.5
    assert _numeric(0) == 0.0
    for bad in (None, True, False, "abc", {}, [], float("nan"),
                float("inf")):
        assert _numeric(bad) is None, bad


def test_ladder_day_series_drops_corrupt_readings_instead_of_zeroing_them():
    """A non-numeric ladder entering the series as 0.0 would manufacture
    a flip in the day-over-day delta."""
    from trade_lab.monitoring.app import _ladder_prev_day_delta

    class _R:
        def cycles(self, n=20):
            return [
                {"signal": {"asof": "2026-08-24T00:00:00+00:00",
                            "ladder_value": 1.0}},
                {"signal": {"asof": "2026-08-25T00:00:00+00:00",
                            "ladder_value": "corrupt"}},
            ]

    # Only one usable day remains -> no delta, rather than a fake -1.00.
    assert _ladder_prev_day_delta(_R()) is None


def test_environment_copy_follows_the_configured_sources(monkeypatch):
    """A testnet-only deployment must not advertise a real-money account
    it cannot see — the #23 defect with the environments swapped."""
    import trade_lab.monitoring.app as app

    monkeypatch.setattr(app, "JOURNAL_SOURCES",
                        {"mainnet": "m.jsonl", "testnet": "t.jsonl"})
    caption, modal = app._environment_copy()
    assert "mainnet (real money, capped)" in caption
    assert "testnet paper environment" in caption
    assert "Environment control above the tabs" in modal
    assert "sidebar" not in modal            # there is no sidebar control

    monkeypatch.setattr(app, "JOURNAL_SOURCES", {"testnet": "t.jsonl"})
    caption, modal = app._environment_copy()
    assert "real money" not in caption, caption
    assert "mainnet" not in caption.lower(), caption
    assert "testnet" in modal

    monkeypatch.setattr(app, "JOURNAL_SOURCES", {"mainnet": "m.jsonl"})
    caption, modal = app._environment_copy()
    assert "mainnet (real money, capped)" in caption
    assert "testnet" not in caption.lower(), caption


def test_kpi_state_deltas_carry_no_direction_arrow(tmp_path, monkeypatch):
    """`12d OPEN` and `SUCCESS` are categorical. delta_color='off' only
    greys the arrow; without delta_arrow='off' Streamlit still paints an
    upward one and state reads as a numeric improvement."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    from streamlit.proto.Metric_pb2 import Metric as _MetricProto

    now = datetime.now(timezone.utc)
    iso = now.isoformat()
    cycle = {
        "cycle_id": "c1", "started_at": iso, "ended_at": iso,
        "duration_ms": 5000, "outcome": "success", "error": None,
        "git_commit": None, "python_version": "3.11.0",
        "context": {"mode": "live", "exchange": "binance", "sandbox": False,
                    "quote_currency": "USDT", "basket": ["BTC"]},
        "signal": {"asof": now.date().isoformat(), "ladder_value": 1.0,
                   "sma_gate_open": True, "basket_close": 100.0,
                   "sma_value": 90.0},
        "basket_close_series": None, "balance": None, "equity_usd": 100.0,
        "target_allocation": None, "current_holdings_quote": None,
        "orders_planned": [], "orders_skipped": [],
        "total_skipped_quote_drift": 0.0,
        "orders_executed": [{"symbol": "BTC/USDT", "side": "buy",
                             "client_order_id": "tl-2026-08-24-BTC-buy",
                             "terminal_status": "closed",
                             "intended_amount": 1.0, "filled_amount": 1.0,
                             "filled_notional_quote": 13.18}],
        "schema_version": 2,
    }
    journal = tmp_path / "cycles.jsonl"
    journal.write_text(json.dumps(cycle) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    none_dir = _MetricProto.MetricDirection.NONE
    for label in ("SMA(200) gate", "Last LIVE"):
        tile = _first_metric(app, label)
        assert tile.proto.direction == none_dir, (
            label, _MetricProto.MetricDirection.Name(tile.proto.direction)
        )


def test_kpi_suppresses_history_deltas_when_the_latest_cycle_has_no_signal(
    tmp_path, monkeypatch,
):
    """A failed newest cycle leaves the tiles at '—'. The history helpers
    still have older signal rows, so the deltas would describe a movement
    of a quantity the tile just said it cannot read."""
    import json
    import pytest as _pytest
    from pathlib import Path

    at = _pytest.importorskip("streamlit.testing.v1")
    now = datetime.now(timezone.utc)

    def _cycle(cid, ended, signal, outcome="success"):
        iso = ended.isoformat()
        return {
            "cycle_id": cid, "started_at": iso, "ended_at": iso,
            "duration_ms": 5000, "outcome": outcome, "error": None,
            "git_commit": None, "python_version": "3.11.0",
            "context": {"mode": "live", "exchange": "binance",
                        "sandbox": False, "quote_currency": "USDT",
                        "basket": ["BTC"]},
            "signal": signal, "basket_close_series": None, "balance": None,
            "equity_usd": 100.0, "target_allocation": None,
            "current_holdings_quote": None, "orders_planned": [],
            "orders_skipped": [], "total_skipped_quote_drift": 0.0,
            "orders_executed": None, "schema_version": 2,
        }

    journal = tmp_path / "cycles.jsonl"
    journal.write_text("\n".join(json.dumps(c) for c in [
        _cycle("d1", now - timedelta(days=2),
               {"asof": (now - timedelta(days=2)).date().isoformat(),
                "ladder_value": 0.0, "sma_gate_open": False}),
        _cycle("d2", now - timedelta(days=1),
               {"asof": (now - timedelta(days=1)).date().isoformat(),
                "ladder_value": 1.0, "sma_gate_open": True}),
        # Newest cycle failed before producing a signal.
        _cycle("d3", now, None, outcome="failed"),
    ]) + "\n")
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)

    repo = Path(__file__).resolve().parents[1]
    app = at.AppTest.from_file(
        str(repo / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30,
    )
    app.run()
    assert not app.exception, app.exception

    ladder = _first_metric(app, "Ladder")
    gate = _first_metric(app, "SMA(200) gate")
    assert ladder.value == "—"
    assert ladder.delta == ""            # not "+1.00 vs prior day"
    assert gate.value == "—"
    assert gate.delta == ""              # not "1d OPEN"
