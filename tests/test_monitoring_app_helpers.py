"""Tests for pure-function helpers in ``trade_lab.monitoring.app``.

The Streamlit rendering itself is verified manually (no ScriptRunContext
in pytest), but the helpers that format timestamps for the Status tab
are pure and worth pinning so a future refactor does not silently
regress the narrow-screen layout."""
from __future__ import annotations

from datetime import datetime, timezone


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

    assert _cycle_mode({"orders_executed": []}) == "LIVE"
    assert _cycle_mode({"orders_executed": [{"symbol": "BTC"}]}) == "LIVE"
    assert _cycle_mode({"orders_executed": None}) == "DRY"
    assert _cycle_mode({}) == "DRY"
    assert _cycle_mode(None) == "DRY"


class _LiveReader:
    def __init__(self, live=None, cycles=None):
        self._live = live
        self._cycles = cycles or []

    def latest_live_cycle(self):
        return self._live

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

        def cumulative_skipped_drift(self):
            return 0.0

    app._render_portfolio(_Reader())   # must not raise
