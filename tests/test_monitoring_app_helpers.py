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


# ---------------------------------------------------------------------------
# Fail-loud robustness (Theme 3)
# ---------------------------------------------------------------------------


def test_series_return_tolerates_null_and_garbage_elements():
    from trade_lab.monitoring.app import _series_return

    assert _series_return([100.0, None], 1) == "—"        # null element
    assert _series_return([100.0, "x"], 1) == "—"         # garbage element
    assert _series_return([100.0, 110.0], 1) == "+10.00%"  # still works


def test_render_read_stats_warns_on_corrupt_and_errors_on_read_error(monkeypatch):
    import trade_lab.monitoring.app as app
    from trade_lab.monitoring.data_source import ReadStats

    cap = {}
    _stub_st(monkeypatch, cap)

    app._render_read_stats(ReadStats(total_lines=10, valid_cycles=8,
                                     corrupt_lines=2))
    assert cap["warning"]                       # corrupt → warning, not caption

    cap2 = {}
    _stub_st(monkeypatch, cap2)
    app._render_read_stats(ReadStats(read_error="PermissionError: denied"))
    assert cap2["error"]                        # unreadable journal → loud error


def test_render_read_stats_silent_when_clean(monkeypatch):
    import trade_lab.monitoring.app as app
    from trade_lab.monitoring.data_source import ReadStats

    cap = {}
    _stub_st(monkeypatch, cap)
    app._render_read_stats(ReadStats(total_lines=5, valid_cycles=5))
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

        def cumulative_skipped_drift(self):
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


def test_render_incidents_shows_stall_banner_after_clean_success(monkeypatch):
    """On a clean window the banner renders AFTER the 'No failed/partial
    cycles...' success line — the operational verdict first, the structural
    SMA(200) notice below it — so a clean window still explains why no
    trades ever happen."""
    import trade_lab.monitoring.app as app

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(app.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(app.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(
        app.st, "success", lambda msg, *a, **k: calls.append(("success", msg)))
    monkeypatch.setattr(
        app.st, "info", lambda msg, *a, **k: calls.append(("info", msg)))

    reader = _stall_reader(sma_value=None, bars=36)
    reader.cycles = lambda n=500: []   # clean window: no incidents at all
    app._render_incidents(reader)

    kinds = [k for k, _ in calls]
    assert kinds == ["success", "info"]      # verdict first, then the notice
    assert "No failed/partial cycles" in calls[0][1]
    assert "SMA(200)" in calls[1][1]
    assert "no buy order is placed" in calls[1][1]
