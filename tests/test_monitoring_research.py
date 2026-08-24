"""Tests for the read-only research loader (``monitoring/research.py``) and a
smoke test that the dashboard app (now with the Research tab + About modal)
still runs headlessly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trade_lab.monitoring import research

_REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# research.py — the loader
# --------------------------------------------------------------------------

def test_groups_shape():
    assert list(research.GROUPS) == [
        "Deployable strategy", "Overlays & wrappers", "Rejected / inconclusive",
        "Validation", "Literature & sessions", "Strategy reference",
        "Results analyses", "Methodology",
    ]
    # 21 findings + 8 strategies + 8 results + 2 methodology.
    assert sum(len(v) for v in research.GROUPS.values()) == 39


def test_all_docs_exist_and_are_nonempty():
    # Every doc the picker can select must exist on disk — this catches a
    # renamed/deleted writeup before it 404s the tab.
    missing = [p for p in research.all_docs() if not (_REPO / p).is_file()]
    assert not missing, f"research docs missing on disk: {missing}"
    for p in research.all_docs():
        assert (_REPO / p).read_text(encoding="utf-8").strip(), f"empty: {p}"


def test_read_markdown_returns_content_for_known_doc():
    text = research.results_markdown()
    assert "master index" in text.lower()


def test_read_markdown_rejects_unknown_and_traversal_paths():
    for bad in ["../secret.md", "/etc/passwd", "src/trade_lab/cli.py",
                "findings/does_not_exist.md"]:
        assert research.read_markdown(bad).startswith("_Unknown document")


def test_doc_title_reads_first_heading():
    title = research.doc_title("findings/han_28d_tsmom.md")
    assert title and not title.startswith("#")
    # falls back to a humanized filename only if there is no heading (there is)
    assert "tsmom" in title.lower() or "han" in title.lower()


# --------------------------------------------------------------------------
# with_github_links — backtick doc-path references become GitHub links
# --------------------------------------------------------------------------

_REPO_URL = "https://github.com/gistrec/trade-lab"


def test_with_github_links_linkifies_allowlisted_path():
    out = research.with_github_links(
        "see `findings/han_28d_tsmom.md` for detail", _REPO_URL)
    assert ("[`findings/han_28d_tsmom.md`]"
            "(https://github.com/gistrec/trade-lab/blob/main/"
            "findings/han_28d_tsmom.md)") in out


def test_with_github_links_linkifies_results_index_and_nested_docs():
    out = research.with_github_links(
        "`RESULTS.md` and `docs/results/yearly_btc.md`", _REPO_URL)
    assert "/blob/main/RESULTS.md)" in out
    assert "/blob/main/docs/results/yearly_btc.md)" in out


def test_with_github_links_leaves_unknown_or_private_path_as_code():
    # gitignored/private and non-corpus paths must NOT become dead links.
    for path in ["docs/systems_visibility_roadmap.md", "docs/SLO.md",
                 "findings/does_not_exist.md"]:
        src = f"note `{path}` here"
        assert research.with_github_links(src, _REPO_URL) == src


def test_with_github_links_ignores_paths_without_backticks():
    src = "a bare findings/han_28d_tsmom.md mention in prose"
    assert research.with_github_links(src, _REPO_URL) == src


def test_with_github_links_noop_without_repo_url():
    src = "see `findings/han_28d_tsmom.md`"
    assert research.with_github_links(src, "") == src


def test_with_github_links_respects_ref_and_trailing_slash():
    out = research.with_github_links(
        "`RESULTS.md`", _REPO_URL + "/", ref="abc123")
    assert "https://github.com/gistrec/trade-lab/blob/abc123/RESULTS.md)" in out
    assert "/blob/abc123//" not in out  # trailing slash stripped, no double //


# --------------------------------------------------------------------------
# App smoke — the app still runs headlessly with the new tab/modal
# --------------------------------------------------------------------------

def test_dashboard_app_runs_headless(tmp_path, monkeypatch):
    at = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv(
        "TRADE_LAB_MONITORING_JOURNAL_PATH", str(tmp_path / "cycles.jsonl"))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)
    app = at.AppTest.from_file(
        str(_REPO / "src" / "trade_lab" / "monitoring" / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    labels = [t.label for t in app.tabs]
    assert any("Research" in lbl for lbl in labels), labels
    # Single-source mode: no environment switcher.
    assert len(app.segmented_control) == 0


def _cycle_line(sandbox: bool) -> str:
    import json
    return json.dumps({
        "cycle_id": "x", "started_at": "2026-07-09T00:00:00+00:00",
        "ended_at": "2026-07-09T00:00:05+00:00", "duration_ms": 5000,
        "outcome": "success", "error": None, "git_commit": None,
        "python_version": "3.11.0",
        "context": {"mode": "dry_run", "exchange": "binance",
                    "sandbox": sandbox, "quote_currency": "USDT",
                    "basket": ["BTC"]},
        "signal": None, "basket_close_series": None, "balance": None,
        "equity_usd": None, "target_allocation": None,
        "current_holdings_quote": None, "orders_planned": None,
        "orders_skipped": None, "total_skipped_quote_drift": None,
        "orders_executed": None, "schema_version": 2,
    }) + "\n"


def test_env_switcher_switches_journals(tmp_path, monkeypatch):
    """Two configured sources -> segmented control renders, defaults to
    mainnet (real money is the environment worth landing on), and
    selecting testnet swaps the journal (banner flips to the
    content-derived TESTNET wording)."""
    at = pytest.importorskip("streamlit.testing.v1")
    testnet = tmp_path / "cycles.jsonl"
    mainnet = tmp_path / "cycles_mainnet.jsonl"
    testnet.write_text(_cycle_line(sandbox=True))
    mainnet.write_text(_cycle_line(sandbox=False))
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(testnet))
    monkeypatch.setenv(
        "TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET", str(mainnet))

    app = at.AppTest.from_file(
        str(_REPO / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30)
    app.run()
    assert not app.exception, app.exception

    assert len(app.segmented_control) == 1
    control = app.segmented_control[0]
    assert control.value == "mainnet"
    page = " ".join(str(m.value) for m in app.markdown)
    assert "MAINNET — BINANCE — REAL MONEY" in page
    # No SOURCE MISMATCH: mainnet label points at a mainnet journal.
    assert "SOURCE MISMATCH" not in page
    # The full-perimeter red frame is a fixed overlay (a box-shadow on
    # the app container is painted over by Streamlit's fixed header).
    assert "border:4px solid #b71c1c" in page
    assert "position:fixed" in page
    markdown_count_mainnet = len(app.markdown)

    control.set_value("testnet")
    app.run()
    assert not app.exception, app.exception
    page = " ".join(str(m.value) for m in app.markdown)
    assert "TESTNET — BINANCE" in page
    # Spacing invariant: the frame must NOT be an extra element in the
    # flow — Streamlit's inter-element gap would push the mainnet
    # banner lower than the testnet one. Same element count both ways.
    assert len(app.markdown) == markdown_count_mainnet


# --------------------------------------------------------------------------
# Unfillable-drift panel — a lone live cycle must not plot as a bare dot
# --------------------------------------------------------------------------

def _live_cycle_line(ended_at: str, skipped_drift: float) -> str:
    """A cycle that placed real orders (orders_executed is a list)."""
    import json
    return json.dumps({
        "cycle_id": "live-" + ended_at[:10], "started_at": ended_at,
        "ended_at": ended_at, "duration_ms": 5000, "outcome": "success",
        "error": None, "git_commit": None, "python_version": "3.11.0",
        "context": {"mode": "live", "exchange": "binance", "sandbox": True,
                    "quote_currency": "USDT", "basket": ["BTC"]},
        "signal": {"asof": ended_at, "ladder_value": 1.0,
                   "sma_gate_open": True, "per_lookback_states": {"28": 1},
                   "basket_close": 100.0, "asset_closes": {"BTC": 50000.0}},
        "basket_close_series": None,
        "balance": {"quote_currency": "USDT", "quote_total": 10.0,
                    "quote_free": 10.0, "quote_used": 0.0,
                    "asset_totals": {"BTC": 0.001}},
        "equity_usd": 100.0, "target_allocation": {"BTC": 100.0},
        "current_holdings_quote": {"BTC": 100.0},
        "orders_planned": [], "orders_skipped": [],
        "total_skipped_quote_drift": skipped_drift,
        "orders_executed": [], "schema_version": 2,
    }) + "\n"


def _run_app_with(journal_text: str, tmp_path, monkeypatch):
    at = pytest.importorskip("streamlit.testing.v1")
    journal = tmp_path / "cycles.jsonl"
    journal.write_text(journal_text)
    monkeypatch.setenv("TRADE_LAB_MONITORING_JOURNAL_PATH", str(journal))
    monkeypatch.delenv("TRADE_LAB_MONITORING_JOURNAL_PATH_MAINNET",
                       raising=False)
    app = at.AppTest.from_file(
        str(_REPO / "src" / "trade_lab" / "monitoring" / "app.py"),
        default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    return app


def test_single_live_cycle_states_the_reading_instead_of_charting(
        tmp_path, monkeypatch):
    """One point plots as a dot with no line — say it in words instead."""
    app = _run_app_with(
        _live_cycle_line("2026-08-24T00:53:25+00:00", 0.0),
        tmp_path, monkeypatch)
    infos = [str(i.value) for i in app.info]
    assert any("1 live cycle so far" in i for i in infos), infos


def test_no_live_cycles_explains_the_empty_panel(tmp_path, monkeypatch):
    app = _run_app_with(_cycle_line(sandbox=True), tmp_path, monkeypatch)
    infos = [str(i.value) for i in app.info]
    assert any("No live cycles yet" in i for i in infos), infos


def test_two_live_cycles_render_the_chart(tmp_path, monkeypatch):
    app = _run_app_with(
        _live_cycle_line("2026-08-24T00:45:00+00:00", 0.0)
        + _live_cycle_line("2026-08-25T00:45:00+00:00", 1.25),
        tmp_path, monkeypatch)
    infos = [str(i.value) for i in app.info]
    assert not any("1 live cycle so far" in i for i in infos), infos
    assert not any("No live cycles yet" in i for i in infos), infos
