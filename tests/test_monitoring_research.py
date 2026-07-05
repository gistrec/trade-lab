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
    app = at.AppTest.from_file(
        str(_REPO / "src" / "trade_lab" / "monitoring" / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    labels = [t.label for t in app.tabs]
    assert any("Research" in lbl for lbl in labels), labels
