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
