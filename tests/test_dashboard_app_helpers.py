"""Tests for the research dashboard's per-tab failure containment.

Streamlit executes every tab body in a single top-down script run, so an
uncaught exception in one tab aborts the whole run and blanks the others.
The dashboard must contain each tab's failure the way the monitoring app
does (regression: R3).
"""
from __future__ import annotations


def test_render_tab_safely_contains_exception(monkeypatch):
    """A tab whose renderer raises (e.g. strftime on a non-datetime index)
    must surface a visible error instead of killing the whole run and
    blanking the sibling tabs."""
    import trade_lab.dashboard.app as app

    errors: list[str] = []
    monkeypatch.setattr(app.st, "error", lambda msg: errors.append(msg))

    def broken_tab():
        raise ValueError("Invalid format string for a non-datetime index")

    app._render_tab_safely("Overview", broken_tab)  # must not raise

    assert len(errors) == 1
    assert "Overview" in errors[0]
    assert "ValueError" in errors[0]


def test_render_tab_safely_passes_through_on_success(monkeypatch):
    import trade_lab.dashboard.app as app

    monkeypatch.setattr(app.st, "error", lambda msg: None)
    rendered = []

    app._render_tab_safely("Trades", lambda: rendered.append(True))

    assert rendered == [True]
