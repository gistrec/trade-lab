"""Tests for the shared per-tab failure containment (``trade_lab.uikit``).

Both dashboards delegate to this one implementation; the copies had already
drifted (the monitoring copy grew a reassurance caption the dashboard copy
lacked), so pin the shared behaviour here.
"""
from __future__ import annotations


def test_render_tab_safely_contains_and_shows_note(monkeypatch):
    import trade_lab.uikit as uikit

    errors, captions = [], []
    monkeypatch.setattr(uikit.st, "error", lambda m: errors.append(m))
    monkeypatch.setattr(uikit.st, "caption", lambda m: captions.append(m))

    def boom():
        raise TypeError("schema drift")

    uikit.render_tab_safely("Signal", boom, note="others unaffected")

    assert len(errors) == 1
    assert "Signal" in errors[0] and "TypeError" in errors[0]
    assert captions == ["others unaffected"]


def test_render_tab_safely_no_note_renders_no_caption(monkeypatch):
    import trade_lab.uikit as uikit

    errors, captions = [], []
    monkeypatch.setattr(uikit.st, "error", lambda m: errors.append(m))
    monkeypatch.setattr(uikit.st, "caption", lambda m: captions.append(m))

    def boom():
        raise ValueError("y")

    uikit.render_tab_safely("Overview", boom)   # dashboard: no note

    assert len(errors) == 1
    assert captions == []


def test_render_tab_safely_passes_through_on_success(monkeypatch):
    import trade_lab.uikit as uikit

    ran = []
    uikit.render_tab_safely("X", lambda: ran.append(True))
    assert ran == [True]
