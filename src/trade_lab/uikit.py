"""Shared Streamlit UI helpers for the two dashboards.

Both the read-only monitoring app and the backtest-inspection dashboard need
the same per-tab failure containment. Keeping ONE implementation here removes
the copy that had already drifted between the two apps (the monitoring copy
grew a reassurance caption the dashboard copy lacked). No exchange access, no
credentials — pure rendering.
"""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st


def render_tab_safely(
    tab_name: str,
    render_fn: Callable[[], None],
    *,
    note: Optional[str] = None,
) -> None:
    """Contain a tab's failure to that tab.

    Streamlit aborts the whole script run on an uncaught exception, so a
    single drifted module or schema-drifted journal row would otherwise blank
    every tab at once. The error stays loud — rendered red inside the failing
    tab — without taking down the siblings. ``note`` renders an optional
    caption under the error (the monitoring app uses it for its
    'other tabs are unaffected' reassurance).
    """
    try:
        render_fn()
    except Exception as exc:
        st.error(f"{tab_name} tab failed: {type(exc).__name__}: {exc}")
        if note:
            st.caption(note)
