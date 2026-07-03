"""CLI --param value coercion.

A strategy boolean flag passed as `--param use_vol_target=false` must
become Python False. The strategy constructor does bool(use_vol_target),
and bool("false") is True — so an un-coerced string flag silently
inverted the operator's intent and backtested a different config
(regression: C1).
"""
from __future__ import annotations

from trade_lab.cli import _coerce, _parse_params


def test_coerce_boolean_word_literals():
    assert _coerce("false") is False
    assert _coerce("true") is True
    assert _coerce("no") is False
    assert _coerce("off") is False
    assert _coerce("yes") is True
    assert _coerce("on") is True
    # Case-insensitive.
    assert _coerce("False") is False
    assert _coerce("TRUE") is True


def test_coerce_numeric_stays_numeric():
    """0/1 must stay int (a strategy may want lookback=1), not become
    bool — only word literals map to bool."""
    assert _coerce("1") == 1 and isinstance(_coerce("1"), int)
    assert _coerce("0") == 0 and isinstance(_coerce("0"), int)
    assert _coerce("0.5") == 0.5
    assert _coerce("28") == 28


def test_coerce_non_boolean_string_passes_through():
    assert _coerce("donchian") == "donchian"


def test_parse_params_disables_bool_flag():
    assert _parse_params(["use_vol_target=false"]) == {"use_vol_target": False}
