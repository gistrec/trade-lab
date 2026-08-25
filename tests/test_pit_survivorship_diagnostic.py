"""Tests for scripts/pit_survivorship_diagnostic.py on synthetic PIT data.

The real-data run is a separate post-merge step; here a tiny stub
(3 assets / 3 rebalances) exercises the full pipeline and the report
structure, plus the fail-loud paths and the equivalence of the
membership-aware index builder with ``build_crypto_market_index``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtest.market_index import build_crypto_market_index
from trade_lab.data.coin_registry import CoinMeta
from trade_lab.data.universe import PITMcapGapError

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "pit_survivorship_diagnostic.py"
)
_spec = importlib.util.spec_from_file_location("pit_survivorship_diagnostic", _SCRIPT)
assert _spec is not None and _spec.loader is not None
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")


def _stub_panel():
    """3 assets, 80 daily bars -> rebalances at Jan 1 / Feb 1 / Mar 1 2020.

    Mcap ranks flip before the March rebalance: B's cap collapses on
    2020-02-20, so top-2 goes {A, B} -> {A, B} -> {A, C}. Volumes are
    equal on purpose (shared min-rank 1) so composition is mcap-driven.
    """
    n = 80
    idx = _idx(n)
    steps = np.arange(n, dtype=float)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.0 + 0.001) ** steps,
            "B": 50.0 * (1.0 + 0.002) ** steps,
            "C": 20.0 * (1.0 + 0.0005) ** steps,
        },
        index=idx,
    )
    b_cap = np.where(idx < pd.Timestamp("2020-02-20", tz="UTC"), 1e10, 1e8)
    market_caps = pd.DataFrame(
        {"A": np.full(n, 1e11), "B": b_cap, "C": np.full(n, 5e9)},
        index=idx,
    )
    volumes = pd.DataFrame(
        {"A": np.full(n, 1e9), "B": np.full(n, 1e9), "C": np.full(n, 1e9)},
        index=idx,
    )
    pool = {
        sym: CoinMeta(f"{sym.lower()}-id", f"{sym}/USDT", "2020-01-01", None)
        for sym in ("A", "B", "C")
    }
    return prices, market_caps, volumes, pool


def test_stub_end_to_end_report_structure(tmp_path):
    prices, market_caps, volumes, pool = _stub_panel()
    payload = diag.run_diagnostic(
        prices, market_caps, volumes, pool,
        out_dir=tmp_path, static_basket=("A", "B"),
    )

    md_path = tmp_path / "pit_survivorship_diagnostic.md"
    json_path = tmp_path / "pit_survivorship_diagnostic.json"
    assert md_path.exists() and json_path.exists()
    js = json.loads(json_path.read_text())
    assert js["num_trials"] == 500  # diagnostic re-run, not a new search
    assert js["walk_forward"]["lookbacks"] == [28, 60]
    assert js["walk_forward"]["sma_filter_periods"] == [200]
    assert js["static_basket"] == ["A", "B"]

    # Composition: 3 rebalances, deviation only at the third.
    assert [r["date"] for r in js["rebalances"]] == [
        "2020-01-01", "2020-02-01", "2020-03-01",
    ]
    r1, r2, r3 = js["rebalances"]
    assert r1["members"] == ["A", "B"] and r1["deviates_from_static"] is False
    assert r2["members"] == ["A", "B"] and r2["removed"] == []
    assert r3["members"] == ["A", "C"]
    assert r3["added"] == ["C"] and r3["removed"] == ["B"]
    assert r3["missing_vs_static"] == ["B"] and r3["extra_vs_static"] == ["C"]
    assert r3["deviates_from_static"] is True

    # Both runs carry the frozen walk-forward summary structure even when
    # the stub window is too short for a single 24m+6m fold.
    for run in ("pit_top_n", "static_control"):
        summary = js["runs"][run]["summary"]
        for key in (
            "concatenated_oos_sharpe",
            "concatenated_oos_dsr",
            "concatenated_oos_max_drawdown_pct",
            "mean_test_sharpe",
            "hit_rate",
            "n_folds",
        ):
            assert key in summary
        assert js["runs"][run]["folds"] == []
        assert summary["n_folds"] == 0
    assert set(js["delta_pit_minus_control"]) >= {
        "concatenated_oos_sharpe", "concatenated_oos_dsr",
    }
    assert payload["rebalances"] == js["rebalances"]

    md = md_path.read_text()
    assert "does not change" in md          # deployed basket is untouched
    assert "**≠**" in md                    # deviation from static highlighted
    assert "PROJECT_NUM_TRIALS" in md and "500" in md


def test_nan_mcap_on_rebalance_date_fails_loud(tmp_path):
    prices, market_caps, volumes, pool = _stub_panel()
    market_caps.loc[pd.Timestamp("2020-02-01", tz="UTC"), "B"] = np.nan
    with pytest.raises(PITMcapGapError, match="B @ 2020-02-01"):
        diag.run_diagnostic(
            prices, market_caps, volumes, pool,
            out_dir=tmp_path, static_basket=("A", "B"),
        )
    # Fail loud means no partial report either.
    assert not (tmp_path / "pit_survivorship_diagnostic.md").exists()


def test_membership_index_matches_deployed_builder_for_constant_membership():
    """With constant membership and all assets listed from bar 0 the
    script's builder must reproduce build_crypto_market_index exactly."""
    n = 400
    idx = _idx(n)
    steps = np.arange(n, dtype=float)
    x = 100.0 * np.cumprod(1.0 + 0.001 * np.sin(steps / 7.0) + 0.0002)
    y = 80.0 * np.cumprod(1.0 + 0.0015 * np.cos(steps / 11.0) + 0.0001)
    closes = pd.DataFrame({"X": x, "Y": y}, index=idx)

    bars = diag.monthly_rebalance_bars(idx)
    baskets = {ts: ("X", "Y") for ts in bars}
    mine = diag.build_membership_basket_index(
        closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
        initial_capital=10_000.0,
    )
    theirs = build_crypto_market_index(
        {sym: pd.DataFrame({"close": closes[sym]}) for sym in ("X", "Y")},
        fee_rate=0.001, slippage_rate=0.0005, initial_capital=10_000.0,
        rebalance_freq="MS",
    )
    np.testing.assert_allclose(
        mine["close"].to_numpy(), theirs["close"].to_numpy(), rtol=1e-10
    )


def test_membership_switch_charges_full_turnover():
    """X -> Y switch on flat prices: sell 1.0 + buy 1.0 at fee+slippage."""
    n = 10
    idx = _idx(n)
    closes = pd.DataFrame(
        {"X": np.full(n, 100.0), "Y": np.full(n, 200.0)}, index=idx
    )
    baskets = {idx[0]: ("X",), idx[5]: ("Y",)}
    out = diag.build_membership_basket_index(
        closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
        initial_capital=10_000.0,
    )
    assert np.isclose(out["close"].iloc[4], 100.0)
    # turnover 2.0 * (0.001 + 0.0005) = 30 bps hit on the switch bar.
    assert np.isclose(out["close"].iloc[5], 100.0 * (1.0 - 0.003))
    assert np.isclose(out["close"].iloc[-1], 100.0 * (1.0 - 0.003))


def test_held_member_losing_prices_fails_loud():
    n = 10
    idx = _idx(n)
    y = np.full(n, 200.0)
    y[6:] = np.nan
    closes = pd.DataFrame({"X": np.full(n, 100.0), "Y": y}, index=idx)
    baskets = {idx[0]: ("Y",)}
    with pytest.raises(SystemExit, match="held member.*Y"):
        diag.build_membership_basket_index(
            closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=10_000.0,
        )


def test_rebalance_member_without_close_fails_loud():
    n = 10
    idx = _idx(n)
    x = np.full(n, 100.0)
    x[0] = np.nan
    closes = pd.DataFrame({"X": x}, index=idx)
    baskets = {idx[0]: ("X",)}
    with pytest.raises(SystemExit, match="basket member.*X"):
        diag.build_membership_basket_index(
            closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=10_000.0,
        )


def test_missing_coverage_requires_explicit_acknowledgement(tmp_path):
    """Empty cache + fetch=False: every registry coin lacks coverage and
    the loader must abort instead of silently shrinking the pool."""
    with pytest.raises(SystemExit, match="no Coin Metrics coverage"):
        diag.load_diagnostic_panel(
            tmp_path, allow_missing=[], static_basket=("BTC",), fetch=False
        )


def test_static_basket_member_cannot_be_excluded(tmp_path):
    with pytest.raises(SystemExit, match="static-basket member"):
        diag.load_diagnostic_panel(
            tmp_path, allow_missing=["BTC"], static_basket=("BTC", "ETH"),
            fetch=False,
        )


def test_unknown_allow_missing_symbol_rejected(tmp_path):
    with pytest.raises(SystemExit, match="not in COIN_REGISTRY"):
        diag.load_diagnostic_panel(
            tmp_path, allow_missing=["ZZZZ"], static_basket=("BTC",),
            fetch=False,
        )
