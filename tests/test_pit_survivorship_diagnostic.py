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
from trade_lab.data.coin_registry import COIN_REGISTRY, CoinMeta
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
    assert r1["members"] == ["A", "B"] and r1["deviates_from_control"] is False
    assert r1["control_members"] == ["A", "B"]
    assert r2["members"] == ["A", "B"] and r2["removed"] == []
    assert r3["members"] == ["A", "C"]
    assert r3["added"] == ["C"] and r3["removed"] == ["B"]
    assert r3["control_members"] == ["A", "B"]
    assert r3["missing_vs_control"] == ["B"] and r3["extra_vs_control"] == ["C"]
    assert r3["deviates_from_control"] is True

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


def test_outgoing_holding_nan_close_on_rebalance_bar_fails_loud():
    """The exit leg realizes Y's final return on the rebalance bar — a NaN
    close there must raise, not become an invented flat exit via fillna(0)."""
    n = 10
    idx = _idx(n)
    y = np.full(n, 200.0)
    y[5:] = np.nan
    closes = pd.DataFrame({"X": np.full(n, 100.0), "Y": y}, index=idx)
    baskets = {idx[0]: ("X", "Y"), idx[5]: ("X",)}
    with pytest.raises(SystemExit, match="held member.*Y"):
        diag.build_membership_basket_index(
            closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=10_000.0,
        )


def test_forced_delist_exit_stops_return_accrual():
    """Coin Metrics keeps pricing a delisted coin — after the forced exit
    at its last tradable bar the pump must contribute exactly nothing."""
    n = 12
    idx = _idx(n)
    y = np.full(n, 200.0)
    y[4:] = 200.0 * 2.0 ** np.arange(1, 9)  # keeps pumping after delist
    closes = pd.DataFrame({"X": np.full(n, 100.0), "Y": y}, index=idx)
    baskets = {idx[0]: ("X", "Y"), idx[8]: ("X",)}
    out = diag.build_membership_basket_index(
        closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
        initial_capital=10_000.0, forced_exits={"Y": idx[3]},
    )
    hit = 1.0 - 0.5 * 0.0015  # sell (later: buy) half the book at fee+slip
    assert np.isclose(out["close"].iloc[3], 100.0 * hit)
    np.testing.assert_allclose(
        out["close"].iloc[4:8].to_numpy(), 100.0 * hit, rtol=1e-12
    )
    # Freed weight sits in cash until the next scheduled rebalance, then
    # redeploys into X — charged like any trade.
    assert np.isclose(out["close"].iloc[8], 100.0 * hit * hit)
    np.testing.assert_allclose(
        out["close"].iloc[9:].to_numpy(), 100.0 * hit * hit, rtol=1e-12
    )


def test_delist_exit_reconciles_with_rebalance_validation():
    """After a forced delist exit the dead coin's NaN tail must not trip
    the rebalance-bar held-member check."""
    n = 10
    idx = _idx(n)
    y = np.full(n, 200.0)
    y[4:] = np.nan  # series goes dark right after the exit bar
    closes = pd.DataFrame({"X": np.full(n, 100.0), "Y": y}, index=idx)
    baskets = {idx[0]: ("X", "Y"), idx[5]: ("X",)}
    out = diag.build_membership_basket_index(
        closes, baskets, fee_rate=0.001, slippage_rate=0.0005,
        initial_capital=10_000.0, forced_exits={"Y": idx[3]},
    )
    hit = 1.0 - 0.5 * 0.0015
    assert np.isclose(out["close"].iloc[3], 100.0 * hit)
    assert np.isclose(out["close"].iloc[-1], 100.0 * hit * hit)


def test_forced_exit_guards_fail_loud():
    n = 10
    idx = _idx(n)
    closes = pd.DataFrame({"X": np.full(n, 100.0)}, index=idx)
    with pytest.raises(SystemExit, match="delisted member"):
        diag.build_membership_basket_index(
            closes, {idx[0]: ("X",)}, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=10_000.0, forced_exits={"X": idx[0]},
        )
    with pytest.raises(SystemExit, match="unknown column"):
        diag.build_membership_basket_index(
            closes, {idx[0]: ("X",)}, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=10_000.0, forced_exits={"Z": idx[3]},
        )


def test_compute_forced_exits_maps_to_last_tradable_bar():
    idx = _idx(30)  # 2020-01-01 .. 2020-01-30
    pool = {
        "LIVE": CoinMeta("live-id", "LIVE/USDT", "2020-01-01", None),
        "DEAD": CoinMeta("dead-id", "DEAD/USDT", "2020-01-01", "2020-01-15"),
        "SOON": CoinMeta("soon-id", "SOON/USDT", "2020-01-01", "2099-01-01"),
        "PRE":  CoinMeta("pre-id",  "PRE/USDT",  "2019-01-01", "2019-06-01"),
    }
    exits = diag.compute_forced_exits(idx, pool)
    # delisted_date itself is already suspended -> exit the bar before.
    assert exits == {"DEAD": pd.Timestamp("2020-01-14", tz="UTC")}


def test_run_diagnostic_forces_exit_on_mid_month_delisting(tmp_path):
    """C delists 2020-02-20 and its series goes dark: both index runs must
    force-exit it at the last tradable bar instead of raising on the
    held-member NaN check (or silently accruing post-delist returns)."""
    prices, market_caps, volumes, pool = _stub_panel()
    cutoff = pd.Timestamp("2020-02-20", tz="UTC")
    prices.loc[prices.index >= cutoff, "C"] = np.nan
    volumes.loc[volumes.index >= cutoff, "C"] = np.nan
    market_caps["C"] = np.where(market_caps.index < cutoff, 5e10, np.nan)
    pool = dict(pool)
    pool["C"] = CoinMeta("c-id", "C/USDT", "2020-01-01", "2020-02-20")
    payload = diag.run_diagnostic(
        prices, market_caps, volumes, pool,
        out_dir=tmp_path, static_basket=("A", "C"),
    )
    # The delisting is an availability change -> its own rebalance bar,
    # not a wait until 2020-03-01 (deployed n_active rule).
    r1, r2, r3, r4 = payload["rebalances"]
    assert r1["members"] == ["A", "C"] and r2["members"] == ["A", "C"]
    assert r3["date"] == "2020-02-20" and r3["trigger"] == "membership"
    assert r3["members"] == ["A", "B"] and r3["removed"] == ["C"]
    assert r4["date"] == "2020-03-01" and r4["trigger"] == "schedule"
    # The control drops C on the same bar, not at month end.
    ctl = {r["date"]: r for r in payload["control_rebalances"]}
    assert ctl["2020-02-20"]["members"] == ["A"]
    assert ctl["2020-02-20"]["trigger"] == "membership"
    assert (tmp_path / "pit_survivorship_diagnostic.md").exists()


def _late_listing_panel():
    """3 assets over 2020-01-01..2020-02-19; B lists mid-month 2020-01-15."""
    n = 50
    idx = _idx(n)
    steps = np.arange(n, dtype=float)
    listing = pd.Timestamp("2020-01-15", tz="UTC")
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.0 + 0.001) ** steps,
            "B": 50.0 * (1.0 + 0.002) ** steps,
            "C": 20.0 * (1.0 + 0.0005) ** steps,
        },
        index=idx,
    )
    prices.loc[idx < listing, "B"] = np.nan
    market_caps = pd.DataFrame(
        {"A": np.full(n, 1e11), "B": np.full(n, 5e10), "C": np.full(n, 1e10)},
        index=idx,
    )
    market_caps.loc[idx < listing, "B"] = np.nan
    volumes = pd.DataFrame(
        {"A": np.full(n, 1e9), "B": np.full(n, 1e9), "C": np.full(n, 1e9)},
        index=idx,
    )
    volumes.loc[idx < listing, "B"] = np.nan
    pool = {
        "A": CoinMeta("a-id", "A/USDT", "2020-01-01", None),
        "B": CoinMeta("b-id", "B/USDT", "2020-01-15", None),
        "C": CoinMeta("c-id", "C/USDT", "2020-01-01", None),
    }
    return prices, market_caps, volumes, pool


def test_availability_change_bars_flags_the_listing_bar():
    prices, _, _, pool = _late_listing_panel()
    bars = diag.availability_change_bars(prices.index, ("A", "B"), prices, pool)
    assert list(bars) == [pd.Timestamp("2020-01-15", tz="UTC")]
    assert bars[pd.Timestamp("2020-01-15", tz="UTC")] == frozenset({"B"})


def test_static_control_rebalances_on_mid_month_listing(tmp_path):
    """Deployed rule (market_index.py: n_active.diff().ne(0)): a member
    that becomes tradable mid-month enters on THAT bar, not at month end."""
    prices, market_caps, volumes, pool = _late_listing_panel()
    payload = diag.run_diagnostic(
        prices, market_caps, volumes, pool,
        out_dir=tmp_path, static_basket=("A", "B"),
    )
    ctl = {r["date"]: r for r in payload["control_rebalances"]}
    assert list(ctl) == ["2020-01-01", "2020-01-15", "2020-02-01"]
    assert ctl["2020-01-01"]["members"] == ["A"]
    assert ctl["2020-01-15"]["members"] == ["A", "B"]
    assert ctl["2020-01-15"]["trigger"] == "membership"
    # ...and the PIT arm reads the same rule off the same bar, so the
    # delta cannot absorb an entry-timing difference.
    pit = {r["date"]: r for r in payload["rebalances"]}
    assert pit["2020-01-15"]["trigger"] == "membership"
    assert pit["2020-01-15"]["members"] == ["A", "B"]
    assert "n_active.diff().ne(0)" in payload["rebalance_rule"]
    md = (tmp_path / "pit_survivorship_diagnostic.md").read_text()
    assert "off-schedule rebalances" in md and "2020-01-15" in md


def test_late_listing_is_not_reported_as_a_composition_deviation(tmp_path):
    """B lists mid-month, so NEITHER arm holds it on 2020-01-01. Scoring the
    PIT basket against the frozen names would flag "≠ −B" on a bar where the
    control holds exactly the same member — an entry-timing artefact, not a
    composition delta."""
    prices, market_caps, volumes, pool = _late_listing_panel()
    volumes["C"] = np.nan  # keep C out of the ranking: PIT == control pre-listing
    payload = diag.run_diagnostic(
        prices, market_caps, volumes, pool,
        out_dir=tmp_path, static_basket=("A", "B"),
    )
    pit = {r["date"]: r for r in payload["rebalances"]}
    ctl = {r["date"]: r for r in payload["control_rebalances"]}
    assert pit["2020-01-01"]["members"] == ctl["2020-01-01"]["members"] == ["A"]
    # The frozen names include B; the control does not hold it yet — scoring
    # against the frozen set is exactly what produced the phantom "−B".
    assert payload["static_basket"] == ["A", "B"]
    assert pit["2020-01-01"]["control_members"] == ["A"]
    assert pit["2020-01-01"]["missing_vs_control"] == []
    assert pit["2020-01-01"]["deviates_from_control"] is False
    # The listing bar itself is shared too — still no composition delta.
    assert pit["2020-01-15"]["members"] == ctl["2020-01-15"]["members"] == ["A", "B"]
    assert pit["2020-01-15"]["deviates_from_control"] is False
    md = (tmp_path / "pit_survivorship_diagnostic.md").read_text()
    assert "**≠**" not in md


def test_missing_coverage_requires_explicit_acknowledgement(tmp_path):
    """Empty cache + fetch=False: every registry coin lacks coverage and
    the loader must abort instead of silently shrinking the pool."""
    with pytest.raises(SystemExit, match="no usable Coin Metrics series at all"):
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


_CM_COLUMNS = ("price", "market_cap", "volume_usd")


def _write_cm_cache(
    cache_dir: Path,
    cm_id: str,
    *,
    empty: bool = False,
    columns: tuple[str, ...] = _CM_COLUMNS,
) -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC", name="time")
    val = np.nan if empty else 1.0
    pd.DataFrame({col: val for col in columns}, index=idx).to_parquet(
        cache_dir / f"coinmetrics_{cm_id}.parquet"
    )


def test_allow_missing_with_usable_coverage_fails_loud(tmp_path):
    """A stale flag must not keep a repaired asset out under a false reason."""
    _write_cm_cache(tmp_path, COIN_REGISTRY["LTC"].cm_id)
    with pytest.raises(SystemExit, match="LTC.*coverage IS present"):
        diag.load_diagnostic_panel(
            tmp_path, allow_missing=["LTC"], static_basket=("BTC",), fetch=False
        )


def test_allow_missing_exclusion_is_verified_against_the_panel(tmp_path):
    """Genuine gaps still pass — a missing file and an all-NaN column both
    count, and the recorded reason says the gap was re-checked."""
    for sym, meta in COIN_REGISTRY.items():
        if sym == "WAVES":
            continue  # no cache file at all
        _write_cm_cache(tmp_path, meta.cm_id, empty=(sym == "OMG"))
    _, market_caps, _, pool, excluded = diag.load_diagnostic_panel(
        tmp_path, allow_missing=["WAVES", "OMG"], static_basket=("BTC",),
        fetch=False,
    )
    assert set(excluded) == {"WAVES", "OMG"}
    assert all("verified" in reason for reason in excluded.values())
    # The recorded reason distinguishes the two waivable shapes.
    assert "no cached Coin Metrics file" in excluded["WAVES"]
    assert "all empty" in excluded["OMG"]
    assert "WAVES" not in pool and "OMG" not in pool
    assert "BTC" in pool and "BTC" in market_caps.columns


def test_missing_mcap_alone_is_not_waivable(tmp_path):
    """A coin with a full price+volume series but no market_cap column is a
    DATA gap, not an absent asset: --allow-missing must refuse it (it would
    otherwise drop a live top-N candidate under a false "no coverage"
    reason) and the PITMcapGapError must stay reachable."""
    for sym, meta in COIN_REGISTRY.items():
        _write_cm_cache(
            tmp_path, meta.cm_id,
            columns=("price", "volume_usd") if sym == "LTC" else _CM_COLUMNS,
        )
    with pytest.raises(SystemExit, match=r"LTC.*market_cap 0 bars"):
        diag.load_diagnostic_panel(
            tmp_path, allow_missing=["LTC"], static_basket=("BTC",), fetch=False
        )

    # Without the flag it is NOT quietly dropped either — it stays in the
    # pool and the strict mcap check aborts the run.
    prices, market_caps, volumes, pool, excluded = diag.load_diagnostic_panel(
        tmp_path, allow_missing=[], static_basket=("BTC",), fetch=False
    )
    assert excluded == {} and "LTC" in pool
    assert market_caps["LTC"].isna().all()
    assert prices["LTC"].notna().any() and volumes["LTC"].notna().any()
    with pytest.raises(PITMcapGapError, match="LTC @ 2020-01-01"):
        diag.run_diagnostic(prices, market_caps, volumes, pool, out_dir=tmp_path)
    assert not (tmp_path / "pit_survivorship_diagnostic.md").exists()
