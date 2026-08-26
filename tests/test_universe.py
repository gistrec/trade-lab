"""Unit tests for the PIT universe builder.

The tests use synthetic market_cap / volume panels and never hit the
CoinGecko API. CoinGecko integration is exercised separately by an
ad-hoc smoke script — it is too slow and network-dependent to run on
every test invocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.data.coin_registry import CoinMeta, stablecoins
from trade_lab.data.universe import (
    PITMcapGapError,
    build_pit_universe,
    closes_for_universe,
)


def _date_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")


def _registry() -> dict[str, CoinMeta]:
    """A small synthetic registry for tests — three majors + one
    delisting + one stablecoin."""
    return {
        "BIG":   CoinMeta("big-id",   "BIG/USDT",   "2020-01-01", None),
        "MED":   CoinMeta("med-id",   "MED/USDT",   "2020-01-01", None),
        "SMALL": CoinMeta("small-id", "SMALL/USDT", "2020-01-01", None),
        "DEAD":  CoinMeta("dead-id",  "DEAD/USDT",  "2020-01-01", "2020-05-01"),
        "FAKE":  CoinMeta("fake-id",  "USDT/USDT",  "2020-01-01", None,
                          notes="placeholder stablecoin entry"),
    }


def _panels(n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (market_caps, volumes) with deliberate rank order:
    BIG > MED > SMALL > DEAD on both axes throughout."""
    idx = _date_index(n)
    market_caps = pd.DataFrame(
        {
            "BIG":   np.full(n, 1e11),
            "MED":   np.full(n, 1e10),
            "SMALL": np.full(n, 1e9),
            "DEAD":  np.full(n, 5e9),
            "FAKE":  np.full(n, 1e12),  # large cap on purpose: tests stablecoin filter
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {
            "BIG":   np.full(n, 5e9),
            "MED":   np.full(n, 2e9),
            "SMALL": np.full(n, 1e8),
            "DEAD":  np.full(n, 1e9),
            "FAKE":  np.full(n, 1e10),
        },
        index=idx,
    )
    return market_caps, volumes


def test_nan_metric_coin_not_eligible_in_small_universe():
    """When fewer than top_n coins have a valid market cap/volume,
    na_option='bottom' assigns NaN cells a rank <= top_n, so a tradable
    coin with a missing (NaN) market cap was marked eligible without its
    cap ever being verified as top-N (regression: C12)."""
    idx = _date_index(120)
    registry = {
        "BIG":   CoinMeta("big-id",   "BIG/USDT",   "2020-01-01", None),
        "NOCAP": CoinMeta("nocap-id", "NOCAP/USDT", "2020-01-01", None),
    }
    market_caps = pd.DataFrame(
        {"BIG": np.full(120, 1e11), "NOCAP": np.full(120, np.nan)},
        index=idx,
    )
    volumes = pd.DataFrame(
        {"BIG": np.full(120, 5e9), "NOCAP": np.full(120, 5e8)},
        index=idx,
    )
    eligibility = build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
    )
    assert eligibility["BIG"].iloc[-1] == True
    # NOCAP's market cap is unknown (NaN) — it must NOT be eligible even
    # though the tiny universe leaves its NaN rank numerically <= top_n.
    assert eligibility["NOCAP"].iloc[-1] == False


def test_top_n_picks_largest_caps_when_tradable():
    market_caps, volumes = _panels(200)
    eligibility = build_pit_universe(
        market_caps, volumes,
        candidates=_registry(),
        top_n=2,
        volume_lookback_days=30,
        exclude_stablecoins=False,
    )
    # Top-2 by both metrics, with FAKE excluded as stablecoin elsewhere.
    # With exclude_stablecoins=False, FAKE has the biggest cap and second
    # biggest volume; BIG also makes top-2 on both.
    assert eligibility["FAKE"].iloc[-1] is np.True_ or eligibility["FAKE"].iloc[-1] == True
    assert eligibility["BIG"].iloc[-1] == True
    # MED has rank-3 on cap (after FAKE+BIG) AND rank-3 on volume — out.
    assert eligibility["MED"].iloc[-1] == False
    assert eligibility["SMALL"].iloc[-1] == False
    assert eligibility["DEAD"].iloc[-1] == False  # delisted in this slice


def test_excluded_stablecoin_never_eligible():
    """USDT/USDT is the synthetic FAKE entry; its base symbol is USDT
    so the stablecoin filter must zero its eligibility regardless of
    market cap or volume."""
    market_caps, volumes = _panels(200)
    eligibility = build_pit_universe(
        market_caps, volumes,
        candidates=_registry(),
        top_n=5,
        volume_lookback_days=30,
        exclude_stablecoins=True,
    )
    assert (eligibility["FAKE"] == False).all()
    # USDT is in the stablecoin denylist by default.
    assert "USDT" in stablecoins()


def test_delisted_pair_loses_eligibility_after_delisting_date():
    """DEAD listed 2020-01-01, delisted 2020-05-01. Before delisting it
    has both top-N rank and tradability, so eligibility = True. After,
    eligibility must be False even if its synthetic cap/volume stayed
    high."""
    market_caps, volumes = _panels(200)
    eligibility = build_pit_universe(
        market_caps, volumes,
        candidates=_registry(),
        top_n=5,
        volume_lookback_days=30,
        exclude_stablecoins=True,
    )
    cutoff = pd.Timestamp("2020-05-01", tz="UTC")
    pre = eligibility["DEAD"][eligibility.index < cutoff]
    post = eligibility["DEAD"][eligibility.index >= cutoff]
    # Pre: cap is rank-3, vol is rank-2 — top-5 on both → eligible.
    assert (pre == True).any()
    # Post: tradable_at returns False, so eligibility must be False.
    assert (post == False).all()


def test_warm_up_period_for_listing_date():
    """A coin listed mid-window must have eligibility False before its
    listing date even if its later cap/volume would qualify."""
    n = 200
    idx = _date_index(n, start="2020-01-01")
    market_caps = pd.DataFrame(
        {
            "BTC":  np.full(n, 1e11),
            "LATE": np.full(n, 9e10),
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {
            "BTC":  np.full(n, 5e9),
            "LATE": np.full(n, 4e9),
        },
        index=idx,
    )
    registry = {
        "BTC":  CoinMeta("bitcoin", "BTC/USDT", "2020-01-01", None),
        "LATE": CoinMeta("late",    "LATE/USDT", "2020-03-15", None),
    }
    eligibility = build_pit_universe(
        market_caps, volumes,
        candidates=registry,
        top_n=2,
        volume_lookback_days=30,
        exclude_stablecoins=False,
    )
    cutoff = pd.Timestamp("2020-03-15", tz="UTC")
    assert (eligibility["LATE"][eligibility.index < cutoff] == False).all()
    assert (eligibility["LATE"][eligibility.index >= cutoff] == True).all()


def test_composite_rank_requires_top_n_on_both_axes():
    """A coin top-N on market cap but bottom on volume (or vice versa)
    must be ineligible — composite is AND, not OR."""
    n = 100
    idx = _date_index(n)
    market_caps = pd.DataFrame(
        {
            "A": np.full(n, 1e11),  # cap rank 1, vol rank 3 — must be out for top_n=2
            "B": np.full(n, 1e10),  # cap rank 2, vol rank 1
            "C": np.full(n, 1e9),   # cap rank 3, vol rank 2
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {
            "A": np.full(n, 1e8),
            "B": np.full(n, 1e10),
            "C": np.full(n, 5e9),
        },
        index=idx,
    )
    registry = {
        sym: CoinMeta(f"{sym.lower()}-id", f"{sym}/USDT", "2020-01-01", None)
        for sym in ("A", "B", "C")
    }
    eligibility = build_pit_universe(
        market_caps, volumes,
        candidates=registry,
        top_n=2,
        volume_lookback_days=10,
        exclude_stablecoins=False,
    )
    # A is cap rank 1 but vol rank 3 — fails the AND.
    # B is cap rank 2 and vol rank 1 — passes both.
    # C is cap rank 3 and vol rank 2 — fails on cap side.
    # Only B survives the composite top-N filter.
    assert (eligibility["A"] == False).all()
    assert (eligibility["B"] == True).all()
    assert (eligibility["C"] == False).all()


def test_closes_for_universe_masks_out_ineligible_cells():
    """closes_for_universe must NaN-out cells where eligibility is False."""
    n = 50
    idx = _date_index(n)
    prices = pd.DataFrame(
        {
            "X": np.arange(100, 100 + n, dtype=float),
            "Y": np.arange(200, 200 + n, dtype=float),
        },
        index=idx,
    )
    eligibility = pd.DataFrame(
        {
            "X": [True] * n,
            "Y": [True] * (n // 2) + [False] * (n - n // 2),
        },
        index=idx,
    )
    masked = closes_for_universe(prices, eligibility)
    assert masked["X"].notna().all()
    assert masked["Y"].iloc[: n // 2].notna().all()
    assert masked["Y"].iloc[n // 2 :].isna().all()


def test_empty_inputs_return_empty_frame():
    eligibility = build_pit_universe(pd.DataFrame(), pd.DataFrame())
    assert eligibility.empty


# ---------------------------------------------------------------------------
# strict_mcap_dates — fail loud on NaN mcap at a rebalance date (issue #14)
# ---------------------------------------------------------------------------


def _nan_mcap_setup(n: int = 120):
    idx = _date_index(n)
    registry = {
        "BIG":   CoinMeta("big-id",   "BIG/USDT",   "2020-01-01", None),
        "NOCAP": CoinMeta("nocap-id", "NOCAP/USDT", "2020-01-01", None),
    }
    market_caps = pd.DataFrame(
        {"BIG": np.full(n, 1e11), "NOCAP": np.full(n, np.nan)},
        index=idx,
    )
    volumes = pd.DataFrame(
        {"BIG": np.full(n, 5e9), "NOCAP": np.full(n, 5e8)},
        index=idx,
    )
    return registry, market_caps, volumes


def test_strict_mcap_dates_raise_on_nan_for_tradable_coin():
    registry, market_caps, volumes = _nan_mcap_setup()
    with pytest.raises(PITMcapGapError, match="NOCAP @ 2020-02-01") as excinfo:
        build_pit_universe(
            market_caps, volumes, candidates=registry,
            top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
            strict_mcap_dates=["2020-02-01"],
        )
    assert excinfo.value.gaps == [
        ("NOCAP", pd.Timestamp("2020-02-01", tz="UTC"))
    ]


def test_strict_mcap_dates_pass_when_all_observed():
    registry, market_caps, volumes = _nan_mcap_setup()
    market_caps["NOCAP"] = 1e9  # gap repaired
    strict = build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
        strict_mcap_dates=["2020-02-01", "2020-03-01"],
    )
    loose = build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
    )
    pd.testing.assert_frame_equal(strict, loose)


def test_strict_mcap_date_without_panel_row_raises():
    registry, market_caps, volumes = _nan_mcap_setup()
    market_caps["NOCAP"] = 1e9
    with pytest.raises(PITMcapGapError, match="no panel row"):
        build_pit_universe(
            market_caps, volumes, candidates=registry,
            top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
            strict_mcap_dates=["2030-01-01"],
        )


def test_default_none_keeps_silent_ineligibility_for_other_dates():
    """Without strict_mcap_dates the legacy behavior stays: NaN mcap only
    makes the coin ineligible; NaN on a non-strict date never raises."""
    registry, market_caps, volumes = _nan_mcap_setup()
    eligibility = build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
    )
    assert (eligibility["NOCAP"] == False).all()
    # NaN outside the strict set is likewise tolerated.
    market_caps["NOCAP"] = 1e9
    market_caps.loc[market_caps.index[5], "BIG"] = np.nan
    build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
        strict_mcap_dates=["2020-02-01"],
    )


def test_strict_mcap_dates_ignore_non_tradable_and_stablecoins():
    """Delisted pairs and excluded stablecoins are not rank candidates —
    their NaN mcap on a strict date must not raise."""
    n = 200
    idx = _date_index(n)
    registry = _registry()
    market_caps = pd.DataFrame(
        {
            "BIG":   np.full(n, 1e11),
            "MED":   np.full(n, 1e10),
            "SMALL": np.full(n, 1e9),
            "DEAD":  np.full(n, np.nan),   # delisted 2020-05-01
            "FAKE":  np.full(n, np.nan),   # stablecoin, excluded
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {c: np.full(n, 1e9) for c in market_caps.columns}, index=idx
    )
    # 2020-06-01 is past DEAD's delisting; FAKE is filtered as stablecoin.
    build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=5, volume_lookback_days=30, exclude_stablecoins=True,
        strict_mcap_dates=["2020-06-01"],
    )
    # Before delisting DEAD is tradable, so its NaN must raise.
    with pytest.raises(PITMcapGapError, match="DEAD @ 2020-03-01"):
        build_pit_universe(
            market_caps, volumes, candidates=registry,
            top_n=5, volume_lookback_days=30, exclude_stablecoins=True,
            strict_mcap_dates=["2020-03-01"],
        )


def test_strict_mcap_dates_on_empty_panel_raise():
    with pytest.raises(PITMcapGapError):
        build_pit_universe(
            pd.DataFrame(), pd.DataFrame(),
            strict_mcap_dates=["2020-01-01"],
        )


def test_strict_mcap_dates_catch_candidate_absent_from_panel():
    """A tradable pool candidate with no panel column at all must gap —
    the column restriction must not drop it before validation."""
    registry, market_caps, volumes = _nan_mcap_setup()
    market_caps["NOCAP"] = 1e9
    registry["GHOST"] = CoinMeta("ghost-id", "GHOST/USDT", "2020-01-01", None)
    with pytest.raises(PITMcapGapError, match="GHOST @ 2020-02-01") as excinfo:
        build_pit_universe(
            market_caps, volumes, candidates=registry,
            top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
            strict_mcap_dates=["2020-02-01"],
        )
    assert ("GHOST", pd.Timestamp("2020-02-01", tz="UTC")) in excinfo.value.gaps


def test_absent_candidate_untradable_on_strict_dates_passes():
    registry, market_caps, volumes = _nan_mcap_setup()
    market_caps["NOCAP"] = 1e9
    registry["LATE"] = CoinMeta("late-id", "LATE/USDT", "2020-03-15", None)
    build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=False,
        strict_mcap_dates=["2020-02-01"],
    )


def test_absent_stablecoin_candidate_ignored_when_excluded():
    registry, market_caps, volumes = _nan_mcap_setup()
    market_caps["NOCAP"] = 1e9
    registry["USDC"] = CoinMeta("usd-coin", "USDC/USDT", "2019-01-01", None)
    build_pit_universe(
        market_caps, volumes, candidates=registry,
        top_n=20, volume_lookback_days=30, exclude_stablecoins=True,
        strict_mcap_dates=["2020-02-01"],
    )
