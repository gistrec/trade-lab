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
from trade_lab.data.universe import build_pit_universe, closes_for_universe


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
