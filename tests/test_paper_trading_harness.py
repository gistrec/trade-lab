"""Harness end-to-end + invariants: hash gate, idempotency, vintage hash
recorded in journal, signal matches what backtest would produce on the
same input.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.paper_trading import harness as harness_mod
from trade_lab.paper_trading import (
    HarnessError,
    read_log,
    run_paper_trading_cycle,
)
from trade_lab.paper_trading.vintage_store import vintage_path


def _synthetic_candles(asof: date, n_bars: int = 400) -> dict[str, pd.DataFrame]:
    """Plausible 7-asset OHLCV data ending at ``asof`` (one bar per day,
    UTC). Different per-asset price trajectories so the basket is
    non-degenerate."""
    end = pd.Timestamp(asof, tz="UTC")
    idx = pd.date_range(end=end, periods=n_bars, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=42)
    out: dict[str, pd.DataFrame] = {}
    for j, sym in enumerate(PRODUCTION_CONFIG.assets):
        # Geometric brownian-ish walk; drift varies by asset
        drift = 0.0005 + 0.0001 * j
        vol = 0.04
        log_returns = rng.normal(drift, vol, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        out[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.003,
                "low": close * 0.997,
                "close": close,
                "volume": np.full(n_bars, 1.0e6),
            },
            index=idx,
        )
    return out


def _stub_fetcher(asof: date):
    """A deterministic stub that any test can use; never hits ccxt."""
    panel = _synthetic_candles(asof)

    def fetch(sym: str, n_bars: int, _asof: date) -> pd.DataFrame:
        return panel[sym].iloc[-n_bars:]

    return fetch


def test_hash_gate_refuses_on_drift(monkeypatch, tmp_path):
    """If CANONICAL_HASH does not match the runtime config hash, the
    harness must refuse to run. This is the contract."""
    # Pretend the canonical hash is something else
    monkeypatch.setattr(harness_mod, "CANONICAL_HASH", "0" * 64)
    with pytest.raises(HarnessError, match="hash drift"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl",
            vintage_root=tmp_path / "v",
            asof=date(2024, 6, 1),
            fetch_callable=_stub_fetcher(date(2024, 6, 1)),
        )


def test_hash_gate_passes_with_canonical_hash(tmp_path):
    """Sanity: the in-repo CANONICAL_HASH is the one the harness reads,
    so unmocked execution must NOT raise the drift error."""
    run_paper_trading_cycle(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
        asof=date(2024, 6, 1),
        fetch_callable=_stub_fetcher(date(2024, 6, 1)),
    )


def test_writes_one_row_per_day(tmp_path):
    asof = date(2024, 6, 1)
    log = tmp_path / "j.jsonl"
    run_paper_trading_cycle(
        log_path=log,
        vintage_root=tmp_path / "v",
        asof=asof,
        fetch_callable=_stub_fetcher(asof),
    )
    rows = read_log(log)
    assert len(rows) == 1
    assert rows[0].date == "2024-06-01"
    assert rows[0].config_hash == CANONICAL_HASH


def _constant_growth_fetcher(end: date, total_bars: int = 500, daily: float = 0.01):
    """All 7 assets identical, growing a constant `daily` per bar. The
    equal-weight basket then grows exactly `daily` per bar with no
    rebalance turnover, so the true one-day basket return is known."""
    idx = pd.date_range(
        end=pd.Timestamp(end, tz="UTC"), periods=total_bars, freq="D",
    )
    close = 100.0 * (1.0 + daily) ** np.arange(total_bars)
    frame = pd.DataFrame(
        {
            "open": close, "high": close * 1.0001,
            "low": close * 0.9999, "close": close,
            "volume": np.full(total_bars, 1.0e6),
        },
        index=idx,
    )
    panel = {sym: frame.copy() for sym in PRODUCTION_CONFIG.assets}

    def fetch(sym: str, n_bars: int, asof: date) -> pd.DataFrame:
        df = panel[sym]
        df = df[df.index <= pd.Timestamp(asof, tz="UTC")]
        return df.iloc[-n_bars:]

    return fetch


def _seed_row(date_str: str, ladder: float):
    from trade_lab.paper_trading.journal import HarnessLogRow
    assets = PRODUCTION_CONFIG.assets
    return HarnessLogRow(
        date=date_str, config_hash=CANONICAL_HASH, vintage_content_hash="seed",
        basket_close=100.0, sma_value=90.0, sma_gate_open=True,
        ladder_state=ladder, prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.1, "60": 0.1},
        target_weights={s: ladder / len(assets) for s in assets},
        current_weights={s: 0.0 for s in assets},
        intended_trades={s: 0.0 for s in assets},
        portfolio_equity=10_000.0, daily_return=0.0,
        gross_position_return=0.0, net_position_return=0.0,
    )


def test_backfill_chains_off_earlier_row_not_future_row(tmp_path):
    """An --asof backfill of a missed earlier date must chain off the most
    recent row BEFORE it, not the last physically appended (future) row.
    Selecting by insertion order chained the return backwards in time
    (regression: C14)."""
    from trade_lab.paper_trading.journal import append_row

    log = tmp_path / "j.jsonl"
    append_row(_seed_row("2024-06-01", 0.5), log)
    append_row(_seed_row("2024-06-03", 1.0), log)  # a LATER date already logged

    row = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=date(2024, 6, 2),
        fetch_callable=_stub_fetcher(date(2024, 6, 2)),
    )
    assert row.date == "2024-06-02"
    # Must chain off 2024-06-01 (ladder 0.5), not the future 2024-06-03 (1.0).
    assert row.prior_ladder_state == 0.5


def test_daily_return_is_within_window_not_cross_window(tmp_path):
    """daily_return must be the basket's one-day return within a single
    normalized window, NOT prior_row.basket_close / current basket_close.
    build_crypto_market_index renormalizes to 100 at each window's first
    bar, so consecutive cycles anchor differently and the cross-window
    ratio collapses a real +1%/day basket to ~0% (regression: C4)."""
    d1 = date(2024, 6, 1)
    d2 = date(2024, 6, 2)
    log = tmp_path / "j.jsonl"
    fetch = _constant_growth_fetcher(end=d2, daily=0.01)

    row1 = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=d1, fetch_callable=fetch,
    )
    row2 = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=d2, fetch_callable=fetch,
    )
    assert row1.daily_return == 0.0            # first cycle: no prior
    # True one-day basket return is +1%; the cross-window bug reports ~0%.
    assert row2.daily_return == pytest.approx(0.01, abs=1e-4)


def test_idempotent_when_run_twice_on_same_date(tmp_path):
    """Re-invocation on the same UTC date must NOT write a duplicate
    row — the cron job is allowed to fire multiple times per day."""
    asof = date(2024, 6, 1)
    log = tmp_path / "j.jsonl"
    fetcher = _stub_fetcher(asof)
    row1 = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=asof,
        fetch_callable=fetcher,
    )
    row2 = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=asof,
        fetch_callable=fetcher,
    )
    assert row1.date == row2.date
    assert row1.vintage_content_hash == row2.vintage_content_hash
    assert row1.ladder_state == row2.ladder_state
    assert len(read_log(log)) == 1


def test_vintage_file_is_written_and_hash_matches_journal(tmp_path):
    """The journal's vintage_content_hash must point at a file that
    exists and whose content actually hashes to that value — this is
    the look-ahead-detector contract."""
    asof = date(2024, 6, 1)
    log = tmp_path / "j.jsonl"
    vroot = tmp_path / "v"
    row = run_paper_trading_cycle(
        log_path=log, vintage_root=vroot, asof=asof,
        fetch_callable=_stub_fetcher(asof),
    )
    p = vintage_path(vroot, row.vintage_content_hash)
    assert p.exists()
    # Independently verify
    import hashlib
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row.vintage_content_hash


def test_empty_candles_raise(tmp_path):
    """Empty fetch is a fail-loud condition, not a "log partial basket"."""
    def empty_fetch(sym, n, asof):
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    with pytest.raises(HarnessError, match="Empty candles"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl",
            vintage_root=tmp_path / "v",
            asof=date(2024, 6, 1),
            fetch_callable=empty_fetch,
        )


def test_fetch_exception_raises_harness_error(tmp_path):
    def broken_fetch(sym, n, asof):
        raise RuntimeError("network down")
    with pytest.raises(HarnessError, match="network down"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl",
            vintage_root=tmp_path / "v",
            asof=date(2024, 6, 1),
            fetch_callable=broken_fetch,
        )


def test_signal_matches_backtest_on_identical_input(tmp_path):
    """The harness's ladder_state must equal what
    TimeSeriesMomentumStrategy + build_crypto_market_index would
    produce on the same input. If they ever diverge it would mean
    paper trading is running a different signal than the backtest —
    the precise failure mode the look-ahead detector is supposed to
    catch.
    """
    asof = date(2024, 6, 1)
    panel = _synthetic_candles(asof)

    # Harness path
    row = run_paper_trading_cycle(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
        asof=asof,
        fetch_callable=lambda s, n, _: panel[s].iloc[-n:],
    )

    # Reference path: build basket and strategy directly
    from trade_lab.backtest.market_index import build_crypto_market_index
    from trade_lab.strategies.tsmom import TimeSeriesMomentumStrategy
    cfg = PRODUCTION_CONFIG
    basket = build_crypto_market_index(
        {s: panel[s] for s in cfg.assets},
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )
    strat = TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        annualization_factor=cfg.annualization_factor,
    )
    ref_signal = float(strat.generate_signals(basket).iloc[-1])
    assert row.ladder_state == ref_signal


def test_target_weights_sum_equals_ladder(tmp_path):
    """target_weights must sum to ladder_state (1/N equal-weight × ladder)."""
    asof = date(2024, 6, 1)
    row = run_paper_trading_cycle(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
        asof=asof,
        fetch_callable=_stub_fetcher(asof),
    )
    assert pytest.approx(sum(row.target_weights.values()), rel=1e-9) == row.ladder_state
    assert sorted(row.target_weights.keys()) == sorted(PRODUCTION_CONFIG.assets)


# ---------------------------------------------------------------------------
# H5 — look-ahead window regressions: the fetch window must never include
# a bar after the cycle's signal date (completed T+1 bar on a backfill,
# the still-forming bar on a same-day run).
# ---------------------------------------------------------------------------


class _FrozenDatetime(datetime):
    """``datetime`` replacement whose ``now()`` is pinned to
    2024-06-05 12:00 UTC, so the same-day tests are deterministic and
    immune to running across a real UTC midnight."""

    @classmethod
    def now(cls, tz=None):
        return cls(2024, 6, 5, 12, 0, tzinfo=tz)


def _contract_fetcher(panel, seen: dict | None = None):
    """A fetcher honoring the harness contract: bars up to and including
    the ``asof`` argument the harness passes in. Optionally records that
    argument in ``seen``."""

    def fetch(sym: str, n_bars: int, asof: date) -> pd.DataFrame:
        if seen is not None:
            seen["cutoff"] = asof
        df = panel[sym]
        return df[df.index <= pd.Timestamp(asof, tz="UTC")].iloc[-n_bars:]

    return fetch


def test_default_fetch_window_excludes_bars_after_asof(monkeypatch):
    """_default_fetch must bound the window at the completed bar dated
    ``asof``. fetch_ohlcv's ``until`` filter is INCLUSIVE and daily bars
    are stamped at their open, so the old ``until = asof + 1 day`` also
    matched the bar stamped ``asof + 1`` — the H5 1-bar look-ahead."""

    def fake_fetch_ohlcv(exchange_id, symbol, timeframe, since, until, limit):
        # Bars past `asof` exist; replicate the real fetch_ohlcv's
        # INCLUSIVE `until` cut so the test pins the composed semantics
        # (bound value × inclusive filter).
        idx = pd.date_range("2024-01-01", "2024-06-05", freq="D", tz="UTC")
        close = np.linspace(100.0, 200.0, len(idx))
        df = pd.DataFrame(
            {
                "open": close, "high": close * 1.001,
                "low": close * 0.999, "close": close,
                "volume": np.full(len(idx), 1.0e6),
            },
            index=idx,
        ).rename_axis("timestamp")
        return df[df.index <= pd.Timestamp(until, tz="UTC")]

    monkeypatch.setattr(harness_mod, "fetch_ohlcv", fake_fetch_ohlcv)
    asof = date(2024, 6, 1)
    df = harness_mod._default_fetch("BTC", 10, asof)
    assert df.index[-1].date() == asof            # completed bar T included
    assert not (df.index.date > asof).any()       # nothing beyond T


def test_backfill_with_completed_next_day_bar_excludes_it(tmp_path):
    """--asof T backfill while the completed T+1 bar already exists (the
    documented missed-cron recovery): T+1 must NOT participate — the row
    (signal, vintage, basket_close) must be identical to one computed on
    a panel that simply ends at T, and the guard must stay silent."""
    asof = date(2024, 6, 1)
    panel = _synthetic_candles(date(2024, 6, 2))  # ends at T+1
    # Poison the T+1 bar so any leak flips lookbacks/SMA loudly.
    for sym in panel:
        panel[sym].iloc[-1] = panel[sym].iloc[-1] * 5.0

    row = run_paper_trading_cycle(
        log_path=tmp_path / "a.jsonl", vintage_root=tmp_path / "va",
        asof=asof, fetch_callable=_contract_fetcher(panel),
    )

    ref_panel = {
        s: df[df.index <= pd.Timestamp(asof, tz="UTC")]
        for s, df in panel.items()
    }
    ref = run_paper_trading_cycle(
        log_path=tmp_path / "b.jsonl", vintage_root=tmp_path / "vb",
        asof=asof, fetch_callable=lambda s, n, _a: ref_panel[s].iloc[-n:],
    )
    assert row.date == "2024-06-01"
    assert row.vintage_content_hash == ref.vintage_content_hash
    assert row.ladder_state == ref.ladder_state
    assert row.basket_close == ref.basket_close
    assert row.per_lookback_returns == ref.per_lookback_returns


def test_lookahead_guard_raises_on_bar_beyond_signal_date(tmp_path):
    """A fetcher that bypasses the window and returns a bar dated after
    the signal date must trip the hard guard BEFORE anything is
    snapshotted or journaled (future filter regressions fail loud)."""
    asof = date(2024, 6, 1)
    panel = _synthetic_candles(date(2024, 6, 2))  # completed T+1 exists

    def leaky_fetch(sym, n_bars, _asof):
        return panel[sym].iloc[-n_bars:]  # ignores the cutoff → leaks T+1

    log = tmp_path / "j.jsonl"
    vroot = tmp_path / "v"
    with pytest.raises(HarnessError, match="Look-ahead guard"):
        run_paper_trading_cycle(
            log_path=log, vintage_root=vroot, asof=asof,
            fetch_callable=leaky_fetch,
        )
    assert not log.exists()                                   # no journal row
    assert not vroot.exists() or not any(vroot.rglob("*"))    # no vintage


def test_same_day_run_uses_last_completed_bar(tmp_path, monkeypatch):
    """Default (cron) invocation during day D: the still-forming bar D
    must not participate; the row is computed on — and dated by — the
    last completed bar D-1. Same convention as the live path
    (execution.signal drops the forming bar) and the backtest
    (signal[T] is a function of the completed close[T])."""
    monkeypatch.setattr(harness_mod, "datetime", _FrozenDatetime)
    today = date(2024, 6, 5)
    yesterday = date(2024, 6, 4)
    panel = _synthetic_candles(today)  # last bar stamped today == forming
    seen: dict = {}

    log = tmp_path / "j.jsonl"
    row = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=None,
        fetch_callable=_contract_fetcher(panel, seen),
    )
    assert seen["cutoff"] == yesterday            # forming bar cut at source
    assert row.date == yesterday.isoformat()      # row keyed by signal date
    # Idempotent re-run within the same UTC day returns the same row.
    row2 = run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=None,
        fetch_callable=_contract_fetcher(panel),
    )
    assert row2.date == row.date
    assert len(read_log(log)) == 1


def test_same_day_forming_bar_leak_trips_guard(tmp_path, monkeypatch):
    """If a fetcher leaks the forming current-day bar on a same-day run,
    the guard must raise rather than let intraday noise into the row."""
    monkeypatch.setattr(harness_mod, "datetime", _FrozenDatetime)
    panel = _synthetic_candles(date(2024, 6, 5))  # forming bar included

    def leaky_fetch(sym, n_bars, _asof):
        return panel[sym].iloc[-n_bars:]

    with pytest.raises(HarnessError, match="Look-ahead guard"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl", vintage_root=tmp_path / "v",
            asof=None, fetch_callable=leaky_fetch,
        )


def test_future_asof_raises(tmp_path, monkeypatch):
    """A future --asof cannot be computed on completed bars — hard error,
    not a silent window into whatever the fetcher returns."""
    monkeypatch.setattr(harness_mod, "datetime", _FrozenDatetime)
    tomorrow = date(2024, 6, 6)
    with pytest.raises(HarnessError, match="future"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl", vintage_root=tmp_path / "v",
            asof=tomorrow, fetch_callable=_stub_fetcher(tomorrow),
        )


def test_stale_candles_raise_instead_of_mislabeling(tmp_path):
    """Candles ending BEFORE the signal date must raise: journaling a row
    dated T computed on older bars would silently break the date-join
    with the backtest (fail loud, not silent)."""
    asof = date(2024, 6, 5)
    panel = _synthetic_candles(date(2024, 6, 1))  # 4 days stale

    with pytest.raises(HarnessError, match="Basket ends"):
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl", vintage_root=tmp_path / "v",
            asof=asof, fetch_callable=lambda s, n, _a: panel[s].iloc[-n:],
        )


def test_journal_row_round_trips_through_disk(tmp_path):
    """One full write+read cycle to make sure the schema is stable."""
    asof = date(2024, 6, 1)
    log = tmp_path / "j.jsonl"
    run_paper_trading_cycle(
        log_path=log, vintage_root=tmp_path / "v", asof=asof,
        fetch_callable=_stub_fetcher(asof),
    )
    rows = read_log(log)
    assert len(rows) == 1
    r = rows[0]
    # Spot-check structural fields
    assert r.date == "2024-06-01"
    assert isinstance(r.per_lookback_states, dict)
    assert "28" in r.per_lookback_states
    assert "60" in r.per_lookback_states
    assert isinstance(r.target_weights, dict)
    assert isinstance(r.intended_trades, dict)
