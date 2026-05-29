"""Harness end-to-end + invariants: hash gate, idempotency, vintage hash
recorded in journal, signal matches what backtest would produce on the
same input.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_lab.config import CANONICAL_HASH, PRODUCTION_CONFIG
from trade_lab.paper_trading import harness as harness_mod
from trade_lab.paper_trading import (
    HarnessError,
    is_already_logged,
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
