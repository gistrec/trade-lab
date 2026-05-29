"""Reference fingerprint + monitor invariants."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from trade_lab.paper_trading.fingerprint import (
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_ROLLING_WINDOW_DAYS,
    compute_reference_fingerprint,
    fingerprint_content_hash,
    load_reference,
    save_reference,
)
from trade_lab.paper_trading.fingerprint_monitor import (
    check_journal_against_reference,
    compute_live_metrics_from_journal,
)
from trade_lab.paper_trading.journal import HarnessLogRow, append_row


def _synthetic_series(n: int = 1000, seed: int = 0):
    """Synthetic basket close, positions, equity, sma — enough structure
    that fingerprint metrics are non-degenerate."""
    idx = pd.date_range("2022-01-21", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.03, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=idx)
    # Positions flip every ~30 bars with values from the ladder
    pos = pd.Series(0.0, index=idx)
    for i in range(n):
        if i % 30 == 0:
            pos.iloc[i:] = rng.choice([0.0, 0.5, 1.0])
    # Equity: positions × close returns, compounded
    bar_returns = close.pct_change().fillna(0.0)
    port_ret = pos.shift(1).fillna(0.0) * bar_returns
    equity = (1.0 + port_ret).cumprod() * 10_000.0
    sma = close.rolling(200).mean()
    return close, pos, equity, sma, idx


# ---------------------------------------------------------------------------
# fingerprint.py
# ---------------------------------------------------------------------------

def test_fingerprint_construction_yields_well_formed_bands():
    close, pos, equity, sma, idx = _synthetic_series()
    fp = compute_reference_fingerprint(
        basket_close=close, positions=pos, equity=equity, sma_series=sma,
        window_start=idx[0], window_end=idx[-1],
        frozen_config_hash="x" * 64,
    )
    # Sanity: required structure
    assert fp.schema_version == "v1"
    assert fp.frozen_config_hash == "x" * 64
    assert fp.n_bars > 0
    for band in (fp.exposure_flip_freq_rolling,
                 fp.regime_gate_flip_freq_rolling,
                 fp.rebalance_turnover_per_event):
        assert set(band.percentiles.keys()) == {"p05", "p25", "p50", "p75", "p95"}
        assert band.extremes["min"] <= band.percentiles["p05"]
        assert band.extremes["max"] >= band.percentiles["p95"]
    # Drawdown is non-positive
    dd = fp.drawdown_profile
    assert dd.max_historical_dd <= 0.0
    # content_hash is set
    assert len(fp.content_hash) == 64


def test_fingerprint_is_deterministic_for_same_inputs():
    a = _synthetic_series(seed=42)
    b = _synthetic_series(seed=42)
    fp_a = compute_reference_fingerprint(
        basket_close=a[0], positions=a[1], equity=a[2], sma_series=a[3],
        window_start=a[4][0], window_end=a[4][-1],
        frozen_config_hash="cafe" * 16,
    )
    fp_b = compute_reference_fingerprint(
        basket_close=b[0], positions=b[1], equity=b[2], sma_series=b[3],
        window_start=b[4][0], window_end=b[4][-1],
        frozen_config_hash="cafe" * 16,
    )
    assert fp_a.content_hash == fp_b.content_hash


def test_fingerprint_save_load_round_trip(tmp_path):
    close, pos, equity, sma, idx = _synthetic_series()
    fp = compute_reference_fingerprint(
        basket_close=close, positions=pos, equity=equity, sma_series=sma,
        window_start=idx[0], window_end=idx[-1],
        frozen_config_hash="x" * 64,
    )
    p = tmp_path / "ref.json"
    save_reference(fp, p)
    loaded = load_reference(p)
    assert loaded.content_hash == fp.content_hash
    assert loaded.n_bars == fp.n_bars


def test_fingerprint_load_raises_on_hash_mismatch(tmp_path):
    """Editing the JSON after writing must be caught at load time."""
    close, pos, equity, sma, idx = _synthetic_series()
    fp = compute_reference_fingerprint(
        basket_close=close, positions=pos, equity=equity, sma_series=sma,
        window_start=idx[0], window_end=idx[-1],
        frozen_config_hash="x" * 64,
    )
    p = tmp_path / "ref.json"
    save_reference(fp, p)
    # Tamper: bump n_bars without recomputing hash
    data = json.loads(p.read_text())
    data["n_bars"] = data["n_bars"] + 1
    p.write_text(json.dumps(data, indent=2))
    with pytest.raises(ValueError, match="content-hash mismatch"):
        load_reference(p)


# ---------------------------------------------------------------------------
# fingerprint_monitor.py
# ---------------------------------------------------------------------------

def _ref_to_tmp(tmp_path, **kwargs):
    close, pos, equity, sma, idx = _synthetic_series()
    fp = compute_reference_fingerprint(
        basket_close=close, positions=pos, equity=equity, sma_series=sma,
        window_start=idx[0], window_end=idx[-1],
        frozen_config_hash="x" * 64,
        **kwargs,
    )
    p = tmp_path / "ref.json"
    save_reference(fp, p)
    return p, fp


def _journal_row(date_iso: str, ladder: float, equity: float, sma_open: bool = True):
    return HarnessLogRow(
        date=date_iso, config_hash="x" * 64, vintage_content_hash="y" * 64,
        basket_close=100.0, sma_value=99.0, sma_gate_open=sma_open,
        ladder_state=ladder, prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.01, "60": 0.01},
        target_weights={"BTC": ladder/7},
        current_weights={"BTC": 0.0},
        intended_trades={"BTC": ladder/7},
        portfolio_equity=equity,
        daily_return=0.0, gross_position_return=0.0, net_position_return=0.0,
    )


def test_monitor_empty_journal(tmp_path):
    ref_p, _ = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    report = check_journal_against_reference(log_p, ref_p)
    assert report.journal_n_rows == 0
    assert "empty" in report.advisory.lower()


def test_monitor_bootstrap_phase(tmp_path):
    """Fewer journal bars than the rolling-window length → bootstrap."""
    ref_p, fp = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    # Write 10 days
    for i in range(10):
        append_row(_journal_row(f"2026-05-{i+1:02d}", 0.5, 10_000.0), log_p)
    report = check_journal_against_reference(log_p, ref_p)
    assert report.journal_n_rows == 10
    assert "bootstrap" in report.advisory.lower()


def test_monitor_within_envelope(tmp_path):
    """Lots of bars with behavior INSIDE the reference bands → green."""
    ref_p, fp = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    # Use the same synthetic series the reference was built on for the
    # last 200 bars → live metrics will sit inside the bands by construction.
    close, pos, equity, sma, idx = _synthetic_series()
    for i in range(200):
        append_row(_journal_row(
            idx[i].strftime("%Y-%m-%d"),
            float(pos.iloc[i]),
            float(equity.iloc[i]),
        ), log_p)
    report = check_journal_against_reference(log_p, ref_p)
    assert report.journal_n_rows == 200
    assert not report.drawdown.breached
    assert "envelope" in report.advisory.lower() or "noise" in report.advisory.lower()


def test_monitor_drawdown_breach(tmp_path):
    """A live drawdown deeper than max_historical_dd must flip the
    drawdown breach flag."""
    ref_p, fp = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    # Build a journal that crashes equity hard
    n = 250
    for i in range(n):
        # Equity rises for half the days then crashes 80%
        eq = 10_000.0 * (1.5 ** (i / n)) if i < n / 2 else 2_000.0
        append_row(_journal_row(
            (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            0.5, eq,
        ), log_p)
    report = check_journal_against_reference(log_p, ref_p)
    assert report.drawdown.breached, (
        f"Expected drawdown breach: max_live_dd={report.drawdown.max_live_drawdown}, "
        f"reference_max_dd={report.drawdown.reference_max_historical_dd}"
    )
    assert "drawdown breach" in report.advisory.lower()


def test_monitor_sustained_metric_breach(tmp_path):
    """A live exposure-flip rate sustained above p95 for many days
    must trigger 'sustained breach'."""
    ref_p, fp = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    # 200 days where ladder flips EVERY bar (much higher than ref p95)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(200):
        ladder = 1.0 if i % 2 == 0 else 0.0  # flip every day
        append_row(_journal_row(
            (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            ladder, 10_000.0,
        ), log_p)
    report = check_journal_against_reference(log_p, ref_p)
    assert report.overall_sustained_breach
    assert "sustained" in report.advisory.lower() or "drawdown" in report.advisory.lower()


def test_live_metrics_helper_returns_aligned_series(tmp_path):
    ref_p, fp = _ref_to_tmp(tmp_path)
    log_p = tmp_path / "log.jsonl"
    # 120 days
    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i in range(120):
        row = _journal_row(
            (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            0.5 if i % 5 == 0 else 0.0, 10_000.0 + i * 10.0,
        )
        rows.append(row)
        append_row(row, log_p)
    out = compute_live_metrics_from_journal(
        rows,
        rolling_window_days=DEFAULT_ROLLING_WINDOW_DAYS,
        annualization_factor=DEFAULT_ANNUALIZATION_FACTOR,
    )
    # 120 - 90 + 1 = 31 valid rolling bars
    assert len(out["roll_flip"]) == 31
    assert len(out["roll_gate"]) == 31
    assert len(out["drawdown"]) == 120


def test_monitor_handles_corrupted_reference_loud(tmp_path):
    """Hand-edited reference file must raise at load — descriptive
    fail-loud rather than silently feeding wrong bands."""
    ref_p, fp = _ref_to_tmp(tmp_path)
    data = json.loads(ref_p.read_text())
    data["n_bars"] = 99999
    ref_p.write_text(json.dumps(data, indent=2))
    log_p = tmp_path / "log.jsonl"
    with pytest.raises(ValueError, match="content-hash mismatch"):
        check_journal_against_reference(log_p, ref_p)
