"""Look-ahead detector (Part B) invariants on synthetic vintages.

Until the forward journal has rows, these tests are the only thing
the detector can be evaluated against — so the offset-detection
logic in particular MUST be exercised here, not deferred."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_lab.config import PRODUCTION_CONFIG
from trade_lab.paper_trading.harness import run_paper_trading_cycle
from trade_lab.paper_trading.journal import HarnessLogRow, append_row
from trade_lab.paper_trading.lookahead_detector import (
    check_journal_for_lookahead,
    classify_signal_match,
    replay_signal_for_vintage,
)


# ---------------------------------------------------------------------------
# classify_signal_match (pure-function classifier)
# ---------------------------------------------------------------------------

def test_classify_match():
    assert classify_signal_match(0.5, 0.5, 0.0) == "match"
    assert classify_signal_match(0.5, 0.5, None) == "match"


def test_classify_offset_1():
    # live equals replay_prev, not replay_last → labeling drift
    assert classify_signal_match(1.0, 0.0, 1.0) == "offset_1_match"
    assert classify_signal_match(0.5, 1.0, 0.5) == "offset_1_match"


def test_classify_random_disagreement():
    # live equals neither replay_last nor replay_prev → potential look-ahead
    assert classify_signal_match(0.5, 0.0, 1.0) == "random_disagreement"
    assert classify_signal_match(1.0, 0.5, 0.0) == "random_disagreement"
    # prev is None and live != last
    assert classify_signal_match(0.5, 0.0, None) == "random_disagreement"
from trade_lab.paper_trading.vintage_store import store_vintage


def _synthetic_candles(asof: date, n_bars: int = 400, seed: int = 0):
    end = pd.Timestamp(asof, tz="UTC")
    idx = pd.date_range(end=end, periods=n_bars, freq="D")
    rng = np.random.default_rng(seed=seed)
    out: dict[str, pd.DataFrame] = {}
    for j, sym in enumerate(PRODUCTION_CONFIG.assets):
        drift = 0.0005 + 0.0001 * j
        vol = 0.04
        log_returns = rng.normal(drift, vol, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        out[sym] = pd.DataFrame(
            {
                "open": close * 0.999, "high": close * 1.003,
                "low": close * 0.997, "close": close,
                "volume": np.full(n_bars, 1.0e6),
            },
            index=idx,
        )
    return out


def _stub_fetcher(panel: dict[str, pd.DataFrame]):
    def fetch(sym: str, n_bars: int, _asof: date) -> pd.DataFrame:
        return panel[sym].iloc[-n_bars:]
    return fetch


# ---------------------------------------------------------------------------
# replay_signal_for_vintage
# ---------------------------------------------------------------------------

def test_replay_reconstructs_signal_from_vintage(tmp_path):
    """A vintage stored by the harness must yield the same signal when
    replayed by the detector."""
    panel = _synthetic_candles(date(2024, 6, 1))
    row = run_paper_trading_cycle(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
        asof=date(2024, 6, 1),
        fetch_callable=_stub_fetcher(panel),
    )
    last, prev = replay_signal_for_vintage(row.vintage_content_hash, tmp_path / "v")
    assert abs(last - row.ladder_state) < 1e-9


# ---------------------------------------------------------------------------
# check_journal_for_lookahead — happy path
# ---------------------------------------------------------------------------

def test_empty_journal_reports_no_rows(tmp_path):
    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_checked == 0
    assert "empty" in report.advisory.lower()
    assert not report.random_disagreement_present


def test_match_when_journal_was_written_by_harness(tmp_path):
    """If we let the harness write rows and then replay, every row
    must match — there is no look-ahead to find."""
    panel = _synthetic_candles(date(2024, 6, 5))
    fetcher = _stub_fetcher(panel)
    for asof in [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 3)]:
        # Use the same panel but trim to asof, so each cycle sees a
        # different vintage
        trimmed = {
            sym: df[df.index <= pd.Timestamp(asof, tz="UTC")]
            for sym, df in panel.items()
        }
        run_paper_trading_cycle(
            log_path=tmp_path / "j.jsonl",
            vintage_root=tmp_path / "v",
            asof=asof,
            fetch_callable=_stub_fetcher(trimmed),
        )
    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_checked == 3
    assert report.n_match == 3
    assert report.n_random_disagreement == 0
    assert report.n_offset_1_match == 0
    assert "match" in report.advisory.lower()


# ---------------------------------------------------------------------------
# Random-disagreement (the load-bearing detector signal)
# ---------------------------------------------------------------------------

def test_random_disagreement_is_flagged(tmp_path):
    """Manually inject a journal row whose ladder_state does NOT match
    the replay signal and does NOT match offset_1 either — this is
    what a real look-ahead in the backtest path would look like."""
    panel = _synthetic_candles(date(2024, 6, 1))
    asset_candles = {sym: df.iloc[-400:] for sym, df in panel.items()}
    vh = store_vintage(asset_candles, tmp_path / "v")
    replay_last, _ = replay_signal_for_vintage(vh, tmp_path / "v")
    # Pick a ladder value that is NEITHER the last nor prev signal
    bogus_ladder = (replay_last + 0.5) % 1.5
    row = HarnessLogRow(
        date="2024-06-01",
        config_hash="x" * 64,
        vintage_content_hash=vh,
        basket_close=100.0, sma_value=99.0, sma_gate_open=True,
        ladder_state=bogus_ladder, prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.01, "60": 0.01},
        target_weights={}, current_weights={}, intended_trades={},
        portfolio_equity=10_000.0, daily_return=0.0,
        gross_position_return=0.0, net_position_return=0.0,
    )
    append_row(row, tmp_path / "j.jsonl")
    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_random_disagreement >= 1
    assert "look-ahead suspect" in report.advisory.lower()


# ---------------------------------------------------------------------------
# Constant-1-bar offset — labeling artifact, NOT look-ahead
# ---------------------------------------------------------------------------

def test_constant_offset_is_classified_as_labeling_not_lookahead(tmp_path):
    """Construct a vintage whose replay signal at [-1] != [-2]. Then
    write a journal row whose ladder_state == replay[-2]. The
    detector should classify this as `offset_1_match` and the
    aggregate advisory must say 'labeling artifact', NOT 'look-ahead'.
    """
    # Build several vintages, each with replay[-1] != replay[-2], so
    # that the offset-classification has signal to lock onto.
    panel = _synthetic_candles(date(2024, 6, 10))
    interesting = []
    for asof in pd.date_range(start="2024-06-05", end="2024-06-09", freq="D"):
        asof_ts = pd.Timestamp(asof.date(), tz="UTC")
        trimmed = {
            sym: df[df.index <= asof_ts] for sym, df in panel.items()
        }
        vh = store_vintage(trimmed, tmp_path / "v")
        last, prev = replay_signal_for_vintage(vh, tmp_path / "v")
        if prev is None or last == prev:
            continue
        # Construct a row where ladder_state == prev (labeling drift)
        row = HarnessLogRow(
            date=asof.strftime("%Y-%m-%d"),
            config_hash="x" * 64,
            vintage_content_hash=vh,
            basket_close=100.0, sma_value=99.0, sma_gate_open=True,
            ladder_state=prev, prior_ladder_state=0.0,
            per_lookback_states={"28": 1, "60": 1},
            per_lookback_returns={"28": 0.01, "60": 0.01},
            target_weights={}, current_weights={}, intended_trades={},
            portfolio_equity=10_000.0, daily_return=0.0,
            gross_position_return=0.0, net_position_return=0.0,
        )
        append_row(row, tmp_path / "j.jsonl")
        interesting.append((asof, last, prev))

    if not interesting:
        pytest.skip("Could not synthesize a non-degenerate offset case "
                    "with this random seed; covered elsewhere.")

    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_offset_1_match == len(interesting)
    assert report.n_random_disagreement == 0
    assert report.constant_offset_pattern is True
    assert "labeling artifact" in report.advisory.lower()
    assert "look-ahead" not in report.advisory.lower() or "not a real" in report.advisory.lower()


# ---------------------------------------------------------------------------
# Vintage missing — handled gracefully
# ---------------------------------------------------------------------------

def test_missing_vintage_is_reported_not_crashed(tmp_path):
    row = HarnessLogRow(
        date="2024-06-01",
        config_hash="x" * 64,
        vintage_content_hash="missing" * 9 + "abcd",  # 64 chars, not on disk
        basket_close=100.0, sma_value=99.0, sma_gate_open=True,
        ladder_state=0.5, prior_ladder_state=0.0,
        per_lookback_states={"28": 1, "60": 1},
        per_lookback_returns={"28": 0.01, "60": 0.01},
        target_weights={}, current_weights={}, intended_trades={},
        portfolio_equity=10_000.0, daily_return=0.0,
        gross_position_return=0.0, net_position_return=0.0,
    )
    append_row(row, tmp_path / "j.jsonl")
    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_vintage_missing == 1
    assert "vintages missing" in report.advisory.lower() or "missing" in report.advisory.lower()


# ---------------------------------------------------------------------------
# Cold-start: detector MUST NOT fabricate a verdict on empty input
# ---------------------------------------------------------------------------

def test_detector_does_not_claim_pass_on_empty_journal(tmp_path):
    """The advisory on an empty journal must explicitly point at the
    Part-A (truncation-audit) result as the dispositive test, NOT
    claim 'all rows match' or similar false-positive."""
    report = check_journal_for_lookahead(
        log_path=tmp_path / "j.jsonl",
        vintage_root=tmp_path / "v",
    )
    assert report.n_match == 0
    advisory_lower = report.advisory.lower()
    # MUST reference Part A as the dispositive test
    assert "part a" in advisory_lower
    assert "truncation" in advisory_lower
    # MUST NOT mistakenly say "all match" / "clean" / "no look-ahead"
    assert "all" not in advisory_lower or "match" not in advisory_lower
