"""Tests for the live-signal computation pipeline.

No network, no real CCXT. Signal computation is deterministic given
candles; we feed canned OHLCV frames through the broker stub and
assert the ladder value (0 / 0.5 / 1.0) and gate diagnostics line up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ccxt

from trade_lab.backtest.market_index import build_crypto_market_index_with_weights
from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.signal import (
    InsufficientWarmupError, MS_REBALANCE_MAX_GAP_DAYS,
    SignalComputationError, SignalSnapshot, compute_live_signal,
    min_live_candles, required_basket_bars,
)


def _config(basket=("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")):
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


class _ExchangeStub:
    """Minimal CCXT stand-in that only needs to satisfy Broker plumbing."""
    id = "stub"
    def set_sandbox_mode(self, enabled): pass
    def fetch_balance(self): return {"USDT": {"free": 0, "used": 0, "total": 0}}
    def fetch_ticker(self, symbol): return {"last": 1.0, "close": 1.0}
    def fetch_status(self): return {"status": "ok"}
    def load_markets(self, reload=False): return {}


# Fixed decision date for deterministic frames: every default frame ENDS at
# _NOW's yesterday, so the asof-freshness guard sees exactly the most
# recently closed bar regardless of when the suite runs. Mid-month on
# purpose — a month-start asof would reset the drifted weights to flat 1/N
# and blind the weight-drift assertions.
_NOW = pd.Timestamp("2024-05-15 00:45:00", tz="UTC")
_END = "2024-05-14"


def _ohlcv(closes, end=_END):
    """Build an OHLCV frame of daily UTC bars ending at ``end``."""
    idx = pd.date_range(end=end, periods=len(closes), freq="1D", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 1.0},
        index=idx,
    )


def _candles_factory(symbol_to_closes):
    """Return a fetch_candles function that emits canned frames."""
    def _fetch(broker, pair, limit):
        sym = pair.split("/")[0]
        closes = symbol_to_closes.get(sym)
        if closes is None:
            raise RuntimeError(f"no canned candles for {sym}")
        return _ohlcv(closes).iloc[-limit:]
    return _fetch


def test_signal_full_long_when_both_lookbacks_positive():
    """7 assets all monotonically up → basket up → both 28d and 60d
    look-back returns positive → signal = 1.0."""
    n = 500
    closes = (100 + np.linspace(0, 200, n)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert isinstance(snap, SignalSnapshot)
    assert snap.signal == 1.0
    assert snap.sma_gate_open is True


def test_signal_zero_when_basket_below_sma200():
    """All assets in a clean downtrend → basket falls under its own
    SMA(200) → regime gate closes → signal = 0."""
    n = 500
    closes = np.linspace(200, 100, n).tolist()  # clean downtrend
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert snap.signal == 0.0
    assert snap.sma_gate_open is False


def test_signal_returns_ladder_only():
    """Across a random walk window, the signal must NEVER take a
    continuous value — only ``{0, 0.5, 1.0}``."""
    rng = np.random.default_rng(0)
    closes = (100 + rng.normal(0.5, 1.5, 500).cumsum()).clip(min=10).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert snap.signal in (0.0, 0.5, 1.0)


def test_signal_records_diagnostics():
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert snap.n_assets_in_basket == 7
    assert set(snap.asset_closes.keys()) == {
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
    }
    assert snap.basket_close > 0
    assert pd.Timestamp(snap.asof).tzinfo is not None  # tz-aware UTC


def test_signal_records_sma_value_and_returns():
    """sma_value is the SMA(200) at asof; per_lookback_returns gives the
    actual pct_change magnitude per lookback (not just the binary state)."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert snap.sma_value is not None
    assert snap.sma_value > 0
    assert set(snap.per_lookback_returns.keys()) == {28, 60}
    # Clean uptrend → both lookback returns positive AND >> 0.
    assert snap.per_lookback_returns[28] > 0
    assert snap.per_lookback_returns[60] > 0


def test_signal_returns_match_states():
    """The sign of per_lookback_returns must match per_lookback_states
    bit-for-bit; otherwise the journal would surface contradictory
    diagnostics to the monitor."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    for L, ret in snap.per_lookback_returns.items():
        expected_state = 1 if ret > 0 else 0
        assert snap.per_lookback_states[L] == expected_state


def test_signal_records_basket_weights():
    """basket_weights carries one drifted weight per basket asset; with
    identical trajectories they stay equal-weight (~1/7) and sum to 1.
    The allocator sizes each asset to signal × w_i × equity (C3)."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in
        ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert set(snap.basket_weights.keys()) == {
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
    }
    assert sum(snap.basket_weights.values()) == pytest.approx(1.0)
    for w in snap.basket_weights.values():
        assert w == pytest.approx(1.0 / 7)


def test_basket_weights_reflect_divergent_performance():
    """When one asset far outperforms the rest, its drifted weight at
    asof is >= the laggards' (strictly greater unless asof lands exactly
    on a monthly rebalance bar, where all reset to 1/N). This drifted
    weight — not a flat 1/N reset — is what the allocator must size to."""
    n = 500
    strong = (100 + np.linspace(0, 400, n)).tolist()   # steep uptrend
    weak = (100 + np.linspace(0, 20, n)).tolist()      # nearly flat
    fetch = _candles_factory({
        "BTC": strong, "ETH": weak, "BNB": weak, "SOL": weak,
        "ADA": weak, "XRP": weak, "DOGE": weak,
    })
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    # Strict, non-flat: an accidental revert to flat 1/N in signal.py would
    # make BTC == 1/7 and fail here. asof (~mid-May 2024 for this window) is
    # not a month-start bar, so the weight is genuinely drifted.
    assert snap.basket_weights["BTC"] > 1.0 / 7 + 1e-4
    assert snap.basket_weights["ETH"] < 1.0 / 7
    assert sum(snap.basket_weights.values()) == pytest.approx(1.0)


def test_basket_weights_are_the_asof_row_not_shifted():
    """Timing lock: basket_weights must be the weight row AT asof (the last
    completed bar) — the backtest's held-into-next-bar convention — NOT the
    prior bar. A shift(1) off-by-one on the weight lookup would pick the
    stale row and fail here."""
    n = 500
    strong = (100 + np.linspace(0, 400, n)).tolist()
    weak = (100 + np.linspace(0, 20, n)).tolist()
    symbol_to_closes = {"BTC": strong, "ETH": weak, "BNB": weak, "SOL": weak,
                        "ADA": weak, "XRP": weak, "DOGE": weak}
    fetch = _candles_factory(symbol_to_closes)
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(broker, fetch_candles=fetch, now=_NOW)

    # Rebuild the index EXACTLY as compute_live_signal does: the last
    # candles_per_asset (400) bars, in-progress bar dropped (a no-op for
    # these historical dates).
    cutoff = _NOW.normalize()
    candles = {}
    for s, c in symbol_to_closes.items():
        df = _ohlcv(c).iloc[-400:]
        candles[s] = df[df.index < cutoff]
    market = build_crypto_market_index_with_weights(candles)

    expected = market.weights.loc[snap.asof]
    loc = market.weights.index.get_loc(snap.asof)
    prev = market.weights.iloc[loc - 1]
    # asof and the prior row must genuinely differ, else the test could not
    # distinguish loc[asof] from a one-bar shift.
    assert not np.allclose(expected.to_numpy(), prev.to_numpy())
    for sym in candles:
        assert snap.basket_weights[sym] == pytest.approx(float(expected[sym]))
        assert snap.basket_weights[sym] != pytest.approx(float(prev[sym]))


def test_signal_missing_asset_raises():
    """If a single basket asset returns no candles, the signal computation
    refuses to proceed — we never want the basket to silently shrink."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    # BTC missing on purpose.
    fetch = _candles_factory({s: closes for s in
        ("ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match="BTC"):
        compute_live_signal(broker, fetch_candles=fetch)


def test_signal_empty_candles_raises():
    def empty_fetch(broker, pair, limit):
        return _ohlcv([])
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match="Empty candles"):
        compute_live_signal(broker, fetch_candles=empty_fetch)


# ---------------------------------------------------------------------------
# Default fetch path (_fetch_recent_candles → Broker.fetch_ohlcv): kline
# fetches get the broker's read-retry; a permanent failure still raises.
# _config() leaves retry_base_delay_s=0.0, so retries are instant.
# ---------------------------------------------------------------------------


class _OhlcvExchangeStub(_ExchangeStub):
    """Exchange stub serving raw OHLCV rows, optionally flaky."""

    def __init__(self, closes, fail_first=0, fail_exc=None):
        self._closes = closes
        self._fail_remaining = fail_first
        self._fail_exc = fail_exc
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=400):
        self.calls += 1
        if self._fail_remaining:
            self._fail_remaining -= 1
            raise self._fail_exc
        ts = pd.date_range(
            end=_END, periods=len(self._closes), freq="1D", tz="UTC",
        ).as_unit("ns").astype("int64") // 10**6
        return [
            [int(t), c, c, c, c, 1.0] for t, c in zip(ts, self._closes)
        ][-limit:]


def test_default_fetch_survives_one_transient_kline_error():
    """One RequestTimeout on the first of the 7 kline fetches is retried
    inside the broker; the cycle completes instead of aborting via
    SignalComputationError (which would slip the rebalance by a day)."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    exch = _OhlcvExchangeStub(
        closes, fail_first=1, fail_exc=ccxt.RequestTimeout("blip"),
    )
    broker = Broker(_config(), exch)
    snap = compute_live_signal(broker, now=_NOW)
    assert snap.signal == 1.0
    assert exch.calls == len(_config().basket) + 1  # one extra for the retry


def test_default_fetch_permanent_failure_still_raises_signal_error():
    """Retry must not mask a dead endpoint: after retry_max_attempts the
    NetworkError propagates and is wrapped as SignalComputationError,
    exactly as before — no silent basket shrinkage."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    exch = _OhlcvExchangeStub(
        closes, fail_first=10**6, fail_exc=ccxt.NetworkError("down"),
    )
    broker = Broker(_config(), exch)
    with pytest.raises(
        SignalComputationError, match=r"Could not fetch candles for BTC/USDT",
    ) as exc_info:
        compute_live_signal(broker, now=_NOW)
    assert isinstance(exc_info.value.__cause__, ccxt.NetworkError)
    assert exch.calls == 3  # retry_max_attempts on the first asset, then abort


# ---------------------------------------------------------------------------
# Warm-up depth guard (H3): truncated history must raise, never
# silently liquidate
# ---------------------------------------------------------------------------

_BASKET_7 = ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")


def test_short_history_raises_instead_of_signal_zero():
    """150 bars of clean uptrend: both lookback states are 1, but the
    SMA(200) is still NaN. Before the guard this silently produced
    signal=0 / sma_value=None — which the allocator reads as 'liquidate
    the whole book'. On mainnet a truncated kline response would become
    a real sell-off with zero errors. Must raise, with the depth and
    the requirement in the message."""
    closes = (100 + np.linspace(0, 200, 150)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"150 completed bars, need >= 200",
    ):
        compute_live_signal(broker, fetch_candles=fetch)


def test_exact_warmup_depth_raises_replication_guard():
    """Exactly 200 completed bars warm the SMA (first non-NaN at asof),
    but the SMA range then spans the whole window including the bar-0
    initial-allocation artifact — an approximate signal the backtest
    never produced. Hard refusal on every venue, NOT the structured
    warm-up skip (that stays strictly len < 200)."""
    closes = (100 + np.linspace(0, 200, 200)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"200 completed bars, need >= 230",
    ) as exc_info:
        compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert not isinstance(exc_info.value, InsufficientWarmupError)


def test_one_bar_below_warmup_boundary_raises():
    """199 bars → SMA(200) NaN at asof → raise. Pins the exact
    rolling(min_periods=window) off-by-one."""
    closes = (100 + np.linspace(0, 200, 199)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"199 completed bars, need >= 200",
    ):
        compute_live_signal(broker, fetch_candles=fetch)


def test_candles_per_asset_below_warmup_refused_before_any_fetch():
    """230 requested bars warm the SMA but leave the SMA range inside
    the window's initial-allocation artifact (exact replication needs
    warm-up + up to 30 bars to the first in-window MS rebalance + the
    dropped in-progress bar) — refused up front, before any exchange
    call."""
    calls: list[str] = []

    def fetch(broker, pair, limit):
        calls.append(pair)
        return _ohlcv((100 + np.linspace(0, 200, 500)).tolist()).iloc[-limit:]

    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError, match=r"candles_per_asset=230.*need >= 231",
    ):
        compute_live_signal(broker, fetch_candles=fetch, candles_per_asset=230)
    assert calls == [], "must refuse before burning exchange calls"


def test_candles_per_asset_at_exact_replication_floor_computes():
    """231 requested bars — the exact-replication floor — must compute
    normally with a fully warmed gate."""
    closes = (100 + np.linspace(0, 200, 500)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(
        broker, fetch_candles=fetch, candles_per_asset=231, now=_NOW,
    )
    assert snap.sma_value is not None
    assert snap.signal == 1.0


def test_one_bar_below_actual_replication_floor_raises():
    """229 ACTUAL bars pass the requested-window guard (feed truncated
    server-side, not by --candles) but sit one bar under the actual
    replication floor — the runtime depth guard must own it."""
    closes = (100 + np.linspace(0, 200, 229)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"229 completed bars, need >= 230",
    ) as exc_info:
        compute_live_signal(broker, fetch_candles=fetch, now=_NOW)
    assert not isinstance(exc_info.value, InsufficientWarmupError)


def test_depth_guard_raises_structured_insufficient_warmup_error():
    """The basket-depth guard raises the InsufficientWarmupError SUBCLASS
    carrying bars_available/bars_required — the structured facts the
    orchestrators journal into skip_reason on testnet. It stays a
    SignalComputationError, so every existing catch keeps working."""
    closes = (100 + np.linspace(0, 20, 36)).tolist()  # testnet shape
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(InsufficientWarmupError) as exc_info:
        compute_live_signal(broker, fetch_candles=fetch)
    exc = exc_info.value
    assert isinstance(exc, SignalComputationError)
    assert exc.bars_available == 36
    assert exc.bars_required == 200
    assert exc.reason_dict() == {
        "type": "insufficient_warmup",
        "bars_available": 36,
        "bars_required": 200,
        "message": str(exc)[:512],
    }


def test_non_depth_errors_are_not_insufficient_warmup():
    """Only the basket-depth guard gets the structured subclass. Every
    other refusal — the up-front candles_per_asset window check, uneven
    per-asset history, empty candles — must stay a plain
    SignalComputationError, so the testnet 'skipped_warmup' branch in the
    orchestrators can never swallow a real data incident."""
    broker = Broker(_config(), _ExchangeStub())

    # Up-front window refusal (before any fetch).
    with pytest.raises(SignalComputationError) as e1:
        compute_live_signal(
            broker,
            fetch_candles=lambda b, p, n: _ohlcv([1.0]),
            candles_per_asset=200,
        )
    assert not isinstance(e1.value, InsufficientWarmupError)

    # Empty candles for one asset.
    with pytest.raises(SignalComputationError) as e2:
        compute_live_signal(broker, fetch_candles=lambda b, p, n: _ohlcv([]))
    assert not isinstance(e2.value, InsufficientWarmupError)

    # Uneven history (M5): one truncated asset.
    full = (100 + np.linspace(0, 200, 400)).tolist()
    trunc = (100 + np.linspace(120, 200, 150)).tolist()
    frames = {s: _ohlcv_ending_at(full) for s in _BASKET_7}
    frames["DOGE"] = _ohlcv_ending_at(trunc)
    with pytest.raises(SignalComputationError, match="Uneven") as e3:
        compute_live_signal(broker, fetch_candles=_frames_fetch(frames))
    assert not isinstance(e3.value, InsufficientWarmupError)


def test_required_basket_bars_semantics():
    """max(SMA period, max lookback + 1): SMA(P) needs P bars,
    pct_change(L) needs L+1 bars (both at the last bar)."""
    assert required_basket_bars() == 200
    assert required_basket_bars((28, 60), 200) == 200
    # A lookback deeper than the SMA dominates (+1 for pct_change).
    assert required_basket_bars((28, 250), 200) == 251
    assert required_basket_bars((28, 60), 50) == 61
    with pytest.raises(ValueError, match="non-empty"):
        required_basket_bars((), 200)


def test_min_live_candles_semantics():
    """Exact-replication floor = warm-up + up to 30 bars to the first
    in-window MS rebalance (worst case: window starts day 2 of a 31-day
    month) + the dropped in-progress candle."""
    assert MS_REBALANCE_MAX_GAP_DAYS == 30
    assert min_live_candles() == 231
    assert min_live_candles((28, 60), 200) == 231
    assert min_live_candles((28, 250), 200) == 251 + 30 + 1


def test_deep_lookback_dominates_warmup_requirement():
    """With lookbacks deeper than the SMA, 250 bars warm SMA(200) but
    NOT pct_change(250) — the guard must key on max(lookback)+1, not
    just the SMA period. Without it, the NaN trailing return silently
    reads as state 0 (warm-up bars are 'flat, never long')."""
    closes = (100 + np.linspace(0, 200, 250)).tolist()
    fetch = _candles_factory({s: closes for s in _BASKET_7})
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"250 completed bars, need >= 251",
    ):
        compute_live_signal(
            broker, fetch_candles=fetch, lookbacks=(28, 250),
            candles_per_asset=400,
        )


# ---------------------------------------------------------------------------
# Window-start invariance (why the floor is 231, not 201): at the minimum
# window the signal must replicate the full history bit-exactly
# ---------------------------------------------------------------------------


def test_min_window_replicates_full_history_bit_exact():
    """At 231 bars the whole SMA/lookback range lies past the first
    in-window MS rebalance, where index levels differ from full history
    only by a constant rebasing factor and weights are identical floats.
    Divergent per-asset paths on purpose: with genuinely drifted weights
    a shorter window's start artifact (flat 1/7 forced at bar 0) would
    shift the gate margin and break these equalities."""
    rng = np.random.default_rng(7)
    closes = {
        s: (100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 500)))).tolist()
        for s in _BASKET_7
    }
    fetch = _candles_factory(closes)
    broker = Broker(_config(), _ExchangeStub())
    full = compute_live_signal(
        broker, fetch_candles=fetch, candles_per_asset=500, now=_NOW,
    )
    tail = compute_live_signal(
        broker, fetch_candles=fetch, candles_per_asset=231, now=_NOW,
    )
    assert tail.asof == full.asof
    assert tail.signal == full.signal
    assert tail.sma_gate_open == full.sma_gate_open
    assert tail.per_lookback_states == full.per_lookback_states
    # Weights depend only on the last rebalance's exact 1/N target and
    # identical per-asset returns since — bit-exact, no tolerance.
    assert tail.basket_weights == full.basket_weights
    # Index levels are rebased to 100 per window, so compare the
    # scale-free quantities the strategy actually keys on.
    assert tail.basket_close / tail.sma_value == pytest.approx(
        full.basket_close / full.sma_value, rel=1e-9,
    )
    for L in (28, 60):
        assert tail.per_lookback_returns[L] == pytest.approx(
            full.per_lookback_returns[L], rel=1e-9,
        )


def test_actual_floor_230_bars_worst_case_alignment_bit_exact():
    """230 ACTUAL bars at the tightest possible alignment: the window
    ends 2024-08-18 so it starts 2024-01-02 — day 2 of a 31-day month,
    exactly MS_REBALANCE_MAX_GAP_DAYS=30 bars before the Feb-1 rebalance.
    The SMA(200) range then starts exactly ON the rebalance bar: still
    bit-exact vs full history, proving the actual floor is tight."""
    end, now = "2024-08-18", pd.Timestamp("2024-08-19 00:45:00", tz="UTC")
    rng = np.random.default_rng(11)
    closes = {
        s: (100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 500)))).tolist()
        for s in _BASKET_7
    }
    broker = Broker(_config(), _ExchangeStub())

    def _fetch_for(depth):
        frames = {s: _ohlcv(c, end=end) for s, c in closes.items()}
        return lambda b, pair, limit: frames[pair.split("/")[0]].iloc[-depth:]

    full = compute_live_signal(
        broker, fetch_candles=_fetch_for(500), candles_per_asset=500, now=now,
    )
    tail = compute_live_signal(broker, fetch_candles=_fetch_for(230), now=now)
    assert tail.asof == full.asof
    assert tail.signal == full.signal
    assert tail.sma_gate_open == full.sma_gate_open
    assert tail.per_lookback_states == full.per_lookback_states
    assert tail.basket_weights == full.basket_weights
    assert tail.basket_close / tail.sma_value == pytest.approx(
        full.basket_close / full.sma_value, rel=1e-9,
    )


# ---------------------------------------------------------------------------
# In-progress candle exclusion (backtest replication)
# ---------------------------------------------------------------------------


def _ohlcv_ending_today(closes):
    """Frame whose last bar opens at the current UTC midnight — i.e.
    the in-progress daily candle a live fetch_ohlcv returns."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    idx = pd.date_range(end=end, periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": 1.0},
        index=idx,
    )


def test_in_progress_candle_is_excluded_from_signal():
    """The backtest decides on the completed close of day t
    (signals.shift(1)); the live cron fires minutes after UTC midnight
    while day t+1's candle is still forming. That partial bar must not
    enter the signal: here a fake intraday -50% crash on the forming
    bar would flip the ladder to 0 if included."""
    n = 320
    up = (100 + np.linspace(0, 200, n - 1)).tolist()
    closes = up + [up[-1] * 0.5]   # last bar = today's partial crash
    fetch = lambda broker, pair, limit: _ohlcv_ending_today(closes).iloc[-limit:]
    broker = Broker(_config(), _ExchangeStub())

    snap = compute_live_signal(broker, fetch_candles=fetch)

    yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    assert pd.Timestamp(snap.asof) == yesterday
    assert snap.signal == 1.0          # crash bar ignored
    assert snap.basket_close == pytest.approx(
        100.0 * (up[-1] / up[0]), rel=1e-6
    )


# ---------------------------------------------------------------------------
# Uneven per-asset history (M5): a truncated history for ONE asset must
# raise, never silently shrink the basket inside the SMA/lookback window
# ---------------------------------------------------------------------------


def _ohlcv_ending_at(closes, end="2026-01-01"):
    """Frame whose bars END at a common date — the shape of a truncated
    kline response (an exchange wipe / short response keeps the most
    recent bars, so the truncated asset starts LATER, not earlier)."""
    idx = pd.date_range(end=end, periods=len(closes), freq="1D", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 1.0},
        index=idx,
    )


def _frames_fetch(frames):
    return lambda broker, pair, limit: frames[pair.split("/")[0]].iloc[-limit:]


def test_one_truncated_asset_raises_instead_of_shrinking_basket():
    """6 assets x 400 bars + DOGE x 150 (same end date): before the guard
    this produced signal=1.0 with ZERO warnings while N_active flipped
    6->7 at bar 250 — inside the SMA(200) window — with a forced
    unscheduled 1/6->1/7 rebalance the backtest never had (with full
    history all 7 majors are listed years before any live window, so a
    late start can only be truncated API data, not a listing). The
    union index still has 400 bars, so the required_basket_bars depth
    guard cannot catch this. Must raise, naming the truncated asset and
    both depths."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    trunc = (100 + np.linspace(120, 200, 150)).tolist()
    frames = {s: _ohlcv_ending_at(full) for s in _BASKET_7}
    frames["DOGE"] = _ohlcv_ending_at(trunc)
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"Uneven basket history.*400 completed bars.*DOGE has 150 bars",
    ):
        compute_live_signal(broker, fetch_candles=_frames_fetch(frames))


def test_equal_full_histories_pass_the_uneven_history_guard():
    """All 7 assets with identical 400-bar aligned-end windows: the
    guard must stay silent and the signal must compute normally."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    frames = {s: _ohlcv_ending_at(full) for s in _BASKET_7}
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(
        broker, fetch_candles=_frames_fetch(frames),
        now=pd.Timestamp("2026-01-02 00:45:00", tz="UTC"),
    )
    assert snap.signal == 1.0
    assert snap.n_assets_in_basket == 7


def test_uniform_truncation_still_hits_depth_guard_not_uneven_guard():
    """All 7 assets truncated ALIKE (150 bars, common start): the starts
    agree, so the uneven-history guard stays silent and the
    required_basket_bars depth guard owns the error — the two checks
    are complementary, never contradictory."""
    trunc = (100 + np.linspace(0, 200, 150)).tolist()
    frames = {s: _ohlcv_ending_at(trunc) for s in _BASKET_7}
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(
        SignalComputationError,
        match=r"150 completed bars, need >= 200",
    ):
        compute_live_signal(broker, fetch_candles=_frames_fetch(frames))


def test_all_nan_closes_for_one_asset_raises():
    """A frame that is non-empty but has no valid close at all has no
    first bar to anchor the coverage check — refuse by asset name
    instead of treating the asset as never-listed."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    frames = {s: _ohlcv_ending_at(full) for s in _BASKET_7}
    frames["XRP"] = _ohlcv_ending_at([float("nan")] * 400)
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match=r"No valid closes for XRP"):
        compute_live_signal(broker, fetch_candles=_frames_fetch(frames))


def test_uniformly_stale_candles_raise():
    """All 7 assets return deep, gap-free history that simply STOPS days
    ago (frozen kline cache / degraded endpoint / stale clock). Depth,
    interior-gap and uneven-start guards all pass — only an asof
    freshness check can refuse to trade the stale signal."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    frames = {s: _ohlcv_ending_at(full, end="2024-05-11") for s in _BASKET_7}
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match=r"[Ss]tale"):
        compute_live_signal(broker, fetch_candles=_frames_fetch(frames))


def test_fresh_asof_passes_with_injected_now():
    """The freshness guard accepts exactly asof == cutoff - 1 day. The
    injected ``now`` keeps the test deterministic against fixed
    historical frames."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    frames = {s: _ohlcv_ending_at(full, end="2024-05-14") for s in _BASKET_7}
    broker = Broker(_config(), _ExchangeStub())
    snap = compute_live_signal(
        broker, fetch_candles=_frames_fetch(frames),
        now=pd.Timestamp("2024-05-15 00:45:00", tz="UTC"),
    )
    assert pd.Timestamp(snap.asof) == pd.Timestamp("2024-05-14", tz="UTC")
    assert snap.signal == 1.0


def test_one_day_stale_candles_raise_with_injected_now():
    """Off-by-one pin: frames ending exactly one bar early (the most
    recently closed bar is missing) must be refused."""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    frames = {s: _ohlcv_ending_at(full, end="2024-05-13") for s in _BASKET_7}
    broker = Broker(_config(), _ExchangeStub())
    with pytest.raises(SignalComputationError, match=r"[Ss]tale"):
        compute_live_signal(
            broker, fetch_candles=_frames_fetch(frames),
            now=pd.Timestamp("2024-05-15 00:45:00", tz="UTC"),
        )


def test_backtest_index_path_stays_lenient_for_late_listing():
    """The strictness lives ONLY in the live signal path. The index
    builder keeps the dynamic-universe leniency the backtest needs
    (leading NaN = not yet listed): the exact frames the live guard
    rejects still build a 400-bar index without raising. (See also
    test_market_index.py::test_leading_nan_late_listing_still_allowed.)"""
    full = (100 + np.linspace(0, 200, 400)).tolist()
    trunc = (100 + np.linspace(120, 200, 150)).tolist()
    frames = {s: _ohlcv_ending_at(full) for s in _BASKET_7}
    frames["DOGE"] = _ohlcv_ending_at(trunc)
    market = build_crypto_market_index_with_weights(frames)
    assert len(market.index) == 400          # union window, no raise
    assert (market.weights["DOGE"].iloc[-1]) > 0  # DOGE onboarded late
