"""Tests for the dry-run orchestrator (end-to-end without orders)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.execution.broker import Broker
from trade_lab.execution.config import PaperConfig
from trade_lab.execution.dry_run import DryRunResult, run_dry_cycle


def _config(basket=("BTC", "ETH")):
    return PaperConfig(
        exchange_id="binance", sandbox=True, api_key="k", api_secret="s",
        allow_mainnet=False, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


class _StubExchange:
    """A more featured stub that supports the dry-run pipeline."""
    id = "stub"

    def __init__(self, balance_usdt=10_000.0, btc_holdings=0.0):
        self.balance = {
            "USDT": {"free": balance_usdt, "used": 0.0, "total": balance_usdt},
            "BTC":  {"free": btc_holdings, "used": 0.0, "total": btc_holdings},
            "ETH":  {"free": 0.0,           "used": 0.0, "total": 0.0},
        }
        self.tickers = {
            "BTC/USDT": {"last": 50_000.0, "close": 50_000.0},
            "ETH/USDT": {"last": 3_000.0,  "close": 3_000.0},
        }
        # OHLCV: clean uptrend so signal=1.0, gate open.
        self._closes = (100 + np.linspace(0, 200, 500)).tolist()

    def set_sandbox_mode(self, enabled): pass

    def fetch_balance(self):
        return self.balance

    def fetch_ticker(self, symbol):
        return self.tickers[symbol]

    def fetch_status(self):
        return {"status": "ok"}

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=400):
        # End at yesterday's UTC midnight — the most recently closed daily
        # bar — so the signal's asof-freshness guard passes at any hour.
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        timestamps = pd.date_range(
            end=end, periods=len(self._closes), freq="1D", tz="UTC",
        ).as_unit("ns").astype("int64") // 10**6
        rows = [
            [int(ts), c, c, c, c, 1.0]
            for ts, c in zip(timestamps, self._closes)
        ]
        return rows[-limit:]

    def load_markets(self, reload=False):
        return {
            "BTC/USDT": {
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10.0}},
                "precision": {"amount": 8},
            },
            "ETH/USDT": {
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 10.0}},
                "precision": {"amount": 8},
            },
        }


def test_dry_run_emits_buys_when_fully_long_signal_and_no_holdings():
    """Clean uptrend → signal=1.0 → buy each asset at 1/N × equity."""
    exch = _StubExchange(balance_usdt=10_000.0, btc_holdings=0.0)
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert isinstance(result, DryRunResult)
    assert result.signal == 1.0
    assert result.sma_gate_open is True

    # 2-asset basket × $10k equity → $5k target each, shaved 10 bp by the
    # full-entry quote reserve (#28): the buys would otherwise sum to
    # exactly quote_free.
    sides = {o["symbol"]: o["side"] for o in result.orders_planned}
    assert sides == {"BTC/USDT": "buy", "ETH/USDT": "buy"}
    btc_order = next(o for o in result.orders_planned if o["symbol"] == "BTC/USDT")
    assert btc_order["notional_quote"] == pytest.approx(5_000.0 * 0.999)


def test_dry_run_does_not_call_create_order():
    """Defensive: confirm the stub's create_order is never invoked.

    The stub doesn't define create_order at all; if the dry-run pipeline
    ever calls it, AttributeError surfaces immediately. The fact that
    this test runs cleanly without that attribute IS the assertion."""
    exch = _StubExchange()
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert isinstance(result, DryRunResult)


def test_dry_run_skips_tiny_delta_below_min_cost():
    """current holdings ~= target → sub-$10 delta should be SKIPPED,
    not sent. The result records it under orders_skipped."""
    # Math: equity = balance_usdt + btc_qty × $50k. For a 2-asset
    # basket at signal=1.0, target per asset = equity / 2. We want
    # BTC delta to be ~$5 below min_cost=$10. Choose
    # btc_holdings=0.02 (= $1000), balance_usdt=$1010 → equity=$2010,
    # target_btc_value=$1005, current=$1000, delta=$5 ⇒ below min_cost.
    exch = _StubExchange(balance_usdt=1_010.0, btc_holdings=0.02)
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    btc_orders = [o for o in result.orders_planned if o["symbol"] == "BTC/USDT"]
    btc_skipped = [s for s in result.orders_skipped if s["symbol"] == "BTC/USDT"]
    # The BTC delta should be in skipped, not in orders.
    assert btc_orders == [], f"BTC delta should have been skipped, got {btc_orders}"
    assert len(btc_skipped) == 1
    assert result.total_skipped_quote_drift > 0


def test_dry_run_with_signal_zero_plans_full_sell():
    """Build a downtrend → signal=0 → if BTC is held, plan a full sell."""
    exch = _StubExchange(balance_usdt=0.0, btc_holdings=0.1)
    # Override candles with a clean downtrend.
    exch._closes = np.linspace(200, 100, 500).tolist()
    broker = Broker(_config(), exch)
    result = run_dry_cycle(broker, candles_per_asset=400)
    assert result.signal == 0.0
    # BTC must be sold (target qty is 0; current is 0.1).
    btc_order = next(
        o for o in result.orders_planned if o["symbol"] == "BTC/USDT"
    )
    assert btc_order["side"] == "sell"
    assert btc_order["base_amount"] == pytest.approx(0.1)


def test_dry_run_returns_structured_data_for_logging():
    """All result fields populated — useful for the JSON
    reconciliation logger in step #2b."""
    exch = _StubExchange()
    broker = Broker(_config(), exch)
    r = run_dry_cycle(broker, candles_per_asset=400)
    assert r.asof is not None
    assert isinstance(r.signal, float)
    assert isinstance(r.total_equity, float)
    assert isinstance(r.target_allocation, dict)
    assert isinstance(r.current_holdings_quote, dict)
    assert isinstance(r.orders_planned, list)
    assert isinstance(r.orders_skipped, list)
    assert isinstance(r.total_skipped_quote_drift, float)


def test_nan_weight_fails_loud_through_dry_cycle(tmp_path, monkeypatch):
    """Mirror of the live-cycle fail-loud guard on the dry-run path: a NaN
    in basket_weights raises and journals outcome='failed' rather than
    silently mis-sizing the printed plan."""
    import json
    import math

    from trade_lab.execution import dry_run as dr
    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import SignalSnapshot

    bad_snap = SignalSnapshot(
        asof=pd.Timestamp("2026-06-11", tz="UTC"), signal=1.0,
        basket_close=150.0, asset_closes={"BTC": 50_000.0, "ETH": 3_000.0},
        sma_gate_open=True, n_assets_in_basket=2,
        basket_weights={"BTC": math.nan, "ETH": 0.5},
    )
    monkeypatch.setattr(dr, "compute_live_signal", lambda *a, **k: bad_snap)

    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    with pytest.raises(ValueError, match="BTC"):
        run_dry_cycle(broker, journal=journal)

    lines = (tmp_path / "cycles.jsonl").read_text().splitlines()
    cycle = json.loads(lines[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "ValueError"


def _mainnet_config(basket=("BTC", "ETH")):
    return PaperConfig(
        exchange_id="binance", sandbox=False, api_key="k", api_secret="s",
        allow_mainnet=True, quote_currency="USDT", basket=basket,
        request_timeout_ms=5000,
    )


def test_dry_run_short_history_on_testnet_journals_skipped_warmup(tmp_path):
    """Binance-testnet shape: the exchange wiped candles and returns only
    ~36 daily bars, so SMA(200) structurally can NEVER warm there. On a
    sandbox config that is a healthy first-class skip, not an incident:
    journal outcome='skipped_warmup' with an explicit skip_reason block
    (bars_available/bars_required), error=None, no plan. The exception
    still re-raises — no result exists — and the CLI maps it to exit 0."""
    import json

    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import InsufficientWarmupError

    exch = _StubExchange(balance_usdt=0.0, btc_holdings=0.1)
    exch._closes = (100 + np.linspace(0, 20, 36)).tolist()  # uptrend, 36 bars
    broker = Broker(_config(), exch)
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(InsufficientWarmupError, match="36 completed bars"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "skipped_warmup"
    assert cycle["error"] is None
    assert cycle["skip_reason"]["type"] == "insufficient_warmup"
    assert cycle["skip_reason"]["bars_available"] == 36
    assert cycle["skip_reason"]["bars_required"] == 200
    assert "36 completed bars" in cycle["skip_reason"]["message"]
    # No plan was produced — nothing that could be mistaken for orders.
    assert cycle["orders_planned"] is None


def test_dry_run_short_history_on_mainnet_stays_failed(tmp_path):
    """PIN of mainnet strictness (H3): the identical short history on a
    sandbox=False config means truncated kline history on the real-money
    path — outcome='failed' + raise, no skipped_warmup softening. This
    test exists to break any future dilution of the mainnet posture."""
    import json

    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import SignalComputationError

    exch = _StubExchange(balance_usdt=0.0, btc_holdings=0.1)
    exch._closes = (100 + np.linspace(0, 20, 36)).tolist()
    broker = Broker(_mainnet_config(), exch)
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(SignalComputationError, match="36 completed bars"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "InsufficientWarmupError"
    assert "36 completed bars" in cycle["error"]["message"]
    assert cycle["skip_reason"] is None


def test_dry_run_other_signal_errors_still_fail_on_testnet(tmp_path, monkeypatch):
    """Only the depth guard's InsufficientWarmupError becomes a
    first-class skip on the sandbox. Every other SignalComputationError
    (e.g. the M5 uneven-history guard) keeps the failed posture on
    testnet too — a real data incident must never hide behind the
    warm-up skip."""
    import json

    from trade_lab.execution import dry_run as dr
    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import SignalComputationError

    def _raise(*a, **k):
        raise SignalComputationError(
            "Uneven basket history: DOGE has 150 bars starting 2025-08-05"
        )

    monkeypatch.setattr(dr, "compute_live_signal", _raise)
    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(SignalComputationError, match="Uneven"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "SignalComputationError"
    assert cycle["skip_reason"] is None


def test_dry_run_keyboard_interrupt_journals_failed_and_reraises(tmp_path):
    """Regression (M4, dry-run mirror): Ctrl-C mid-cycle (KeyboardInterrupt
    is a BaseException, not an Exception) must still journal
    outcome='failed' and re-raise — same guarantee as an ordinary
    exception. Before the fix `except Exception` let the interrupt bypass
    the journal write entirely."""
    import json

    from trade_lab.execution.journal import JournalWriter

    class _CtrlCStub(_StubExchange):
        def fetch_balance(self):
            raise KeyboardInterrupt()

    broker = Broker(_config(), _CtrlCStub())
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(KeyboardInterrupt):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["error"]["type"] == "KeyboardInterrupt"


def test_dry_run_journals_candle_close_price_fallback(tmp_path, caplog):
    """A failed ticker falls back to the candle close for sizing (documented
    posture) — but the journal must carry a per-symbol marker with the
    price's source and age, so a stale-priced sizing decision stays
    attributable post-hoc, not buried in cron logs."""
    import json
    import logging

    from trade_lab.execution.journal import JournalWriter
    from trade_lab.execution.signal import decision_age_seconds

    class _EthTickerDownStub(_StubExchange):
        def fetch_ticker(self, symbol):
            if symbol == "ETH/USDT":
                return {}  # no last/close → BrokerError in fetch_ticker_price
            return super().fetch_ticker(symbol)

    broker = Broker(_config(), _EthTickerDownStub())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    with caplog.at_level(logging.WARNING, logger="trade_lab.execution.dry_run"):
        result = run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    # The warning stays (fail-loud letter), on the orchestrator's logger.
    warn = [
        r for r in caplog.records
        if "Ticker for ETH failed" in r.getMessage()
    ]
    assert len(warn) == 1
    assert warn[0].name == "trade_lab.execution.dry_run"

    # Sizing used the candle close (stub closes end at 300.0), not 0.0.
    eth = next(o for o in result.orders_planned if o["symbol"] == "ETH/USDT")
    assert eth["price_used"] == pytest.approx(300.0)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    fb = cycle["price_fallbacks"]
    assert set(fb) == {"ETH"}
    assert fb["ETH"]["source"] == "candle_close_fallback"
    expected_age = decision_age_seconds(pd.Timestamp(cycle["signal"]["asof"]))
    assert fb["ETH"]["age_s"] == pytest.approx(expected_age, abs=60.0)
    assert fb["ETH"]["age_s"] >= 0.0
    # The BTC ticker succeeded — no marker for it.
    assert "BTC" not in fb


def test_dry_run_all_tickers_ok_journals_no_price_fallbacks(tmp_path):
    """Healthy tickers → price_fallbacks stays None: the marker means
    'stale price used', never 'field present on every cycle'."""
    import json

    from trade_lab.execution.journal import JournalWriter

    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["price_fallbacks"] is None


def test_dry_run_failed_after_read_phase_keeps_price_fallbacks(
        tmp_path, monkeypatch):
    """A cycle that dies AFTER the read phase must journal the fallback
    markers in the failed-cycle entry — same posture as the live cycle."""
    import json

    from trade_lab.execution import dry_run as dr
    from trade_lab.execution.journal import JournalWriter

    class _EthTickerDownStub(_StubExchange):
        def fetch_ticker(self, symbol):
            if symbol == "ETH/USDT":
                return {}  # no last/close → BrokerError in fetch_ticker_price
            return super().fetch_ticker(symbol)

    def _boom(plan):
        raise RuntimeError("post-read failure")

    # First failure point after the read phase (DryRunResult assembly).
    monkeypatch.setattr(dr, "total_skipped_quote_drift", _boom)
    broker = Broker(_config(), _EthTickerDownStub())
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(RuntimeError, match="post-read failure"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    fb = cycle["price_fallbacks"]
    assert set(fb) == {"ETH"}
    assert fb["ETH"]["source"] == "candle_close_fallback"


def test_dry_run_failed_before_read_phase_price_fallbacks_none(tmp_path):
    """A failure before the read phase produced any prices journals
    price_fallbacks=None (and must not NameError on the pre-bound
    read variable)."""
    import json

    from trade_lab.execution.journal import JournalWriter

    class _BalanceDownStub(_StubExchange):
        def fetch_balance(self):
            raise RuntimeError("balance down")

    broker = Broker(_config(), _BalanceDownStub())
    journal = JournalWriter(tmp_path / "cycles.jsonl")

    with pytest.raises(RuntimeError, match="balance down"):
        run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["outcome"] == "failed"
    assert cycle["price_fallbacks"] is None


def test_dry_run_records_exchange_latency_in_journal(tmp_path):
    """A successful dry cycle stamps context.exchange_latency — read-only
    telemetry the /metrics exporter surfaces. Metadata only."""
    import json

    from trade_lab.execution.journal import JournalWriter

    broker = Broker(_config(), _StubExchange())
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    lat = cycle["context"]["exchange_latency"]
    assert lat["count"] > 0  # fetch_balance / ticker / ohlcv / markets were timed
    assert lat["errors"] == 0
    assert set(lat) >= {"count", "errors", "max_ms", "p95_ms", "by_endpoint"}


# ---------------------------------------------------------------------------
# Reserve cap in the operator's preview (#71): the cap's shave is metered
# apart from sub-min drift in the live path and the journal — the dry-run
# result and printout must show it too, or the preview hides the reserve.
# ---------------------------------------------------------------------------


def test_dry_run_reports_funding_cap_apart_from_submin_drift(tmp_path):
    """Full-cash entry: the whole divergence is the 10 bp reserve, not
    work the exchange refused. It must land in total_funding_cap_quote
    (result AND journal) with the sub-min drift left at zero."""
    import json

    from trade_lab.execution.journal import JournalWriter

    broker = Broker(_config(), _StubExchange(balance_usdt=10_000.0))
    journal = JournalWriter(tmp_path / "cycles.jsonl")
    result = run_dry_cycle(broker, journal=journal, candles_per_asset=400)

    assert {s["reason"] for s in result.orders_skipped} == {"funding_cap"}
    assert result.total_skipped_quote_drift == 0.0
    assert result.total_funding_cap_quote == pytest.approx(10_000.0 * 0.001)

    cycle = json.loads((tmp_path / "cycles.jsonl").read_text().splitlines()[-1])
    assert cycle["total_funding_cap_quote"] == pytest.approx(10_000.0 * 0.001)
    assert cycle["total_skipped_quote_drift"] == 0.0


def _printable_result(orders_skipped, *, drift=0.0, cap=0.0) -> DryRunResult:
    return DryRunResult(
        asof=pd.Timestamp("2026-08-25T00:00:00Z"), signal=1.0,
        sma_gate_open=True, total_equity=10_000.0,
        target_allocation={"BTC": 5_000.0},
        current_holdings_quote={"BTC": 0.0},
        orders_planned=[], orders_skipped=orders_skipped,
        total_skipped_quote_drift=drift, total_funding_cap_quote=cap,
    )


def _skip(symbol: str, reason: str, notional: float) -> dict:
    return {
        "symbol": symbol, "desired_side": "buy", "desired_amount": 0.1,
        "desired_notional": notional, "reason": reason,
    }


def test_print_dry_run_meters_reserve_cap_apart_from_submin(capsys):
    """One header over both classes prices the reserve's deliberate shave
    into 'unfillable drift': the sub-min block must count only sub-min
    skips, and the cap gets its own labelled figure."""
    from trade_lab.execution.dry_run import print_dry_run

    print_dry_run(
        _printable_result(
            [_skip("BTC/USDT", "notional 4.00 < min_cost 10.0", 4.0),
             _skip("ETH/USDT", "funding_cap", 5.0)],
            drift=4.0, cap=5.0,
        ),
        quote="USDT",
    )
    out = capsys.readouterr().out
    assert "Sub-min divergence (1, cumulative 4.00 USDT)" in out
    assert "Reserve cap (10 bp) (1, held back 5.00 USDT)" in out
    assert "CAP  BUY  ETH/USDT" in out
    # The cap entry must not be counted or listed as sub-min drift.
    assert "SKIP BUY  ETH/USDT" not in out


def test_print_dry_run_flags_known_preview_vs_live_cap_divergence(capsys):
    """#70: the dry run has no order-state store, so it caps buys whose
    same-day coid the live cycle exempts. Documented divergence — the
    operator must read it next to the figure, not in a docstring."""
    from trade_lab.execution.dry_run import print_dry_run

    print_dry_run(
        _printable_result([_skip("ETH/USDT", "funding_cap", 5.0)], cap=5.0),
        quote="USDT",
    )
    out = capsys.readouterr().out
    assert "preview-only figure" in out
    assert "caps LESS than shown, never more" in out


def test_print_dry_run_uncapped_cycle_says_so_without_the_caveat(capsys):
    """No shave → an explicit zero line and no divergence note: the
    caveat is about a figure that exists, not boilerplate."""
    from trade_lab.execution.dry_run import print_dry_run

    print_dry_run(_printable_result([]), quote="USDT")
    out = capsys.readouterr().out
    assert "Reserve cap (10 bp): 0.00" in out
    assert "preview-only figure" not in out
