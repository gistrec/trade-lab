"""Daily forward-test cycle: signal + vintage + journal.

Each invocation does, in this order:

1. **Frozen-hash gate.** Refuse to run if ``production_config_hash()``
   has drifted from the ``FROZEN_CONFIG_HASH`` literal pinned in this
   module. There is no other legitimate path to changing the strategy
   on the forward-test branch — if the hash differs, *something*
   changed the canonical parameters since the test started, and the
   resulting log row would be measuring a different strategy. Raise
   loud, before anything is journaled. (The pin must be a literal:
   comparing against ``config.CANONICAL_HASH`` — recomputed from the
   same object at import time — was a tautology that let a config
   hotfix through silently; regression M8.)
2. **Idempotency check.** If the cycle's signal date is already in
   the journal, return that row without recomputing — the cron job
   can be invoked multiple times per day safely.
3. **Fetch candles** for the 7 frozen-basket assets: completed daily
   bars up to and including the signal date, never beyond. The signal
   date is ``asof`` for a backfill and ``today - 1`` for a same-day
   run (the bar stamped today is still forming — the live path in
   ``execution.signal.compute_live_signal`` drops it the same way).
   A fetched bar dated after the signal date trips a hard look-ahead
   guard: a backfilled row must equal what a same-day run would have
   written, and both must equal the backtest's signal at that date.
4. **Vintage snapshot.** Canonical-serialize the OHLCV bytes,
   content-hash, write immutably (no overwrite).
5. **Compute signal** with the frozen strategy on the frozen basket
   construction.
6. **Paper portfolio update.** Maintain virtual USD equity by
   compounding the prior cycle's ladder state against today's basket
   return, less simulated turnover cost on position changes.
7. **Append row.** Atomic JSONL append.

Step 1 is the contract that protects step 5: paper trading is only
meaningful as a forward-test if the strategy under test has been the
same one for the whole horizon.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..backtest.market_index import build_crypto_market_index
from ..config import PRODUCTION_CONFIG, production_config_hash
from ..data.fetch_ohlcv import fetch_ohlcv
from ..strategies.tsmom import TimeSeriesMomentumStrategy

from .journal import HarnessLogRow, append_row, get_row_for_date, is_already_logged
from .vintage_store import store_vintage

# Frozen pin of the DSR-validated production config: SHA256 of the
# canonical JSON serialization (see ``production_config_hash``). This is
# a hardcoded LITERAL on purpose. The gate used to compare against
# ``config.CANONICAL_HASH``, which is recomputed from PRODUCTION_CONFIG
# at import time — a hotfix to the config moved both sides of the
# comparison together, so the gate could never fire (M8 tautology).
#
# This literal must always equal ``_EXPECTED_HASH`` in
# ``tests/test_production_config.py`` (the double pin is itself tested,
# so editing one without the other fails CI). To change it INTENTIONALLY
# follow the new-research-cycle procedure documented in that file, then
# recompute the value with:
#
#   .venv/bin/python -c "from trade_lab.config import \
#       production_config_hash; print(production_config_hash())"
FROZEN_CONFIG_HASH = (
    "ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753"
)


class HarnessError(RuntimeError):
    """Raised by the harness on any condition that must NOT be papered
    over (hash drift, empty candle fetch, etc.). The cron job should
    treat a HarnessError as a failed cycle and surface it for human
    review, not retry blindly."""


def run_paper_trading_cycle(
    *,
    log_path: Path,
    vintage_root: Path,
    asof: Optional[date] = None,
    candles_per_asset: int = 400,
    fetch_callable: Optional[Callable[[str, int, date], pd.DataFrame]] = None,
) -> HarnessLogRow:
    """Run one cycle. Returns the written (or already-existing) row.

    ``asof`` is the cycle date (default: today UTC). The row is keyed
    and labeled by the cycle's *signal date* — the last completed daily
    bar: ``asof`` itself for a backfill (``asof < today``), yesterday
    for a same-day run (the bar stamped today is still forming). The
    journal row dated ``T`` is therefore always the backtest's signal
    at index ``T``, computed on closes up to and including the
    completed ``close[T]``, regardless of when the cycle ran.
    """
    log_path = Path(log_path)
    vintage_root = Path(vintage_root)

    # --- Step 1: frozen-hash gate ---
    runtime_hash = production_config_hash(PRODUCTION_CONFIG)
    if runtime_hash != FROZEN_CONFIG_HASH:
        raise HarnessError(
            f"Frozen-config hash drift: runtime={runtime_hash}, "
            f"frozen={FROZEN_CONFIG_HASH}. Refusing to run — paper "
            f"trading must replicate the validated config exactly. "
            f"If this drift is intentional, open a new research cycle, "
            f"update tests/test_production_config.py AND the "
            f"FROZEN_CONFIG_HASH literal in this module."
        )

    # --- Step 2: idempotency check ---
    today_utc = datetime.now(tz=timezone.utc).date()
    if asof is None:
        asof = today_utc
    if asof > today_utc:
        raise HarnessError(
            f"asof={asof.isoformat()} is in the future (today UTC is "
            f"{today_utc.isoformat()}). A cycle can only be computed on "
            f"completed daily bars — refusing."
        )
    # The signal date is the last COMPLETED daily bar as of ``asof``.
    # Daily bars are stamped at their open (00:00 UTC), so the bar
    # stamped today is still forming while a same-day cycle runs —
    # using it would let intraday noise into signal/SMA and diverge
    # from the live path (execution.signal.compute_live_signal drops
    # the forming bar) and from the backtest (signal[T] is a function
    # of the completed close[T]). A backfill (asof < today) uses the
    # completed bar dated ``asof`` itself; the next-day bar never
    # participates even if it already exists.
    signal_date = asof if asof < today_utc else today_utc - timedelta(days=1)
    signal_date_str = signal_date.isoformat()
    if is_already_logged(signal_date_str, log_path):
        existing = get_row_for_date(signal_date_str, log_path)
        # type guard — is_already_logged True implies a row exists
        assert existing is not None
        return existing

    # --- Step 3: fetch candles ---
    fetch = fetch_callable if fetch_callable is not None else _default_fetch
    asset_candles: dict[str, pd.DataFrame] = {}
    for sym in PRODUCTION_CONFIG.assets:
        try:
            df = fetch(sym, candles_per_asset, signal_date)
        except Exception as exc:
            raise HarnessError(
                f"Could not fetch candles for {sym}: {exc}"
            ) from exc
        if df.empty:
            raise HarnessError(
                f"Empty candles returned for {sym} at asof={signal_date_str}. "
                "Failing loud rather than logging a row with a partial basket."
            )
        # Normalise tz so the canonical hash is stable
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        # Hard look-ahead guard: no bar after the signal date may reach
        # the vintage snapshot or the signal. Raise instead of silently
        # clamping — a future bar here means the fetcher's window
        # regressed (H5: `until = asof + 1 day` against fetch_ohlcv's
        # INCLUSIVE bound let the completed T+1 bar into a backfill and
        # the forming bar into a same-day run, so the "immutable" row
        # for T was computed on data unavailable at T's close).
        last_bar_date = df.index.max().date()
        if last_bar_date > signal_date:
            raise HarnessError(
                f"Look-ahead guard tripped for {sym}: fetched candles end "
                f"at {last_bar_date.isoformat()}, after the signal date "
                f"{signal_date_str}. Fix the fetcher's window; refusing to "
                f"journal a row computed on future bars."
            )
        asset_candles[sym] = df

    # --- Step 4: immutable vintage snapshot ---
    vintage_hash = store_vintage(asset_candles, vintage_root)

    # --- Step 5: frozen-strategy signal ---
    cfg = PRODUCTION_CONFIG
    basket = build_crypto_market_index(
        asset_candles,
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        rebalance_freq=cfg.basket_rebalance_freq,
    )
    if basket.empty:
        raise HarnessError("Basket construction returned an empty frame.")
    basket_last_date = basket.index[-1].date()
    if basket_last_date != signal_date:
        raise HarnessError(
            f"Basket ends at {basket_last_date.isoformat()} but the "
            f"cycle's signal date is {signal_date_str}. Journaling this "
            f"row would silently mislabel a stale signal as "
            f"{signal_date_str}'s (the date-join with the backtest would "
            f"break) — refusing."
        )

    strategy = TimeSeriesMomentumStrategy(
        lookbacks=cfg.lookbacks,
        sma_filter_periods=cfg.sma_filter_periods,
        use_vol_target=cfg.use_vol_target,
        vol_lookback=cfg.vol_lookback,
        annual_vol_target=cfg.annual_vol_target,
        max_position_size=cfg.max_position_size,
        rebalance_threshold=cfg.rebalance_threshold,
        annualization_factor=cfg.annualization_factor,
    )
    signal_series = strategy.generate_signals(basket)
    if signal_series.empty:
        raise HarnessError("Strategy produced an empty signal series.")

    close = basket["close"]
    basket_close = float(close.iloc[-1])
    sma_period = cfg.sma_filter_periods[0]
    sma_value_raw = close.rolling(sma_period).mean().iloc[-1]
    sma_value: Optional[float] = (
        float(sma_value_raw) if pd.notna(sma_value_raw) else None
    )
    sma_gate_open = bool(sma_value is not None and basket_close > sma_value)
    ladder_state = float(signal_series.iloc[-1])

    per_lookback_states: dict[str, int] = {}
    per_lookback_returns: dict[str, float] = {}
    for L in cfg.lookbacks:
        pr = close.pct_change(L, fill_method=None).iloc[-1]
        per_lookback_returns[str(L)] = float(pr) if pd.notna(pr) else 0.0
        per_lookback_states[str(L)] = int(pr > 0) if pd.notna(pr) else 0

    # --- Step 6: paper portfolio update ---
    n_assets = len(cfg.assets)
    target_per_asset = ladder_state / n_assets
    target_weights = {sym: target_per_asset for sym in cfg.assets}

    prior_row = _prior_row(log_path, signal_date_str)
    if prior_row is None:
        prior_ladder = 0.0
        current_weights = {sym: 0.0 for sym in cfg.assets}
        portfolio_equity = cfg.initial_capital
        daily_return = 0.0
    else:
        prior_ladder = prior_row.ladder_state
        current_weights = dict(prior_row.target_weights)
        # Return since the prior cycle, measured WITHIN this cycle's
        # normalized window. build_crypto_market_index renormalizes the
        # index to 100 at the first bar of whatever window it is handed,
        # so prior_row.basket_close (stored from a differently-anchored
        # earlier window) is on another scale — ratioing it against
        # basket_close does NOT yield the basket return. Locate the bar in
        # THIS window dated prior_row.date and ratio against it, so the
        # return spans exactly the holding period even across a missed
        # cron day.
        prior_close_same_window = _basket_close_on_date(close, prior_row.date)
        if prior_close_same_window is not None and prior_close_same_window > 0:
            daily_return = float(basket_close / prior_close_same_window - 1.0)
        elif len(close) >= 2 and float(close.iloc[-2]) > 0:
            # Prior date rolled out of the window (or is unparseable):
            # fall back to the one-bar return within the same window.
            daily_return = float(basket_close / float(close.iloc[-2]) - 1.0)
        else:
            daily_return = 0.0
        # Mark prior equity forward by prior-ladder × period return.
        portfolio_equity = prior_row.portfolio_equity * (
            1.0 + prior_ladder * daily_return
        )

    intended_trades = {
        sym: target_weights[sym] - current_weights[sym]
        for sym in cfg.assets
    }
    # Simulated turnover cost on the size of the position change.
    turnover = sum(abs(v) for v in intended_trades.values())
    sim_cost_rate = cfg.fee_rate + cfg.slippage_rate
    sim_cost = portfolio_equity * turnover * sim_cost_rate
    portfolio_equity -= sim_cost

    gross_position_return = prior_ladder * daily_return
    net_position_return = gross_position_return - turnover * sim_cost_rate

    # --- Step 7: append row ---
    row = HarnessLogRow(
        date=signal_date_str,
        # Gate-verified above: runtime_hash == FROZEN_CONFIG_HASH, so the
        # journaled hash is both "what actually ran" and the frozen pin.
        config_hash=runtime_hash,
        vintage_content_hash=vintage_hash,
        basket_close=basket_close,
        sma_value=sma_value,
        sma_gate_open=sma_gate_open,
        ladder_state=ladder_state,
        prior_ladder_state=prior_ladder,
        per_lookback_states=per_lookback_states,
        per_lookback_returns=per_lookback_returns,
        target_weights=target_weights,
        current_weights=current_weights,
        intended_trades=intended_trades,
        portfolio_equity=float(portfolio_equity),
        daily_return=float(daily_return),
        gross_position_return=float(gross_position_return),
        net_position_return=float(net_position_return),
    )
    append_row(row, log_path)
    return row


def _prior_row(log_path: Path, asof_str: str) -> Optional[HarnessLogRow]:
    """The most recent row strictly before ``asof_str``.

    NOT the last physically appended row: an ``--asof`` backfill of a
    missed earlier date leaves later dates already in the journal, so the
    last line can be a FUTURE row — chaining the return backwards in time
    and marking equity off the wrong anchor. ISO date strings sort
    chronologically, so the max row with ``date < asof`` is the correct
    predecessor.
    """
    from .journal import read_log
    prior = (r for r in read_log(log_path) if r.date < asof_str)
    return max(prior, key=lambda r: r.date, default=None)


def _basket_close_on_date(close: pd.Series, date_str: str) -> Optional[float]:
    """Close of the normalized basket index on the bar dated ``date_str``.

    Returns ``None`` if no bar matches (the date rolled out of the window,
    or ``date_str`` is unparseable). Used to read the holding-period return
    within a single normalized window instead of across two differently
    anchored ones.
    """
    try:
        target = pd.Timestamp(date_str).date()
    except (TypeError, ValueError):
        return None
    matches = [float(v) for ts, v in close.items() if ts.date() == target]
    return matches[-1] if matches else None


def _default_fetch(sym: str, n_bars: int, asof: date) -> pd.DataFrame:
    """Default live fetch: paginated Binance daily OHLCV of completed
    bars up to and including the bar dated ``asof``.

    Daily bars are stamped at their open (00:00 UTC) and
    ``fetch_ohlcv``'s ``until`` bound is INCLUSIVE, so the bound must
    be midnight of ``asof`` itself. The previous ``asof + 1 day`` bound
    also matched the bar stamped ``asof + 1`` — on a backfill that is
    the completed next-day bar (a literal 1-bar look-ahead into signal,
    SMA(200) and the immutable vintage), on a same-day run it is the
    still-forming current bar. ``run_paper_trading_cycle`` only ever
    passes a completed-bar date here (yesterday at the latest).

    A test or replay can pass a custom ``fetch_callable`` to bypass
    the live exchange (so tests don't hit the network).
    """
    since_date = asof - timedelta(days=int(n_bars * 1.2))
    since = datetime(since_date.year, since_date.month, since_date.day)
    until = datetime(asof.year, asof.month, asof.day)
    df = fetch_ohlcv(
        "binance",
        f"{sym}/USDT",
        "1d",
        since=since,
        until=until,
        limit=1000,
    )
    if df.empty:
        return df
    # Cut to the last n_bars
    return df.iloc[-n_bars:]
