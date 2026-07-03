"""Daily forward-test cycle: signal + vintage + journal.

Each invocation does, in this order:

1. **Frozen-hash gate.** Refuse to run if ``production_config_hash()``
   has drifted from ``CANONICAL_HASH``. There is no other legitimate
   path to changing the strategy on the forward-test branch — if the
   hash differs, *something* changed the canonical parameters since
   the test started, and the resulting log row would be measuring a
   different strategy. Raise loud.
2. **Idempotency check.** If today's UTC date is already in the
   journal, return that row without recomputing — the cron job can
   be invoked multiple times per day safely.
3. **Fetch candles** for the 7 frozen-basket assets.
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
from ..config import CANONICAL_HASH, PRODUCTION_CONFIG, production_config_hash
from ..data.fetch_ohlcv import fetch_ohlcv
from ..strategies.tsmom import TimeSeriesMomentumStrategy

from .journal import HarnessLogRow, append_row, get_row_for_date, is_already_logged
from .vintage_store import store_vintage


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
    """Run one cycle. Returns the written (or already-existing) row."""
    log_path = Path(log_path)
    vintage_root = Path(vintage_root)

    # --- Step 1: frozen-hash gate ---
    runtime_hash = production_config_hash(PRODUCTION_CONFIG)
    if runtime_hash != CANONICAL_HASH:
        raise HarnessError(
            f"Frozen-config hash drift: runtime={runtime_hash}, "
            f"canonical={CANONICAL_HASH}. Refusing to run — paper "
            f"trading must replicate the validated config exactly. "
            f"If this drift is intentional, open a new research cycle "
            f"and update tests/test_production_config.py."
        )

    # --- Step 2: idempotency check ---
    if asof is None:
        asof = datetime.now(tz=timezone.utc).date()
    asof_str = asof.isoformat()
    if is_already_logged(asof_str, log_path):
        existing = get_row_for_date(asof_str, log_path)
        # type guard — is_already_logged True implies a row exists
        assert existing is not None
        return existing

    # --- Step 3: fetch candles ---
    fetch = fetch_callable if fetch_callable is not None else _default_fetch
    asset_candles: dict[str, pd.DataFrame] = {}
    for sym in PRODUCTION_CONFIG.assets:
        try:
            df = fetch(sym, candles_per_asset, asof)
        except Exception as exc:
            raise HarnessError(
                f"Could not fetch candles for {sym}: {exc}"
            ) from exc
        if df.empty:
            raise HarnessError(
                f"Empty candles returned for {sym} at asof={asof_str}. "
                "Failing loud rather than logging a row with a partial basket."
            )
        # Normalise tz so the canonical hash is stable
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
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

    prior_row = _last_row(log_path)
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
        date=asof_str,
        config_hash=CANONICAL_HASH,
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


def _last_row(log_path: Path) -> Optional[HarnessLogRow]:
    from .journal import read_log
    rows = read_log(log_path)
    return rows[-1] if rows else None


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
    """Default live fetch: paginated Binance daily OHLCV up to ``asof``.

    A test or replay can pass a custom ``fetch_callable`` to bypass
    the live exchange (so tests don't hit the network).
    """
    since_date = asof - timedelta(days=int(n_bars * 1.2))
    since = datetime(since_date.year, since_date.month, since_date.day)
    until = datetime(asof.year, asof.month, asof.day) + timedelta(days=1)
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
