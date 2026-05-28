"""Exchange-agnostic broker abstraction built on CCXT.

Three goals:

1. **No exchange-specific code outside this module.** Callers only see
   the methods on :class:`Broker`. Switching from Binance testnet to
   Kraken live is a config change, not a code change.
2. **Exchange is the source of truth.** Every call that touches state
   (balances, positions, orders) goes through CCXT, not through any
   in-memory cache. Memory caches are dangerous when the process
   restarts or when the testnet wipes state mid-session.
3. **Refuse-by-default to mainnet.** The :func:`Broker.connect`
   constructor sets ``set_sandbox_mode(True)`` based on the config
   flag; the config layer already refuses to load when sandbox=False
   without an explicit mainnet-allow flag.

The CCXT exchange object is held as ``self.exchange`` but callers
should never reach for it directly — anything they need should be
exposed through a Broker method. That discipline keeps the abstraction
honest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import ccxt

from .config import PaperConfig


logger = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Base for broker errors — connection, auth, request failures.

    Wraps the underlying CCXT exception type when relevant so callers
    can branch on connectivity vs auth without importing ccxt
    themselves.
    """


class ConnectionRefused(BrokerError):
    """Raised at construction time if the broker refuses to point at
    mainnet without explicit operator clearance. Distinct from network
    errors so paper-trading scripts can fail loudly on this case."""


class _CcxtExchange(Protocol):
    """Subset of the CCXT exchange interface the broker actually uses.

    Defining this as a Protocol makes the broker testable with a mock
    that only implements these methods. Real ``ccxt.binance()`` and
    ``ccxt.kraken()`` instances satisfy the protocol structurally.
    """

    id: str

    def set_sandbox_mode(self, enabled: bool) -> None: ...
    def fetch_balance(self) -> dict: ...
    def fetch_ticker(self, symbol: str) -> dict: ...
    def fetch_status(self) -> dict: ...
    def load_markets(self, reload: bool = ...) -> dict: ...


@dataclass
class BalanceSnapshot:
    """Snapshot of balances pulled live from the exchange.

    `free` is what the caller can spend; `used` is locked into open
    orders; `total` = free + used. Quote-currency balance is broken
    out separately because it's what the strategy uses as the base
    capital reference.
    """

    raw: dict                     # ccxt.fetch_balance() output, for audit
    quote_currency: str           # e.g. "USDT"
    quote_free: float
    quote_used: float
    quote_total: float
    asset_totals: dict[str, float]  # base symbol -> total (free+used)


class Broker:
    """Exchange-agnostic broker.

    Construct via :meth:`connect`. Never instantiate directly — the
    constructor takes a pre-built exchange object so it can be tested
    with mocks; production code should always go through
    :meth:`connect` so the safety gates are enforced.
    """

    def __init__(
        self,
        config: PaperConfig,
        exchange: _CcxtExchange,
    ) -> None:
        self.config = config
        self.exchange = exchange

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls, config: PaperConfig) -> "Broker":
        """Build a CCXT exchange and verify connectivity.

        The verification step performs the cheapest authenticated call
        the exchange supports (``fetch_balance``). If that succeeds we
        know:

        * Network is reachable.
        * API key + secret are valid for the configured exchange.
        * The sandbox flag was set on the correct exchange instance
          (not, say, a stale mainnet one from a prior process).

        Failures here are :class:`BrokerError` subclasses; the caller
        is expected to surface them to the operator, not retry silently.
        """
        if not config.sandbox and not config.allow_mainnet:
            # The config loader is supposed to catch this — defensive check.
            raise ConnectionRefused(
                "Refusing to connect: mainnet not explicitly enabled."
            )

        exchange_cls = getattr(ccxt, config.exchange_id, None)
        if exchange_cls is None:
            raise BrokerError(
                f"Unknown CCXT exchange id: {config.exchange_id!r}"
            )

        exchange = exchange_cls({
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
            "timeout": int(config.request_timeout_ms),
        })

        # Order matters: set sandbox mode BEFORE the first authenticated
        # request, otherwise we'd briefly point at mainnet. The
        # exchange object hasn't issued any requests at this point.
        try:
            exchange.set_sandbox_mode(config.sandbox)
        except Exception as exc:
            raise BrokerError(
                f"Could not set sandbox mode on {config.exchange_id}: {exc}"
            ) from exc

        broker = cls(config=config, exchange=exchange)
        broker._verify_connection()
        return broker

    # ------------------------------------------------------------------
    # Connection verification
    # ------------------------------------------------------------------

    def _verify_connection(self) -> None:
        """Make one authenticated round-trip and surface failures."""
        try:
            # fetch_balance is the standard "are we connected and
            # authenticated?" probe. Most exchanges return immediately.
            balance = self.exchange.fetch_balance()
        except ccxt.AuthenticationError as exc:
            raise BrokerError(
                f"Authentication failed for {self.config.exchange_id} "
                f"(sandbox={self.config.sandbox}). Check API key/secret."
            ) from exc
        except ccxt.NetworkError as exc:
            raise BrokerError(
                f"Network error contacting {self.config.exchange_id}: {exc}"
            ) from exc
        except Exception as exc:
            raise BrokerError(
                f"Unexpected error during connection probe to "
                f"{self.config.exchange_id}: {exc}"
            ) from exc

        if not isinstance(balance, dict):
            raise BrokerError(
                f"Connection probe returned a non-dict balance "
                f"({type(balance).__name__}); the CCXT exchange object "
                "may be misconfigured."
            )
        logger.info(
            "Broker connected: exchange=%s sandbox=%s",
            self.config.exchange_id, self.config.sandbox,
        )

    # ------------------------------------------------------------------
    # Balance / state queries (always live, never cached)
    # ------------------------------------------------------------------

    def fetch_balance_snapshot(self) -> BalanceSnapshot:
        """Pull balances from the exchange and decompose them.

        ALWAYS calls the exchange — never returns cached state. This
        is on purpose: a testnet balance reset, a partial-fill update,
        or a manual deposit must all be visible immediately.
        """
        raw = self.exchange.fetch_balance()
        if not isinstance(raw, dict):
            raise BrokerError("fetch_balance did not return a dict.")
        quote = self.config.quote_currency
        # CCXT exposes per-currency dicts with 'free', 'used', 'total'.
        # Missing currencies mean "no balance in that asset" — treat as 0.
        per_currency = raw.get(quote, {}) or {}
        quote_free = float(per_currency.get("free", 0.0) or 0.0)
        quote_used = float(per_currency.get("used", 0.0) or 0.0)
        quote_total = float(per_currency.get("total", 0.0) or 0.0)

        asset_totals: dict[str, float] = {}
        for sym in self.config.basket:
            entry = raw.get(sym, {}) or {}
            total = float(entry.get("total", 0.0) or 0.0)
            asset_totals[sym] = total

        return BalanceSnapshot(
            raw=raw,
            quote_currency=quote,
            quote_free=quote_free,
            quote_used=quote_used,
            quote_total=quote_total,
            asset_totals=asset_totals,
        )

    def fetch_ticker_price(self, symbol: str) -> float:
        """Latest last-trade price for ``symbol`` (CCXT format BASE/QUOTE)."""
        ticker = self.exchange.fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            raise BrokerError(f"fetch_ticker did not return a dict for {symbol}.")
        last = ticker.get("last")
        if last is None:
            # Fall back to ticker close. Some exchanges report only close.
            last = ticker.get("close")
        if last is None:
            raise BrokerError(
                f"Ticker for {symbol} has no last/close field; cannot mark."
            )
        return float(last)

    def estimate_total_equity_usd(
        self,
        snapshot: Optional[BalanceSnapshot] = None,
    ) -> float:
        """Mark-to-market: ``quote_total + sum(asset_total × ticker_last)``.

        Uses the snapshot's quote balance plus a live ticker call for
        each non-zero asset. Cheap enough to run every cycle on a
        7-asset basket; for larger universes consider batching.
        """
        snap = snapshot if snapshot is not None else self.fetch_balance_snapshot()
        equity = snap.quote_total
        for sym, total in snap.asset_totals.items():
            if total <= 0.0:
                continue
            try:
                price = self.fetch_ticker_price(f"{sym}/{snap.quote_currency}")
            except Exception as exc:
                # If a single ticker fails, mark the position at zero
                # rather than blowing up the equity number. Log loudly.
                logger.warning(
                    "Could not mark %s for equity calc: %s — counting as 0",
                    sym, exc,
                )
                continue
            equity += total * price
        return equity
