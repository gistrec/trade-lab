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
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import ccxt

from .config import PaperConfig


logger = logging.getLogger(__name__)

# A single exchange round-trip slower than this logs at WARNING (else DEBUG).
SLOW_CALL_MS = 3000.0


def _pctl(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile over a pre-sorted, non-empty list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


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
    def fetch_time(self) -> int: ...
    def load_markets(self, reload: bool = ...) -> dict: ...
    # Phase #2b — order placement and reconstruction:
    def create_order(
        self, symbol: str, type: str, side: str, amount: float,
        price: Optional[float] = ..., params: Optional[dict] = ...,
    ) -> dict: ...
    def fetch_order(
        self, id: str, symbol: Optional[str] = ...,
        params: Optional[dict] = ...,
    ) -> dict: ...
    def fetch_open_orders(self, symbol: Optional[str] = ...) -> list: ...
    def fetch_my_trades(
        self, symbol: Optional[str] = ...,
        since: Optional[int] = ..., limit: Optional[int] = ...,
    ) -> list: ...


def _coerce_float_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _precision_to_decimals(value, *, tick_size: bool) -> Optional[int]:
    """Normalize a ccxt ``precision.amount`` to a decimal-place count.

    ccxt reports precision either as a decimal count (DECIMAL_PLACES
    mode, e.g. ``8``) or as a step size (TICK_SIZE mode — the Binance
    default, e.g. ``1e-05``). ``int()`` on a step silently yields 0,
    which would claim "whole units only". Steps that are not a power
    of ten (``0.5``, ``10``) have no decimal-count equivalent and map
    to ``None``; the raw market dict keeps the original value.
    """
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    if not tick_size:
        return int(v) if v.is_integer() else None
    decimals = -math.log10(v)
    rounded = round(decimals)
    if abs(decimals - rounded) > 1e-9 or rounded < 0:
        return None
    return int(rounded)


@dataclass
class MarketConstraints:
    """Minimum-size / precision constraints for one trading pair.

    ``min_amount`` is the smallest order in BASE units. ``min_cost`` is
    the smallest order in QUOTE notional. ``amount_precision`` is the
    decimal-place count CCXT exposes; treat it as a rough lot-step
    proxy. Any field may be ``None`` if the exchange doesn't expose it.
    """

    symbol: str
    min_amount: Optional[float]
    min_cost: Optional[float]
    amount_precision: Optional[int]
    raw: dict


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
        # Per-instance accumulator of exchange round-trip timings. A Broker is
        # built fresh per cycle, so this covers exactly one cycle's calls.
        self._calls: list[dict] = []
        # Backoff sleep, injectable so tests can zero it out.
        self._sleep = time.sleep

    # ------------------------------------------------------------------
    # Exchange round-trip instrumentation
    # ------------------------------------------------------------------

    def _timed_call(self, endpoint: str, fn, *args, **kwargs):
        """Time one CCXT round-trip, record it, and re-raise on failure.

        Latency lands in ``self._calls`` (see :meth:`exchange_call_stats`) and
        is logged structured — WARNING when a call is slow, else DEBUG. Otherwise
        identical to calling ``fn`` directly: result and exception pass straight
        through, so this changes no control flow.

        A raised ``ccxt.OrderNotFound`` is recorded as ``errored=False``: it is
        a *successful* round-trip returning a definitive "no such order", which
        ``place_order``'s query-before-place relies on as control flow (a clean
        cycle fires one per order). Counting it as an error would inflate the
        error tally the /metrics exporter and any alarm read. Every other
        exception is a real failure (``errored=True``).
        """
        start = time.perf_counter()
        ok = False
        errored = False
        try:
            result = fn(*args, **kwargs)
            ok = True
            return result
        except ccxt.OrderNotFound:
            raise  # expected negative answer, not a failure — errored stays False
        except Exception:
            errored = True
            raise
        finally:
            ms = (time.perf_counter() - start) * 1000.0
            self._calls.append({"endpoint": endpoint, "ms": ms, "ok": ok,
                                "errored": errored})
            level = logging.WARNING if ms >= SLOW_CALL_MS else logging.DEBUG
            logger.log(
                level, "exchange call %s %.0fms ok=%s", endpoint, ms, ok,
                extra={"endpoint": endpoint, "latency_ms": round(ms, 1),
                       "ok": ok, "slow": ms >= SLOW_CALL_MS},
            )

    def _read_call(self, endpoint: str, fn, *args, **kwargs):
        """Timed READ-ONLY call, retried on transient network errors.

        Retries ``ccxt.NetworkError`` — which covers RequestTimeout /
        DDoSProtection / ExchangeNotAvailable — with exponential backoff +
        jitter. Non-transient errors (auth, bad symbol, order-not-found, ...)
        are not NetworkErrors and propagate on the first attempt. Only
        idempotent reads are routed here; ``create_order`` is NOT (a retried
        placement could double-fill — reconstruction handles its failures).
        """
        attempts = max(1, self.config.retry_max_attempts)
        base = max(0.0, self.config.retry_base_delay_s)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._timed_call(endpoint, fn, *args, **kwargs)
            except ccxt.NetworkError as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                delay = base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
                logger.warning(
                    "transient exchange error on %s (attempt %d/%d), "
                    "retrying in %.2fs: %s",
                    endpoint, attempt, attempts, delay, exc,
                    extra={"endpoint": endpoint, "attempt": attempt,
                           "max_attempts": attempts,
                           "retry_delay_s": round(delay, 3),
                           "error_type": type(exc).__name__},
                )
                self._sleep(delay)
        raise last_exc  # reached only after a caught, non-final NetworkError

    def exchange_call_stats(self) -> dict:
        """Summary of this cycle's exchange round-trips.

        ``{count, errors, max_ms, p95_ms, total_ms, by_endpoint}`` — a read-only
        telemetry view a caller can journal or log at cycle end. ``count`` is
        round-trips (a retried read counts each attempt, since each is a real
        round-trip that consumed wall-clock time); ``errors`` counts real
        failures only — a query-before-place ``OrderNotFound`` is a definitive
        answer, not a failure, so it is excluded (see :meth:`_timed_call`). The
        shape is stable even for a zero-call cycle.
        """
        calls = self._calls
        if not calls:
            return {"count": 0, "errors": 0, "max_ms": 0.0, "p95_ms": 0.0,
                    "total_ms": 0.0, "by_endpoint": {}}

        def _is_error(c) -> bool:
            # Fall back to the pre-``errored`` semantics if the flag is absent.
            return c.get("errored", not c.get("ok", True))

        durs = sorted(c["ms"] for c in calls)
        by_ep: dict[str, dict] = {}
        for c in calls:
            e = by_ep.setdefault(
                c["endpoint"], {"count": 0, "max_ms": 0.0, "errors": 0},
            )
            e["count"] += 1
            e["max_ms"] = max(e["max_ms"], c["ms"])
            e["errors"] += 1 if _is_error(c) else 0
        return {
            "count": len(calls),
            "errors": sum(1 for c in calls if _is_error(c)),
            "max_ms": round(durs[-1], 1),
            "p95_ms": round(_pctl(durs, 95), 1),
            "total_ms": round(sum(durs), 1),
            "by_endpoint": {
                k: {"count": v["count"], "max_ms": round(v["max_ms"], 1),
                    "errors": v["errors"]}
                for k, v in by_ep.items()
            },
        }

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
            # Signed requests carry a timestamp; the exchange rejects any that
            # fall outside recvWindow of ITS clock. Set it explicitly rather
            # than inherit the CCXT default.
            "options": {"recvWindow": int(config.recv_window_ms)},
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
            balance = self._read_call("fetch_balance", self.exchange.fetch_balance)
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
        self._check_clock_skew()
        logger.info(
            "Broker connected: exchange=%s sandbox=%s",
            self.config.exchange_id, self.config.sandbox,
        )

    def _check_clock_skew(self) -> None:
        """Fail loud if the local clock is too far from the exchange's clock.

        The idempotency key is derived from the UTC date and signed requests
        must fall within recvWindow of server time, so a skewed clock either
        double-places across midnight or gets every request rejected (-1021).
        Aborting with a clear message beats discovering it as a cryptic
        rejection deep in a cycle. ``clock_skew_max_ms=0`` disables the check.
        """
        max_ms = self.config.clock_skew_max_ms
        if max_ms <= 0:
            return
        try:
            server_ms = self._read_call("fetch_time", self.exchange.fetch_time)
        except Exception as exc:
            raise BrokerError(
                "Could not fetch exchange server time for the clock-skew "
                f"check on {self.config.exchange_id}: {exc}"
            ) from exc
        skew_ms = float(server_ms) - time.time() * 1000.0
        if abs(skew_ms) > max_ms:
            raise BrokerError(
                f"Clock skew {skew_ms:.0f}ms exceeds {max_ms}ms vs "
                f"{self.config.exchange_id} server time. Fix host NTP (chrony) "
                "before trading: the idempotency key is UTC-derived and signed "
                "requests must be within recvWindow."
            )
        logger.info(
            "clock skew ok: %.0fms (limit %dms)", skew_ms, max_ms,
            extra={"clock_skew_ms": round(skew_ms, 1), "limit_ms": max_ms},
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
        raw = self._read_call("fetch_balance", self.exchange.fetch_balance)
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
        ticker = self._read_call("fetch_ticker", self.exchange.fetch_ticker, symbol)
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

    def fetch_market_constraints(self, symbol: str) -> "MarketConstraints":
        """Pull minimum notional and amount-step for a symbol via CCXT.

        CCXT normalizes per-exchange ``markets[symbol]['limits']`` to a
        uniform shape:

        * ``limits.amount.min`` — minimum base-asset quantity per order.
        * ``limits.cost.min``   — minimum quote-asset notional per order.
        * ``precision.amount``  — number of decimals (lot-step proxy).

        Different exchanges populate different subsets; we accept None
        for missing fields and let the caller decide the policy
        (skip the order, round up to min, etc.). For Binance: cost.min
        is usually populated; for Kraken: amount.min and precision.
        """
        markets = self._read_call("load_markets", self.exchange.load_markets)
        if symbol not in markets:
            raise BrokerError(
                f"Market {symbol!r} not found on {self.config.exchange_id}. "
                "Pair may not be listed or quote-currency may not match."
            )
        m = markets[symbol]
        limits = (m.get("limits") or {})
        amount = (limits.get("amount") or {})
        cost = (limits.get("cost") or {})
        precision = (m.get("precision") or {})
        is_tick_size = getattr(self.exchange, "precisionMode", None) == ccxt.TICK_SIZE
        return MarketConstraints(
            symbol=symbol,
            min_amount=_coerce_float_or_none(amount.get("min")),
            min_cost=_coerce_float_or_none(cost.get("min")),
            amount_precision=_precision_to_decimals(
                precision.get("amount"), tick_size=is_tick_size,
            ),
            raw=m,
        )

    # ------------------------------------------------------------------
    # Order placement (phase #2b)
    # ------------------------------------------------------------------

    def create_order_safe(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: str,
    ) -> dict:
        """Place a market order with a deterministic client order ID.

        Re-raises CCXT exceptions verbatim. Mapping to rejection /
        timeout / partial outcomes happens in
        :mod:`trade_lab.execution.orders`, not here — the broker stays
        a thin transport.

        The ``newClientOrderId`` param is the Binance name; if a future
        exchange uses a different one, this is the single line to swap.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        # NOTE: create_order is timed but deliberately NOT retried here — a
        # transient failure is resolved by the reconstruction path (query by
        # clientOrderId next cycle), which keeps placement idempotency-safe.
        return self._timed_call(
            "create_order", self.exchange.create_order,
            symbol, "market", side, amount, None,
            {"newClientOrderId": client_order_id},
        )

    def fetch_order_by_coid(
        self,
        client_order_id: str,
        symbol: str,
    ) -> dict:
        """Fetch an order by its client order ID.

        Binance reads the ID from ``origClientOrderId`` in params and
        ignores the positional ``id`` arg when both are present. We
        pass the clientOrderId in both spots so mocks can match on
        either path; real ccxt+Binance prefers the params version.

        Re-raises ``ccxt.OrderNotFound`` when the exchange has no
        record of the ID — the caller turns that into a "needs
        placement" decision.
        """
        return self._read_call(
            "fetch_order", self.exchange.fetch_order,
            client_order_id, symbol,
            {"origClientOrderId": client_order_id},
        )

    def fetch_open_orders(self, symbol: Optional[str] = None) -> list:
        """Open orders, optionally filtered to one symbol.

        Used at cycle startup to discover orders that exist on the
        exchange but are not in our local state — for example because
        the state file was wiped or a previous cycle crashed before
        persisting.
        """
        return self._read_call(
            "fetch_open_orders", self.exchange.fetch_open_orders, symbol)

    def fetch_my_trades_since(
        self,
        symbol: str,
        since_ms: Optional[int] = None,
    ) -> list:
        """Recent trades for a symbol. Used by reconstruction fallback.

        ``since_ms`` is a Unix epoch milliseconds timestamp; ``None``
        lets the exchange decide the default window (Binance returns
        the last 24h by default).
        """
        return self._read_call(
            "fetch_my_trades", self.exchange.fetch_my_trades, symbol, since_ms)

    # ------------------------------------------------------------------
    # Equity estimate
    # ------------------------------------------------------------------

    def estimate_total_equity_usd(
        self,
        snapshot: Optional[BalanceSnapshot] = None,
    ) -> float:
        """Mark-to-market: ``quote_total + sum(asset_total × ticker_last)``.

        Uses the snapshot's quote balance plus a live ticker call for
        each non-zero asset. Cheap enough to run every cycle on a
        7-asset basket; for larger universes consider batching.

        A ticker failure propagates (``BrokerError`` / ccxt error) —
        marking a held position at zero would understate equity and
        shrink every target downstream, turning one missing price into
        spurious sells across the whole basket. Hard rule: missing
        prices raise; the cycle fails loud and is journaled as failed.
        """
        snap = snapshot if snapshot is not None else self.fetch_balance_snapshot()
        equity = snap.quote_total
        for sym, total in snap.asset_totals.items():
            if total <= 0.0:
                continue
            price = self.fetch_ticker_price(f"{sym}/{snap.quote_currency}")
            equity += total * price
        return equity
