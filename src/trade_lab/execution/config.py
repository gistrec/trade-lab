"""Configuration for the paper-trading / live execution layer.

All values come from environment variables. ``.env`` support lives at
the CLI entrypoint (``trade_lab.cli.main``), NOT at module import:
importing this module from a credential-free process (monitoring)
must not pull API keys from ``.env`` into its environment.

**No secrets in code. Ever.** The :class:`PaperConfig` dataclass holds
API key + secret as Python strings only after the runtime has read
them from the environment. Loading code never logs them, and the
:meth:`PaperConfig.__repr__` masks them for safe printing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


class PaperConfigError(RuntimeError):
    """Raised when the paper-trading configuration is missing or
    inconsistent. Always carries a human-readable message that tells
    the operator which env var to fix; we never want a silent fallback
    to a mainnet default."""


_DEFAULT_BASKET = ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")


@dataclass(frozen=True)
class PaperConfig:
    """Runtime configuration for the paper-trading execution layer."""

    exchange_id: str             # CCXT id, e.g. "binance" or "kraken"
    sandbox: bool                # set_sandbox_mode flag
    api_key: str                 # masked in __repr__
    api_secret: str              # masked in __repr__
    allow_mainnet: bool          # required True to connect when sandbox=False
    quote_currency: str          # e.g. "USDT"
    basket: Tuple[str, ...]      # universe symbols, e.g. ("BTC","ETH",...)
    request_timeout_ms: int      # CCXT timeout

    def __repr__(self) -> str:
        # Never leak credentials in logs or REPL inspection.
        return (
            f"PaperConfig(exchange_id={self.exchange_id!r}, "
            f"sandbox={self.sandbox}, "
            f"api_key='***{self.api_key[-4:] if len(self.api_key) >= 4 else ''}', "
            f"api_secret='***', "
            f"allow_mainnet={self.allow_mainnet}, "
            f"quote_currency={self.quote_currency!r}, "
            f"basket={self.basket}, "
            f"request_timeout_ms={self.request_timeout_ms})"
        )


def _coerce_bool(value: str | None, name: str) -> bool:
    """Parse a strict bool env value. Refuse anything ambiguous."""
    if value is None:
        raise PaperConfigError(f"{name} must be set (true or false).")
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise PaperConfigError(
        f"{name} must be a bool-like string (true/false). Got {value!r}."
    )


def _require_env(name: str) -> str:
    """Fetch a required env var. Empty/missing fails loudly."""
    value = os.getenv(name, "").strip()
    if not value:
        raise PaperConfigError(
            f"{name} is required but missing or empty. "
            f"Add it to your .env (see paper.env.example)."
        )
    return value


def _parse_basket(value: str | None) -> Tuple[str, ...]:
    """Comma-separated symbol list. Default is the 7-asset basket
    matching `data/binance_*_USDT_1d.parquet`."""
    if not value:
        return _DEFAULT_BASKET
    parts = [p.strip().upper() for p in value.split(",") if p.strip()]
    if not parts:
        return _DEFAULT_BASKET
    return tuple(parts)


def load_paper_config() -> PaperConfig:
    """Read all ``TRADE_LAB_PAPER_*`` env vars into a :class:`PaperConfig`.

    Refuses to return a config that would allow mainnet trading unless
    ``TRADE_LAB_PAPER_ALLOW_MAINNET=true`` is set in addition to
    ``TRADE_LAB_PAPER_SANDBOX=false``. The two-flag requirement makes
    it impossible to accidentally point at mainnet by flipping a
    single env value.
    """
    exchange_id = _require_env("TRADE_LAB_PAPER_EXCHANGE").lower()
    sandbox = _coerce_bool(
        os.getenv("TRADE_LAB_PAPER_SANDBOX"), "TRADE_LAB_PAPER_SANDBOX"
    )
    api_key = _require_env("TRADE_LAB_PAPER_API_KEY")
    api_secret = _require_env("TRADE_LAB_PAPER_API_SECRET")
    allow_mainnet = _coerce_bool(
        os.getenv("TRADE_LAB_PAPER_ALLOW_MAINNET", "false"),
        "TRADE_LAB_PAPER_ALLOW_MAINNET",
    )
    quote_currency = os.getenv("TRADE_LAB_PAPER_QUOTE", "USDT").strip().upper()
    basket = _parse_basket(os.getenv("TRADE_LAB_PAPER_BASKET"))
    timeout_raw = os.getenv("TRADE_LAB_PAPER_TIMEOUT_MS", "20000")
    try:
        timeout_ms = int(timeout_raw)
    except ValueError:
        raise PaperConfigError(
            f"TRADE_LAB_PAPER_TIMEOUT_MS must be an integer (milliseconds), "
            f"got {timeout_raw!r}."
        ) from None
    if timeout_ms <= 0:
        raise PaperConfigError(
            f"TRADE_LAB_PAPER_TIMEOUT_MS must be positive, got {timeout_ms}."
        )

    # Refuse-by-default to mainnet. Two flags must agree.
    if not sandbox and not allow_mainnet:
        raise PaperConfigError(
            "Mainnet trading refused: TRADE_LAB_PAPER_SANDBOX is false but "
            "TRADE_LAB_PAPER_ALLOW_MAINNET is not true. Set both flags "
            "explicitly to leave the testnet."
        )

    # Kraken has no CCXT sandbox. Whether set_sandbox_mode crashes or
    # is silently ignored is a CCXT implementation detail per version;
    # a silently ignored sandbox flag would send live mainnet requests
    # from a config that claims to be paper-safe. Refuse explicitly
    # (CLAUDE.md hard rule).
    if exchange_id == "kraken" and sandbox:
        raise PaperConfigError(
            "TRADE_LAB_PAPER_EXCHANGE=kraken with TRADE_LAB_PAPER_SANDBOX="
            "true is invalid: Kraken has no CCXT sandbox, so the sandbox "
            "flag cannot be honored. Use binance for testnet paper trading."
        )

    return PaperConfig(
        exchange_id=exchange_id,
        sandbox=sandbox,
        api_key=api_key,
        api_secret=api_secret,
        allow_mainnet=allow_mainnet,
        quote_currency=quote_currency,
        basket=basket,
        request_timeout_ms=timeout_ms,
    )
