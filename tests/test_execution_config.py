"""Tests for the paper-trading config layer.

No network, no real CCXT. The config layer is pure env-var parsing
with strict safety gates; these tests verify those gates fire when
they should.
"""
from __future__ import annotations

import os

import pytest

from trade_lab.execution.config import (
    PaperConfigError, load_paper_config,
)


# A reasonable env to start from; tests override individual keys.
_DEFAULT_ENV = {
    "TRADE_LAB_PAPER_EXCHANGE": "binance",
    "TRADE_LAB_PAPER_SANDBOX": "true",
    "TRADE_LAB_PAPER_API_KEY": "test_key_12345",
    "TRADE_LAB_PAPER_API_SECRET": "test_secret_12345",
}


def _apply_env(monkeypatch, env: dict, clear: list[str] | None = None):
    """Set the env vars in ``env`` and unset any in ``clear``."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in (clear or []):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Required vars
# ---------------------------------------------------------------------------


def test_missing_exchange_raises(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_EXCHANGE"])
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_EXCHANGE"):
        load_paper_config()


def test_missing_api_key_raises(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_API_KEY"])
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_API_KEY"):
        load_paper_config()


def test_missing_api_secret_raises(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_API_SECRET"])
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_API_SECRET"):
        load_paper_config()


def test_missing_sandbox_flag_raises(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_SANDBOX"])
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_SANDBOX"):
        load_paper_config()


def test_empty_required_var_raises(monkeypatch):
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_API_KEY": "   "}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_API_KEY"):
        load_paper_config()


# ---------------------------------------------------------------------------
# Sandbox / mainnet safety gate
# ---------------------------------------------------------------------------


def test_sandbox_true_loads_cleanly(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV)
    cfg = load_paper_config()
    assert cfg.sandbox is True
    assert cfg.allow_mainnet is False  # default
    assert cfg.exchange_id == "binance"


def test_mainnet_refused_without_allow_flag(monkeypatch):
    """sandbox=False but TRADE_LAB_PAPER_ALLOW_MAINNET unset → refuse."""
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_SANDBOX": "false"}
    _apply_env(monkeypatch, env, clear=["TRADE_LAB_PAPER_ALLOW_MAINNET"])
    with pytest.raises(PaperConfigError, match="Mainnet trading refused"):
        load_paper_config()


def test_mainnet_refused_with_allow_false(monkeypatch):
    """sandbox=False AND allow_mainnet=false → still refuse."""
    env = {
        **_DEFAULT_ENV,
        "TRADE_LAB_PAPER_SANDBOX": "false",
        "TRADE_LAB_PAPER_ALLOW_MAINNET": "false",
    }
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="Mainnet trading refused"):
        load_paper_config()


def test_mainnet_allowed_with_both_flags_true(monkeypatch):
    """The only path to mainnet: two explicit flags."""
    env = {
        **_DEFAULT_ENV,
        "TRADE_LAB_PAPER_SANDBOX": "false",
        "TRADE_LAB_PAPER_ALLOW_MAINNET": "true",
    }
    _apply_env(monkeypatch, env)
    cfg = load_paper_config()
    assert cfg.sandbox is False
    assert cfg.allow_mainnet is True


def test_kraken_with_sandbox_true_raises(monkeypatch):
    """Kraken has no CCXT sandbox — sandbox=true must refuse explicitly
    (CLAUDE.md hard rule), never rely on CCXT to crash or ignore it."""
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_EXCHANGE": "kraken"}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="[Kk]raken"):
        load_paper_config()


def test_kraken_case_insensitive_sandbox_raise(monkeypatch):
    """Exchange id is lowercased before the guard — 'Kraken' counts."""
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_EXCHANGE": "Kraken"}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="[Kk]raken"):
        load_paper_config()


def test_ambiguous_bool_raises(monkeypatch):
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_SANDBOX": "maybe"}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="bool-like"):
        load_paper_config()


# ---------------------------------------------------------------------------
# Defaults and parsing
# ---------------------------------------------------------------------------


def test_default_basket_is_seven_assets(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_BASKET"])
    cfg = load_paper_config()
    assert cfg.basket == ("BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE")


def test_custom_basket_is_parsed_and_upcased(monkeypatch):
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_BASKET": "btc,eth,sol "}
    _apply_env(monkeypatch, env)
    cfg = load_paper_config()
    assert cfg.basket == ("BTC", "ETH", "SOL")


def test_default_quote_currency_is_usdt(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_QUOTE"])
    cfg = load_paper_config()
    assert cfg.quote_currency == "USDT"


def test_default_timeout_is_20s(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV, clear=["TRADE_LAB_PAPER_TIMEOUT_MS"])
    cfg = load_paper_config()
    assert cfg.request_timeout_ms == 20_000


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------


def test_repr_masks_api_key_and_secret(monkeypatch):
    _apply_env(monkeypatch, _DEFAULT_ENV)
    cfg = load_paper_config()
    r = repr(cfg)
    # The full secret must never appear.
    assert "test_secret_12345" not in r
    # The full key must not appear either (only the tail).
    assert "test_key_12345" not in r
    # Last 4 chars of the key are intentionally shown for log debugging.
    assert "2345" in r
    # Secret is fully masked.
    assert "api_secret='***'" in r


# ---------------------------------------------------------------------------
# Credential isolation: importing config modules must not load .env
# ---------------------------------------------------------------------------


def test_importing_config_modules_does_not_inject_env():
    """The monitoring dashboard imports trade_lab.config (and research
    modules import this package); a module-level load_dotenv() used to
    pull API keys from .env into any importing process. .env loading
    now happens only in the CLI entrypoint. Trivially green on machines
    without a .env; the regression shows up wherever one exists."""
    import subprocess
    import sys

    code = (
        "import os; before = set(os.environ); "
        "import trade_lab.config, trade_lab.execution.config; "
        "leaked = [k for k in set(os.environ) - before "
        "if k.startswith('TRADE_LAB')]; "
        "print(','.join(sorted(leaked)))"
    )
    clean_env = {
        k: v for k, v in os.environ.items() if not k.startswith("TRADE_LAB")
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=clean_env, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        f"importing config modules injected env vars: {out.stdout.strip()}"
    )


def test_garbage_timeout_raises_config_error(monkeypatch):
    """Every other config mistake reports as PaperConfigError with the
    offending env var named — a bare ValueError traceback from int()
    broke that contract."""
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_TIMEOUT_MS": "twenty seconds"}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="TRADE_LAB_PAPER_TIMEOUT_MS"):
        load_paper_config()


def test_non_positive_timeout_raises_config_error(monkeypatch):
    env = {**_DEFAULT_ENV, "TRADE_LAB_PAPER_TIMEOUT_MS": "0"}
    _apply_env(monkeypatch, env)
    with pytest.raises(PaperConfigError, match="positive"):
        load_paper_config()
