"""Environment isolation between testnet and mainnet.

Three layers, each fail-loud:

* ``assert_journal_env`` — a journal file never mixes environments.
* ``OrderStateStore(expected_env=...)`` — a state file is stamped with
  its environment and refuses cross-environment reuse (clientOrderIds
  carry no environment component, so a shared state file would make the
  idempotency fast-path skip real mainnet placements).
* ``--env-file`` CLI flag — a typo'd path is an error, not a silent
  fallback to the default ``.env``.
"""
from __future__ import annotations

import json

import pytest

from trade_lab.execution.journal import JournalEnvMismatch, assert_journal_env
from trade_lab.execution.order_state import (
    OrderStateEntry, OrderStateEnvMismatch, OrderStateStore,
)


def _cycle_line(sandbox: bool, exchange: str = "binance") -> str:
    return json.dumps({
        "cycle_id": "x", "outcome": "success",
        "context": {"mode": "dry_run", "exchange": exchange,
                    "sandbox": sandbox},
    }) + "\n"


def _entry(coid: str = "tsmom_20260709_BTCUSDT_buy") -> OrderStateEntry:
    return OrderStateEntry(
        client_order_id=coid, symbol="BTC/USDT", side="buy",
        intended_amount=1.0, status="closed", exchange_order_id="1",
        placed_at="2026-07-09T00:05:00+00:00",
        last_seen_at="2026-07-09T00:05:10+00:00",
    )


# ---------------------------------------------------------------------------
# Journal guard
# ---------------------------------------------------------------------------


def test_journal_guard_missing_file_passes(tmp_path):
    assert_journal_env(
        tmp_path / "new.jsonl", exchange_id="binance", sandbox=False,
    )


def test_journal_guard_matching_env_passes(tmp_path):
    p = tmp_path / "cycles.jsonl"
    p.write_text(_cycle_line(sandbox=True))
    assert_journal_env(p, exchange_id="binance", sandbox=True)


def test_journal_guard_sandbox_mismatch_raises(tmp_path):
    p = tmp_path / "cycles.jsonl"
    p.write_text(_cycle_line(sandbox=True))
    with pytest.raises(JournalEnvMismatch, match="never share a journal"):
        assert_journal_env(p, exchange_id="binance", sandbox=False)


def test_journal_guard_exchange_mismatch_raises(tmp_path):
    p = tmp_path / "cycles.jsonl"
    p.write_text(_cycle_line(sandbox=False, exchange="kraken"))
    with pytest.raises(JournalEnvMismatch):
        assert_journal_env(p, exchange_id="binance", sandbox=False)


def test_journal_guard_uses_last_valid_context(tmp_path):
    """Corrupt lines and context-free records (smoke tests) are skipped,
    matching the reader's tolerance — the LAST valid context decides."""
    p = tmp_path / "cycles.jsonl"
    p.write_text(
        "{not json\n"
        + json.dumps({"kind": "smoke_test", "result": {}}) + "\n"
        + _cycle_line(sandbox=False)
    )
    assert_journal_env(p, exchange_id="binance", sandbox=False)
    with pytest.raises(JournalEnvMismatch):
        assert_journal_env(p, exchange_id="binance", sandbox=True)


def test_journal_guard_context_free_file_passes(tmp_path):
    p = tmp_path / "smoke.jsonl"
    p.write_text(json.dumps({"kind": "smoke_test", "result": {}}) + "\n")
    assert_journal_env(p, exchange_id="binance", sandbox=False)


# ---------------------------------------------------------------------------
# Order state stamping
# ---------------------------------------------------------------------------

_TESTNET = {"exchange": "binance", "sandbox": True}
_MAINNET = {"exchange": "binance", "sandbox": False}


def test_state_write_stamps_env_and_same_env_reopens(tmp_path):
    p = tmp_path / "orders.json"
    store = OrderStateStore(p, expected_env=_TESTNET)
    store.put(_entry())

    raw = json.loads(p.read_text())
    assert raw["__meta__"] == {"exchange": "binance", "sandbox": True}

    again = OrderStateStore(p, expected_env=_TESTNET)
    assert set(again.all_entries()) == {"tsmom_20260709_BTCUSDT_buy"}


def test_state_cross_env_reopen_raises(tmp_path):
    p = tmp_path / "orders.json"
    OrderStateStore(p, expected_env=_TESTNET).put(_entry())

    mainnet_store = OrderStateStore(p, expected_env=_MAINNET)
    with pytest.raises(OrderStateEnvMismatch, match="stamped"):
        mainnet_store.all_entries()


def test_state_legacy_unstamped_file_ok_for_testnet(tmp_path):
    """Pre-stamp files are testnet by construction (mainnet placement was
    refused before the stamp existed) — testnet keeps working, and the
    next write adds the stamp."""
    p = tmp_path / "orders.json"
    from dataclasses import asdict
    p.write_text(json.dumps({_entry().client_order_id: asdict(_entry())}))

    store = OrderStateStore(p, expected_env=_TESTNET)
    assert set(store.all_entries()) == {"tsmom_20260709_BTCUSDT_buy"}

    store.put(_entry("tsmom_20260709_ETHUSDT_buy"))
    assert json.loads(p.read_text())["__meta__"]["sandbox"] is True


def test_state_legacy_unstamped_file_refused_for_mainnet(tmp_path):
    p = tmp_path / "orders.json"
    from dataclasses import asdict
    p.write_text(json.dumps({_entry().client_order_id: asdict(_entry())}))

    store = OrderStateStore(p, expected_env=_MAINNET)
    with pytest.raises(OrderStateEnvMismatch, match="environment stamp"):
        store.all_entries()


def test_state_empty_file_adopts_any_env(tmp_path):
    p = tmp_path / "orders.json"
    store = OrderStateStore(p, expected_env=_MAINNET)
    assert store.all_entries() == {}
    store.put(_entry())
    assert json.loads(p.read_text())["__meta__"]["sandbox"] is False


def test_state_meta_never_leaks_as_entry(tmp_path):
    p = tmp_path / "orders.json"
    store = OrderStateStore(p, expected_env=_TESTNET)
    store.put(_entry())
    assert store.get("__meta__") is None
    assert "__meta__" not in store.all_entries()
    assert "__meta__" not in store.open_entries()


def test_state_no_expected_env_ignores_stamp(tmp_path):
    """Legacy callers (expected_env=None) read a stamped file without
    tripping over the meta key — forward compatible."""
    p = tmp_path / "orders.json"
    OrderStateStore(p, expected_env=_MAINNET).put(_entry())

    plain = OrderStateStore(p)
    assert set(plain.all_entries()) == {"tsmom_20260709_BTCUSDT_buy"}


# ---------------------------------------------------------------------------
# --env-file CLI flag
# ---------------------------------------------------------------------------


def test_env_file_missing_is_an_error(tmp_path):
    from trade_lab.cli import main

    with pytest.raises(SystemExit, match="env file not found"):
        main(["paper-status", "--env-file", str(tmp_path / "nope.env")])


def test_paper_default_env_file_is_testnet(monkeypatch, tmp_path):
    """Without --env-file, paper commands read .env.testnet — never a
    legacy .env, whose environment would be anyone's guess."""
    import trade_lab.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.testnet").write_text("# testnet env\n")
    # A legacy .env with mainnet content must be ignored entirely.
    (tmp_path / ".env").write_text("TRADE_LAB_PAPER_SANDBOX=false\n")

    loaded: list = []
    monkeypatch.setattr(
        cli_mod, "load_dotenv",
        lambda *a, **kw: loaded.append((a, kw.get("override"))),
    )
    monkeypatch.delenv("TRADE_LAB_PAPER_EXCHANGE", raising=False)

    with pytest.raises(SystemExit, match="Config error"):
        cli_mod.main(["paper-status"])

    assert loaded == [((".env.testnet",), True)]


def test_paper_missing_default_env_file_fails_loud(monkeypatch, tmp_path):
    """A bare paper command in a directory without .env.testnet must
    error with a migration hint — no silent fallback to .env."""
    from trade_lab.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TRADE_LAB_PAPER_SANDBOX=true\n")

    with pytest.raises(SystemExit, match=r"\.env\.testnet"):
        main(["paper-status"])


def test_env_file_is_loaded_before_config(monkeypatch, tmp_path):
    """The flagged file (not the default .env) feeds load_dotenv, and the
    command proceeds to the config stage."""
    import trade_lab.cli as cli_mod

    env_file = tmp_path / "custom.env"
    env_file.write_text("# empty on purpose\n")

    loaded: list = []
    monkeypatch.setattr(
        cli_mod, "load_dotenv", lambda *a, **kw: loaded.append(a),
    )
    # Force the config stage to fail deterministically: no exchange var.
    monkeypatch.delenv("TRADE_LAB_PAPER_EXCHANGE", raising=False)

    with pytest.raises(SystemExit, match="Config error"):
        cli_mod.main(["paper-status", "--env-file", str(env_file)])

    assert loaded == [(str(env_file),)]


def test_env_file_overrides_lingering_shell_vars(monkeypatch, tmp_path):
    """CRITICAL regression: --env-file must beat ambient process env.

    python-dotenv defaults to override=False; without an explicit
    override, TRADE_LAB_PAPER_* vars lingering in the shell (e.g. an
    earlier `source .env.mainnet`) would silently win over the selected
    testnet file, carrying a MAINNET config through a run the operator
    labelled testnet."""
    import trade_lab.cli as cli_mod
    from trade_lab.execution.broker import Broker, BrokerError

    # Lingering mainnet environment from an earlier debugging session.
    monkeypatch.setenv("TRADE_LAB_PAPER_EXCHANGE", "binance")
    monkeypatch.setenv("TRADE_LAB_PAPER_SANDBOX", "false")
    monkeypatch.setenv("TRADE_LAB_PAPER_ALLOW_MAINNET", "true")
    monkeypatch.setenv("TRADE_LAB_PAPER_MAINNET_LIVE_ORDERS", "true")
    monkeypatch.setenv("TRADE_LAB_PAPER_API_KEY", "MAINNET_KEY")
    monkeypatch.setenv("TRADE_LAB_PAPER_API_SECRET", "MAINNET_SECRET")

    env_file = tmp_path / "testnet.env"
    env_file.write_text(
        "TRADE_LAB_PAPER_EXCHANGE=binance\n"
        "TRADE_LAB_PAPER_SANDBOX=true\n"
        "TRADE_LAB_PAPER_ALLOW_MAINNET=false\n"
        "TRADE_LAB_PAPER_MAINNET_LIVE_ORDERS=false\n"
        "TRADE_LAB_PAPER_API_KEY=TESTNET_KEY\n"
        "TRADE_LAB_PAPER_API_SECRET=TESTNET_SECRET\n"
    )

    seen: list = []

    def fake_connect(cls, config):
        seen.append(config)
        raise BrokerError("stop here — config captured")

    monkeypatch.setattr(Broker, "connect", classmethod(fake_connect))

    with pytest.raises(SystemExit, match="Broker connection failed"):
        cli_mod.main(["paper-status", "--env-file", str(env_file)])

    assert len(seen) == 1
    cfg = seen[0]
    assert cfg.sandbox is True, "testnet file must win over lingering env"
    assert cfg.mainnet_live_orders is False
    assert cfg.api_key == "TESTNET_KEY"


# ---------------------------------------------------------------------------
# Hardened degrade paths (post-review fixes)
# ---------------------------------------------------------------------------


def test_state_corrupt_file_refused_for_mainnet(tmp_path):
    """An existing non-empty state file that cannot be parsed must not
    degrade to empty on MAINNET — the next write would restamp and
    overwrite state of unknown provenance."""
    p = tmp_path / "orders.json"
    p.write_text("{corrupt json")

    store = OrderStateStore(p, expected_env=_MAINNET)
    with pytest.raises(OrderStateEnvMismatch, match="cannot be verified"):
        store.all_entries()


def test_state_corrupt_file_still_degrades_for_testnet(tmp_path):
    """Testnet keeps the documented degrade-and-rediscover recovery."""
    p = tmp_path / "orders.json"
    p.write_text("{corrupt json")

    store = OrderStateStore(p, expected_env=_TESTNET)
    assert store.all_entries() == {}


def test_state_meta_is_reserved_in_mutators(tmp_path):
    p = tmp_path / "orders.json"
    store = OrderStateStore(p, expected_env=_TESTNET)
    store.put(_entry())

    with pytest.raises(KeyError):
        store.mark_terminal("__meta__", "closed")
    with pytest.raises(ValueError, match="reserved"):
        store.put(_entry("__meta__"))
