"""Tests for the persistent OrderStateStore.

Coverage focus:

* Roundtrip: put → get returns the same entry.
* Filtering: ``open_entries`` excludes terminal statuses.
* Persistence: a fresh Store instance over the same path sees prior writes.
* Atomicity: a crash during rename leaves the prior state intact.
* Corruption tolerance: a corrupt JSON file is logged and treated as
  empty (NOT raised) — exchange is the source of truth, the store is
  a fast-path cache.
* Permissions: file lands with mode 0640 so the monitoring group can
  read it without granting write.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_lab.execution.order_state import (
    NON_TERMINAL_STATUSES,
    OrderStateEntry,
    OrderStateStore,
    TERMINAL_STATUSES,
)


def _entry(coid: str = "c1", status: str = "open") -> OrderStateEntry:
    return OrderStateEntry(
        client_order_id=coid,
        symbol="BTC/USDT",
        side="buy",
        intended_amount=0.001,
        status=status,
        exchange_order_id=None,
        placed_at="2026-05-30T00:05:01+00:00",
        last_seen_at="2026-05-30T00:05:01+00:00",
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_get_unknown_returns_none(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    assert store.get("nonexistent") is None


def test_put_then_get_roundtrip(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    e = _entry("c1")
    store.put(e)
    assert store.get("c1") == e


def test_all_entries_returns_all(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    store.put(_entry("c1"))
    store.put(_entry("c2", status="closed"))
    assert set(store.all_entries().keys()) == {"c1", "c2"}


def test_open_entries_excludes_terminal(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    store.put(_entry("open1", status="open"))
    store.put(_entry("partial1", status="partial"))
    store.put(_entry("closed1", status="closed"))
    store.put(_entry("canceled1", status="canceled"))
    store.put(_entry("rejected1", status="rejected"))
    store.put(_entry("timeout1", status="timeout"))
    store.put(_entry("lost1", status="lost_track"))
    assert set(store.open_entries().keys()) == {
        "open1", "partial1", "timeout1", "lost1",
    }


def test_mark_terminal_updates_status(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    store.put(_entry("c1", status="open"))
    store.mark_terminal("c1", "closed")
    assert store.get("c1").status == "closed"


def test_mark_terminal_requires_terminal_status(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    store.put(_entry("c1"))
    with pytest.raises(ValueError, match="terminal status"):
        store.mark_terminal("c1", "open")


def test_mark_terminal_unknown_id_raises(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    with pytest.raises(KeyError):
        store.mark_terminal("ghost", "closed")


def test_mark_terminal_updates_last_seen_at(tmp_path):
    store = OrderStateStore(tmp_path / "orders.json")
    store.put(_entry("c1", status="open"))
    original_last_seen = store.get("c1").last_seen_at
    store.mark_terminal("c1", "closed")
    assert store.get("c1").last_seen_at != original_last_seen


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persists_across_instances(tmp_path):
    path = tmp_path / "orders.json"
    OrderStateStore(path).put(_entry("c1"))
    assert OrderStateStore(path).get("c1") == _entry("c1")


def test_parent_directory_auto_created(tmp_path):
    nested = tmp_path / "deep" / "nested" / "state"
    store = OrderStateStore(nested / "orders.json")
    store.put(_entry("c1"))
    assert (nested / "orders.json").exists()


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_atomic_write_preserves_old_state_on_rename_failure(tmp_path, monkeypatch):
    """If rename fails, the prior state must be intact and the tmp
    file must not pollute future reads."""
    path = tmp_path / "orders.json"
    store = OrderStateStore(path)
    store.put(_entry("first"))

    def boom(src, dst):
        raise OSError("simulated disk failure during rename")
    monkeypatch.setattr(os, "rename", boom)

    with pytest.raises(OSError, match="simulated"):
        store.put(_entry("second"))

    monkeypatch.undo()
    # The original state is intact, the failed write did not corrupt it.
    assert store.get("first") is not None
    assert store.get("second") is None


def test_partial_tmp_file_is_ignored(tmp_path):
    """A leftover .tmp file from a previous crashed write does not
    contaminate reads from the canonical path."""
    path = tmp_path / "orders.json"
    store = OrderStateStore(path)
    store.put(_entry("real"))
    (path.with_suffix(path.suffix + ".tmp")).write_text(
        '{"phantom": {"bogus": "should not be read"}}'
    )
    assert set(store.all_entries().keys()) == {"real"}


# ---------------------------------------------------------------------------
# Corruption tolerance
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_store(tmp_path):
    path = tmp_path / "orders.json"
    path.write_text("")
    assert OrderStateStore(path).all_entries() == {}


def test_missing_file_returns_empty_store(tmp_path):
    assert OrderStateStore(tmp_path / "missing.json").all_entries() == {}


def test_corrupt_json_returns_empty_with_warning(tmp_path, caplog):
    path = tmp_path / "orders.json"
    path.write_text("{not valid json[[[")
    store = OrderStateStore(path)
    with caplog.at_level("WARNING"):
        assert store.all_entries() == {}
    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_non_dict_root_returns_empty_with_warning(tmp_path, caplog):
    path = tmp_path / "orders.json"
    path.write_text('["array", "not", "dict"]')
    store = OrderStateStore(path)
    with caplog.at_level("WARNING"):
        assert store.all_entries() == {}


def test_entry_missing_field_is_skipped_not_crashed(tmp_path, caplog):
    """A valid-JSON entry missing a required field must be skipped with a
    warning, not raise TypeError out of all_entries()/open_entries() — the
    store promises corrupt state degrades to empty, and open_entries()
    runs first thing in the daily cron (regression: R5)."""
    import json
    path = tmp_path / "orders.json"
    good = {
        "client_order_id": "good", "symbol": "BTC/USDT", "side": "buy",
        "intended_amount": 0.001, "status": "open", "exchange_order_id": None,
        "placed_at": "2026-05-30T00:05:01+00:00",
        "last_seen_at": "2026-05-30T00:05:01+00:00",
    }
    missing = {k: v for k, v in good.items() if k != "side"}
    missing["client_order_id"] = "missing"
    path.write_text(json.dumps({"good": good, "missing": missing}))
    store = OrderStateStore(path)
    with caplog.at_level("WARNING"):
        entries = store.all_entries()
    assert set(entries) == {"good"}          # malformed entry dropped
    assert store.get("missing") is None
    # open_entries (first cron step) must not raise.
    assert set(store.open_entries()) == {"good"}
    assert caplog.records, "a dropped entry must be logged, not silent"


def test_entry_with_unknown_field_survives(tmp_path):
    """A newer-schema entry carrying an extra field older code doesn't
    know must still load — the unknown key is dropped (forward-compatible)
    rather than crashing the store (regression: R5)."""
    import json
    path = tmp_path / "orders.json"
    entry = {
        "client_order_id": "c1", "symbol": "BTC/USDT", "side": "buy",
        "intended_amount": 0.001, "status": "open", "exchange_order_id": None,
        "placed_at": "2026-05-30T00:05:01+00:00",
        "last_seen_at": "2026-05-30T00:05:01+00:00",
        "some_future_field": 123,
    }
    path.write_text(json.dumps({"c1": entry}))
    store = OrderStateStore(path)
    loaded = store.get("c1")
    assert loaded is not None
    assert loaded.status == "open"
    assert loaded.client_order_id == "c1"


def test_corrupt_state_does_not_block_subsequent_writes(tmp_path):
    """After a corrupt read, the next put() succeeds and the file is
    repaired with valid JSON."""
    path = tmp_path / "orders.json"
    path.write_text("{bad")
    OrderStateStore(path).put(_entry("recovery"))
    assert set(OrderStateStore(path).all_entries().keys()) == {"recovery"}


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def test_file_permissions_0640_after_write(tmp_path):
    path = tmp_path / "orders.json"
    OrderStateStore(path).put(_entry("c1"))
    mode = path.stat().st_mode & 0o777
    assert mode == 0o640


# ---------------------------------------------------------------------------
# Status taxonomy sanity
# ---------------------------------------------------------------------------


def test_terminal_and_non_terminal_are_disjoint():
    assert TERMINAL_STATUSES & NON_TERMINAL_STATUSES == frozenset()


def test_all_known_statuses_covered():
    """Quick guard: if anyone adds a status, it must land in one bucket."""
    expected = {
        "closed", "canceled", "rejected",
        "open", "partial", "timeout", "lost_track",
    }
    assert TERMINAL_STATUSES | NON_TERMINAL_STATUSES == expected
