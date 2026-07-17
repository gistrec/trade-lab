"""Tests for the MySQL data mirror (execution/db_mirror.py).

The DB is faked (CLAUDE.md: mock external services); the planning logic
is pure and tested directly. What must hold:

* only lines past the mirrored high-water mark are sent (append-only
  incremental sync);
* a local file shorter than its mirror is loud DRIFT, never silently
  repaired;
* broken journal lines are skipped, their physical numbers reserved;
* restore refuses to overwrite existing non-empty files without force;
* the post-cycle hook NEVER raises — a mirror failure must not take a
  completed trading cycle down with it;
* credentials never leak through repr.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_lab.execution.db_mirror import (
    MirrorConfigError,
    collect_journal_lines,
    mirror_after_cycle,
    mirror_config_from_env,
    plan_journal_inserts,
    reconcile,
    restore,
)


# ── fake DB ──────────────────────────────────────────────────────────

class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._rows = []

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if s.startswith("CREATE TABLE"):
            pass
        elif s.startswith("SELECT COALESCE(MAX(line_no), 0), COUNT(*)"):
            rows = self.store["journal"].get(params[0], {})
            self._rows = [(max(rows) if rows else 0, len(rows))]
        elif s.startswith("SELECT DISTINCT source FROM journal_lines"):
            self._rows = [(k,) for k in sorted(self.store["journal"])]
        elif s.startswith("SELECT payload FROM journal_lines"):
            rows = self.store["journal"][params[0]]
            self._rows = [(p,) for _, p in sorted(rows.items())]
        elif s.startswith("SELECT source, payload FROM state_files"):
            self._rows = sorted(self.store["state"].items())
        elif s.startswith("INSERT INTO state_files"):
            self.store["state"][params[0]] = params[1]
        else:  # pragma: no cover - unexpected SQL is a test failure
            raise AssertionError(f"unexpected SQL: {s}")

    def executemany(self, sql, rows):
        assert "INSERT IGNORE INTO journal_lines" in sql
        for source, line_no, payload, _ in rows:
            self.store["journal"].setdefault(source, {}).setdefault(
                line_no, payload
            )

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self):
        self.store = {"journal": {}, "state": {}}
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.store)

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def _write_journal(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec if isinstance(rec, str) else json.dumps(rec))
            fh.write("\n")


# ── config ───────────────────────────────────────────────────────────

def test_config_unset_means_disabled(monkeypatch):
    monkeypatch.delenv("TRADE_LAB_DB_URL", raising=False)
    assert mirror_config_from_env() is None


def test_config_parses_url_encoded_credentials(monkeypatch):
    monkeypatch.setenv(
        "TRADE_LAB_DB_URL", "mysql://trade-lab:p%40ss@db.host:3307/trade-lab"
    )
    cfg = mirror_config_from_env()
    assert cfg.host == "db.host"
    assert cfg.port == 3307
    assert cfg.user == "trade-lab"
    assert cfg.password == "p@ss"
    assert cfg.database == "trade-lab"


def test_config_rejects_non_mysql_scheme(monkeypatch):
    monkeypatch.setenv("TRADE_LAB_DB_URL", "postgres://u:p@h/db")
    with pytest.raises(MirrorConfigError):
        mirror_config_from_env()


def test_config_repr_masks_password(monkeypatch):
    monkeypatch.setenv("TRADE_LAB_DB_URL", "mysql://u:supersecret@h:3306/db")
    assert "supersecret" not in repr(mirror_config_from_env())


# ── collection & planning (pure) ─────────────────────────────────────

def test_collect_skips_broken_lines_but_reserves_numbers(tmp_path):
    path = tmp_path / "cycles.jsonl"
    _write_journal(path, [{"cycle_id": "a"}, "{broken", {"cycle_id": "b"}])
    lines = collect_journal_lines(path)
    assert [n for n, _ in lines] == [1, 3]  # 2 stays reserved


def test_plan_inserts_only_past_high_water_mark():
    local = [(1, "a"), (2, "b"), (3, "c")]
    to_insert, drift = plan_journal_inserts(local, 2, 2)
    assert to_insert == [(3, "c")]
    assert drift is None


def test_plan_reports_truncation_as_drift():
    # Mirror says 3 lines up to line 3; local file only has line 1 left.
    to_insert, drift = plan_journal_inserts([(1, "a")], 3, 3)
    assert to_insert == []
    assert drift is not None and "truncation" in drift


# ── reconcile / restore round-trip ───────────────────────────────────

def test_reconcile_is_incremental_and_round_trips(tmp_path):
    data = tmp_path / "data"
    _write_journal(
        data / "journal" / "cycles.jsonl",
        [{"cycle_id": "a"}, {"cycle_id": "b"}],
    )
    _write_journal(data / "journal" / "cycles_mainnet.jsonl", [{"c": "m"}])
    (data / "state").mkdir(parents=True)
    (data / "state" / "orders.json").write_text('{"__meta__": {}}')

    conn = FakeConn()
    first = reconcile(conn, data)
    assert first.journal_lines_inserted == 3
    assert first.state_files_mirrored == 1
    assert not first.drift

    # Same files again: nothing new to send.
    assert reconcile(conn, data).journal_lines_inserted == 0

    # One appended cycle: exactly one new row.
    with open(data / "journal" / "cycles.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cycle_id": "c"}) + "\n")
    assert reconcile(conn, data).journal_lines_inserted == 1

    # Fresh-host restore reproduces the files exactly.
    fresh = tmp_path / "fresh"
    written = restore(conn, fresh)
    assert sorted(written) == [
        "journal/cycles.jsonl",
        "journal/cycles_mainnet.jsonl",
        "state/orders.json",
    ]
    assert (fresh / "journal" / "cycles.jsonl").read_text() == (
        data / "journal" / "cycles.jsonl"
    ).read_text()
    assert (fresh / "state" / "orders.json").read_text() == '{"__meta__": {}}'


def test_reconcile_flags_local_truncation(tmp_path):
    data = tmp_path / "data"
    _write_journal(
        data / "journal" / "cycles.jsonl", [{"a": 1}, {"b": 2}]
    )
    conn = FakeConn()
    reconcile(conn, data)

    _write_journal(data / "journal" / "cycles.jsonl", [{"a": 1}])
    report = reconcile(conn, data)
    assert report.drift and "cycles.jsonl" in report.drift[0]


def test_restore_refuses_existing_files_without_force(tmp_path):
    data = tmp_path / "data"
    _write_journal(data / "journal" / "cycles.jsonl", [{"a": 1}])
    conn = FakeConn()
    reconcile(conn, data)

    # The live file is ahead of the mirror — must not be rolled back.
    with open(data / "journal" / "cycles.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"b": 2}) + "\n")
    assert restore(conn, data) == []
    assert len(collect_journal_lines(data / "journal" / "cycles.jsonl")) == 2

    written = restore(conn, data, force=True)
    assert written == ["journal/cycles.jsonl"]
    assert len(collect_journal_lines(data / "journal" / "cycles.jsonl")) == 1


# ── the post-cycle hook must never raise ─────────────────────────────

def test_mirror_after_cycle_swallows_connection_failure(monkeypatch, caplog):
    monkeypatch.setenv("TRADE_LAB_DB_URL", "mysql://u:p@h:3306/db")

    def boom(config):
        raise RuntimeError("db down")

    monkeypatch.setattr("trade_lab.execution.db_mirror.connect", boom)
    mirror_after_cycle()  # must not raise
    assert any("db mirror failed" in r.message for r in caplog.records)


def test_mirror_after_cycle_disabled_without_url(monkeypatch, caplog):
    import logging

    monkeypatch.delenv("TRADE_LAB_DB_URL", raising=False)
    with caplog.at_level(logging.INFO):
        mirror_after_cycle()  # must not raise, must say it's disabled
    assert any("disabled" in r.message for r in caplog.records)
