"""Regression test for the future ``--asof`` CLI scenario (review
finding M7): a typo like ``2026-07-19`` instead of ``-09`` must fail
loud with exit code 2 and a readable message — and must write NOTHING
(no journal row, no vintage). Before the H5 guard, such a row was
journaled under the future date; when that day arrived, the
idempotency check silently returned the poisoned row and the real
cycle never ran.

The harness-level guard itself is covered in
``test_paper_trading_harness.py::test_future_asof_raises``; this file
pins the CLI contract on top of it: ``HarnessError`` becomes
``HARNESS ERROR: ...`` on stderr plus exit code 2 (cron-visible), not
a raw traceback.
"""
from __future__ import annotations

from datetime import datetime

from trade_lab.paper_trading import harness as harness_mod
from trade_lab.paper_trading.cli import main


class _FrozenDatetime(datetime):
    """``now()`` pinned to 2024-06-05 12:00 UTC (same convention as the
    H5 tests in test_paper_trading_harness.py) so "future" is
    deterministic and immune to running across a real UTC midnight."""

    @classmethod
    def now(cls, tz=None):
        return cls(2024, 6, 5, 12, 0, tzinfo=tz)


def test_future_asof_via_cli_exits_2_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(harness_mod, "datetime", _FrozenDatetime)

    # If the guard ever regresses, the cycle proceeds to the fetch step —
    # block the live exchange so the regression fails loud here instead
    # of hitting the network (hard rule: never hit the live API in tests).
    def _no_network(*_args, **_kwargs):
        raise AssertionError(
            "future-asof guard regressed: CLI reached the live fetch"
        )

    monkeypatch.setattr(harness_mod, "fetch_ohlcv", _no_network)

    log = tmp_path / "journal.jsonl"
    vroot = tmp_path / "vintages"
    rc = main([
        "--asof", "2024-06-15",  # ten days past the frozen "today"
        "--log-path", str(log),
        "--vintage-root", str(vroot),
    ])

    assert rc == 2  # cron-visible failure code
    err = capsys.readouterr().err
    assert "HARNESS ERROR" in err  # clean message, not a raw traceback
    assert "future" in err
    # Nothing journaled or snapshotted: the poisoned-row chain (future
    # row -> idempotency check silently returns it when the day
    # arrives) can never start.
    assert not log.exists()
    assert not vroot.exists() or not any(vroot.rglob("*"))
