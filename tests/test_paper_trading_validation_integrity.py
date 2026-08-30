"""Целостность валидационного слоя: сортировка, гейты, durability."""
from __future__ import annotations

from pathlib import Path

import pytest

from trade_lab.paper_trading.harness import FROZEN_CONFIG_HASH
from trade_lab.paper_trading.journal import HarnessLogRow, append_row


def _row(date: str, *, ladder: float = 0.5, equity: float = 10_000.0) -> HarnessLogRow:
    return HarnessLogRow(
        date=date,
        config_hash="abc123",
        vintage_content_hash="def456",
        ladder_state=ladder,
        sma_gate_open=True,
        basket_close=100.0,
        sma_value=90.0,
        prior_ladder_state=ladder,
        per_lookback_states={"28": 1, "60": 0},
        per_lookback_returns={"28": 0.01, "60": -0.01},
        target_weights={},
        current_weights={},
        intended_trades={},
        portfolio_equity=equity,
        daily_return=0.0,
        gross_position_return=0.0,
        net_position_return=0.0,
    )


# ---------------------------------------------------------------------------
# Backfill: positional metrics over file order
# ---------------------------------------------------------------------------


def test_live_metrics_sort_by_date_not_file_order():
    """The harness supports backfilling a missed day, which appends an
    EARLIER date after later ones. diff()/cummax()/rolling() are all
    positional, so file order silently corrupts flip and drawdown."""
    from trade_lab.paper_trading.fingerprint_monitor import (
        compute_live_metrics_from_journal,
    )

    # Equity peaks on 06-02 and falls on 06-03; the 06-02 row is BACKFILLED
    # afterwards, so in file order the peak arrives last.
    in_order = [_row("2026-06-01", equity=10_000.0),
                _row("2026-06-02", equity=12_000.0),
                _row("2026-06-03", equity=9_000.0)]
    backfilled = [_row("2026-06-01", equity=10_000.0),
                  _row("2026-06-03", equity=9_000.0),
                  _row("2026-06-02", equity=12_000.0)]

    a = compute_live_metrics_from_journal(
        in_order, rolling_window_days=30, annualization_factor=365)
    b = compute_live_metrics_from_journal(
        backfilled, rolling_window_days=30, annualization_factor=365)

    assert list(a["drawdown"].index) == list(b["drawdown"].index)
    assert a["drawdown"].tolist() == pytest.approx(b["drawdown"].tolist())
    # The real drawdown is 9000 vs the 12000 peak = -25%.
    assert a["drawdown"].min() == pytest.approx(-0.25)


def test_live_metrics_backfill_does_not_manufacture_flips():
    """A ladder that never moved must report no rebalance events, even
    when a backfilled row lands out of order."""
    import pandas as pd

    from trade_lab.paper_trading.fingerprint_monitor import (
        compute_live_metrics_from_journal,
    )

    # The ladder moves exactly ONCE, on 06-03. With 06-02 backfilled after
    # 06-03, file order reads 0 -> 1 -> 0 and reports TWO events.
    rows = [_row("2026-06-01", ladder=0.0),
            _row("2026-06-03", ladder=1.0),
            _row("2026-06-02", ladder=0.0)]     # backfilled last
    live = compute_live_metrics_from_journal(
        rows, rolling_window_days=30, annualization_factor=365)
    assert len(live["rebalance_events"]) == 1
    assert live["rebalance_events"].index[0] == pd.Timestamp(
        "2026-06-03", tz="UTC")


# ---------------------------------------------------------------------------
# Two gates that could not fire
# ---------------------------------------------------------------------------


def _reference(tmp_path: Path, *, config_hash: str) -> Path:
    """Reference built on the same synthetic series the fingerprint tests
    use — enough structure that the bands are non-degenerate."""
    import numpy as np
    import pandas as pd

    from trade_lab.paper_trading.fingerprint import (
        compute_reference_fingerprint, save_reference,
    )

    n = 1000
    idx = pd.date_range("2022-01-21", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, size=n))), index=idx)
    pos = pd.Series(0.0, index=idx)
    for i in range(n):
        if i % 30 == 0:
            pos.iloc[i:] = rng.choice([0.0, 0.5, 1.0])
    port_ret = pos.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)
    equity = (1.0 + port_ret).cumprod() * 10_000.0

    fp = compute_reference_fingerprint(
        basket_close=close, positions=pos, equity=equity,
        sma_series=close.rolling(200).mean(),
        window_start=idx[0], window_end=idx[-1],
        frozen_config_hash=config_hash,
    )
    out = tmp_path / "reference.json"
    save_reference(fp, out)
    return out


def test_monitor_refuses_a_reference_built_for_another_config(tmp_path):
    """Nothing checked reference.frozen_config_hash, so a fingerprint
    generated for a DIFFERENT strategy config would silently supply the
    bands every live cycle is judged against."""
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    ref = _reference(tmp_path, config_hash="0" * 64)

    with pytest.raises(ValueError, match="built for config"):
        check_journal_against_reference(log, ref)


def test_monitor_accepts_a_reference_for_the_pinned_config(tmp_path):
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    ref = _reference(tmp_path, config_hash=FROZEN_CONFIG_HASH)
    report = check_journal_against_reference(log, ref)
    assert report is not None


def test_monitor_refuses_a_missing_journal_instead_of_bootstrapping(tmp_path):
    """read_log returns [] for a path that does not exist, so a typo in
    --log-path or a cron in the wrong cwd (both CLI defaults are relative)
    left the monitor printing 'no live data yet' and exiting 0 forever —
    the shape of a healthy system with no monitoring at all."""
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )

    ref = _reference(tmp_path, config_hash=FROZEN_CONFIG_HASH)
    with pytest.raises(FileNotFoundError, match="not an empty one"):
        check_journal_against_reference(tmp_path / "typo.jsonl", ref)

    # A genuinely empty file is still the bootstrap case: the file was
    # found, there is simply nothing in it yet.
    empty = tmp_path / "journal.jsonl"
    empty.write_text("")
    report = check_journal_against_reference(empty, ref)
    assert report.journal_window is None


# ---------------------------------------------------------------------------
# Vintage durability: the evidence cannot be re-derived
# ---------------------------------------------------------------------------


def test_store_vintage_fsyncs_file_and_directory(tmp_path, monkeypatch):
    """rename() is atomic for readers but not durable on its own: after a
    power loss the entry can point at bytes that never reached the disk.
    Unlike the journal, a vintage cannot be re-derived."""
    import os

    import pandas as pd

    from trade_lab.paper_trading import vintage_store

    synced: list[str] = []
    real_fsync = os.fsync

    def _tracking_fsync(fd):
        try:
            synced.append("dir" if os.path.isdir(f"/dev/fd/{fd}") else "file")
        except OSError:                      # platform without /dev/fd
            synced.append("fd")
        return real_fsync(fd)

    monkeypatch.setattr(vintage_store.os, "fsync", _tracking_fsync)

    idx = pd.date_range("2026-06-01", periods=3, freq="D", tz="UTC")
    candles = {"BTC": pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=idx)}
    h = vintage_store.store_vintage(candles, tmp_path)

    # file, shard directory, and vintage_root (the shard is brand new).
    assert synced == ["file", "dir", "dir"], synced
    assert vintage_store.vintage_path(tmp_path, h).exists()
    # Round-trip still works.
    assert "BTC" in vintage_store.load_vintage(h, tmp_path)


# ---------------------------------------------------------------------------
# The reference builder must not stamp a recomputed hash
# ---------------------------------------------------------------------------


def test_reference_builder_stamps_the_literal_not_a_recomputed_hash():
    """It used to write frozen_config_hash=CANONICAL_HASH — recomputed
    from the very config object it was describing. On a machine with a
    hotfixed config that generates a fingerprint FOR the edited strategy
    and labels it 'frozen', with no warning."""
    source = Path("scripts/build_reference_fingerprint.py").read_text()
    assert "frozen_config_hash=FROZEN_CONFIG_HASH" in source
    assert "CANONICAL_HASH" not in source.replace(
        "# one. This script used to stamp the reference with CANONICAL_HASH,",
        "")
    # And it refuses outright when the running config has drifted.
    assert "Refusing to build a reference" in source


# ---------------------------------------------------------------------------
# A torn tail must not swallow the records written after it
# ---------------------------------------------------------------------------


def test_append_after_a_torn_write_keeps_the_new_row_readable(tmp_path):
    """Reproduces the swallow: 06-01 written, 06-02 torn by a crash, then
    06-03 appended. Without the repair the last two fuse into one invalid
    line, read_log tolerates it as the final line, and 06-03 vanishes
    while the cycle reports success."""
    from trade_lab.paper_trading.journal import read_log

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-06-02", "config_h')     # power loss
    assert [r.date for r in read_log(log)] == ["2026-06-01"]

    append_row(_row("2026-06-03"), log)
    assert [r.date for r in read_log(log)] == ["2026-06-01", "2026-06-03"]
    # And the file is left clean, so the NEXT read does not trip the
    # corrupt-line-in-the-middle guard either.
    assert all(line.startswith("{") and line.endswith("}")
               for line in log.read_text().splitlines() if line)


def test_append_preserves_a_complete_row_missing_only_its_newline(tmp_path):
    """An editor stripping the trailing newline leaves a COMPLETE record.
    Truncating it would discard real history, so it is terminated
    instead."""
    from trade_lab.paper_trading.journal import read_log

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    append_row(_row("2026-06-02"), log)
    log.write_text(log.read_text().rstrip("\n"))        # newline stripped

    append_row(_row("2026-06-03"), log)
    assert [r.date for r in read_log(log)] == [
        "2026-06-01", "2026-06-02", "2026-06-03"]


def test_corrupt_journal_raises_a_declared_error_not_a_bare_valueerror():
    """The harness CLI maps declared failures to exit 2; an uncaught
    ValueError would exit 1 and cron would misclassify the cycle."""
    from trade_lab.paper_trading.journal import JournalCorruptionError

    assert issubclass(JournalCorruptionError, RuntimeError)


def test_report_window_follows_dates_not_append_order(tmp_path):
    """Metrics are computed on chronologically sorted rows, so a report
    window taken from append order would name an end date earlier than
    the data the latest status describes."""
    from trade_lab.paper_trading.fingerprint_monitor import (
        check_journal_against_reference,
    )

    log = tmp_path / "journal.jsonl"
    for d in ("2026-06-01", "2026-06-03", "2026-06-02"):   # 06-02 backfilled
        append_row(_row(d), log)
    ref = _reference(tmp_path, config_hash=FROZEN_CONFIG_HASH)

    report = check_journal_against_reference(log, ref)
    assert report.journal_window == ("2026-06-01", "2026-06-03")


def test_store_vintage_syncs_the_parent_when_the_shard_is_new(
    tmp_path, monkeypatch,
):
    """vintage_path shards on h[:2], so the first vintage of a prefix
    creates a directory entry under vintage_root. Syncing only the shard
    leaves the file durable inside a directory whose own entry never
    landed."""
    import os

    import pandas as pd

    from trade_lab.paper_trading import vintage_store

    synced: list[str] = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        vintage_store.os, "fsync",
        lambda fd: (synced.append(fd), real_fsync(fd))[1])

    idx = pd.date_range("2026-06-01", periods=3, freq="D", tz="UTC")

    def _candles(v: float):
        return {"BTC": pd.DataFrame(
            {"open": v, "high": v, "low": v, "close": v, "volume": 1.0},
            index=idx)}

    # First vintage of its shard: file + shard dir + vintage_root.
    vintage_store.store_vintage(_candles(1.0), tmp_path)
    first = len(synced)
    assert first == 3, synced

    # A second vintage landing in an EXISTING shard syncs one dir less.
    synced.clear()
    vintage_store.store_vintage(_candles(2.0), tmp_path)
    assert len(synced) in (2, 3), synced      # 3 only if it opened a new shard


def test_monitor_cli_maps_journal_corruption_to_exit_2(tmp_path, capsys):
    """JournalCorruptionError is a RuntimeError, so without naming it the
    monitor CLI would emit a traceback and exit 1 — which is also the
    documented --fail-on-breach code."""
    from trade_lab.paper_trading.fingerprint_cli import main as monitor_main

    log = tmp_path / "journal.jsonl"
    for d in ("2026-06-01", "2026-06-02", "2026-06-03"):
        append_row(_row(d), log)
    lines = log.read_text().splitlines()
    lines[1] = '{"date": "2026-06-02", "conf'          # corrupt, mid-file
    log.write_text("\n".join(lines) + "\n")
    ref = _reference(tmp_path, config_hash=FROZEN_CONFIG_HASH)

    code = monitor_main(["--log-path", str(log), "--reference-path", str(ref)])
    assert code == 2
    assert "MONITOR ERROR" in capsys.readouterr().err


def test_newline_terminated_corruption_is_not_a_torn_append(tmp_path):
    """A torn append cannot have written its own terminator. Malformed
    AND newline-terminated means the write finished and the bytes rotted
    afterwards — corruption, not a crash."""
    from trade_lab.paper_trading.journal import (
        JournalCorruptionError, read_log,
    )

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-06-02", "conf\n')       # rotted, terminated

    with pytest.raises(JournalCorruptionError):
        read_log(log)

    # The same bytes WITHOUT the terminator are the tolerated torn tail.
    log2 = tmp_path / "journal2.jsonl"
    append_row(_row("2026-06-01"), log2)
    with open(log2, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-06-02", "conf')
    assert [r.date for r in read_log(log2)] == ["2026-06-01"]


def test_schema_invalid_row_raises_declared_corruption(tmp_path):
    """Valid JSON, wrong shape. Left as TypeError it escapes both CLIs'
    declared-corruption handling and exits 1."""
    from trade_lab.paper_trading.journal import (
        JournalCorruptionError, read_log,
    )

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-06-02"}\n')             # missing every field

    with pytest.raises(JournalCorruptionError, match="HarnessLogRow"):
        read_log(log)


def test_concurrent_appends_do_not_delete_each_others_rows(tmp_path):
    """The repair is destructive, so it must not race the append. Two
    processes observing the same torn tail would otherwise both compute
    the same cut, and the second truncate would delete the row the first
    had already written."""
    import subprocess
    import sys
    import textwrap

    log = tmp_path / "journal.jsonl"
    append_row(_row("2026-06-01"), log)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"date": "2026-06-02", "conf')         # torn tail

    repo = Path(__file__).resolve().parents[1]
    prog = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(repo / "src")!r})
        sys.path.insert(0, {str(repo / "tests")!r})
        from test_paper_trading_validation_integrity import _row
        from trade_lab.paper_trading.journal import append_row
        append_row(_row(sys.argv[1]), {str(log)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", prog, d])
             for d in ("2026-06-03", "2026-06-04")]
    for pr in procs:
        assert pr.wait(timeout=60) == 0

    from trade_lab.paper_trading.journal import read_log
    dates = sorted(r.date for r in read_log(log))
    # 06-02 was torn and is legitimately gone; both new rows survive.
    assert dates == ["2026-06-01", "2026-06-03", "2026-06-04"], dates


def test_store_vintage_syncs_every_newly_created_ancestor(tmp_path):
    """When vintage_root itself does not exist, mkdir creates it AND the
    shard; syncing only those two leaves the root's own entry in its
    parent unsynced, which loses everything below it."""
    import os

    import pandas as pd

    from trade_lab.paper_trading import vintage_store

    synced: list[str] = []
    real_fsync = os.fsync
    orig_fsync_dir = vintage_store._fsync_dir

    def _tracking(path):
        synced.append(str(path))
        return orig_fsync_dir(path)

    vintage_store._fsync_dir = _tracking
    try:
        idx = pd.date_range("2026-06-01", periods=2, freq="D", tz="UTC")
        candles = {"BTC": pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
             "volume": 1.0}, index=idx)}
        root = tmp_path / "does" / "not" / "exist"     # nothing exists yet
        vintage_store.store_vintage(candles, root)
    finally:
        vintage_store._fsync_dir = orig_fsync_dir
        del real_fsync

    # The shard, plus the parent of every directory mkdir had to create.
    assert str(root) in synced, synced
    assert str(tmp_path / "does" / "not") in synced, synced
    assert str(tmp_path / "does") in synced, synced
