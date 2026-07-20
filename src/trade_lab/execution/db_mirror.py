"""Write-only MySQL mirror of the local paper-trading data files.

Fleet data-locality policy: a host must be disposable — the forward-test
journal and order state must never exist ONLY on its disk. This module
mirrors ``data/journal/*.jsonl`` (line-by-line, append-only) and
``data/state/*.json`` (whole file) into the fleet's managed MySQL, and
can rematerialise them on a fresh host (``trade-lab db-restore``).

Strictly one-way
================
The trading path never reads MySQL. The exchange stays the single
source of truth for balance/positions; the local files stay the source
of truth for the journal and state. MySQL is a durability mirror — the
environment-isolation guards (``assert_journal_env``, the order-state
env stamp) keep their teeth because nothing here feeds data back into
a cycle.

Environment isolation
=====================
Rows carry ``source`` = the file's path relative to ``data/``
(``journal/cycles.jsonl`` vs ``journal/cycles_mainnet.jsonl``,
``state/orders.json`` vs ``state/orders_mainnet.json``). Testnet and
mainnet rows are never merged; ``db-restore`` writes each source back
to its own file, byte-for-line.

Dedup contract
==============
Journal files are append-only (rows are never edited in place — see
``journal.py``), so ``(source, physical line number)`` is a stable
identity: each reconcile inserts only lines past the mirrored high-water
mark. A mirror holding MORE lines than the local file means the local
file was truncated — that is reported loudly as drift, never repaired
silently in either direction. Lines that fail to parse as JSON (crash
mid-write) are skipped with a warning, mirroring the journal reader's
own contract; their physical line numbers stay reserved.

Failure semantics
=================
``trade-lab db-mirror`` fails loud (non-zero exit) — it is the manual /
recovery entry point. The post-cycle hook (``mirror_after_cycle``) must
never take a completed trading cycle down with it: a mirror failure is
logged as a structured warning and the next cycle (or a manual
``db-mirror``) reconciles everything — the scan is always full, so
nothing is lost, only delayed. With ``MYSQL_HOST`` unset the
mirror is disabled and says so once per run.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pymysql


logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS journal_lines (
        source      VARCHAR(190)  NOT NULL,
        line_no     INT           NOT NULL,
        payload     MEDIUMTEXT    NOT NULL,
        mirrored_at DATETIME(3)   NOT NULL,
        PRIMARY KEY (source, line_no)
    ) DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS state_files (
        source      VARCHAR(190)  NOT NULL,
        payload     MEDIUMTEXT    NOT NULL,
        mirrored_at DATETIME(3)   NOT NULL,
        PRIMARY KEY (source)
    ) DEFAULT CHARSET=utf8mb4
    """,
)


class MirrorConfigError(RuntimeError):
    """The MYSQL_* env is present but unusable."""


@dataclass(frozen=True)
class MirrorConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_ca: Optional[str] = None

    def __repr__(self) -> str:  # never leak the password (CLAUDE.md rule)
        return (
            f"MirrorConfig(host={self.host!r}, port={self.port}, "
            f"user={self.user!r}, password='***', "
            f"database={self.database!r}, ssl_ca={self.ssl_ca!r})"
        )


def mirror_config_from_env() -> Optional[MirrorConfig]:
    """Build the mirror config from the fleet-standard ``MYSQL_*`` env, or
    None if unset.

    Reads discrete ``MYSQL_HOST`` / ``MYSQL_PORT`` / ``MYSQL_USER`` /
    ``MYSQL_PASSWORD`` / ``MYSQL_DB``. With ``MYSQL_HOST`` unset the mirror is
    disabled (None). ``MYSQL_SSL_CA`` selects the CA bundle for TLS
    verification, defaulting to the system bundle (trusts public CAs).
    """
    host = os.getenv("MYSQL_HOST", "").strip()
    if not host:
        return None
    user = os.getenv("MYSQL_USER", "").strip()
    database = os.getenv("MYSQL_DB", "").strip()
    if not user or not database:
        raise MirrorConfigError(
            "MYSQL_HOST is set but MYSQL_USER / MYSQL_DB is missing"
        )
    try:
        port = int(os.getenv("MYSQL_PORT", "3306"))
    except ValueError as exc:
        raise MirrorConfigError("MYSQL_PORT must be an integer") from exc
    ssl_ca = os.getenv(
        "MYSQL_SSL_CA", "/etc/ssl/certs/ca-certificates.crt"
    ).strip() or None
    return MirrorConfig(
        host=host,
        port=port,
        user=user,
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=database,
        ssl_ca=ssl_ca,
    )


def connect(config: MirrorConfig) -> pymysql.connections.Connection:
    conn = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset="utf8mb4",
        ssl={"ca": config.ssl_ca} if config.ssl_ca else None,
        connect_timeout=15,
    )
    with conn.cursor() as cur:
        for ddl in _SCHEMA:
            cur.execute(ddl)
    conn.commit()
    return conn


# ── local file collection (pure, no DB) ──────────────────────────────

def collect_journal_lines(path: Path) -> list[tuple[int, str]]:
    """Physical-line-numbered valid JSON lines of one journal file.

    Line numbers are 1-based and count PHYSICAL lines, so they stay
    stable forever in an append-only file. Unparsable lines (crash
    mid-write) are skipped with a warning — same tolerance as the
    journal reader — and their numbers stay reserved.
    """
    lines: list[tuple[int, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except ValueError:
                logger.warning(
                    "db mirror: skipping unparsable line %s:%d "
                    "(crash mid-write?)", path, line_no,
                )
                continue
            lines.append((line_no, stripped))
    return lines


def plan_journal_inserts(
    local_lines: list[tuple[int, str]],
    mirrored_max_line: int,
    mirrored_count: int,
) -> tuple[list[tuple[int, str]], Optional[str]]:
    """Lines to insert past the high-water mark, plus a drift complaint.

    Pure planning: the caller supplies the mirror's ``MAX(line_no)`` and
    ``COUNT(*)`` for this source. Drift = the mirror holds lines the
    local file no longer has (truncated/rewritten file) — reported, not
    repaired.
    """
    to_insert = [(n, p) for n, p in local_lines if n > mirrored_max_line]
    local_at_or_below_mark = sum(
        1 for n, _ in local_lines if n <= mirrored_max_line
    )
    drift = None
    if mirrored_count > local_at_or_below_mark:
        drift = (
            f"mirror holds {mirrored_count} lines up to line "
            f"{mirrored_max_line} but the local file has only "
            f"{local_at_or_below_mark} there — local truncation? "
            f"NOT repaired automatically"
        )
    return to_insert, drift


# ── reconcile / restore ──────────────────────────────────────────────

@dataclass
class MirrorReport:
    journal_lines_inserted: int = 0
    state_files_mirrored: int = 0
    drift: list[str] = field(default_factory=list)

    def summary(self) -> str:
        out = (
            f"journal lines inserted: {self.journal_lines_inserted}, "
            f"state files mirrored: {self.state_files_mirrored}"
        )
        if self.drift:
            out += f", DRIFT: {'; '.join(self.drift)}"
        return out


def reconcile(conn, data_dir: Path = DEFAULT_DATA_DIR) -> MirrorReport:
    """Mirror every journal/state file under ``data_dir`` into MySQL."""
    now = datetime.now(timezone.utc)
    report = MirrorReport()

    with conn.cursor() as cur:
        for path in sorted(data_dir.glob("journal/*.jsonl")):
            source = path.relative_to(data_dir).as_posix()
            cur.execute(
                "SELECT COALESCE(MAX(line_no), 0), COUNT(*) "
                "FROM journal_lines WHERE source = %s",
                (source,),
            )
            mirrored_max, mirrored_count = cur.fetchone()
            to_insert, drift = plan_journal_inserts(
                collect_journal_lines(path), int(mirrored_max),
                int(mirrored_count),
            )
            if drift:
                report.drift.append(f"{source}: {drift}")
                logger.warning("db mirror drift — %s: %s", source, drift)
            if to_insert:
                # IGNORE: a concurrent mirror of the same tail must be a
                # no-op, not a PK explosion.
                cur.executemany(
                    "INSERT IGNORE INTO journal_lines "
                    "(source, line_no, payload, mirrored_at) "
                    "VALUES (%s, %s, %s, %s)",
                    [(source, n, p, now) for n, p in to_insert],
                )
                report.journal_lines_inserted += len(to_insert)

        for path in sorted(data_dir.glob("state/*.json")):
            source = path.relative_to(data_dir).as_posix()
            payload = path.read_text(encoding="utf-8")
            cur.execute(
                "INSERT INTO state_files (source, payload, mirrored_at) "
                "VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload), "
                "mirrored_at = VALUES(mirrored_at)",
                (source, payload, now),
            )
            report.state_files_mirrored += 1

    conn.commit()
    return report


def restore(
    conn, data_dir: Path = DEFAULT_DATA_DIR, force: bool = False
) -> list[str]:
    """Rematerialise the mirrored files under ``data_dir`` (fresh host).

    Refuses to overwrite an existing non-empty file unless ``force`` —
    a live host's files are ahead of the mirror by up to one cycle, and
    silently rolling them back would be data loss.
    """
    written: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM journal_lines")
        journal_sources = [row[0] for row in cur.fetchall()]
        for source in journal_sources:
            target = data_dir / source
            if target.exists() and target.stat().st_size > 0 and not force:
                logger.warning(
                    "db-restore: %s exists — refusing to overwrite "
                    "(--force to override)", target,
                )
                continue
            cur.execute(
                "SELECT payload FROM journal_lines WHERE source = %s "
                "ORDER BY line_no",
                (source,),
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                for (payload,) in cur.fetchall():
                    fh.write(payload + "\n")
            written.append(source)

        cur.execute("SELECT source, payload FROM state_files")
        for source, payload in cur.fetchall():
            target = data_dir / source
            if target.exists() and target.stat().st_size > 0 and not force:
                logger.warning(
                    "db-restore: %s exists — refusing to overwrite "
                    "(--force to override)", target,
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(payload)
            # Same contract as OrderStateStore: owner rw, group r.
            os.chmod(target, 0o640)
            written.append(source)
    return written


def mirror_after_cycle(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Best-effort post-cycle mirror — never raises.

    A completed (even failed) cycle is already journaled on disk; the
    mirror must not change the cycle's exit code. Every failure mode
    lands as a structured warning, and the next cycle's full-scan
    reconcile (or a manual ``trade-lab db-mirror``) self-heals.
    """
    try:
        config = mirror_config_from_env()
        if config is None:
            logger.info("db mirror disabled (MYSQL_HOST unset)")
            return
        conn = connect(config)
        try:
            report = reconcile(conn, data_dir)
        finally:
            conn.close()
        logger.info("db mirror: %s", report.summary())
    except Exception:
        logger.warning(
            "db mirror failed (trading unaffected; the next cycle or "
            "`trade-lab db-mirror` reconciles)", exc_info=True,
        )
