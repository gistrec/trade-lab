"""Write-only MySQL mirror of the local paper-trading data files.

Fleet data-locality policy: a host must be disposable — the forward-test
journal and order state must never exist ONLY on its disk. This module
mirrors ``data/journal/*.jsonl`` (line-by-line, append-only),
``data/state/*.json`` (whole file) and the harness vintage store
(``paper_trading/vintages/``, content-addressed blobs) into the fleet's
managed MySQL, and can rematerialise them on a fresh host
(``trade-lab db-restore``).

Vintages are the exact OHLCV bytes the harness saw on one decision day
(the look-ahead detector replays against them) and are not reliably
reproducible after the fact — re-fetching a lost day may return revised
history. ~267 KB/day raw, ~4x smaller gzipped.

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

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import pymysql


logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
# Vintages live outside data/ (harness, not execution) — own root.
DEFAULT_VINTAGE_ROOT = Path("paper_trading/vintages")

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
    # Content-addressed: the SHA-256 IS the identity, so it is the PK and
    # inserts are idempotent. Gzipped BLOB, never normalised into rows —
    # the detector needs the exact bytes back.
    """
    CREATE TABLE IF NOT EXISTS vintages (
        content_hash CHAR(64)      NOT NULL,
        payload      MEDIUMBLOB    NOT NULL,
        bytes_raw    INT UNSIGNED  NOT NULL,
        mirrored_at  DATETIME(3)   NOT NULL,
        PRIMARY KEY (content_hash)
    ) DEFAULT CHARSET=utf8mb4
    """,
)


class MirrorConfigError(RuntimeError):
    """The MYSQL_* env is present but unusable."""


class MirrorIntegrityError(RuntimeError):
    """Mirrored data cannot be trusted (bad content hash, hostile path).

    Distinct from MirrorConfigError so `except MirrorConfigError` cannot
    swallow data corruption along with a typo'd env var. Carries the
    count of items that DID verify, so a caller can still report them.
    """

    def __init__(self, message: str, written: int = 0) -> None:
        super().__init__(message)
        self.written = written


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
    Cleartext only via an explicit ``MYSQL_SSL_DISABLED=true``.
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
    ssl_ca = _ssl_ca_from_env()
    return MirrorConfig(
        host=host,
        port=port,
        user=user,
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=database,
        ssl_ca=ssl_ca,
    )


def _ssl_ca_from_env() -> Optional[str]:
    # None means cleartext downstream (connect passes ssl=None), so it is
    # only ever produced by the explicit opt-out flag — never by an empty
    # MYSQL_SSL_CA (a template artefact must not silently drop TLS).
    disabled_raw = os.getenv("MYSQL_SSL_DISABLED")
    ca_raw = os.getenv("MYSQL_SSL_CA")
    if disabled_raw is None:
        disabled = False
    else:
        v = disabled_raw.strip().lower()
        if v not in ("true", "false"):
            raise MirrorConfigError(
                f"MYSQL_SSL_DISABLED must be 'true' or 'false', "
                f"got {disabled_raw!r}"
            )
        disabled = v == "true"
    if disabled:
        if ca_raw is not None and ca_raw.strip():
            raise MirrorConfigError(
                "MYSQL_SSL_DISABLED=true contradicts a non-empty "
                "MYSQL_SSL_CA — drop one of them"
            )
        logger.warning(
            "db mirror: MYSQL_SSL_DISABLED=true — the MySQL connection "
            "is CLEARTEXT (credentials and journal data unencrypted)"
        )
        return None
    if ca_raw is None:
        return "/etc/ssl/certs/ca-certificates.crt"
    ssl_ca = ca_raw.strip()
    if not ssl_ca:
        raise MirrorConfigError(
            "MYSQL_SSL_CA is set but empty — set a CA bundle path, or set "
            "MYSQL_SSL_DISABLED=true to deliberately connect without TLS"
        )
    return ssl_ca


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
    repaired, and inserts FREEZE: appending onto a broken numbering
    identity mints duplicate rows (e.g. a renumbered mirror against a
    not-yet-republished file), so a drifted source sends nothing until
    an operator re-runs restore/--force.
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
            f"NOT repaired automatically; inserts frozen for this source"
        )
        to_insert = []
    return to_insert, drift


def collect_vintage_files(vintage_root: Path) -> dict[str, Path]:
    """``{content_hash: path}`` for every vintage under ``vintage_root``.

    Hash comes from the filename (``hh/hash.txt`` layout), not from
    reading the file — a rescan would otherwise hash ~22 MB every cycle.
    Contents are verified only for the ones actually inserted.
    """
    out: dict[str, Path] = {}
    if not vintage_root.exists():
        return out
    for path in sorted(vintage_root.glob("*/*.txt")):
        out[path.stem] = path
    return out


# ── reconcile / restore ──────────────────────────────────────────────

def _insert_journal_lines(
    cur, source: str, lines: list[tuple[int, str]], now: datetime
) -> None:
    # IGNORE: a concurrent mirror of the same tail must be a no-op, not
    # a PK explosion.
    cur.executemany(
        "INSERT IGNORE INTO journal_lines "
        "(source, line_no, payload, mirrored_at) "
        "VALUES (%s, %s, %s, %s)",
        [(source, n, p, now) for n, p in lines],
    )

@dataclass
class MirrorReport:
    journal_lines_inserted: int = 0
    state_files_mirrored: int = 0
    vintages_mirrored: int = 0
    drift: list[str] = field(default_factory=list)

    def summary(self) -> str:
        out = (
            f"journal lines inserted: {self.journal_lines_inserted}, "
            f"state files mirrored: {self.state_files_mirrored}, "
            f"vintages mirrored: {self.vintages_mirrored}"
        )
        if self.drift:
            out += f", DRIFT: {'; '.join(self.drift)}"
        return out


def _mirror_lock_name(cur) -> str:
    # Named locks are server-wide; scope to the schema so two databases
    # on one shared MySQL never contend. Hashed: GET_LOCK caps names at
    # 64 chars and a schema name alone can exceed that.
    cur.execute("SELECT DATABASE()")
    row = cur.fetchone()
    schema = str(row[0]) if row and row[0] else ""
    digest = hashlib.sha256(schema.encode("utf-8")).hexdigest()[:16]
    return f"trade_lab_db_mirror:{digest}"


def _acquire_mirror_lock(conn) -> bool:
    # Advisory lock: reconcile and restore mutually exclude. The races a
    # concurrent pair can produce (restore renumbering under a reconcile
    # planned on the old numbering, stale state payloads) are impossible
    # to detect after the fact — so they are prevented, not detected.
    # Limitation: a writer running a pre-lock binary bypasses this.
    with conn.cursor() as cur:
        cur.execute("SELECT GET_LOCK(%s, 0)", (_mirror_lock_name(cur),))
        row = cur.fetchone()
    return bool(row and row[0] == 1)


def _release_mirror_lock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT RELEASE_LOCK(%s)", (_mirror_lock_name(cur),))


def reconcile(
    conn,
    data_dir: Path = DEFAULT_DATA_DIR,
    vintage_root: Path = DEFAULT_VINTAGE_ROOT,
) -> MirrorReport:
    """Mirror journal/state files and harness vintages into MySQL."""
    if not _acquire_mirror_lock(conn):
        raise MirrorIntegrityError(
            "another mirror/restore pass holds the db-mirror lock — "
            "refusing a concurrent write"
        )
    try:
        return _reconcile_locked(conn, data_dir, vintage_root)
    finally:
        _release_mirror_lock(conn)


def _reconcile_locked(
    conn,
    data_dir: Path,
    vintage_root: Path,
) -> MirrorReport:
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
                _insert_journal_lines(cur, source, to_insert, now)
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

        local_vintages = collect_vintage_files(vintage_root)
        if local_vintages:
            cur.execute("SELECT content_hash FROM vintages")
            already = {row[0] for row in cur.fetchall()}
            for content_hash, path in sorted(local_vintages.items()):
                if content_hash in already:
                    continue
                raw = path.read_bytes()
                actual = hashlib.sha256(raw).hexdigest()
                if actual != content_hash:
                    # Bytes disagree with the name — never mirror that, or
                    # the mirror starts vouching for corruption.
                    msg = (
                        f"vintage {path} hashes to {actual} but is named "
                        f"{content_hash} — NOT mirrored"
                    )
                    report.drift.append(msg)
                    logger.warning("db mirror drift — %s", msg)
                    continue
                cur.execute(
                    "INSERT IGNORE INTO vintages "
                    "(content_hash, payload, bytes_raw, mirrored_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (content_hash, gzip.compress(raw), len(raw), now),
                )
                report.vintages_mirrored += 1

    conn.commit()
    return report


def restore_vintages(conn, vintage_root: Path = DEFAULT_VINTAGE_ROOT) -> int:
    """Rematerialise mirrored vintages under ``vintage_root``. Returns count.

    No ``force`` flag: a local file hashing to its own name IS the
    mirrored one and is left alone; one that does not is corruption and
    gets repaired. Every payload is re-hashed before it touches the disk;
    failures are collected and raised at the end, so a bad row neither
    passes unnoticed nor withholds the good ones.
    """
    written = 0
    corrupt: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash, payload, bytes_raw FROM vintages")
        rows = cur.fetchall()
    for content_hash, payload, bytes_raw in rows:
        try:
            raw = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            corrupt.append(f"{content_hash}: undecompressable payload ({exc})")
            continue
        actual = hashlib.sha256(raw).hexdigest()
        if actual != content_hash or len(raw) != int(bytes_raw):
            corrupt.append(
                f"{content_hash}: mirror payload hashes to {actual} "
                f"({len(raw)} bytes, expected {bytes_raw})"
            )
            continue
        target = vintage_root / content_hash[:2] / f"{content_hash}.txt"
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == content_hash:
                continue
            logger.warning(
                "db-restore: local vintage %s is corrupt — repairing from "
                "the mirror", target,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.rename(target)        # atomic, as in store_vintage
        written += 1
    if corrupt:
        raise MirrorIntegrityError(
            "%d mirrored vintage(s) failed verification and were NOT "
            "written: %s" % (len(corrupt), "; ".join(corrupt)),
            written=written,
        )
    return written


def _restore_target(data_dir: Path, source: str) -> Path:
    """Validated write target for a mirrored ``source``.

    ``source`` comes straight from the DB: a compromised mirror must not
    turn restore into an arbitrary file write, so anything that could
    land outside ``data_dir`` (absolute, drive/UNC-qualified, ``..``)
    raises before any filesystem touch.
    """
    posix, windows = PurePosixPath(source), PureWindowsPath(source)
    if (
        posix.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise MirrorIntegrityError(
            f"refusing to restore {source!r}: escapes {data_dir}"
        )
    target = data_dir / source
    root = data_dir.resolve()
    resolved = target.resolve()
    # Backstop (symlinks, oddball separators): strictly inside data_dir.
    if resolved == root or not resolved.is_relative_to(root):
        raise MirrorIntegrityError(
            f"refusing to restore {source!r}: escapes {data_dir}"
        )
    return target


def _open_staged(path: Path):
    # 0640 from the first byte (fchmod pins it against umask) — a chmod
    # after writing leaves a window where a default-umask file is
    # readable by any local user.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    os.fchmod(fd, 0o640)
    return os.fdopen(fd, "w", encoding="utf-8")


def restore(
    conn, data_dir: Path = DEFAULT_DATA_DIR, force: bool = False
) -> list[str]:
    """Rematerialise the mirrored files under ``data_dir`` (fresh host).

    Refuses to overwrite an existing non-empty file unless ``force`` —
    a live host's files are ahead of the mirror by up to one cycle, and
    silently rolling them back would be data loss.

    A restored journal file is compact (torn lines' reserved numbers
    collapse), so each written source's mirror rows are renumbered to
    the restored file. Order per source: stage the replacement next to
    the target, renumber + COMMIT, publish with an atomic rename — a DB
    failure (missing DELETE/INSERT grant, dead connection) aborts with
    the live file untouched, and a crash between commit and publish
    leaves original-file-vs-compact-mirror drift that reconcile reports
    loudly with inserts frozen (rerun with --force repairs it).
    Restore takes the same advisory lock as reconcile: a renumber
    interleaved with a live mirror pass can strand rows on the old
    numbering with no detectable drift, so concurrency is refused
    outright, not reconciled after the fact.
    """
    if not _acquire_mirror_lock(conn):
        raise MirrorIntegrityError(
            "a mirror pass holds the db-mirror lock — refusing to "
            "restore against a moving mirror"
        )
    try:
        return _restore_locked(conn, data_dir, force)
    finally:
        _release_mirror_lock(conn)


def _restore_locked(conn, data_dir: Path, force: bool) -> list[str]:
    now = datetime.now(timezone.utc)
    written: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM journal_lines")
        journal_sources = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT source FROM state_files")
        state_sources = [row[0] for row in cur.fetchall()]
        # Every source vetted before the first write: one hostile row
        # must not leave a partial restore behind.
        journal_targets = {
            s: _restore_target(data_dir, s) for s in journal_sources
        }
        state_targets = {
            s: _restore_target(data_dir, s) for s in state_sources
        }
        for source in journal_sources:
            target = journal_targets[source]
            if target.exists() and target.stat().st_size > 0 and not force:
                logger.warning(
                    "db-restore: %s exists — refusing to overwrite "
                    "(--force to override)", target,
                )
                continue
            cur.execute(
                "SELECT payload FROM journal_lines WHERE source = %s "
                "ORDER BY line_no FOR UPDATE",
                (source,),
            )
            payloads = [payload for (payload,) in cur.fetchall()]
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(target.name + ".restoring")
            try:
                with _open_staged(staged) as fh:
                    for payload in payloads:
                        fh.write(payload + "\n")
                    # Staged bytes must be durable BEFORE the renumber
                    # commits — a post-commit power loss must not leave a
                    # compact mirror against a hollow replacement.
                    fh.flush()
                    os.fsync(fh.fileno())
                cur.execute(
                    "DELETE FROM journal_lines WHERE source = %s", (source,)
                )
                _insert_journal_lines(
                    cur, source, list(enumerate(payloads, start=1)), now
                )
                # Stage → commit → atomic publish: see docstring.
                conn.commit()
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)
            written.append(source)

        for source in state_sources:
            target = state_targets[source]
            if target.exists() and target.stat().st_size > 0 and not force:
                logger.warning(
                    "db-restore: %s exists — refusing to overwrite "
                    "(--force to override)", target,
                )
                continue
            # Locking read at write time: a plain SELECT under REPEATABLE
            # READ would return the transaction's opening snapshot, not
            # the payload current now (matters only vs pre-lock writers,
            # but FOR UPDATE costs nothing under the advisory lock).
            cur.execute(
                "SELECT payload FROM state_files WHERE source = %s "
                "FOR UPDATE",
                (source,),
            )
            (payload,) = cur.fetchone()
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(target.name + ".restoring")
            try:
                with _open_staged(staged) as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)
            written.append(source)
    conn.commit()
    return written


def mirror_status_path(data_dir: Path, sandbox: bool) -> Path:
    # Same suffix rule as cycles.jsonl / cycles_mainnet.jsonl — testnet
    # and mainnet never share this file.
    suffix = "" if sandbox else "_mainnet"
    return data_dir / "journal" / f"mirror_status{suffix}.json"


def _write_mirror_status(
    data_dir: Path, sandbox: bool, error: Optional[str]
) -> None:
    """Best-effort status drop for /metrics — never raises."""
    try:
        path = mirror_status_path(data_dir, sandbox)
        prev: dict = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prev = loaded
        except (OSError, ValueError):
            pass  # first run or corrupt file — start fresh
        now_iso = datetime.now(timezone.utc).isoformat()
        status = {
            "last_attempt_at": now_iso,
            # A failure must not reset the success clock: the age metric
            # keeps growing until a mirror actually completes.
            "last_success_at": (
                now_iso if error is None else prev.get("last_success_at")
            ),
            "last_error": error,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(status) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.warning("db mirror: status file write failed", exc_info=True)


def mirror_after_cycle(
    data_dir: Path = DEFAULT_DATA_DIR,
    vintage_root: Path = DEFAULT_VINTAGE_ROOT,
    *,
    sandbox: bool,
) -> None:
    """Best-effort post-cycle mirror — never raises.

    A completed (even failed) cycle is already journaled on disk; the
    mirror must not change the cycle's exit code. Every failure mode
    lands as a structured warning, and the next cycle's full-scan
    reconcile (or a manual ``trade-lab db-mirror``) self-heals.

    Success and failure both drop a per-environment status file next to
    the journal (``mirror_status[_mainnet].json``) so /metrics and the
    netdata alarms can see mirror health; the drop itself is best-effort.
    """
    try:
        config = mirror_config_from_env()
        if config is None:
            logger.info("db mirror disabled (MYSQL_HOST unset)")
            return
        conn = connect(config)
        try:
            report = reconcile(conn, data_dir, vintage_root)
        finally:
            conn.close()
        logger.info("db mirror: %s", report.summary())
        _write_mirror_status(data_dir, sandbox, error=None)
    except Exception as exc:
        logger.warning(
            "db mirror failed (trading unaffected; the next cycle or "
            "`trade-lab db-mirror` reconciles)", exc_info=True,
        )
        _write_mirror_status(
            data_dir, sandbox, error=f"{type(exc).__name__}: {exc}"
        )
