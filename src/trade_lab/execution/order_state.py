"""Persistent client-order-ID state for idempotent placement.

The state file stores, for each known clientOrderId: the last-known
exchange-side status, the exchange's own order ID, when we last
observed it, and the intent fields needed for reconstruction.

Atomicity
=========
Writes go to ``{path}.tmp``, are fsync'd, then renamed to ``{path}``.
POSIX rename is atomic at the inode level — a crash mid-write leaves
either the old state or the new state, never a partial JSON.

Recovery
========
A corrupt state file is treated as empty with a loud warning, NOT a
hard error. The exchange is the single source of truth: query-before-
place plus discovery via ``fetch_open_orders`` will reconstruct state
from the exchange's history on the next cycle. Blocking the bot on a
recoverable corruption is worse than starting fresh and rediscovering.

Permissions
===========
Created with mode 0640. Group ownership is the operator's job (see
``execution/README.md``) — the code can't know whether the
``monitoring`` group exists on this host. The bot must own the file
to write; monitoring reads via the group bit if granted.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


TERMINAL_STATUSES = frozenset({"closed", "canceled", "rejected", "expired"})
NON_TERMINAL_STATUSES = frozenset({"open", "partial", "timeout", "lost_track"})
ALL_STATUSES = TERMINAL_STATUSES | NON_TERMINAL_STATUSES

# Reserved root key that stamps which environment (exchange + sandbox
# flag) owns this state file. Never a clientOrderId: real IDs are
# prefixed tsmom_/smoke_.
_META_KEY = "__meta__"


class OrderStateEnvMismatch(RuntimeError):
    """Raised when a state file belongs to a different environment.

    The store is keyed solely by clientOrderId, and the ID scheme has no
    environment component: a terminal testnet entry for today's
    ``tsmom_...`` ID would make the mainnet run's state fast-path skip
    the real placement entirely, and reconstruction would query the
    wrong venue. Unlike ordinary corruption (which safely degrades to
    an empty store), a cross-environment file must be a hard error.
    """


@dataclass
class OrderStateEntry:
    """Last-known state of one placed order.

    ``status`` is one of :data:`TERMINAL_STATUSES` or
    :data:`NON_TERMINAL_STATUSES`. ``exchange_order_id`` is ``None``
    until the exchange acks the placement.
    """

    client_order_id: str
    symbol: str
    side: str
    intended_amount: float
    status: str
    exchange_order_id: Optional[str]
    placed_at: str          # ISO-8601 UTC
    last_seen_at: str       # ISO-8601 UTC


def utcnow_iso() -> str:
    """ISO-8601 UTC, used to stamp last_seen_at."""
    return datetime.now(timezone.utc).isoformat()


_ENTRY_FIELDS = frozenset(f.name for f in fields(OrderStateEntry))


def _entry_from_raw(coid: str, raw: object) -> Optional[OrderStateEntry]:
    """Build an :class:`OrderStateEntry` from an on-disk dict, defensively.

    A valid-JSON entry whose shape has drifted from the current schema
    must NOT crash the store: ``OrderStateEntry(**raw)`` raises
    ``TypeError`` on a missing or unexpected field, and the docstring
    guarantees corrupt state degrades to empty with a warning, not a hard
    error — ``open_entries()`` runs first thing in the daily cron.

    * Unknown keys (a newer-schema file read by older code) are dropped
      so the entry still loads — forward-compatible.
    * A missing required field cannot be reconstructed, so the entry is
      skipped with a warning (the exchange, the source of truth, will
      rediscover it next cycle).
    """
    if not isinstance(raw, dict):
        logger.warning(
            "Order state entry %s is not a JSON object (got %s); skipping.",
            coid, type(raw).__name__,
        )
        return None
    known = {k: v for k, v in raw.items() if k in _ENTRY_FIELDS}
    try:
        return OrderStateEntry(**known)
    except TypeError as exc:
        logger.warning(
            "Order state entry %s has a drifted shape (%s); skipping it. "
            "The exchange will be queried for this order next cycle.",
            coid, exc,
        )
        return None


class OrderStateStore:
    """Persistent dict of :class:`OrderStateEntry`, keyed by clientOrderId.

    Open/close per call to stay crash-safe across the bot's lifecycle.
    No long-lived file handles, no caches — every read goes to disk so
    multi-process scenarios (rare but possible: smoke test running while
    cron fires) cannot diverge.
    """

    def __init__(
        self,
        path: Path | str,
        expected_env: Optional[dict] = None,
    ) -> None:
        """``expected_env`` — optional ``{"exchange": str, "sandbox": bool}``.

        When set, every read verifies the file's ``__meta__`` stamp
        against it (see :class:`OrderStateEnvMismatch`) and every write
        (re)stamps it. ``None`` keeps the legacy unstamped behaviour for
        callers that manage isolation themselves (tests).
        """
        self.path = Path(path)
        self.expected_env = expected_env
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, client_order_id: str) -> Optional[OrderStateEntry]:
        if client_order_id == _META_KEY:
            return None
        raw = self._read().get(client_order_id)
        return None if raw is None else _entry_from_raw(client_order_id, raw)

    def put(self, entry: OrderStateEntry) -> None:
        """Insert or overwrite an entry. Atomic per call."""
        if entry.client_order_id == _META_KEY:
            raise ValueError(
                f"{_META_KEY!r} is a reserved metadata key, "
                f"not a clientOrderId."
            )
        state = self._read()
        state[entry.client_order_id] = asdict(entry)
        self._write_atomic(state)

    def all_entries(self) -> dict[str, OrderStateEntry]:
        out: dict[str, OrderStateEntry] = {}
        for coid, raw in self._read().items():
            if coid == _META_KEY:
                continue
            entry = _entry_from_raw(coid, raw)
            if entry is not None:
                out[coid] = entry
        return out

    def open_entries(self) -> dict[str, OrderStateEntry]:
        """Non-terminal entries — what reconstruction must resolve."""
        return {
            coid: entry
            for coid, entry in self.all_entries().items()
            if entry.status not in TERMINAL_STATUSES
        }

    def mark_terminal(self, client_order_id: str, status: str) -> None:
        """Update an existing entry's status to a terminal value.

        Raises :class:`KeyError` if the ID is unknown — the caller has a
        bug if they try to terminate something that was never placed.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(
                f"mark_terminal requires a terminal status, got {status!r}; "
                f"valid: {sorted(TERMINAL_STATUSES)}"
            )
        state = self._read()
        # _META_KEY is present in the raw dict but is not an entry — it
        # must behave as unknown here, per get()/all_entries().
        if client_order_id == _META_KEY or client_order_id not in state:
            raise KeyError(f"Unknown client_order_id: {client_order_id!r}")
        state[client_order_id]["status"] = status
        state[client_order_id]["last_seen_at"] = utcnow_iso()
        self._write_atomic(state)

    # ------------------------------------------------------------------
    # Read / write internals
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, dict]:
        """Load the state file. Corrupt or missing → empty store.

        Exception: a MAINNET store (``expected_env`` with sandbox=False)
        must not degrade an existing non-empty file it cannot verify —
        treating it as empty would restamp and overwrite state whose
        provenance is unknown, and the next write makes the damage
        permanent. Testnet keeps the degrade-and-rediscover recovery
        (the exchange is the source of truth and the stakes allow it).
        """
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            self._refuse_unverifiable_mainnet_file(f"cannot be read: {exc}")
            logger.warning(
                "Could not read order state file %s: %s — treating as empty.",
                self.path, exc,
            )
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._refuse_unverifiable_mainnet_file(f"is corrupt JSON: {exc}")
            logger.warning(
                "Order state file %s is corrupt JSON: %s. Treating as empty; "
                "exchange will be queried for discovery on next cycle. "
                "Inspect the file manually before deleting if you want to "
                "recover it.",
                self.path, exc,
            )
            return {}
        if not isinstance(data, dict):
            self._refuse_unverifiable_mainnet_file(
                f"root is not a dict (got {type(data).__name__})"
            )
            logger.warning(
                "Order state file %s root is not a dict (got %s). Treating "
                "as empty.",
                self.path, type(data).__name__,
            )
            return {}
        self._check_env(data)
        return data

    def _refuse_unverifiable_mainnet_file(self, why: str) -> None:
        if self.expected_env is None or bool(self.expected_env["sandbox"]):
            return
        raise OrderStateEnvMismatch(
            f"Order state file {self.path} exists but {why} — its "
            f"environment cannot be verified for a MAINNET run. Refusing "
            f"to treat it as empty: the next write would restamp and "
            f"overwrite state that may belong to another environment. "
            f"Inspect or move the file, or point --state at a fresh path."
        )

    def _check_env(self, data: dict) -> None:
        """Hard-fail on a cross-environment state file.

        Runs on every successful read when ``expected_env`` is set.
        This is deliberately NOT part of the corrupt-degrades-to-empty
        recovery path: a parseable file stamped with the wrong
        environment is not corruption, it is an operator mixing testnet
        and mainnet state — degrading it to empty would be the silent
        failure this guard exists to prevent.
        """
        if self.expected_env is None:
            return
        expected_sandbox = bool(self.expected_env["sandbox"])
        expected_exchange = str(self.expected_env["exchange"]).lower()
        meta = data.get(_META_KEY)
        if isinstance(meta, dict) and isinstance(meta.get("sandbox"), bool):
            stamped_exchange = str(meta.get("exchange") or "").lower()
            if (meta["sandbox"] != expected_sandbox
                    or stamped_exchange != expected_exchange):
                raise OrderStateEnvMismatch(
                    f"Order state file {self.path} is stamped for "
                    f"{stamped_exchange or 'unknown'}/"
                    f"{'testnet' if meta['sandbox'] else 'MAINNET'} but the "
                    f"current config is {expected_exchange}/"
                    f"{'testnet' if expected_sandbox else 'MAINNET'}. Use a "
                    f"separate --state path per environment "
                    f"(e.g. data/state/orders_mainnet.json)."
                )
            return
        # No usable stamp. Unstamped non-empty files predate the meta
        # stamp and are testnet by construction (mainnet placement was
        # refused before the stamp existed) — a mainnet config must not
        # adopt one.
        has_entries = any(k != _META_KEY for k in data)
        if has_entries and not expected_sandbox:
            raise OrderStateEnvMismatch(
                f"Order state file {self.path} has entries but no "
                f"environment stamp — it predates mainnet support and is "
                f"presumed testnet. Refusing to reuse it for mainnet; "
                f"point --state at a fresh file "
                f"(e.g. data/state/orders_mainnet.json)."
            )

    def _write_atomic(self, state: dict[str, dict]) -> None:
        """Write to ``{path}.tmp`` then rename — POSIX-atomic per inode.

        ``os.chmod`` runs after rename so the canonical path lands with
        mode 0640 regardless of the umask in effect.
        """
        if self.expected_env is not None:
            state[_META_KEY] = {
                "exchange": str(self.expected_env["exchange"]).lower(),
                "sandbox": bool(self.expected_env["sandbox"]),
            }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(json.dumps(state, separators=(",", ":")).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self.path)
        os.chmod(self.path, 0o640)
