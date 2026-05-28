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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


TERMINAL_STATUSES = frozenset({"closed", "canceled", "rejected"})
NON_TERMINAL_STATUSES = frozenset({"open", "partial", "timeout", "lost_track"})
ALL_STATUSES = TERMINAL_STATUSES | NON_TERMINAL_STATUSES


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


class OrderStateStore:
    """Persistent dict of :class:`OrderStateEntry`, keyed by clientOrderId.

    Open/close per call to stay crash-safe across the bot's lifecycle.
    No long-lived file handles, no caches — every read goes to disk so
    multi-process scenarios (rare but possible: smoke test running while
    cron fires) cannot diverge.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, client_order_id: str) -> Optional[OrderStateEntry]:
        raw = self._read().get(client_order_id)
        return None if raw is None else OrderStateEntry(**raw)

    def put(self, entry: OrderStateEntry) -> None:
        """Insert or overwrite an entry. Atomic per call."""
        state = self._read()
        state[entry.client_order_id] = asdict(entry)
        self._write_atomic(state)

    def all_entries(self) -> dict[str, OrderStateEntry]:
        return {coid: OrderStateEntry(**raw) for coid, raw in self._read().items()}

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
        if client_order_id not in state:
            raise KeyError(f"Unknown client_order_id: {client_order_id!r}")
        state[client_order_id]["status"] = status
        state[client_order_id]["last_seen_at"] = utcnow_iso()
        self._write_atomic(state)

    # ------------------------------------------------------------------
    # Read / write internals
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, dict]:
        """Load the state file. Corrupt or missing → empty store."""
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
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
            logger.warning(
                "Order state file %s is corrupt JSON: %s. Treating as empty; "
                "exchange will be queried for discovery on next cycle. "
                "Inspect the file manually before deleting if you want to "
                "recover it.",
                self.path, exc,
            )
            return {}
        if not isinstance(data, dict):
            logger.warning(
                "Order state file %s root is not a dict (got %s). Treating "
                "as empty.",
                self.path, type(data).__name__,
            )
            return {}
        return data

    def _write_atomic(self, state: dict[str, dict]) -> None:
        """Write to ``{path}.tmp`` then rename — POSIX-atomic per inode.

        ``os.chmod`` runs after rename so the canonical path lands with
        mode 0640 regardless of the umask in effect.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(json.dumps(state, separators=(",", ":")).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self.path)
        os.chmod(self.path, 0o640)
