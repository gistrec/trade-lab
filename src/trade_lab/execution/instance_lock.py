"""Per-environment single-instance lock for order-placing commands.

Why this exists
===============
Idempotency in :mod:`.orders` is check-then-act: state fast-path, then
query-before-place, then ``create_order``. There is no mutual exclusion
between processes — two concurrent runs (the daily cron plus a manual
invocation) each spend seconds computing the signal, both get
``OrderNotFound`` from query-before-place, and both call
``create_order`` with the same clientOrderId. Binance deduplicates
``newClientOrderId`` only against a *live* order: a market order fills
instantly and frees the ID, so the second create places a second real
order — a doubled position on mainnet. The deterministic clientOrderId
scheme cannot help here; concurrent runs must be mutually exclusive.

Scope
=====
One exclusive ``fcntl.flock`` per environment. The lock file lives next
to the resolved order-state file (``orders.json.lock`` on testnet,
``orders_mainnet.json.lock`` on mainnet): ``paper-place-orders`` and
``paper-place-test-order`` of one environment share the state file and
therefore share the lock, while testnet and mainnet have separate state
files and never block each other.

Lifecycle
=========
The lock is held until the process exits — acquired handles are pinned
in a module-level registry so a caller cannot accidentally drop the fd
and release early. The OS releases a flock on process termination,
including crash and SIGKILL, so there is no stale-lock cleanup and no
pid-liveness heuristic. The ``.lock`` file itself is never unlinked:
deleting a flocked path would let a third process lock a fresh inode
while the original holder is still running.
"""
from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path


class InstanceLockHeld(RuntimeError):
    """Another order-placing process of this environment is running."""


@dataclass(eq=False)
class InstanceLock:
    """A held lock. Released by the OS at process exit; :meth:`release`
    exists so tests can simulate a holder going away in-process."""

    path: Path
    fd: int

    def release(self) -> None:
        if self.fd < 0:
            return
        os.close(self.fd)  # closing the fd drops the flock
        self.fd = -1
        try:
            _HELD.remove(self)
        except ValueError:
            pass


# Pins every acquired lock for the life of the process — the guarantee
# "held until exit" must not depend on call sites keeping a reference.
_HELD: list[InstanceLock] = []


def lock_path_for_state(state_path: Path | str) -> Path:
    """``data/state/orders_mainnet.json`` → ``.../orders_mainnet.json.lock``."""
    p = Path(state_path)
    return p.with_name(p.name + ".lock")


def acquire_instance_lock(state_path: Path | str) -> InstanceLock:
    """Take the exclusive per-environment placement lock, non-blocking.

    Raises :class:`InstanceLockHeld` immediately if another process
    holds it — the caller must refuse loudly, never wait: by the time a
    blocked second run woke up, its signal snapshot and balance reads
    would be stale, and a queued duplicate cycle is exactly the failure
    mode this lock exists to prevent.
    """
    lock_file = lock_path_for_state(state_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o640)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = "pid unknown"
        try:
            raw = os.read(fd, 64).decode("ascii", "replace").strip()
            if raw:
                holder = f"pid {raw}"
        except OSError:
            pass
        os.close(fd)
        raise InstanceLockHeld(
            f"another order-placing process ({holder}) holds {lock_file}. "
            f"Concurrent runs can both pass query-before-place and create "
            f"the same clientOrderId twice — wait for the other run "
            f"(cron?) to finish, then retry."
        )
    # Best-effort holder pid for the refusal message of the next comer.
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode("ascii"))
    lock = InstanceLock(path=lock_file, fd=fd)
    _HELD.append(lock)
    return lock
