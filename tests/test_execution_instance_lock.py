"""Tests for the per-environment single-instance placement lock.

Finding H2: idempotency in ``orders.py`` is check-then-act with no
mutual exclusion — two concurrent runs (cron + manual) both get
``OrderNotFound`` from query-before-place and both create the same
clientOrderId; Binance dedups only against a live order, so a filled
market order lets the duplicate through (doubled position on mainnet).
The lock must make the second process fail loudly instead.

No exchange involved anywhere here — this is pure fcntl.flock
semantics: a second acquire refuses while the first is held, separate
environments (separate state files) never block each other, and the
lock disappears with its holder (release() in-process, OS cleanup on
process exit).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from trade_lab.execution.instance_lock import (
    InstanceLockHeld,
    acquire_instance_lock,
    lock_path_for_state,
)


# ---------------------------------------------------------------------------
# Naming: the lock lives next to the per-environment state file
# ---------------------------------------------------------------------------


def test_lock_path_derives_from_state_path():
    assert lock_path_for_state("data/state/orders.json") == Path(
        "data/state/orders.json.lock"
    )
    assert lock_path_for_state("data/state/orders_mainnet.json") == Path(
        "data/state/orders_mainnet.json.lock"
    )


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


def test_second_acquire_refused_while_first_held(tmp_path):
    """The core guarantee: while one holder lives, the next acquire
    must raise loudly — never wait, never proceed."""
    state = tmp_path / "orders.json"
    lock = acquire_instance_lock(state)
    try:
        with pytest.raises(
            InstanceLockHeld, match="another order-placing process"
        ):
            acquire_instance_lock(state)
    finally:
        lock.release()


def test_refusal_names_the_holder_pid(tmp_path):
    """The refusal is structured enough to act on: it carries the
    holder's pid and the lock path."""
    state = tmp_path / "orders.json"
    lock = acquire_instance_lock(state)
    try:
        with pytest.raises(InstanceLockHeld) as exc_info:
            acquire_instance_lock(state)
    finally:
        lock.release()
    message = str(exc_info.value)
    assert f"pid {os.getpid()}" in message
    assert str(lock_path_for_state(state)) in message


def test_environments_never_block_each_other(tmp_path):
    """Testnet and mainnet have separate state files, hence separate
    locks — a running testnet cycle must not delay a mainnet one."""
    testnet = acquire_instance_lock(tmp_path / "orders.json")
    try:
        mainnet = acquire_instance_lock(tmp_path / "orders_mainnet.json")
        mainnet.release()
    finally:
        testnet.release()


# ---------------------------------------------------------------------------
# Lifecycle: the lock dies with its holder
# ---------------------------------------------------------------------------


def test_release_frees_the_lock(tmp_path):
    state = tmp_path / "orders.json"
    first = acquire_instance_lock(state)
    first.release()
    second = acquire_instance_lock(state)  # must not raise
    second.release()


def test_release_is_idempotent(tmp_path):
    lock = acquire_instance_lock(tmp_path / "orders.json")
    lock.release()
    lock.release()  # second release is a no-op, not a crash


def test_lock_released_when_holder_process_exits(tmp_path):
    """flock is tied to the process: when the holder exits (cleanly or
    not), the OS releases it — no stale-lock cleanup is ever needed."""
    state = tmp_path / "orders.json"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from trade_lab.execution.instance_lock import "
                "acquire_instance_lock\n"
                f"acquire_instance_lock({str(state)!r})\n"
                "print('locked', flush=True)\n"
                "sys.stdin.read()\n"  # hold until the parent closes stdin
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline().strip() == b"locked"
        # Held by a live foreign process -> refused.
        with pytest.raises(InstanceLockHeld):
            acquire_instance_lock(state)
        child.stdin.close()
        assert child.wait(timeout=10) == 0
        # Holder exited -> the OS released the flock; acquire succeeds.
        acquire_instance_lock(state).release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdout.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_state_directory_is_created(tmp_path):
    """First run on a fresh host: data/state/ does not exist yet."""
    state = tmp_path / "data" / "state" / "orders.json"
    lock = acquire_instance_lock(state)
    try:
        assert lock_path_for_state(state).exists()
    finally:
        lock.release()
