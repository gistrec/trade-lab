"""Structured (JSON) logging with a ``cycle_id`` trace field.

Call :func:`setup_logging` once at process start (the CLI entrypoint does).
Bind the cycle id for the duration of a cycle with :func:`set_cycle_id` /
:func:`cycle_context` so every log line emitted during that cycle carries the
same UUID the journal already stamps — a monitoring incident then links
straight to the log lines of that exact cycle.

JSON is the default; set ``TRADE_LAB_LOG_JSON=false`` for a human-readable
line format. Dependency-free: a small :class:`JsonFormatter`, no
``python-json-logger``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

_CYCLE_ID: ContextVar[Optional[str]] = ContextVar(
    "trade_lab_cycle_id", default=None,
)

# Standard LogRecord machinery — everything a caller attaches via
# ``logger.info(..., extra={...})`` is NOT in here and is surfaced as a field.
_STD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "cycle_id"}


def set_cycle_id(cycle_id: Optional[str]):
    """Bind ``cycle_id`` for the current context; returns a reset token."""
    return _CYCLE_ID.set(cycle_id)


def reset_cycle_id(token) -> None:
    """Undo a :func:`set_cycle_id` using its token."""
    _CYCLE_ID.reset(token)


@contextlib.contextmanager
def cycle_context(cycle_id: Optional[str]):
    """Bind ``cycle_id`` for the duration of the ``with`` block."""
    token = _CYCLE_ID.set(cycle_id)
    try:
        yield
    finally:
        _CYCLE_ID.reset(token)


class CycleIdFilter(logging.Filter):
    """Inject the context-local ``cycle_id`` onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.cycle_id = _CYCLE_ID.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per log record (UTC ISO timestamp), dependency-free."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(
                record.created, timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "cycle_id": getattr(record, "cycle_id", "-"),
            "msg": record.getMessage(),
        }
        # Surface caller-supplied extras (extra={...}).
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def setup_logging(
    *, level: Optional[str] = None, json_output: Optional[bool] = None,
) -> None:
    """Attach one structured handler to the root logger. Idempotent.

    Appends (does not clobber) so it coexists with pytest's caplog handler.
    """
    global _configured
    if _configured:
        return
    level = (level or os.environ.get("TRADE_LAB_LOG_LEVEL", "INFO")).upper()
    if json_output is None:
        json_output = (
            os.environ.get("TRADE_LAB_LOG_JSON", "true").lower() != "false"
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(CycleIdFilter())
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [cid=%(cycle_id)s] %(message)s",
        ))

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True
