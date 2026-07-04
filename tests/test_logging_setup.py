"""Tests for structured JSON logging + cycle_id trace propagation."""
from __future__ import annotations

import io
import json
import logging
import sys

import pytest

import trade_lab.logging_setup as ls


@pytest.fixture(autouse=True)
def _clean_cycle_id():
    # run_live_cycle/run_dry_cycle set the cycle_id and never reset it (a
    # one-shot cron process exits right after), so in a multi-test process the
    # contextvar can carry a leftover id. Pin it to None around each test.
    token = ls.set_cycle_id(None)
    try:
        yield
    finally:
        ls.reset_cycle_id(token)


def _rec(name="t", level=logging.INFO, msg="m", args=()):
    return logging.LogRecord(name, level, "f", 1, msg, args, None)


def test_cycle_id_filter_default_is_dash():
    rec = _rec()
    ls.CycleIdFilter().filter(rec)
    assert rec.cycle_id == "-"


def test_cycle_context_sets_and_resets():
    with ls.cycle_context("abc123"):
        inside = _rec()
        ls.CycleIdFilter().filter(inside)
        assert inside.cycle_id == "abc123"
    after = _rec()
    ls.CycleIdFilter().filter(after)
    assert after.cycle_id == "-"


def test_set_reset_cycle_id():
    tok = ls.set_cycle_id("xyz")
    try:
        rec = _rec()
        ls.CycleIdFilter().filter(rec)
        assert rec.cycle_id == "xyz"
    finally:
        ls.reset_cycle_id(tok)


def test_json_formatter_shape():
    rec = _rec("mylogger", logging.WARNING, "hello %s", ("world",))
    rec.cycle_id = "cid-1"
    d = json.loads(ls.JsonFormatter().format(rec))
    assert d["level"] == "WARNING"
    assert d["logger"] == "mylogger"
    assert d["cycle_id"] == "cid-1"
    assert d["msg"] == "hello world"
    assert d["ts"].endswith("+00:00")  # UTC ISO


def test_json_formatter_includes_extras_and_exc():
    rec = _rec("l", logging.ERROR, "boom")
    rec.cycle_id = "-"
    rec.symbol = "BTC/USDT"  # caller extra
    try:
        raise ValueError("nope")
    except ValueError:
        rec.exc_info = sys.exc_info()
    d = json.loads(ls.JsonFormatter().format(rec))
    assert d["symbol"] == "BTC/USDT"
    assert "ValueError: nope" in d["exc"]


def test_setup_logging_idempotent():
    root = logging.getLogger()
    before_handlers, before_level, before_cfg = (
        list(root.handlers), root.level, ls._configured,
    )
    try:
        ls._configured = False
        ls.setup_logging(json_output=True)
        one = len(root.handlers)
        ls.setup_logging(json_output=True)  # second call is a no-op
        two = len(root.handlers)
        assert one == len(before_handlers) + 1
        assert two == one
    finally:
        root.handlers[:] = before_handlers
        root.setLevel(before_level)
        ls._configured = before_cfg


def test_end_to_end_json_line_carries_cycle_id():
    logger = logging.getLogger("trade_lab.test_e2e")
    logger.propagate = False
    logger.handlers[:] = []
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.addFilter(ls.CycleIdFilter())
    h.setFormatter(ls.JsonFormatter())
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    try:
        with ls.cycle_context("run-42"):
            logger.info("placed order", extra={"symbol": "ETH/USDT"})
        d = json.loads(buf.getvalue().strip())
        assert d["cycle_id"] == "run-42"
        assert d["symbol"] == "ETH/USDT"
        assert d["msg"] == "placed order"
    finally:
        logger.handlers[:] = []
