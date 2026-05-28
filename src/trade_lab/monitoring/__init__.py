"""Read-only monitoring dashboard for the paper-trading bot.

The monitoring process reads the bot's append-only journal and exposes
a Streamlit dashboard. By design it has NO write capability, NO direct
exchange access, and NO knowledge of API credentials. The bot writes,
monitoring reads — never the other way.

The data source (``data_source.py``) is intentionally isolated from
``trade_lab.execution``: monitoring runs as a separate Unix user that
cannot import broker/credential code. The journal format is the only
contract between the two layers.
"""
from .data_source import (
    JournalReader, ReadStats, Staleness, parse_iso,
    cycle_orders_executed, KNOWN_SCHEMA_VERSIONS,
)

__all__ = [
    "JournalReader",
    "KNOWN_SCHEMA_VERSIONS",
    "ReadStats",
    "Staleness",
    "cycle_orders_executed",
    "parse_iso",
]
