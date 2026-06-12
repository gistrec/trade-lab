"""Runtime configuration loaded from environment variables.

This is the *operational* config — paths and exchange defaults the
``trade_lab.cli`` family reads at process startup. Distinct from
``production_config.py``, which is the *strategy* parameter freeze
(hash-pinned, never read from env).

``.env`` support lives at the CLI entrypoint (``trade_lab.cli.main``),
NOT at module import: importing ``trade_lab.config`` from a process
that must stay credential-free (the monitoring dashboard) must not
pull API keys from ``.env`` into its environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Runtime configuration for trade-lab."""

    data_dir: Path
    default_exchange: str
    initial_capital: float
    fee_rate: float
    slippage_rate: float


def load_config() -> Config:
    """Build a :class:`Config` from environment variables, falling back to defaults."""
    return Config(
        data_dir=Path(os.getenv("TRADE_LAB_DATA_DIR", "data")),
        default_exchange=os.getenv("TRADE_LAB_EXCHANGE", "binance"),
        initial_capital=float(os.getenv("TRADE_LAB_INITIAL_CAPITAL", "10000")),
        fee_rate=float(os.getenv("TRADE_LAB_FEE_RATE", "0.001")),
        slippage_rate=float(os.getenv("TRADE_LAB_SLIPPAGE_RATE", "0.0005")),
    )
