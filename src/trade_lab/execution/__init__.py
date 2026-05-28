"""Paper-trading / live execution layer.

Exchange-agnostic via CCXT. The entry point is :class:`Broker` in
``broker.py``. Configuration comes from environment variables — see
``paper.env.example`` at the repo root for the variable list. Never
hard-code API keys.

Default posture is **refuse-by-default to mainnet**: even when the
sandbox flag is false, the broker will not connect unless
``TRADE_LAB_PAPER_ALLOW_MAINNET=true`` is also set. This makes the
"oops, I'm sending real orders" failure mode require two
independent decisions, not one.
"""
from .config import PaperConfig, load_paper_config, PaperConfigError
from .broker import Broker, BrokerError, ConnectionRefused

__all__ = [
    "Broker",
    "BrokerError",
    "ConnectionRefused",
    "PaperConfig",
    "PaperConfigError",
    "load_paper_config",
]
