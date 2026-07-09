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
from .broker import Broker, BrokerError, ConnectionRefused, MarketConstraints
from .signal import SignalSnapshot, SignalComputationError, compute_live_signal
from .allocator import TargetAllocation, compute_target_allocation
from .delta import (
    DeltaPlan, OrderIntent, SkippedDelta,
    compute_delta_plan, total_skipped_quote_drift,
)
from .journal import (
    Cycle, JournalEntryTooLarge, JournalEnvMismatch, JournalWriter,
    JOURNAL_SCHEMA_VERSION, MAX_LINE_BYTES, assert_journal_env,
)
from .dry_run import DryRunResult, print_dry_run, run_dry_cycle
from .live_cycle import LiveCycleResult, run_live_cycle
from .order_state import (
    OrderStateEntry, OrderStateEnvMismatch, OrderStateStore,
    TERMINAL_STATUSES, NON_TERMINAL_STATUSES,
)
from .orders import OrderResult, place_order, sort_orders_for_placement

__all__ = [
    "Broker",
    "BrokerError",
    "ConnectionRefused",
    "Cycle",
    "DeltaPlan",
    "DryRunResult",
    "JOURNAL_SCHEMA_VERSION",
    "JournalEntryTooLarge",
    "JournalEnvMismatch",
    "JournalWriter",
    "LiveCycleResult",
    "MAX_LINE_BYTES",
    "MarketConstraints",
    "NON_TERMINAL_STATUSES",
    "OrderIntent",
    "OrderResult",
    "OrderStateEntry",
    "OrderStateEnvMismatch",
    "OrderStateStore",
    "PaperConfig",
    "PaperConfigError",
    "SignalComputationError",
    "SignalSnapshot",
    "SkippedDelta",
    "TERMINAL_STATUSES",
    "TargetAllocation",
    "assert_journal_env",
    "compute_delta_plan",
    "compute_live_signal",
    "compute_target_allocation",
    "load_paper_config",
    "place_order",
    "print_dry_run",
    "run_dry_cycle",
    "run_live_cycle",
    "sort_orders_for_placement",
    "total_skipped_quote_drift",
]
