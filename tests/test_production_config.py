"""Pin the canonical production-config hash + cross-check execution defaults.

The whole point of ``src/trade_lab/config/production_config.py`` is to
make accidental drift loud, not silent. These tests are the contract:

* ``test_canonical_hash_pinned`` will fail if any field in
  ``PRODUCTION_CONFIG`` is changed. That failure means the engineer
  must (a) open a new ``findings/<name>.md`` documenting why the
  strategy is changing and (b) update the pinned hash here. There is
  no other legitimate reason for the hash to change.

* ``test_execution_defaults_match_*`` will fail if the execution layer
  silently diverges from the canonical config — e.g. if someone
  refactors ``compute_live_signal`` and forgets to thread a parameter
  through. Paper trading running a different config than the
  DSR-validated one is precisely the failure mode this test catches.
"""
from __future__ import annotations

import inspect

import pytest

from trade_lab.config import (
    CANONICAL_HASH,
    PRODUCTION_CONFIG,
    production_config_hash,
)


# ---------------------------------------------------------------------------
# Hash pin — the contract
# ---------------------------------------------------------------------------

# Procedure for changing this hash (ANY of: assets, lookbacks, SMA period,
# vol-target flag, fee/slippage rates, basket rebalance freq, or any other
# field of ProductionConfig):
#
#   1. Open findings/<descriptive_name>.md documenting the new config as a
#      new research cycle (counts against PROJECT_NUM_TRIALS).
#   2. Re-run walk-forward + DSR on the new config; record results in
#      that finding.
#   3. Update CLAUDE.md "Deployable strategy" section if appropriate.
#   4. ONLY THEN update _EXPECTED_HASH below.
#
# Bumping this hash without steps 1-3 is the policy violation this test
# is designed to detect at review time.
_EXPECTED_HASH = "ac8919618ca6d5c6515ad9c26437f3fe28f1b4af3d4f37aeefcf989d0bce8753"


def test_canonical_hash_pinned():
    assert CANONICAL_HASH == _EXPECTED_HASH, (
        "Production config hash changed. If this is intentional, follow "
        "the procedure in this file's docstring."
    )


def test_harness_frozen_literal_matches_pin():
    """The harness's runtime gate compares against its own hardcoded
    FROZEN_CONFIG_HASH literal (not the import-time CANONICAL_HASH,
    which recomputes from the same object and can never drift — M8).
    Both pins must point at the same value: updating one without the
    other is exactly the half-done config change this test catches.
    """
    from trade_lab.paper_trading.harness import FROZEN_CONFIG_HASH

    assert FROZEN_CONFIG_HASH == _EXPECTED_HASH, (
        "FROZEN_CONFIG_HASH in paper_trading/harness.py and "
        "_EXPECTED_HASH here must be updated together, following the "
        "procedure in this file's docstring."
    )


def test_hash_is_deterministic():
    # Same config -> same hash (sanity: the hash is not random per import).
    h1 = production_config_hash(PRODUCTION_CONFIG)
    h2 = production_config_hash(PRODUCTION_CONFIG)
    assert h1 == h2


def test_hash_changes_when_any_field_changes():
    from dataclasses import replace

    base = production_config_hash(PRODUCTION_CONFIG)
    # Pick a field that's "obviously" core to the strategy:
    different = replace(PRODUCTION_CONFIG, lookbacks=(28, 60, 90))
    assert production_config_hash(different) != base

    # And a field that's INACTIVE in the current config but recorded:
    different = replace(PRODUCTION_CONFIG, vol_lookback=15)
    assert production_config_hash(different) != base, (
        "Inactive knobs must still affect the hash — otherwise toggling "
        "use_vol_target=True later would silently inherit the default."
    )


# ---------------------------------------------------------------------------
# Field-level pins (defense in depth — the hash already covers these)
# ---------------------------------------------------------------------------

def test_assets_canonical_order():
    # Order is part of the hash; column order downstream depends on it.
    assert PRODUCTION_CONFIG.assets == (
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
    )


def test_lookbacks():
    assert PRODUCTION_CONFIG.lookbacks == (28, 60)


def test_sma_filter():
    assert PRODUCTION_CONFIG.sma_filter_periods == (200,)


def test_use_vol_target_off():
    assert PRODUCTION_CONFIG.use_vol_target is False


def test_basket_rebalance_freq_monthly():
    assert PRODUCTION_CONFIG.basket_rebalance_freq == "MS"


def test_cost_model_defaults_binance():
    # The frozen config encodes the Binance-realistic cost regime.
    # Test 2 (validation_execution) varies these but does NOT rebind
    # PRODUCTION_CONFIG.
    assert PRODUCTION_CONFIG.fee_rate == 0.001
    assert PRODUCTION_CONFIG.slippage_rate == 0.0005


# ---------------------------------------------------------------------------
# Cross-check the execution layer against the canonical config
# ---------------------------------------------------------------------------

def test_execution_signal_defaults_match_canonical():
    """``compute_live_signal`` must default to the canonical parameters.

    If this fails, paper trading is running a different config than
    the one DSR-validated in ``findings/han_28d_tsmom.md``.
    """
    from trade_lab.execution.signal import compute_live_signal

    sig = inspect.signature(compute_live_signal)
    defaults = {name: p.default for name, p in sig.parameters.items()}

    assert tuple(defaults["lookbacks"]) == PRODUCTION_CONFIG.lookbacks
    assert defaults["sma_filter_period"] == PRODUCTION_CONFIG.sma_filter_periods[0]
    assert defaults["fee_rate"] == PRODUCTION_CONFIG.fee_rate
    assert defaults["slippage_rate"] == PRODUCTION_CONFIG.slippage_rate


def test_execution_dry_run_defaults_match_canonical():
    from trade_lab.execution.dry_run import run_dry_cycle

    sig = inspect.signature(run_dry_cycle)
    defaults = {name: p.default for name, p in sig.parameters.items()}

    assert tuple(defaults["lookbacks"]) == PRODUCTION_CONFIG.lookbacks


def test_execution_live_cycle_defaults_match_canonical():
    from trade_lab.execution.live_cycle import run_live_cycle

    sig = inspect.signature(run_live_cycle)
    defaults = {name: p.default for name, p in sig.parameters.items()}

    assert tuple(defaults["lookbacks"]) == PRODUCTION_CONFIG.lookbacks


# ---------------------------------------------------------------------------
# Frozen-ness: accidental mutation must raise
# ---------------------------------------------------------------------------

def test_production_config_is_immutable():
    with pytest.raises(Exception):
        PRODUCTION_CONFIG.lookbacks = (28, 60, 90)  # type: ignore[misc]
