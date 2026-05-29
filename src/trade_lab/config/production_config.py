"""Frozen production config for TSMOM(28, 60) + SMA(200) + 7-major basket.

This module is the single source of truth for the deployable strategy.
Every parameter that defines what the backtest measured (and therefore
what paper trading must replicate) lives here.

The hash exported below is the *contract*: validation tests, the
paper-trading harness, and the behavioral fingerprint all reference
``CANONICAL_HASH`` so a config drift is detected, not silently merged.

What this hash protects
-----------------------
Any of the following requires bumping the pinned hash in
``tests/test_production_config.py``:

* Asset list or order (basket composition).
* TSMOM lookbacks ``(28, 60)``.
* SMA regime gate ``(200,)`` on basket close.
* The ``use_vol_target=False`` decision.
* Cost-model rates ``fee_rate=0.001``, ``slippage_rate=0.0005``.
* Basket rebalance frequency ``"MS"`` (monthly).
* Any inactive-but-recorded knob (e.g. ``rebalance_threshold``).

Each such change is, by project rule, a **new research cycle**: it
adds to ``PROJECT_NUM_TRIALS``, invalidates prior DSR numbers
referencing the old config, and requires a new ``findings/<name>.md``
documenting the decision. The hash exists to make that contract
mechanical, not vibes-based.

What this hash does NOT cover
-----------------------------
* Code-level refactors of strategy / engine / basket that preserve
  output (the hash is parameter-level, not behavior-level — the
  behavioral fingerprint in Test 4 covers output-level drift).
* Data vintage: replaying the harness against revised historical
  candles is a known source of small divergences. Test 4's look-ahead
  detector explicitly distinguishes data-vintage drift from logic drift.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class ProductionConfig:
    """Immutable parameter bundle for the deployable strategy.

    ``frozen=True`` means accidental in-place mutation raises. To run
    a non-canonical variant (e.g. Kraken cost regime for Test 2),
    construct a new instance with ``dataclasses.replace(...)`` — the
    hash will differ, which is the desired signal.
    """

    # --- Basket composition (equal-weight; order is part of the hash) ---
    assets: Tuple[str, ...] = (
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOGE",
    )

    # --- TSMOM signal ---
    # Both elements are lookback windows in days; signal is the mean
    # of binary sign-of-return states over each, producing the ladder
    # {0, 0.5, 1.0}.
    lookbacks: Tuple[int, ...] = (28, 60)

    # SMA(200) regime gate on basket close (market-level, not per-asset).
    # Strategy zeros the signal when basket close <= SMA(200).
    sma_filter_periods: Tuple[int, ...] = (200,)

    # Han DSR 0.770 was without vol-targeting. Including the knob in
    # the hash means a future flip to True is detected as drift even
    # though it would silently change the strategy semantics.
    use_vol_target: bool = False

    # --- Vol-target layer (INACTIVE while use_vol_target=False) ---
    # Recorded so the hash is sensitive to a future toggle that would
    # otherwise inherit the strategy's default values.
    vol_lookback: int = 30
    annual_vol_target: float = 0.25
    max_position_size: float = 1.0
    rebalance_threshold: float = 0.05

    # --- Basket construction ---
    basket_rebalance_freq: str = "MS"          # monthly start
    initial_capital: float = 10_000.0

    # --- Cost model (Binance-realistic; Test 2 explicitly varies these) ---
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005

    # --- Engine conventions ---
    annualization_factor: int = 365
    signal_shift_bars: int = 1                 # signal at t -> position at t+1
    warmup_days: int = 200                     # max(SMA period, max lookback)


PRODUCTION_CONFIG = ProductionConfig()


def production_config_hash(cfg: ProductionConfig | None = None) -> str:
    """Deterministic SHA256 hash of the canonical config.

    Serialization is canonical JSON (sorted keys, no whitespace) so the
    hash is stable across Python versions and machines.
    """
    cfg = cfg if cfg is not None else PRODUCTION_CONFIG
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CANONICAL_HASH: str = production_config_hash(PRODUCTION_CONFIG)
