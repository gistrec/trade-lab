"""Frozen, hash-pinned production config (single source of truth).

The validation phase treats the candidate strategy as locked. Every
test in ``findings/validation_*.md`` reads ``PRODUCTION_CONFIG`` and
its ``CANONICAL_HASH`` so a change to the config is loud, not silent.
Any drift between the hash here and the hash pinned in tests is a
new research cycle, not a config refactor.
"""
from .production_config import (
    CANONICAL_HASH,
    PRODUCTION_CONFIG,
    ProductionConfig,
    production_config_hash,
)

__all__ = [
    "CANONICAL_HASH",
    "PRODUCTION_CONFIG",
    "ProductionConfig",
    "production_config_hash",
]
