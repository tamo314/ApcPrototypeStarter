"""Distillation, merging, pruning, and shadow validation."""

from apc.consolidation.distill import (
    ConsolidationConfig,
    ConsolidationReport,
    consolidate,
    measure_activity,
)

__all__ = [
    "ConsolidationConfig",
    "ConsolidationReport",
    "consolidate",
    "measure_activity",
]
