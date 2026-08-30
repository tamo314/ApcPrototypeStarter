"""Distillation, merging, pruning, and shadow validation."""

from apc.consolidation.distill import (
    ConsolidationConfig,
    ConsolidationReport,
    consolidate,
    measure_activity,
)
from apc.consolidation.shadow import (
    ShadowValidationConfig,
    ShadowValidationReport,
    evaluate_shadow,
    run_shadow_validation,
)

__all__ = [
    "ConsolidationConfig",
    "ConsolidationReport",
    "consolidate",
    "measure_activity",
    "ShadowValidationConfig",
    "ShadowValidationReport",
    "evaluate_shadow",
    "run_shadow_validation",
]
