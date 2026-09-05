"""Temporary plastic capacity, allocation, and residual learning."""

from apc.plastic.allocator import Allocator, AllocatorPreset
from apc.plastic.lifecycle import (
    CompactLifecycleConfig,
    CompactLifecycleReport,
    CompactPlasticLifecyclePolicy,
)
from apc.plastic.residual import (
    ParameterBreakdown,
    execute_plastic_residual,
    get_parameter_breakdown,
    verify_frozen_invariants,
)
from apc.plastic.workspace import PlasticWorkspace

__all__ = [
    "Allocator",
    "AllocatorPreset",
    "CompactLifecycleConfig",
    "CompactLifecycleReport",
    "CompactPlasticLifecyclePolicy",
    "ParameterBreakdown",
    "PlasticWorkspace",
    "execute_plastic_residual",
    "get_parameter_breakdown",
    "verify_frozen_invariants",
]
