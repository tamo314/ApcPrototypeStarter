"""Conditional Neural Primitives (CNP) v1 implementation surface.

This package is intentionally separate from APC's token-operation registry.  CNP
uses typed continuous set states and explicit conditional calls; it does not
extend or mutate the legacy :mod:`apc.environments` operation IDs.
"""

from apc.cnp.contracts import (
    CNPPrimitiveCall,
    CNPRecipe,
    SelectArguments,
    SelectionResult,
    SetState,
)
from apc.cnp.primitive import ConditionalSelectPrimitive, ResidualAdapter

__all__ = [
    "CNPPrimitiveCall",
    "CNPRecipe",
    "ConditionalSelectPrimitive",
    "ResidualAdapter",
    "SelectArguments",
    "SelectionResult",
    "SetState",
]
