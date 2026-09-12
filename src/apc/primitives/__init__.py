"""Primitive representation, persistent bank, and sparse router."""

from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import (
    CompositionLibrary,
    CompositionRecipe,
    execute_composition_recipe,
)
from apc.primitives.composition_search import (
    SearchResult,
    search_composition_recipe,
)
from apc.primitives.primitive import (
    CD_DPCA_ARCHITECTURE_SIGNATURE,
    PRIMITIVE_TYPE_REGISTRY,
    CDDPCAPrimitive,
    CDDPCAPrimitiveConfig,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PointwisePrimitive,
    Primitive,
    PrimitiveBase,
    PrimitiveConfig,
    PrimitiveStatus,
    ReverseRelativePrimitive,
    ReverseRelativePrimitiveConfig,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
    build_primitive_from_config_dict,
)

__all__ = [
    "CDDPCAPrimitive",
    "CDDPCAPrimitiveConfig",
    "CD_DPCA_ARCHITECTURE_SIGNATURE",
    "CompositionLibrary",
    "CompositionRecipe",
    "ContentDecoupledDiscretePositionalCrossAttentionPrimitive",
    "ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig",
    "CrossPositionLengthBiasPrimitive",
    "CrossPositionLengthBiasPrimitiveConfig",
    "CrossPositionPrimitive",
    "CrossPositionPrimitiveConfig",
    "PRIMITIVE_TYPE_REGISTRY",
    "PointwisePrimitive",
    "Primitive",
    "PrimitiveBase",
    "PrimitiveBank",
    "PrimitiveConfig",
    "PrimitiveStatus",
    "ReverseRelativePrimitive",
    "ReverseRelativePrimitiveConfig",
    "SearchResult",
    "ShiftRelativePrimitive",
    "ShiftRelativePrimitiveConfig",
    "build_primitive_from_config_dict",
    "execute_composition_recipe",
    "search_composition_recipe",
]
