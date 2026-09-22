"""Independent continuous-control domain for Adaptive Primitive Consolidation.

See docs/research/EMBODIED_3D_POC.md. Importing this package does not train a model,
open a renderer, change the discrete operation registry or access old run bundles.
"""
from apc.embodied.control import (
    EmbodiedPrimitiveCall,
    LinearMotionPrimitive,
    MotionBank,
    MotionCompositionLibrary,
    MotionRecipe,
    run_episode,
    try_library,
)
from apc.embodied.world import Box, EmbodiedEnv, NavigationTask, Observation, WorldConfig

__all__ = [
    "Box", "EmbodiedEnv", "EmbodiedPrimitiveCall", "LinearMotionPrimitive", "MotionBank",
    "MotionCompositionLibrary", "MotionRecipe", "NavigationTask", "Observation", "WorldConfig",
    "run_episode", "try_library",
]
