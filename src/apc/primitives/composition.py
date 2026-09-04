"""Composition Library and multi-step recipe execution.

Phase A.1 Post-Diagnostic: Branch B Integration.

Provides:
- `CompositionRecipe`: Dataclass representing an ordered chain of primitive calls.
- `CompositionLibrary`: Dedicated registry storing named and procedural recipes,
  strictly separate from `PrimitiveBank`.
- `execute_composition_recipe`: Sequential execution engine piping latent representations
  across ordered primitive steps over the frozen task-blind Stable Core and PrimitiveBank,
  with strict sparse call tracking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from apc.core.data import collate_content_only_batch, pad_token_sequences
from apc.environments.generator import Example, oracle_calls_for_example
from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)

if TYPE_CHECKING:
    from apc.evaluation.shared_encoder_architecture_gate import SharedContentEncoder

__all__ = [
    "CompositionLibrary",
    "CompositionRecipe",
    "execute_composition_recipe",
    "extract_argument_value",
]


def extract_argument_value(operation: str, call: PrimitiveCall) -> Any:
    """Extract the primary typed argument value for a canonical primitive call."""
    if operation == "SHIFT":
        return call.arguments.get("amount", 0)
    if operation == "SELECT":
        return call.arguments.get("indices", ())
    if operation == "COUNT":
        return call.arguments.get("target", 0)
    if operation == "BIND":
        return call.arguments.get("query_key", 0)
    return None


@dataclass(frozen=True)
class CompositionRecipe:
    """An ordered sequence of `PrimitiveCall`s that defines a composite computation."""

    name: str
    steps: tuple[PrimitiveCall, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"CompositionRecipe '{self.name}' must contain at least one step.")

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(step.operation for step in self.steps)


class CompositionLibrary:
    """Registry for composite execution recipes.

    Kept distinct from `PrimitiveBank` (AGENTS.md requirement): a composition of known
    primitives is not a new primitive and allocates zero persistent weights.
    """

    def __init__(self, recipes: Sequence[CompositionRecipe] | None = None) -> None:
        self._recipes: dict[str, CompositionRecipe] = {}
        for recipe in recipes or []:
            self.register_recipe(recipe)

    def __len__(self) -> int:
        return len(self._recipes)

    def names(self) -> list[str]:
        return sorted(self._recipes.keys())

    def register_recipe(self, recipe: CompositionRecipe) -> None:
        """Register a recipe under its unique name."""
        if recipe.name in self._recipes:
            raise ValueError(f"Recipe '{recipe.name}' is already registered.")
        self._recipes[recipe.name] = recipe

    def get_recipe(self, name: str) -> CompositionRecipe:
        """Retrieve a registered recipe by name."""
        try:
            return self._recipes[name]
        except KeyError:
            raise KeyError(f"Recipe '{name}' not found in CompositionLibrary.") from None

    def has_recipe(self, name: str) -> bool:
        return name in self._recipes


def _encode_intermediate_tokens(
    core: SharedContentEncoder,
    token_sequences: Sequence[Sequence[int]],
    device: torch.device,
) -> torch.Tensor:
    """Encode intermediate tokens through the frozen task-blind Stable Core without task tokens."""
    framed = []
    for seq in token_sequences:
        framed.append((core.tokens.bos,) + tuple(seq) + (core.tokens.sep,))
    padded = pad_token_sequences(framed, core.tokens.pad, device=device)
    hidden = core.model.encode(padded)
    max_len = max(len(s) for s in token_sequences)
    return hidden[:, 1 : 1 + max_len, :]


def _encode_initial_content(
    core: Any,
    examples: Sequence[Example],
) -> tuple[torch.Tensor, list[int]]:
    """Task-blind initial content encoding through the frozen Stable Core."""
    content_lengths = [len(example.input_tokens) for example in examples]
    lmax = max(content_lengths)
    content_ids = collate_content_only_batch(examples, core.tokens, device=core.device)
    content_state = core.model.encode(content_ids)
    content_features = content_state[:, 1 : 1 + lmax, :]
    return content_features, content_lengths


def execute_composition_recipe(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    examples: Sequence[Example],
    *,
    recipe: CompositionRecipe | None = None,
) -> torch.Tensor:
    """Execute sequential recipes by piping latent activations across ordered primitive calls.

    Strict Invariants Enforced:
    1. Task-blind core encoding: no task, operation, or argument tokens enter `core.model.encode`.
    2. Strict sparse execution: only the participating primitives in `bank` receive forward calls;
       all other primitives retain zero forward calls.
    3. Zero plastic allocation: no temporary parameters or workspace modules are touched.

    Args:
        core: SharedContentEncoder (task-blind content encoder).
        bank: PrimitiveBank holding the compact heterogeneous primitives.
        op_to_id: Mapping from canonical operation name to bank primitive_id.
        examples: Batch of input examples to evaluate.
        recipe: Optional static `CompositionRecipe`. If None, per-example oracle recipes
            from `oracle_calls_for_example(ex)` are used.

    Returns:
        Final output logits of shape `[batch, final_output_length, vocab_size]`.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    batch_size = len(examples)
    device = core.device

    # Resolve per-example call sequences
    if recipe is not None:
        calls_per_example: list[tuple[PrimitiveCall, ...]] = [recipe.steps] * batch_size
        num_steps = len(recipe.steps)
    else:
        calls_per_example = [oracle_calls_for_example(ex) for ex in examples]
        num_steps = len(calls_per_example[0])
        if not all(len(calls) == num_steps for calls in calls_per_example):
            raise ValueError("All examples in batch must have identical recipe depth.")

    # 1. Initial task-blind content encoding
    h_current, current_lengths = _encode_initial_content(core, examples)

    # 2. Sequential execution through ordered primitive steps
    final_logits: torch.Tensor | None = None

    for step_idx in range(num_steps):
        step_calls = [calls_per_example[b][step_idx] for b in range(batch_size)]
        op_name = step_calls[0].operation
        if not all(call.operation == op_name for call in step_calls):
            raise ValueError(f"Step {step_idx} has heterogeneous operations across batch items.")

        if op_name not in op_to_id:
            raise KeyError(f"Operation '{op_name}' not mapped in op_to_id.")

        prim_id = op_to_id[op_name]
        primitive = bank.get(prim_id)

        op_def = get_operation(op_name)
        output_lengths = [op_def.output_length(length) for length in current_lengths]

        # Extract typed arguments for parameterized primitives
        is_parameterized = bool(op_def.required_argument_names)
        if is_parameterized:
            arg_values: list[Any] | None = [
                extract_argument_value(op_name, call) for call in step_calls
            ]
        else:
            arg_values = None

        # Execute only the selected primitive
        if isinstance(
            primitive,
            (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
        ):
            logits = primitive(h_current, current_lengths, output_lengths, arg_values)
        else:
            logits = primitive(h_current)

        if step_idx < num_steps - 1:
            # Intermediate step: pipe transformed state to next step via task-blind encoder
            predictions = logits.argmax(dim=-1)
            token_seqs = [
                tuple(predictions[b, : output_lengths[b]].tolist()) for b in range(batch_size)
            ]
            h_current = _encode_intermediate_tokens(core, token_seqs, device)
            current_lengths = list(output_lengths)
        else:
            # Final step: record output logits
            final_logits = logits

    assert final_logits is not None
    return final_logits
