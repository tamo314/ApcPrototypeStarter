"""Independent reference execution and deterministic downstream CNP functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from apc.cnp.contracts import SelectArguments, SelectionResult, SetState
from apc.cnp.data import CNPWorld, phi


def reference_select(
    state: SetState, arguments: SelectArguments, world: CNPWorld
) -> SelectionResult:
    """Evaluate the private-world selection relation without a learned fallback path."""

    arguments.validate_batch_size(state.batch_size)
    delta = phi(state.values) - phi(arguments.query).unsqueeze(1)
    distance = torch.einsum("bni,ij,bnj->bn", delta, world.matrix, delta) / 16.0
    logits = (arguments.threshold.unsqueeze(1) - distance).to(torch.float32)
    selected = state.valid & (logits >= 0.0)
    return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))


def wrong_family_select(
    state: SetState, arguments: SelectArguments, world: CNPWorld
) -> SelectionResult:
    """Reference wrong-family control: select the complement of the correct relation."""

    correct = reference_select(state, arguments, world)
    logits = -correct.logits
    selected = state.valid & (logits >= 0.0)
    return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))


def count_selected(state: SetState) -> torch.Tensor:
    """Deterministically count selected elements per batch item."""

    return state.valid.sum(dim=1, dtype=torch.int64)


def sum_first_selected(state: SetState) -> torch.Tensor:
    """Deterministically sum coordinate zero of selected elements, including empty sets."""

    return (state.values[..., 0] * state.valid.to(state.values.dtype)).sum(dim=1)


@dataclass(frozen=True)
class ReferenceControlResult:
    """Reference outputs for correct, none, wrong-family, and wrong-argument controls."""

    correct: SelectionResult
    none: SetState
    wrong_family: SelectionResult
    wrong_argument: SelectionResult


def reference_controls(
    state: SetState,
    arguments: SelectArguments,
    wrong_arguments: SelectArguments,
    world: CNPWorld,
) -> ReferenceControlResult:
    """Create causal controls from fixed arguments without adaptive example selection."""

    return ReferenceControlResult(
        correct=reference_select(state, arguments, world),
        none=state,
        wrong_family=wrong_family_select(state, arguments, world),
        wrong_argument=reference_select(state, wrong_arguments, world),
    )
