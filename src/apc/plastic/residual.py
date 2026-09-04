"""Residual plastic execution layer (Task A1-B005 / Milestone B-M5).

Implements residual plastic capacity execution from
`docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md` section 9
and `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 10:
    F_target(h) ~= F_existing(h) + R_plastic(h)

Key Invariants:
1. Stable Core is strictly task-blind and frozen (`requires_grad == False`).
2. Persistent bank primitives are strictly frozen (`requires_grad == False`).
3. Base recipe logits F_existing(h) are evaluated with no gradients (`torch.no_grad()` / detached).
4. All gradients flow exclusively to temporary parameters inside `PlasticWorkspace`.
5. Strict parameter accounting separates resident core, resident bank, and temporary capacity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from apc.core.data import collate_content_only_batch
from apc.environments.generator import Example
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)


@dataclass(frozen=True)
class ParameterBreakdown:
    """Accounting breakdown of network parameters across modules."""

    core_parameters: int
    bank_parameters: int
    temporary_parameters: int
    active_parameters: int
    trainable_parameters: int

    @property
    def total_resident_parameters(self) -> int:
        return self.core_parameters + self.bank_parameters + self.temporary_parameters


def verify_frozen_invariants(core: Any, bank: PrimitiveBank) -> bool:
    """Verify that all parameters in Stable Core and PrimitiveBank are frozen.

    Returns True if strictly all parameters have `requires_grad == False`.
    Raises AssertionError if any parameter is trainable.
    """
    # Verify core model
    for name, param in core.model.named_parameters():
        if param.requires_grad:
            raise AssertionError(
                f"Stable Core parameter '{name}' is not frozen (requires_grad=True)."
            )

    # Verify persistent primitive bank
    for pid in bank.ids():
        prim = bank.get(pid)
        for name, param in prim.named_parameters():
            if param.requires_grad:
                raise AssertionError(
                    f"Persistent primitive {pid} ('{prim}') parameter '{name}' is not frozen."
                )

    return True


def get_parameter_breakdown(
    core: Any,
    bank: PrimitiveBank,
    workspace: PlasticWorkspace,
    *,
    active_bank_pids: Sequence[int] = (),
    active_workspace_ids: Sequence[int] = (),
) -> ParameterBreakdown:
    """Compute explicit parameter breakdown isolating temporary from persistent weights."""
    core_params = sum(p.numel() for p in core.model.parameters())
    bank_params = bank.total_parameter_count()
    temp_params = workspace.total_parameter_count()

    active_bank = sum(bank.get(pid).num_parameters() for pid in active_bank_pids)
    active_temp = workspace.active_parameter_count(active_workspace_ids)
    active_total = active_bank + active_temp

    trainable = workspace.total_parameter_count(trainable_only=True)

    return ParameterBreakdown(
        core_parameters=core_params,
        bank_parameters=bank_params,
        temporary_parameters=temp_params,
        active_parameters=active_total,
        trainable_parameters=trainable,
    )


def execute_plastic_residual(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    workspace: PlasticWorkspace,
    plastic_id: int,
    examples: Sequence[Example],
    *,
    base_candidate_operations: Sequence[str] | None = None,
    output_length_fn: Any = None,
    target_vocab_size: int = 10,
) -> torch.Tensor:
    """Execute base recipe + temporary plastic residual over frozen core.

    Args:
        core: SharedContentEncoder (task-blind content encoder).
        bank: Frozen PrimitiveBank.
        op_to_id: Mapping from canonical operation name to bank primitive_id.
        workspace: PlasticWorkspace holding temporary plastic modules.
        plastic_id: ID of the active plastic operator in workspace.
        examples: Batch of input examples.
        base_candidate_operations: Sequence of operation names forming the base recipe
            in the bank (F_existing). If None, evaluates full-task plastic mode (F_existing = 0).
        output_length_fn: Optional callable (input_length -> output_length). If omitted,
            inferred from example target length or operation definition.
        target_vocab_size: Size of vocabulary for logit tensor.

    Returns:
        Logits tensor of shape `[batch, max(output_lengths), vocab_size]`.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    batch_size = len(examples)
    device = core.device
    plastic_op = workspace.get(plastic_id)

    # 1. Task-blind content encoding through frozen core (no gradients)
    content_lengths = [len(ex.input_tokens) for ex in examples]
    lmax = max(content_lengths)
    b_ids = collate_content_only_batch(examples, core.tokens, device=device)
    with torch.no_grad():
        h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]

    # Incur output lengths
    if output_length_fn is not None:
        output_lengths = [output_length_fn(c_len) for c_len in content_lengths]
    elif hasattr(examples[0], "target_tokens") and examples[0].target_tokens:
        output_lengths = [len(ex.target_tokens) for ex in examples]
    else:
        output_lengths = list(content_lengths)

    max_out_len = max(output_lengths)

    # 2. Base recipe execution F_existing(h) (strictly detached / no_grad)
    if base_candidate_operations:
        with torch.no_grad():
            base_logits = execute_composition_recipe(
                core,
                bank,
                op_to_id,
                examples,
                candidate_operations=base_candidate_operations,
            ).detach()

        # Handle potential length mismatch between base recipe output and target length
        base_len = base_logits.shape[1]
        if base_len != max_out_len:
            # Pad or slice base logits to match max_out_len
            aligned_base = base_logits.new_zeros(batch_size, max_out_len, target_vocab_size)
            common_len = min(base_len, max_out_len)
            aligned_base[:, :common_len, :] = base_logits[:, :common_len, :]
            base_logits = aligned_base
    else:
        base_logits = h_content.new_zeros(batch_size, max_out_len, target_vocab_size)

    # 3. Plastic residual execution R_plastic(h) (trainable)
    if isinstance(
        plastic_op,
        (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
    ):
        plastic_logits = plastic_op(h_content, content_lengths, output_lengths)
    else:
        plastic_logits = plastic_op(h_content)

    return base_logits + plastic_logits
