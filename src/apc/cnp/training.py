"""Small, explicit training primitives used by later CNP experiment runners."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as functional

from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.primitive import ConditionalSelectPrimitive, ResidualAdapter


def masked_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean BCE over valid elements; reject empty batches rather than hiding them."""

    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("logits, target, and valid must have the same shape")
    if not bool(valid.any()):
        raise ValueError("cannot compute a training loss for an all-invalid batch")
    losses = functional.binary_cross_entropy_with_logits(
        logits, target.to(torch.float32), reduction="none"
    )
    return losses[valid].mean()


def clone_local_candidate(parent: ConditionalSelectPrimitive) -> ConditionalSelectPrimitive:
    """Copy a stable parent, attach one zero-output adapter, and freeze base weights."""

    if parent.adapter is not None:
        raise ValueError("CNP v1 does not stack adapters on an existing candidate")
    candidate = copy.deepcopy(parent)
    candidate.attach_adapter(ResidualAdapter())
    candidate.freeze_base()
    return candidate


def adapter_parameters(candidate: ConditionalSelectPrimitive) -> list[torch.nn.Parameter]:
    """Return exactly the local parameters and fail closed for an invalid candidate."""

    if candidate.adapter is None:
        raise ValueError("candidate has no residual adapter")
    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    if set(parameters) != set(candidate.adapter.parameters()):
        raise RuntimeError("CNP local candidate exposes trainable base weights")
    return parameters


def one_training_step(
    primitive: ConditionalSelectPrimitive,
    state: SetState,
    arguments: SelectArguments,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform exactly one explicit update; experiment loops own all step budgets."""

    optimizer.zero_grad(set_to_none=True)
    result = primitive(state, arguments)
    loss = masked_bce_loss(result.logits, target, state.valid)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(primitive.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())
