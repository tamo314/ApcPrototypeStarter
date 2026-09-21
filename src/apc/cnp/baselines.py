"""CNP v1 comparators with the same observed inputs as the learned primitive."""

from __future__ import annotations

import torch
from torch import nn

from apc.cnp.contracts import SelectArguments, SelectionResult, SetState
from apc.cnp.data import phi


class RawDistanceBaseline(nn.Module):
    """Two-parameter raw-space distance comparator; it never receives the private metric."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(()))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        arguments.validate_batch_size(state.batch_size)
        distance = (state.values - arguments.query.unsqueeze(1)).square().mean(dim=-1)
        logits = arguments.threshold.unsqueeze(1) - (
            torch.nn.functional.softplus(self.raw_scale) * distance + self.offset
        )
        logits = logits.to(torch.float32)
        selected = state.valid & (logits >= 0.0)
        return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))


class LearnedMetricBaseline(nn.Module):
    """A PSD metric over public phi features and a learned positive logit temperature."""

    def __init__(self) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.eye(16, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        arguments.validate_batch_size(state.batch_size)
        delta = phi(state.values) - phi(arguments.query).unsqueeze(1)
        projected = delta @ self.factor.T
        distance = projected.square().sum(dim=-1) / 16.0
        logits = torch.nn.functional.softplus(self.log_temperature) * (
            arguments.threshold.unsqueeze(1) - distance
        )
        logits = logits.to(torch.float32)
        selected = state.valid & (logits >= 0.0)
        return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))


class UnconditionedBaseline(nn.Module):
    """A shortcut detector with the same architecture but query and threshold zeroed."""

    def __init__(self, conditional: nn.Module) -> None:
        super().__init__()
        self.conditional = conditional

    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        zero_arguments = SelectArguments(
            query=torch.zeros_like(arguments.query),
            threshold=torch.zeros_like(arguments.threshold),
        )
        return self.conditional(state, zero_arguments)
