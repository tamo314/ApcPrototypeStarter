"""Typed continuous-state contracts for CNP v1.

The public forward-facing types deliberately exclude labels, generator state,
split names, condition IDs, and all other evaluation-only information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CNP_SCHEMA_VERSION = "cnp_v1"
FEATURE_DIM = 8
FamilyName = Literal["CONDITIONAL_SELECT", "COUNT", "SUM_FIRST"]


def _require_tensor(name: str, value: torch.Tensor, *, dtype: torch.dtype, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")


@dataclass(frozen=True)
class SetState:
    """A padded batch of continuous items and its execution-time validity mask."""

    values: torch.Tensor
    valid: torch.Tensor
    item_ids: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("values", self.values, dtype=torch.float32, ndim=3)
        _require_tensor("valid", self.valid, dtype=torch.bool, ndim=2)
        _require_tensor("item_ids", self.item_ids, dtype=torch.int64, ndim=2)
        batch, width, features = self.values.shape
        if features != FEATURE_DIM:
            raise ValueError(f"values must have feature dimension {FEATURE_DIM}, got {features}")
        if self.valid.shape != (batch, width) or self.item_ids.shape != (batch, width):
            raise ValueError("valid and item_ids must match values' batch and item dimensions")
        if self.valid.device != self.values.device or self.item_ids.device != self.values.device:
            raise ValueError("SetState tensors must be on the same device")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    def with_valid(self, valid: torch.Tensor) -> SetState:
        """Return the same values and IDs with a replacement validity mask."""

        return SetState(values=self.values, valid=valid, item_ids=self.item_ids)


@dataclass(frozen=True)
class SelectArguments:
    """Model-visible arguments for one conditional selection call."""

    query: torch.Tensor
    threshold: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("query", self.query, dtype=torch.float32, ndim=2)
        _require_tensor("threshold", self.threshold, dtype=torch.float32, ndim=1)
        if self.query.shape[1] != FEATURE_DIM:
            raise ValueError(f"query must have feature dimension {FEATURE_DIM}")
        if self.query.shape[0] != self.threshold.shape[0]:
            raise ValueError("query and threshold must have the same batch size")
        if self.query.device != self.threshold.device:
            raise ValueError("query and threshold must be on the same device")

    def validate_batch_size(self, batch_size: int) -> None:
        if self.query.shape[0] != batch_size:
            raise ValueError(
                f"arguments batch size {self.query.shape[0]} does not match state {batch_size}"
            )


@dataclass(frozen=True)
class CNPPrimitiveCall:
    """One execution call: a CNP family plus its typed arguments."""

    family: FamilyName
    arguments: SelectArguments | None = None
    schema_version: str = CNP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CNP_SCHEMA_VERSION:
            raise ValueError(f"Unsupported CNP schema {self.schema_version!r}")
        expects_args = self.family == "CONDITIONAL_SELECT"
        if expects_args != (self.arguments is not None):
            raise ValueError(
                f"{self.family} {'requires' if expects_args else 'does not accept'} SelectArguments"
            )


@dataclass(frozen=True)
class CNPRecipe:
    """A statically declared CNP call order without weights or training data."""

    name: str
    steps: tuple[CNPPrimitiveCall, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Recipe name must be non-empty")
        if not self.steps:
            raise ValueError("Recipe must contain at least one call")


@dataclass(frozen=True)
class SelectionResult:
    """Logits, thresholded mask, and the state forwarded to a following call."""

    logits: torch.Tensor
    selected: torch.Tensor
    state: SetState

    def __post_init__(self) -> None:
        _require_tensor("logits", self.logits, dtype=torch.float32, ndim=2)
        _require_tensor("selected", self.selected, dtype=torch.bool, ndim=2)
        expected = (self.state.batch_size, self.state.width)
        if self.logits.shape != expected or self.selected.shape != expected:
            raise ValueError("logits and selected must match the result state")
        if (
            self.logits.device != self.state.values.device
            or self.selected.device != self.state.values.device
        ):
            raise ValueError("SelectionResult tensors must be on the same device")
