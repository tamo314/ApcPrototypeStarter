"""Learned conditional-select primitive and its local residual adapter."""

from __future__ import annotations

import torch
from torch import nn

from apc.cnp.contracts import SelectArguments, SelectionResult, SetState
from apc.cnp.data import phi
from apc.primitives.primitive import PrimitiveBase, PrimitiveStatus

ARCHITECTURE_SIGNATURE = "cnp_conditional_select_mlp_v1"
HIDDEN_DIM = 64
ADAPTER_RANK = 8


class ResidualAdapter(nn.Module):
    """A zero-initialized rank-eight residual adapter for local adaptation."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, rank: int = ADAPTER_RANK) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(torch.nn.functional.gelu(self.down(hidden)))


class ConditionalSelectPrimitive(PrimitiveBase):
    """A sparse, argument-conditioned elementwise selection primitive.

    Its only forward inputs are the continuous content state and typed query and
    threshold arguments.  It cannot receive labels, condition IDs, split names,
    or a private reference-world matrix.
    """

    ARCHITECTURE_SIGNATURE = ARCHITECTURE_SIGNATURE

    def __init__(
        self,
        primitive_id: int,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, object] | None = None,
        adapter: ResidualAdapter | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.input_proj = nn.Linear(65, HIDDEN_DIM)
        self.hidden_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.readout = nn.Linear(HIDDEN_DIM, 1)
        self.adapter = adapter

    @property
    def architecture_signature(self) -> str:
        return self.ARCHITECTURE_SIGNATURE

    @property
    def base_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("adapter.")
        )

    @property
    def adapter_parameter_count(self) -> int:
        return 0 if self.adapter is None else self.adapter.num_parameters

    def attach_adapter(self, adapter: ResidualAdapter | None = None) -> ResidualAdapter:
        """Attach one local adapter; a second adapter is rejected by design."""

        if self.adapter is not None:
            raise RuntimeError("CNP v1 permits only one adapter per stable primitive")
        self.adapter = adapter or ResidualAdapter()
        return self.adapter

    def freeze_base(self) -> None:
        """Freeze all stable weights while leaving an attached adapter trainable."""

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("adapter."))

    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        arguments.validate_batch_size(state.batch_size)
        self.forward_call_count += 1
        if not self.enabled:
            logits = torch.full(
                (state.batch_size, state.width),
                float("-inf"),
                dtype=state.values.dtype,
                device=state.values.device,
            )
            selected = torch.zeros_like(state.valid)
            return SelectionResult(
                logits=logits, selected=selected, state=state.with_valid(selected)
            )

        content = phi(state.values)
        query = phi(arguments.query).unsqueeze(1).expand(-1, state.width, -1)
        threshold = arguments.threshold.reshape(-1, 1, 1).expand(-1, state.width, 1)
        features = torch.cat(
            (content, query, torch.abs(content - query), content * query, threshold), dim=-1
        )
        hidden = torch.nn.functional.gelu(self.input_proj(features))
        hidden = torch.nn.functional.gelu(self.hidden_proj(hidden))
        if self.adapter is not None:
            hidden = self.adapter(hidden)
        logits = self.readout(hidden).squeeze(-1).to(torch.float32)
        selected = state.valid & (logits >= 0.0)
        return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))
