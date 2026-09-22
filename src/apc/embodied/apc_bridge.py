"""Native APC integration. Imported only when the 'apc' backend is requested.

The token operation registry and its integer IDs are deliberately untouched.
Embodied bundles use their own strict serializer; do NOT call the legacy bank's
save/load for these modules, whose type is absent from its token-only registry.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from apc.embodied.control import EmbodiedPrimitiveCall, LinearMotionPrimitive, policy_features
from apc.embodied.world import Array, Observation
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import PrimitiveBase, PrimitiveStatus


class EmbodiedAPCPrimitive(PrimitiveBase):
    """Continuous policy using APC's actual lifecycle and sparse instrumentation."""
    ARCHITECTURE_SIGNATURE = "embodied_body_frame_linear_v1"

    def __init__(self, primitive_id: int, policy: LinearMotionPrimitive) -> None:
        super().__init__(primitive_id, status=PrimitiveStatus.STABLE,
                         metadata={"domain": "embodied_3d", "operation": policy.name})
        self.weight = nn.Parameter(torch.tensor(policy.weights(), dtype=torch.float64))
        self.freeze()
        self.eval()

    def forward(self, observation: Observation, call: EmbodiedPrimitiveCall) -> torch.Tensor:
        if not self.enabled or self.status != PrimitiveStatus.STABLE:
            raise RuntimeError("Only enabled stable primitives may execute")
        if call.operation != self.metadata["operation"]:
            raise ValueError("Operation does not match native primitive")
        self.forward_call_count += 1
        self.record_usage()
        # Goal binding stays inside the selected primitive, not in a shared core.
        features = torch.as_tensor(policy_features(observation, call.arguments),
                                   dtype=self.weight.dtype, device=self.weight.device)
        return self.weight @ features


class NativeMotionBank:
    def __init__(self) -> None:
        self.bank = PrimitiveBank()
        self.operation_ids: dict[str, int] = {}

    def add(self, policy: LinearMotionPrimitive) -> None:
        if policy.name in self.operation_ids:
            raise ValueError("Duplicate operation")
        primitive_id = max(self.bank.ids(), default=-1)+1
        primitive = EmbodiedAPCPrimitive(primitive_id, policy)
        self.bank.add_primitive(primitive)
        self.operation_ids[policy.name] = primitive_id

    def act(self, observation: Observation, call: EmbodiedPrimitiveCall) -> Array:
        selected = self.bank.get(self.operation_ids[call.operation])
        with torch.inference_mode():
            force = selected(observation, call).detach().cpu().numpy()
        # No task-conditioned action decoding or dense bank evaluation.
        return np.concatenate((force, np.zeros(3)))

    def policy(self, name: str) -> LinearMotionPrimitive:
        selected = self.bank.get(self.operation_ids[name])
        assert isinstance(selected, EmbodiedAPCPrimitive)
        return LinearMotionPrimitive(name, selected.weight.detach().cpu().numpy())

    @property
    def resident_parameters(self) -> int:
        return sum(self.bank.get(pid).num_parameters() for pid in self.bank.ids())
