"""Integration with the real repository classes, never mock or silently emulate APC."""
import importlib.util

import numpy as np
import pytest

from apc.embodied.control import EmbodiedPrimitiveCall, MotionBank
from apc.embodied.learning import expert_policy
from apc.embodied.world import EmbodiedEnv, WorldConfig, make_task

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("apc.primitives") is None,
    reason="Full APC source tree is required for native bank integration tests",
)


def test_real_primitive_base_bank_freeze_and_sparse_calls():
    from apc.primitives.bank import PrimitiveBank
    from apc.primitives.primitive import PrimitiveBase, PrimitiveStatus
    config = WorldConfig()
    bank = MotionBank("apc")
    bank.add(expert_policy(config, name="selected"))
    bank.add(expert_policy(config, name="unused"))
    native = bank._native
    assert isinstance(native.bank, PrimitiveBank)
    selected = native.bank.get(native.operation_ids["selected"])
    unused = native.bank.get(native.operation_ids["unused"])
    assert isinstance(selected, PrimitiveBase) and selected.status == PrimitiveStatus.STABLE
    assert selected.is_frozen() and selected.num_parameters(trainable_only=True) == 0
    obs, _ = EmbodiedEnv().reset(make_task(1000))
    bank.act(obs, EmbodiedPrimitiveCall("selected", {"target": (2, 1, 4)}))
    assert selected.forward_call_count == 1 and selected.usage_count == 1
    assert unused.forward_call_count == 0 and unused.usage_count == 0
    assert bank.resident_parameters == 42


def test_native_reference_action_parity_and_registry_unchanged():
    from apc.environments.operations import registered_operation_names
    before = registered_operation_names()
    a, b = MotionBank("apc"), MotionBank("numpy")
    policy = expert_policy(WorldConfig(), name="motion")
    a.add(policy)
    b.add(policy)
    obs, _ = EmbodiedEnv().reset(make_task(1000))
    call = EmbodiedPrimitiveCall("motion", {"target": (2, 1, 4)})
    assert np.allclose(a.act(obs, call), b.act(obs, call), atol=1e-12, rtol=0)
    assert registered_operation_names() == before
