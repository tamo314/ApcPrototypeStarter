"""Tests for B-C005REC-004Z: I03 Key/Value Content-Prep Functional-Role Isolation Pilot."""

from typing import cast

import torch

from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_kv_role_split_pilot as pilot
from apc.evaluation import mirror_position_bias_repair as mpbr


def test_role_split_primitive_parameter_structure() -> None:
    core, _, _ = rec004t._load_runtime_core(seed=10)
    config = pilot.CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=core.model.config.d_model,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=64,
        max_sequence_length=32,
        arg_dim=16,
        bias_hidden_dim=32,
        length_ref=32,
    )
    model = pilot.RoleSplitCrossPositionLengthBiasPrimitive(primitive_id=999, config=config)
    named_params = dict(model.named_parameters())

    # Key branch exists
    assert "key_content_in_proj.weight" in named_params
    assert "key_content_in_proj.bias" in named_params
    assert "key_content_position_embedding.weight" in named_params

    # Value branch exists under legacy names
    assert "content_in_proj.weight" in named_params
    assert "content_in_proj.bias" in named_params
    assert "content_position_embedding.weight" in named_params

    # Check key parameter dimensions
    assert named_params["key_content_in_proj.weight"].shape == (32, core.model.config.d_model)
    assert named_params["key_content_position_embedding.weight"].shape == (32, 32)

    # Added resident parameters = 192*32 + 32 + 32*32 = 6144 + 32 + 1024 = 7200
    added_params = (
        named_params["key_content_in_proj.weight"].numel()
        + named_params["key_content_in_proj.bias"].numel()
        + named_params["key_content_position_embedding.weight"].numel()
    )
    assert added_params == 7200


def test_role_split_clone_and_migration() -> None:
    core, _, _ = rec004t._load_runtime_core(seed=10)
    start_ts_path = rec004t._training_state_path("I03", 7500)
    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")

    model, optimizer, migration_audit = pilot.create_role_split_primitive_and_optimizer(
        core, start_ts, device
    )

    assert migration_audit["status"] == "PASS"
    assert len(migration_audit["entries"]) == 3

    # Verify exact clone of weights
    assert torch.equal(
        model.key_content_in_proj.weight.data,
        model.content_in_proj.weight.data,
    )
    assert torch.equal(
        model.key_content_in_proj.bias.data,
        model.content_in_proj.bias.data,
    )
    assert torch.equal(
        model.key_content_position_embedding.weight.data,
        model.content_position_embedding.weight.data,
    )

    # Verify exact clone of optimizer states
    key_w_opt = optimizer.state[model.key_content_in_proj.weight]
    val_w_opt = optimizer.state[model.content_in_proj.weight]
    assert torch.equal(key_w_opt["exp_avg"], val_w_opt["exp_avg"])
    assert torch.equal(key_w_opt["exp_avg_sq"], val_w_opt["exp_avg_sq"])
    assert key_w_opt["step"] == val_w_opt["step"]


def test_role_split_exact_parity_at_7500() -> None:
    core, _, _ = rec004t._load_runtime_core(seed=10)
    start_ts_path = rec004t._training_state_path("I03", 7500)
    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)
    device = core.device

    ref_7500 = cast(
        pilot.CrossPositionLengthBiasPrimitive,
        mpbr._new_arm_primitive(core, "P_LENGTH_POSITION_BIAS"),
    )
    ref_7500.load_state_dict(start_ts["primitive_state_dict"])
    ref_7500.eval()

    role_model, _, _ = pilot.create_role_split_primitive_and_optimizer(core, start_ts, device)
    role_model.eval()

    # Generate a small parity test set
    exs, _ = pilot.build_parity_fixture_v1(10, set(), n=32)

    report = pilot.verify_role_split_initial_parity(core, role_model, ref_7500, exs, device)
    assert report["status"] == "PASS"
    assert report["discrete_prediction_mismatches"] == 0
    assert report["attn_probs_max_abs_diff"] <= 1e-5
    assert report["final_logits_max_abs_diff"] <= 1e-4


def test_role_split_gradient_isolation() -> None:
    core, _, _ = rec004t._load_runtime_core(seed=10)
    start_ts_path = rec004t._training_state_path("I03", 7500)
    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)
    device = core.device

    role_model, _, _ = pilot.create_role_split_primitive_and_optimizer(core, start_ts, device)

    exs, _ = pilot.build_parity_fixture_v1(10, set(), n=16)

    report = pilot.verify_role_split_gradient_path_isolation(core, role_model, exs, device)
    assert report["status"] == "PASS"
    assert report["j0_key_active"] is True
    assert report["o1_key_isolated"] is True
    assert report["o1_value_path_active"] is True
