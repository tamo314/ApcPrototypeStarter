"""Unit and regression tests for Task B-C005REC-004X:
I03 CVOF Pre-Transition Trust-Region Plasticity Pilot.
"""

from __future__ import annotations

import torch

from apc.evaluation import mirror_cvof_trust_region_pilot as rec004x
from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive


def test_rec004x_dataset_disjointness() -> None:
    """Verifies that fresh evaluation datasets for REC-004X are disjoint from
    all prior protected registries (1..18000, REC-004T..W) and from each other."""
    seed = rec004x.REC004X_SEED
    datasets, manifest = rec004x.prepare_rec004x_datasets(seed)

    fresh_val = datasets[rec004x.REC004X_FRESH_NORMAL_VALIDATION]
    fresh_conf = datasets[rec004x.REC004X_FRESH_LENGTH10_CONFIRMATION]

    assert len(fresh_val) == rec004x.REC004X_FRESH_VALIDATION_EXAMPLES
    assert len(fresh_conf) == rec004x.REC004X_FRESH_LENGTH10_EXAMPLES

    val_digests = rec004x._digest_examples(fresh_val)
    conf_digests = rec004x._digest_examples(fresh_conf)

    # 1. Internal disjointness
    assert len(val_digests) == len(fresh_val)
    assert len(conf_digests) == len(fresh_conf)
    assert len(val_digests.intersection(conf_digests)) == 0

    # 2. Manifest status verified
    assert manifest["fresh_normal_validation"]["disjoint_verified"] is True
    assert manifest["fresh_length10_confirmation"]["disjoint_verified"] is True


def test_rec004x_trust_region_radial_projection_math() -> None:
    """Verifies that radial projection correctly preserves parameters within radius,
    and projects parameters exceeding radius back to exact sphere boundary."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004x.REC004X_SEED)
    model = mpbr._new_arm_primitive(core, rec004x.REC004X_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)

    current_cvof = rec004x.get_cvof_group_tensors(model)
    anchor_groups = {g: [t.detach().clone() for t in t_list] for g, t_list in current_cvof.items()}

    # Radii setup
    radii = {"C": 0.1, "V": 0.05, "O": 0.08, "F": 0.15}

    # Case 1: Perturb within radius -> No projection
    with torch.no_grad():
        for t in current_cvof["C"]:
            t.add_(0.001)

    c_diff = rec004x.compute_group_l2_distance(
        rec004x.get_cvof_group_tensors(model)["C"], anchor_groups["C"]
    )
    assert c_diff < radii["C"]

    dyn1 = rec004x.project_cvof_trust_region(model, anchor_groups, radii)
    assert dyn1["C"]["projection_applied"] is False
    assert abs(dyn1["C"]["current_displacement_norm"] - c_diff) < 1e-6
    assert dyn1["C"]["projection_correction_norm"] == 0.0

    # Case 2: Perturb beyond radius -> Radial projection to exactly radius
    with torch.no_grad():
        for t in current_cvof["V"]:
            t.add_(1.0)  # Huge perturbation

    v_raw_diff = rec004x.compute_group_l2_distance(
        rec004x.get_cvof_group_tensors(model)["V"], anchor_groups["V"]
    )
    assert v_raw_diff > radii["V"]

    dyn2 = rec004x.project_cvof_trust_region(model, anchor_groups, radii)
    assert dyn2["V"]["projection_applied"] is True
    assert abs(dyn2["V"]["current_displacement_norm"] - radii["V"]) < 1e-5
    assert dyn2["V"]["projection_correction_norm"] > 0.0

    # Verify actual tensor distance matches radius
    v_actual_diff = rec004x.compute_group_l2_distance(
        rec004x.get_cvof_group_tensors(model)["V"], anchor_groups["V"]
    )
    assert abs(v_actual_diff - radii["V"]) < 1e-5


def test_rec004x_fused_qkv_row_slicing() -> None:
    """Verifies that in_proj_weight row slicing affects only V rows (64:96),
    leaving Q rows (0:32) and K rows (32:64) completely unconstrained and unprojected."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004x.REC004X_SEED)
    model = mpbr._new_arm_primitive(core, rec004x.REC004X_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)

    embed_dim = model.d_operator  # 32
    current_cvof = rec004x.get_cvof_group_tensors(model)
    anchor_groups = {g: [t.detach().clone() for t in t_list] for g, t_list in current_cvof.items()}

    # Perturb Q, K, and V rows
    with torch.no_grad():
        # Q rows (0:32)
        model.cross_attn.in_proj_weight[0:embed_dim].add_(5.0)
        # K rows (32:64)
        model.cross_attn.in_proj_weight[embed_dim : 2 * embed_dim].add_(7.0)
        # V rows (64:96)
        model.cross_attn.in_proj_weight[2 * embed_dim : 3 * embed_dim].add_(10.0)

    q_perturbed = model.cross_attn.in_proj_weight[0:embed_dim].clone()
    k_perturbed = model.cross_attn.in_proj_weight[embed_dim : 2 * embed_dim].clone()

    radii = {"C": 0.1, "V": 0.05, "O": 0.08, "F": 0.15}
    dyn = rec004x.project_cvof_trust_region(model, anchor_groups, radii)

    # V must be projected
    assert dyn["V"]["projection_applied"] is True

    # Q and K rows must be 100% untouched by projection!
    assert torch.equal(model.cross_attn.in_proj_weight[0:embed_dim], q_perturbed)
    assert torch.equal(model.cross_attn.in_proj_weight[embed_dim : 2 * embed_dim], k_perturbed)


def test_rec004x_initial_parity_at_7500() -> None:
    """Verifies that at step 7500, pilot model and reference forward match within tolerance."""
    core, _, _ = rec004t._load_runtime_core(seed=rec004x.REC004X_SEED)
    start_ts_path = rec004t._training_state_path(
        rec004x.REC004X_DECISIVE_INIT, rec004x.REC004X_START_STEP
    )
    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)

    model = mpbr._new_arm_primitive(core, rec004x.REC004X_BASE_PRIMITIVE)
    ref = mpbr._new_arm_primitive(core, rec004x.REC004X_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    assert isinstance(ref, CrossPositionLengthBiasPrimitive)

    model.load_state_dict(start_ts["primitive_state_dict"], strict=True)
    ref.load_state_dict(start_ts["primitive_state_dict"], strict=True)

    sample_exs, _ = rec004x.build_cvof_trust_region_length10_v1(rec004x.REC004X_SEED, set(), n=8)
    report = rec004x.verify_initial_parity(core, model, ref, sample_exs, core.device)
    assert report["status"] == "PASS"
    assert report["score_logits_max_abs_diff"] < 1e-4
    assert report["discrete_prediction_mismatch_count"] == 0
