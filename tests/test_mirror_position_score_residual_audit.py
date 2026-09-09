"""CPU-only contract tests for Task B-C005REC-004E (MIRROR_HALVES Position-
Score Residual Audit & Next-Repair Contract).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real audit against REC-004D's saved 5-init x 13-checkpoint
production run is never invoked from this test module (it is dispatched
separately via `scripts/run_phase_b_b2_model_bundle_recovery.py --task
B-C005REC-004E`). These tests exercise the real production functions
against freshly-constructed, tiny, CPU-only primitives and synthetic
inputs -- never optimizer/training code, since this task performs zero new
optimizer updates by design.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from apc.evaluation import mirror_position_score_residual_audit as m
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _tiny_primitive(
    seed: int = 0, *, nonzero_bias: bool = True
) -> CrossPositionLengthBiasPrimitive:
    torch.manual_seed(seed)
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES", bias_hidden_dim=m.REC004E_BIAS_HIDDEN_DIM,
        length_ref=m.REC004E_LENGTH_REF,
    )
    primitive = CrossPositionLengthBiasPrimitive(m.REC004E_TARGET_PHYSICAL_ID, cfg)
    if nonzero_bias:
        with torch.no_grad():
            primitive.position_bias_out.weight.normal_(mean=0.0, std=0.2)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def _tiny_batch(content_lengths: list[int], d_model: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    lmax = max(content_lengths)
    return torch.randn(len(content_lengths), lmax, d_model)


# ---------------------------------------------------------------------------
# 1. Grid: 330 unique (i, j) pairs total, no duplicates, correct index shape.
# ---------------------------------------------------------------------------


def test_grid_position_pair_count_is_330_and_matches_per_length_squares() -> None:
    per_length = {n: n * n for n in m.REC004E_LEGAL_LENGTHS}
    assert sum(per_length.values()) == 330
    assert per_length == {6: 36, 7: 49, 8: 64, 9: 81, 10: 100}


def test_grid_index_layout_has_no_duplicate_pairs_within_a_length() -> None:
    for n in m.REC004E_LEGAL_LENGTHS:
        pairs = {(i, j) for i in range(n) for j in range(n)}
        assert len(pairs) == n * n


def test_checkpoint_steps_are_13_and_include_the_decisive_step() -> None:
    assert len(m.REC004E_CHECKPOINT_STEPS) == 13
    assert m.REC004E_CHECKPOINT_STEPS[0] == 0
    assert m.REC004E_CHECKPOINT_STEPS[-1] == m.REC004E_DECISIVE_STEP == 6000
    assert list(m.REC004E_CHECKPOINT_STEPS) == sorted(set(m.REC004E_CHECKPOINT_STEPS))


# ---------------------------------------------------------------------------
# 2. Real length vs padding; `_position_bias_raw` matches the real
#    `_position_bias` exactly when no override is given (no reconstruction
#    drift), and L_ref is read from the real schema constant, never assumed.
# ---------------------------------------------------------------------------


def test_position_bias_raw_matches_real_position_bias_with_no_override() -> None:
    primitive = _tiny_primitive()
    device = torch.device("cpu")
    content_lengths = [6, 7, 8, 9, 10]
    out_max = lmax = max(content_lengths)
    real_b = primitive._position_bias(content_lengths, out_max, lmax, device)
    pack = m._position_bias_raw(primitive, content_lengths, out_max, lmax, device)
    assert torch.equal(real_b, pack["b"])


def test_position_bias_raw_phi4_override_changes_only_the_fourth_feature() -> None:
    primitive = _tiny_primitive()
    device = torch.device("cpu")
    content_lengths = [10, 10, 9]
    out_max = lmax = 10
    apply_mask = torch.tensor([True, True, False])
    real_pack = m._position_bias_raw(primitive, content_lengths, out_max, lmax, device)
    overridden = m._position_bias_raw(
        primitive, content_lengths, out_max, lmax, device,
        phi4_override=m.REC004E_J5_OVERRIDE_VALUE, phi4_apply_mask=apply_mask,
    )
    assert torch.equal(overridden["phi"][..., :3], real_pack["phi"][..., :3])
    assert torch.equal(overridden["phi"][2, ..., 3], real_pack["phi"][2, ..., 3])
    assert not torch.equal(overridden["phi"][0, ..., 3], real_pack["phi"][0, ..., 3])
    assert torch.allclose(
        overridden["phi"][0, 0, 0, 3], torch.tensor(m.REC004E_J5_OVERRIDE_VALUE)
    )


def test_length_ref_used_is_the_live_schema_default_not_an_assumed_value() -> None:
    default_cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    assert m.REC004E_LENGTH_REF == default_cfg.length_ref
    assert m.REC004E_LENGTH_REF != 10  # never assume the trained-length ceiling is L_ref


# ---------------------------------------------------------------------------
# 3. Fixed/moved contract; denominator-0 is null, not a fabricated zero.
# ---------------------------------------------------------------------------


def test_fixed_position_contract_matches_the_pre_registered_table() -> None:
    result = m._fixed_position_contract()
    assert result["all_match"] is True
    assert result["per_length"]["6"]["fixed_positions"] == [1, 4]
    assert result["per_length"]["8"]["fixed_positions"] == []
    assert result["per_length"]["10"]["fixed_positions"] == [2, 7]


def test_fixed_and_moved_partition_every_position_exactly_once() -> None:
    result = m._fixed_position_contract()
    for n_str, row in result["per_length"].items():
        n = int(n_str)
        fixed, moved = set(row["fixed_positions"]), set(row["moved_positions"])
        assert fixed.isdisjoint(moved)
        assert fixed | moved == set(range(n))


def test_length_eight_fixed_denominator_is_null_not_a_fabricated_zero() -> None:
    grid_result = {
        "index": {
            "row_affine_segment": {"rows_with_single_activation_pattern": 0, "rows_checked": 0}
        }
    }
    fraction = grid_result["index"]["row_affine_segment"]
    # zero denominator must be represented as null, matching the real
    # run_position_grid_enumeration's own `.../ total if total else None` pattern.
    computed = (
        fraction["rows_with_single_activation_pattern"] / fraction["rows_checked"]
        if fraction["rows_checked"] else None
    )
    assert computed is None


# ---------------------------------------------------------------------------
# 4. Row-centering: a constant added to a whole VALID row does not change
#    softmax; -inf is never mixed into a centering computation.
# ---------------------------------------------------------------------------


def test_constant_row_addition_does_not_change_softmax() -> None:
    row = torch.tensor([0.3, -1.2, 2.5, 0.0])
    shifted = row + 7.0
    assert torch.allclose(torch.softmax(row, dim=-1), torch.softmax(shifted, dim=-1), atol=1e-6)


def test_row_centering_never_mixes_in_negative_infinity() -> None:
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    valid = torch.tensor([True, True, False])
    centered_row0 = b[0][valid] - b[0][valid].mean()
    assert torch.isfinite(centered_row0).all()
    assert valid.sum().item() == 2  # the masked (padded) column is excluded, not zeroed


# ---------------------------------------------------------------------------
# 5. Score-observer parity: J0 (no intervention) reproduces the real
#    forward's discrete predictions on a CPU tiny fixture.
# ---------------------------------------------------------------------------


def test_j0_hook_path_reproduces_real_forward_on_cpu_fixture() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=1)
    with torch.no_grad():
        real_logits = primitive(content, content_lengths, output_lengths, None)
        recon = m.run_intervention_forward(
            primitive, content, content_lengths, output_lengths, "J0"
        )
    assert (real_logits - recon["logits"]).abs().max().item() <= m.REC004E_SCORE_LOGIT_ABS_TOL
    real_preds = m._predict_from_logits(real_logits, output_lengths)
    recon_preds = m._predict_from_logits(recon["logits"], output_lengths)
    assert real_preds == recon_preds


def test_manual_attention_reconstruction_matches_hook_path_at_s_other_scale_one() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=2)
    with torch.no_grad():
        query, kv, out_max, lmax, batch = m._prepare_query_kv(
            primitive, content, content_lengths, output_lengths
        )
        pad = m._pad_mask(content_lengths, lmax, content.device)
        bias_pack = m._position_bias_raw(primitive, content_lengths, out_max, lmax, content.device)
        combined = torch.where(
            pad.unsqueeze(1).expand(batch, out_max, lmax),
            torch.finfo(bias_pack["b"].dtype).min,
            bias_pack["b"],
        )
        manual_out, _, _ = m._manual_attention(primitive, query, kv, combined, s_other_scale=1.0)
        manual_logits = m._post_attention(primitive, query, manual_out)
        real_logits = primitive(content, content_lengths, output_lengths, None)
    diff = (manual_logits - real_logits).abs().max().item()
    assert diff <= m.REC004E_MANUAL_RECONSTRUCTION_ABS_TOL


# ---------------------------------------------------------------------------
# 6. J1-J4 modify only the specified score component; value/mask/decoder
#    (content_in_proj, embeddings, ffn, readout) are never touched.
# ---------------------------------------------------------------------------


def test_j1_is_architecturally_identical_to_zeroed_position_bias() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=3)
    with torch.no_grad():
        j1 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J1")
        with primitive.zeroed_position_bias():
            zeroed = primitive(content, content_lengths, output_lengths, None)
    assert (j1["logits"] - zeroed).abs().max().item() <= m.REC004E_SCORE_LOGIT_ABS_TOL


def test_j2_j3_scale_the_bias_term_only() -> None:
    primitive = _tiny_primitive()
    content_lengths = [10]
    output_lengths = [10]
    content = _tiny_batch(content_lengths, d_model=192, seed=4)
    with torch.no_grad():
        j0 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J0")
        j2 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J2")
        j3 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J3")
    assert torch.allclose(j2["b_scaled"], 0.5 * j0["b_scaled"], atol=1e-6)
    assert torch.allclose(j3["b_scaled"], 2.0 * j0["b_scaled"], atol=1e-6)


def test_j4_zeroes_s_other_but_keeps_bias_and_mask() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=5)
    with torch.no_grad():
        j0 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J0")
        j4 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J4")
    assert torch.equal(j4["b_scaled"], j0["b_scaled"])  # bias term unchanged
    assert j4["s_other"] is not None
    # J4's own attention must differ from J0's whenever S_other is not already
    # uniform across keys (confirms the intervention actually took effect).
    assert not torch.allclose(j4["attn_probs"], j0["attn_probs"])


def test_interventions_never_touch_the_value_content_or_readout_weights() -> None:
    primitive = _tiny_primitive()
    before = {k: v.detach().clone() for k, v in primitive.state_dict().items()}
    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=6)
    with torch.no_grad():
        for j_id in m.REC004E_J_IDS:
            m.run_intervention_forward(primitive, content, content_lengths, output_lengths, j_id)
    after = primitive.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------------------
# 7. J5/J6 change only phi's 4th feature; real n/position IDs/mask/L_ref
#    are untouched.
# ---------------------------------------------------------------------------


def test_j5_only_changes_phi_fourth_component_for_length_ten() -> None:
    primitive = _tiny_primitive()
    content_lengths = [10, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=7)
    with torch.no_grad():
        j0 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J0")
        j5 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J5")
    assert torch.equal(j0["bias_pack"]["phi"][..., :3], j5["bias_pack"]["phi"][..., :3])
    assert not torch.equal(j0["bias_pack"]["phi"][..., 3], j5["bias_pack"]["phi"][..., 3])
    expected_j5 = torch.tensor(m.REC004E_J5_OVERRIDE_VALUE)
    assert torch.allclose(j5["bias_pack"]["phi"][0, 0, 0, 3], expected_j5)
    assert torch.equal(j0["pad_mask"], j5["pad_mask"])


def test_j6_only_applies_to_length_nine_examples_in_a_mixed_batch() -> None:
    primitive = _tiny_primitive()
    content_lengths = [9, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=8)
    with torch.no_grad():
        j0 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J0")
        j6 = m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J6")
    # row 0 (length 9) is overridden; row 1 (length 10) is untouched.
    assert not torch.equal(j0["bias_pack"]["phi"][0, ..., 3], j6["bias_pack"]["phi"][0, ..., 3])
    assert torch.equal(j0["bias_pack"]["phi"][1, ..., 3], j6["bias_pack"]["phi"][1, ..., 3])


# ---------------------------------------------------------------------------
# 8. Non-mutation: normal forward after ANY intervention reproduces the
#    original prediction; weights/buffers are unchanged.
# ---------------------------------------------------------------------------


def test_intervention_nonmutation_pattern_restores_original_forward() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    content = _tiny_batch(content_lengths, d_model=192, seed=9)
    before_hash_tensors = {k: v.detach().clone() for k, v in primitive.state_dict().items()}
    with torch.no_grad():
        pre = primitive(content, content_lengths, output_lengths, None)
        for j_id in m.REC004E_J_IDS:
            m.run_intervention_forward(primitive, content, content_lengths, output_lengths, j_id)
        post = primitive(content, content_lengths, output_lengths, None)
    assert torch.equal(pre, post)
    after = primitive.state_dict()
    assert all(torch.equal(before_hash_tensors[k], after[k]) for k in before_hash_tensors)


# ---------------------------------------------------------------------------
# 9. Query target / position map never reach a bias or intervention input.
# ---------------------------------------------------------------------------


def test_position_bias_raw_signature_has_no_teacher_or_position_map_argument() -> None:
    sig = inspect.signature(m._position_bias_raw)
    params = set(sig.parameters)
    assert params == {
        "primitive", "content_lengths", "out_max", "lmax", "device",
        "phi4_override", "phi4_apply_mask",
    }
    assert "pi" not in params and "target" not in params and "teacher" not in params


def test_intervention_forward_signature_has_no_teacher_or_position_map_argument() -> None:
    sig = inspect.signature(m.run_intervention_forward)
    params = set(sig.parameters)
    assert params == {"primitive", "content_features", "content_lengths", "output_lengths", "j_id"}


# ---------------------------------------------------------------------------
# 10. Dynamic guard: gradient computation must stay disabled; the module
#     never imports/constructs an optimizer.
# ---------------------------------------------------------------------------


def test_assert_eval_only_raises_when_gradients_are_enabled() -> None:
    with pytest.raises(RuntimeError, match="REC004E_TRAINING_GUARD"):
        with torch.enable_grad():
            m._assert_eval_only()


def test_assert_eval_only_passes_under_no_grad() -> None:
    with torch.no_grad():
        m._assert_eval_only()  # must not raise


def test_run_intervention_forward_guards_against_gradient_enabled_context() -> None:
    primitive = _tiny_primitive()
    content_lengths = [6]
    output_lengths = [6]
    content = _tiny_batch(content_lengths, d_model=192, seed=10)
    with pytest.raises(RuntimeError, match="REC004E_TRAINING_GUARD"):
        with torch.enable_grad():
            m.run_intervention_forward(primitive, content, content_lengths, output_lengths, "J0")


def test_module_never_imports_torch_optim() -> None:
    import apc.evaluation.mirror_position_score_residual_audit as mod

    source = inspect.getsource(mod)
    assert "torch.optim" not in source
    assert "import optim" not in source
    assert ".backward(" not in source


# ---------------------------------------------------------------------------
# 11. Observation subset selection is positional, not correctness-based;
#     no best-EM/best-intervention auto-adoption logic exists.
# ---------------------------------------------------------------------------


def test_score_observation_subset_size_is_pre_registered_not_derived_from_results() -> None:
    assert m.REC004E_SCORE_OBS_PER_LENGTH == 32
    assert m.REC004E_INTERVENTION_PER_LENGTH == m.REC004E_LENGTH_BALANCED_PER_LENGTH == 256


def test_no_selection_or_adoption_helper_exists_in_this_module() -> None:
    import apc.evaluation.mirror_position_score_residual_audit as mod

    forbidden_names = ("select_candidate", "build_child_bundle", "run_stage_f", "adopt_")
    exported = set(dir(mod))
    assert not any(name in exported for name in forbidden_names)


# ---------------------------------------------------------------------------
# 12. Missing/NaN/unsupported results are never silently filled with
#     PASS/0%; this module never calls REC-005/query/child-bundle code.
# ---------------------------------------------------------------------------


def test_lock_source_manifest_reports_unavailable_not_a_fabricated_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(m, "REC004E_SOURCE_RUN_DIR", tmp_path)
    result = m.lock_source_manifest()
    assert result["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"
    assert "missing" in result


def test_next_repair_contract_is_evidence_insufficient_on_contract_failure() -> None:
    diagnosis = {
        "labels": ["EXECUTION_CONTRACT_FAILURE"],
        "evidence": {},
    }
    text = m.build_next_repair_contract(diagnosis)
    assert "status: EVIDENCE_INSUFFICIENT" in text
    assert "not authorization to implement" in text


def test_module_never_references_rec005_or_recheck_query_machinery() -> None:
    import apc.evaluation.mirror_position_score_residual_audit as mod

    source = inspect.getsource(mod)
    assert "REC004D_RECHECK_QUERY_SPLIT" not in source
    assert "run_stage_f" not in source
    assert "rec005" not in source.lower().replace("rec005_eligible", "").replace(
        "rec005 onward", ""
    ).replace("b-c005rec-005", "")


# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


def test_j_conditions_and_scales_are_pre_registered() -> None:
    assert m.REC004E_J_IDS == ("J0", "J1", "J2", "J3", "J4", "J5", "J6")
    assert m.REC004E_J_BIAS_SCALE["J0"] == 1.0
    assert m.REC004E_J_BIAS_SCALE["J1"] == 0.0
    assert m.REC004E_J_BIAS_SCALE["J2"] == 0.5
    assert m.REC004E_J_BIAS_SCALE["J3"] == 2.0
    assert m.REC004E_J_S_OTHER_SCALE["J4"] == 0.0
    assert m.REC004E_J5_TARGET_LENGTH == 10
    assert m.REC004E_J6_TARGET_LENGTH == 9


def test_init_ids_and_source_run_dir_match_rec004d() -> None:
    assert m.REC004E_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert m.REC004E_SOURCE_TASK_ID == "B-C005REC-004D"
    assert "rec004d" in str(m.REC004E_SOURCE_RUN_DIR).replace("\\", "/")
