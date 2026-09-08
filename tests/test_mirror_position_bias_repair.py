"""CPU-only contract tests for Task B-C005REC-004D (MIRROR_HALVES
Length-Conditioned Position Bias & RG3 Recheck).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (10 real training runs against the
real REC-004 parent bundle + REC-004C's real saved I01-I05 initial states)
is never invoked from this test module. These tests exercise the real
production functions -- including genuine (tiny-ladder) end-to-end CPU
training calls -- against small fixtures.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from apc.evaluation import mirror_position_bias_repair as m
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


def test_target_operation_and_fixed_candidates_are_disjoint() -> None:
    assert m.REC004D_TARGET_OPERATION == "MIRROR_HALVES"
    assert set(m.REC004D_FIXED_CANDIDATE_OPERATIONS) == {
        "CYCLE_FOUR", "ROTATE_TRIPLETS", "SWAP_ENDS",
    }
    changed_ops = {m.REC004D_TARGET_OPERATION, *m.REC004D_FIXED_CANDIDATE_OPERATIONS}
    assert len(changed_ops) == 4
    assert changed_ops.isdisjoint(set(m.REC004D_PROTECTED_OPERATIONS))


def test_protected_operations_are_twelve_including_shift() -> None:
    assert len(m.REC004D_PROTECTED_OPERATIONS) == 12
    assert "SHIFT" in m.REC004D_PROTECTED_OPERATIONS


def test_arms_init_ids_and_budget_are_pre_registered() -> None:
    assert set(m.REC004D_ARMS) == {"U_CURRENT_OPERATOR", "P_LENGTH_POSITION_BIAS"}
    assert m.REC004D_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert m.REC004D_MAX_UPDATES_PER_RUN == 6000 == m.REC004D_DECISIVE_STEP
    assert m.REC004D_TOTAL_MAX_UPDATES == 60000
    assert m.REC004D_T_MAX == 1000


def test_parent_bundle_is_rec004s() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc

    assert m.REC004D_PARENT_BUNDLE_ID == ibc.REC004A_PARENT_BUNDLE_ID
    assert m.REC004D_PARENT_MANIFEST_PATH == ibc.REC004A_PARENT_MANIFEST_PATH


def test_existing_validation_split_reuses_rec004as_and_recheck_is_new_namespace() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc

    assert m.REC004D_EXISTING_VALIDATION_SPLIT == ibc.REC004A_BUDGET_VALIDATION_SPLIT
    assert m.REC004D_RECHECK_QUERY_SPLIT != m.REC004D_EXISTING_VALIDATION_SPLIT
    assert m.REC004D_RECHECK_QUERY_SPLIT == "rec004d_recheck_query"


# ---------------------------------------------------------------------------
# 1. Fixed/moved position audit -- n=6/10 have 2 fixed each, n=8 has none
#    (null/zero denominator), fixed+moved == valid positions, apparent
#    "changed" is a distinct classification from structural "moved".
# ---------------------------------------------------------------------------


def test_fixed_position_audit_matches_the_source_formula_ground_truth() -> None:
    audit = m._audit_fixed_position_claim()
    per_length = audit["per_length"]
    assert per_length["6"]["fixed_positions"] == [1, 4]
    assert per_length["6"]["fixed_count"] == 2
    assert per_length["10"]["fixed_positions"] == [2, 7]
    assert per_length["10"]["fixed_count"] == 2
    assert per_length["8"]["fixed_positions"] == []
    assert per_length["8"]["fixed_count"] == 0
    assert per_length["7"]["fixed_count"] == 1
    assert per_length["9"]["fixed_count"] == 1
    assert audit["lengths_with_fixed_position"] == [6, 7, 9, 10]
    assert audit["classification"] == "REPORT_ONLY_ERRATUM"


def test_fixed_and_moved_partition_every_valid_position_exactly_once() -> None:
    for n in m.REC004D_LEGAL_LENGTHS:
        pi = m.mpid.mirror_halves_position_map(n)
        fixed = {i for i in range(n) if pi[i] == i}
        moved = {i for i in range(n) if pi[i] != i}
        assert fixed | moved == set(range(n))
        assert fixed & moved == set()


def test_apparent_changed_is_distinct_from_structural_moved() -> None:
    # n=6: pi_6 = (2,1,0,5,4,3). Position 1 is structurally FIXED (pi[1]=1)
    # but a sequence with x[1] != y[1] would still be "apparent changed" at
    # a structurally fixed position IF x[1] != x[pi[1]]=x[1] -- impossible
    # for a fixed position since y[i]=x[pi[i]]=x[i]. So a structurally fixed
    # position is always "apparent unchanged"; this test documents that
    # entailment rather than assuming it.
    n = 6
    pi = m.mpid.mirror_halves_position_map(n)
    x = (3, 7, 1, 9, 2, 5)
    y = tuple(x[pi[i]] for i in range(n))
    for i in range(n):
        if pi[i] == i:
            assert x[i] == y[i], "a structurally fixed position must be apparent-unchanged"


# ---------------------------------------------------------------------------
# 2. Metric reconciliation -- REC-004C's own JSON aggregates are recomputed
#    from raw by-position cells and must match exactly (no correction to the
#    underlying predictions/EM, only to the prose).
# ---------------------------------------------------------------------------


def test_metric_reconciliation_confirms_rec004c_json_without_changing_it() -> None:
    if not m.REC004C_POSITION_ERROR_SUMMARY_PATH.is_file():
        pytest.skip("REC-004C run artifacts not present in this checkout")
    audit = m._audit_fixed_position_claim()
    result = m._reconcile_historical_position_metrics(audit)
    assert result["status"] == "RECONCILED"
    assert result["all_match"] is True


# ---------------------------------------------------------------------------
# 3. Position feature uses only i, j, n, length_ref -- no teacher/pi/half id.
# ---------------------------------------------------------------------------


def test_position_bias_hidden_input_dimension_is_exactly_four() -> None:
    cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    primitive = CrossPositionLengthBiasPrimitive(12, cfg)
    assert primitive.position_bias_hidden.in_features == 4


def test_position_bias_forward_signature_has_no_teacher_or_half_id_argument() -> None:
    import inspect

    sig = inspect.signature(CrossPositionLengthBiasPrimitive._position_bias)
    params = list(sig.parameters)
    assert params == ["self", "content_lengths", "out_max", "lmax", "device"]


# ---------------------------------------------------------------------------
# 4. Heterogeneous-length batch / padding / position correspondence.
# ---------------------------------------------------------------------------


def test_mask_layout_tests_pass_on_a_heterogeneous_length_batch() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    core.model.eval()
    result = m.run_mask_layout_tests(core)
    assert result["all_logits_finite"] is True
    assert result["bias_shape_distinct_per_real_length"] is True


def test_different_batch_orderings_of_the_same_lengths_give_identical_per_row_bias() -> None:
    cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    primitive = CrossPositionLengthBiasPrimitive(12, cfg)
    with torch.no_grad():
        primitive.position_bias_out.weight.normal_(mean=0.0, std=0.1)
    device = torch.device("cpu")
    bias_ab = primitive._position_bias([6, 10], 10, 10, device)
    bias_ba = primitive._position_bias([10, 6], 10, 10, device)
    assert torch.allclose(bias_ab[0], bias_ba[1])
    assert torch.allclose(bias_ab[1], bias_ba[0])


# ---------------------------------------------------------------------------
# 5/6. Zero-bias parity (forward + gradient equivalence, softmax-pre-add,
#      scale/mask/dtype preserved) and the gradient path itself.
# ---------------------------------------------------------------------------


def test_zero_bias_parity_check_passes_on_cpu() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    core.model.eval()
    shared_bias_state = m.build_shared_bias_initial_state(core)
    result = m.run_zero_bias_parity_check(core, shared_bias_state)
    assert result["results"]["cpu"]["bias_output_weight_is_zero"] is True
    assert result["results"]["cpu"]["forward_max_abs_diff"] == 0.0
    assert result["results"]["cpu"]["parity_passed"] is True


def test_gradient_path_audit_confirms_real_not_detached_branch() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    core.model.eval()
    shared_bias_state = m.build_shared_bias_initial_state(core)
    result = m.run_gradient_path_audit(core, shared_bias_state)
    assert result["at_init"]["output_grad_nonzero"] is True
    assert result["at_init"]["hidden_grad_zero_as_expected"] is True
    assert result["after_one_optimizer_step"]["hidden_grad_now_nonzero"] is True
    assert result["gradient_path_confirmed_real"] is True
    assert result["both_layers_permanently_zero"] is False


# ---------------------------------------------------------------------------
# 7. I01-I05 shared common part + shared bias initial state; init RNG
#    consumption doesn't leak into the training-stream RNG.
# ---------------------------------------------------------------------------


def test_shared_bias_initial_state_is_deterministic_and_output_weight_zero() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    state_a = m.build_shared_bias_initial_state(core)
    state_b = m.build_shared_bias_initial_state(core)
    for k in state_a:
        assert torch.equal(state_a[k], state_b[k])
    assert torch.count_nonzero(state_a["position_bias_out.weight"]).item() == 0
    assert torch.count_nonzero(state_a["position_bias_hidden.weight"]).item() > 0


def test_building_shared_bias_state_does_not_perturb_the_training_stream_rng() -> None:
    seed, step = 10, 1
    examples_before = m.ibc._generate_step_training_examples(
        seed, step, m.REC004D_TARGET_OPERATION,
        vocab_size=m.REC004D_VOCAB_SIZE, sequence_length_range=m.REC004D_SEQUENCE_LENGTH_RANGE,
    )
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    m.build_shared_bias_initial_state(arch.core)
    examples_after = m.ibc._generate_step_training_examples(
        seed, step, m.REC004D_TARGET_OPERATION,
        vocab_size=m.REC004D_VOCAB_SIZE, sequence_length_range=m.REC004D_SEQUENCE_LENGTH_RANGE,
    )
    assert [tuple(e.input_tokens) for e in examples_before] == [
        tuple(e.input_tokens) for e in examples_after
    ]


# ---------------------------------------------------------------------------
# Real (tiny) end-to-end CPU training: U touches only the base 17,098
# params, P touches base+192; both arms share the same per-step data.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_core_and_bank() -> tuple:
    arch_cfg = SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)
    bank, op_to_id = m._rec004d_reconstruct_16_op_bank_structure(core, seed=0)
    bank.freeze_all()
    bank.eval()
    return core, bank, op_to_id


def test_run_one_arm_u_touches_only_base_params_and_respects_checkpoint_interval(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)

    core_state_before = {k: v.clone() for k, v in core.model.state_dict().items()}
    other_slices_before = {
        op: {k: v.clone() for k, v in bank.get(op_to_id[op]).state_dict().items()}
        for op in op_to_id if op != m.REC004D_TARGET_OPERATION
    }

    base_cfg = CrossPositionPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    torch.manual_seed(0)
    initial_state = CrossPositionPrimitive(pid, base_cfg).state_dict()

    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=2, max_updates_per_run=4,
        existing_validation_examples=8,
    )
    outcome = m.run_one_arm(
        core, bank, op_to_id, "I01", m.REC004D_ARM_U, initial_state, config, tmp_path
    )
    bank.replace_primitive(pid, original_slot)

    for k, v in core.model.state_dict().items():
        assert torch.equal(v, core_state_before[k])
    for op, before in other_slices_before.items():
        after = bank.get(op_to_id[op]).state_dict()
        for k, v in after.items():
            assert torch.equal(v, before[k]), f"{op}.{k} changed but was not the training target"

    steps = [c["step"] for c in outcome["checkpoints"] if "existing_validation" in c]
    assert steps == [0, 2, 4]
    assert set(outcome["final_primitive_state_dict"].keys()) == set(initial_state.keys())
    assert outcome["arm"] == m.REC004D_ARM_U
    assert (tmp_path / "I01" / m.REC004D_ARM_U / "checkpoints" / "step0.pt").is_file()
    final_ckpt = next(c for c in outcome["checkpoints"] if c["step"] == 4)
    assert "bias_ablation" not in (final_ckpt["final_step_extras"] or {})


def test_run_one_arm_p_touches_base_plus_192_and_produces_bias_ablation(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)

    bias_cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(0)
    initial_state = CrossPositionLengthBiasPrimitive(pid, bias_cfg).state_dict()

    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=2, max_updates_per_run=4,
        existing_validation_examples=8,
    )
    outcome = m.run_one_arm(
        core, bank, op_to_id, "I02", m.REC004D_ARM_P, initial_state, config, tmp_path
    )
    bank.replace_primitive(pid, original_slot)

    assert len(outcome["final_primitive_state_dict"]) == len(initial_state)
    final_ckpt = next(c for c in outcome["checkpoints"] if c["step"] == 4)
    extras = final_ckpt["final_step_extras"]
    assert extras is not None
    assert "bias_ablation" in extras
    assert extras["bias_ablation"]["restore_reproduces_original_predictions_exactly"] is True


def test_run_one_arm_optimizer_receives_exactly_the_arms_own_parameter_count(
    tiny_core_and_bank: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)

    captured: dict[str, int] = {}
    original_adamw = torch.optim.AdamW

    def _spy_adamw(params, **kwargs):  # type: ignore[no-untyped-def]
        params = list(params)
        captured["n_params"] = sum(p.numel() for p in params)
        return original_adamw(params, **kwargs)

    monkeypatch.setattr(m.torch.optim, "AdamW", _spy_adamw)
    base_cfg = CrossPositionPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    torch.manual_seed(0)
    initial_state = CrossPositionPrimitive(pid, base_cfg).state_dict()
    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=1, max_updates_per_run=1,
        existing_validation_examples=4,
    )
    m.run_one_arm(core, bank, op_to_id, "I01", m.REC004D_ARM_U, initial_state, config, tmp_path)
    bank.replace_primitive(pid, original_slot)

    expected = sum(p.numel() for p in CrossPositionPrimitive(pid, base_cfg).parameters())
    assert captured["n_params"] == expected == 17098


def test_run_one_arm_records_nan_divergence_as_not_executed(
    tiny_core_and_bank: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    original_cross_entropy = m.F.cross_entropy

    def _nan_cross_entropy(*args, **kwargs):  # type: ignore[no-untyped-def]
        return original_cross_entropy(*args, **kwargs) * float("nan")

    monkeypatch.setattr(m.F, "cross_entropy", _nan_cross_entropy)
    base_cfg = CrossPositionPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    torch.manual_seed(0)
    initial_state = CrossPositionPrimitive(pid, base_cfg).state_dict()
    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=1, max_updates_per_run=2,
        existing_validation_examples=4,
    )
    outcome = m.run_one_arm(
        core, bank, op_to_id, "I01", m.REC004D_ARM_U, initial_state, config, tmp_path
    )
    bank.replace_primitive(pid, original_slot)
    assert outcome["diverged_at_step"] == 1
    statuses = {c["step"]: c.get("status") for c in outcome["checkpoints"] if c["step"] != 0}
    assert statuses[1] == "NOT_EXECUTED"
    assert statuses[2] == "NOT_EXECUTED"


def test_u_and_p_from_the_same_step_receive_identical_training_data(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)

    base_cfg = CrossPositionPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    bias_cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    torch.manual_seed(0)
    u_initial = CrossPositionPrimitive(pid, base_cfg).state_dict()
    torch.manual_seed(1)
    p_initial = CrossPositionLengthBiasPrimitive(pid, bias_cfg).state_dict()

    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=2, max_updates_per_run=2,
        existing_validation_examples=4,
    )
    outcome_u = m.run_one_arm(
        core, bank, op_to_id, "I03", m.REC004D_ARM_U, u_initial, config, tmp_path
    )
    bank.replace_primitive(pid, original_slot)
    outcome_p = m.run_one_arm(
        core, bank, op_to_id, "I03", m.REC004D_ARM_P, p_initial, config, tmp_path
    )
    bank.replace_primitive(pid, original_slot)
    assert outcome_u["data_digests"] == outcome_p["data_digests"]


# ---------------------------------------------------------------------------
# 10. Selection: 4/5 P passing selects nothing; 5/5 fixes I01 (not best-of-5).
# ---------------------------------------------------------------------------


def _fake_outcome(diverged: bool, em_final: float | None) -> dict:
    checkpoints: list[dict] = [{"step": 0}]
    if em_final is not None:
        checkpoints.append(
            {
                "step": m.REC004D_DECISIVE_STEP,
                "existing_validation": {"correct_exact_match": em_final},
            }
        )
    return {"diverged_at_step": (1 if diverged else None), "checkpoints": checkpoints}


def test_select_candidate_requires_all_five_p_runs_not_best_of_five() -> None:
    outcomes = {}
    per_init = {}
    for i, init_id in enumerate(m.REC004D_INIT_IDS, start=1):
        em_p = 0.96 if i < 5 else 0.90  # only 4 of 5 clear the floor
        outcomes[(init_id, m.REC004D_ARM_U)] = _fake_outcome(False, 0.3)
        outcomes[(init_id, m.REC004D_ARM_P)] = _fake_outcome(False, em_p)
        per_init[init_id] = {"em_u": 0.3, "em_p": em_p, "delta_p_minus_u": em_p - 0.3}
    selection = m.select_candidate(outcomes, per_init, 0.95)
    assert selection["status"] == "POSITION_BIAS_VALIDATION_NOT_MET"
    assert selection["selected_arm"] is None


def test_select_candidate_all_five_pass_fixes_p_i01_even_if_not_the_best() -> None:
    outcomes = {}
    per_init = {}
    # I01 is deliberately the WORST of the 5 P runs (still above floor) to
    # confirm the fixed rule never substitutes the best-performing init.
    ems = {"I01": 0.951, "I02": 0.999, "I03": 0.98, "I04": 0.97, "I05": 0.96}
    for init_id in m.REC004D_INIT_IDS:
        outcomes[(init_id, m.REC004D_ARM_U)] = _fake_outcome(False, 0.3)
        outcomes[(init_id, m.REC004D_ARM_P)] = _fake_outcome(False, ems[init_id])
        per_init[init_id] = {
            "em_u": 0.3, "em_p": ems[init_id], "delta_p_minus_u": ems[init_id] - 0.3,
        }
    selection = m.select_candidate(outcomes, per_init, 0.95)
    assert selection["status"] == "FIVE_INIT_VALIDATION_FLOOR_PASS"
    assert selection["selected_arm"] == m.REC004D_ARM_P
    assert selection["selected_init"] == "I01"


def test_select_candidate_incomplete_pair_short_circuits() -> None:
    outcomes = {}
    per_init = {}
    for init_id in m.REC004D_INIT_IDS:
        outcomes[(init_id, m.REC004D_ARM_U)] = _fake_outcome(False, 0.3)
        outcomes[(init_id, m.REC004D_ARM_P)] = _fake_outcome(init_id == "I03", None)
        per_init[init_id] = {"em_u": 0.3, "em_p": 0.99, "delta_p_minus_u": 0.69}
    selection = m.select_candidate(outcomes, per_init, 0.95)
    assert selection["status"] == "PAIR_INCOMPLETE"
    assert selection["selected_arm"] is None


# ---------------------------------------------------------------------------
# 11. Bias-zero ablation intervention leaves the trained weights unmodified.
# ---------------------------------------------------------------------------


def test_bias_ablation_does_not_mutate_the_trained_primitive_weights() -> None:
    cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    torch.manual_seed(0)
    primitive = CrossPositionLengthBiasPrimitive(12, cfg, status=PrimitiveStatus.STABLE)
    with torch.no_grad():
        primitive.position_bias_out.weight.normal_(mean=0.0, std=0.1)
    before = {k: v.clone() for k, v in primitive.state_dict().items()}

    rng = random.Random(0)
    examples = _tiny_examples(rng, n=16)
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    arch.core.model.eval()
    m.run_bias_ablation(arch.core, primitive, examples)

    after = primitive.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"{k} was mutated by the ablation intervention"


def test_zeroed_position_bias_context_manager_restores_weight_on_exit() -> None:
    cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    torch.manual_seed(0)
    primitive = CrossPositionLengthBiasPrimitive(12, cfg)
    with torch.no_grad():
        primitive.position_bias_out.weight.normal_(mean=0.0, std=0.1)
    original = primitive.position_bias_out.weight.detach().clone()
    with primitive.zeroed_position_bias():
        assert torch.count_nonzero(primitive.position_bias_out.weight).item() == 0
    assert torch.equal(primitive.position_bias_out.weight, original)


def _tiny_examples(rng: random.Random, n: int) -> list:
    from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
    from apc.environments.interpreter import run_program
    from apc.environments.task_spec import TaskSpec

    examples = []
    for _ in range(n):
        seq_len = rng.randint(6, 10)
        seq = tuple(rng.randrange(10) for _ in range(seq_len))
        prog = Program(steps=(ProgramStep(operation="MIRROR_HALVES", params={}),))
        res = run_program(prog, seq, 10)
        examples.append(
            Example(
                input_tokens=seq, target_tokens=res.output_tokens, program=prog,
                operation_graph=res.graph, category="known", split="test_fixture",
                vocab_size=10, task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(label="K", primitive_operations=("MIRROR_HALVES",)),
            )
        )
    return examples


# ---------------------------------------------------------------------------
# 12. New-type fresh-process round-trip / cross-type strict rejection.
# ---------------------------------------------------------------------------


def test_operator_serialization_contract_check_passes() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    core.model.eval()
    shared_bias_state = m.build_shared_bias_initial_state(core)
    result = m.run_operator_serialization_contract_check(core, shared_bias_state)
    assert result["new_type_round_trip_ok"] is True
    assert result["old_class_strict_load_of_bias_state_dict_rejected"] is True
    assert result["new_class_strict_load_of_base_only_state_dict_rejected"] is True
    assert result["state_abi_hash_distinguishes_architecture"] is True
    assert result["contract_passed"] is True


def test_state_abi_hash_ignores_tensor_content_but_not_architecture_or_shape() -> None:
    cfg = CrossPositionLengthBiasPrimitiveConfig(operation="MIRROR_HALVES")
    torch.manual_seed(0)
    sd_a = CrossPositionLengthBiasPrimitive(12, cfg).state_dict()
    torch.manual_seed(1)
    sd_b = CrossPositionLengthBiasPrimitive(12, cfg).state_dict()
    assert not all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    hash_a = mb.compute_state_abi_hash(sd_a, architecture_signature="cross_position_length_bias_v1")
    hash_b = mb.compute_state_abi_hash(sd_b, architecture_signature="cross_position_length_bias_v1")
    assert hash_a == hash_b, "ABI hash must be content-independent"

    hash_other_arch = mb.compute_state_abi_hash(sd_a, architecture_signature="cross_position_v1")
    assert hash_other_arch != hash_a


# ---------------------------------------------------------------------------
# 13. Architecture spec / parameter budget.
# ---------------------------------------------------------------------------


def test_architecture_spec_reports_exactly_192_new_params_under_budget() -> None:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    spec = m.build_architecture_spec(core)
    assert spec["new_param_count"] == 192
    assert spec["new_param_count_matches_expected"] is True
    assert spec["budget_status"] == "PASS"
    assert spec["budget_fraction_of_base"] < m.REC004D_BUDGET_FRACTION_CEILING


# ---------------------------------------------------------------------------
# 14. Dynamic guard against build calls during frozen evaluation.
# ---------------------------------------------------------------------------


def test_guard_not_frozen_blocks_run_one_arm_during_frozen_evaluation(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    from apc.evaluation.model_bundle_recovery import EvaluationFrozenError, frozen_evaluation

    core, bank, op_to_id = tiny_core_and_bank
    pid = m.REC004D_TARGET_PHYSICAL_ID
    base_cfg = CrossPositionPrimitiveConfig(
        operation=m.REC004D_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
    )
    torch.manual_seed(0)
    initial_state = CrossPositionPrimitive(pid, base_cfg).state_dict()
    config = m.MirrorPositionBiasRepairConfig(
        output_dir=tmp_path, checkpoint_interval=1, max_updates_per_run=1,
        existing_validation_examples=4,
    )
    with frozen_evaluation(), pytest.raises(EvaluationFrozenError):
        m.run_one_arm(core, bank, op_to_id, "I01", m.REC004D_ARM_U, initial_state, config, tmp_path)


def test_guard_not_frozen_blocks_build_child_bundle_during_frozen_evaluation(
    tmp_path: Path,
) -> None:
    from apc.evaluation.model_bundle_recovery import EvaluationFrozenError, frozen_evaluation

    config = m.MirrorPositionBiasRepairConfig(output_dir=tmp_path)
    with frozen_evaluation(), pytest.raises(EvaluationFrozenError):
        m.build_child_bundle(config, None, {}, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 15. NaN/missing/empty-denominator results are surfaced, not silently
#     dropped -- already covered by test_run_one_arm_records_nan_divergence_
#     as_not_executed above; this adds the empty-denominator (n=8 fixed) case.
# ---------------------------------------------------------------------------


def test_length_eight_has_a_null_fixed_denominator_not_a_fabricated_zero() -> None:
    audit = m._audit_fixed_position_claim()
    assert audit["per_length"]["8"]["fixed_count"] == 0
    # fixed_count == 0 is a real fact (no fixed positions exist at n=8), not
    # a missing-data placeholder -- distinguish it from a null/None result.
    assert audit["per_length"]["8"]["fixed_count"] is not None


# ---------------------------------------------------------------------------
# Dispatcher config loader.
# ---------------------------------------------------------------------------


def test_dispatcher_rec004d_config_loader_reads_yaml(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")
    config_path = tmp_path / "rec004d.yaml"
    config_path.write_text(
        "seed: 10\n"
        f"output_dir: {tmp_path / 'run_001'}\n"
        "existing_validation_examples: 512\n"
        "checkpoint_interval: 250\n"
        "max_updates_per_run: 3000\n",
        encoding="utf-8",
    )
    config = dispatcher._load_rec004d_config(config_path)
    assert config.seed == 10
    assert config.existing_validation_examples == 512
    assert config.checkpoint_interval == 250
    assert config.max_updates_per_run == 3000


def test_dispatcher_rec004d_config_loader_rejects_non_pilot_seed(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")
    config_path = tmp_path / "rec004d.yaml"
    config_path.write_text("seed: 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        dispatcher._load_rec004d_config(config_path)


# ---------------------------------------------------------------------------
# Orchestrator seed guard.
# ---------------------------------------------------------------------------


def test_run_mirror_position_bias_repair_task_refuses_a_non_pilot_seed(tmp_path: Path) -> None:
    bad_config = m.MirrorPositionBiasRepairConfig(output_dir=tmp_path, seed=11)
    with pytest.raises(ValueError, match="pre-registered"):
        m.run_mirror_position_bias_repair_task(bad_config)
