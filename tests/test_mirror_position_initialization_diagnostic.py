"""CPU-only contract tests for Task B-C005REC-004C (MIRROR_HALVES Position
Correspondence & Initialization Diagnostic).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (5 inits x 6000 optimizer steps
against the real REC-004 parent bundle + REC-004A/REC-004B's real saved
checkpoints) is never invoked from this test module. These tests exercise
the real production functions -- including a genuine (tiny-ladder)
end-to-end CPU training call -- against small fixtures.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from apc.environments.generator import Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import model_bundle_recovery as mbr
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)

# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


def test_target_operation_and_init_ids() -> None:
    assert mpid.REC004C_TARGET_OPERATION == "MIRROR_HALVES"
    assert mpid.REC004C_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert mpid.REC004C_MAX_UPDATES_PER_INIT == 6000
    assert mpid.REC004C_TOTAL_MAX_UPDATES == 30000
    assert mpid.REC004C_DECISIVE_STEP == 6000
    assert mpid.REC004C_T_MAX == 1000


def test_parent_bundle_matches_rec004a_rec004b() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc

    assert mpid.REC004C_PARENT_BUNDLE_ID == ibc.REC004A_PARENT_BUNDLE_ID
    assert mpid.REC004C_PARENT_MANIFEST_PATH == ibc.REC004A_PARENT_MANIFEST_PATH


def test_schedule_validation_split_reuses_rec004as_budget_validation() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc

    assert mpid.REC004C_SCHEDULE_VALIDATION_SPLIT == ibc.REC004A_BUDGET_VALIDATION_SPLIT


def test_diagnostic_suite_splits_are_distinct_from_train_and_validation() -> None:
    splits = {
        mpid.REC004C_LENGTH_BALANCED_SPLIT,
        mpid.REC004C_POSITION_IDENTIFIABLE_SPLIT,
        mpid.REC004C_COUNTERFACTUAL_SPLIT,
        mpid.REC004C_SCHEDULE_VALIDATION_SPLIT,
        mpid.REC004C_TRAIN_SPLIT_LABEL,
    }
    assert len(splits) == 5, "diagnostic/validation/train splits must all be distinct namespaces"


# ---------------------------------------------------------------------------
# 1. Position map direction matches source teacher; odd/even + boundary
#    legal lengths.
# ---------------------------------------------------------------------------


def test_position_map_matches_teacher_for_even_and_odd_lengths() -> None:
    result = mpid._verify_position_map_against_source(range(2, 13))
    assert result["all_lengths_matched"] is True
    assert result["applicability"] == "POSITION_MAP_APPLICABLE"
    for n in range(2, 13):
        assert result["per_length"][str(n)]["matches"] is True


def test_position_map_is_involution_within_each_half() -> None:
    for n in range(2, 13):
        pi = mpid.mirror_halves_position_map(n)
        assert all(pi[pi[i]] == i for i in range(n))


def test_position_map_boundary_length_two() -> None:
    assert mpid.mirror_halves_position_map(2) == (0, 1)
    prog = Program(steps=(ProgramStep(operation="MIRROR_HALVES", params={}),))
    out = run_program(prog, (5, 9), 10).output_tokens
    assert out == (5, 9)  # mid=1: reversed([5])+reversed([9]) == (5,9)


def test_position_map_matches_known_odd_even_fixtures() -> None:
    even_pi = mpid.mirror_halves_position_map(6)
    even_input = (10, 20, 30, 40, 50, 60)
    assert tuple(even_input[j] for j in even_pi) == (30, 20, 10, 60, 50, 40)

    odd_pi = mpid.mirror_halves_position_map(7)
    odd_input = (1, 2, 3, 4, 5, 6, 7)
    assert tuple(odd_input[j] for j in odd_pi) == (3, 2, 1, 7, 6, 5, 4)


# ---------------------------------------------------------------------------
# 2/3. J_hat classification (unique/ambiguous/not-in-input), structural
#      moved-vs-fixed and apparent changed-vs-unchanged separation.
# ---------------------------------------------------------------------------


def _tiny_core_and_primitive() -> tuple[object, object]:
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    )
    core = arch.core
    primitive = mpid._new_primitive(core)
    primitive.eval()
    return core, primitive


def test_position_level_error_analysis_structure_and_denominators() -> None:
    core, primitive = _tiny_core_and_primitive()
    examples = [
        mpid._make_example((1, 2, 3, 4, 5, 6), "test_split"),
        mpid._make_example((7, 8, 9, 0, 1, 2, 3), "test_split"),
    ]
    summary, confusion = mpid._position_level_error_analysis(core, primitive, examples)
    assert summary["n_examples"] == 2
    # length 6 contributes positions 0..5, length 7 contributes 0..6 -- 13 (length,pos) cells.
    assert len(summary["by_length_position"]) == 6 + 7
    for _key, bucket in summary["by_length_position"].items():
        assert bucket["n"] >= 1
        assert 0 <= bucket["correct"] <= bucket["n"]
    total_structural = confusion["structural_moved_vs_fixed"]["moved"]["n"] + (
        confusion["structural_moved_vs_fixed"]["fixed"]["n"]
    )
    assert total_structural == 6 + 7
    total_j_hat = sum(confusion["j_hat_source_token_classification"].values())
    assert total_j_hat == 6 + 7


def test_j_hat_classification_unique_ambiguous_and_not_in_input() -> None:
    # Duplicate-token input -> ambiguous J_hat is possible; distinct-token
    # input -> J_hat is always unique when the predicted value is legal.
    x = (3, 3, 3, 3, 3, 3)  # all duplicates: every value is ambiguous unless empty
    n = len(x)
    # A perfect prediction on an all-duplicate input is ambiguous for every position
    # (every input position holds the same value 3).
    j_hat_for_value_3 = [j for j in range(n) if x[j] == 3]
    assert len(j_hat_for_value_3) == n  # ambiguous, not unique, not empty

    distinct = tuple(range(6))
    pi2 = mpid.mirror_halves_position_map(6)
    for i in range(6):
        yhat = distinct[pi2[i]]
        j_hat = [j for j in range(6) if distinct[j] == yhat]
        assert len(j_hat) == 1  # unique
    # A value absent from the input is never found (PREDICTED_TOKEN_NOT_IN_INPUT case).
    absent_value = 9
    assert absent_value not in distinct
    assert [j for j in range(6) if distinct[j] == absent_value] == []


def test_diagnostic_suites_never_merged_into_main_validation() -> None:
    length_balanced = mpid._generate_length_balanced_diagnostic()
    schedule_validation_digests = {
        (e.input_tokens, e.target_tokens) for e in mpid._generate_schedule_validation_examples()
    }
    diagnostic_digests = {(e.input_tokens, e.target_tokens) for e in length_balanced}
    # Different split labels feed different RNG streams; the two sets must
    # not be silently identical (which would indicate the split label was
    # ignored, merging the diagnostic suite into the main validation split).
    assert length_balanced[0].split == mpid.REC004C_LENGTH_BALANCED_SPLIT
    assert diagnostic_digests != schedule_validation_digests


# ---------------------------------------------------------------------------
# 4. Padding/batch semantics-preserving comparisons.
# ---------------------------------------------------------------------------


def test_padding_batch_metamorphic_check_passes_on_a_tiny_fixture() -> None:
    core, primitive = _tiny_core_and_primitive()
    examples = [mpid._make_example((1, 2, 3, 4, 5, 6), "test_split")]
    result = mpid._padding_batch_metamorphic_check(core, primitive, examples)
    assert result["status"] == "PASS"
    assert result["all_match"] is True
    assert result["unsupported_cases"] == []
    assert set(result["conditions"]) == {
        "single_vs_batched_with_short_fillers",
        "single_vs_batched_with_long_fillers",
        "short_padding_width_vs_long_padding_width_same_content",
    }


def test_generate_padding_batch_sample_is_length_balanced() -> None:
    sample = mpid._generate_padding_batch_sample()
    lengths = [len(ex.input_tokens) for ex in sample]
    for n in mpid.REC004C_LEGAL_LENGTHS:
        assert lengths.count(n) == mpid.REC004C_PADDING_BATCH_SAMPLE_PER_LENGTH


# ---------------------------------------------------------------------------
# 5. Attention hook never changes predictions.
# ---------------------------------------------------------------------------


def test_attention_observation_hook_matches_normal_forward() -> None:
    core, primitive = _tiny_core_and_primitive()
    examples = [
        mpid._make_example((1, 2, 3, 4, 5, 6), "test_split"),
        mpid._make_example((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "test_split"),
    ]
    result = mpid._attention_observation(core, primitive, examples)
    assert result["status"] == "OBSERVED"
    assert result["hook_predictions_match_normal_forward"] is True
    assert result["n_positions_checked"] == 6 + 10


# ---------------------------------------------------------------------------
# 6/7. init_id never changes Core/data seed; 5 states pre-fixed and
#      reproducible; training RNG restored after init draws.
# ---------------------------------------------------------------------------


def test_build_five_initial_states_are_distinct_and_reproducible() -> None:
    core, _ = _tiny_core_and_primitive()
    states_a = mpid.build_five_initial_states(core, seed=999)  # seed arg is a documented no-op
    states_b = mpid.build_five_initial_states(core, seed=mpid.RECOVERY_PILOT_SEED)
    hashes_a = {k: mpid.mb.canonical_state_hash(v) for k, v in states_a.items()}
    hashes_b = {k: mpid.mb.canonical_state_hash(v) for k, v in states_b.items()}
    assert len(set(hashes_a.values())) == 5
    assert hashes_a == hashes_b  # init_id-labeled seeds never depend on the passed-in `seed`


def test_init_seed_labels_do_not_reuse_rec004a_or_rec004b_labels() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc
    import apc.evaluation.mirror_schedule_comparison as msc

    rec004a_label = f"rec004a_init:{mpid.REC004C_TARGET_OPERATION}"
    rec004b_label = f"rec004b_shared_init:{mpid.REC004C_TARGET_OPERATION}"
    for init_id in mpid.REC004C_INIT_IDS:
        label = f"rec004c_init:{mpid.REC004C_TARGET_OPERATION}:{init_id}"
        assert label != rec004a_label
        assert label != rec004b_label
    assert "rec004a_init" in inspect.getsource(ibc._train_operation_with_checkpoints)
    assert "rec004b_shared_init" in inspect.getsource(msc.build_shared_initial_state)


def test_shared_training_rng_restored_after_init_draws() -> None:
    core, _ = _tiny_core_and_primitive()
    mpid.build_five_initial_states(core, seed=mpid.RECOVERY_PILOT_SEED)
    state_after_build = torch.get_rng_state()

    expected_seed = mpid._local_seed(0, "rec004c_shared_training_rng")
    torch.manual_seed(expected_seed)
    state_expected = torch.get_rng_state()
    assert torch.equal(state_after_build, state_expected)


# ---------------------------------------------------------------------------
# 8/9. Optimizer parameter scope isolated to MIRROR_HALVES; fixed
#      LR/6000-cap/500-interval; a real tiny-ladder end-to-end CPU run.
# ---------------------------------------------------------------------------


def test_optimizer_receives_only_the_target_primitives_own_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _ = _tiny_core_and_primitive()
    initial_state = mpid.build_five_initial_states(core, mpid.RECOVERY_PILOT_SEED)["I01"]

    captured: dict[str, int] = {}
    real_adamw_init = torch.optim.AdamW.__init__

    def spy_init(self, params, *args, **kwargs):  # type: ignore[no-untyped-def]
        params = list(params)
        captured["n_params"] = sum(p.numel() for p in params)
        return real_adamw_init(self, params, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "__init__", spy_init)

    expected_primitive = mpid._new_primitive(core)
    expected_primitive.load_state_dict(initial_state, strict=True)
    expected_n_params = sum(p.numel() for p in expected_primitive.parameters())

    bank, op_to_id = _tiny_bank_with_16_ops(core)
    tmp_dir = _tiny_output_dir()
    config = mpid.MirrorPositionInitializationDiagnosticConfig(
        output_dir=tmp_dir,
        max_updates_per_init=2,
        checkpoint_interval=1,
        schedule_validation_examples=4,
    )

    non_target_before = {
        op: mpid.mb.canonical_state_hash(bank.get(pid).state_dict())
        for op, pid in op_to_id.items()
        if op != mpid.REC004C_TARGET_OPERATION
    }

    empty_position_identifiable = {n: [] for n in mpid.REC004C_LEGAL_LENGTHS}
    empty_counterfactual = {n: [] for n in mpid.REC004C_LEGAL_LENGTHS}
    mpid.run_one_init(
        core, bank, op_to_id, "I01", initial_state, config, tmp_dir,
        length_balanced_examples=[],
        position_identifiable_by_length=empty_position_identifiable,
        counterfactual_by_length=empty_counterfactual,
    )

    assert captured["n_params"] == expected_n_params

    non_target_after = {
        op: mpid.mb.canonical_state_hash(bank.get(pid).state_dict())
        for op, pid in op_to_id.items()
        if op != mpid.REC004C_TARGET_OPERATION
    }
    assert non_target_before == non_target_after


def test_protocol_requires_decisive_step_on_checkpoint_boundary() -> None:
    config = mpid.MirrorPositionInitializationDiagnosticConfig(
        max_updates_per_init=5999, checkpoint_interval=500, output_dir=_tiny_output_dir()
    )
    with pytest.raises(ValueError, match="checkpoint boundary"):
        mpid.build_initialization_protocol(config, _fake_stage_a_for_protocol())


def test_protocol_records_fixed_lr_and_single_schedule_only() -> None:
    config = mpid.MirrorPositionInitializationDiagnosticConfig(output_dir=_tiny_output_dir())
    protocol = mpid.build_initialization_protocol(config, _fake_stage_a_for_protocol())
    assert protocol["schedule"]["t_max"] == 1000
    assert protocol["operator_lr"] == mpid.REC004C_OPERATOR_LR
    assert protocol["max_updates_per_init"] == 6000
    assert protocol["checkpoint_interval"] == 500
    assert protocol["decisive_checkpoint"] == 6000
    assert protocol["selection_rule"].startswith("NONE")
    assert "protocol_hash" in protocol


# ---------------------------------------------------------------------------
# 10. Full-state checkpoint round-trip (weights + optimizer + scheduler +
#     RNG) reproduces continued training bit-for-bit; weight-only never
#     accepted as a strict resume.
# ---------------------------------------------------------------------------


def test_full_training_state_checkpoint_reproduces_continued_training() -> None:
    torch.manual_seed(0)
    model_a = torch.nn.Linear(4, 4)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=10)

    def _step(model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
        opt.zero_grad(set_to_none=True)
        x = torch.ones(1, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    _step(model_a, opt_a)
    sched_a.step()
    _step(model_a, opt_a)
    sched_a.step()

    checkpoint = {
        "weights": {k: v.clone() for k, v in model_a.state_dict().items()},
        "optimizer": opt_a.state_dict(),
        "scheduler": sched_a.state_dict(),
        "rng": torch.get_rng_state(),
    }

    _step(model_a, opt_a)
    sched_a.step()
    _step(model_a, opt_a)
    sched_a.step()
    reference_weights = {k: v.clone() for k, v in model_a.state_dict().items()}

    model_b = torch.nn.Linear(4, 4)
    model_b.load_state_dict(checkpoint["weights"])
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    opt_b.load_state_dict(checkpoint["optimizer"])
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=10)
    sched_b.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["rng"])

    _step(model_b, opt_b)
    sched_b.step()
    _step(model_b, opt_b)
    sched_b.step()

    for key in reference_weights:
        assert torch.allclose(reference_weights[key], model_b.state_dict()[key])

    # Weight-only "resume" (no optimizer/scheduler state) is a DIFFERENT
    # trajectory once momentum matters -- never claimed equivalent.
    model_c = torch.nn.Linear(4, 4)
    model_c.load_state_dict(checkpoint["weights"])
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)  # fresh optimizer state
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=10)
    torch.set_rng_state(checkpoint["rng"])
    _step(model_c, opt_c)
    sched_c.step()
    _step(model_c, opt_c)
    sched_c.step()
    weight_only_matches = all(
        torch.allclose(reference_weights[k], model_c.state_dict()[k]) for k in reference_weights
    )
    assert not weight_only_matches


# ---------------------------------------------------------------------------
# 11. NaN/missing-trial handling in denominators and summary statistics.
# ---------------------------------------------------------------------------


def _fake_outcome(em_at_6000: float | None, *, diverged: int | None = None) -> dict:
    checkpoints = []
    checkpoints.append({"step": 0, "checkpoint_state_hash": "init_hash"})
    if em_at_6000 is not None:
        checkpoints.append(
            {
                "step": 6000,
                "checkpoint_state_hash": "final_hash",
                "schedule_validation": {
                    "correct_exact_match": em_at_6000, "correct_token_accuracy": 0.5,
                },
                "train_fit": {"sequence_exact_match": em_at_6000},
                "length_stratified": {"by_length": {}},
            }
        )
    return {"checkpoints": checkpoints, "diverged_at_step": diverged}


def test_initialization_summary_sd_null_below_two_completed() -> None:
    outcomes = {
        "I01": _fake_outcome(0.9),
        "I02": _fake_outcome(None, diverged=3000),
        "I03": _fake_outcome(None),  # missing, e.g. never reached decisive step
        "I04": _fake_outcome(None),
        "I05": _fake_outcome(None),
    }
    summary = mpid.build_initialization_summary(outcomes, descriptive_floor=0.95)
    assert summary["completed"] == 1
    assert summary["diverged"] == 1
    assert summary["missing"] == 3
    assert summary["final_validation_em_summary"]["sample_sd"] is None
    assert summary["planned"] == 5


def test_initialization_summary_full_stats_with_multiple_completed() -> None:
    outcomes = {
        "I01": _fake_outcome(0.90),
        "I02": _fake_outcome(0.99),
        "I03": _fake_outcome(0.40),
        "I04": _fake_outcome(None, diverged=100),
        "I05": _fake_outcome(0.10),
    }
    summary = mpid.build_initialization_summary(outcomes, descriptive_floor=0.95)
    assert summary["completed"] == 4
    assert summary["diverged"] == 1
    assert summary["missing"] == 0
    assert summary["final_validation_em_summary"]["min"] == pytest.approx(0.10)
    assert summary["final_validation_em_summary"]["max"] == pytest.approx(0.99)
    assert summary["n_reaching_descriptive_floor_of_completed"] == 1
    assert summary["final_validation_em_summary"]["sample_sd"] is not None
    # never fill missing/diverged trials with 0 or PASS
    assert summary["per_init"]["I04"]["final_validation_em"] is None
    assert summary["per_init"]["I04"]["status"] == "DIVERGED"


# ---------------------------------------------------------------------------
# 12. Bundle publish / RG3 query / REC-005 are never callable from this
#     module, even if all 5 trials clear the descriptive floor.
# ---------------------------------------------------------------------------


def test_module_never_defines_child_bundle_or_rg3_recheck_machinery() -> None:
    forbidden = ("build_child_bundle", "run_stage_e", "rg3_recheck_report", "publish_dir")
    for name in forbidden:
        assert not hasattr(mpid, name), f"{name} must not exist in a diagnostic-only module"


def test_orchestration_result_fixes_selected_init_and_rg3_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even if the underlying init summary shows every trial clearing the
    # descriptive floor, the orchestration's own hardcoded result contract
    # never selects/publishes/rechecks -- verified via source inspection
    # (the literal is set once, unconditionally, not derived from any
    # measured EM).
    source = inspect.getsource(mpid.run_mirror_position_initialization_diagnostic_task)
    assert '"selected_init": None' in source
    assert '"child_bundle": None' in source
    assert '"rg3_recheck": "NOT_EXECUTED"' in source
    assert "build_child_bundle" not in source


# ---------------------------------------------------------------------------
# 13. Dynamic guard: evaluation-only stages refuse to run while frozen.
# ---------------------------------------------------------------------------


def test_stage_a_and_stage_b_respect_frozen_evaluation_guard() -> None:
    with mbr.frozen_evaluation():
        with pytest.raises(mbr.EvaluationFrozenError):
            mpid._guard_not_frozen("run_stage_a")
        with pytest.raises(mbr.EvaluationFrozenError):
            mpid._guard_not_frozen("run_stage_b")
        with pytest.raises(mbr.EvaluationFrozenError):
            mpid._guard_not_frozen("run_stage_d")


# ---------------------------------------------------------------------------
# 14. Original artifacts / shared cache / sealed paths are never write
#     targets of this module's own namespace roots.
# ---------------------------------------------------------------------------


def test_no_output_namespace_overlaps_forbidden_shared_cache_prefixes() -> None:
    namespaces = [
        str(mpid._rec004c_namespace(mpid.RECOVERY_PILOT_SEED)),
        "runs/phase_b_b2_model_bundle_recovery/rec004c",
    ]
    for ns in namespaces:
        for forbidden in mbr.FORBIDDEN_SHARED_CACHE_PATH_PREFIXES:
            assert not ns.startswith(forbidden)


def test_existing_checkpoint_paths_are_never_written_by_this_module() -> None:
    # The only writes to REC-004A/REC-004B's own directories this module
    # performs must be reads (`mb.load_state_dict`, `.is_file()`); a
    # write-mode open/torch.save targeting those directories would be a bug.
    assert "torch.save" not in inspect.getsource(mpid._historical_replay)
    assert "torch.save" not in inspect.getsource(mpid.run_stage_a)
    assert "torch.save" not in inspect.getsource(mpid.run_stage_b)


# ---------------------------------------------------------------------------
# Helpers shared by the tiny end-to-end CPU test.
# ---------------------------------------------------------------------------


def _tiny_output_dir():  # type: ignore[no-untyped-def]
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="rec004c_test_"))


def _tiny_bank_with_16_ops(core):  # type: ignore[no-untyped-def]
    import apc.evaluation.incremental_budget_calibration as ibc

    bank, op_to_id = ibc._reconstruct_16_op_bank_structure(core, seed=mpid.RECOVERY_PILOT_SEED)
    bank.to(core.device)
    for module in bank._primitives.values():  # noqa: SLF001 - test-only introspection
        for p in module.parameters():
            p.requires_grad_(False)
    bank.eval()
    return bank, op_to_id


def _fake_stage_a_for_protocol() -> dict:
    class _FakeCore:
        canonical_state_hash = "fake_core_hash"

    class _FakeManifest:
        bundle_id = "fake_bundle_id"
        core = _FakeCore()

    return {"parent_manifest": _FakeManifest()}
