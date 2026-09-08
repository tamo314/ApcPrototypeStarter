"""CPU-only contract tests for Task B-C005REC-004B (MIRROR_HALVES Schedule
Comparison & RG3 Recheck).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (2 conditions x 6000 optimizer steps
against the real REC-004 parent bundle + REC-004A's real step=4000
checkpoints) is never invoked from this test module. These tests exercise
the real production functions -- including genuine (tiny-ladder) end-to-end
CPU training calls -- against small fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apc.evaluation import mirror_schedule_comparison as msc
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)

# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


def test_target_operation_and_fixed_candidates_are_disjoint() -> None:
    assert msc.REC004B_TARGET_OPERATION == "MIRROR_HALVES"
    assert set(msc.REC004B_FIXED_CANDIDATE_OPERATIONS) == {
        "CYCLE_FOUR",
        "ROTATE_TRIPLETS",
        "SWAP_ENDS",
    }
    assert msc.REC004B_TARGET_OPERATION not in msc.REC004B_FIXED_CANDIDATE_OPERATIONS
    changed_ops = {msc.REC004B_TARGET_OPERATION, *msc.REC004B_FIXED_CANDIDATE_OPERATIONS}
    assert len(changed_ops) == 4
    assert changed_ops.isdisjoint(set(msc.REC004B_PROTECTED_OPERATIONS))


def test_protected_operations_are_twelve_plus_shift() -> None:
    assert len(msc.REC004B_PROTECTED_OPERATIONS) == 12
    assert "SHIFT" in msc.REC004B_PROTECTED_OPERATIONS


def test_conditions_and_t_max_are_the_two_preregistered_schedules() -> None:
    assert set(msc.REC004B_CONDITIONS) == {"A_FIXED_TMAX_1000", "B_SINGLE_DECAY_6000"}
    assert msc.REC004B_CONDITION_T_MAX["A_FIXED_TMAX_1000"] == 1000
    assert msc.REC004B_CONDITION_T_MAX["B_SINGLE_DECAY_6000"] == 6000
    assert msc.REC004B_DECISIVE_STEP == 6000 == msc.REC004B_MAX_UPDATES


def test_parent_bundle_is_rec004s_not_rec004as() -> None:
    # REC-004A never published a child bundle, so REC-004B's parent must be
    # the exact same bundle REC-004A itself used as parent (REC-004's own).
    import apc.evaluation.incremental_budget_calibration as ibc

    assert msc.REC004B_PARENT_BUNDLE_ID == ibc.REC004A_PARENT_BUNDLE_ID
    assert msc.REC004B_PARENT_MANIFEST_PATH == ibc.REC004A_PARENT_MANIFEST_PATH


def test_schedule_validation_split_reuses_rec004as_budget_validation() -> None:
    import apc.evaluation.incremental_budget_calibration as ibc

    assert msc.REC004B_SCHEDULE_VALIDATION_SPLIT == ibc.REC004A_BUDGET_VALIDATION_SPLIT
    assert msc.REC004B_RECHECK_QUERY_SPLIT != msc.REC004B_SCHEDULE_VALIDATION_SPLIT


# ---------------------------------------------------------------------------
# MIRROR_HALVES contract fixture (A2).
# ---------------------------------------------------------------------------


def test_mirror_halves_contract_fixture_matches_split_reverse_transform() -> None:
    result = msc._check_mirror_halves_contract()
    assert result["even_length_fixture"]["matches"] is True
    assert result["odd_length_fixture"]["matches"] is True
    assert result["is_parameter_free_live_check"] is True


# ---------------------------------------------------------------------------
# LR trace preflight against the closed-form CosineAnnealingLR formula (B2).
# ---------------------------------------------------------------------------


def test_closed_form_eta_matches_known_table_values() -> None:
    # From the task doc's own pre-registered LR table (section 3.2).
    assert msc._closed_form_eta(0, 1000) == pytest.approx(0.0008)
    assert msc._closed_form_eta(1000, 1000) == pytest.approx(0.00001, abs=1e-9)
    assert msc._closed_form_eta(2000, 1000) == pytest.approx(0.0008, abs=1e-9)
    assert msc._closed_form_eta(1000, 6000) == pytest.approx(0.00074708, abs=1e-7)
    assert msc._closed_form_eta(6000, 6000) == pytest.approx(0.00001, abs=1e-9)


def test_verify_lr_trace_confirms_real_scheduler_matches_closed_form() -> None:
    checkpoints = list(range(0, 6001, 500))
    check_a = msc._verify_lr_trace(1000, checkpoints)
    check_b = msc._verify_lr_trace(6000, checkpoints)
    assert check_a["contract_ok"] is True
    assert check_b["contract_ok"] is True
    assert check_a["trace"][1000] == pytest.approx(0.00001, abs=1e-9)
    assert check_a["trace"][2000] == pytest.approx(0.0008, abs=1e-9)
    assert check_b["trace"][6000] == pytest.approx(0.00001, abs=1e-9)


def test_verify_lr_trace_detects_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tolerance so tight no real float trace can satisfy it -- the
    # LR_TRACE_CONTRACT_FAILURE guard's own falsifiability check.
    check = msc._verify_lr_trace(1000, [0, 1000], tol=-1.0)
    assert check["contract_ok"] is False
    assert 1000 in check["mismatches"] or 0 in check["mismatches"]


# ---------------------------------------------------------------------------
# select_candidate -- pure selection logic (D1): B primary, A fallback,
# both-fail, incomplete pair.
# ---------------------------------------------------------------------------


def _fake_condition_outcome(em_at_6000: float | None, *, diverged: int | None = None) -> dict:
    checkpoints = []
    if em_at_6000 is not None:
        checkpoints.append(
            {"step": 6000, "schedule_validation": {"correct_exact_match": em_at_6000}}
        )
    return {"diverged_at_step": diverged, "checkpoints": checkpoints}


def test_select_candidate_prefers_b_when_b_clears_floor() -> None:
    outcomes = {
        "A_FIXED_TMAX_1000": _fake_condition_outcome(0.40),
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.99),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["selected_condition"] == "B_SINGLE_DECAY_6000"
    assert selection["status"] == "B_SELECTED"


def test_select_candidate_falls_back_to_a_only_when_b_misses_and_a_clears() -> None:
    outcomes = {
        "A_FIXED_TMAX_1000": _fake_condition_outcome(0.97),
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.50),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["selected_condition"] == "A_FIXED_TMAX_1000"
    assert selection["status"] == "A_SELECTED"
    assert selection["note"] == "CONTROL_SELECTED_REPAIR_NOT_SUPPORTED"


def test_select_candidate_picks_b_even_when_both_clear_floor() -> None:
    outcomes = {
        "A_FIXED_TMAX_1000": _fake_condition_outcome(0.96),
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.98),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["selected_condition"] == "B_SINGLE_DECAY_6000"
    assert selection["status"] == "B_SELECTED"


def test_select_candidate_null_when_both_miss_floor() -> None:
    outcomes = {
        "A_FIXED_TMAX_1000": _fake_condition_outcome(0.40),
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.50),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["selected_condition"] is None
    assert selection["status"] == "VALIDATION_TARGET_NOT_MET"


def test_select_candidate_incomplete_when_a_condition_diverged() -> None:
    outcomes = {
        "A_FIXED_TMAX_1000": _fake_condition_outcome(None, diverged=123),
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.98),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["selected_condition"] is None
    assert selection["status"] == "PAIR_INCOMPLETE"


def test_select_candidate_ignores_intermediate_checkpoints() -> None:
    # A passing intermediate checkpoint must never substitute for a missing
    # decisive step=6000 checkpoint (task doc D1: "6000のみ").
    outcomes = {
        "A_FIXED_TMAX_1000": {
            "diverged_at_step": None,
            "checkpoints": [
                {"step": 500, "schedule_validation": {"correct_exact_match": 0.99}},
            ],
        },
        "B_SINGLE_DECAY_6000": _fake_condition_outcome(0.98),
    }
    selection = msc.select_candidate(outcomes, floor=0.95)
    assert selection["em_a_at_6000"] is None
    assert selection["status"] == "PAIR_INCOMPLETE"


# ---------------------------------------------------------------------------
# Real (tiny) end-to-end CPU training: shared init, only MIRROR_HALVES
# touched, checkpoints only at the configured interval.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_core_and_bank() -> tuple:
    import apc.evaluation.incremental_budget_calibration as ibc

    arch_cfg = SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)
    bank, op_to_id = ibc._reconstruct_16_op_bank_structure(core, seed=0)
    bank.freeze_all()
    bank.eval()
    return core, bank, op_to_id


def test_build_shared_initial_state_is_deterministic_and_seed_specific(
    tiny_core_and_bank: tuple,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    state_a = msc.build_shared_initial_state(core, seed=0)
    state_b = msc.build_shared_initial_state(core, seed=0)
    state_c = msc.build_shared_initial_state(core, seed=1)
    for k in state_a:
        assert torch.equal(state_a[k], state_b[k])
    assert any(not torch.equal(state_a[k], state_c[k]) for k in state_a)


def test_run_one_condition_touches_only_mirror_halves_and_respects_checkpoint_interval(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = msc.REC004B_PHYSICAL_IDS[msc.REC004B_TARGET_OPERATION]

    core_state_before = {k: v.clone() for k, v in core.model.state_dict().items()}
    other_slices_before = {
        op: {k: v.clone() for k, v in bank.get(op_to_id[op]).state_dict().items()}
        for op in op_to_id
        if op != msc.REC004B_TARGET_OPERATION
    }

    initial_state = msc.build_shared_initial_state(core, seed=0)
    config = msc.MirrorScheduleComparisonConfig(
        output_dir=tmp_path, checkpoint_interval=2, max_updates=4, schedule_validation_examples=8,
    )
    outcome = msc.run_one_condition(
        core, bank, op_to_id, "A_FIXED_TMAX_1000", initial_state, config, tmp_path
    )

    for k, v in core.model.state_dict().items():
        assert torch.equal(v, core_state_before[k]), f"Core parameter {k} changed"
    for op, before in other_slices_before.items():
        after = bank.get(op_to_id[op]).state_dict()
        for k, v in after.items():
            assert torch.equal(v, before[k]), f"{op}.{k} changed but was not the training target"

    steps = [c["step"] for c in outcome["checkpoints"] if "schedule_validation" in c]
    assert steps == [0, 2, 4]
    assert outcome["condition_id"] == "A_FIXED_TMAX_1000"
    assert outcome["t_max"] == 1000
    assert len(outcome["data_digests"]) == 4
    assert (tmp_path / "A_FIXED_TMAX_1000" / "checkpoints" / "step0.pt").is_file()
    assert (tmp_path / "A_FIXED_TMAX_1000" / "training_states" / "step4.pt").is_file()
    # The eval_bank's MIRROR_HALVES slot reflects the last checkpoint's
    # trained weights (production restores it via its own original-slice
    # save/reload around each condition, exercised by the orchestrator, not
    # this single-condition call).
    pid_after = bank.get(pid).state_dict()
    assert any(not torch.equal(v, initial_state[k].to(v.device)) for k, v in pid_after.items())


def test_run_one_condition_optimizer_receives_only_target_primitive_parameters(
    tiny_core_and_bank: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    pid = msc.REC004B_PHYSICAL_IDS[msc.REC004B_TARGET_OPERATION]
    expected_param_count = sum(p.numel() for p in bank.get(pid).parameters())

    captured: dict[str, int] = {}
    original_adamw = torch.optim.AdamW

    def _spy_adamw(params, **kwargs):  # type: ignore[no-untyped-def]
        params = list(params)
        captured["n_params"] = sum(p.numel() for p in params)
        return original_adamw(params, **kwargs)

    monkeypatch.setattr(msc.torch.optim, "AdamW", _spy_adamw)
    initial_state = msc.build_shared_initial_state(core, seed=0)
    config = msc.MirrorScheduleComparisonConfig(
        output_dir=tmp_path, checkpoint_interval=1, max_updates=1, schedule_validation_examples=4,
    )
    msc.run_one_condition(
        core, bank, op_to_id, "B_SINGLE_DECAY_6000", initial_state, config, tmp_path
    )

    assert captured["n_params"] == expected_param_count


def test_run_one_condition_records_nan_divergence_as_not_executed(
    tiny_core_and_bank: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    original_cross_entropy = msc.F.cross_entropy

    def _nan_cross_entropy(*args, **kwargs):  # type: ignore[no-untyped-def]
        return original_cross_entropy(*args, **kwargs) * float("nan")

    monkeypatch.setattr(msc.F, "cross_entropy", _nan_cross_entropy)
    initial_state = msc.build_shared_initial_state(core, seed=0)
    config = msc.MirrorScheduleComparisonConfig(
        output_dir=tmp_path, checkpoint_interval=1, max_updates=2, schedule_validation_examples=4,
    )
    outcome = msc.run_one_condition(
        core, bank, op_to_id, "A_FIXED_TMAX_1000", initial_state, config, tmp_path
    )
    assert outcome["diverged_at_step"] == 1
    statuses = {c["step"]: c.get("status") for c in outcome["checkpoints"] if c["step"] != 0}
    assert statuses[1] == "NOT_EXECUTED"
    assert statuses[2] == "NOT_EXECUTED"


def test_two_conditions_from_same_shared_init_receive_identical_data_stream(
    tiny_core_and_bank: tuple, tmp_path: Path
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    initial_state = msc.build_shared_initial_state(core, seed=0)
    config = msc.MirrorScheduleComparisonConfig(
        output_dir=tmp_path, checkpoint_interval=2, max_updates=2, schedule_validation_examples=4,
    )
    outcome_a = msc.run_one_condition(
        core, bank, op_to_id, "A_FIXED_TMAX_1000", initial_state, config, tmp_path
    )
    outcome_b = msc.run_one_condition(
        core, bank, op_to_id, "B_SINGLE_DECAY_6000", initial_state, config, tmp_path
    )
    assert outcome_a["data_digests"] == outcome_b["data_digests"]
    # But the two conditions' checkpoint weights must differ after training
    # (different T_max -> different LR -> different optimizer trajectory).
    step2_a = outcome_a["final_primitive_state_dict"]
    step2_b = outcome_b["final_primitive_state_dict"]
    assert any(not torch.equal(step2_a[k], step2_b[k]) for k in step2_a)


# ---------------------------------------------------------------------------
# Dispatcher config loader.
# ---------------------------------------------------------------------------


def test_dispatcher_rec004b_config_loader_reads_yaml(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004b.yaml"
    config_path.write_text(
        "seed: 10\noutput_dir: runs/x/rec004b/run_001\n"
        "schedule_validation_examples: 512\nschedule_validation_floor: 0.9\n"
        "recheck_query_examples: 512\ncheckpoint_interval: 250\nmax_updates: 3000\n",
        encoding="utf-8",
    )
    config = dispatcher._load_rec004b_config(config_path)
    assert config.seed == 10
    assert config.schedule_validation_examples == 512
    assert config.schedule_validation_floor == pytest.approx(0.9)
    assert config.checkpoint_interval == 250
    assert config.max_updates == 3000


def test_dispatcher_rec004b_config_loader_rejects_non_pilot_seed(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004b.yaml"
    config_path.write_text("seed: 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        dispatcher._load_rec004b_config(config_path)


# ---------------------------------------------------------------------------
# Orchestrator seed guard.
# ---------------------------------------------------------------------------


def test_run_mirror_schedule_comparison_task_refuses_a_non_pilot_seed(tmp_path: Path) -> None:
    bad_config = msc.MirrorScheduleComparisonConfig(output_dir=tmp_path, seed=11)
    with pytest.raises(ValueError, match="pre-registered"):
        msc.run_mirror_schedule_comparison_task(bad_config)
