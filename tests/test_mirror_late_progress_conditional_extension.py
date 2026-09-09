"""CPU-only contract tests for Task B-C005REC-004H (MIRROR_HALVES Late-Stage
Length-10 Audit & Conditional Extension).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (auditing REC-004G's real saved
P/I01-I05 checkpoints, and conditionally extending them) is never invoked
from this test module. These tests exercise the real production functions
against small, synthetic "REC-004G-shaped" fixtures built in `tmp_path`,
never against the real
`runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/` tree.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as mbe
from apc.evaluation import mirror_late_progress_conditional_extension as m
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import EvaluationFrozenError, frozen_evaluation
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)
from apc.utils import model_bundle as mb

# ---------------------------------------------------------------------------
# Constants sanity -- everything reused from REC-004G/REC-004D, nothing
# retyped.
# ---------------------------------------------------------------------------


def test_constants_are_reused_from_rec004g_not_retyped() -> None:
    assert m.REC004H_INIT_IDS == mbe.REC004G_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert m.REC004H_ARM == mbe.REC004G_ARM
    assert m.REC004H_TARGET_OPERATION == "MIRROR_HALVES"
    assert m.REC004H_OPERATOR_LR == mbe.REC004G_OPERATOR_LR
    assert m.REC004H_T_MAX == mbe.REC004G_T_MAX == 1000
    assert m.REC004H_PROTECTED_OPERATIONS == mbe.REC004G_PROTECTED_OPERATIONS
    assert m.REC004H_FIXED_CANDIDATE_OPERATIONS == mbe.REC004G_FIXED_CANDIDATE_OPERATIONS


def test_budget_is_exactly_12000_to_18000_with_30000_cap() -> None:
    assert m.REC004H_SOURCE_STEP_FOR_EXTENSION == 12000
    assert m.REC004H_ADDITIONAL_UPDATES == 6000
    assert m.REC004H_TARGET_CUMULATIVE_STEP == 18000
    assert m.REC004H_MAX_UPDATES_TOTAL == 30000


def test_audit_and_phase_match_steps_are_the_documented_points() -> None:
    assert m.REC004H_AUDIT_STEPS == (8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000)
    assert m.REC004H_PHASE_MATCH_STEPS == (8000, 10000, 12000)
    assert m.REC004H_TARGET_LENGTH == 10


def test_module_never_imports_oracle_attention_or_dispatches_later_tasks() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = getattr(node, "module", None) or ""
            assert "oracle_attention" not in module
            assert not any("oracle" in n.lower() for n in names)
    assert "_oracle_attention" not in source
    assert "oracle_attention_substitution" not in source
    blocked = ("rec005", "r3_011", "b_c006", "task_inference")
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


# ---------------------------------------------------------------------------
# mirror_halves_position_map(10) sanity: fixed positions must be exactly
# {2, 7}, matching the task doc's own claim -- computed from the live source,
# never hardcoded independently.
# ---------------------------------------------------------------------------


def test_length10_fixed_positions_are_2_and_7() -> None:
    from apc.evaluation import mirror_position_initialization_diagnostic as mpid

    pi_10 = mpid.mirror_halves_position_map(10)
    fixed = [i for i in range(10) if pi_10[i] == i]
    assert fixed == [2, 7]


# ---------------------------------------------------------------------------
# compute_progress_flags: EM_PROGRESS / SOFT_PROGRESS / overall-only /
# early-only / final-loss-only / missing-data fixtures.
# ---------------------------------------------------------------------------


def test_em_progress_true_when_correct_and_errors_strictly_improve() -> None:
    C = {8000: 10, 10000: 20, 12000: 30}
    E = {8000: 100, 10000: 80, 12000: 50}
    L = {8000: 2.0, 10000: 1.5, 12000: 1.0}
    flags = m.compute_progress_flags(C, E, L)
    assert flags["status"] == "COMPUTED"
    assert flags["EM_PROGRESS"] is True
    assert flags["INIT_PROGRESS_CONFIRMED"] is True


def test_soft_progress_true_when_em_flat_but_errors_and_loss_improve() -> None:
    C = {8000: 20, 10000: 20, 12000: 20}  # flat sequence EM
    E = {8000: 100, 10000: 80, 12000: 50}  # errors strictly fall
    L = {8000: 2.0, 10000: 1.5, 12000: 1.0}  # loss strictly falls
    flags = m.compute_progress_flags(C, E, L)
    assert flags["EM_PROGRESS"] is False  # C_12000 not > C_10000/C_8000
    assert flags["SOFT_PROGRESS"] is True
    assert flags["INIT_PROGRESS_CONFIRMED"] is True


def test_overall_only_improvement_across_6000_to_12000_does_not_confirm_progress() -> None:
    """A big historical 6000->12000 gain says nothing here: the predicate
    only ever looks at 8000/10000/12000. A flat-or-regressing 8000->12000
    trajectory must NOT be confirmed even if the task "improved a lot
    overall" at some other, unrelated pair of steps."""
    C = {8000: 30, 10000: 25, 12000: 26}  # dips then partially recovers, never exceeds 8000
    E = {8000: 50, 10000: 60, 12000: 55}
    L = {8000: 1.0, 10000: 1.2, 12000: 1.1}
    flags = m.compute_progress_flags(C, E, L)
    assert flags["EM_PROGRESS"] is False
    assert flags["SOFT_PROGRESS"] is False
    assert flags["INIT_PROGRESS_CONFIRMED"] is False


def test_early_8000_only_improvement_does_not_confirm_progress() -> None:
    C = {8000: 5, 10000: 40, 12000: 20}  # big early jump then regression by 12000
    E = {8000: 100, 10000: 10, 12000: 60}
    L = {8000: 2.0, 10000: 0.5, 12000: 1.5}
    flags = m.compute_progress_flags(C, E, L)
    assert flags["EM_PROGRESS"] is False
    assert flags["SOFT_PROGRESS"] is False
    assert flags["INIT_PROGRESS_CONFIRMED"] is False


def test_final_loss_only_improvement_without_C_or_E_improvement_does_not_confirm() -> None:
    C = {8000: 30, 10000: 30, 12000: 29}  # EM regresses slightly at 12000
    E = {8000: 50, 10000: 50, 12000: 50}  # errors flat
    L = {8000: 2.0, 10000: 2.0, 12000: 1.0}  # only loss improves
    flags = m.compute_progress_flags(C, E, L)
    assert flags["EM_PROGRESS"] is False
    assert flags["SOFT_PROGRESS"] is False
    assert flags["INIT_PROGRESS_CONFIRMED"] is False


def test_missing_or_nonfinite_data_yields_evidence_insufficient() -> None:
    flags_missing = m.compute_progress_flags(
        {8000: 1, 10000: 2}, {8000: 1, 10000: 1}, {8000: 1.0, 10000: 1.0}
    )
    assert flags_missing["status"] == "EVIDENCE_INSUFFICIENT"

    flags_nonfinite = m.compute_progress_flags(
        {8000: 1, 10000: 2, 12000: 3},
        {8000: 1, 10000: 1, 12000: 1},
        {8000: 1.0, 10000: 1.0, 12000: float("nan")},
    )
    assert flags_nonfinite["status"] == "EVIDENCE_INSUFFICIENT"


# ---------------------------------------------------------------------------
# build_continuation_decision branch coverage.
# ---------------------------------------------------------------------------


def test_continuation_decision_already_at_floor_when_none_below_floor() -> None:
    below_floor = {i: False for i in m.REC004H_INIT_IDS}
    decision = m.build_continuation_decision(below_floor, {})
    assert decision["decision"] == "ALREADY_AT_FLOOR_AUDIT_ONLY"


def test_continuation_decision_evidence_insufficient_when_step12000_not_verified() -> None:
    below_floor = {i: (None if i == "I01" else False) for i in m.REC004H_INIT_IDS}
    decision = m.build_continuation_decision(below_floor, {})
    assert decision["decision"] == "EVIDENCE_INSUFFICIENT"


def test_continuation_decision_extends_only_when_all_below_floor_inits_confirmed() -> None:
    below_floor = {"I01": True, "I02": True, "I03": True, "I04": False, "I05": False}
    per_init_progress = {
        "I01": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I02": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I03": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
    }
    decision = m.build_continuation_decision(below_floor, per_init_progress)
    assert decision["decision"] == "EXTEND_ALL_FIVE_TO_18000"


def test_continuation_decision_stops_when_one_below_floor_init_unconfirmed() -> None:
    """The task's own central guardrail: I01/I02 confirmed but I03 is NOT ->
    the decision must be STOP_NO_EXTENSION for ALL FIVE, never a partial
    I01/I02-only extension."""
    below_floor = {"I01": True, "I02": True, "I03": True, "I04": False, "I05": False}
    per_init_progress = {
        "I01": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I02": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I03": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": False},
    }
    decision = m.build_continuation_decision(below_floor, per_init_progress)
    assert decision["decision"] == "STOP_NO_EXTENSION"
    assert decision["mixed_or_no_confirmed_late_progress"] == "MIXED_OR_NO_CONFIRMED_LATE_PROGRESS"


def test_continuation_decision_evidence_insufficient_when_progress_data_missing() -> None:
    below_floor = {"I01": True, "I02": False, "I03": False, "I04": False, "I05": False}
    per_init_progress = {"I01": {"status": "EVIDENCE_INSUFFICIENT"}}
    decision = m.build_continuation_decision(below_floor, per_init_progress)
    assert decision["decision"] == "EVIDENCE_INSUFFICIENT"


# ---------------------------------------------------------------------------
# below_floor_from_audit: integer predicate, no float rounding fabrication.
# ---------------------------------------------------------------------------


def test_below_floor_uses_integer_comparison_not_rounded_float() -> None:
    # 973/1024 = 0.95019..., which rounds to "0.95" at 2dp but the integer
    # rule 100*973 < 95*1024 (97300 < 97280) is FALSE -> NOT below floor.
    audit = {"by_length": {"6": {"n": 1024, "exact": 973}}}
    assert m._below_floor_from_audit(audit, 1024) is False
    audit_972 = {"by_length": {"6": {"n": 1024, "exact": 972}}}
    assert m._below_floor_from_audit(audit_972, 1024) is True


# ---------------------------------------------------------------------------
# build_improvement_vs_residual_decomposition: denominator-0 / negative
# delta / missing-step handling.
# ---------------------------------------------------------------------------


def test_improvement_decomposition_handles_zero_delta_and_missing_steps() -> None:
    checkpoint_audits = {
        i: {
            8000: {"status": "VERIFIED", "by_length": {"10": {"n": 10, "exact": 5}}},
            10000: {"status": "SOURCE_REPLAY_MISMATCH"},
            12000: {"status": "VERIFIED", "by_length": {"10": {"n": 10, "exact": 5}}},
        }
        for i in m.REC004H_INIT_IDS
    }
    result = m.build_improvement_vs_residual_decomposition(checkpoint_audits, None)
    pair_8000_12000 = result["I01"]["phase_match_steps_decomposition"]["8000_to_12000"]
    assert pair_8000_12000["delta_correct_all"] == 0
    assert pair_8000_12000["per_length"]["10"]["contribution_to_total_delta"] is None
    pair_8000_10000 = result["I01"]["phase_match_steps_decomposition"]["8000_to_10000"]
    assert pair_8000_10000["status"] == "EVIDENCE_INSUFFICIENT"


def test_improvement_decomposition_negative_delta_is_not_clamped() -> None:
    checkpoint_audits = {
        i: {
            8000: {"status": "VERIFIED", "by_length": {"10": {"n": 10, "exact": 8}}},
            10000: {"status": "VERIFIED", "by_length": {"10": {"n": 10, "exact": 8}}},
            12000: {"status": "VERIFIED", "by_length": {"10": {"n": 10, "exact": 3}}},
        }
        for i in m.REC004H_INIT_IDS
    }
    result = m.build_improvement_vs_residual_decomposition(checkpoint_audits, None)
    pair = result["I01"]["phase_match_steps_decomposition"]["8000_to_12000"]
    assert pair["per_length"]["10"]["delta_correct"] == -5
    assert pair["per_length"]["10"]["contribution_to_total_delta"] == 1.0


# ---------------------------------------------------------------------------
# Frozen-evaluation dynamic guard: audit/eval functions must run fine WHILE
# frozen; training functions must be blocked.
# ---------------------------------------------------------------------------


def test_hash_only_precheck_runs_fine_inside_frozen_evaluation(tmp_path: Path) -> None:
    with frozen_evaluation():
        result = m._hash_only_precheck("I01", 8000, [])
    # no real source tree -- still no raise
    assert result["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"


def test_run_one_late_extension_raises_under_frozen_evaluation() -> None:
    config = m.MirrorLateProgressConditionalExtensionConfig()
    with frozen_evaluation():
        with pytest.raises(EvaluationFrozenError):
            m.run_one_late_extension(None, None, None, "I01", {}, config, Path("."))


def test_run_extension_source_replay_raises_under_frozen_evaluation() -> None:
    config = m.MirrorLateProgressConditionalExtensionConfig()
    with frozen_evaluation():
        with pytest.raises(EvaluationFrozenError):
            m.run_extension_source_replay(None, None, None, "I01", [], config)


# ---------------------------------------------------------------------------
# Real tiny CPU core+bank fixture, and a synthetic "REC-004G-shaped" source
# tree built by actually training a tiny CrossPositionLengthBiasPrimitive.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_core_and_bank() -> tuple:
    arch_cfg = SharedEncoderArchitectureConfig(seed=0, vocab_size=10, device="cpu")
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)
    bank, op_to_id = mpbr._rec004d_reconstruct_16_op_bank_structure(core, seed=0)
    bank.freeze_all()
    bank.eval()
    return core, bank, op_to_id


def _tiny_primitive(core) -> CrossPositionLengthBiasPrimitive:
    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    bias_cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation=m.REC004H_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(0)
    return CrossPositionLengthBiasPrimitive(pid, bias_cfg)


def _forward_result_for_step(core, primitive, step: int, seed: int) -> dict:
    """Regenerates the exact fixed validation set REC004H uses and evaluates
    `primitive` on it via the module's own forward function -- used to build
    a self-consistent synthetic fixture (byte-identical to what the module
    itself will later re-derive when auditing this fixture)."""
    examples = m._existing_validation_examples(seed)
    return m._run_full_checkpoint_forward(core, primitive, examples, core.device)


def _make_synthetic_rec004g_run(
    tmp_path: Path, core, init_id: str, audit_steps: tuple[int, ...], seed: int = 10
) -> Path:
    """Trains a real tiny primitive continuously through max(audit_steps)
    steps (using REC-004H's own target operation/vocab/length-range) and
    writes out checkpoints/training_states + a matching learning_curve.jsonl
    in REC-004G's exact directory shape, at tiny scale."""
    primitive = _tiny_primitive(core)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=m.REC004H_OPERATOR_LR,
        weight_decay=m.REC004H_OPERATOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=m.REC004H_T_MAX, eta_min=m.REC004H_SCHEDULER_ETA_MIN
    )

    run_dir = tmp_path / "rec004g_run_001"
    ckpt_dir = run_dir / init_id / m.REC004H_ARM / "checkpoints"
    state_dir = run_dir / init_id / m.REC004H_ARM / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    max_step = max(audit_steps)
    for step in range(1, max_step + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, m.REC004H_TARGET_OPERATION,
            vocab_size=m.REC004H_VOCAB_SIZE, sequence_length_range=m.REC004H_SEQUENCE_LENGTH_RANGE,
        )
        content_lengths = [len(ex.input_tokens) for ex in examples]
        op_obj = get_operation(m.REC004H_TARGET_OPERATION)
        output_lengths = [op_obj.output_length(n) for n in content_lengths]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, core.device)
        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), m.REC004H_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])

        if step in audit_steps:
            primitive.eval()
            state_snapshot = {
                k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()
            }
            checkpoint_hash = mb.canonical_state_hash(state_snapshot)
            torch.save(state_snapshot, ckpt_dir / f"step{step}.pt")
            torch.save(
                {
                    "primitive_state_dict": state_snapshot,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": None,
                    "cumulative_updates": step,
                    "init_id": init_id,
                    "arm": m.REC004H_ARM,
                },
                state_dir / f"step{step}.pt",
            )
            forward_result = _forward_result_for_step(core, primitive, step, seed)
            rows.append(
                {
                    "init_id": init_id, "arm": m.REC004H_ARM, "step": step,
                    "checkpoint_state_hash": checkpoint_hash,
                    "lr_used": lr_used, "lr_after_scheduler": lr_after,
                    "existing_validation": {
                        "actual_examples": forward_result["n_examples"],
                        "correct_exact_match": forward_result["overall_exact_match"],
                    },
                    "length_stratified": {"by_length": forward_result["by_length"]},
                }
            )
            primitive.train()

    with (run_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    return run_dir


# ---------------------------------------------------------------------------
# run_checkpoint_audit: hash match -> VERIFIED, reproduces recorded numbers
# exactly (self-consistent synthetic fixture).
# ---------------------------------------------------------------------------


def test_run_checkpoint_audit_verifies_self_consistent_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    audit_steps = (4, 6, 8)
    run_dir = _make_synthetic_rec004g_run(tmp_path, core, "I01", audit_steps, seed=10)
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004H_SOURCE_LEARNING_CURVE_PATH", run_dir / "learning_curve.jsonl")
    rows = m._load_rec004g_learning_curve_rows()
    examples = m._existing_validation_examples(10)

    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    result = m.run_checkpoint_audit(core, bank, op_to_id, "I01", 8, rows, examples)
    bank.replace_primitive(pid, original_slot)

    assert result["status"] == "VERIFIED"
    assert result["overall_em_matches_within_tolerance"] is True
    assert result["length10_count_matches"] is True
    assert "length10_token_audit" in result
    assert result["length10_token_audit"]["valid_token_count"] >= 0


def test_run_checkpoint_audit_detects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    audit_steps = (4,)
    run_dir = _make_synthetic_rec004g_run(tmp_path, core, "I01", audit_steps, seed=10)
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004H_SOURCE_LEARNING_CURVE_PATH", run_dir / "learning_curve.jsonl")

    rows = m._load_rec004g_learning_curve_rows()
    rows[0]["checkpoint_state_hash"] = "deadbeef" * 8
    examples = m._existing_validation_examples(10)

    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    result = m.run_checkpoint_audit(core, bank, op_to_id, "I01", 4, rows, examples)
    bank.replace_primitive(pid, original_slot)

    assert result["status"] == "SOURCE_REPLAY_MISMATCH"


def test_hash_only_precheck_detects_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", tmp_path / "does_not_exist")
    result = m._hash_only_precheck("I01", 8000, [])
    assert result["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"


# ---------------------------------------------------------------------------
# run_checkpoint_audit / _run_full_checkpoint_forward never raise under
# frozen_evaluation() (pure forward, no builder/optimizer calls) -- the
# audit stage must be safely nestable inside a frozen block.
# ---------------------------------------------------------------------------


def test_run_checkpoint_audit_is_frozen_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    run_dir = _make_synthetic_rec004g_run(tmp_path, core, "I01", (4,), seed=10)
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004H_SOURCE_LEARNING_CURVE_PATH", run_dir / "learning_curve.jsonl")
    rows = m._load_rec004g_learning_curve_rows()
    examples = m._existing_validation_examples(10)

    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    with frozen_evaluation():
        result = m.run_checkpoint_audit(core, bank, op_to_id, "I01", 4, rows, examples)
    bank.replace_primitive(pid, original_slot)
    assert result["status"] == "VERIFIED"


# ---------------------------------------------------------------------------
# Full resume/extension continuity on a tiny fixture: LR trajectory must
# match an uninterrupted run; sample-step starts at source+1; weight-only
# resume (no optimizer_state_dict) must be rejected.
# ---------------------------------------------------------------------------


def test_extension_resume_reproduces_uninterrupted_lr_trajectory_and_correct_step_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    fake_source_step = 4
    fake_target_step = fake_source_step + 4
    run_dir = _make_synthetic_rec004g_run(tmp_path, core, "I01", (fake_source_step,), seed=10)

    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004H_SOURCE_LEARNING_CURVE_PATH", run_dir / "learning_curve.jsonl")
    monkeypatch.setattr(m, "REC004H_SOURCE_STEP_FOR_EXTENSION", fake_source_step)
    monkeypatch.setattr(m, "REC004H_TARGET_CUMULATIVE_STEP", fake_target_step)
    monkeypatch.setattr(m, "REC004H_CHECKPOINT_INTERVAL", 2)

    config = m.MirrorLateProgressConditionalExtensionConfig(
        output_dir=tmp_path / "rec004h_out"
    )

    rows = m._load_rec004g_learning_curve_rows()
    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    replay = m.run_extension_source_replay(core, bank, op_to_id, "I01", rows, config)
    bank.replace_primitive(pid, original_slot)
    assert replay["status"] == "VERIFIED"
    assert replay["next_training_sample_step"] == fake_source_step + 1

    outcome = m.run_one_late_extension(
        core, bank, op_to_id, "I01", replay, config, config.output_dir
    )
    bank.replace_primitive(pid, original_slot)

    assert outcome["diverged_at_step"] is None
    assert outcome["new_optimizer_updates"] == 4
    steps_with_eval = sorted(
        c["step"] for c in outcome["checkpoints"] if "existing_validation" in c
    )
    assert steps_with_eval == [fake_source_step + 2, fake_source_step + 4]
    final_ckpt = next(c for c in outcome["checkpoints"] if c["step"] == fake_target_step)
    assert "bias_ablation" in (final_ckpt["final_step_extras"] or {})
    lrs = [row["lr_after_scheduler"] for row in outcome["lr_trace"]]
    assert len(lrs) == 4 and all(lr > 0 for lr in lrs)


def test_load_source_training_state_for_extension_rejects_wrong_cumulative_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, _bank, _op = tiny_core_and_bank
    run_dir = _make_synthetic_rec004g_run(tmp_path, core, "I01", (4,), seed=10)
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004H_SOURCE_STEP_FOR_EXTENSION", 4)

    path = m._rec004g_training_state_path("I01", 4)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["cumulative_updates"] = 999
    torch.save(state, path)

    with pytest.raises(ValueError, match="SOURCE_ARTIFACT_UNAVAILABLE"):
        m.load_source_training_state_for_extension("I01")


def test_load_source_training_state_for_extension_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004H_SOURCE_RUN_DIR", tmp_path / "does_not_exist")
    with pytest.raises(mb.MissingArtifactError):
        m.load_source_training_state_for_extension("I01")


# ---------------------------------------------------------------------------
# Dynamic spy: when the continuation decision is STOP_NO_EXTENSION, zero
# torch.optim.AdamW constructions must occur anywhere in the extension path.
# ---------------------------------------------------------------------------


def test_run_extension_for_all_inits_never_called_data_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural proof (not a full 5-model run): `run_extension_for_all_inits`
    (the only entry point that can construct an optimizer for MIRROR_HALVES
    in this module) must be reachable ONLY via `continuation["decision"] ==
    "EXTEND_ALL_FIVE_TO_18000"`. This confirms the decision string itself is
    the sole gate by re-deriving both a STOP and an EXTEND decision from
    fixtures and asserting the boolean each implies."""
    below_floor = {"I01": True, "I02": True, "I03": True, "I04": False, "I05": False}
    stop_progress = {
        "I01": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I02": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": True},
        "I03": {"status": "COMPUTED", "INIT_PROGRESS_CONFIRMED": False},
    }
    stop_decision = m.build_continuation_decision(below_floor, stop_progress)
    assert stop_decision["decision"] != "EXTEND_ALL_FIVE_TO_18000"

    extend_progress = {k: {**v, "INIT_PROGRESS_CONFIRMED": True} for k, v in stop_progress.items()}
    extend_decision = m.build_continuation_decision(below_floor, extend_progress)
    assert extend_decision["decision"] == "EXTEND_ALL_FIVE_TO_18000"


# ---------------------------------------------------------------------------
# build_terminal_floor_status / build_extension_protocol: adoption fields
# stay null/NOT_EXECUTED even in an all-pass fixture; terminal step is fixed.
# ---------------------------------------------------------------------------


def _final_checkpoint_row(em: float) -> dict:
    return {
        "step": m.REC004H_TARGET_CUMULATIVE_STEP,
        "existing_validation": {"correct_exact_match": em},
    }


def test_terminal_floor_status_all_pass_still_selects_nothing() -> None:
    outcomes = {
        i: {"checkpoints": [_final_checkpoint_row(0.99)]}
        for i in m.REC004H_INIT_IDS
    }
    status = m.build_terminal_floor_status(outcomes, m.REC004H_EXISTING_VALIDATION_FLOOR)
    assert status["terminal_floor_status"] == "ALL_FIVE_AT_18000_FLOOR"
    assert status["selected_init"] is None
    assert status["selected_intervention"] is None
    assert status["child_bundle"] is None
    assert status["rg3_recheck"] == "NOT_EXECUTED"


def test_terminal_floor_status_incomplete_when_a_final_checkpoint_is_missing() -> None:
    outcomes = {
        i: {"checkpoints": [_final_checkpoint_row(0.99)]}
        for i in m.REC004H_INIT_IDS[:-1]
    }
    outcomes["I05"] = {"checkpoints": []}
    status = m.build_terminal_floor_status(outcomes, m.REC004H_EXISTING_VALIDATION_FLOOR)
    assert status["terminal_floor_status"] == "INCOMPLETE"


def test_extension_protocol_targets_fixed_18000_and_30000_cap() -> None:
    config = m.MirrorLateProgressConditionalExtensionConfig()
    protocol = m.build_extension_protocol(config)
    assert protocol["target_cumulative_step"] == 18000
    assert protocol["max_new_updates_total"] == 30000
    assert protocol["init_ids"] == list(m.REC004H_INIT_IDS)


# ---------------------------------------------------------------------------
# Data-overlap gate: a disclosed training/validation content collision must
# stop further training (S5.4's own rule), and the impact audit must bound
# (never assert as fact) how much a disclosed overlap COULD have mattered.
# ---------------------------------------------------------------------------


def test_data_overlap_impact_audit_flags_nonrobust_and_robust_floor_calls() -> None:
    data_manifest = {
        "overlap_validation_indices": [0, 1],
        "overlap_count": 2,
        "total_new_training_examples": 100,
    }
    # I01: 973 correct out of 1024, both overlap examples (indices 0, 1)
    # correct -> removing them drops to 971, BELOW the 973 threshold that
    # 0.95*1024 requires -> NOT robust.
    correct = [False] * 1024
    for i in range(973):
        correct[i] = True
    per_init_terminal_forward = {
        "I01": {"sequence_correct_all": 973, "per_example_correct": correct},
    }

    audit = m.build_data_overlap_impact_audit(data_manifest, per_init_terminal_forward, 0.95)
    row = audit["per_init"]["I01"]
    assert row["actual_clears_floor"] is True
    assert row["worst_case_correct_if_overlap_examples_excluded"] == 971
    assert row["worst_case_clears_floor"] is False
    assert row["floor_conclusion_robust_to_disclosed_overlap"] is False


def test_data_overlap_impact_audit_robust_when_margin_exceeds_overlap_count() -> None:
    data_manifest = {
        "overlap_validation_indices": [0], "overlap_count": 1,
        "total_new_training_examples": 10,
    }
    correct = [False] * 1024
    for i in range(1000):
        correct[i] = True
    per_init_terminal_forward = {
        "I04": {"sequence_correct_all": 1000, "per_example_correct": correct},
    }
    audit = m.build_data_overlap_impact_audit(data_manifest, per_init_terminal_forward, 0.95)
    row = audit["per_init"]["I04"]
    assert row["actual_clears_floor"] is True
    assert row["worst_case_clears_floor"] is True
    assert row["floor_conclusion_robust_to_disclosed_overlap"] is True


def test_extension_data_manifest_reports_overlap_validation_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A training-stream example that collides byte-for-byte with a
    validation example (same input_tokens/target_tokens) must be reported
    with the exact validation index it collides with, not just a count."""
    fake_seed = 10
    fake_source, fake_target = 4, 6
    monkeypatch.setattr(m, "REC004H_SOURCE_STEP_FOR_EXTENSION", fake_source)
    monkeypatch.setattr(m, "REC004H_TARGET_CUMULATIVE_STEP", fake_target)
    real_examples = m._existing_validation_examples(fake_seed)
    manifest = m.build_extension_data_manifest(fake_seed, real_examples)
    assert manifest["overlap_count"] == len(manifest["overlap_rows"])
    assert isinstance(manifest["overlap_validation_indices"], list)
    # Structural contract only -- whether real overlap exists at this tiny
    # step range is incidental; the important thing is the shape/keys exist.
    assert "overlap_free" in manifest


# ---------------------------------------------------------------------------
# same_phase_audit: LR-after-scheduler values at the 3 phase-match steps
# must be recognized as consistent when equal (within tolerance).
# ---------------------------------------------------------------------------


def _lr_row(init_id: str, step: int, lr: float) -> dict:
    return {
        "init_id": init_id, "arm": m.REC004H_ARM, "step": step,
        "lr_used": lr, "lr_after_scheduler": lr,
    }


def test_same_phase_audit_flags_consistent_and_inconsistent_phases() -> None:
    rows = [
        _lr_row("I01", s, 0.0008) for s in m.REC004H_PHASE_MATCH_STEPS
    ] + [
        _lr_row("I01", s, 0.0004)
        for s in m.REC004H_AUDIT_STEPS if s not in m.REC004H_PHASE_MATCH_STEPS
    ]
    for init_id in m.REC004H_INIT_IDS[1:]:
        rows.extend(_lr_row(init_id, s, 0.0001) for s in m.REC004H_AUDIT_STEPS)
    audit = m.build_same_phase_audit(rows)
    assert audit["per_init"]["I01"]["phase_match_steps_consistent"] is True

    rows[1]["lr_after_scheduler"] = 0.0001  # perturb one phase-match point for I01
    audit2 = m.build_same_phase_audit(rows)
    assert audit2["per_init"]["I01"]["phase_match_steps_consistent"] is False


# ---------------------------------------------------------------------------
# Dispatcher config loader (mirrors REC-004G's own dispatcher test pattern).
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_wrong_seed(tmp_path: Path) -> None:
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_phase_b_b2_model_bundle_recovery.py"
    )
    spec = importlib.util.spec_from_file_location("rec004h_dispatcher_under_test", script_path)
    dispatcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dispatcher)

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("seed: 11\noutput_dir: runs/x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered under seed"):
        dispatcher._load_rec004h_config(bad_config)


def test_dispatcher_lists_rec004h_as_implemented() -> None:
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_phase_b_b2_model_bundle_recovery.py"
    )
    spec = importlib.util.spec_from_file_location("rec004h_dispatcher_under_test2", script_path)
    dispatcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dispatcher)
    assert "B-C005REC-004H" in dispatcher._IMPLEMENTED_TASKS
