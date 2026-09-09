"""CPU-only contract tests for Task B-C005REC-004I (Contamination-Free
Checkpoint Trajectory Audit & Early-Stopping Feasibility).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (auditing every real saved P/I01-I05
checkpoint across REC-004D's/REC-004G's/REC-004H's real `run_001` trees) is
never invoked from this test module. These tests exercise the real
production functions either as pure-Python unit tests or against small,
synthetic "REC-004D/G/H-shaped" fixtures built in `tmp_path`.
"""

from __future__ import annotations

import ast
import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as m
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
# Constants sanity -- reused from REC-004G/REC-004D, nothing retyped.
# ---------------------------------------------------------------------------


def test_constants_are_reused_not_retyped() -> None:
    from apc.evaluation import mirror_budget_extension as mbe

    assert m.REC004I_INIT_IDS == mbe.REC004G_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert m.REC004I_ARM == mbe.REC004G_ARM
    assert m.REC004I_TARGET_OPERATION == "MIRROR_HALVES"
    assert m.REC004I_FLOOR == mbe.REC004G_EXISTING_VALIDATION_FLOOR == 0.95
    assert m.REC004I_PROTECTED_OPERATIONS == mbe.REC004G_PROTECTED_OPERATIONS
    assert m.REC004I_FIXED_CANDIDATE_OPERATIONS == mbe.REC004G_FIXED_CANDIDATE_OPERATIONS


def test_checkpoint_steps_cover_6000_to_18000_every_500() -> None:
    assert m.REC004I_CHECKPOINT_STEPS[0] == 6000
    assert m.REC004I_CHECKPOINT_STEPS[-1] == 18000
    assert len(m.REC004I_CHECKPOINT_STEPS) == 25
    assert all(b - a == 500 for a, b in pairwise(m.REC004I_CHECKPOINT_STEPS))


def test_target_lengths_are_6_through_10() -> None:
    assert m.REC004I_TARGET_LENGTHS == (6, 7, 8, 9, 10)


def test_max_prediction_budget_matches_5_inits_25_steps_1024_examples() -> None:
    assert m.REC004I_MAX_PREDICTION_EXAMPLES_BUDGET == 5 * 25 * 1024 == 128000


def test_source_for_step_resolves_all_three_directories() -> None:
    assert m._source_for_step(6000) == "B-C005REC-004D"
    assert m._source_for_step(6500) == "B-C005REC-004G"
    assert m._source_for_step(12000) == "B-C005REC-004G"
    assert m._source_for_step(12500) == "B-C005REC-004H"
    assert m._source_for_step(18000) == "B-C005REC-004H"
    with pytest.raises(ValueError):
        m._source_for_step(5999)
    with pytest.raises(ValueError):
        m._source_for_step(18001)


def test_module_never_imports_oracle_attention_or_dispatches_later_tasks() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = getattr(node, "module", None) or ""
            assert "oracle_attention" not in module
            assert not any("oracle_attention" in n.lower() for n in names)
    assert "_oracle_attention" not in source
    assert "oracle_attention_substitution" not in source
    blocked = ("rec005", "r3_011", "b_c006", "task_inference")
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


def test_module_never_trains_outside_of_nothing_zero_training_functions() -> None:
    """No function in this module calls `.backward()` or `optimizer.step()` --
    unlike REC-004G/H, this task has no training stage at all."""
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "AdamW" not in source
    assert "CosineAnnealingLR" not in source


# ---------------------------------------------------------------------------
# Digest helpers.
# ---------------------------------------------------------------------------


def test_digest_is_deterministic_and_order_sensitive() -> None:
    d1 = m._digest((1, 2, 3), (3, 2, 1))
    d2 = m._digest((1, 2, 3), (3, 2, 1))
    d3 = m._digest((3, 2, 1), (1, 2, 3))
    assert d1 == d2
    assert d1 != d3


# ---------------------------------------------------------------------------
# build_clean_selection_validation_v2: deterministic collision-avoidance.
# ---------------------------------------------------------------------------


def test_clean_v2_disjoint_from_empty_protected_set_has_zero_substitutions() -> None:
    examples, detail = m.build_clean_selection_validation_v2(10, set(), n=16)
    assert len(examples) == 16
    assert detail["substitution_count"] == 0
    assert detail["total_candidate_draws"] == 16


def test_clean_v2_substitutes_on_collision_deterministically() -> None:
    baseline, _ = m.build_clean_selection_validation_v2(10, set(), n=4)
    first_digest = m._digest_example(baseline[0])

    examples, detail = m.build_clean_selection_validation_v2(10, {first_digest}, n=4)
    assert detail["substitution_count"] == 1
    assert detail["substitutions"][0]["slot"] == 0
    # The substituted example must differ from the rejected (protected) one.
    assert m._digest_example(examples[0]) != first_digest
    assert len(examples) == 4
    # Consuming one extra draw for slot 0 shifts every later slot's draw from
    # the same continuing RNG stream -- later slots are NOT expected to match
    # the (no-collision) baseline; determinism is checked separately below.

    # Determinism: repeating with the same inputs gives the identical set.
    examples2, detail2 = m.build_clean_selection_validation_v2(10, {first_digest}, n=4)
    assert [m._digest_example(e) for e in examples2] == [m._digest_example(e) for e in examples]
    assert detail2 == detail


def test_clean_v2_never_collides_with_a_large_protected_set(monkeypatch) -> None:
    monkeypatch.setattr(m, "REC004I_TRAIN_STREAM_LAST_STEP", 200)
    protected, _ = m.build_protected_digest_registry(10)
    examples, _detail = m.build_clean_selection_validation_v2(10, protected, n=32)
    digests = m._digest_examples(examples)
    assert not (digests & protected)
    assert len(digests) == 32  # no within-set collisions in this draw


# ---------------------------------------------------------------------------
# build_protected_digest_registry: per-source counts, existing_validation
# subset check. Uses a tiny monkeypatched train-stream range for speed.
# ---------------------------------------------------------------------------


def test_protected_digest_registry_includes_existing_validation(monkeypatch) -> None:
    monkeypatch.setattr(m, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    protected, counts = m.build_protected_digest_registry(10)
    existing_val_digests = m._digest_examples(m._existing_validation_examples(10))
    assert existing_val_digests <= protected
    assert counts["rec004a_budget_validation"] == len(existing_val_digests)
    assert counts["training_stream_steps_1_to_18000"] > 0
    assert "rec004c_length_balanced_diagnostic" in counts
    assert "rec004c_content_counterfactual_diagnostic" in counts


# ---------------------------------------------------------------------------
# compute_trajectory_derived_stats: floor-crossing / duration / peak /
# regression semantics.
# ---------------------------------------------------------------------------


def test_derived_stats_never_crosses_floor() -> None:
    points = [(1, 0.5), (2, 0.6), (3, 0.7)]
    stats = m.compute_trajectory_derived_stats(points, floor=0.95)
    assert stats["ever_reaches_floor"] is False
    assert stats["first_step_at_or_above_floor"] is None
    assert stats["duration_at_or_above_floor_from_first_cross"] == 0
    assert stats["peak_step"] == 3
    assert stats["peak_value"] == pytest.approx(0.7)


def test_derived_stats_crosses_and_stays() -> None:
    points = [(1, 0.9), (2, 0.96), (3, 0.97), (4, 0.98)]
    stats = m.compute_trajectory_derived_stats(points, floor=0.95)
    assert stats["first_step_at_or_above_floor"] == 2
    assert stats["duration_at_or_above_floor_from_first_cross"] == 3
    assert stats["peak_step"] == 4
    assert stats["regression_from_peak_to_final"] == pytest.approx(0.0, abs=1e-9)


def test_derived_stats_crosses_then_regresses_without_recovery() -> None:
    points = [(1, 0.96), (2, 0.97), (3, 0.80), (4, 0.75)]
    stats = m.compute_trajectory_derived_stats(points, floor=0.95)
    assert stats["first_step_at_or_above_floor"] == 1
    assert stats["duration_at_or_above_floor_from_first_cross"] == 2  # steps 1,2 only
    assert stats["peak_step"] == 2
    assert stats["final_step"] == 4
    assert stats["regression_from_peak_to_final"] == pytest.approx(0.97 - 0.75)
    assert stats["recovered_to_peak_after_regressing"] is False


def test_derived_stats_regresses_then_recovers_to_peak() -> None:
    points = [(1, 0.96), (2, 0.80), (3, 0.961)]
    stats = m.compute_trajectory_derived_stats(points, floor=0.95, eps=1e-6)
    assert stats["peak_step"] == 3
    assert stats["peak_value"] == pytest.approx(0.961)
    # Peak is the final point here, so "recovered after regressing" from the
    # step-1 local peak is moot; re-derive with a clean peak-then-fall-then-
    # recover-to-that-earlier-peak shape instead.
    points2 = [(1, 0.97), (2, 0.80), (3, 0.9695)]
    stats2 = m.compute_trajectory_derived_stats(points2, floor=0.95, eps=1e-3)
    assert stats2["peak_step"] == 1
    assert stats2["peak_value"] == pytest.approx(0.97)
    assert stats2["recovered_to_peak_after_regressing"] is True


def test_derived_stats_evidence_insufficient_when_all_none() -> None:
    stats = m.compute_trajectory_derived_stats([(1, None), (2, None)])
    assert stats["status"] == "EVIDENCE_INSUFFICIENT"


# ---------------------------------------------------------------------------
# classify_pattern: the task's own three-way outcome classification.
# ---------------------------------------------------------------------------


def test_classify_all_five_clear() -> None:
    all_inits = ("I01", "I02", "I03", "I04", "I05")
    pattern = m.classify_pattern(set(all_inits), set(all_inits), all_inits)
    assert pattern["classification"] == "ALL_FIVE_CLEAR_CLEAN_V2"
    assert pattern["never_clear_clean_v2_inits"] == []


def test_classify_some_inits_never_clear() -> None:
    all_inits = ("I01", "I02", "I03", "I04", "I05")
    old_pass = {"I01", "I02", "I04", "I05"}
    clean_pass = {"I01", "I02", "I04", "I05"}  # I03 never clears either set
    pattern = m.classify_pattern(old_pass, clean_pass, all_inits)
    assert pattern["classification"] == "SOME_INITS_NEVER_CLEAR_CLEAN_V2"
    assert pattern["never_clear_clean_v2_inits"] == ["I03"]
    assert pattern["invalidated_inits"] == []


def test_classify_old_passes_largely_invalidated() -> None:
    all_inits = ("I01", "I02", "I03", "I04", "I05")
    old_pass = {"I01", "I02", "I04"}
    clean_pass = {"I04"}  # I01, I02 lose their old-validation pass on clean-v2
    pattern = m.classify_pattern(old_pass, clean_pass, all_inits)
    assert pattern["classification"] == "OLD_PASSES_LARGELY_INVALIDATED"
    assert pattern["invalidated_inits"] == ["I01", "I02"]
    assert pattern["invalidation_fraction_of_old_passes"] == pytest.approx(2 / 3)


def test_proposed_next_step_text_differs_per_classification() -> None:
    texts = {
        m._proposed_next_step_text(c)
        for c in (
            "ALL_FIVE_CLEAR_CLEAN_V2",
            "SOME_INITS_NEVER_CLEAR_CLEAN_V2",
            "OLD_PASSES_LARGELY_INVALIDATED",
        )
    }
    assert len(texts) == 3  # all distinct, none empty
    assert all(texts)


# ---------------------------------------------------------------------------
# _old_dev_metric_row: key-normalization across REC-004D/G's
# "length_stratified.by_length" vs REC-004H's top-level "by_length".
# ---------------------------------------------------------------------------


def test_old_dev_metric_row_handles_both_by_length_key_shapes() -> None:
    rows_d_shaped = {
        "B-C005REC-004D": [
            {
                "init_id": "I01", "arm": m.REC004I_ARM, "step": 6000,
                "existing_validation": {"correct_exact_match": 0.5},
                "length_stratified": {"by_length": {"6": {"n": 1, "exact": 1}}},
                "checkpoint_state_hash": "abc",
            }
        ],
        "B-C005REC-004G": [],
        "B-C005REC-004H": [
            {
                "init_id": "I01", "arm": m.REC004I_ARM, "step": 12500,
                "existing_validation": {"correct_exact_match": 0.9},
                "by_length": {"6": {"n": 1, "exact": 1}},
                "checkpoint_state_hash": "def",
            }
        ],
    }
    row_d = m._old_dev_metric_row("I01", 6000, rows_d_shaped)
    assert row_d is not None
    assert row_d["existing_validation_em"] == 0.5
    assert row_d["by_length"] == {"6": {"n": 1, "exact": 1}}

    row_h = m._old_dev_metric_row("I01", 12500, rows_d_shaped)
    assert row_h is not None
    assert row_h["existing_validation_em"] == 0.9
    assert row_h["by_length"] == {"6": {"n": 1, "exact": 1}}

    assert m._old_dev_metric_row("I01", 6500, rows_d_shaped) is None  # missing row


# ---------------------------------------------------------------------------
# frozen_evaluation() guard: reconstructing runtime must be forbidden inside
# a frozen block (mirrors REC-004H's own guard tests).
# ---------------------------------------------------------------------------


def test_reconstruct_parent_runtime_is_guarded_inside_frozen_evaluation() -> None:
    with frozen_evaluation(), pytest.raises(EvaluationFrozenError):
        ibc._reconstruct_parent_runtime(
            ibc.IncrementalBudgetCalibrationConfig(seed=10), object()  # never reached
        )


# ---------------------------------------------------------------------------
# Real-fixture integration: tiny 3-source synthetic tree, full orchestrator.
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
        operation=m.REC004I_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(0)
    return CrossPositionLengthBiasPrimitive(pid, bias_cfg)


def _tiny_forward_em(core, primitive, examples) -> dict:
    return m._run_full_checkpoint_forward_all_lengths(core, primitive, examples, core.device)


def _build_tiny_three_source_tree(
    tmp_path: Path, core, init_id: str, checkpoint_steps: tuple[int, ...], seed: int = 10
) -> dict[str, Path]:
    """Trains one real tiny primitive continuously through
    `max(checkpoint_steps)` tiny steps and writes checkpoints/learning_curve
    rows at exactly `checkpoint_steps`, split across 3 directories the same
    way `_source_for_step` (boundaries 6000/6500/12000/12500/18000) resolves
    real steps -- but at this fixture's own tiny step numbers via a
    test-local resolver the test monkeypatches in."""
    primitive = _tiny_primitive(core)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(primitive.parameters(), lr=0.0008, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

    dirs = {
        "SRC_D": tmp_path / "rec004d_run_001",
        "SRC_G": tmp_path / "rec004g_run_001",
        "SRC_H": tmp_path / "rec004h_run_001",
    }
    boundary_d, boundary_g = checkpoint_steps[0], checkpoint_steps[2]  # 1st, 3rd points

    def source_for(step: int) -> str:
        if step <= boundary_d:
            return "SRC_D"
        if step <= boundary_g:
            return "SRC_G"
        return "SRC_H"

    rows_by_source: dict[str, list[dict]] = {"SRC_D": [], "SRC_G": [], "SRC_H": []}
    clean_examples_seed_examples = m._existing_validation_examples(seed)
    max_step = max(checkpoint_steps)

    for step in range(1, max_step + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, m.REC004I_TARGET_OPERATION, vocab_size=10, sequence_length_range=(6, 10),
        )
        content_lengths = [len(ex.input_tokens) for ex in examples]
        op_obj = get_operation(m.REC004I_TARGET_OPERATION)
        output_lengths = [op_obj.output_length(n) for n in content_lengths]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, core.device)
        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step in checkpoint_steps:
            source = source_for(step)
            run_dir = dirs[source]
            ckpt_dir = run_dir / init_id / m.REC004I_ARM / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            primitive.eval()
            state_snapshot = {
                k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()
            }
            checkpoint_hash = mb.canonical_state_hash(state_snapshot)
            torch.save(state_snapshot, ckpt_dir / f"step{step}.pt")
            forward_result = _tiny_forward_em(core, primitive, clean_examples_seed_examples)
            rows_by_source[source].append(
                {
                    "init_id": init_id, "arm": m.REC004I_ARM, "step": step,
                    "checkpoint_state_hash": checkpoint_hash,
                    "existing_validation": {
                        "correct_exact_match": forward_result["overall_exact_match"]
                    },
                    "by_length": forward_result["by_length"],
                }
            )
            primitive.train()

    for source, run_dir in dirs.items():
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows_by_source[source]:
                fh.write(json.dumps(row) + "\n")

    return dirs


def test_full_orchestrator_on_tiny_synthetic_three_source_tree(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    tiny_steps = (20, 40, 60, 80, 100)  # 1st/3rd points define D/G boundary per helper above

    dirs = _build_tiny_three_source_tree(tmp_path, core, "I01", tiny_steps, seed=10)

    def fake_source_for_step(step: int) -> str:
        if step <= tiny_steps[0]:
            return "SRC_D"
        if step <= tiny_steps[2]:
            return "SRC_G"
        return "SRC_H"

    monkeypatch.setattr(m, "REC004I_INIT_IDS", ("I01",))
    monkeypatch.setattr(m, "REC004I_CHECKPOINT_STEPS", tiny_steps)
    monkeypatch.setattr(m, "REC004I_SOURCE_RUN_DIRS", dirs)
    monkeypatch.setattr(m, "_source_for_step", fake_source_for_step)
    # NOTE: build_clean_selection_validation_v2's `n` default is bound at
    # def-time, so it is NOT affected by monkeypatching REC004I_CLEAN_V2_EXAMPLES
    # here; the orchestrator's own call site uses the real default (1024).
    monkeypatch.setattr(m, "REC004I_TRAIN_STREAM_FIRST_STEP", 1)
    monkeypatch.setattr(m, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    budget = 1 * len(tiny_steps) * m.REC004I_CLEAN_V2_EXAMPLES
    monkeypatch.setattr(m, "REC004I_MAX_PREDICTION_EXAMPLES_BUDGET", budget)
    monkeypatch.setattr(m, "REC004I_CONTRACT_FILE", Path(m.__file__))  # any existing file
    monkeypatch.setattr(
        ibc, "_reconstruct_parent_runtime", lambda config, parent_manifest: (core, bank, op_to_id)
    )
    monkeypatch.setattr(ibc, "_load_parent_manifest", lambda: (object(), {}))

    config = m.MirrorContaminationFreeCheckpointTrajectoryAuditConfig(
        output_dir=tmp_path / "rec004i_run_001", seed=10,
    )
    summary = m.run_mirror_contamination_free_checkpoint_trajectory_audit_task(config)

    assert summary["implementation_status"] == "COMPLETE"
    assert summary["new_optimizer_updates"] == 0
    assert summary["checkpoints_evaluated"] == 1 * len(tiny_steps)
    assert summary["source_replay_status"] == "VERIFIED"
    assert summary["selected_init"] is None
    assert summary["selected_step"] is None
    assert summary["child_bundle"] is None
    assert summary["rg3_recheck"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["freeze_audit_passed"] is True
    assert summary["side_effect_audit_passed"] is True

    out = config.output_dir
    for name in (
        "clean_selection_validation_v2_manifest.json", "checkpoint_trajectory.jsonl",
        "trajectory_decomposition.json", "old_vs_clean_v2_comparison.json",
        "pattern_classification.json", "freeze_audit.json", "side_effect_audit.json",
        "cost_accounting.json", "summary.json", "report.md", "next_step_recommendation.md",
    ):
        assert (out / name).is_file(), name

    manifest = json.loads((out / "clean_selection_validation_v2_manifest.json").read_text())
    assert manifest["n"] == m.REC004I_CLEAN_V2_EXAMPLES
    assert manifest["disjoint_from_protected_set_confirmed"] is True

    trajectory_lines = (out / "checkpoint_trajectory.jsonl").read_text().splitlines()
    assert len(trajectory_lines) == len(tiny_steps)
    for line in trajectory_lines:
        row = json.loads(line)
        assert row["status"] == "VERIFIED"
        assert row["clean_v2"]["n_examples"] == m.REC004I_CLEAN_V2_EXAMPLES


def test_run_checkpoint_trajectory_point_detects_missing_checkpoint(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    monkeypatch.setattr(
        m, "REC004I_SOURCE_RUN_DIRS",
        {
            "B-C005REC-004D": tmp_path / "d", "B-C005REC-004G": tmp_path / "g",
            "B-C005REC-004H": tmp_path / "h",
        },
    )
    result = m.run_checkpoint_trajectory_point(core, bank, op_to_id, "I01", 6000, {}, [])
    assert result["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"


def test_run_checkpoint_trajectory_point_detects_hash_mismatch(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core)
    state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    run_dir = tmp_path / "d"
    ckpt_dir = run_dir / "I01" / m.REC004I_ARM / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save(state, ckpt_dir / "step6000.pt")
    row = {
        "init_id": "I01", "arm": m.REC004I_ARM, "step": 6000,
        "existing_validation": {"correct_exact_match": 0.5},
        "by_length": {}, "checkpoint_state_hash": "not-the-real-hash",
    }
    (run_dir / "learning_curve.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        m, "REC004I_SOURCE_RUN_DIRS",
        {
            "B-C005REC-004D": run_dir, "B-C005REC-004G": tmp_path / "g",
            "B-C005REC-004H": tmp_path / "h",
        },
    )
    examples = m._existing_validation_examples(10)[:8]
    source_rows = {"B-C005REC-004D": [row], "B-C005REC-004G": [], "B-C005REC-004H": []}
    result = m.run_checkpoint_trajectory_point(
        core, bank, op_to_id, "I01", 6000, source_rows, examples
    )
    assert result["status"] == "SOURCE_REPLAY_MISMATCH"
    assert result["clean_v2"]["n_examples"] == 8  # forward pass still runs and is reported


# ---------------------------------------------------------------------------
# Dispatcher sanity.
# ---------------------------------------------------------------------------


def test_dispatcher_lists_rec004i_and_validates_seed(tmp_path) -> None:
    import importlib.util
    import sys as _sys

    script_path = Path("scripts/run_phase_b_b2_model_bundle_recovery.py")
    spec = importlib.util.spec_from_file_location("rec004i_dispatcher_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        assert "B-C005REC-004I" in module._IMPLEMENTED_TASKS
        wrong_seed_config = tmp_path / "wrong_seed.yaml"
        wrong_seed_config.write_text("seed: 999\noutput_dir: runs/tmp\n", encoding="utf-8")
        with pytest.raises(ValueError):
            module._load_rec004i_config(wrong_seed_config)
    finally:
        del _sys.modules[spec.name]
