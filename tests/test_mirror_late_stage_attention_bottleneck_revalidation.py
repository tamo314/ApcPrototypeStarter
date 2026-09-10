"""CPU-only contract tests for Task B-C005REC-004J (Late-Stage Attention
Bottleneck Revalidation (I03) & Collapse Contrast (I05)).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (reading REC-004H's real, multi-
gigabyte `run_001` tree for I03/I04/I05 at step=17500/18000) is never invoked
from this test module. These tests exercise the real production functions
either as pure-Python unit tests or against small, synthetic "REC-004H-
shaped" fixtures built in `tmp_path`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as m
from apc.evaluation import mirror_position_bias_repair as mpbr
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
# Constants sanity -- reused from REC-004F/REC-004I, nothing retyped.
# ---------------------------------------------------------------------------


def test_targets_are_the_fixed_five_checkpoints() -> None:
    assert m.REC004J_TARGETS == (
        ("I03", 17500), ("I03", 18000), ("I04", 18000), ("I05", 17500), ("I05", 18000),
    )


def test_all_targets_resolve_to_rec004h() -> None:
    for _init_id, step in m.REC004J_TARGETS:
        assert traj_audit._source_for_step(step) == "B-C005REC-004H"


def test_decisive_point_and_thresholds_are_preregistered() -> None:
    assert m.REC004J_DECISIVE_INIT == "I03"
    assert m.REC004J_DECISIVE_STEP == 17500
    assert (m.REC004J_DECISIVE_INIT, m.REC004J_DECISIVE_STEP) in m.REC004J_TARGETS
    assert m.REC004J_ORACLE_EM_THRESHOLD == 0.95
    assert m.REC004J_LARGE_DELTA_THRESHOLD == 0.30
    assert m.REC004J_TARGET_LENGTH == 10
    assert m.REC004J_PROBE_EXAMPLES == 512


def test_module_never_trains() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "AdamW" not in source
    assert "CosineAnnealingLR" not in source


def test_module_never_selects_or_starts_further_repair() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    blocked = ("rec005", "r3_011", "b_c006", "task_inference")
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


def test_module_only_runs_j0_and_o1_never_other_j_conditions() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ('"J1"', '"J2"', '"J3"', '"J4"', '"J5"', '"J6"'):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# length10_mechanism_probe_v1 generation: pure function, length-fixed,
# disjoint-by-construction, deterministic collision substitution.
# ---------------------------------------------------------------------------


def test_probe_v1_examples_are_always_length_10() -> None:
    examples, detail = m.build_length10_mechanism_probe_v1(10, set(), n=32)
    assert len(examples) == 32
    assert all(len(ex.input_tokens) == 10 for ex in examples)
    assert all(len(ex.target_tokens) == 10 for ex in examples)
    assert detail["target_length"] == 10
    assert detail["substitution_count"] == 0


def test_probe_v1_is_disclosed_as_development_exposed_not_sealed() -> None:
    _examples, detail = m.build_length10_mechanism_probe_v1(10, set(), n=4)
    assert detail["development_exposed"] is True
    assert detail["sealed_or_rg3_query"] is False


def test_probe_v1_substitutes_on_collision_deterministically() -> None:
    baseline, _ = m.build_length10_mechanism_probe_v1(10, set(), n=4)
    first_digest = traj_audit._digest_example(baseline[0])

    examples, detail = m.build_length10_mechanism_probe_v1(10, {first_digest}, n=4)
    assert detail["substitution_count"] == 1
    assert detail["substitutions"][0]["slot"] == 0
    assert traj_audit._digest_example(examples[0]) != first_digest
    assert len(examples) == 4

    examples2, detail2 = m.build_length10_mechanism_probe_v1(10, {first_digest}, n=4)
    assert [traj_audit._digest_example(e) for e in examples2] == [
        traj_audit._digest_example(e) for e in examples
    ]
    assert detail2 == detail


def test_probe_v1_never_collides_with_a_large_protected_set(monkeypatch) -> None:
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_LAST_STEP", 200)
    protected, _counts = traj_audit.build_protected_digest_registry(10)
    examples, _detail = m.build_length10_mechanism_probe_v1(10, protected, n=32)
    digests = traj_audit._digest_examples(examples)
    assert not (digests & protected)
    assert len(digests) == 32


def test_probe_v1_disjoint_from_clean_v2_itself(monkeypatch) -> None:
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    protected, _counts = traj_audit.build_protected_digest_registry(10)
    clean_v2_examples, _ = traj_audit.build_clean_selection_validation_v2(10, protected, n=64)
    clean_v2_digests = traj_audit._digest_examples(clean_v2_examples)
    protected_plus_clean_v2 = protected | clean_v2_digests

    probe_examples, _detail = m.build_length10_mechanism_probe_v1(
        10, protected_plus_clean_v2, n=32
    )
    probe_digests = traj_audit._digest_examples(probe_examples)
    assert not (probe_digests & clean_v2_digests)
    assert not (probe_digests & protected)


# ---------------------------------------------------------------------------
# regenerate_clean_selection_validation_v2: length-10 filter.
# ---------------------------------------------------------------------------


def test_regenerate_clean_v2_filters_to_length_10(monkeypatch) -> None:
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    all_1024, length_10_subset, detail, counts = m.regenerate_clean_selection_validation_v2(10)
    assert len(all_1024) == 1024
    assert all(len(ex.input_tokens) == 10 for ex in length_10_subset)
    assert len(length_10_subset) == sum(1 for ex in all_1024 if len(ex.input_tokens) == 10)
    assert detail["n"] == 1024
    assert "training_stream_steps_1_to_18000" in counts


def test_regenerate_clean_v2_matches_rec004i_manifest_at_real_scale() -> None:
    """At the REAL (non-monkeypatched) protected-set scale, this must
    reproduce REC-004I's own saved manifest numbers exactly -- a pure
    function of (seed, protected registry), never re-derived. This is the
    one slow test in this module (~5s, matching REC-004I's own
    admission)."""
    all_1024, length_10_subset, detail, _counts = m.regenerate_clean_selection_validation_v2(10)
    assert detail["n"] == 1024
    assert detail["substitution_count"] == 32
    assert detail["total_candidate_draws"] == 1056
    assert len(length_10_subset) == 230  # ADR-0104's own recorded length-10 subset size


# ---------------------------------------------------------------------------
# Checkpoint loading: missing / hash-mismatch / verified.
# ---------------------------------------------------------------------------


@pytest.fixture()
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


def _tiny_primitive(core, *, seed: int) -> CrossPositionLengthBiasPrimitive:
    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    bias_cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation=m.REC004J_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(seed)
    primitive = CrossPositionLengthBiasPrimitive(pid, bias_cfg)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def test_load_and_verify_primitive_missing_checkpoint(
    monkeypatch, tmp_path, tiny_core_and_bank
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    monkeypatch.setitem(
        traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004H", tmp_path / "rec004h_run_001"
    )
    primitive, info = m._load_and_verify_primitive(core, "I03", 17500)
    assert primitive is None
    assert info["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"


def test_load_and_verify_primitive_hash_mismatch(monkeypatch, tmp_path, tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    run_dir = tmp_path / "rec004h_run_001"
    monkeypatch.setitem(traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004H", run_dir)
    ckpt_dir = run_dir / "I03" / "P_LENGTH_POSITION_BIAS" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    primitive = _tiny_primitive(core, seed=1)
    torch.save(primitive.state_dict(), ckpt_dir / "step17500.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "learning_curve.jsonl").write_text(
        json.dumps(
            {
                "init_id": "I03", "arm": "P_LENGTH_POSITION_BIAS", "step": 17500,
                "checkpoint_state_hash": "deadbeef" * 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded, info = m._load_and_verify_primitive(core, "I03", 17500)
    assert loaded is not None  # still loads (never a silent no-op) but flags the mismatch
    assert info["status"] == "SOURCE_REPLAY_MISMATCH"


def test_load_and_verify_primitive_verified(monkeypatch, tmp_path, tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    run_dir = tmp_path / "rec004h_run_001"
    monkeypatch.setitem(traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004H", run_dir)
    ckpt_dir = run_dir / "I03" / "P_LENGTH_POSITION_BIAS" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    primitive = _tiny_primitive(core, seed=1)
    sd = primitive.state_dict()
    torch.save(sd, ckpt_dir / "step17500.pt")
    real_hash = mb.canonical_state_hash(sd)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "learning_curve.jsonl").write_text(
        json.dumps(
            {
                "init_id": "I03", "arm": "P_LENGTH_POSITION_BIAS", "step": 17500,
                "checkpoint_state_hash": real_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded, info = m._load_and_verify_primitive(core, "I03", 17500)
    assert loaded is not None
    assert info["status"] == "VERIFIED"
    assert info["checkpoint_state_hash"] == real_hash


# ---------------------------------------------------------------------------
# compute_diagnostic_point: structural correctness on a real (tiny,
# untrained) primitive -- never asserts a specific EM value, since the
# primitive is not trained; asserts shape/range invariants instead.
# ---------------------------------------------------------------------------


def _tiny_length10_examples(seed: int, n: int) -> list:
    import random

    from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
    from apc.environments.interpreter import run_program
    from apc.environments.task_spec import TaskSpec

    rng = random.Random(seed)
    op_obj = get_operation("MIRROR_HALVES")
    examples = []
    for _ in range(n):
        seq = tuple(rng.randrange(10) for _ in range(10))
        params = op_obj.sample_params(rng, seq, 10)
        prog = Program(steps=(ProgramStep(operation="MIRROR_HALVES", params=params),))
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


def test_compute_diagnostic_point_structural_invariants(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    examples = _tiny_length10_examples(seed=123, n=17)

    with torch.no_grad():
        point = m.compute_diagnostic_point(core, primitive, examples)

    assert point["n"] == 17
    assert 0.0 <= point["j0_sequence_exact_match"] <= 1.0
    assert 0.0 <= point["oracle_sequence_exact_match"] <= 1.0
    assert point["paired_delta_oracle_minus_j0"] == pytest.approx(
        point["oracle_sequence_exact_match"] - point["j0_sequence_exact_match"]
    )
    assert set(point["per_output_position"].keys()) == {str(k) for k in range(10)}
    for row in point["per_output_position"].values():
        assert 0.0 <= row["j0_accuracy"] <= 1.0
        assert 0.0 <= row["oracle_accuracy"] <= 1.0
    assert 1 <= point["correct_key_rank_margin"]["mean_rank"] <= 10
    assert point["attention_entropy"]["j0_head_averaged_mean"] >= 0.0
    assert len(point["attention_entropy"]["j0_per_head_mean"]) == primitive.n_head
    # Oracle attention is a one-hot substitution by construction -- its
    # entropy is EXACTLY 0, a sanity check on the construction, not a finding.
    assert point["attention_entropy"]["oracle_entropy"] == 0.0
    assert 0.0 <= point["head_agreement"]["argmax_agreement_rate"] <= 1.0
    assert point["head_agreement"]["mean_l1_divergence_from_head_mean"] >= 0.0
    assert (
        point["both_correct"] + point["both_wrong"] + point["j0_only_correct"]
        + point["oracle_only_correct"] == 17
    )


def test_compute_diagnostic_point_residual_after_oracle_matches_o1_wrong_count(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=3)
    examples = _tiny_length10_examples(seed=456, n=13)
    with torch.no_grad():
        point = m.compute_diagnostic_point(core, primitive, examples)
    expected_wrong = round((1.0 - point["oracle_sequence_exact_match"]) * 13)
    assert point["residual_after_oracle"]["wrong_example_count"] == expected_wrong
    assert sum(point["residual_after_oracle"]["by_position_wrong_count"].values()) >= 0


# ---------------------------------------------------------------------------
# build_i03_decision: pre-registered gate, both datasets, both conditions.
# ---------------------------------------------------------------------------


def _row(em: float, delta: float) -> dict:
    return {"oracle_sequence_exact_match": em, "paired_delta_oracle_minus_j0": delta}


def test_i03_decision_supported_when_both_datasets_clear_both_conditions() -> None:
    points = {
        "I03@17500:clean_v2_length10": _row(0.97, 0.78),
        "I03@17500:length10_mechanism_probe_v1": _row(0.96, 0.75),
    }
    decision = m.build_i03_decision(points)
    assert decision["label"] == "PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED"
    assert decision["clean_v2_passes"] is True
    assert decision["probe_passes"] is True


def test_i03_decision_mixed_when_only_one_dataset_passes() -> None:
    points = {
        "I03@17500:clean_v2_length10": _row(0.97, 0.78),
        "I03@17500:length10_mechanism_probe_v1": _row(0.40, 0.20),
    }
    decision = m.build_i03_decision(points)
    assert decision["label"] == "LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM"
    assert decision["clean_v2_passes"] is True
    assert decision["probe_passes"] is False


def test_i03_decision_mixed_when_em_high_but_delta_too_small() -> None:
    """High EM alone is not enough -- a checkpoint already near-ceiling
    under J0 could show a technically-high oracle EM with a tiny delta;
    the >=0.30 paired-delta bar rules that reading out."""
    points = {
        "I03@17500:clean_v2_length10": _row(0.97, 0.10),
        "I03@17500:length10_mechanism_probe_v1": _row(0.96, 0.08),
    }
    decision = m.build_i03_decision(points)
    assert decision["label"] == "LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM"


def test_i03_decision_missing_rows_default_to_mixed() -> None:
    decision = m.build_i03_decision({})
    assert decision["label"] == "LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM"
    assert decision["clean_v2_passes"] is False
    assert decision["probe_passes"] is False


# ---------------------------------------------------------------------------
# build_i05_collapse_contrast.
# ---------------------------------------------------------------------------


def _i05_points(j0_17500, oracle_17500, j0_18000, oracle_18000) -> dict:
    base = {
        "attention_entropy": {"j0_head_averaged_mean": 1.0},
        "head_agreement": {"argmax_agreement_rate": 0.5},
    }
    row_17500 = {
        "j0_sequence_exact_match": j0_17500, "oracle_sequence_exact_match": oracle_17500, **base,
    }
    row_18000 = {
        "j0_sequence_exact_match": j0_18000, "oracle_sequence_exact_match": oracle_18000, **base,
    }
    return {
        "I05@17500:clean_v2_length10": row_17500,
        "I05@18000:clean_v2_length10": row_18000,
        "I05@17500:length10_mechanism_probe_v1": row_17500,
        "I05@18000:length10_mechanism_probe_v1": row_18000,
    }


def test_i05_contrast_no_collapse() -> None:
    points = _i05_points(1.0, 0.95, 0.98, 0.94)
    contrast = m.build_i05_collapse_contrast(points)
    assert contrast["classification"] == "NO_COLLAPSE_OBSERVED_ON_THIS_SET"


def test_i05_contrast_collapse_persists_under_oracle_downstream_implicated() -> None:
    points = _i05_points(1.0, 0.90, 0.74, 0.60)
    contrast = m.build_i05_collapse_contrast(points)
    assert contrast["j0_collapses"] is True
    assert contrast["oracle_collapses"] is True
    expected = "COLLAPSE_PERSISTS_UNDER_ORACLE_ATTENTION_DOWNSTREAM_IMPLICATED"
    assert contrast["classification"] == expected


def test_i05_contrast_rescued_by_oracle_attention_implicated() -> None:
    points = _i05_points(1.0, 0.97, 0.70, 0.96)
    contrast = m.build_i05_collapse_contrast(points)
    assert contrast["j0_collapses"] is True
    assert contrast["oracle_collapses"] is False
    expected = "ORACLE_ATTENTION_RESCUES_COLLAPSE_ATTENTION_DISTRIBUTION_IMPLICATED"
    assert contrast["classification"] == expected


# ---------------------------------------------------------------------------
# Full orchestrator, tiny synthetic REC-004H-shaped tree.
# ---------------------------------------------------------------------------


def _write_tiny_checkpoint(run_dir: Path, core, init_id: str, step: int, *, seed: int) -> dict:
    primitive = _tiny_primitive(core, seed=seed)
    ckpt_dir = run_dir / init_id / "P_LENGTH_POSITION_BIAS" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sd = primitive.state_dict()
    torch.save(sd, ckpt_dir / f"step{step}.pt")
    return {
        "init_id": init_id, "arm": "P_LENGTH_POSITION_BIAS", "step": step,
        "checkpoint_state_hash": mb.canonical_state_hash(sd),
    }


def test_full_orchestrator_on_tiny_synthetic_rec004h_tree(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    run_dir = tmp_path / "rec004h_run_001"
    rows = [
        _write_tiny_checkpoint(run_dir, core, "I03", 17500, seed=1),
        _write_tiny_checkpoint(run_dir, core, "I03", 18000, seed=2),
        _write_tiny_checkpoint(run_dir, core, "I04", 18000, seed=3),
        _write_tiny_checkpoint(run_dir, core, "I05", 17500, seed=4),
        _write_tiny_checkpoint(run_dir, core, "I05", 18000, seed=5),
    ]
    with (run_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    monkeypatch.setitem(traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004H", run_dir)
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_FIRST_STEP", 1)
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    # `build_length10_mechanism_probe_v1`'s own `n` default is bound at
    # def-time (same caveat REC-004I's own tests note for
    # `build_clean_selection_validation_v2`), so it is NOT shrunk here; the
    # orchestrator's own call site uses the real default (512) -- cheap pure
    # -Python generation, and the forward passes below are on a tiny model.
    monkeypatch.setattr(
        ibc, "_reconstruct_parent_runtime", lambda config, parent_manifest: (core, bank, op_to_id)
    )
    monkeypatch.setattr(ibc, "_load_parent_manifest", lambda: (object(), {}))

    config = m.MirrorLateStageAttentionBottleneckRevalidationConfig(
        output_dir=tmp_path / "rec004j_run_001", seed=10,
    )
    summary = m.run_mirror_late_stage_attention_bottleneck_revalidation_task(config)

    assert summary["implementation_status"] == "COMPLETE"
    assert summary["checkpoint_source_replay_status"] == "VERIFIED"
    assert summary["new_optimizer_updates"] == 0
    assert summary["selected_init"] is None
    assert summary["selected_step"] is None
    assert summary["child_bundle"] is None
    assert summary["rg3_recheck"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["freeze_audit"]["core_unchanged"] is True
    assert summary["freeze_audit"]["protected_operations_unchanged"] is True
    assert summary["side_effect_audit"]["shared_cache_unchanged"] is True
    assert summary["i03_17500_decision"] in (
        "PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED",
        "LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM",
    )

    out = config.output_dir
    for name in (
        "clean_v2_source_replay.json", "length10_mechanism_probe_v1_manifest.json",
        "checkpoint_load_info.json", "diagnostic_points.jsonl",
        "diagnostic_points_summary.json", "i03_17500_decision.json",
        "i05_collapse_contrast.json", "freeze_audit.json", "side_effect_audit.json",
        "cost_accounting.json", "summary.json", "config.yaml", "system.json",
        "protocol.json",
    ):
        assert (out / name).is_file(), name

    points_lines = (out / "diagnostic_points.jsonl").read_text().splitlines()
    assert len(points_lines) == len(m.REC004J_TARGETS) * 2  # 5 checkpoints x 2 datasets
    for line in points_lines:
        row = json.loads(line)
        assert row["status"] == "VERIFIED"


def test_dispatcher_lists_rec004j_and_validates_seed(tmp_path) -> None:
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")
    assert "B-C005REC-004J" in dispatcher._IMPLEMENTED_TASKS

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("seed: 99\noutput_dir: runs/x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dispatcher._load_rec004j_config(bad_config)
