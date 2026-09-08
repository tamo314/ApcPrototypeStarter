"""CPU-only contract tests for Task B-C005REC-004A (Incremental Primitive
Budget Calibration & RG3 Recheck).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (6000 optimizer steps x 4 operations
against the real REC-004 parent bundle) is never invoked from this test
module. These tests exercise the real production functions -- including a
genuine (tiny-ladder) end-to-end CPU training call -- against small fixtures,
never against `runs/phase_b_b2_model_bundle_recovery/bundles/<real parent>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.utils import model_bundle as mb

# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


def test_target_operations_and_physical_ids_are_the_four_rec004_floor_failures() -> None:
    assert set(ibc.REC004A_TARGET_OPERATIONS) == {
        "CYCLE_FOUR",
        "MIRROR_HALVES",
        "ROTATE_TRIPLETS",
        "SWAP_ENDS",
    }
    assert set(ibc.REC004A_PHYSICAL_IDS) == set(ibc.REC004A_TARGET_OPERATIONS)
    assert len(set(ibc.REC004A_PHYSICAL_IDS.values())) == 4


def test_protected_operations_are_the_twelve_rec004_floor_passes_plus_shift() -> None:
    assert len(ibc.REC004A_PROTECTED_NON_SHIFT_OPERATIONS) == 11
    assert ibc.REC004A_PROTECTED_OPERATIONS == (
        *ibc.REC004A_PROTECTED_NON_SHIFT_OPERATIONS,
        "SHIFT",
    )
    assert set(ibc.REC004A_PROTECTED_OPERATIONS).isdisjoint(set(ibc.REC004A_TARGET_OPERATIONS))


def test_step_ladder_caps_at_6000_and_is_increasing() -> None:
    assert ibc.REC004A_STEP_LADDER == (1000, 2000, 4000, 6000)
    assert max(ibc.REC004A_STEP_LADDER) == 6000
    assert list(ibc.REC004A_STEP_LADDER) == sorted(ibc.REC004A_STEP_LADDER)


# ---------------------------------------------------------------------------
# _check_resume_state_available -- real filesystem check (A3).
# ---------------------------------------------------------------------------


def test_check_resume_state_available_empty_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = ibc._check_resume_state_available(seed=999999)
    assert result["namespace_exists"] is False
    assert result["candidate_files_found"] == []


def test_check_resume_state_available_detects_optimizer_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ns = tmp_path / "runs" / "phase_b_b2_model_bundle_recovery" / "staging" / "seed_10" / "rec004"
    ns.mkdir(parents=True)
    (ns / "optimizer_state.pt").write_bytes(b"fake")
    (ns / "primitive_bank_16.pt").write_bytes(b"fake")
    result = ibc._check_resume_state_available(seed=10)
    assert result["namespace_exists"] is True
    assert result["candidate_files_found"] == ["optimizer_state.pt"]


# ---------------------------------------------------------------------------
# select_candidates -- pure selection logic (D1).
# ---------------------------------------------------------------------------


def _fake_outcome(steps_and_em: dict[int, float]) -> dict:
    return {
        "diverged_at_step": None,
        "checkpoints": [
            {"step": step, "budget_validation": {"correct_exact_match": em}}
            for step, em in steps_and_em.items()
        ],
    }


def test_select_candidates_picks_first_step_at_or_above_floor() -> None:
    results = {
        "CYCLE_FOUR": _fake_outcome({1000: 0.80, 2000: 0.90, 4000: 0.97, 6000: 0.99}),
    }
    selection, all_selected = ibc.select_candidates(results, floor=0.95)
    assert selection["CYCLE_FOUR"]["selected_step"] == 4000
    assert all_selected is True


def test_select_candidates_null_when_no_step_clears_floor() -> None:
    results = {
        "MIRROR_HALVES": _fake_outcome({1000: 0.01, 2000: 0.02, 4000: 0.03, 6000: 0.05}),
    }
    selection, all_selected = ibc.select_candidates(results, floor=0.95)
    assert selection["MIRROR_HALVES"]["selected_step"] is None
    assert all_selected is False


def test_select_candidates_requires_all_operations_to_pass() -> None:
    results = {
        "CYCLE_FOUR": _fake_outcome({1000: 0.99}),
        "MIRROR_HALVES": _fake_outcome({1000: 0.01}),
    }
    selection, all_selected = ibc.select_candidates(results, floor=0.95)
    assert selection["CYCLE_FOUR"]["selected_step"] == 1000
    assert selection["MIRROR_HALVES"]["selected_step"] is None
    assert all_selected is False


def test_select_candidates_ignores_non_monotonic_later_drops() -> None:
    # A later checkpoint dropping back below floor must not un-select an
    # earlier passing checkpoint, and the earliest passing step still wins.
    results = {
        "ROTATE_TRIPLETS": _fake_outcome({1000: 0.30, 2000: 0.96, 4000: 0.80, 6000: 0.97}),
    }
    selection, all_selected = ibc.select_candidates(results, floor=0.95)
    assert selection["ROTATE_TRIPLETS"]["selected_step"] == 2000
    assert selection["ROTATE_TRIPLETS"]["all_checkpoint_validation_em"][4000] == 0.80
    assert all_selected is True


# ---------------------------------------------------------------------------
# build_budget_protocol -- SCHEDULE_EXTENSION_REQUIRED guard (B3).
# ---------------------------------------------------------------------------


def _stub_stage_a() -> dict:
    class _FakeCore:
        canonical_state_hash = "core-hash"

    class _FakeManifest:
        bundle_id = "fake-bundle"
        core = _FakeCore()

    return {
        "parent_manifest": _FakeManifest(),
        "parent_audit": {"protected_operation_physical_ids": {}},
    }


def test_build_budget_protocol_detects_lr_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DriftedConfig:
        operator_lr = 0.999
        operator_weight_decay = ibc.REC004A_OPERATOR_WEIGHT_DECAY
        operator_grad_clip = ibc.REC004A_OPERATOR_GRAD_CLIP

    monkeypatch.setattr(ibc, "UnifiedBenchmarkConfig", lambda: _DriftedConfig())
    config = ibc.IncrementalBudgetCalibrationConfig(output_dir=tmp_path)
    with pytest.raises(ValueError, match="SCHEDULE_EXTENSION_REQUIRED"):
        ibc.build_budget_protocol(config, _stub_stage_a())


def test_build_budget_protocol_fixes_t_max_at_1000_regardless_of_ladder(tmp_path: Path) -> None:
    config = ibc.IncrementalBudgetCalibrationConfig(
        output_dir=tmp_path, step_ladder=(2, 4, 6, 8)
    )
    protocol = ibc.build_budget_protocol(config, _stub_stage_a())
    assert protocol["scheduler_t_max"] == 1000
    assert protocol["step_ladder"] == [2, 4, 6, 8]
    assert "protocol_hash" in protocol
    on_disk = json.loads((tmp_path / "budget_protocol.json").read_text(encoding="utf-8"))
    assert on_disk["protocol_hash"] == protocol["protocol_hash"]


# ---------------------------------------------------------------------------
# _dedup_audit -- real (unmocked) generator calls confirm no leakage across
# train/budget_validation/recheck_query for a real target operation (B4).
# ---------------------------------------------------------------------------


def test_dedup_audit_reports_zero_overlap_for_real_generators(tmp_path: Path) -> None:
    config = ibc.IncrementalBudgetCalibrationConfig(
        output_dir=tmp_path, validation_examples=64, recheck_query_examples=64
    )
    stage_c_results = {"SWAP_ENDS": _fake_outcome({1000: 0.5})}
    report = ibc._dedup_audit(stage_c_results, config)
    assert report["SWAP_ENDS"]["train_fit_vs_budget_validation_overlap"] == 0
    assert report["SWAP_ENDS"]["train_fit_vs_recheck_query_overlap"] == 0
    assert report["SWAP_ENDS"]["budget_validation_vs_recheck_query_overlap"] == 0


# ---------------------------------------------------------------------------
# _generate_step_training_examples -- exact reproduction of
# _train_single_primitive's per-step generation (same seed formula).
# ---------------------------------------------------------------------------


def test_generate_step_training_examples_is_deterministic_and_matches_seed_formula() -> None:
    from apc.evaluation.unified_oracle_causal_benchmark import _derive_local_seed

    a = ibc._generate_step_training_examples(
        seed=10, step=1, operation="CYCLE_FOUR", vocab_size=10, sequence_length_range=(6, 10)
    )
    b = ibc._generate_step_training_examples(
        seed=10, step=1, operation="CYCLE_FOUR", vocab_size=10, sequence_length_range=(6, 10)
    )
    assert [e.input_tokens for e in a] == [e.input_tokens for e in b]
    assert [e.target_tokens for e in a] == [e.target_tokens for e in b]
    assert len(a) == ibc.REC004A_EXAMPLES_PER_STEP
    # The seed derivation matches the exact label _train_single_primitive uses
    # ("train:{operation}"), so a future drift in that label would be caught
    # by a mismatch here against the same helper's own derivation.
    assert _derive_local_seed(10, 1, "train:CYCLE_FOUR") == _derive_local_seed(
        10, 1, f"{ibc.REC004A_TRAIN_SPLIT_LABEL}:CYCLE_FOUR"
    )


# ---------------------------------------------------------------------------
# Real (tiny-ladder) end-to-end CPU training: Core/other-primitives frozen,
# only the target primitive's parameters reach the optimizer, checkpoints
# land only on ladder steps and never exceed the ladder's cap.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_core_and_bank() -> tuple:
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


def test_train_operation_with_checkpoints_touches_only_the_target_primitive(
    tiny_core_and_bank: tuple,
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    operation = "CYCLE_FOUR"
    pid = ibc.REC004A_PHYSICAL_IDS[operation]

    core_state_before = {k: v.clone() for k, v in core.model.state_dict().items()}
    other_slices_before = {
        op: {k: v.clone() for k, v in bank.get(op_to_id[op]).state_dict().items()}
        for op in op_to_id
        if op != operation
    }
    target_slice_before = {k: v.clone() for k, v in bank.get(pid).state_dict().items()}

    config = ibc.IncrementalBudgetCalibrationConfig(
        output_dir=Path("unused"), step_ladder=(2, 4), validation_examples=8
    )
    outcome = ibc._train_operation_with_checkpoints(core, bank, op_to_id, operation, config=config)

    # Core is untouched.
    for k, v in core.model.state_dict().items():
        assert torch.equal(v, core_state_before[k]), f"Core parameter {k} changed"

    # The 15 non-target primitives are untouched.
    for op, before in other_slices_before.items():
        after = bank.get(op_to_id[op]).state_dict()
        for k, v in after.items():
            assert torch.equal(v, before[k]), f"{op}.{k} changed but was not the training target"

    # The target primitive itself DID change (training actually happened),
    # and only ladder steps produced checkpoints, never beyond the ladder.
    target_after = bank.get(pid).state_dict()
    changed = any(
        not torch.equal(v, target_slice_before[k]) for k, v in target_after.items()
    )
    assert changed, "target primitive weights did not change after training"

    steps = [c["step"] for c in outcome["checkpoints"] if "budget_validation" in c]
    assert steps == [2, 4]
    assert max(steps) <= max(config.step_ladder) == 6 or max(steps) <= 4
    assert set(outcome["checkpoint_state_dicts"].keys()) == {2, 4}


def test_train_operation_optimizer_receives_only_target_primitive_parameters(
    tiny_core_and_bank: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    operation = "SWAP_ENDS"
    pid = ibc.REC004A_PHYSICAL_IDS[operation]
    expected_param_count = sum(p.numel() for p in bank.get(pid).parameters())

    captured: dict[str, int] = {}
    original_adamw = torch.optim.AdamW

    def _spy_adamw(params, **kwargs):  # type: ignore[no-untyped-def]
        params = list(params)
        captured["n_params"] = sum(p.numel() for p in params)
        return original_adamw(params, **kwargs)

    monkeypatch.setattr(ibc.torch.optim, "AdamW", _spy_adamw)
    config = ibc.IncrementalBudgetCalibrationConfig(
        output_dir=Path("unused"), step_ladder=(1,), validation_examples=4
    )
    ibc._train_operation_with_checkpoints(core, bank, op_to_id, operation, config=config)

    assert captured["n_params"] == expected_param_count


def test_train_operation_records_nan_divergence_as_not_executed(
    tiny_core_and_bank: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    operation = "ROTATE_TRIPLETS"

    original_cross_entropy = ibc.F.cross_entropy

    def _nan_cross_entropy(*args, **kwargs):  # type: ignore[no-untyped-def]
        return original_cross_entropy(*args, **kwargs) * float("nan")

    monkeypatch.setattr(ibc.F, "cross_entropy", _nan_cross_entropy)
    config = ibc.IncrementalBudgetCalibrationConfig(
        output_dir=Path("unused"), step_ladder=(1, 2), validation_examples=4
    )
    outcome = ibc._train_operation_with_checkpoints(core, bank, op_to_id, operation, config=config)
    assert outcome["diverged_at_step"] == 1
    statuses = {c["step"]: c.get("status") for c in outcome["checkpoints"]}
    assert statuses[1] == "NOT_EXECUTED"
    assert statuses[2] == "NOT_EXECUTED"
    assert outcome["checkpoint_state_dicts"] == {}


# ---------------------------------------------------------------------------
# Manifest round trip (mirrors model_bundle_recovery's own pattern; reused
# here because REC-004A's child manifest goes through the same JSON shape).
# ---------------------------------------------------------------------------


def _tiny_manifest(tmp_path: Path, *, suffix: str = "") -> mb.ModelBundleManifest:
    core_path = tmp_path / f"core{suffix}.pt"
    vocab_path = tmp_path / f"vocab{suffix}.pt"
    bank_path = tmp_path / f"bank{suffix}.pt"
    router_path = tmp_path / f"router{suffix}.pt"
    scorer_path = tmp_path / f"scorer{suffix}.pt"

    torch.save({"embed.weight": torch.randn(4, 3)}, core_path)
    torch.save({"marker": torch.zeros(1)}, vocab_path)
    torch.save(
        {"_primitives.14.weight": torch.randn(4, 4), "_primitives.14.bias": torch.randn(4)},
        bank_path,
    )
    torch.save(
        {
            "query_proj.weight": torch.randn(4, 4),
            "query_proj.bias": torch.randn(4),
            "_keys.14": torch.randn(4),
        },
        router_path,
    )
    torch.save({"heads.SELECT.weight": torch.randn(8, 4)}, scorer_path)

    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    schema_hash = "vocab10_ops16_argspan10_v1"
    core_component = mb.ComponentManifest("core", str(core_path), core_raw, core_state, schema_hash)
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    vocab_component = mb.ComponentManifest(
        "vocabulary", str(vocab_path), vocab_raw, vocab_state, schema_hash
    )
    bank_sd = mb.load_state_dict(bank_path)
    primitive = mb.PrimitiveManifestEntry(
        physical_id=14,
        operation_name="CYCLE_FOUR",
        version="v1",
        architecture_signature="cross_position_v1",
        state_abi_hash=mb.canonical_state_hash(mb.primitive_state_dict(bank_sd, 14)),
        core_dependency_hash=core_state,
        decoder_dependency_hash="NOT_APPLICABLE_NO_SEPARATE_DECODER_COMPONENT",
        weights_hash=mb.canonical_state_hash(mb.primitive_state_dict(bank_sd, 14)),
        source_artifact=str(bank_path),
        provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD,
        training_receipt="test-receipt",
    )
    router_sd = mb.load_state_dict(router_path)
    router = mb.RouterManifest(
        source_artifact=str(router_path),
        weights_hash=mb.canonical_state_hash(mb.router_non_key_state_dict(router_sd)),
        task_state_dependency_hash=core_state,
        key_to_primitive_mapping_hash=mb.router_key_to_primitive_mapping_hash(router_sd, [14]),
    )
    scorer_sd = mb.load_state_dict(scorer_path)
    scorer = mb.ArgumentScorerManifest(
        source_artifact=str(scorer_path),
        weights_hash=mb.canonical_state_hash(scorer_sd),
        input_dependency_hash=core_state,
        argument_schema_hash="arg-schema-v1",
    )
    policy = mb.ScoringPolicyManifest(
        family_formula_version="dot_product_v1",
        argument_formula_version="select_sigmoid_v2_adr0088",
        lambda_weight=2.0,
        application_policy_version="rec004a_recovery_v1",
        calibrated_pair_id="pair-1",
    )
    return mb.build_manifest(
        schema_version=1,
        source_commit="test",
        runtime_recipe_version="rec004a-v1",
        environment_record={"python_version": "3.12"},
        model_id="seed10",
        model_seed=10,
        training_run_id="run-1",
        parent_bundle_ids=("parent-bundle-id",),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "diagnostic_only"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=(primitive,),
        router=router,
        argument_scorer=scorer,
        scoring_policy=policy,
    )


def test_manifest_to_json_dict_round_trips_through_load_bundle(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    as_json = ibc._manifest_to_json_dict(manifest)
    reconstructed = ibc._manifest_from_json_dict(as_json)
    assert reconstructed.bundle_id == manifest.bundle_id
    assert reconstructed.content_manifest_digest == manifest.content_manifest_digest
    loaded = mb.load_bundle(reconstructed, mode="nominal", expected_primitive_count=1)
    assert loaded.mode == "nominal"
    assert 14 in loaded.primitive_state_dicts


def test_single_bank_path_rejects_manifest_with_inconsistent_source_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _tiny_manifest(tmp_path)
    import dataclasses as dc

    bad_entry = dc.replace(manifest.primitives[0], source_artifact="somewhere/else.pt")
    bad_manifest = dc.replace(manifest, primitives=(manifest.primitives[0], bad_entry))
    with pytest.raises(mb.IncompleteBundleError):
        ibc._single_bank_path(bad_manifest)


# ---------------------------------------------------------------------------
# Dispatcher config loader.
# ---------------------------------------------------------------------------


def test_dispatcher_rec004a_config_loader_reads_yaml(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004a.yaml"
    config_path.write_text(
        "seed: 10\noutput_dir: runs/x/rec004a/run_001\n"
        "step_ladder: [1000, 2000, 4000, 6000]\n"
        "validation_examples: 512\nvalidation_floor: 0.9\n"
        "recheck_query_examples: 512\n",
        encoding="utf-8",
    )
    config = dispatcher._load_rec004a_config(config_path)
    assert config.seed == 10
    assert config.step_ladder == (1000, 2000, 4000, 6000)
    assert config.validation_examples == 512
    assert config.validation_floor == pytest.approx(0.9)


def test_dispatcher_rec004a_config_loader_rejects_non_pilot_seed(tmp_path: Path) -> None:
    import importlib
    import sys

    sys.path.insert(0, "scripts")
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004a.yaml"
    config_path.write_text("seed: 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        dispatcher._load_rec004a_config(config_path)


# ---------------------------------------------------------------------------
# Orchestrator seed guard.
# ---------------------------------------------------------------------------


def test_run_incremental_budget_calibration_task_refuses_a_non_pilot_seed(tmp_path: Path) -> None:
    bad_config = ibc.IncrementalBudgetCalibrationConfig(output_dir=tmp_path, seed=11)
    with pytest.raises(ValueError, match="pre-registered"):
        ibc.run_incremental_budget_calibration_task(bad_config)
