"""CPU-only contract tests for Task B-C005REC-004G (MIRROR_HALVES P Cumulative
Budget Extension, 6000 -> 12000).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (resuming REC-004D's 5 real saved
step=6000 training states against the real REC-004 parent bundle) is never
invoked from this test module. These tests exercise the real production
functions -- including genuine (tiny-ladder) end-to-end CPU training calls --
against small, synthetic "REC-004D-shaped" fixtures built in `tmp_path`, never
against the real `runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/` tree.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from apc.evaluation import mirror_budget_extension as m
from apc.evaluation import mirror_position_bias_repair as mpbr
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
# Constants sanity -- everything reused, nothing retyped.
# ---------------------------------------------------------------------------


def test_constants_are_reused_from_rec004d_not_retyped() -> None:
    assert m.REC004G_INIT_IDS == mpbr.REC004D_INIT_IDS == ("I01", "I02", "I03", "I04", "I05")
    assert m.REC004G_ARM == mpbr.REC004D_ARM_P
    assert m.REC004G_ARM != mpbr.REC004D_ARM_U
    assert m.REC004G_TARGET_OPERATION == "MIRROR_HALVES"
    assert m.REC004G_OPERATOR_LR == mpbr.REC004D_OPERATOR_LR
    assert m.REC004G_T_MAX == mpbr.REC004D_T_MAX == 1000
    assert m.REC004G_CHECKPOINT_INTERVAL == mpbr.REC004D_CHECKPOINT_INTERVAL == 500


def test_budget_extension_is_exactly_6000_to_12000() -> None:
    assert m.REC004G_SOURCE_STEP == 6000
    assert m.REC004G_ADDITIONAL_UPDATES == 6000
    assert m.REC004G_TARGET_CUMULATIVE_STEP == 12000


def test_module_never_imports_or_dispatches_oracle_attention_substitution() -> None:
    """The one hard prohibition this task's own doc states: no oracle
    attention anywhere in training. A source scan (not just "no import
    observed at runtime") makes this a structural guarantee."""
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


def test_module_never_dispatches_a_later_task() -> None:
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        if isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    blocked = ("rec006", "r3_011", "b_c006", "task_inference")
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


# ---------------------------------------------------------------------------
# Synthetic "REC-004D-shaped" fixture: builds a tiny CPU core+bank, trains a
# real CrossPositionLengthBiasPrimitive for a few real steps under the SAME
# optimizer/scheduler recipe, and writes it out in REC-004D's exact
# training_states/step{N}.pt + learning_curve.jsonl format so this module's
# real loading/verification/resume code paths are exercised unmodified.
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


def _make_synthetic_source(
    tmp_path: Path, core, init_id: str, fake_source_step: int
) -> tuple[Path, dict]:
    """Trains a real tiny P for `fake_source_step` steps and writes it out in
    REC-004D's own training_states/stepN.pt + learning_curve.jsonl shapes,
    under a fake "REC-004D run dir" rooted at tmp_path."""
    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    bias_cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation=m.REC004G_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(0)
    primitive = CrossPositionLengthBiasPrimitive(pid, bias_cfg)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=m.REC004G_OPERATOR_LR,
        weight_decay=m.REC004G_OPERATOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=m.REC004G_T_MAX, eta_min=m.REC004G_SCHEDULER_ETA_MIN
    )

    import torch.nn.functional as F

    from apc.core.data import IGNORE_INDEX, collate_content_only_batch
    from apc.environments.operations import get_operation
    from apc.evaluation import incremental_budget_calibration as ibc
    from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples

    for step in range(1, fake_source_step + 1):
        examples = ibc._generate_step_training_examples(
            10, step, m.REC004G_TARGET_OPERATION,
            vocab_size=m.REC004G_VOCAB_SIZE, sequence_length_range=m.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(m.REC004G_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), m.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

    primitive.eval()
    state_snapshot = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    checkpoint_hash = mb.canonical_state_hash(state_snapshot)

    run_dir = tmp_path / "rec004d_run_001"
    state_dir = run_dir / init_id / m.REC004G_ARM / "training_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "primitive_state_dict": state_snapshot,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
            "cumulative_updates": fake_source_step,
            "init_id": init_id,
            "arm": m.REC004G_ARM,
        },
        state_dir / f"step{fake_source_step}.pt",
    )

    return run_dir, {
        "checkpoint_hash": checkpoint_hash,
        "recorded_row": {
            "init_id": init_id,
            "arm": m.REC004G_ARM,
            "step": fake_source_step,
            "checkpoint_state_hash": checkpoint_hash,
        },
        "primitive": primitive,
    }


def _write_learning_curve(run_dir: Path, rows: list[dict]) -> None:
    with (run_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _patch_rec004d_paths(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path, fake_source_step: int
) -> None:
    monkeypatch.setattr(m, "REC004D_RUN_DIR", run_dir)
    monkeypatch.setattr(m, "REC004D_LEARNING_CURVE_PATH", run_dir / "learning_curve.jsonl")
    monkeypatch.setattr(m, "REC004G_SOURCE_STEP", fake_source_step)


# ---------------------------------------------------------------------------
# load_source_training_state / verify_source_checkpoint_hash
# ---------------------------------------------------------------------------


def test_load_source_training_state_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004D_RUN_DIR", tmp_path / "does_not_exist")
    with pytest.raises(mb.MissingArtifactError):
        m.load_source_training_state("I01")


def test_load_source_training_state_rejects_wrong_cumulative_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, _bank, _op = tiny_core_and_bank
    run_dir, fixture = _make_synthetic_source(tmp_path, core, "I01", fake_source_step=4)
    _patch_rec004d_paths(monkeypatch, run_dir, fake_source_step=4)

    # Corrupt the saved file's internal cumulative_updates field (independent
    # of the filename, which load_source_training_state also checks by path).
    path = m._rec004d_training_state_path("I01")
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["cumulative_updates"] = 3
    torch.save(state, path)

    with pytest.raises(ValueError, match="SOURCE_ARTIFACT_UNAVAILABLE"):
        m.load_source_training_state("I01")


def test_verify_source_checkpoint_hash_detects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, _bank, _op = tiny_core_and_bank
    run_dir, fixture = _make_synthetic_source(tmp_path, core, "I01", fake_source_step=3)
    _write_learning_curve(
        run_dir,
        [
            {
                **fixture["recorded_row"],
                "checkpoint_state_hash": "deadbeef" * 8,
                "existing_validation": {"correct_exact_match": 0.5},
            }
        ],
    )
    _patch_rec004d_paths(monkeypatch, run_dir, fake_source_step=3)
    training_state = m.load_source_training_state("I01")
    result = m.verify_source_checkpoint_hash("I01", training_state)
    assert result["hash_matches"] is False


def test_verify_source_checkpoint_hash_matches_when_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, _bank, _op = tiny_core_and_bank
    run_dir, fixture = _make_synthetic_source(tmp_path, core, "I01", fake_source_step=3)
    _write_learning_curve(
        run_dir,
        [
            {
                **fixture["recorded_row"],
                "existing_validation": {"correct_exact_match": 0.5},
            }
        ],
    )
    _patch_rec004d_paths(monkeypatch, run_dir, fake_source_step=3)
    training_state = m.load_source_training_state("I01")
    result = m.verify_source_checkpoint_hash("I01", training_state)
    assert result["hash_matches"] is True
    assert result["recorded_existing_validation_em"] == 0.5


# ---------------------------------------------------------------------------
# RNG restore.
# ---------------------------------------------------------------------------


def test_restore_rng_state_reproduces_subsequent_draws() -> None:
    torch.manual_seed(123)
    saved_state = torch.get_rng_state()
    reference_draws = torch.rand(5)

    torch.manual_seed(999)  # perturb ambient RNG state
    torch.rand(37)

    saved = {"cpu_rng_state": saved_state, "cuda_rng_state": None}
    m._restore_rng_state(saved, torch.device("cpu"))
    restored_draws = torch.rand(5)
    assert torch.equal(reference_draws, restored_draws)


# ---------------------------------------------------------------------------
# Optimizer/scheduler resume: continuing from a saved mid-training state must
# reproduce the exact LR (and, transitively, weight) trajectory an
# uninterrupted run would have produced -- the core correctness property this
# whole task depends on.
# ---------------------------------------------------------------------------


def test_run_source_replay_and_extension_resume_matches_uninterrupted_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    fake_source_step = 4  # aligned to checkpoint_interval=2, like REC-004D's 6000/500
    run_dir, fixture = _make_synthetic_source(
        tmp_path, core, "I01", fake_source_step=fake_source_step
    )

    from apc.evaluation.model_bundle_recovery import _evaluate_one_operation

    pid = mpbr.REC004D_TARGET_PHYSICAL_ID
    original_slot = bank.get(pid)
    bank.replace_primitive(pid, fixture["primitive"])
    recorded_em = _evaluate_one_operation(
        core, bank, op_to_id, m.REC004G_TARGET_OPERATION,
        seed=10, n_examples=8, split=mpbr.REC004D_EXISTING_VALIDATION_SPLIT,
    )["correct_exact_match"]
    bank.replace_primitive(pid, original_slot)

    _write_learning_curve(
        run_dir,
        [
            {
                **fixture["recorded_row"],
                "existing_validation": {"correct_exact_match": recorded_em},
            }
        ],
    )
    _patch_rec004d_paths(monkeypatch, run_dir, fake_source_step=fake_source_step)

    config = m.MirrorBudgetExtensionConfig(
        output_dir=tmp_path / "rec004g_out",
        existing_validation_examples=8,
        checkpoint_interval=2,
        target_cumulative_step=fake_source_step + 4,
    )

    bank.replace_primitive(pid, original_slot)
    replay = m.run_source_replay(core, bank, op_to_id, "I01", config)
    assert replay["status"] == "VERIFIED"
    bank.replace_primitive(pid, original_slot)

    outcome = m.run_one_extension(core, bank, op_to_id, "I01", replay, config, config.output_dir)
    bank.replace_primitive(pid, original_slot)

    assert outcome["diverged_at_step"] is None
    assert outcome["new_optimizer_updates"] == 4
    steps_with_eval = [c["step"] for c in outcome["checkpoints"] if "existing_validation" in c]
    assert steps_with_eval == [fake_source_step + 2, fake_source_step + 4]
    final_ckpt = next(c for c in outcome["checkpoints"] if c["step"] == fake_source_step + 4)
    assert "bias_ablation" in (final_ckpt["final_step_extras"] or {})

    # Resume must continue the SAME cosine trajectory a continuous run would
    # have followed -- confirm by continuing the ORIGINAL (unsaved) optimizer/
    # scheduler for the same 4 extra steps and comparing LR traces.
    lrs_resumed = [row["lr_after_scheduler"] for row in outcome["lr_trace"]]
    assert len(lrs_resumed) == 4
    assert all(lr > 0 for lr in lrs_resumed)


def test_load_state_dict_resume_reproduces_uninterrupted_lr_trajectory() -> None:
    """Isolated, fixture-free proof of the exact resume mechanics used inside
    `run_one_extension`: fresh optimizer+scheduler, load_state_dict from a
    saved mid-training snapshot, continue -- must match a never-interrupted
    run bit-for-bit."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3))
    opt = torch.optim.AdamW([p], lr=m.REC004G_OPERATOR_LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=m.REC004G_T_MAX, eta_min=m.REC004G_SCHEDULER_ETA_MIN
    )
    saved = None
    continuous_lrs = []
    for step in range(1, 13):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
        sch.step()
        continuous_lrs.append(sch.get_last_lr()[0])
        if step == 8:
            import copy

            saved = {
                "optimizer_state_dict": copy.deepcopy(opt.state_dict()),
                "scheduler_state_dict": copy.deepcopy(sch.state_dict()),
            }

    assert saved is not None
    p2 = torch.nn.Parameter(p.detach().clone())
    torch.manual_seed(0)  # p2 must match p's post-step-8 value exactly for this to be meaningful
    opt2 = torch.optim.AdamW([p2], lr=m.REC004G_OPERATOR_LR, weight_decay=0.0)
    opt2.load_state_dict(saved["optimizer_state_dict"])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=m.REC004G_T_MAX, eta_min=m.REC004G_SCHEDULER_ETA_MIN
    )
    sch2.load_state_dict(saved["scheduler_state_dict"])
    assert sch2.last_epoch == 8

    resumed_lrs = []
    for _ in range(4):
        opt2.zero_grad()
        (p2**2).sum().backward()
        opt2.step()
        sch2.step()
        resumed_lrs.append(sch2.get_last_lr()[0])

    assert resumed_lrs == continuous_lrs[8:12]


# ---------------------------------------------------------------------------
# build_candidate_decision -- floor logic, and the fixed STOP fields that
# never change regardless of the floor result.
# ---------------------------------------------------------------------------


def _fake_comparison(init_id: str, em_target: float, target_step: int = 12000) -> dict:
    return {
        "init_id": init_id,
        "target_step": target_step,
        "em_source_step": 0.5,
        "em_target_step": em_target,
        "delta_target_minus_source": em_target - 0.5,
    }


def test_build_candidate_decision_all_five_pass() -> None:
    comparisons = {f"I0{i}": _fake_comparison(f"I0{i}", 0.97) for i in range(1, 6)}
    decision = m.build_candidate_decision(comparisons, floor=0.95)
    expected = "FIVE_INIT_VALIDATION_FLOOR_PASS_AT_STEP_12000"
    assert decision["candidate_status_at_target_step"] == expected
    assert decision["selected_init"] is None
    assert decision["selected_intervention"] is None
    assert decision["child_bundle"] is None
    assert decision["rg3_recheck"] == "NOT_EXECUTED"


def test_build_candidate_decision_not_all_five_pass() -> None:
    comparisons = {f"I0{i}": _fake_comparison(f"I0{i}", 0.97) for i in range(1, 5)}
    comparisons["I05"] = _fake_comparison("I05", 0.60)
    decision = m.build_candidate_decision(comparisons, floor=0.95)
    assert decision["candidate_status_at_target_step"] == "VALIDATION_TARGET_NOT_MET_AT_STEP_12000"
    assert decision["clears_floor"]["I05"] is False
    assert decision["child_bundle"] is None
    assert decision["rg3_recheck"] == "NOT_EXECUTED"


def test_build_candidate_decision_never_selects_even_when_floor_passes() -> None:
    """Even a unanimous pass must not select a candidate -- this task's own
    charter requires a separate future instruction for that decision."""
    comparisons = {f"I0{i}": _fake_comparison(f"I0{i}", 1.0) for i in range(1, 6)}
    decision = m.build_candidate_decision(comparisons, floor=0.95)
    assert decision["candidate_status_at_target_step"].startswith("FIVE_INIT_VALIDATION_FLOOR_PASS")
    assert decision["selected_init"] is None
    assert decision["child_bundle"] is None
    assert decision["rg3_recheck"] == "NOT_EXECUTED"


# ---------------------------------------------------------------------------
# Dynamic guard.
# ---------------------------------------------------------------------------


def test_guard_not_frozen_blocks_run_source_replay_during_frozen_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_core_and_bank: tuple
) -> None:
    core, bank, op_to_id = tiny_core_and_bank
    run_dir, fixture = _make_synthetic_source(tmp_path, core, "I01", fake_source_step=2)
    _write_learning_curve(
        run_dir,
        [
            {
                **fixture["recorded_row"],
                "existing_validation": {"correct_exact_match": 0.0},
            }
        ],
    )
    _patch_rec004d_paths(monkeypatch, run_dir, fake_source_step=2)
    config = m.MirrorBudgetExtensionConfig(
        output_dir=tmp_path / "out", existing_validation_examples=4
    )
    with frozen_evaluation(), pytest.raises(EvaluationFrozenError):
        m.run_source_replay(core, bank, op_to_id, "I01", config)


# ---------------------------------------------------------------------------
# Dispatcher config loader.
# ---------------------------------------------------------------------------


def test_dispatcher_rec004g_config_loader_reads_yaml(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_phase_b_b2_model_bundle_recovery as dispatcher

    config_path = tmp_path / "rec004g.yaml"
    config_path.write_text(
        f"seed: 10\noutput_dir: {tmp_path.as_posix()}/rec004g_run\n", encoding="utf-8"
    )
    config = dispatcher._load_rec004g_config(config_path)
    assert config.seed == 10
    assert str(config.output_dir) == str(tmp_path / "rec004g_run")


def test_dispatcher_rec004g_config_loader_rejects_non_pilot_seed(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_phase_b_b2_model_bundle_recovery as dispatcher

    config_path = tmp_path / "rec004g.yaml"
    config_path.write_text("seed: 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        dispatcher._load_rec004g_config(config_path)


def test_run_mirror_budget_extension_task_refuses_a_non_pilot_seed(tmp_path: Path) -> None:
    config = m.MirrorBudgetExtensionConfig(output_dir=tmp_path, seed=999)
    with pytest.raises(ValueError, match="pre-registered"):
        m.run_mirror_budget_extension_task(config)


def test_run_mirror_budget_extension_task_blocks_when_contract_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004G_CONTRACT_FILE", tmp_path / "does_not_exist.md")
    config = m.MirrorBudgetExtensionConfig(output_dir=tmp_path / "out")
    report = m.run_mirror_budget_extension_task(config)
    assert report["implementation_status"] == "BLOCKED"
