"""CPU-only contract tests for Task B-C005REC-004K (I03 Matched-Data
Temporal Mechanism Recheck & Same-Init Downstream Rollback Audit).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real seed-10 milestone run (reading REC-004D's and REC-004H's
real, multi-gigabyte `run_001` trees for I03 at step=6000/17500/18000) is
never invoked from this test module. These tests exercise the real
production functions either as pure-Python unit tests or against small,
synthetic "REC-004D/H-shaped" fixtures built in `tmp_path`.
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
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_temporal_mechanism_rollback_audit as m
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
# Constants sanity -- reused from REC-004J wherever the recipe is unchanged.
# ---------------------------------------------------------------------------


def test_temporal_steps_and_decisive_init_are_preregistered() -> None:
    assert m.REC004K_DECISIVE_INIT == "I03"
    assert m.REC004K_EARLY_STEP == 6000
    assert m.REC004K_LATE_STEP == 17500
    assert m.REC004K_TERMINAL_STEP == 18000
    assert m.REC004K_TEMPORAL_STEPS == (6000, 17500, 18000)
    assert m.REC004K_TARGET_LENGTH == 10


def test_oracle_em_threshold_reused_verbatim_from_rec004j() -> None:
    import apc.evaluation.mirror_late_stage_attention_bottleneck_revalidation as rec004j

    assert m.REC004K_ORACLE_EM_THRESHOLD == rec004j.REC004J_ORACLE_EM_THRESHOLD == 0.95


def test_decisive_positions_match_mirror_halves_pivot() -> None:
    pi_10 = mpid.mirror_halves_position_map(10)
    assert pi_10[m.REC004K_POSITION_MAIN_RESIDUAL] == 0
    assert pi_10[m.REC004K_POSITION_SYMMETRIC_CONTROL] == 9


def test_component_and_rollback_ids_are_the_fixed_sets() -> None:
    assert m.REC004K_COMPONENT_IDS == (
        "VALUE_OUTPROJ", "QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "FFN_BLOCK", "READOUT",
    )
    assert m.REC004K_ROLLBACK_IDS == ("R0", "R1", "R2", "R3", "R4", "R5", "R_ALL")
    assert m.REC004K_ROLLBACK_TO_COMPONENT["R0"] == ()
    assert m.REC004K_ROLLBACK_TO_COMPONENT["R_ALL"] == m.REC004K_COMPONENT_IDS
    for rid, component_ids in m.REC004K_ROLLBACK_TO_COMPONENT.items():
        if rid not in ("R0", "R_ALL"):
            assert len(component_ids) == 1


def test_module_never_trains_or_writes_a_checkpoint_file() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "AdamW" not in source
    assert "CosineAnnealingLR" not in source
    assert "torch.save(" not in source


def test_module_never_selects_or_starts_further_repair() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    blocked = ("rec005", "r3_011", "b_c006", "task_inference")
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


def test_module_never_searches_a_rollback_combination_beyond_the_fixed_set() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ('"R6"', '"R7"', "itertools.combinations", "itertools.product"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Stage B: partition_state_dict_keys -- exhaustive, non-overlapping, derived
# from the REAL state_dict of a real CrossPositionLengthBiasPrimitive.
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
        operation=m.REC004K_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(seed)
    primitive = CrossPositionLengthBiasPrimitive(pid, bias_cfg)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def test_partition_state_dict_keys_is_exhaustive_and_nonoverlapping(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=1)
    keys = list(primitive.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)

    all_grouped = [k for ks in groups.values() for k in ks]
    assert sorted(all_grouped) == sorted(keys)
    assert len(all_grouped) == len(set(all_grouped))  # no key assigned twice

    assert groups["VALUE_OUTPROJ"] == [
        "content_in_proj.weight", "content_in_proj.bias", "content_position_embedding.weight",
        "cross_attn.in_proj_weight", "cross_attn.in_proj_bias",
        "cross_attn.out_proj.weight", "cross_attn.out_proj.bias",
    ]
    assert groups["QUERY_RESIDUAL_PATH"] == ["answer_query_embedding.weight"]
    assert groups["POST_ATTN_NORM"] == ["attn_norm.weight", "attn_norm.bias"]
    assert groups["FFN_BLOCK"] == [
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
        "ffn_norm.weight", "ffn_norm.bias",
    ]
    assert groups["READOUT"] == ["readout.weight", "readout.bias"]
    assert groups["EXCLUDED_FROM_O1"] == [
        "position_bias_hidden.weight", "position_bias_hidden.bias", "position_bias_out.weight",
    ]


def test_partition_state_dict_keys_raises_on_an_unrecognized_key() -> None:
    with pytest.raises(AssertionError):
        m.partition_state_dict_keys(["some_new_submodule.weight"])


def test_partition_state_dict_keys_raises_if_a_key_double_matches(monkeypatch) -> None:
    monkeypatch.setitem(
        m.REC004K_COMPONENT_PREFIXES, "READOUT_DUP", ("readout.",)
    )
    with pytest.raises(AssertionError):
        m.partition_state_dict_keys(["readout.weight"])


# ---------------------------------------------------------------------------
# Stage B: empirical forward-graph invariance check.
# ---------------------------------------------------------------------------


def test_o1_invariant_to_excluded_params_and_restores_exactly(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=1)
    n = 10
    content_lengths = [n] * 6
    output_lengths = [n] * 6
    torch.manual_seed(42)
    content = torch.randn(6, n, core.model.config.d_model)

    before_hash = mb.canonical_state_hash(primitive.state_dict())
    with torch.no_grad():
        result = m.verify_o1_invariance_to_excluded_params(
            primitive, content, content_lengths, output_lengths
        )
    after_hash = mb.canonical_state_hash(primitive.state_dict())

    assert result["max_abs_logit_diff_after_perturbation"] == 0.0
    assert result["invariant"] is True
    assert result["restore_check_max_abs_diff"] == 0.0
    assert result["weights_restored_exactly"] is True
    assert after_hash == before_hash  # primitive is byte-identical after the check


def test_o1_is_not_invariant_to_value_outproj(tiny_core_and_bank) -> None:
    """Negative control: perturbing a parameter that DOES feed O1 (the
    out_proj bias, part of VALUE_OUTPROJ) must change O1's output -- proves
    the invariance check in the positive test isn't vacuous."""
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=1)
    n = 10
    content_lengths = [n] * 6
    output_lengths = [n] * 6
    torch.manual_seed(42)
    content = torch.randn(6, n, core.model.config.d_model)
    with torch.no_grad():
        baseline = oracle_probe.run_oracle_forward(
            primitive, content, content_lengths, output_lengths
        )
        primitive.cross_attn.out_proj.bias.add_(5.0)
        perturbed = oracle_probe.run_oracle_forward(
            primitive, content, content_lengths, output_lengths
        )
    diff = (perturbed["logits"] - baseline["logits"]).abs().max().item()
    assert diff > 0.0


# ---------------------------------------------------------------------------
# Stage C: rollback primitive construction and mechanics.
# ---------------------------------------------------------------------------


def test_r0_rollback_reproduces_the_late_primitive_under_o1(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    r0 = m._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004K_ROLLBACK_TO_COMPONENT["R0"], groups
    )

    n = 10
    content_lengths = [n] * 5
    output_lengths = [n] * 5
    torch.manual_seed(9)
    content = torch.randn(5, n, core.model.config.d_model)
    with torch.no_grad():
        r0_out = oracle_probe.run_oracle_forward(r0, content, content_lengths, output_lengths)
        late_out = oracle_probe.run_oracle_forward(late, content, content_lengths, output_lengths)
    assert (r0_out["logits"] - late_out["logits"]).abs().max().item() == 0.0


def test_r_all_rollback_reproduces_the_early_primitive_under_o1(tiny_core_and_bank) -> None:
    """The core positive-control property (task doc Section 6): since
    R_ALL replaces every O1-relevant parameter with the early checkpoint's
    values, and the excluded params (proven inert to O1) stay at the late
    checkpoint's values, R_ALL's O1 forward must EXACTLY reproduce the early
    primitive's own O1 forward."""
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    r_all = m._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004K_ROLLBACK_TO_COMPONENT["R_ALL"], groups
    )

    n = 10
    content_lengths = [n] * 5
    output_lengths = [n] * 5
    torch.manual_seed(11)
    content = torch.randn(5, n, core.model.config.d_model)
    with torch.no_grad():
        r_all_out = oracle_probe.run_oracle_forward(r_all, content, content_lengths, output_lengths)
        early_out = oracle_probe.run_oracle_forward(early, content, content_lengths, output_lengths)
    assert (r_all_out["logits"] - early_out["logits"]).abs().max().item() == 0.0


def test_late_and_early_primitives_are_unmutated_by_rollback_construction(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    late_hash_before = mb.canonical_state_hash(late.state_dict())
    early_hash_before = mb.canonical_state_hash(early.state_dict())
    keys = list(late.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    for rollback_id in m.REC004K_ROLLBACK_IDS:
        m._build_rollback_primitive(
            core, late_sd, early_sd, m.REC004K_ROLLBACK_TO_COMPONENT[rollback_id], groups
        )
    assert mb.canonical_state_hash(late.state_dict()) == late_hash_before
    assert mb.canonical_state_hash(early.state_dict()) == early_hash_before


# ---------------------------------------------------------------------------
# compute_rollback_diagnostic_point: structural invariants.
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


def test_compute_rollback_diagnostic_point_structural_invariants(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    examples = _tiny_length10_examples(seed=123, n=17)

    with torch.no_grad():
        point = m.compute_rollback_diagnostic_point(core, primitive, examples)

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
        assert isinstance(row["j0_mean_target_logit_margin"], float)
        assert isinstance(row["oracle_mean_target_logit_margin"], float)


# ---------------------------------------------------------------------------
# run_component_decomposition_parity_check.
# ---------------------------------------------------------------------------


def test_parity_check_passes_when_r_all_is_built_from_the_compared_early_primitive(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    r_all = m._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004K_ROLLBACK_TO_COMPONENT["R_ALL"], groups
    )
    examples = _tiny_length10_examples(seed=321, n=9)
    datasets = {"ds_a": examples}
    with torch.no_grad():
        result = m.run_component_decomposition_parity_check(core, r_all, early, datasets)
    assert result["status"] == "VERIFIED"
    assert result["per_dataset"]["ds_a"]["max_abs_logit_diff"] == 0.0
    assert result["per_dataset"]["ds_a"]["predictions_match"] is True


def test_parity_check_fails_when_compared_against_a_mismatched_primitive(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    wrong_early = _tiny_primitive(core, seed=99)  # NOT what R_ALL was built from
    keys = list(late.state_dict().keys())
    groups = m.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    r_all = m._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004K_ROLLBACK_TO_COMPONENT["R_ALL"], groups
    )
    examples = _tiny_length10_examples(seed=321, n=9)
    datasets = {"ds_a": examples}
    with torch.no_grad():
        result = m.run_component_decomposition_parity_check(core, r_all, wrong_early, datasets)
    assert result["status"] == "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP"
    assert result["per_dataset"]["ds_a"]["passed"] is False


# ---------------------------------------------------------------------------
# build_stage_a_temporal_decision: pre-registered gate.
# ---------------------------------------------------------------------------


def _stage_a_row(em: float) -> dict:
    return {"status": "VERIFIED", "oracle_sequence_exact_match": em}


def _stage_a_points(early_v2, early_probe, late_v2, late_probe) -> dict:
    return {
        f"{m.REC004K_EARLY_STEP}:{m.REC004K_CLEAN_V2_DATASET}": _stage_a_row(early_v2),
        f"{m.REC004K_EARLY_STEP}:{m.REC004K_PROBE_DATASET}": _stage_a_row(early_probe),
        f"{m.REC004K_LATE_STEP}:{m.REC004K_CLEAN_V2_DATASET}": _stage_a_row(late_v2),
        f"{m.REC004K_LATE_STEP}:{m.REC004K_PROBE_DATASET}": _stage_a_row(late_probe),
    }


def test_stage_a_confirmed_when_early_passes_both_and_late_fails_both() -> None:
    points = _stage_a_points(0.98, 0.97, 0.75, 0.76)
    decision = m.build_stage_a_temporal_decision(points)
    assert decision["label"] == "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED"
    assert decision["authorizes_component_rollback"] is True


def test_stage_a_confirmed_when_late_fails_only_one_dataset() -> None:
    points = _stage_a_points(0.98, 0.97, 0.75, 0.99)
    decision = m.build_stage_a_temporal_decision(points)
    assert decision["label"] == "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED"


def test_stage_a_not_established_when_early_fails_one_dataset() -> None:
    points = _stage_a_points(0.98, 0.40, 0.75, 0.76)
    decision = m.build_stage_a_temporal_decision(points)
    assert decision["label"] == "TEMPORAL_SHIFT_NOT_ESTABLISHED"
    assert decision["authorizes_component_rollback"] is False


def test_stage_a_not_established_when_late_also_passes_both() -> None:
    points = _stage_a_points(0.98, 0.97, 0.96, 0.95)
    decision = m.build_stage_a_temporal_decision(points)
    assert decision["label"] == "TEMPORAL_SHIFT_NOT_ESTABLISHED"


def test_stage_a_not_established_on_missing_rows() -> None:
    decision = m.build_stage_a_temporal_decision({})
    assert decision["label"] == "TEMPORAL_SHIFT_NOT_ESTABLISHED"
    assert decision["authorizes_component_rollback"] is False


# ---------------------------------------------------------------------------
# build_i03_component_decision.
# ---------------------------------------------------------------------------


def _rollback_row(em: float, pos4_acc: float) -> dict:
    return {
        "oracle_sequence_exact_match": em,
        "per_output_position": {"4": {"oracle_accuracy": pos4_acc}},
    }


def _rollback_points(**per_rollback_em_pos4: tuple[float, float]) -> dict:
    points = {}
    for rollback_id, (em, pos4) in per_rollback_em_pos4.items():
        for ds in m.REC004K_DATASETS:
            points[f"{rollback_id}:{ds}"] = _rollback_row(em, pos4)
    return points


_PARITY_OK = {"status": "VERIFIED"}
_PARITY_FAIL = {"status": "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP"}


def test_i03_decision_parity_failed_stop() -> None:
    decision = m.build_i03_component_decision({}, _PARITY_FAIL)
    assert decision["label"] == "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP"


def test_i03_decision_single_component_sufficient() -> None:
    points = _rollback_points(
        R0=(0.20, 0.10), R1=(0.40, 0.30), R2=(0.40, 0.30), R3=(0.40, 0.30),
        R4=(0.40, 0.30), R5=(0.97, 0.98), R_ALL=(0.99, 0.99),
    )
    decision = m.build_i03_component_decision(points, _PARITY_OK)
    assert decision["label"] == "READOUT_ROLLBACK_SUFFICIENT"
    assert decision["sufficient_single_components"] == ["READOUT"]
    assert decision["r_all_passes"] is True


def test_i03_decision_multiple_components_sufficient() -> None:
    points = _rollback_points(
        R0=(0.20, 0.10), R1=(0.40, 0.30), R2=(0.40, 0.30), R3=(0.40, 0.30),
        R4=(0.96, 0.97), R5=(0.97, 0.98), R_ALL=(0.99, 0.99),
    )
    decision = m.build_i03_component_decision(points, _PARITY_OK)
    assert decision["label"] == "MULTIPLE_COMPONENTS_ROLLBACK_SUFFICIENT"
    assert set(decision["sufficient_single_components"]) == {"FFN_BLOCK", "READOUT"}


def test_i03_decision_distributed_downstream_coadaptation() -> None:
    points = _rollback_points(
        R0=(0.20, 0.10), R1=(0.40, 0.30), R2=(0.40, 0.30), R3=(0.40, 0.30),
        R4=(0.40, 0.30), R5=(0.40, 0.30), R_ALL=(0.99, 0.99),
    )
    decision = m.build_i03_component_decision(points, _PARITY_OK)
    assert decision["label"] == "DISTRIBUTED_DOWNSTREAM_COADAPTATION"
    assert decision["sufficient_single_components"] == []
    assert decision["r_all_passes"] is True


def test_i03_decision_downstream_decomposition_incomplete() -> None:
    points = _rollback_points(
        R0=(0.20, 0.10), R1=(0.40, 0.30), R2=(0.40, 0.30), R3=(0.40, 0.30),
        R4=(0.40, 0.30), R5=(0.40, 0.30), R_ALL=(0.50, 0.40),
    )
    decision = m.build_i03_component_decision(points, _PARITY_OK)
    assert decision["label"] == "DOWNSTREAM_DECOMPOSITION_INCOMPLETE"
    assert decision["r_all_passes"] is False


# ---------------------------------------------------------------------------
# Full orchestrator, tiny synthetic REC-004D+REC-004H-shaped tree.
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


def _patch_common(monkeypatch, tmp_path, core, bank, op_to_id) -> tuple[Path, Path]:
    rec004d_dir = tmp_path / "rec004d_run_001"
    rec004h_dir = tmp_path / "rec004h_run_001"
    monkeypatch.setitem(traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004D", rec004d_dir)
    monkeypatch.setitem(traj_audit.REC004I_SOURCE_RUN_DIRS, "B-C005REC-004H", rec004h_dir)
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_FIRST_STEP", 1)
    monkeypatch.setattr(traj_audit, "REC004I_TRAIN_STREAM_LAST_STEP", 5)
    monkeypatch.setattr(
        ibc, "_reconstruct_parent_runtime", lambda config, parent_manifest: (core, bank, op_to_id)
    )
    monkeypatch.setattr(ibc, "_load_parent_manifest", lambda: (object(), {}))
    return rec004d_dir, rec004h_dir


def test_full_orchestrator_stops_after_stage_a_on_untrained_checkpoints(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    """Untrained, random tiny checkpoints essentially never clear the 0.95
    oracle-EM floor -- exercises the real Stage A STOP-gate path end to
    end."""
    core, bank, op_to_id = tiny_core_and_bank
    rec004d_dir, rec004h_dir = _patch_common(monkeypatch, tmp_path, core, bank, op_to_id)
    row_6000 = _write_tiny_checkpoint(rec004d_dir, core, "I03", 6000, seed=1)
    row_17500 = _write_tiny_checkpoint(rec004h_dir, core, "I03", 17500, seed=2)
    row_18000 = _write_tiny_checkpoint(rec004h_dir, core, "I03", 18000, seed=3)
    (rec004d_dir / "learning_curve.jsonl").write_text(json.dumps(row_6000) + "\n", encoding="utf-8")
    with (rec004h_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for row in (row_17500, row_18000):
            fh.write(json.dumps(row) + "\n")

    config = m.MirrorTemporalMechanismRollbackAuditConfig(
        output_dir=tmp_path / "rec004k_run_001", seed=10,
    )
    summary = m.run_mirror_temporal_mechanism_rollback_audit_task(config)

    assert summary["implementation_status"] == "COMPLETE"
    assert summary["checkpoint_source_replay_status"] == "VERIFIED"
    assert summary["new_optimizer_updates"] == 0
    assert summary["selected_init"] is None
    assert summary["selected_step"] is None
    assert summary["selected_component"] is None
    assert summary["child_bundle"] is None
    assert summary["rg3_recheck"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["freeze_audit"]["core_unchanged"] is True
    assert summary["freeze_audit"]["protected_operations_unchanged"] is True
    assert summary["side_effect_audit"]["shared_cache_unchanged"] is True
    assert summary["rec004j_cross_check_status"] in (
        "VERIFIED", "MISMATCH", "REC004J_ARTIFACT_UNAVAILABLE",
    )
    assert summary["stage_a_temporal_decision"] in (
        "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED", "TEMPORAL_SHIFT_NOT_ESTABLISHED",
    )
    if summary["stage_a_temporal_decision"] == "TEMPORAL_SHIFT_NOT_ESTABLISHED":
        assert summary["stage_c_status"] == "NOT_EXECUTED_TEMPORAL_SHIFT_NOT_ESTABLISHED"
        assert summary["i03_component_decision"] is None

    out = config.output_dir
    for name in (
        "clean_v2_detail.json", "length10_mechanism_probe_v1_manifest.json",
        "stage_a_temporal_points.jsonl", "stage_a_temporal_points_summary.json",
        "stage_a_checkpoint_load_info.json", "rec004j_cross_check.json",
        "stage_a_temporal_decision.json", "freeze_audit.json", "side_effect_audit.json",
        "cost_accounting.json", "summary.json", "config.yaml", "system.json", "protocol.json",
        "report.md",
    ):
        assert (out / name).is_file(), name


def test_full_orchestrator_executes_stage_c_when_temporal_decision_forced_confirmed(
    tmp_path, monkeypatch, tiny_core_and_bank
) -> None:
    """Forces the Stage A gate via a minimal, disclosed test seam (overriding
    only the `oracle_sequence_exact_match` field the gate reads, leaving
    every real forward computation untouched) so Stage B/C/D's real
    integration path -- component partition, invariance check, 7-variant
    rollback, parity check, decision -- runs for real on the tiny fixture."""
    core, bank, op_to_id = tiny_core_and_bank
    rec004d_dir, rec004h_dir = _patch_common(monkeypatch, tmp_path, core, bank, op_to_id)
    row_6000 = _write_tiny_checkpoint(rec004d_dir, core, "I03", 6000, seed=1)
    row_17500 = _write_tiny_checkpoint(rec004h_dir, core, "I03", 17500, seed=2)
    row_18000 = _write_tiny_checkpoint(rec004h_dir, core, "I03", 18000, seed=3)
    (rec004d_dir / "learning_curve.jsonl").write_text(json.dumps(row_6000) + "\n", encoding="utf-8")
    with (rec004h_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for row in (row_17500, row_18000):
            fh.write(json.dumps(row) + "\n")

    real_compute = m.rec004j.compute_diagnostic_point
    call_state = {"count": 0}

    def fake_compute(core_arg, primitive, examples):
        point = real_compute(core_arg, primitive, examples)
        step_index = call_state["count"] // 2  # (6000,6000,17500,17500,18000,18000)
        call_state["count"] += 1
        point["oracle_sequence_exact_match"] = 1.0 if step_index == 0 else 0.5
        return point

    monkeypatch.setattr(m.rec004j, "compute_diagnostic_point", fake_compute)

    config = m.MirrorTemporalMechanismRollbackAuditConfig(
        output_dir=tmp_path / "rec004k_run_001", seed=10,
    )
    summary = m.run_mirror_temporal_mechanism_rollback_audit_task(config)

    assert summary["stage_a_temporal_decision"] == "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED"
    assert summary["stage_c_status"] == "EXECUTED"
    assert summary["component_decomposition_parity_status"] == "VERIFIED"
    assert summary["i03_component_decision"] in (
        "DOWNSTREAM_DECOMPOSITION_INCOMPLETE", "DISTRIBUTED_DOWNSTREAM_COADAPTATION",
        "MULTIPLE_COMPONENTS_ROLLBACK_SUFFICIENT",
        "VALUE_OUTPROJ_ROLLBACK_SUFFICIENT", "QUERY_RESIDUAL_PATH_ROLLBACK_SUFFICIENT",
        "POST_ATTN_NORM_ROLLBACK_SUFFICIENT", "FFN_BLOCK_ROLLBACK_SUFFICIENT",
        "READOUT_ROLLBACK_SUFFICIENT",
    )
    assert summary["selected_init"] is None
    assert summary["selected_component"] is None
    assert summary["new_optimizer_updates"] == 0

    out = config.output_dir
    for name in (
        "component_key_groups.json", "forward_graph_invariance_check.json",
        "stage_c_rollback_points.jsonl", "stage_c_rollback_points_summary.json",
        "component_decomposition_parity_check.json", "i03_component_decision.json",
    ):
        assert (out / name).is_file(), name

    rollback_lines = (out / "stage_c_rollback_points.jsonl").read_text().splitlines()
    assert len(rollback_lines) == len(m.REC004K_ROLLBACK_IDS) * 2  # 7 variants x 2 datasets


def test_dispatcher_lists_rec004k_and_validates_seed(tmp_path) -> None:
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")
    assert "B-C005REC-004K" in dispatcher._IMPLEMENTED_TASKS

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("seed: 99\noutput_dir: runs/x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dispatcher._load_rec004k_config(bad_config)
