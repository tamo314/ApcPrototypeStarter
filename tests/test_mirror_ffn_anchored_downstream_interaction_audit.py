"""CPU-only contract tests for Task B-C005REC-004L (FFN-Anchored Downstream
Interaction Audit).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real milestone run (reading REC-004D's and REC-004H's real,
multi-gigabyte `run_001` trees for I03 at step=6000/17500) is never invoked
from this test module. These tests exercise the real production functions
either as pure-Python unit tests or against small, synthetic
"REC-004D/H-shaped" fixtures built in-process.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest
import torch

from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import mirror_ffn_anchored_downstream_interaction_audit as m
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_temporal_mechanism_rollback_audit as rec004k
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
# Constants sanity -- reused from REC-004K wherever the recipe is unchanged;
# only the FFN-anchored condition set and new probe dataset are new here.
# ---------------------------------------------------------------------------


def test_decisive_checkpoint_identity_reused_verbatim_from_rec004k() -> None:
    assert m.REC004L_DECISIVE_INIT == rec004k.REC004K_DECISIVE_INIT == "I03"
    assert m.REC004L_EARLY_STEP == rec004k.REC004K_EARLY_STEP == 6000
    assert m.REC004L_LATE_STEP == rec004k.REC004K_LATE_STEP == 17500
    assert m.REC004L_TARGET_LENGTH == rec004k.REC004K_TARGET_LENGTH == 10


def test_thresholds_reused_verbatim_from_rec004k() -> None:
    assert m.REC004L_ROLLBACK_EM_THRESHOLD == rec004k.REC004K_ROLLBACK_EM_THRESHOLD == 0.95
    assert (
        m.REC004L_ROLLBACK_POSITION_ACC_THRESHOLD
        == rec004k.REC004K_ROLLBACK_POSITION_ACC_THRESHOLD
        == 0.95
    )


def test_condition_set_is_the_fixed_ffn_anchored_pairs_only() -> None:
    assert m.REC004L_CONDITION_IDS == ("R0", "F", "F_V", "F_Q", "F_N", "F_R", "R_ALL", "EARLY")
    assert m.REC004L_PAIR_CONDITION_IDS == ("F_V", "F_Q", "F_N", "F_R")
    assert m.REC004L_CONDITION_TO_COMPONENTS["R0"] == ()
    assert m.REC004L_CONDITION_TO_COMPONENTS["F"] == ("FFN_BLOCK",)
    for pair_id in m.REC004L_PAIR_CONDITION_IDS:
        components = m.REC004L_CONDITION_TO_COMPONENTS[pair_id]
        assert len(components) == 2
        assert "FFN_BLOCK" in components
    assert m.REC004L_CONDITION_TO_COMPONENTS["R_ALL"] == rec004k.REC004K_COMPONENT_IDS
    # Every partner named across the 4 pairs is a distinct non-FFN component.
    partners = {
        c for cid in m.REC004L_PAIR_CONDITION_IDS for c in m.REC004L_CONDITION_TO_COMPONENTS[cid]
        if c != "FFN_BLOCK"
    }
    assert partners == {"VALUE_OUTPROJ", "QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "READOUT"}


def test_datasets_include_both_existing_sets_plus_exactly_one_new_set() -> None:
    assert m.REC004L_DATASETS == (
        m.REC004L_CLEAN_V2_DATASET, m.REC004L_PROBE_DATASET, m.REC004L_NEW_PROBE_SPLIT,
    )
    assert m.REC004L_CLEAN_V2_DATASET == rec004k.REC004K_CLEAN_V2_DATASET
    assert m.REC004L_PROBE_DATASET == rec004k.REC004K_PROBE_DATASET
    assert m.REC004L_NEW_PROBE_SPLIT == "length10_downstream_interaction_probe_v1"
    assert m.REC004L_NEW_PROBE_EXAMPLES == 512


def test_module_never_trains_or_writes_a_checkpoint_file() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "AdamW" not in source
    assert "CosineAnnealingLR" not in source
    assert "torch.save(" not in source


def test_module_never_selects_or_starts_further_repair() -> None:
    """Checks function/class NAMES only (not docstring prose, which
    legitimately mentions `B-C005REC-005`/`rec005_eligible` as a fixed,
    always-`False` non-adoption field) -- same convention as REC-004K's own
    test."""
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    blocked = ("rec005", "r3_011", "b_c006", "task_inference")
    assert not any(b in n.lower().replace("-", "_") for n in names for b in blocked)


def test_module_never_searches_beyond_the_fixed_condition_set() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ("itertools.combinations", "itertools.product", '"F_V_Q"', '"F_ALL_PAIRS"'):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# New disjoint dataset generation -- pure function of (seed, protected set).
# ---------------------------------------------------------------------------


def test_new_probe_dataset_is_deterministic_and_disjoint_from_a_protected_set() -> None:
    protected = {"deadbeef" * 4}  # a single, irrelevant digest -- never collides
    examples_a, detail_a = m.build_length10_downstream_interaction_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    examples_b, detail_b = m.build_length10_downstream_interaction_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    assert len(examples_a) == len(examples_b) == 32
    assert [e.input_tokens for e in examples_a] == [e.input_tokens for e in examples_b]
    assert [e.target_tokens for e in examples_a] == [e.target_tokens for e in examples_b]
    assert detail_a["development_exposed"] is True
    assert detail_a["sealed_or_rg3_query"] is False


def test_new_probe_dataset_never_draws_a_protected_digest() -> None:
    from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit

    # Generate a small unprotected set first, then protect ITS digests, and
    # confirm a second draw with that protected set never reproduces them.
    baseline_examples, _ = m.build_length10_downstream_interaction_probe_v1(
        seed=11, protected_digests=set(), n=16
    )
    protected = traj_audit._digest_examples(baseline_examples)
    guarded_examples, detail = m.build_length10_downstream_interaction_probe_v1(
        seed=11, protected_digests=protected, n=16
    )
    guarded_digests = traj_audit._digest_examples(guarded_examples)
    assert not (guarded_digests & protected)
    assert detail["substitution_count"] >= 0


def test_new_probe_dataset_uses_a_distinct_rng_stream_from_rec004j_probe() -> None:
    """Different split label -> different `_derive_local_seed` output ->
    not simply a re-draw of REC-004J/004K's own probe sequence."""
    from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j

    ours, _ = m.build_length10_downstream_interaction_probe_v1(
        seed=10, protected_digests=set(), n=8
    )
    theirs, _ = rec004j.build_length10_mechanism_probe_v1(seed=10, protected_digests=set(), n=8)
    assert [e.input_tokens for e in ours] != [e.input_tokens for e in theirs]


def test_dataset_digest_is_stable_and_order_independent() -> None:
    examples, _ = m.build_length10_downstream_interaction_probe_v1(
        seed=12, protected_digests=set(), n=8
    )
    digest_forward = m._dataset_digest(examples)
    digest_reversed = m._dataset_digest(list(reversed(examples)))
    assert digest_forward == digest_reversed
    assert digest_forward == m._dataset_digest(examples)


# ---------------------------------------------------------------------------
# Condition primitive construction -- built on REC-004K's rollback merge
# mechanism, unmodified.
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
        operation=m.REC004L_TARGET_OPERATION, d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10, max_sequence_length=32,
        bias_hidden_dim=32, length_ref=32,
    )
    torch.manual_seed(seed)
    primitive = CrossPositionLengthBiasPrimitive(pid, bias_cfg)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def _tiny_length10_examples(seed: int, n: int) -> list[Example]:
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


def test_f_condition_rolls_back_exactly_the_ffn_block_keys(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = rec004k.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    f_primitive = rec004k._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004L_CONDITION_TO_COMPONENTS["F"], groups
    )
    f_sd = f_primitive.state_dict()
    for key in groups["FFN_BLOCK"]:
        assert torch.equal(f_sd[key], early_sd[key])
    for cid in ("VALUE_OUTPROJ", "QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "READOUT"):
        for key in groups[cid]:
            assert torch.equal(f_sd[key], late_sd[key])


def test_f_v_condition_rolls_back_ffn_and_value_outproj_only(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = rec004k.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    fv_primitive = rec004k._build_rollback_primitive(
        core, late_sd, early_sd, m.REC004L_CONDITION_TO_COMPONENTS["F_V"], groups
    )
    fv_sd = fv_primitive.state_dict()
    for cid in ("FFN_BLOCK", "VALUE_OUTPROJ"):
        for key in groups[cid]:
            assert torch.equal(fv_sd[key], early_sd[key])
    for cid in ("QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "READOUT"):
        for key in groups[cid]:
            assert torch.equal(fv_sd[key], late_sd[key])


def test_early_condition_is_the_real_loaded_primitive_not_a_merge(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = rec004k.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    assert primitives["EARLY"] is early


def test_r_all_condition_reproduces_early_primitive_under_o1(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = rec004k.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)

    n = 10
    content_lengths = [n] * 5
    output_lengths = [n] * 5
    torch.manual_seed(9)
    content = torch.randn(5, n, core.model.config.d_model)
    with torch.no_grad():
        r_all_out = oracle_probe.run_oracle_forward(
            primitives["R_ALL"], content, content_lengths, output_lengths
        )
        early_out = oracle_probe.run_oracle_forward(early, content, content_lengths, output_lengths)
    assert (r_all_out["logits"] - early_out["logits"]).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# compute_interaction_diagnostic_point -- structural invariants.
# ---------------------------------------------------------------------------


def test_diagnostic_point_vs_self_reference_has_zero_diff(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    examples = _tiny_length10_examples(seed=123, n=13)

    with torch.no_grad():
        point = m.compute_interaction_diagnostic_point(core, primitive, primitive, examples)

    assert point["n"] == 13
    assert point["max_abs_hidden_state_diff_vs_early"] == 0.0
    assert point["max_abs_logit_diff_vs_early"] == 0.0
    assert set(point["per_output_position"].keys()) == {str(k) for k in range(10)}
    assert 0.0 <= point["oracle_sequence_exact_match"] <= 1.0


def test_diagnostic_point_vs_a_different_reference_is_generally_nonzero(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    reference = _tiny_primitive(core, seed=8)
    examples = _tiny_length10_examples(seed=123, n=13)

    with torch.no_grad():
        point = m.compute_interaction_diagnostic_point(core, primitive, reference, examples)

    assert point["max_abs_hidden_state_diff_vs_early"] > 0.0
    assert point["max_abs_logit_diff_vs_early"] > 0.0


def test_diagnostic_point_never_mutates_either_primitive(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    reference = _tiny_primitive(core, seed=8)
    examples = _tiny_length10_examples(seed=123, n=13)
    hash_before = mb.canonical_state_hash(primitive.state_dict())
    ref_hash_before = mb.canonical_state_hash(reference.state_dict())
    with torch.no_grad():
        m.compute_interaction_diagnostic_point(core, primitive, reference, examples)
    assert mb.canonical_state_hash(primitive.state_dict()) == hash_before
    assert mb.canonical_state_hash(reference.state_dict()) == ref_hash_before


# ---------------------------------------------------------------------------
# Decision rule -- synthetic condition_points, exercising every label branch.
# ---------------------------------------------------------------------------


def _synthetic_point(em: float, pos4: float) -> dict:
    per_position = {str(k): {"oracle_accuracy": 1.0} for k in range(10)}
    per_position["4"] = {"oracle_accuracy": pos4}
    return {"oracle_sequence_exact_match": em, "per_output_position": per_position}


def _full_condition_points(passing_condition_ids: set[str]) -> dict:
    points = {}
    for condition_id in m.REC004L_CONDITION_IDS:
        for dataset_name in m.REC004L_DATASETS:
            if condition_id in passing_condition_ids:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
            else:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(0.5, 0.5)
    return points


_PARITY_OK = {"status": "VERIFIED", "per_dataset": {}}
_PARITY_FAILED = {"status": "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP", "per_dataset": {}}


def test_decision_exactly_one_pair_sufficient() -> None:
    points = _full_condition_points({"F_R"})
    decision = m.build_ffn_interaction_decision(points, _PARITY_OK)
    assert decision["label"] == "FFN_READOUT_INTERACTION_SUFFICIENT"
    assert decision["sufficient_pairs"] == ["F_R"]


def test_decision_multiple_pairs_sufficient_does_not_pick_highest_em() -> None:
    points = _full_condition_points({"F_V", "F_Q"})
    decision = m.build_ffn_interaction_decision(points, _PARITY_OK)
    assert decision["label"] == "MULTIPLE_FFN_INTERACTION_PATHS_SUFFICIENT"
    assert set(decision["sufficient_pairs"]) == {"F_V", "F_Q"}


def test_decision_no_pair_sufficient_but_r_all_passes() -> None:
    points = _full_condition_points({"R_ALL"})
    decision = m.build_ffn_interaction_decision(points, _PARITY_OK)
    assert decision["label"] == "HIGHER_ORDER_OR_MULTI_COMPONENT_COADAPTATION"
    assert decision["sufficient_pairs"] == []
    assert decision["r_all_passes"] is True


def test_decision_nothing_passes() -> None:
    points = _full_condition_points(set())
    decision = m.build_ffn_interaction_decision(points, _PARITY_OK)
    assert decision["label"] == "DOWNSTREAM_DECOMPOSITION_INCOMPLETE"


def test_decision_parity_failure_stops_before_any_interpretation() -> None:
    points = _full_condition_points({"F_V", "F_Q", "F_N", "F_R", "R_ALL"})
    decision = m.build_ffn_interaction_decision(points, _PARITY_FAILED)
    assert decision["label"] == "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP"
    assert decision["sufficient_pairs"] == []


def test_decision_requires_all_three_datasets_to_pass() -> None:
    points = _full_condition_points({"F_R"})
    # Break just one dataset's pass for F_R -- must no longer count as sufficient.
    points[f"F_R:{m.REC004L_NEW_PROBE_SPLIT}"] = _synthetic_point(0.5, 0.5)
    decision = m.build_ffn_interaction_decision(points, _PARITY_OK)
    assert decision["label"] == "DOWNSTREAM_DECOMPOSITION_INCOMPLETE"


# ---------------------------------------------------------------------------
# Regression check -- descriptive only, never gates the decision above.
# ---------------------------------------------------------------------------


def test_regression_check_flags_a_position_that_got_worse_than_r0() -> None:
    points = {}
    for dataset_name in m.REC004L_DATASETS:
        points[f"R0:{dataset_name}"] = _synthetic_point(1.0, 1.0)
        row = _synthetic_point(1.0, 1.0)
        row["per_output_position"]["1"] = {"oracle_accuracy": 0.2}
        points[f"F:{dataset_name}"] = row
    report = m.build_regression_check(points)
    for dataset_name in m.REC004L_DATASETS:
        entry = report[f"F:{dataset_name}"]
        assert entry["any_regression"] is True
        assert any(r["position"] == 1 for r in entry["regressed_positions"])


def test_regression_check_reports_no_regression_when_nothing_dropped() -> None:
    points = {}
    for condition_id in m.REC004L_CONDITION_IDS:
        for dataset_name in m.REC004L_DATASETS:
            points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
    report = m.build_regression_check(points)
    for entry in report.values():
        assert entry["any_regression"] is False
