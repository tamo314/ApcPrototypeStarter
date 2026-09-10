"""CPU-only contract tests for Task B-C005REC-004N (I03 FFN-Value-Path
Leave-One-Out Necessity Audit & Freezeability Gate).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real milestone run (reading REC-004D's/REC-004H's real,
multi-gigabyte `run_001` trees for I03/I04/I05) is never invoked from this
test module. These tests exercise the real production functions either as
pure-Python unit tests or against small, synthetic "REC-004D/H-shaped"
fixtures built in-process.
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
from apc.evaluation import mirror_ffn_anchored_downstream_interaction_audit as rec004l
from apc.evaluation import mirror_ffn_value_path_leave_one_out_necessity_audit as m
from apc.evaluation import mirror_ffn_value_path_subcomponent_attribution as rec004m
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

# ---------------------------------------------------------------------------
# Constants sanity -- reused from REC-004K/L/M wherever the recipe is
# unchanged; only the leave-one-out condition set, the new probe dataset,
# the source-replay check, and the J0 attention-equivalence check are new
# here.
# ---------------------------------------------------------------------------


def test_decisive_checkpoint_identity_reused_verbatim() -> None:
    assert m.REC004N_DECISIVE_INIT == rec004m.REC004M_DECISIVE_INIT == "I03"
    assert m.REC004N_EARLY_STEP == rec004m.REC004M_EARLY_STEP == 6000
    assert m.REC004N_LATE_STEP == rec004m.REC004M_LATE_STEP == 17500
    assert m.REC004N_TARGET_LENGTH == rec004m.REC004M_TARGET_LENGTH == 10


def test_thresholds_reused_verbatim() -> None:
    assert m.REC004N_ROLLBACK_EM_THRESHOLD == rec004m.REC004M_ROLLBACK_EM_THRESHOLD == 0.95
    assert (
        m.REC004N_ROLLBACK_POSITION_ACC_THRESHOLD
        == rec004m.REC004M_ROLLBACK_POSITION_ACC_THRESHOLD
        == 0.95
    )


def test_condition_set_is_the_fixed_seven() -> None:
    assert m.REC004N_CONDITION_IDS == ("R0", "F", "F_CV", "F_CO", "F_VO", "F_CVO", "EARLY")
    assert m.REC004N_LEAVE_ONE_OUT_CONDITION_IDS == ("F_CV", "F_CO", "F_VO")
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["R0"] == ()
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["F"] == ()
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["F_CV"] == ("CONTENT_PREP", "V_PROJECTION")
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["F_CO"] == ("CONTENT_PREP", "ATTN_OUT_PROJ")
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["F_VO"] == ("V_PROJECTION", "ATTN_OUT_PROJ")
    assert m.REC004N_CONDITION_TO_SUBCOMPONENTS["F_CVO"] == (
        "CONTENT_PREP", "V_PROJECTION", "ATTN_OUT_PROJ",
    )
    assert m.REC004N_FFN_ROLLED_BACK["R0"] is False
    assert all(
        m.REC004N_FFN_ROLLED_BACK[cid] for cid in m.REC004N_MERGE_CONDITION_IDS if cid != "R0"
    )


def test_leave_one_out_excludes_map_is_correct() -> None:
    assert m.REC004N_LEAVE_ONE_OUT_EXCLUDES == {
        "F_CV": "ATTN_OUT_PROJ",
        "F_CO": "V_PROJECTION",
        "F_VO": "CONTENT_PREP",
    }


def test_datasets_include_all_four_existing_sets_plus_exactly_one_new_set() -> None:
    assert m.REC004N_DATASETS == (
        m.REC004N_CLEAN_V2_DATASET, m.REC004N_MECHANISM_PROBE_DATASET,
        m.REC004N_DOWNSTREAM_PROBE_DATASET, m.REC004N_SUBCOMPONENT_PROBE_DATASET,
        m.REC004N_NEW_PROBE_SPLIT,
    )
    assert m.REC004N_CLEAN_V2_DATASET == rec004k.REC004K_CLEAN_V2_DATASET
    assert m.REC004N_MECHANISM_PROBE_DATASET == rec004k.REC004K_PROBE_DATASET
    assert m.REC004N_DOWNSTREAM_PROBE_DATASET == rec004l.REC004L_NEW_PROBE_SPLIT
    assert m.REC004N_SUBCOMPONENT_PROBE_DATASET == rec004m.REC004M_NEW_PROBE_SPLIT
    assert m.REC004N_NEW_PROBE_SPLIT == "length10_value_path_necessity_probe_v1"
    assert m.REC004N_NEW_PROBE_EXAMPLES == 512


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


def test_module_never_searches_beyond_the_fixed_condition_set() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "itertools.combinations" not in source
    assert "itertools.product" not in source
    # REC-004M's own single-subcomponent condition IDs (F_C/F_V/F_O/F_QK/
    # F_ALLV) are never recomputed as NEW Stage B conditions here -- "F_ALLV"
    # appears exactly once, only as a read-only cross-check label mapping
    # this task's own F_CVO to REC-004M's saved F_ALLV artifact.
    for forbidden in ("F_C", "F_V", "F_O", "F_QK"):
        assert forbidden not in m.REC004N_CONDITION_IDS
        assert forbidden not in m.REC004N_MERGE_CONDITION_IDS
    assert "F_ALLV" not in m.REC004N_CONDITION_IDS
    assert "F_ALLV" not in m.REC004N_MERGE_CONDITION_IDS


def test_module_never_feeds_pi_n_into_a_forward_input() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "mirror_halves_position_map" not in source


# ---------------------------------------------------------------------------
# New disjoint dataset generation -- pure function of (seed, protected set).
# ---------------------------------------------------------------------------


def test_new_probe_dataset_is_deterministic_and_disjoint_from_a_protected_set() -> None:
    protected = {"deadbeef" * 4}
    examples_a, detail_a = m.build_length10_value_path_necessity_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    examples_b, detail_b = m.build_length10_value_path_necessity_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    assert len(examples_a) == len(examples_b) == 32
    assert [e.input_tokens for e in examples_a] == [e.input_tokens for e in examples_b]
    assert [e.target_tokens for e in examples_a] == [e.target_tokens for e in examples_b]
    assert detail_a["development_exposed"] is True
    assert detail_a["sealed_or_rg3_query"] is False


def test_new_probe_dataset_never_draws_a_protected_digest() -> None:
    from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit

    baseline_examples, _ = m.build_length10_value_path_necessity_probe_v1(
        seed=11, protected_digests=set(), n=16
    )
    protected = traj_audit._digest_examples(baseline_examples)
    guarded_examples, detail = m.build_length10_value_path_necessity_probe_v1(
        seed=11, protected_digests=protected, n=16
    )
    guarded_digests = traj_audit._digest_examples(guarded_examples)
    assert not (guarded_digests & protected)
    assert detail["substitution_count"] >= 0


def test_new_probe_dataset_uses_a_distinct_rng_stream_from_rec004m_probe() -> None:
    ours, _ = m.build_length10_value_path_necessity_probe_v1(
        seed=10, protected_digests=set(), n=8
    )
    theirs, _ = rec004m.build_length10_value_path_subcomponent_probe_v1(
        seed=10, protected_digests=set(), n=8
    )
    assert [e.input_tokens for e in ours] != [e.input_tokens for e in theirs]


# ---------------------------------------------------------------------------
# Fixtures for real-primitive tests.
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
        operation=m.REC004N_TARGET_OPERATION, d_model=core.model.config.d_model,
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


def _build_condition_pair(core, *, late_seed: int, early_seed: int):
    late = _tiny_primitive(core, seed=late_seed)
    early = _tiny_primitive(core, seed=early_seed)
    keys = list(late.state_dict().keys())
    groups = rec004m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    return late, early, groups, late_sd, early_sd


# ---------------------------------------------------------------------------
# Condition primitive construction delegates to REC-004M's own merge
# function unmodified -- confirm the leave-one-out combinations compose
# correctly (no new merge logic is written for this task).
# ---------------------------------------------------------------------------


def test_f_vo_condition_touches_v_and_o_but_not_content_prep(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    f_vo_sd = primitives["F_VO"].state_dict()

    for key in groups["CONTENT_PREP"]:
        assert torch.equal(f_vo_sd[key], late_sd[key])
    for key in groups["FFN_BLOCK"]:
        assert torch.equal(f_vo_sd[key], early_sd[key])
    for key in groups["ATTN_OUT_PROJ"]:
        assert torch.equal(f_vo_sd[key], early_sd[key])
    d = late_sd["cross_attn.in_proj_weight"].shape[0] // 3
    assert torch.equal(
        f_vo_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
        early_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
    )
    assert torch.equal(
        f_vo_sd["cross_attn.in_proj_weight"][0 : 2 * d],
        late_sd["cross_attn.in_proj_weight"][0 : 2 * d],
    )


def test_f_cvo_condition_reproduces_all_three_subcomponents_rolled_back(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    f_cvo_sd = primitives["F_CVO"].state_dict()
    for key in groups["CONTENT_PREP"] + groups["ATTN_OUT_PROJ"] + groups["FFN_BLOCK"]:
        assert torch.equal(f_cvo_sd[key], early_sd[key])
    d = late_sd["cross_attn.in_proj_weight"].shape[0] // 3
    assert torch.equal(
        f_cvo_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
        early_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
    )
    # Q/K rows are never touched by F_CVO (SCORE_PROJECTION_CONTROL excluded).
    assert torch.equal(
        f_cvo_sd["cross_attn.in_proj_weight"][0 : 2 * d],
        late_sd["cross_attn.in_proj_weight"][0 : 2 * d],
    )


def test_r0_and_early_conditions_are_pure_states(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    for key, val in primitives["R0"].state_dict().items():
        assert torch.equal(val, late_sd[key])
    assert primitives["EARLY"] is early


# ---------------------------------------------------------------------------
# Stage C -- J0 attention equivalence: the core mechanistic claim. Two
# primitives sharing CONTENT_PREP/Q,K/position-bias/QUERY_RESIDUAL_PATH must
# produce IDENTICAL J0 attention regardless of FFN/V_PROJECTION/ATTN_OUT_PROJ
# differences; a CONTENT_PREP difference must break the invariance.
# ---------------------------------------------------------------------------


def test_j0_attention_equivalence_is_exactly_zero_when_only_v_o_ffn_differ(
    tiny_core_and_bank,
) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    f_vo_primitive = rec004m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("V_PROJECTION", "ATTN_OUT_PROJ"), groups, ffn_rolled_back=True
    )
    n = 10
    torch.manual_seed(21)
    content = torch.randn(4, n, core.model.config.d_model)
    content_lengths = [n] * 4
    output_lengths = [n] * 4
    with torch.no_grad():
        check = m.compute_j0_attention_equivalence(
            f_vo_primitive, late, content, content_lengths, output_lengths
        )
    assert check["max_abs_diff_query"] == 0.0
    assert check["max_abs_diff_kv"] == 0.0
    assert check["max_abs_diff_position_bias"] == 0.0
    assert check["max_abs_diff_raw_qk_score"] == 0.0
    assert check["max_abs_diff_attn_probs"] == 0.0
    assert check["attention_score_path_invariant"] is True
    assert check["bitwise_exact"] is True


def test_j0_attention_equivalence_is_nonzero_when_content_prep_differs(
    tiny_core_and_bank,
) -> None:
    """Sanity contrast: F_CV rolls CONTENT_PREP back too -- unlike F_VO, its
    J0 attention MUST differ from R0's, confirming the check isn't vacuous."""
    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    f_cv_primitive = rec004m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("CONTENT_PREP", "V_PROJECTION"), groups, ffn_rolled_back=True
    )
    n = 10
    torch.manual_seed(22)
    content = torch.randn(4, n, core.model.config.d_model)
    content_lengths = [n] * 4
    output_lengths = [n] * 4
    with torch.no_grad():
        check = m.compute_j0_attention_equivalence(
            f_cv_primitive, late, content, content_lengths, output_lengths
        )
    assert check["max_abs_diff_kv"] > 0.0
    assert check["max_abs_diff_attn_probs"] > 0.0
    assert check["attention_score_path_invariant"] is False
    assert check["bitwise_exact"] is False


def test_j0_attention_equivalence_is_zero_against_self(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=5)
    n = 10
    torch.manual_seed(23)
    content = torch.randn(3, n, core.model.config.d_model)
    content_lengths = [n] * 3
    output_lengths = [n] * 3
    with torch.no_grad():
        check = m.compute_j0_attention_equivalence(
            primitive, primitive, content, content_lengths, output_lengths
        )
    assert check["max_diff_overall"] == 0.0
    assert check["bitwise_exact"] is True


def test_j0_attention_check_never_mutates_either_primitive(tiny_core_and_bank) -> None:
    from apc.utils import model_bundle as mb

    core, _bank, _op_to_id = tiny_core_and_bank
    late, early, groups, late_sd, early_sd = _build_condition_pair(
        core, late_seed=1, early_seed=2
    )
    f_vo_primitive = rec004m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("V_PROJECTION", "ATTN_OUT_PROJ"), groups, ffn_rolled_back=True
    )
    hash_before_a = mb.canonical_state_hash(f_vo_primitive.state_dict())
    hash_before_b = mb.canonical_state_hash(late.state_dict())
    n = 10
    torch.manual_seed(24)
    content = torch.randn(2, n, core.model.config.d_model)
    with torch.no_grad():
        m.compute_j0_attention_equivalence(f_vo_primitive, late, content, [n, n], [n, n])
    assert mb.canonical_state_hash(f_vo_primitive.state_dict()) == hash_before_a
    assert mb.canonical_state_hash(late.state_dict()) == hash_before_b


# ---------------------------------------------------------------------------
# Decision rule -- synthetic condition_points, exercising every branch.
# ---------------------------------------------------------------------------


def _synthetic_point(em: float, pos4: float) -> dict:
    per_position = {str(k): {"oracle_accuracy": 1.0} for k in range(10)}
    per_position["4"] = {"oracle_accuracy": pos4}
    return {"oracle_sequence_exact_match": em, "per_output_position": per_position}


def _full_condition_points(passing_condition_ids: set[str]) -> dict:
    points = {}
    for condition_id in m.REC004N_CONDITION_IDS:
        for dataset_name in m.REC004N_DATASETS:
            if condition_id in passing_condition_ids:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
            else:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(0.5, 0.5)
    return points


def test_decision_f_vo_passes_is_the_most_useful_result() -> None:
    points = _full_condition_points({"F_VO"})
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"
    assert "CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY" in decision["tags"]


def test_decision_f_vo_priority_wins_even_if_others_also_pass() -> None:
    points = _full_condition_points({"F_VO", "F_CV", "F_CVO"})
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"
    assert decision["per_condition_passes"]["F_CV"] is True


def test_decision_shared_content_prep_required_when_only_f_cv_passes() -> None:
    points = _full_condition_points({"F_CV", "F_CVO"})
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "SHARED_CONTENT_PREP_REQUIRED"
    assert "FREEZE_REPAIR_NOT_SCORE_SAFE" in decision["tags"]


def test_decision_shared_content_prep_required_when_only_f_co_passes() -> None:
    points = _full_condition_points({"F_CO", "F_CVO"})
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "SHARED_CONTENT_PREP_REQUIRED"


def test_decision_all_three_jointly_necessary_when_only_f_cvo_passes() -> None:
    points = _full_condition_points({"F_CVO"})
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN"
    assert decision["tags"] == []


def test_decision_incomplete_when_nothing_passes() -> None:
    points = _full_condition_points(set())
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] == "VALUE_PATH_NECESSITY_INCOMPLETE"


def test_decision_requires_all_five_datasets_to_pass() -> None:
    points = _full_condition_points({"F_VO"})
    points[f"F_VO:{m.REC004N_NEW_PROBE_SPLIT}"] = _synthetic_point(0.5, 0.5)
    decision = m.build_value_path_necessity_decision(points)
    assert decision["label"] != "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"


def test_decision_records_caveat_regardless_of_label() -> None:
    for passing in (set(), {"F_VO"}, {"F_CV"}, {"F_CVO"}):
        decision = m.build_value_path_necessity_decision(_full_condition_points(passing))
        assert "content_prep_shared_with_score_path_caveat" in decision


# ---------------------------------------------------------------------------
# SCORE_PATH_PRESERVED tag -- only when F_VO passes AND J0 attention holds.
# ---------------------------------------------------------------------------


def test_score_path_preserved_tag_added_when_both_conditions_hold() -> None:
    decision = {
        "label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT",
        "tags": ["CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY"],
    }
    j0_check = {"attention_score_path_invariant": True}
    updated = m.apply_score_path_preserved_tag(decision, j0_check)
    assert "SCORE_PATH_PRESERVED" in updated["tags"]


def test_score_path_preserved_tag_not_added_when_attention_check_fails() -> None:
    decision = {"label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT", "tags": []}
    j0_check = {"attention_score_path_invariant": False}
    updated = m.apply_score_path_preserved_tag(decision, j0_check)
    assert "SCORE_PATH_PRESERVED" not in updated["tags"]


def test_score_path_preserved_tag_not_added_for_other_labels() -> None:
    decision = {"label": "SHARED_CONTENT_PREP_REQUIRED", "tags": ["FREEZE_REPAIR_NOT_SCORE_SAFE"]}
    j0_check = {"attention_score_path_invariant": True}
    updated = m.apply_score_path_preserved_tag(decision, j0_check)
    assert "SCORE_PATH_PRESERVED" not in updated["tags"]


def test_score_path_preserved_tag_does_not_mutate_input_decision() -> None:
    decision = {"label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT", "tags": []}
    m.apply_score_path_preserved_tag(decision, {"attention_score_path_invariant": True})
    assert decision["tags"] == []


# ---------------------------------------------------------------------------
# Safety-check gating -- only runs when F_VO passes.
# ---------------------------------------------------------------------------


def test_safety_check_not_executed_when_shared_content_prep_required() -> None:
    decision = {"label": "SHARED_CONTENT_PREP_REQUIRED"}
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_F_VO_DID_NOT_PASS"


def test_safety_check_not_executed_when_all_three_jointly_necessary() -> None:
    decision = {"label": "ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN"}
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_F_VO_DID_NOT_PASS"


def test_safety_check_not_executed_when_incomplete() -> None:
    decision = {"label": "VALUE_PATH_NECESSITY_INCOMPLETE"}
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_F_VO_DID_NOT_PASS"


def test_safety_targets_are_i04_and_i05_own_late_steps() -> None:
    assert m.REC004N_SAFETY_TARGETS == (("I04", 18000), ("I05", 17500))


# ---------------------------------------------------------------------------
# Regression check -- descriptive only, never gates the decision above.
# ---------------------------------------------------------------------------


def test_regression_check_flags_a_position_that_got_worse_than_r0() -> None:
    points = {}
    for dataset_name in m.REC004N_DATASETS:
        points[f"R0:{dataset_name}"] = _synthetic_point(1.0, 1.0)
        row = _synthetic_point(1.0, 1.0)
        row["per_output_position"]["1"] = {"oracle_accuracy": 0.2}
        points[f"F_VO:{dataset_name}"] = row
    report = m.build_regression_check(points)
    for dataset_name in m.REC004N_DATASETS:
        entry = report[f"F_VO:{dataset_name}"]
        assert entry["any_regression"] is True
        assert any(r["position"] == 1 for r in entry["regressed_positions"])


def test_regression_check_reports_no_regression_when_nothing_dropped() -> None:
    points = {}
    for condition_id in m.REC004N_CONDITION_IDS:
        for dataset_name in m.REC004N_DATASETS:
            points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
    report = m.build_regression_check(points)
    for entry in report.values():
        assert entry["any_regression"] is False


# ---------------------------------------------------------------------------
# Source replay -- cross-check gating.
# ---------------------------------------------------------------------------


def test_cross_check_reports_artifact_unavailable_when_rec004m_run_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "REC004M_RUN_DIR", Path("does/not/exist"))
    result = m.cross_check_against_rec004m({})
    assert result["status"] == "REC004M_ARTIFACT_UNAVAILABLE"


def test_source_replay_mismatch_when_f_cvo_not_perfect() -> None:
    replay_points = {}
    for cid in ("R0", "F", "F_CVO", "EARLY"):
        for ds in m.REC004N_EXISTING_DATASETS:
            em = 0.88 if cid == "F" else (1.0 if cid != "F_CVO" else 0.9)
            replay_points[f"{cid}:{ds}"] = _synthetic_point(em, em)
    qk_check = {"invariant": True, "max_abs_logit_diff": 0.0}
    result = m.run_source_replay(replay_points, qk_check)
    assert result["status"] == "SOURCE_REPLAY_MISMATCH"
    assert result["f_cvo_em_is_perfect"] is False


def test_source_replay_mismatch_when_qk_not_invariant() -> None:
    replay_points = {}
    for cid in ("R0", "F", "F_CVO", "EARLY"):
        for ds in m.REC004N_EXISTING_DATASETS:
            em = 0.88 if cid == "F" else 1.0
            replay_points[f"{cid}:{ds}"] = _synthetic_point(em, em)
    qk_check = {"invariant": False, "max_abs_logit_diff": 0.5}
    result = m.run_source_replay(replay_points, qk_check)
    assert result["status"] == "SOURCE_REPLAY_MISMATCH"
    assert result["qk_rollback_invariant"] is False


def test_source_replay_mismatch_when_f_out_of_range() -> None:
    replay_points = {}
    for cid in ("R0", "F", "F_CVO", "EARLY"):
        for ds in m.REC004N_EXISTING_DATASETS:
            em = 0.5 if cid == "F" else 1.0
            replay_points[f"{cid}:{ds}"] = _synthetic_point(em, em)
    qk_check = {"invariant": True, "max_abs_logit_diff": 0.0}
    result = m.run_source_replay(replay_points, qk_check)
    assert result["status"] == "SOURCE_REPLAY_MISMATCH"
    assert result["f_em_in_expected_range"] is False


# ---------------------------------------------------------------------------
# next_step_repair_contract.md -- proposal gating.
# ---------------------------------------------------------------------------


def test_contract_not_proposed_when_shared_content_prep_required() -> None:
    decision = {"label": "SHARED_CONTENT_PREP_REQUIRED", "tags": ["FREEZE_REPAIR_NOT_SCORE_SAFE"]}
    j0_check = {
        "attention_score_path_invariant": False, "max_diff_overall": 1.0, "bitwise_exact": False,
    }
    not_executed = {"status": "NOT_EXECUTED_F_VO_DID_NOT_PASS"}
    text = m.build_next_step_repair_contract(decision, j0_check, not_executed)
    assert "status: NOT_PROPOSED" in text
    assert "architecture-level" in text.lower()


def test_contract_not_proposed_when_all_three_jointly_necessary() -> None:
    decision = {"label": "ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN", "tags": []}
    j0_check = {
        "attention_score_path_invariant": False, "max_diff_overall": 1.0, "bitwise_exact": False,
    }
    not_executed = {"status": "NOT_EXECUTED_F_VO_DID_NOT_PASS"}
    text = m.build_next_step_repair_contract(decision, j0_check, not_executed)
    assert "status: NOT_PROPOSED" in text
    assert "jointly necessary" in text.lower()


def test_contract_not_proposed_when_f_vo_passes_but_attention_check_fails() -> None:
    decision = {
        "label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT",
        "tags": ["CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY"],
    }
    j0_check = {
        "attention_score_path_invariant": False, "max_diff_overall": 0.5, "bitwise_exact": False,
    }
    not_executed = {"status": "NOT_EXECUTED_F_VO_DID_NOT_PASS"}
    text = m.build_next_step_repair_contract(decision, j0_check, not_executed)
    assert "status: NOT_PROPOSED" in text


def test_contract_proposed_when_f_vo_passes_and_score_path_preserved() -> None:
    decision = {
        "label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT",
        "tags": ["CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY", "SCORE_PATH_PRESERVED"],
    }
    j0_check = {
        "attention_score_path_invariant": True, "max_diff_overall": 0.0, "bitwise_exact": True,
    }
    no_degradation = {
        "classification": "NO_DEGRADATION_OBSERVED",
        "max_j0_em_drop": 0.0,
        "degradation_threshold": 0.05,
    }
    safety_check = {
        "status": "EXECUTED",
        "any_degradation_observed": False,
        "per_init": {
            "I04": {
                "late_step": 18000, "j0_attention_equivalence_check": j0_check, **no_degradation,
            },
            "I05": {
                "late_step": 17500, "j0_attention_equivalence_check": j0_check, **no_degradation,
            },
        },
    }
    text = m.build_next_step_repair_contract(decision, j0_check, safety_check)
    assert "status: PROPOSED_NOT_AUTHORIZED" in text
    assert "Phase 1:" in text and "Phase 2:" in text
    assert "V_PROJECTION" in text and "ATTN_OUT_PROJ" in text
    assert "B-C005REC-004O" in text


def test_contract_flags_caution_when_safety_check_shows_degradation() -> None:
    decision = {
        "label": "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT",
        "tags": ["CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY", "SCORE_PATH_PRESERVED"],
    }
    j0_check = {
        "attention_score_path_invariant": True, "max_diff_overall": 0.0, "bitwise_exact": True,
    }
    degraded = {
        "classification": "DEGRADATION_OBSERVED",
        "max_j0_em_drop": 0.3,
        "degradation_threshold": 0.05,
    }
    safety_check = {
        "status": "EXECUTED",
        "any_degradation_observed": True,
        "per_init": {
            "I04": {"late_step": 18000, "j0_attention_equivalence_check": j0_check, **degraded},
        },
    }
    text = m.build_next_step_repair_contract(decision, j0_check, safety_check)
    assert "status: PROPOSED_NOT_AUTHORIZED" in text
    assert "Caution" in text
