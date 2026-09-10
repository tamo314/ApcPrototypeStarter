"""CPU-only contract tests for Task B-C005REC-004M (I03 FFN-Value-Path
Subcomponent Attribution & Freeze-Repair Contract).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real milestone run (reading REC-004D's and REC-004H's real,
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
from apc.evaluation import mirror_ffn_value_path_subcomponent_attribution as m
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
# Constants sanity -- reused from REC-004K/L wherever the recipe is
# unchanged; only the 4-subcomponent grouping, new probe dataset, and
# FFN+subcomponent condition set are new here.
# ---------------------------------------------------------------------------


def test_decisive_checkpoint_identity_reused_verbatim() -> None:
    assert m.REC004M_DECISIVE_INIT == rec004k.REC004K_DECISIVE_INIT == "I03"
    assert m.REC004M_EARLY_STEP == rec004k.REC004K_EARLY_STEP == 6000
    assert m.REC004M_LATE_STEP == rec004k.REC004K_LATE_STEP == 17500
    assert m.REC004M_TARGET_LENGTH == rec004k.REC004K_TARGET_LENGTH == 10


def test_thresholds_reused_verbatim() -> None:
    assert m.REC004M_ROLLBACK_EM_THRESHOLD == rec004k.REC004K_ROLLBACK_EM_THRESHOLD == 0.95
    assert (
        m.REC004M_ROLLBACK_POSITION_ACC_THRESHOLD
        == rec004k.REC004K_ROLLBACK_POSITION_ACC_THRESHOLD
        == 0.95
    )


def test_subcomponent_ids_are_the_fixed_four() -> None:
    assert m.REC004M_SUBCOMPONENT_IDS == (
        "CONTENT_PREP", "V_PROJECTION", "ATTN_OUT_PROJ", "SCORE_PROJECTION_CONTROL",
    )


def test_condition_set_is_the_fixed_eight() -> None:
    assert m.REC004M_CONDITION_IDS == (
        "R0", "F", "F_C", "F_V", "F_O", "F_QK", "F_ALLV", "EARLY",
    )
    assert m.REC004M_PAIR_CONDITION_IDS == ("F_C", "F_V", "F_O", "F_QK")
    assert m.REC004M_CONDITION_TO_SUBCOMPONENTS["R0"] == ()
    assert m.REC004M_CONDITION_TO_SUBCOMPONENTS["F"] == ()
    assert m.REC004M_CONDITION_TO_SUBCOMPONENTS["F_ALLV"] == m.REC004M_SUBCOMPONENT_IDS
    for pair_id in m.REC004M_PAIR_CONDITION_IDS:
        assert len(m.REC004M_CONDITION_TO_SUBCOMPONENTS[pair_id]) == 1
    assert m.REC004M_FFN_ROLLED_BACK["R0"] is False
    assert all(
        m.REC004M_FFN_ROLLED_BACK[cid]
        for cid in m.REC004M_MERGE_CONDITION_IDS
        if cid != "R0"
    )


def test_datasets_include_all_three_existing_sets_plus_exactly_one_new_set() -> None:
    assert m.REC004M_DATASETS == (
        m.REC004M_CLEAN_V2_DATASET, m.REC004M_PROBE_DATASET,
        m.REC004M_DOWNSTREAM_PROBE_DATASET, m.REC004M_NEW_PROBE_SPLIT,
    )
    assert m.REC004M_CLEAN_V2_DATASET == rec004k.REC004K_CLEAN_V2_DATASET
    assert m.REC004M_PROBE_DATASET == rec004k.REC004K_PROBE_DATASET
    assert m.REC004M_DOWNSTREAM_PROBE_DATASET == rec004l.REC004L_NEW_PROBE_SPLIT
    assert m.REC004M_NEW_PROBE_SPLIT == "length10_value_path_subcomponent_probe_v1"
    assert m.REC004M_NEW_PROBE_EXAMPLES == 512


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
    for forbidden in (
        "itertools.combinations", "itertools.product",
        '"F_C_V"', '"F_V_O"', '"F_C_O"', '"F_ALL_TRIPLES"',
    ):
        assert forbidden not in source


def test_module_never_feeds_pi_n_into_a_forward_input() -> None:
    """This task never calls the oracle position map itself -- only the
    already-verified `oracle_probe.run_oracle_forward`/`_oracle_attention`
    do, imported unmodified."""
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "mirror_halves_position_map" not in source


# ---------------------------------------------------------------------------
# Stage A -- key-group partition, self-verified against a real state_dict.
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
        operation=m.REC004M_TARGET_OPERATION, d_model=core.model.config.d_model,
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


def test_key_groups_exactly_partition_value_outproj(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=1)
    keys = list(primitive.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)

    assert set(groups["CONTENT_PREP"]) == {
        "content_in_proj.weight", "content_in_proj.bias", "content_position_embedding.weight",
    }
    assert set(groups["ATTN_OUT_PROJ"]) == {
        "cross_attn.out_proj.weight", "cross_attn.out_proj.bias",
    }
    assert set(groups["IN_PROJ_FUSED_KEYS"]) == {
        "cross_attn.in_proj_weight", "cross_attn.in_proj_bias",
    }
    # Every other REC-004K group passes through unchanged.
    for cid in (
        "QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "FFN_BLOCK", "READOUT", "EXCLUDED_FROM_O1",
    ):
        assert cid in groups
    assert "VALUE_OUTPROJ" not in groups


def test_key_groups_raise_on_an_unaccounted_key() -> None:
    with pytest.raises(AssertionError):
        m.build_value_path_subcomponent_key_groups(
            ["content_in_proj.weight", "some_unexpected_new_key.weight"]
        )


# ---------------------------------------------------------------------------
# Row-slice merge -- V and QK slices must be disjoint, additive, and
# reproduce the whole tensor when both are requested.
# ---------------------------------------------------------------------------


def test_v_slice_replaces_only_the_v_rows() -> None:
    late_t = torch.zeros(9, 3)
    early_t = torch.ones(9, 3)
    merged = {"w": late_t.clone()}
    m._rollback_in_proj_slice(merged, {"w": late_t}, {"w": early_t}, "w", "V")
    assert torch.equal(merged["w"][0:6], torch.zeros(6, 3))  # Q, K rows untouched
    assert torch.equal(merged["w"][6:9], torch.ones(3, 3))  # V rows replaced


def test_qk_slice_replaces_only_the_qk_rows() -> None:
    late_t = torch.zeros(9, 3)
    early_t = torch.ones(9, 3)
    merged = {"w": late_t.clone()}
    m._rollback_in_proj_slice(merged, {"w": late_t}, {"w": early_t}, "w", "QK")
    assert torch.equal(merged["w"][0:6], torch.ones(6, 3))  # Q, K rows replaced
    assert torch.equal(merged["w"][6:9], torch.zeros(3, 3))  # V rows untouched


def test_v_then_qk_slices_compose_to_the_full_early_tensor() -> None:
    late_t = torch.zeros(9, 3)
    early_t = torch.ones(9, 3)
    merged = {"w": late_t.clone()}
    m._rollback_in_proj_slice(merged, {"w": late_t}, {"w": early_t}, "w", "V")
    m._rollback_in_proj_slice(merged, {"w": late_t}, {"w": early_t}, "w", "QK")
    assert torch.equal(merged["w"], early_t)


def test_slice_rejects_a_non_divisible_by_three_shape() -> None:
    late_t = torch.zeros(10, 3)
    early_t = torch.ones(10, 3)
    with pytest.raises(AssertionError):
        m._rollback_in_proj_slice({"w": late_t.clone()}, {"w": late_t}, {"w": early_t}, "w", "V")


def test_slice_rejects_an_unknown_which() -> None:
    late_t = torch.zeros(9, 3)
    early_t = torch.ones(9, 3)
    with pytest.raises(ValueError):
        m._rollback_in_proj_slice(
            {"w": late_t.clone()}, {"w": late_t}, {"w": early_t}, "w", "BOGUS"
        )


# ---------------------------------------------------------------------------
# Condition primitive construction.
# ---------------------------------------------------------------------------


def test_f_condition_rolls_back_only_ffn_block(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    f_primitive = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, (), groups, ffn_rolled_back=True
    )
    f_sd = f_primitive.state_dict()
    for key in groups["FFN_BLOCK"]:
        assert torch.equal(f_sd[key], early_sd[key])
    for key in groups["CONTENT_PREP"] + groups["ATTN_OUT_PROJ"] + groups["IN_PROJ_FUSED_KEYS"]:
        assert torch.equal(f_sd[key], late_sd[key])


def test_r0_condition_is_pure_late_state_with_no_rollback(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    r0_primitive = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, (), groups, ffn_rolled_back=False
    )
    for key, val in r0_primitive.state_dict().items():
        assert torch.equal(val, late_sd[key])


def test_f_v_condition_touches_only_v_rows_of_the_fused_tensor(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    fv_primitive = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("V_PROJECTION",), groups, ffn_rolled_back=True
    )
    fv_sd = fv_primitive.state_dict()
    d = late_sd["cross_attn.in_proj_weight"].shape[0] // 3
    assert torch.equal(
        fv_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
        early_sd["cross_attn.in_proj_weight"][2 * d : 3 * d],
    )
    assert torch.equal(
        fv_sd["cross_attn.in_proj_weight"][0 : 2 * d],
        late_sd["cross_attn.in_proj_weight"][0 : 2 * d],
    )
    # CONTENT_PREP/ATTN_OUT_PROJ untouched by F_V.
    for key in groups["CONTENT_PREP"] + groups["ATTN_OUT_PROJ"]:
        assert torch.equal(fv_sd[key], late_sd[key])


def test_f_allv_reproduces_ffn_plus_whole_value_outproj_rollback(tiny_core_and_bank) -> None:
    """The row-slice-composed F_ALLV must be state-dict-identical to
    REC-004K's own whole-tensor FFN_BLOCK+VALUE_OUTPROJ rollback -- the
    decomposition-completeness invariant this task's own Stage B parity
    gate checks empirically via O1 forward equality."""
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    new_groups = m.build_value_path_subcomponent_key_groups(keys)
    old_groups = rec004k.partition_state_dict_keys(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    f_allv = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, m.REC004M_SUBCOMPONENT_IDS, new_groups, ffn_rolled_back=True
    )
    old_reference = rec004k._build_rollback_primitive(
        core, late_sd, early_sd, ("FFN_BLOCK", "VALUE_OUTPROJ"), old_groups
    )
    for key, val in f_allv.state_dict().items():
        assert torch.equal(val, old_reference.state_dict()[key]), key


def test_early_condition_is_the_real_loaded_primitive_not_a_merge(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    assert primitives["EARLY"] is early


# ---------------------------------------------------------------------------
# QK negative-control invariance -- must be exactly 0.0 diff under O1.
# ---------------------------------------------------------------------------


def test_qk_rollback_is_exactly_o1_invariant(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    n = 10
    torch.manual_seed(5)
    content = torch.randn(4, n, core.model.config.d_model)
    content_lengths = [n] * 4
    output_lengths = [n] * 4
    with torch.no_grad():
        check = m.verify_qk_rollback_is_o1_invariant(
            core, late_sd, early_sd, groups, content, content_lengths, output_lengths
        )
    assert check["invariant"] is True
    assert check["max_abs_logit_diff"] == 0.0
    assert check["predictions_match"] is True


def test_v_rollback_is_generally_not_o1_invariant(tiny_core_and_bank) -> None:
    """Sanity contrast: unlike QK, V rollback IS expected to move O1's
    output in general (it feeds the value vectors oracle attention
    aggregates) -- confirms the invariance check isn't vacuously trivial."""
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}

    f_primitive = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, (), groups, ffn_rolled_back=True
    )
    f_v_primitive = m._build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("V_PROJECTION",), groups, ffn_rolled_back=True
    )
    n = 10
    torch.manual_seed(6)
    content = torch.randn(4, n, core.model.config.d_model)
    content_lengths = [n] * 4
    output_lengths = [n] * 4
    with torch.no_grad():
        f_out = oracle_probe.run_oracle_forward(
            f_primitive, content, content_lengths, output_lengths
        )
        fv_out = oracle_probe.run_oracle_forward(
            f_v_primitive, content, content_lengths, output_lengths
        )
    assert (fv_out["logits"] - f_out["logits"]).abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# New disjoint dataset generation -- pure function of (seed, protected set).
# ---------------------------------------------------------------------------


def test_new_probe_dataset_is_deterministic_and_disjoint_from_a_protected_set() -> None:
    protected = {"deadbeef" * 4}
    examples_a, detail_a = m.build_length10_value_path_subcomponent_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    examples_b, detail_b = m.build_length10_value_path_subcomponent_probe_v1(
        seed=10, protected_digests=protected, n=32
    )
    assert len(examples_a) == len(examples_b) == 32
    assert [e.input_tokens for e in examples_a] == [e.input_tokens for e in examples_b]
    assert [e.target_tokens for e in examples_a] == [e.target_tokens for e in examples_b]
    assert detail_a["development_exposed"] is True
    assert detail_a["sealed_or_rg3_query"] is False


def test_new_probe_dataset_never_draws_a_protected_digest() -> None:
    from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit

    baseline_examples, _ = m.build_length10_value_path_subcomponent_probe_v1(
        seed=11, protected_digests=set(), n=16
    )
    protected = traj_audit._digest_examples(baseline_examples)
    guarded_examples, detail = m.build_length10_value_path_subcomponent_probe_v1(
        seed=11, protected_digests=protected, n=16
    )
    guarded_digests = traj_audit._digest_examples(guarded_examples)
    assert not (guarded_digests & protected)
    assert detail["substitution_count"] >= 0


def test_new_probe_dataset_uses_a_distinct_rng_stream_from_rec004l_probe() -> None:
    ours, _ = m.build_length10_value_path_subcomponent_probe_v1(
        seed=10, protected_digests=set(), n=8
    )
    theirs, _ = rec004l.build_length10_downstream_interaction_probe_v1(
        seed=10, protected_digests=set(), n=8
    )
    assert [e.input_tokens for e in ours] != [e.input_tokens for e in theirs]


# ---------------------------------------------------------------------------
# compute_interaction_diagnostic_point reuse -- structural invariants.
# ---------------------------------------------------------------------------


def test_diagnostic_point_vs_self_reference_has_zero_diff(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    primitive = _tiny_primitive(core, seed=7)
    examples = _tiny_length10_examples(seed=123, n=13)

    with torch.no_grad():
        point = rec004l.compute_interaction_diagnostic_point(core, primitive, primitive, examples)

    assert point["n"] == 13
    assert point["max_abs_hidden_state_diff_vs_early"] == 0.0
    assert point["max_abs_logit_diff_vs_early"] == 0.0


def test_condition_points_never_mutate_any_primitive(tiny_core_and_bank) -> None:
    core, _bank, _op_to_id = tiny_core_and_bank
    late = _tiny_primitive(core, seed=1)
    early = _tiny_primitive(core, seed=2)
    keys = list(late.state_dict().keys())
    groups = m.build_value_path_subcomponent_key_groups(keys)
    late_sd = {k: v.clone() for k, v in late.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early.state_dict().items()}
    primitives = m.build_condition_primitives(core, late_sd, early_sd, groups, early)
    datasets = {ds: _tiny_length10_examples(seed=1, n=4) for ds in m.REC004M_DATASETS}
    hashes_before = {
        cid: mb.canonical_state_hash(p.state_dict()) for cid, p in primitives.items()
    }
    with torch.no_grad():
        m.run_stage_conditions(core, primitives, datasets)
    for cid, p in primitives.items():
        assert mb.canonical_state_hash(p.state_dict()) == hashes_before[cid]


# ---------------------------------------------------------------------------
# Decision rule -- synthetic condition_points, exercising every label branch.
# ---------------------------------------------------------------------------


def _synthetic_point(em: float, pos4: float) -> dict:
    per_position = {str(k): {"oracle_accuracy": 1.0} for k in range(10)}
    per_position["4"] = {"oracle_accuracy": pos4}
    return {"oracle_sequence_exact_match": em, "per_output_position": per_position}


def _full_condition_points(passing_condition_ids: set[str]) -> dict:
    points = {}
    for condition_id in m.REC004M_CONDITION_IDS:
        for dataset_name in m.REC004M_DATASETS:
            if condition_id in passing_condition_ids:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
            else:
                points[f"{condition_id}:{dataset_name}"] = _synthetic_point(0.5, 0.5)
    return points


_PARITY_OK = {"status": "VERIFIED", "per_dataset": {}}
_PARITY_FAILED = {"status": "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP", "per_dataset": {}}


def test_decision_exactly_one_subcomponent_sufficient() -> None:
    points = _full_condition_points({"F_V"})
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "FFN_V_PROJECTION_INTERACTION_SUFFICIENT"
    assert decision["sufficient_condition_ids"] == ["F_V"]
    assert decision["sufficient_subcomponents"] == ["V_PROJECTION"]


def test_decision_multiple_subcomponents_sufficient_does_not_pick_highest_em() -> None:
    points = _full_condition_points({"F_C", "F_O"})
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "MULTIPLE_VALUE_SUBPATHS_SUFFICIENT"
    assert set(decision["sufficient_condition_ids"]) == {"F_C", "F_O"}


def test_decision_no_subcomponent_sufficient_but_f_allv_passes() -> None:
    points = _full_condition_points({"F_ALLV"})
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "DISTRIBUTED_WITHIN_VALUE_PATH"
    assert decision["sufficient_condition_ids"] == []
    assert decision["f_allv_passes"] is True
    assert "content_prep_shared_with_score_path_caveat" in decision


def test_decision_nothing_passes() -> None:
    points = _full_condition_points(set())
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "VALUE_PATH_DECOMPOSITION_INCOMPLETE"


def test_decision_parity_failure_stops_before_any_interpretation() -> None:
    points = _full_condition_points({"F_C", "F_V", "F_O", "F_QK", "F_ALLV"})
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_FAILED)
    assert decision["label"] == "VALUE_PATH_DECOMPOSITION_PARITY_FAILED_STOP"
    assert decision["sufficient_subcomponents"] == []


def test_decision_requires_all_four_datasets_to_pass() -> None:
    points = _full_condition_points({"F_O"})
    points[f"F_O:{m.REC004M_NEW_PROBE_SPLIT}"] = _synthetic_point(0.5, 0.5)
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "VALUE_PATH_DECOMPOSITION_INCOMPLETE"


def test_negative_control_subcomponent_can_in_principle_yield_its_own_label() -> None:
    """Purely a decision-function contract check: the label formatter does
    not special-case F_QK -- if it were ever the sole passer (which the
    invariance check should make mathematically impossible in the real
    run), it would report a normal SCORE_PROJECTION_CONTROL label rather
    than silently dropping or misnaming it."""
    points = _full_condition_points({"F_QK"})
    decision = m.build_value_path_subcomponent_decision(points, _PARITY_OK)
    assert decision["label"] == "FFN_SCORE_PROJECTION_CONTROL_INTERACTION_SUFFICIENT"


# ---------------------------------------------------------------------------
# Regression check -- descriptive only, never gates the decision above.
# ---------------------------------------------------------------------------


def test_regression_check_flags_a_position_that_got_worse_than_r0() -> None:
    points = {}
    for dataset_name in m.REC004M_DATASETS:
        points[f"R0:{dataset_name}"] = _synthetic_point(1.0, 1.0)
        row = _synthetic_point(1.0, 1.0)
        row["per_output_position"]["1"] = {"oracle_accuracy": 0.2}
        points[f"F:{dataset_name}"] = row
    report = m.build_regression_check(points)
    for dataset_name in m.REC004M_DATASETS:
        entry = report[f"F:{dataset_name}"]
        assert entry["any_regression"] is True
        assert any(r["position"] == 1 for r in entry["regressed_positions"])


def test_regression_check_reports_no_regression_when_nothing_dropped() -> None:
    points = {}
    for condition_id in m.REC004M_CONDITION_IDS:
        for dataset_name in m.REC004M_DATASETS:
            points[f"{condition_id}:{dataset_name}"] = _synthetic_point(1.0, 1.0)
    report = m.build_regression_check(points)
    for entry in report.values():
        assert entry["any_regression"] is False


# ---------------------------------------------------------------------------
# Stage E safety-check gating -- only runs when Stage D localizes to one.
# ---------------------------------------------------------------------------


def test_safety_check_not_executed_when_multiple_subcomponents_sufficient() -> None:
    decision = {
        "label": "MULTIPLE_VALUE_SUBPATHS_SUFFICIENT",
        "sufficient_condition_ids": ["F_C", "F_O"],
    }
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT"


def test_safety_check_not_executed_when_distributed() -> None:
    decision = {"label": "DISTRIBUTED_WITHIN_VALUE_PATH", "sufficient_condition_ids": []}
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT"


def test_safety_check_not_executed_when_incomplete() -> None:
    decision = {"label": "VALUE_PATH_DECOMPOSITION_INCOMPLETE", "sufficient_condition_ids": []}
    result = m.run_safety_check(core=None, decision=decision, datasets={})
    assert result["status"] == "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT"


def test_safety_targets_are_i04_and_i05_own_late_steps() -> None:
    assert m.REC004M_SAFETY_TARGETS == (("I04", 18000), ("I05", 17500))


# ---------------------------------------------------------------------------
# next_training_repair_contract.md -- proposal gating.
# ---------------------------------------------------------------------------


def test_contract_not_proposed_when_multiple_sufficient() -> None:
    decision = {
        "label": "MULTIPLE_VALUE_SUBPATHS_SUFFICIENT",
        "sufficient_condition_ids": ["F_C", "F_O"],
        "sufficient_subcomponents": ["CONTENT_PREP", "ATTN_OUT_PROJ"],
    }
    not_executed = {"status": "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT"}
    text = m.build_next_training_repair_contract(decision, not_executed)
    assert "status: NOT_PROPOSED" in text


def test_contract_not_proposed_when_distributed_and_warns_against_full_freeze() -> None:
    decision = {
        "label": "DISTRIBUTED_WITHIN_VALUE_PATH",
        "sufficient_condition_ids": [],
        "sufficient_subcomponents": [],
    }
    not_executed = {"status": "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT"}
    text = m.build_next_training_repair_contract(decision, not_executed)
    assert "status: NOT_PROPOSED" in text
    assert "freezing the entire" in text.lower() or "freezing it" in text.lower()


def test_contract_proposed_when_localized_and_safe() -> None:
    decision = {
        "label": "FFN_V_PROJECTION_INTERACTION_SUFFICIENT",
        "sufficient_condition_ids": ["F_V"],
        "sufficient_subcomponents": ["V_PROJECTION"],
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
            "I04": {"late_step": 18000, **no_degradation},
            "I05": {"late_step": 17500, **no_degradation},
        },
    }
    text = m.build_next_training_repair_contract(decision, safety_check)
    assert "status: PROPOSED_NOT_AUTHORIZED" in text
    assert "Phase 1:" in text and "Phase 2:" in text
    assert "V_PROJECTION" in text


def test_contract_still_proposed_but_flags_caution_when_degraded() -> None:
    decision = {
        "label": "FFN_ATTN_OUT_PROJ_INTERACTION_SUFFICIENT",
        "sufficient_condition_ids": ["F_O"],
        "sufficient_subcomponents": ["ATTN_OUT_PROJ"],
    }
    degraded = {
        "classification": "DEGRADATION_OBSERVED",
        "max_j0_em_drop": 0.3,
        "degradation_threshold": 0.05,
    }
    not_degraded = {
        "classification": "NO_DEGRADATION_OBSERVED",
        "max_j0_em_drop": 0.0,
        "degradation_threshold": 0.05,
    }
    safety_check = {
        "status": "EXECUTED",
        "any_degradation_observed": True,
        "per_init": {
            "I04": {"late_step": 18000, **degraded},
            "I05": {"late_step": 17500, **not_degraded},
        },
    }
    text = m.build_next_training_repair_contract(decision, safety_check)
    assert "status: PROPOSED_NOT_AUTHORIZED" in text
    assert "Caution" in text
