"""Tests for the counterfactual COUNT gate (Phase A.1 Post-Correction Task
A1-R005D-004, STOP GATE).

Fast, small-step tests only -- mirroring `tests/
test_parameterized_primitive_gate.py`'s convention of exercising the wiring
(group construction, config validation, primitive-bank construction, report
shape, multi-seed aggregation) with a tiny model and a handful of training
steps. The gate's own scientific claim (5 seeds, the full training budget)
is run via `scripts/count_counterfactual_gate.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.environments.task_spec import operation_id
from apc.evaluation.count_counterfactual_gate import (
    CORRECT_THRESHOLD,
    DEFAULT_GROUP_SIZE,
    DEFAULT_SEEDS,
    EFFECTFUL_WRONG_ARGUMENT_CEILING,
    MIN_CAUSAL_GAP,
    MIN_GATE_SEEDS,
    MIN_GROUP_SIZE,
    NONE_CEILING,
    CountCounterfactualGateConfig,
    CountCounterfactualGateReport,
    CountCounterfactualGroup,
    PrimitiveTrainConfig,
    _argument_and_content_path_params,
    _build_primitive_bank,
    _flatten_groups,
    _grad_norm,
    count_counterfactual_gate_config_from_dict,
    generate_count_counterfactual_groups,
    run_count_counterfactual_gate,
    run_count_counterfactual_gate_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig
from apc.primitives.conditioning import (
    BasisModulatedConditionedPrimitive,
    ConditionedPrimitive,
    ConditioningVariant,
    FiLMConditionedPrimitive,
)
from apc.primitives.primitive import PrimitiveStatus


def _tiny_config(**overrides: object) -> CountCounterfactualGateConfig:
    base = CountCounterfactualGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        model={
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        core_train=SharedCoreGateTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        primitive_rank=4,
        arg_dim=6,
        max_sequence_length=8,
        primitive_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- generate_count_counterfactual_groups / CountCounterfactualGroup ----------


def test_generated_groups_have_pairwise_distinct_outputs() -> None:
    groups = generate_count_counterfactual_groups(0, 50, step=0, split="train")
    for group in groups:
        outputs = [example.target_tokens for example in group.examples]
        assert len(set(outputs)) == len(outputs)


def test_generated_groups_share_the_same_content() -> None:
    groups = generate_count_counterfactual_groups(0, 20, step=0, split="train")
    for group in groups:
        for example in group.examples:
            assert example.input_tokens == group.input_tokens


def test_generated_groups_meet_minimum_size() -> None:
    groups = generate_count_counterfactual_groups(0, 200, step=0, split="train")
    assert all(len(group.examples) >= MIN_GROUP_SIZE for group in groups)


def test_generated_groups_default_to_the_documented_group_size_most_of_the_time() -> None:
    groups = generate_count_counterfactual_groups(0, 500, step=0, split="train")
    at_default_size = sum(1 for group in groups if len(group.examples) == DEFAULT_GROUP_SIZE)
    # Empirically ~95% for vocab_size=10, sequence_length_range=(6, 10) (see
    # module docstring); a loose floor keeps this a regression guard, not a
    # flaky exact match.
    assert at_default_size / len(groups) > 0.5


def test_generate_groups_is_deterministic_given_the_same_arguments() -> None:
    first = generate_count_counterfactual_groups(0, 10, step=3, split="train")
    second = generate_count_counterfactual_groups(0, 10, step=3, split="train")
    assert [g.input_tokens for g in first] == [g.input_tokens for g in second]
    assert [g.targets for g in first] == [g.targets for g in second]


def test_generate_groups_differ_by_step() -> None:
    first = generate_count_counterfactual_groups(0, 10, step=0, split="train")
    second = generate_count_counterfactual_groups(0, 10, step=1, split="train")
    assert [g.input_tokens for g in first] != [g.input_tokens for g in second]


def test_generate_groups_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_groups"):
        generate_count_counterfactual_groups(0, 0, step=0, split="train")
    with pytest.raises(ValueError, match="step"):
        generate_count_counterfactual_groups(0, 1, step=-1, split="train")
    with pytest.raises(ValueError, match="group_size"):
        generate_count_counterfactual_groups(0, 1, step=0, split="train", group_size=1)


def test_group_rejects_mismatched_examples_and_targets() -> None:
    groups = generate_count_counterfactual_groups(0, 1, step=0, split="train")
    group = groups[0]
    with pytest.raises(ValueError, match="len\\(examples\\) == len\\(targets\\)"):
        CountCounterfactualGroup(
            input_tokens=group.input_tokens,
            examples=group.examples,
            targets=group.targets[:-1],
        )


def test_group_rejects_fewer_than_min_group_size_members() -> None:
    groups = generate_count_counterfactual_groups(0, 1, step=0, split="train")
    group = groups[0]
    with pytest.raises(ValueError, match="requires >="):
        CountCounterfactualGroup(
            input_tokens=group.input_tokens,
            examples=group.examples[:1],
            targets=group.targets[:1],
        )


def test_group_rejects_non_distinct_outputs() -> None:
    groups = generate_count_counterfactual_groups(0, 1, step=0, split="train")
    group = groups[0]
    with pytest.raises(ValueError, match="pairwise distinct outputs"):
        CountCounterfactualGroup(
            input_tokens=group.input_tokens,
            examples=(group.examples[0], group.examples[0]),
            targets=(group.targets[0], group.targets[0]),
        )


def test_wrong_target_index_points_to_a_different_member() -> None:
    groups = generate_count_counterfactual_groups(0, 20, step=0, split="train")
    for group in groups:
        for index in range(len(group.examples)):
            partner = group.wrong_target_index(index)
            assert partner != index
            assert group.examples[partner].target_tokens != group.examples[index].target_tokens


# --- _flatten_groups -----------------------------------------------------------


def test_flatten_groups_preserves_total_example_count() -> None:
    groups = generate_count_counterfactual_groups(0, 10, step=0, split="train")
    examples, wrong_call_by_id, wrong_target_tokens_by_id = _flatten_groups(groups)
    assert len(examples) == sum(len(g.examples) for g in groups)
    assert len(wrong_call_by_id) == len(examples)
    assert len(wrong_target_tokens_by_id) == len(examples)


def test_flatten_groups_wrong_call_is_always_effectful() -> None:
    groups = generate_count_counterfactual_groups(0, 30, step=0, split="train")
    examples, _, wrong_target_tokens_by_id = _flatten_groups(groups)
    for example in examples:
        assert wrong_target_tokens_by_id[id(example)] != example.target_tokens


def test_flatten_groups_wrong_call_is_a_well_formed_count_call() -> None:
    groups = generate_count_counterfactual_groups(0, 5, step=0, split="train")
    examples, wrong_call_by_id, _ = _flatten_groups(groups)
    for example in examples:
        call = wrong_call_by_id[id(example)]
        assert call.operation == "COUNT"
        assert "target" in call.arguments


# --- CountCounterfactualGateConfig validation ----------------------------------


def test_config_rejects_group_size_below_minimum() -> None:
    with pytest.raises(ValueError, match="group_size"):
        _tiny_config(group_size=1)


def test_config_rejects_max_sequence_length_smaller_than_length_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(sequence_length_range=(4, 20), max_sequence_length=8)


def test_config_rejects_non_positive_num_unseen_eval_groups() -> None:
    with pytest.raises(ValueError, match="num_unseen_eval_groups"):
        _tiny_config(num_unseen_eval_groups=0)


def test_config_rejects_non_positive_min_unseen_eval_examples() -> None:
    with pytest.raises(ValueError, match="min_unseen_eval_examples"):
        _tiny_config(min_unseen_eval_examples=0)


def test_config_defaults_to_documented_group_size() -> None:
    config = CountCounterfactualGateConfig()
    assert config.group_size == DEFAULT_GROUP_SIZE


# --- count_counterfactual_gate_config_from_dict --------------------------------


def test_config_from_dict_fills_in_defaults() -> None:
    config = count_counterfactual_gate_config_from_dict({"seed": 3})
    assert config.seed == 3
    assert config.group_size == DEFAULT_GROUP_SIZE
    assert config.primitive_rank == CountCounterfactualGateConfig().primitive_rank


def test_config_from_dict_parses_nested_train_configs() -> None:
    raw = {
        "core_train": {"steps": 5},
        "primitive_train": {"steps": 7, "lr": 0.001},
        "arg_dim": 12,
    }
    config = count_counterfactual_gate_config_from_dict(raw)
    assert config.core_train.steps == 5
    assert config.primitive_train.steps == 7
    assert config.primitive_train.lr == 0.001
    assert config.arg_dim == 12


# --- _build_primitive_bank ------------------------------------------------------


def test_build_primitive_bank_registers_exactly_one_count_primitive() -> None:
    bank = _build_primitive_bank(d_model=16, rank=4, vocab_size=6, max_sequence_length=8, arg_dim=6)
    assert len(bank) == 1
    primitive = bank.get(operation_id("COUNT"))
    assert isinstance(primitive, ConditionedPrimitive)
    assert primitive.config.d_model == 16
    assert primitive.config.rank == 4
    assert primitive.status == PrimitiveStatus.CANDIDATE
    assert primitive.enabled is True


def test_build_primitive_bank_primitive_is_trainable() -> None:
    bank = _build_primitive_bank(d_model=16, rank=4, vocab_size=6, max_sequence_length=8, arg_dim=6)
    assert not bank.get(operation_id("COUNT")).is_frozen()


def test_build_primitive_bank_size_does_not_grow_with_argument_values() -> None:
    bank = _build_primitive_bank(d_model=16, rank=4, vocab_size=6, max_sequence_length=8, arg_dim=6)
    size_before = len(bank)
    h = torch.randn(1, 16)
    count = bank.get(operation_id("COUNT"))
    for target in range(5):
        count(h, argument_values=[target])
        assert len(bank) == size_before


# --- run_count_counterfactual_gate ----------------------------------------------


def test_run_returns_report_with_expected_fields() -> None:
    report = run_count_counterfactual_gate(_tiny_config())
    assert isinstance(report, CountCounterfactualGateReport)
    assert report.core_steps_trained == 3
    assert report.primitive_steps_trained == 3
    assert report.bank_size == 1
    assert report.num_unseen_eval_groups == 20
    assert report.num_unseen_eval_examples >= 8
    for value in (
        report.correct_exact_match,
        report.effectful_wrong_argument_exact_match,
        report.none_exact_match,
        report.argument_effect_rate,
    ):
        assert 0.0 <= value <= 1.0


def test_run_argument_effect_rate_is_always_one_by_construction() -> None:
    """The gate's whole point is that its Wrong-argument control is
    effectful by construction (module docstring), not incidentally --
    unlike `apc.evaluation.parameterized_primitive_gate`'s "+1 modulo"
    construction."""
    report = run_count_counterfactual_gate(_tiny_config())
    assert report.argument_effect_rate == pytest.approx(1.0)


def test_run_freezes_every_stable_core_parameter() -> None:
    report = run_count_counterfactual_gate(_tiny_config())
    assert report.core_trainable_param_count == 0
    assert report.core_param_count > 0


def test_run_primitive_bank_has_nonzero_parameter_count() -> None:
    report = run_count_counterfactual_gate(_tiny_config())
    assert report.primitive_param_count > 0


def test_run_causal_gap_matches_correct_minus_max_controls() -> None:
    report = run_count_counterfactual_gate(_tiny_config())
    expected = report.correct_exact_match - max(
        report.effectful_wrong_argument_exact_match, report.none_exact_match
    )
    assert report.causal_gap == pytest.approx(expected)


def test_run_is_deterministic_given_the_same_seed() -> None:
    first = run_count_counterfactual_gate(_tiny_config())
    second = run_count_counterfactual_gate(_tiny_config())
    assert first.correct_exact_match == second.correct_exact_match
    assert first.effectful_wrong_argument_exact_match == second.effectful_wrong_argument_exact_match
    assert first.none_exact_match == second.none_exact_match


def test_run_rejects_undersized_eval_batch() -> None:
    with pytest.raises(ValueError, match="min_unseen_eval_examples"):
        run_count_counterfactual_gate(
            _tiny_config(num_unseen_eval_groups=1, min_unseen_eval_examples=10_000)
        )


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_count_counterfactual_gate(_tiny_config())
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["correct_exact_match"] == report.correct_exact_match
    assert payload["argument_effect_rate"] == report.argument_effect_rate


# --- _grad_norm / argument-path gradient-norm logging (A1-R005D-005) -----------


def test_grad_norm_is_zero_with_no_gradients() -> None:
    param = torch.nn.Parameter(torch.zeros(3))
    assert _grad_norm([param]) == 0.0


def test_grad_norm_matches_manual_l2_norm() -> None:
    param = torch.nn.Parameter(torch.zeros(3))
    param.grad = torch.tensor([3.0, 4.0, 0.0])
    assert _grad_norm([param]) == pytest.approx(5.0)


def test_grad_norm_combines_multiple_parameters() -> None:
    first = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0])
    second = torch.nn.Parameter(torch.zeros(1))
    second.grad = torch.tensor([4.0])
    assert _grad_norm([first, second]) == pytest.approx(5.0)


def test_run_logs_argument_and_content_path_grad_norms(tmp_path: Path) -> None:
    metrics_path = tmp_path / "primitive_metrics.jsonl"
    run_count_counterfactual_gate(_tiny_config(), primitive_metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    for line in lines:
        row = json.loads(line)
        assert row["argument_path_grad_norm"] >= 0.0
        assert row["content_path_grad_norm"] >= 0.0


# --- ConditioningVariant threading (Task A1-R005D-006) -------------------------


def test_build_primitive_bank_defaults_to_additive_variant() -> None:
    bank = _build_primitive_bank(d_model=16, rank=4, vocab_size=6, max_sequence_length=8, arg_dim=6)
    assert type(bank.get(operation_id("COUNT"))) is ConditionedPrimitive


def test_build_primitive_bank_film_variant_registers_film_primitive() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
        variant=ConditioningVariant.FILM,
    )
    assert len(bank) == 1
    primitive = bank.get(operation_id("COUNT"))
    assert isinstance(primitive, FiLMConditionedPrimitive)


def test_build_primitive_bank_basis_variant_registers_basis_primitive() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
        variant=ConditioningVariant.BASIS,
        num_basis=3,
    )
    assert len(bank) == 1
    primitive = bank.get(operation_id("COUNT"))
    assert isinstance(primitive, BasisModulatedConditionedPrimitive)
    assert primitive.num_basis == 3


@pytest.mark.parametrize(
    "variant", [ConditioningVariant.ADDITIVE, ConditioningVariant.FILM, ConditioningVariant.BASIS]
)
def test_run_with_each_variant_returns_valid_report(variant: ConditioningVariant) -> None:
    report = run_count_counterfactual_gate(_tiny_config(), variant=variant)
    assert isinstance(report, CountCounterfactualGateReport)
    assert report.bank_size == 1
    for value in (
        report.correct_exact_match,
        report.effectful_wrong_argument_exact_match,
        report.none_exact_match,
        report.argument_effect_rate,
    ):
        assert 0.0 <= value <= 1.0


def test_film_and_basis_variants_have_different_parameter_counts_than_additive() -> None:
    additive = run_count_counterfactual_gate(_tiny_config())
    film = run_count_counterfactual_gate(_tiny_config(), variant=ConditioningVariant.FILM)
    basis = run_count_counterfactual_gate(
        _tiny_config(), variant=ConditioningVariant.BASIS, num_basis=3
    )
    assert film.primitive_param_count != additive.primitive_param_count
    assert basis.primitive_param_count != additive.primitive_param_count


def test_run_multi_seed_threads_variant_through_every_seed() -> None:
    multi_seed = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0, 1), variant=ConditioningVariant.FILM
    )
    assert len(multi_seed.per_seed) == 2
    for report in multi_seed.per_seed:
        assert report.bank_size == 1


@pytest.mark.parametrize(
    ("variant", "expected_type", "proj_attr"),
    [
        (ConditioningVariant.ADDITIVE, ConditionedPrimitive, "c_proj"),
        (ConditioningVariant.FILM, FiLMConditionedPrimitive, "film_proj"),
    ],
)
def test_argument_and_content_path_params_includes_the_right_projection(
    variant: ConditioningVariant, expected_type: type, proj_attr: str
) -> None:
    bank = _build_primitive_bank(
        d_model=16, rank=4, vocab_size=6, max_sequence_length=8, arg_dim=6, variant=variant
    )
    primitive = bank.get(operation_id("COUNT"))
    assert isinstance(primitive, expected_type)
    argument_params, content_params = _argument_and_content_path_params(primitive, variant)
    proj_param_ids = {id(p) for p in getattr(primitive, proj_attr).parameters()}
    argument_param_ids = {id(p) for p in argument_params}
    content_param_ids = {id(p) for p in content_params}
    assert proj_param_ids <= argument_param_ids
    assert proj_param_ids.isdisjoint(content_param_ids)
    assert {id(p) for p in primitive.a_proj.parameters()} <= content_param_ids
    assert {id(p) for p in primitive.b_proj.parameters()} <= content_param_ids


def test_argument_and_content_path_params_basis_includes_mix_proj_and_basis() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
        variant=ConditioningVariant.BASIS,
        num_basis=3,
    )
    primitive = bank.get(operation_id("COUNT"))
    assert isinstance(primitive, BasisModulatedConditionedPrimitive)
    argument_params, content_params = _argument_and_content_path_params(
        primitive, ConditioningVariant.BASIS
    )
    argument_param_ids = {id(p) for p in argument_params}
    assert {id(p) for p in primitive.mix_proj.parameters()} <= argument_param_ids
    assert id(primitive.basis) in argument_param_ids


# --- run_count_counterfactual_gate_multi_seed -----------------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    seeds = (0, 1)
    result = run_count_counterfactual_gate_multi_seed(_tiny_config(), seeds=seeds, run_dir=tmp_path)
    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    for seed in seeds:
        assert (tmp_path / f"seed_{seed}" / "report.json").exists()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    result = run_count_counterfactual_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert [report.config.seed for report in result.per_seed] == [0, 1, 2]


def test_multi_seed_mean_causal_gap_matches_mean_arms() -> None:
    result = run_count_counterfactual_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    expected = result.mean_correct_exact_match - max(
        result.mean_effectful_wrong_argument_exact_match, result.mean_none_exact_match
    )
    assert result.mean_causal_gap == pytest.approx(expected)


def test_multi_seed_family_count_passed_reflects_bank_size() -> None:
    result = run_count_counterfactual_gate_multi_seed(_tiny_config(), seeds=(0,))
    assert result.family_count_passed is True
    assert all(report.bank_size == 1 for report in result.per_seed)


def test_multi_seed_passed_requires_all_four_conditions() -> None:
    """Force each of the four gating conditions to fail in turn via an
    unreachable threshold, without depending on real training dynamics."""
    fail_correct = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0,), correct_threshold=2.0
    )
    assert fail_correct.correct_passed is False
    assert fail_correct.passed is False

    fail_wrong_argument = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0,), effectful_wrong_argument_ceiling=-1.0
    )
    assert fail_wrong_argument.effectful_wrong_argument_passed is False
    assert fail_wrong_argument.passed is False

    fail_none = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0,), none_ceiling=-1.0
    )
    assert fail_none.none_passed is False
    assert fail_none.passed is False

    fail_gap = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0,), min_causal_gap=2.0
    )
    assert fail_gap.causal_gap_passed is False
    assert fail_gap.passed is False


def test_multi_seed_meets_seed_policy_reflects_seed_count() -> None:
    short = run_count_counterfactual_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert short.meets_seed_policy is False
    full = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=tuple(range(MIN_GATE_SEEDS))
    )
    assert full.meets_seed_policy is True


def test_multi_seed_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    result = run_count_counterfactual_gate_multi_seed(
        _tiny_config(), seeds=(0, 1), run_dir=tmp_path
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] == result.passed
    assert len(payload["per_seed"]) == 2


# --- module constants -----------------------------------------------------------


def test_default_seeds_matches_min_gate_seeds_length() -> None:
    assert len(DEFAULT_SEEDS) == MIN_GATE_SEEDS


def test_thresholds_match_a1_r005d_004() -> None:
    assert CORRECT_THRESHOLD == 0.90
    assert EFFECTFUL_WRONG_ARGUMENT_CEILING == 0.30
    assert MIN_CAUSAL_GAP == 0.50
    assert NONE_CEILING == 0.30


def test_no_cuda_required_for_cpu_device_override() -> None:
    """Sanity check that the tiny fixture config actually runs on CPU (no
    hidden CUDA dependency), matching every other Phase A.1 gate's test
    convention."""
    report = run_count_counterfactual_gate(_tiny_config())
    assert report.device == str(torch.device("cpu"))
