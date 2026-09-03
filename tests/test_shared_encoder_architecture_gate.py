"""Tests for the shared queryable representation architecture wiring gate
(Phase A.1 diagnostic Task A1-R005E-S001).

Fast, small-model tests only, mirroring `tests/test_compact_cross_position_
operator_probe.py`/`tests/test_joint_representation_compact_operator_probe.py`'s
convention of exercising the wiring with a tiny model. This task runs no
training loop (task Acceptance: "No milestone benchmark yet"), so there is no
separate milestone script invocation to keep out of this file -- `scripts/
shared_encoder_architecture_gate.py` runs the same fast gate at the config's
own default (still small) sizes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.core.model import DecoderOnlyTransformer
from apc.environments.operations import get_operation
from apc.evaluation.compact_cross_position_operator_probe import CompactOperatorConfig
from apc.evaluation.shared_encoder_architecture_gate import (
    DEFAULT_SEEDS,
    MIN_GATE_SEEDS,
    SharedEncoderArchitectureConfig,
    SharedEncoderArchitectureGateReport,
    _dummy_argument_values,
    _encoder_grad_connected_for_operation,
    _oracle_dispatch_rejects_unknown_operation,
    _sample_probe_examples,
    build_shared_encoder_architecture,
    content_state_max_abs_diff_across_operations,
    encode_content,
    encoder_is_free_of_operation_specific_modules,
    operator_holds_no_encoder_submodule,
    run_shared_encoder_architecture_gate,
    run_shared_encoder_architecture_gate_multi_seed,
    run_shared_operator,
    shared_encoder_architecture_config_from_dict,
)

OPERATIONS = ("SHIFT", "SELECT", "COUNT", "BIND")


def _tiny_config(**overrides: object) -> SharedEncoderArchitectureConfig:
    base = SharedEncoderArchitectureConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=OPERATIONS,
        group_size=3,
        model={
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=8,
        n_operator_head=2,
        d_operator_ff=16,
        arg_dim=6,
        max_sequence_length=8,
        num_probe_groups=6,
        min_probe_examples=6,
        atol=1e-5,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- Config validation / round-trip -----------------------------------------


def test_config_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=())


def test_config_rejects_non_divisible_operator_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(d_operator=15, n_operator_head=2)


def test_config_rejects_max_sequence_length_below_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(max_sequence_length=3, sequence_length_range=(4, 6))


def test_config_rejects_non_positive_num_probe_groups() -> None:
    with pytest.raises(ValueError, match="num_probe_groups"):
        _tiny_config(num_probe_groups=0)


def test_config_rejects_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol"):
        _tiny_config(atol=-1.0)


def test_config_from_dict_round_trips_defaults_and_overrides() -> None:
    raw = {
        "seed": 2,
        "vocab_size": 6,
        "sequence_length_range": [4, 6],
        "operation_names": ["SHIFT", "BIND"],
        "group_size": 3,
        "model": {"d_model": 16, "n_layer": 2, "n_head": 2, "d_ff": 32, "max_seq_len": 32},
        "device": "cpu",
        "d_operator": 8,
        "n_operator_head": 2,
        "d_operator_ff": 16,
        "arg_dim": 6,
        "max_sequence_length": 8,
        "num_probe_groups": 6,
        "min_probe_examples": 6,
        "atol": 1e-5,
    }
    config = shared_encoder_architecture_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("SHIFT", "BIND")
    assert config.model["d_model"] == 16
    assert config.device == "cpu"
    assert config.d_operator == 8

    defaulted = shared_encoder_architecture_config_from_dict({})
    assert defaulted == SharedEncoderArchitectureConfig()


def test_default_operation_names_are_the_four_parameterized_operations() -> None:
    assert set(SharedEncoderArchitectureConfig().operation_names) == {
        "SHIFT",
        "SELECT",
        "COUNT",
        "BIND",
    }


def test_default_seed_policy_matches_every_other_gate() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert MIN_GATE_SEEDS == 5


def test_default_operator_dimensions_match_e005_unchanged() -> None:
    # Work item 1: "Reuse E-006A compact operator architecture unchanged."
    shared_defaults = SharedEncoderArchitectureConfig()
    compact_defaults = CompactOperatorConfig()
    assert shared_defaults.d_operator == compact_defaults.d_operator
    assert shared_defaults.n_operator_head == compact_defaults.n_operator_head
    assert shared_defaults.d_operator_ff == compact_defaults.d_operator_ff
    assert shared_defaults.arg_dim == compact_defaults.arg_dim
    assert shared_defaults.max_sequence_length == compact_defaults.max_sequence_length
    assert shared_defaults.model == compact_defaults.model


# --- build_shared_encoder_architecture: one shared encoder ------------------


def test_build_shared_encoder_architecture_has_one_encoder_and_four_operators() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    assert isinstance(architecture.core.model, DecoderOnlyTransformer)
    assert set(architecture.operators) == set(OPERATIONS)
    # Every operator was built from the SAME core's own d_model.
    for operator in architecture.operators.values():
        assert operator.content_in_proj.in_features == architecture.core.model.config.d_model


def test_build_shared_encoder_architecture_operators_are_distinct_objects() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    operator_ids = {id(operator) for operator in architecture.operators.values()}
    assert len(operator_ids) == len(OPERATIONS)


def test_operators_hold_no_encoder_submodule() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    for operator in architecture.operators.values():
        assert operator_holds_no_encoder_submodule(operator)


def test_encoder_is_free_of_operation_specific_modules() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    hits = encoder_is_free_of_operation_specific_modules(
        architecture.core, architecture.config.operation_names
    )
    assert hits == ()


def test_per_operation_argument_encoders_and_readouts_are_distinct() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    arg_encoder_ids = {id(operator.arg_encoder) for operator in architecture.operators.values()}
    readout_ids = {id(operator.readout) for operator in architecture.operators.values()}
    assert len(arg_encoder_ids) == len(OPERATIONS)
    assert len(readout_ids) == len(OPERATIONS)


# --- encode_content is task-blind -------------------------------------------


def test_encode_content_has_no_operation_parameter() -> None:
    import inspect

    signature = inspect.signature(encode_content)
    assert "operation" not in signature.parameters
    assert "argument_values" not in signature.parameters


def test_encode_content_shape() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    content_features, content_lengths = encode_content(architecture.core, examples)
    assert content_features.shape[0] == len(examples)
    assert content_features.shape[1] == max(content_lengths)
    assert content_features.shape[2] == architecture.core.model.config.d_model


# --- run_shared_operator: oracle dispatch ------------------------------------


def test_run_shared_operator_dispatches_every_operation() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    for operation in OPERATIONS:
        argument_values = _dummy_argument_values(operation, examples)
        logits = run_shared_operator(architecture, operation, examples, argument_values)
        content_lengths = [len(example.input_tokens) for example in examples]
        expected_out_max = max(
            get_operation(operation).output_length(length) for length in content_lengths
        )
        expected_shape = (len(examples), expected_out_max, architecture.config.vocab_size)
        assert tuple(logits.shape) == expected_shape


def test_run_shared_operator_raises_on_unconfigured_operation() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    with pytest.raises(KeyError):
        run_shared_operator(architecture, "NOT_A_REAL_OPERATION", examples, None)


def test_oracle_dispatch_rejects_unknown_operation_helper() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    assert _oracle_dispatch_rejects_unknown_operation(architecture, examples)


# --- Content-state invariance across operations ------------------------------


def test_content_state_identical_across_operations() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    diff = content_state_max_abs_diff_across_operations(
        architecture, examples, architecture.config.operation_names
    )
    assert diff == 0.0


def test_content_state_identical_for_single_operation_is_zero_by_definition() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    diff = content_state_max_abs_diff_across_operations(architecture, examples, ("SHIFT",))
    assert diff == 0.0


# --- Gradient connectivity: all operations call the same encoder ------------


def test_encoder_grad_connected_for_every_operation() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    for operation in OPERATIONS:
        argument_values = _dummy_argument_values(operation, examples)
        connected = _encoder_grad_connected_for_operation(
            architecture, operation, examples, argument_values
        )
        assert connected


def test_two_different_operations_backprop_into_the_same_encoder_parameters() -> None:
    architecture = build_shared_encoder_architecture(_tiny_config())
    examples = _sample_probe_examples(architecture.config)
    shared_weight = architecture.core.model.token_emb.weight

    shift_args = _dummy_argument_values("SHIFT", examples)
    shift_logits = run_shared_operator(architecture, "SHIFT", examples, shift_args)
    shift_logits.sum().backward()
    assert shared_weight.grad is not None
    shift_grad = shared_weight.grad.clone()
    shared_weight.grad = None

    bind_args = _dummy_argument_values("BIND", examples)
    bind_logits = run_shared_operator(architecture, "BIND", examples, bind_args)
    bind_logits.sum().backward()
    assert shared_weight.grad is not None
    bind_grad = shared_weight.grad.clone()

    # Both operations reached the SAME encoder parameter tensor (not two
    # independent copies) -- their gradients need not be equal, but both
    # must be non-trivial.
    assert torch.isfinite(shift_grad).all()
    assert torch.isfinite(bind_grad).all()
    assert (shift_grad != 0).any()
    assert (bind_grad != 0).any()


# --- Full gate report --------------------------------------------------------


def test_run_shared_encoder_architecture_gate_passes_on_a_tiny_config() -> None:
    report = run_shared_encoder_architecture_gate(_tiny_config())
    assert isinstance(report, SharedEncoderArchitectureGateReport)
    assert report.passed
    assert report.operators_hold_no_encoder_submodule
    assert report.no_operation_specific_encoder_modules
    assert report.operation_specific_encoder_module_names == ()
    assert report.per_operation_argument_encoder_distinct
    assert report.per_operation_readout_distinct
    assert report.content_state_invariant_across_operations
    assert report.content_state_max_abs_diff_across_operations == 0.0
    assert report.all_operations_encoder_grad_connected
    assert all(report.per_operation_encoder_grad_connected.values())
    assert report.all_output_shapes_valid
    assert all(report.per_operation_output_shape_valid.values())
    assert report.oracle_dispatch_rejects_unknown_operation
    assert report.core_param_count > 0
    assert report.core_trainable_param_count == report.core_param_count
    assert set(report.operator_param_counts) == set(OPERATIONS)
    assert all(count > 0 for count in report.operator_param_counts.values())
    assert report.num_probe_examples >= report.config.min_probe_examples


def test_run_shared_encoder_architecture_gate_raises_below_min_probe_examples() -> None:
    with pytest.raises(ValueError, match="min_probe_examples"):
        run_shared_encoder_architecture_gate(_tiny_config(min_probe_examples=10_000))


def test_run_shared_encoder_architecture_gate_multi_seed_aggregates_and_passes() -> None:
    result = run_shared_encoder_architecture_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert result.seeds == (0, 1, 2)
    assert len(result.per_seed) == 3
    assert result.passed
    assert result.max_content_state_absolute_difference == 0.0


def test_run_shared_encoder_architecture_gate_multi_seed_reports_seed_policy() -> None:
    result = run_shared_encoder_architecture_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert result.meets_seed_policy is False
    result_full = run_shared_encoder_architecture_gate_multi_seed(
        _tiny_config(), seeds=(0, 1, 2, 3, 4)
    )
    assert result_full.meets_seed_policy is True


def test_run_shared_encoder_architecture_gate_multi_seed_writes_per_seed_reports(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    run_shared_encoder_architecture_gate_multi_seed(_tiny_config(), seeds=(0, 1), run_dir=run_dir)
    for seed in (0, 1):
        report_path = run_dir / f"seed_{seed}" / "report.json"
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["config"]["seed"] == seed
        assert payload["passed"] is True
