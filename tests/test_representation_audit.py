"""Tests for the frozen `h_content` information audit (Phase A.1 diagnostic
Task A1-R005E-002).

Fast, small-step tests only -- mirroring `tests/
test_count_counterfactual_gate.py`'s convention of exercising the wiring
(config validation, per-operation frozen-core pretraining, probe fitting,
report shape, multi-seed aggregation) with a tiny model and a handful of
training steps. The audit's own scientific claim (the retry's full
`core_train` budget, 3 seeds) is run via
`scripts/representation_audit.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.evaluation.representation_audit import (
    BIND_ROLE_OPERATIONS,
    DEFAULT_SEEDS,
    PAIR_PROBE_DESIRABLE_THRESHOLD,
    POSITION_DESIRABLE_THRESHOLD,
    RECONSTRUCTION_EXACT_MATCH_DESIRABLE_THRESHOLD,
    ROLE_PROBE_DESIRABLE_THRESHOLD,
    TOKEN_IDENTITY_DESIRABLE_THRESHOLD,
    ProbeTrainConfig,
    RepresentationAuditConfig,
    RepresentationAuditReport,
    _extract_representation,
    _pretrain_frozen_stable_core,
    representation_audit_config_from_dict,
    run_representation_audit,
    run_representation_audit_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig


def _tiny_config(**overrides: object) -> RepresentationAuditConfig:
    base = RepresentationAuditConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=("COUNT", "BIND"),
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
        probe_train=ProbeTrainConfig(steps=5, lr=0.03, weight_decay=0.0),
        num_probe_train_examples=32,
        num_probe_eval_examples=16,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def test_config_validation_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=())


def test_config_validation_rejects_non_positive_probe_example_counts() -> None:
    with pytest.raises(ValueError, match="num_probe_train_examples"):
        _tiny_config(num_probe_train_examples=0)
    with pytest.raises(ValueError, match="num_probe_eval_examples"):
        _tiny_config(num_probe_eval_examples=0)


def test_config_from_dict_round_trips_defaults_and_overrides() -> None:
    raw = {
        "seed": 2,
        "vocab_size": 6,
        "sequence_length_range": [4, 6],
        "operation_names": ["COUNT"],
        "model": {"d_model": 16, "n_layer": 2, "n_head": 2, "d_ff": 32, "max_seq_len": 32},
        "core_train": {"steps": 3, "batch_size": 4},
        "probe_train": {"steps": 5},
        "num_probe_train_examples": 32,
        "num_probe_eval_examples": 16,
    }
    config = representation_audit_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("COUNT",)
    assert config.model["d_model"] == 16
    assert config.core_train.steps == 3
    assert config.probe_train.steps == 5
    assert config.num_probe_train_examples == 32

    defaulted = representation_audit_config_from_dict({})
    assert defaulted == RepresentationAuditConfig()


def test_pretrain_frozen_stable_core_freezes_every_parameter() -> None:
    config = _tiny_config()
    core = _pretrain_frozen_stable_core(config, "COUNT")
    assert core.operation == "COUNT"
    assert all(not p.requires_grad for p in core.model.parameters())
    assert core.model.training is False


def test_pretrain_frozen_stable_core_is_deterministic_given_same_seed() -> None:
    config = _tiny_config()
    core_a = _pretrain_frozen_stable_core(config, "COUNT")
    core_b = _pretrain_frozen_stable_core(config, "COUNT")
    for p_a, p_b in zip(core_a.model.parameters(), core_b.model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
    assert core_a.final_core_train_loss == core_b.final_core_train_loss


def test_extract_representation_shapes_for_non_bind_operation() -> None:
    config = _tiny_config()
    core = _pretrain_frozen_stable_core(config, "COUNT")
    examples = core.generator.generate_online(8, step=10_000, split="test")
    max_content_length = config.sequence_length_range[1]
    extracted = _extract_representation(
        core, examples, max_content_length=max_content_length, include_roles=False
    )

    total_content_length = sum(len(example.input_tokens) for example in examples)
    d_model = config.model["d_model"]
    assert extracted.content_features.shape == (total_content_length, d_model)
    assert extracted.content_token_labels.shape == (total_content_length,)
    assert extracted.content_position_labels.shape == (total_content_length,)
    assert extracted.summary_features.shape == (len(examples), d_model)
    assert extracted.reconstruction_targets.shape == (len(examples), max_content_length)
    assert extracted.reconstruction_mask.shape == (len(examples), max_content_length)
    assert extracted.role_labels is None
    assert extracted.key_features is None
    assert extracted.key_pair_targets is None

    # Position labels are 0-indexed within each example's own content span.
    for row, example in enumerate(examples):
        length = len(example.input_tokens)
        assert extracted.reconstruction_mask[row, :length].sum().item() == length


def test_extract_representation_role_and_pair_labels_for_bind() -> None:
    config = _tiny_config()
    core = _pretrain_frozen_stable_core(config, "BIND")
    examples = core.generator.generate_online(8, step=10_000, split="test")
    max_content_length = config.sequence_length_range[1]
    extracted = _extract_representation(
        core, examples, max_content_length=max_content_length, include_roles=True
    )

    assert extracted.role_labels is not None
    assert extracted.key_features is not None
    assert extracted.key_pair_targets is not None

    # Every content length is even (apc.environments.operations.BindOp), and
    # role labels alternate key(0)/value(1) starting at position 0.
    offset = 0
    num_keys = 0
    for example in examples:
        length = len(example.input_tokens)
        assert length % 2 == 0
        role_slice = extracted.role_labels[offset : offset + length]
        expected = torch.tensor([j % 2 for j in range(length)], dtype=torch.long)
        assert torch.equal(role_slice, expected)
        offset += length
        num_keys += length // 2

    assert extracted.key_features.shape == (num_keys, config.model["d_model"])
    assert extracted.key_pair_targets.shape == (num_keys,)

    # Every pair target must be a paired *value* token actually present at
    # that key's own (key_position + 1) in its own example.
    pair_index = 0
    for example in examples:
        length = len(example.input_tokens)
        for j in range(0, length, 2):
            expected_value = example.input_tokens[j + 1]
            assert int(extracted.key_pair_targets[pair_index].item()) == expected_value
            pair_index += 1


def test_run_representation_audit_report_shape_and_ranges() -> None:
    config = _tiny_config()
    report = run_representation_audit(config)

    assert isinstance(report, RepresentationAuditReport)
    assert set(report.per_operation) == {"COUNT", "BIND"}

    count_report = report.per_operation["COUNT"]
    assert 0.0 <= count_report.token_identity_accuracy <= 1.0
    assert 0.0 <= count_report.position_accuracy <= 1.0
    assert 0.0 <= count_report.reconstruction_token_accuracy <= 1.0
    assert 0.0 <= count_report.reconstruction_exact_match <= 1.0
    assert count_report.token_identity_meets_threshold == (
        count_report.token_identity_accuracy >= TOKEN_IDENTITY_DESIRABLE_THRESHOLD
    )
    assert count_report.position_meets_threshold == (
        count_report.position_accuracy >= POSITION_DESIRABLE_THRESHOLD
    )
    assert count_report.reconstruction_meets_threshold == (
        count_report.reconstruction_exact_match >= RECONSTRUCTION_EXACT_MATCH_DESIRABLE_THRESHOLD
    )
    assert count_report.role_accuracy is None
    assert count_report.role_meets_threshold is None
    assert count_report.pair_accuracy is None
    assert count_report.pair_meets_threshold is None
    assert count_report.num_probe_train_examples == config.num_probe_train_examples
    assert count_report.num_probe_eval_examples == config.num_probe_eval_examples

    bind_report = report.per_operation["BIND"]
    assert bind_report.role_accuracy is not None
    assert 0.0 <= bind_report.role_accuracy <= 1.0
    assert bind_report.role_meets_threshold == (
        bind_report.role_accuracy >= ROLE_PROBE_DESIRABLE_THRESHOLD
    )
    assert bind_report.pair_accuracy is not None
    assert 0.0 <= bind_report.pair_accuracy <= 1.0
    assert bind_report.pair_meets_threshold == (
        bind_report.pair_accuracy >= PAIR_PROBE_DESIRABLE_THRESHOLD
    )

    # Round-trips through JSON (report.json / summary.json shape).
    json.dumps(report.to_dict())


def test_run_representation_audit_operations_default_to_parameterized() -> None:
    config = RepresentationAuditConfig()
    assert set(config.operation_names) == {"SHIFT", "SELECT", "COUNT", "BIND"}


def test_run_representation_audit_multi_seed_aggregates_per_operation(tmp_path: Path) -> None:
    base_config = _tiny_config()
    seeds = (0, 1)
    result = run_representation_audit_multi_seed(base_config, seeds=seeds, run_dir=tmp_path)

    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    assert set(result.per_operation_summary) == {"COUNT", "BIND"}

    count_summary = result.per_operation_summary["COUNT"]
    expected_mean = sum(
        report.per_operation["COUNT"].token_identity_accuracy for report in result.per_seed
    ) / len(result.per_seed)
    assert count_summary.mean_token_identity_accuracy == pytest.approx(expected_mean)
    assert count_summary.mean_role_accuracy is None
    assert count_summary.mean_pair_accuracy is None

    bind_summary = result.per_operation_summary["BIND"]
    assert bind_summary.mean_role_accuracy is not None
    assert bind_summary.mean_pair_accuracy is not None

    for seed in seeds:
        seed_report_path = tmp_path / f"seed_{seed}" / "report.json"
        assert seed_report_path.is_file()
        json.loads(seed_report_path.read_text(encoding="utf-8"))

    json.dumps(result.to_dict())


def test_default_seeds_and_bind_role_operations_constants() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert BIND_ROLE_OPERATIONS == ("BIND",)
