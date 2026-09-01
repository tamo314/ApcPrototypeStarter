"""Tests for Task A1-R005D-001's re-analysis of the existing, frozen A1-R005
run (`docs/CODEX_TASKS_A1_R005_RETRY.md`).

Fast, small-step tests only, mirroring `tests/test_parameterized_primitive_
gate.py`'s own convention: `tiny_run_dir` builds a real (but tiny) two-seed
`parameterized_primitive_gate` run via the existing gate's own multi-seed
entry point, then every test below re-analyzes that real run directory --
never a hand-rolled fake report shape that could drift from the real schema.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from apc.environments.generator import TaskGenerator, oracle_call_for_example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.evaluation.parameterized_primitive_gate import (
    ParameterizedPrimitiveGateConfig,
    PrimitiveTrainConfig,
    _wrong_argument_value,
    run_parameterized_primitive_gate_multi_seed,
)
from apc.evaluation.parameterized_primitive_gate_reanalysis import (
    NOT_RECOVERABLE_REASON,
    compute_argument_effect,
    reanalyze_run,
    reanalyze_seed,
    reconstruct_unseen_eval_examples,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig


def _tiny_config(**overrides: object) -> ParameterizedPrimitiveGateConfig:
    base = ParameterizedPrimitiveGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=PARAMETERIZED_OPERATION_NAMES,
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
        num_unseen_eval_examples=128,
        min_examples_per_operation=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


@pytest.fixture(scope="module")
def tiny_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("a1_r005d001_tiny_run")
    run_parameterized_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1), run_dir=run_dir)
    return run_dir


# --- compute_argument_effect ---------------------------------------------------


@pytest.mark.parametrize("operation_name", PARAMETERIZED_OPERATION_NAMES)
def test_argument_effect_matches_independent_interpreter_computation(
    operation_name: str,
) -> None:
    """Cross-check `compute_argument_effect`'s `PrimitiveCall.execute` path
    against directly calling `Operation.apply` -- an independent code path
    to the same ground truth, to catch any wiring bug rather than only
    re-deriving the same computation twice."""
    generator = TaskGenerator(
        seed=0, operation_names=(operation_name,), vocab_size=6, sequence_length_range=(4, 6)
    )
    examples = generator.generate_online(200, step=0, split="test")
    operation = get_operation(operation_name)

    observed_effects = []
    for example in examples:
        correct_call = oracle_call_for_example(example)
        wrong_arguments = _wrong_argument_value(correct_call, example)
        expected_target = operation.apply(
            example.input_tokens, example.vocab_size, wrong_arguments
        )
        expected_effect = expected_target != example.target_tokens
        assert compute_argument_effect(example) == expected_effect
        observed_effects.append(expected_effect)

    # Not a vacuous check: at this batch size every operation should show at
    # least one effectful example (a random +1-modulo argument shift should
    # essentially always change something for a non-trivial domain).
    assert any(observed_effects)


# --- reconstruct_unseen_eval_examples -------------------------------------------


def test_reconstruct_unseen_eval_examples_is_deterministic(tiny_run_dir: Path) -> None:
    report = json.loads((tiny_run_dir / "seed_0" / "report.json").read_text())
    first = reconstruct_unseen_eval_examples(report["config"])
    second = reconstruct_unseen_eval_examples(report["config"])
    assert [example.input_tokens for example in first] == [
        example.input_tokens for example in second
    ]
    assert [example.target_tokens for example in first] == [
        example.target_tokens for example in second
    ]


def test_reconstruct_unseen_eval_examples_matches_saved_batch_size(tiny_run_dir: Path) -> None:
    report = json.loads((tiny_run_dir / "seed_0" / "report.json").read_text())
    examples = reconstruct_unseen_eval_examples(report["config"])
    assert len(examples) == report["num_unseen_eval_examples"]


# --- reanalyze_seed --------------------------------------------------------------


def test_reanalyze_seed_reproduces_saved_eval_counts(tiny_run_dir: Path) -> None:
    seed_report = reanalyze_seed(tiny_run_dir, 0)
    saved = json.loads((tiny_run_dir / "seed_0" / "report.json").read_text())
    assert seed_report.per_operation_eval_counts == saved["per_operation_eval_counts"]


def test_reanalyze_seed_covers_all_operations(tiny_run_dir: Path) -> None:
    seed_report = reanalyze_seed(tiny_run_dir, 0)
    assert set(seed_report.operation_names) == set(PARAMETERIZED_OPERATION_NAMES)
    assert set(seed_report.per_operation_argument_effect_rate) == set(
        PARAMETERIZED_OPERATION_NAMES
    )
    for rate in seed_report.per_operation_argument_effect_rate.values():
        assert 0.0 <= rate <= 1.0


def test_reanalyze_seed_flags_effectful_wrong_argument_as_not_recoverable(
    tiny_run_dir: Path,
) -> None:
    seed_report = reanalyze_seed(tiny_run_dir, 0)
    for name in PARAMETERIZED_OPERATION_NAMES:
        # raw score: a real number, taken from the existing artifact.
        assert isinstance(seed_report.per_operation_wrong_argument_exact_match[name], float)
        # effectful score: explicitly unavailable, never silently substituted
        # or conflated with the raw score above.
        assert seed_report.per_operation_effectful_wrong_argument_exact_match[name] is None
    assert (
        seed_report.not_recoverable["per_operation_effectful_wrong_argument_exact_match"]
        == NOT_RECOVERABLE_REASON
    )


def test_reanalyze_seed_flags_token_accuracy_as_not_recoverable(tiny_run_dir: Path) -> None:
    seed_report = reanalyze_seed(tiny_run_dir, 0)
    assert all(value is None for value in seed_report.per_operation_correct_token_accuracy.values())
    assert (
        seed_report.not_recoverable["per_operation_correct_token_accuracy"]
        == NOT_RECOVERABLE_REASON
    )


def test_reanalyze_seed_does_not_modify_existing_artifacts(tiny_run_dir: Path) -> None:
    report_path = tiny_run_dir / "seed_0" / "report.json"
    before = report_path.read_bytes()
    reanalyze_seed(tiny_run_dir, 0)
    after = report_path.read_bytes()
    assert before == after


def test_reanalyze_seed_raises_on_mismatched_eval_counts(tmp_path: Path) -> None:
    saved = {
        "config": {
            "seed": 0,
            "vocab_size": 6,
            "sequence_length_range": [4, 6],
            "operation_names": list(PARAMETERIZED_OPERATION_NAMES),
            "num_unseen_eval_examples": 128,
        },
        "num_unseen_eval_examples": 128,
        "per_operation_eval_counts": {"SHIFT": 999, "SELECT": 1, "COUNT": 1, "BIND": 1},
        "per_operation_correct_exact_match": dict.fromkeys(PARAMETERIZED_OPERATION_NAMES, 0.0),
        "per_operation_wrong_argument_exact_match": dict.fromkeys(
            PARAMETERIZED_OPERATION_NAMES, 0.0
        ),
        "per_operation_wrong_family_exact_match": dict.fromkeys(
            PARAMETERIZED_OPERATION_NAMES, 0.0
        ),
        "per_operation_none_exact_match": dict.fromkeys(PARAMETERIZED_OPERATION_NAMES, 0.0),
        "causal_gap": 0.0,
    }
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "report.json").write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="per-operation eval counts"):
        reanalyze_seed(tmp_path, 0)


# --- reanalyze_run -----------------------------------------------------------------


def test_reanalyze_run_covers_all_operations_and_seeds(tiny_run_dir: Path) -> None:
    result = reanalyze_run(tiny_run_dir, seeds=(0, 1))
    assert set(result.operation_names) == set(PARAMETERIZED_OPERATION_NAMES)
    assert result.seeds == (0, 1)
    assert len(result.per_seed) == 2
    for name in PARAMETERIZED_OPERATION_NAMES:
        assert name in result.mean_per_operation_argument_effect_rate
        assert name in result.mean_per_operation_target_output_length_mean


def test_reanalyze_run_does_not_modify_existing_artifacts(tiny_run_dir: Path) -> None:
    before = {
        path: path.read_bytes() for path in sorted(tiny_run_dir.rglob("*")) if path.is_file()
    }
    reanalyze_run(tiny_run_dir, seeds=(0, 1))
    after = {
        path: path.read_bytes() for path in sorted(tiny_run_dir.rglob("*")) if path.is_file()
    }
    assert before == after


def test_reanalyze_run_separates_raw_from_effectful_wrong_argument(tiny_run_dir: Path) -> None:
    result = reanalyze_run(tiny_run_dir, seeds=(0, 1))
    for name in PARAMETERIZED_OPERATION_NAMES:
        assert isinstance(result.mean_per_operation_wrong_argument_exact_match[name], float)
    assert "per_operation_effectful_wrong_argument_exact_match" in result.not_recoverable


def test_reanalyze_run_rejects_mismatched_operation_names_across_seeds(
    tiny_run_dir: Path, tmp_path: Path
) -> None:
    # seed_0 with the tiny run's real 4-operation config, seed_1 a real (but
    # self-consistent, so it passes reanalyze_seed on its own) 2-operation
    # run -- must not silently aggregate two different operation pools.
    mixed_dir = tmp_path / "mixed_run"
    (mixed_dir / "seed_0").mkdir(parents=True)
    (mixed_dir / "seed_0" / "report.json").write_text(
        (tiny_run_dir / "seed_0" / "report.json").read_text(), encoding="utf-8"
    )
    two_operation_config = _tiny_config(operation_names=("SHIFT", "SELECT"), seed=1)
    run_parameterized_primitive_gate_multi_seed(
        two_operation_config, seeds=(1,), run_dir=mixed_dir
    )

    with pytest.raises(ValueError, match="operation_names"):
        reanalyze_run(mixed_dir, seeds=(0, 1))


def test_reanalyze_run_rejects_empty_seeds(tiny_run_dir: Path) -> None:
    with pytest.raises(ValueError, match="seeds"):
        reanalyze_run(tiny_run_dir, seeds=())


def test_run_reanalysis_to_dict_round_trips_through_json(tiny_run_dir: Path) -> None:
    result = reanalyze_run(tiny_run_dir, seeds=(0, 1))
    round_tripped = json.loads(json.dumps(result.to_dict()))
    assert round_tripped["seeds"] == [0, 1]
    assert set(round_tripped["operation_names"]) == set(PARAMETERIZED_OPERATION_NAMES)
