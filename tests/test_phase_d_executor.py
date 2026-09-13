"""Focused fail-closed checks for the D-008 execution boundary."""

from __future__ import annotations

import pytest

from apc.evaluation.phase_d_executor import (
    CAUSAL_ARMS,
    CAUSAL_LENGTH_GROUPS,
    EVAL_SEEDS,
    MODEL_SEEDS,
    PhaseDExecutorConfig,
    PhaseDStopGateError,
    _assert_causal_controls_complete,
    _assert_new_namespace,
    _evaluate_causal_controls,
    _supported_torch_version,
    dry_run_manifest,
    run,
)


def test_d008_dry_run_records_only_the_fixed_authorization(monkeypatch) -> None:
    monkeypatch.setattr(
        "apc.evaluation.phase_d_executor._static_gate", lambda _config: {"status": "PASS"}
    )
    report = dry_run_manifest()

    assert report["mode"] == "DRY_RUN"
    assert tuple(report["config"]["model_seeds"]) == MODEL_SEEDS
    assert report["static_gate"]["status"] == "PASS"
    assert report["information_boundary"]["sealed_access"] == 0
    assert report["conditions"] == (
        "FROZEN_PARENT",
        "LOCAL_SORT_REPAIR",
        "SYMBOLIC_REFERENCE",
    )


def test_d008_runtime_dependency_boundary_is_fail_closed() -> None:
    assert _supported_torch_version("2.12.0+cu128")
    assert _supported_torch_version("2.13.0+cu130")
    assert not _supported_torch_version("2.11.0+cu128")
    assert not _supported_torch_version("2.14.0")


@pytest.mark.parametrize(
    "field, value", [("model_seeds", (40, 41, 42, 43, 45)), ("repair_steps", 6_001)]
)
def test_d008_rejects_seed_or_recipe_deviation(field: str, value: object) -> None:
    with pytest.raises(PhaseDStopGateError):
        PhaseDExecutorConfig(**{field: value})  # type: ignore[arg-type]


def test_d008_refuses_to_overwrite_a_run_namespace(tmp_path) -> None:
    namespace = tmp_path / "existing-run"
    namespace.mkdir()

    with pytest.raises(PhaseDStopGateError, match="cannot be overwritten"):
        _assert_new_namespace(namespace)


def test_causal_controls_record_every_registered_arm_seed_and_length_group(monkeypatch) -> None:
    class FakePrimitive:
        def __init__(self, name: str) -> None:
            self.name = name
            self.enabled = True

    class FakeBank:
        def __init__(self) -> None:
            self.primitives = {0: FakePrimitive("SORT"), 1: FakePrimitive("NEGATE")}

        def get(self, primitive_id: int) -> FakePrimitive:
            return self.primitives[primitive_id]

    def fake_examples(*_args, **_kwargs):
        return [object()] * 1_000

    def fake_em(_core, primitive, examples, _operation):
        em = 990 if primitive.name == "SORT" and primitive.enabled else 0
        return em, len(examples), f"{primitive.name}:{primitive.enabled}"

    monkeypatch.setattr(
        "apc.evaluation.phase_d_executor._generate_parameter_free_examples", fake_examples
    )
    monkeypatch.setattr("apc.evaluation.phase_d_executor._primitive_em", fake_em)

    report = _evaluate_causal_controls(object(), FakeBank(), {"SORT": 0, "NEGATE": 1})

    assert set(report) == {name for name, _ in CAUSAL_LENGTH_GROUPS}
    for group in report.values():
        assert set(group["arms"]) == set(CAUSAL_ARMS)
        assert group["wrong_argument"] == "NOT_APPLICABLE_PARAMETER_FREE_SORT"
        for arm in CAUSAL_ARMS:
            seeds = tuple(row["seed"] for row in group["arms"][arm]["per_evaluation_seed"])
            assert seeds == EVAL_SEEDS


def test_causal_control_completeness_rejects_missing_evaluation_cell() -> None:
    incomplete = {
        name: {
            "arms": {
                arm: {"per_evaluation_seed": [{"seed": seed, "n": 1_000} for seed in EVAL_SEEDS]}
                for arm in CAUSAL_ARMS
            }
        }
        for name, _ in CAUSAL_LENGTH_GROUPS
    }
    incomplete["target_L3_L5"]["arms"]["none"]["per_evaluation_seed"].pop()

    with pytest.raises(PhaseDStopGateError, match="missing or reordered"):
        _assert_causal_controls_complete(incomplete)


def test_d011_run_executes_static_gate_before_creating_directories(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "runs" / "test_executor"
    cohort_root = tmp_path / "runs" / "test_cohort"
    candidate_root = tmp_path / "runs" / "test_candidate"

    def failing_gate(_config):
        raise PhaseDStopGateError("forced static gate stop")

    monkeypatch.setattr("apc.evaluation.phase_d_executor._static_gate", failing_gate)

    cfg = PhaseDExecutorConfig(
        output_root=output_root,
        cohort_root=cohort_root,
        candidate_root=candidate_root,
    )

    with pytest.raises(PhaseDStopGateError, match="forced static gate stop"):
        run(cfg)

    assert not output_root.exists()
    assert not cohort_root.exists()
    assert not candidate_root.exists()
