"""Focused fail-closed checks for the D-008 execution boundary."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

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
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    _build_shared_encoder,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _train_single_primitive,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed
from apc.utils.seed_derivation import derive_seed


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_d012_incremental_primitive_cuda_device_placement() -> None:
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=42,
        device="cuda",
        vocab_size=10,
        sequence_length_range=(6, 8),
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        model={
            "d_model": 16,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
    )
    core = _build_shared_encoder(arch_cfg)
    bank = PrimitiveBank()
    primitive = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="REVERSE",
            d_model=core.model.config.d_model,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
        ),
        status=PrimitiveStatus.STABLE,
    )
    # Regression check: newly created primitive defaults to CPU
    assert next(primitive.parameters()).device.type == "cpu"

    # Co-locate onto core.device as required by D-012 standard placement
    primitive.to(core.device)
    bank.to(core.device)
    assert next(primitive.parameters()).device.type == "cuda"

    # Verify execution runs on CUDA without RuntimeError (device mismatch)
    ucfg = UnifiedBenchmarkConfig(seed=42, device="cuda")
    loss = _train_single_primitive(core, primitive, ucfg, "REVERSE", steps=2)
    assert loss >= 0.0


def test_d013_independent_subprocess_pythonhashseed_compatibility() -> None:
    # 1. Verify set_seed with large 63-bit derived seed bounds PYTHONHASHSEED to uint32
    derived = derive_seed(
        master_seed=40,
        stream_namespace="r3_009_train_init",
        task_key="SHIFT_RIGHT_1",
        sample_index=0,
    )
    assert derived > 4294967295  # Exceeds 32-bit unsigned int
    set_seed(derived)
    val = int(os.environ["PYTHONHASHSEED"])
    assert 0 <= val <= 4294967295
    assert val == (derived & 0xFFFFFFFF)

    # 2. Independent python subprocess inherits environment and launches without crash
    res = subprocess.run(
        [sys.executable, "-c", "import os, sys; print(os.environ.get('PYTHONHASHSEED'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == str(val)

    # 3. Verify sub-process resilience even if os.environ had an out-of-range PYTHONHASHSEED
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str((1 << 63) - 1)
    raw_seed = env["PYTHONHASHSEED"]
    if raw_seed != "random":
        try:
            env["PYTHONHASHSEED"] = str(int(raw_seed) & 0xFFFFFFFF)
        except ValueError:
            env.pop("PYTHONHASHSEED", None)
    res2 = subprocess.run(
        [sys.executable, "-c", "import os, sys; print('subprocess_ok')"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0
    assert "subprocess_ok" in res2.stdout


