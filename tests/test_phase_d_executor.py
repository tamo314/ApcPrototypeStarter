"""Focused fail-closed checks for the D-008 execution boundary."""

from __future__ import annotations

import pytest

from apc.evaluation.phase_d_executor import (
    MODEL_SEEDS,
    PhaseDExecutorConfig,
    PhaseDStopGateError,
    _assert_new_namespace,
    dry_run_manifest,
)


def test_d008_dry_run_records_only_the_fixed_authorization() -> None:
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
