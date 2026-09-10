"""Task B-C005REC-004L-ENV1: SciPy dependency contract & clean-import regression.

Confirms structurally (not by grepping text) that `pyproject.toml` declares
`scipy` as a core runtime dependency of the installed `apc` package -- Stage A
of the task found an unconditional, unguarded `from scipy.stats import beta`
inside `apc.evaluation.functional_metrics_v2._beta_quantile`, reached from the
normal (non-test, non-optional) import graph of the installed package via
`apc.evaluation.{paired_integration_regression,paired_baseline_repair,
shift_functional_generalization_repair,safe_bounded_verification}` -- so
`CORE_RUNTIME_REQUIRED` is the correct classification, not `DEV_OR_TEST_ONLY`
or `OPTIONAL_RUNTIME`.

This file does not change, and does not re-derive, any statistical semantics
in `functional_metrics_v2.py`; the numeric check below only pins the already
existing `one_sided_exact_bounds` contract to a fixed expected value so a
future regression in the dependency contract (e.g. scipy silently missing
again) fails loudly here instead of only inside the 28 downstream tests.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

# scipy.stats.beta.ppf(0.005, 30, 3) / beta.ppf(0.995, 31, 2) via the real
# _beta_quantile call path (successes=30, trials=32, alpha=0.005 both sides),
# computed directly from this task's own clean-room scipy 1.18.1 install
# (runs/phase_b_b2_model_bundle_recovery/rec004l_env1/run_001/
# environment_requalification.json) -- not hand-derived.
_EXPECTED_LOWER = 0.7411892141721008
_EXPECTED_UPPER = 0.9967194234603913


def _load_pyproject() -> dict:
    with _PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def test_scipy_declared_as_core_runtime_dependency() -> None:
    """`scipy` must be a structurally-parsed member of `[project.dependencies]`.

    Classified `CORE_RUNTIME_REQUIRED` (see
    `docs/CODEX_TASKS_PHASE_B_B2_SCIPY_DEPENDENCY_REPAIR.md` Stage A/B and
    `runs/phase_b_b2_model_bundle_recovery/rec004l_env1/run_001/
    dependency_contract_decision.json`): it must live in core `dependencies`,
    not the `dev`/`plots` optional-dependency groups, since the unconditional
    import site has no optional-feature guard around it.
    """
    data = _load_pyproject()
    core_deps = data["project"]["dependencies"]
    names = {Requirement(dep).name for dep in core_deps}
    assert "scipy" in names, (
        f"scipy must be declared in [project].dependencies, found: {sorted(names)}"
    )

    optional = data["project"].get("optional-dependencies", {})
    for group, deps in optional.items():
        group_names = {Requirement(dep).name for dep in deps}
        assert "scipy" not in group_names, (
            f"scipy must not be duplicated into the '{group}' optional group; "
            "it is a core dependency, not dev/test/plots-only"
        )


def test_scipy_importable() -> None:
    """`scipy` must actually import in whatever environment runs this test."""
    import scipy  # noqa: F401
    from scipy.stats import beta, norm  # noqa: F401


def test_one_sided_exact_bounds_unchanged_with_scipy_present() -> None:
    """Pin `one_sided_exact_bounds`'s existing numeric contract post-repair.

    Same call already exercised by `tests/test_functional_metrics_v2.py`;
    repeated here, scoped to this task, to fail loudly if a future dependency
    change silently alters `_beta_quantile`'s exact Beta-quantile behavior.
    """
    from apc.evaluation.functional_metrics_v2 import one_sided_exact_bounds

    lower, upper = one_sided_exact_bounds(
        successes=30, trials=32, alpha_lower=0.005, alpha_upper=0.005
    )
    assert lower == pytest.approx(_EXPECTED_LOWER, abs=1e-9)
    assert upper == pytest.approx(_EXPECTED_UPPER, abs=1e-9)
