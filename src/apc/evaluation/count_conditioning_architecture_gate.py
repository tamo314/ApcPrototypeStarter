"""Conditioning architecture comparison gate (Phase A.1 Post-Correction Task
A1-R005D-006, `docs/CODEX_TASKS_A1_R005_RETRY.md`, STOP GATE).

`docs/DECISIONS.md` ADR-0034 (Task A1-R005D-005) found that capacity/
training increases alone (`primitive_train.steps` x2/x4, `primitive_rank`
16, `arg_dim` x2, each isolated against A1-R005D-004's own baseline) do not
close `COUNT`'s causal gap, and pointed at F3 (`docs/design-docs/
PARAMETERIZED_PRIMITIVE_RETRY.md`: "additive conditioning is too weak to
make the residual actually depend on the argument") as the leading
remaining candidate. This task compares `apc.primitives.conditioning.
ConditionedPrimitive`'s additive formula (**V0**) against two alternative
conditioning formulas that change *how* the argument affects the residual --
`FiLMConditionedPrimitive` (**V1**, gated multiplicative) and
`BasisModulatedConditionedPrimitive` (**V2**, basis modulation, the task
text's own "optional" variant) -- see `apc.primitives.conditioning`'s own
module docstring for each formula.

## Maximal reuse of A1-R005D-004's validated machinery

This module reuses `apc.evaluation.count_counterfactual_gate` completely
unchanged in its scientific behavior: the same task-blind frozen Stable
Core pretraining, the same counterfactual-group generation/flattening
(`generate_count_counterfactual_groups`), and the same three-arm Correct /
effectful Wrong argument / None evaluation. That module's own A1-R005D-006
update threads a `variant: ConditioningVariant` parameter through
`_build_primitive_bank`/`_train_primitives`/`run_count_counterfactual_gate`/
`run_count_counterfactual_gate_multi_seed`, defaulting to
`ConditioningVariant.ADDITIVE` (byte-identical to every pre-A1-R005D-006
call) -- this module's only job is to call that generalized entry point once
per variant and compare the results, exactly mirroring `apc.evaluation.
count_capacity_sweep_gate` (A1-R005D-005)'s own relationship to the same
underlying gate.

## Keep Stable Core and data fixed

A1-R005D-006's own "Work" item says "Keep Stable Core and data fixed."
`run_conditioning_architecture_comparison` therefore takes one
`base_config` and runs every variant against it unmodified (no capacity or
training-budget change per variant) -- the conditioning formula is this
experiment's only controlled variable. `base_config` is expected to be
A1-R005D-004's own baseline (`configs/phase_a1_count_counterfactual_gate.
yaml`'s `primitive_rank=8`/`primitive_train.steps=10000`/`arg_dim=16`),
matching every earlier retry task's "isolate exactly one variable" discipline
(`docs/DECISIONS.md` ADR-0034's own reasoning for the same choice), so V0's
own result under this module is directly comparable to A1-R005D-004's
already-recorded number (ADR-0033) -- V0 is re-run here (not spliced in from
that earlier run) so every variant's report sits in the same self-consistent
artifact layout, produced by the same code path, in the same run.

## No early stopping

Unlike A1-R005D-005's staged sweep (which stops at the first passing stage
to bound compute across four stages), this task runs **all** three variants
unconditionally: A1-R005D-006's own acceptance ("choose by causal gap")
presupposes having every candidate's causal gap to compare, not merely the
first one that happens to pass. Each variant is cheap at this experiment's
fixed baseline capacity/budget (comparable to A1-R005D-004's own ~410s/seed),
so running all three costs roughly 3x one D-004 gate run, not a multi-stage
budget escalation.

## Selection rule

A1-R005D-006's "Acceptance" gives three requirements, implemented here as:

1. **"report parameter counts."** Every `ConditioningVariantReport` carries
   `primitive_param_count` (`PrimitiveBank.total_parameter_count()`, deter-
   ministic across seeds for one variant since the architecture -- not the
   learned weights -- determines parameter count).
2. **"do not choose a model that raises Correct and Wrong together."** A
   non-baseline variant is *disqualified* from selection if its mean Correct
   **and** mean effectful-Wrong-argument both exceed V0/additive's own (the
   `baseline_variant`) -- that pattern suggests the variant became more
   confident/plausible in general, not more argument-*selective*, which the
   causal-gap number alone would not necessarily catch (a variant could
   raise both arms by a similar amount and still show an improved-looking
   gap). V0 itself is never disqualified (there is nothing to compare it
   against).
3. **"choose by causal gap."** Among the current gate's own thresholds
   (`apc.evaluation.count_counterfactual_gate`'s `CORRECT_THRESHOLD`/
   `EFFECTFUL_WRONG_ARGUMENT_CEILING`/`NONE_CEILING`/`MIN_CAUSAL_GAP`, the
   same numbers A1-R005D-004/A1-R005D-005 already gate on), `selected_
   variant` is the highest-causal-gap variant among the *passing*,
   non-disqualified candidates (`None` if none pass -- matching A1-R005D-005's
   own `selected_stage: str | None` convention). `best_by_causal_gap` is
   reported separately and unconditionally (highest causal gap among
   non-disqualified variants regardless of pass/fail), since A1-R005D-006's
   own STOP condition ("STOP if controlled COUNT still cannot pass") still
   needs to say *which* variant came closest even when none actually passed
   -- exactly the role `docs/DECISIONS.md` ADR-0034 played for A1-R005D-005's
   own four failing stages.

## Scope discipline

Same restriction as every earlier Phase A.1 primitive gate: `apc.plastic`,
`apc.consolidation`, and `apc.meta` are never imported here; no `Router` is
imported or reachable. This module builds and discards a fresh single-
primitive `PrimitiveBank` per (variant, seed) pair; no promotion to a
persistent/`STABLE` bank, consolidation, or composition. `apc.primitives.
conditioning.FiLMConditionedPrimitive`/`BasisModulatedConditionedPrimitive`
are the only new architecture introduced by this task -- everything else
(Stable Core, data, oracle routing, evaluation) is reused unmodified.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apc.evaluation.count_counterfactual_gate import (
    CORRECT_THRESHOLD,
    DEFAULT_SEEDS,
    EFFECTFUL_WRONG_ARGUMENT_CEILING,
    MIN_CAUSAL_GAP,
    NONE_CEILING,
    CountCounterfactualGateConfig,
    CountCounterfactualGateMultiSeedReport,
    run_count_counterfactual_gate_multi_seed,
)
from apc.primitives.conditioning import DEFAULT_NUM_BASIS_VECTORS, ConditioningVariant

__all__ = [
    "CONDITIONING_VARIANTS",
    "ConditioningVariantReport",
    "ConditioningArchitectureComparisonReport",
    "run_conditioning_architecture_comparison",
]

# Order matches docs/CODEX_TASKS_A1_R005_RETRY.md A1-R005D-006's own
# "Variants" list: "V0 additive, V1 FiLM/gated multiplicative, optional V2
# basis modulation."
CONDITIONING_VARIANTS: tuple[ConditioningVariant, ...] = (
    ConditioningVariant.ADDITIVE,
    ConditioningVariant.FILM,
    ConditioningVariant.BASIS,
)

_BASELINE_VARIANT = ConditioningVariant.ADDITIVE


@dataclass(frozen=True)
class ConditioningVariantReport:
    """One conditioning variant's full multi-seed gate result."""

    variant: str
    multi_seed: CountCounterfactualGateMultiSeedReport
    primitive_param_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "multi_seed": self.multi_seed.to_dict(),
            "primitive_param_count": self.primitive_param_count,
        }


@dataclass(frozen=True)
class ConditioningArchitectureComparisonReport:
    """A1-R005D-006's overall verdict across all three conditioning variants
    (module docstring, "Selection rule")."""

    variants: tuple[ConditioningVariantReport, ...]
    baseline_variant: str
    disqualified_variants: tuple[str, ...]
    best_by_causal_gap: str | None
    selected_variant: str | None
    any_variant_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": [variant.to_dict() for variant in self.variants],
            "baseline_variant": self.baseline_variant,
            "disqualified_variants": list(self.disqualified_variants),
            "best_by_causal_gap": self.best_by_causal_gap,
            "selected_variant": self.selected_variant,
            "any_variant_passed": self.any_variant_passed,
        }


@dataclass(frozen=True)
class _VariantSelectionInput:
    """The only fields `_select_variant` needs from one variant's result --
    factored out from `ConditioningVariantReport` so the selection rule
    itself is a pure function of plain values, testable without constructing
    a full `CountCounterfactualGateMultiSeedReport` (and its nested per-seed
    reports) by hand."""

    variant: str
    mean_correct_exact_match: float
    mean_effectful_wrong_argument_exact_match: float
    mean_causal_gap: float
    passed: bool


def _selection_input(report: ConditioningVariantReport) -> _VariantSelectionInput:
    return _VariantSelectionInput(
        variant=report.variant,
        mean_correct_exact_match=report.multi_seed.mean_correct_exact_match,
        mean_effectful_wrong_argument_exact_match=(
            report.multi_seed.mean_effectful_wrong_argument_exact_match
        ),
        mean_causal_gap=report.multi_seed.mean_causal_gap,
        passed=report.multi_seed.passed,
    )


def _select_variant(
    inputs: Sequence[_VariantSelectionInput], baseline_variant: str
) -> tuple[tuple[str, ...], str | None, str | None]:
    """A1-R005D-006's selection rule (module docstring, "Selection rule"):
    returns `(disqualified_variants, best_by_causal_gap, selected_variant)`.

    `baseline_variant` is never disqualified (there is nothing to compare it
    against); every other variant is disqualified if its mean Correct *and*
    mean effectful-Wrong-argument both exceed the baseline's own ("do not
    choose a model that raises Correct and Wrong together"). `best_by_
    causal_gap` is the highest-causal-gap variant among the non-disqualified
    ones, regardless of pass/fail; `selected_variant` is the same but
    restricted to variants that actually passed (`None` if none did).
    """
    baseline = next(v for v in inputs if v.variant == baseline_variant)
    disqualified: list[str] = []
    eligible: list[_VariantSelectionInput] = [baseline]
    for candidate in inputs:
        if candidate.variant == baseline.variant:
            continue
        raises_correct_and_wrong_together = (
            candidate.mean_correct_exact_match > baseline.mean_correct_exact_match
            and candidate.mean_effectful_wrong_argument_exact_match
            > baseline.mean_effectful_wrong_argument_exact_match
        )
        if raises_correct_and_wrong_together:
            disqualified.append(candidate.variant)
        else:
            eligible.append(candidate)

    best_by_causal_gap = (
        max(eligible, key=lambda v: v.mean_causal_gap).variant if eligible else None
    )
    passing = [v for v in eligible if v.passed]
    selected_variant = max(passing, key=lambda v: v.mean_causal_gap).variant if passing else None
    return tuple(disqualified), best_by_causal_gap, selected_variant


def run_conditioning_architecture_comparison(
    base_config: CountCounterfactualGateConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    run_dir: str | Path | None = None,
    num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
    correct_threshold: float = CORRECT_THRESHOLD,
    effectful_wrong_argument_ceiling: float = EFFECTFUL_WRONG_ARGUMENT_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
) -> ConditioningArchitectureComparisonReport:
    """Run A1-R005D-006's full comparison: every entry of `CONDITIONING_
    VARIANTS`, each through the same `seeds`-seed gate policy against
    `base_config` unmodified (module docstring, "Keep Stable Core and data
    fixed"). Writes `run_dir/<variant>/{report.json, seed_<n>/...}` per
    variant when `run_dir` is given. The four threshold parameters default
    to A1-R005D-004's own numbers, matching `apc.evaluation.
    count_capacity_sweep_gate.run_count_capacity_sweep_gate`'s own
    convention (mainly so tests can force a pass/fail without depending on
    real training dynamics).
    """
    run_dir_path = Path(run_dir) if run_dir is not None else None
    variant_reports: list[ConditioningVariantReport] = []

    for variant in CONDITIONING_VARIANTS:
        variant_run_dir = run_dir_path / variant.value if run_dir_path is not None else None
        multi_seed = run_count_counterfactual_gate_multi_seed(
            base_config,
            seeds=seeds,
            run_dir=variant_run_dir,
            correct_threshold=correct_threshold,
            effectful_wrong_argument_ceiling=effectful_wrong_argument_ceiling,
            none_ceiling=none_ceiling,
            min_causal_gap=min_causal_gap,
            variant=variant,
            num_basis=num_basis,
        )
        variant_report = ConditioningVariantReport(
            variant=variant.value,
            multi_seed=multi_seed,
            primitive_param_count=multi_seed.per_seed[0].primitive_param_count,
        )
        variant_reports.append(variant_report)
        if variant_run_dir is not None:
            variant_run_dir.mkdir(parents=True, exist_ok=True)
            (variant_run_dir / "report.json").write_text(
                json.dumps(variant_report.to_dict(), indent=2), encoding="utf-8"
            )

    disqualified, best_by_causal_gap, selected_variant = _select_variant(
        [_selection_input(v) for v in variant_reports], _BASELINE_VARIANT.value
    )
    any_variant_passed = any(v.multi_seed.passed for v in variant_reports)

    return ConditioningArchitectureComparisonReport(
        variants=tuple(variant_reports),
        baseline_variant=_BASELINE_VARIANT.value,
        disqualified_variants=disqualified,
        best_by_causal_gap=best_by_causal_gap,
        selected_variant=selected_variant,
        any_variant_passed=any_variant_passed,
    )
