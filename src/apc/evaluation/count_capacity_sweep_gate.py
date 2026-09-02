"""Minimal capacity/training sweep gate (Phase A.1 Post-Correction Task
A1-R005D-005, `docs/CODEX_TASKS_A1_R005_RETRY.md`).

`docs/DECISIONS.md` ADR-0033 (Task A1-R005D-004) found that forcing maximal
counterfactual argument diversity (`argument_effect_rate=1.0` every seed)
did not close `COUNT`'s causal gap at A1-R005's own baseline budget
(`primitive_rank=8`, `primitive_train.steps=10000`) -- F4 ("training does
not force argument use") is not sufficient by itself to explain A1-R005's
failure. Per `docs/exec-plans/active/A1_R005_RETRY.md` R005-M4 ("Minimal
sufficient capacity: Only after M3 setup is valid" -- not "only after M3
passes") and ADR-0033's own Consequence section, this task proceeds against
that now-*identifiable* result without re-running A1-R005D-004: this module
reuses `apc.evaluation.count_counterfactual_gate`'s validated machinery
(counterfactual-group generation, frozen task-blind Stable Core, one
`ConditionedPrimitive` family, the same three-arm Correct/effectful Wrong
argument/None ablation) completely unchanged, and sweeps only the capacity/
training knobs `docs/CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-005 names.

## Staged sweep: one variable at a time, in the task's own order

A1-R005D-005's "Work" item lists four stages: "1. steps x2, 2. steps x4 if
needed, 3. rank 16, 4. arg_dim x2". Read literally this is ambiguous between
a *cumulative* escalation (each stage stacks on top of the previous one's
winning change) and an *isolated* sweep (each stage changes exactly one knob
relative to the A1-R005D-004 baseline, independent of the others). This
module adopts the isolated reading:

- stage 1 (`steps_x2`): `primitive_train.steps` doubled (10000 -> 20000),
  everything else at the A1-R005D-004 baseline;
- stage 2 (`steps_x4`, only reached if stage 1 fails): `primitive_train.steps`
  quadrupled (10000 -> 40000), everything else at baseline;
- stage 3 (`rank_16`, only reached if stage 2 fails or was skipped because
  stage 1 already passed): `primitive_rank` doubled (8 -> 16), `steps` back
  at the *baseline* 10000, `arg_dim` at baseline;
- stage 4 (`arg_dim_x2`, only reached if stage 3 fails): `arg_dim` doubled
  (16 -> 32), `primitive_rank`/`steps` back at baseline.

Reasons for the isolated reading over the cumulative one:

1. Every earlier task in this retry sequence (A1-R005D-001 through
   A1-R005D-004) isolates exactly one variable relative to the previous
   failing run, specifically so a passing or failing result can be
   attributed to that one change (`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s
   entire "Required diagnostic order" is built on this discipline). A
   cumulative sweep would conflate "more steps" with "more rank" the moment
   stage 3 ran on top of stage 2's already-quadrupled step count, undoing
   that discipline right at the one point in the queue explicitly reserved
   for capacity/training (R005-M4's own title, "Minimal sufficient
   capacity").
2. "Select smallest robust passing configuration" (A1-R005D-005's own
   "Acceptance") is only a well-posed question if each candidate
   configuration is comparable to the single A1-R005D-004 baseline it is
   trying to beat, not to whatever the previous stage happened to leave
   the config at.
3. `docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "No blind scaling" rule
   ("Do not begin with more rank, more steps, larger Stable Core, or larger
   arg embedding") warns against *combining* scale-ups without a specific
   diagnostic reason for each one; isolating them is the direct way to keep
   that discipline while still running the sweep A1-R005D-005 itself asks
   for.

This interpretation choice is recorded here (and in `docs/DECISIONS.md`) per
`AGENTS.md`'s guidance for underspecified acceptance/work criteria.

## Stopping early

`run_count_capacity_sweep_gate` runs stages *in order* and stops at the
first stage whose full 5-seed gate (`apc.evaluation.count_counterfactual_gate.
run_count_counterfactual_gate_multi_seed`, same acceptance thresholds as
A1-R005D-004: Correct >= 0.90, effectful Wrong argument <= 0.30, causal gap
>= 0.50, plus the same filled-in `None <= 0.30` convention ADR-0033 already
applied) reports `passed=True` -- matching A1-R005D-005's own "if needed"/
"select smallest ... configuration" framing (the smallest of the four fixed
candidates that passes, not a search over arbitrarily many configurations).
If no stage passes, `sweep_exhausted_without_pass=True` and
`selected_stage=None`; A1-R005D-005's own "Acceptance" states "If none pass,
continue to D-006" -- implementing D-006 is explicitly not done here.

Each stage runs the *same* seed policy as every other Phase A.1 gate
(`DEFAULT_SEEDS`, `MIN_GATE_SEEDS=5`) rather than a cheaper few-seed
screening pass promoted to 5 seeds only for a winning candidate: this keeps
"passing" measured identically to A1-R005D-004's own verdict at every stage,
so a stage's `passed=True` carries the same evidentiary weight the rest of
this retry sequence relies on, at the cost of running the full budget for
every stage attempted (still bounded: at most 4 stages, each `<=` a few
times A1-R005D-004's own ~410s/seed x 5 seeds).

## Logging

`primitive_train`/`core_train` per-step learning curves and the argument-
path/content-path gradient norms (`apc.evaluation.count_counterfactual_gate`'s
own A1-R005D-005 update) are written to `run_dir/<stage_name>/seed_<n>/
{core_metrics.jsonl, primitive_metrics.jsonl}` by `run_count_counterfactual_
gate_multi_seed` itself, unmodified -- this module adds no additional
per-step logging of its own, only the stage-level aggregation.

## Scope discipline

Same restriction as every earlier Phase A.1 primitive gate: `apc.plastic`,
`apc.consolidation`, and `apc.meta` are never imported here; no `Router` is
imported or reachable. No architecture change is made in this module --
`ConditionedPrimitive`'s additive-conditioning formula is untouched; only
its declared capacity (`primitive_rank`, `arg_dim`) and its optimization
budget (`primitive_train.steps`) vary across stages
(`docs/CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-006 owns conditioning-
*architecture* comparisons, explicitly out of scope here).
"""

from __future__ import annotations

import dataclasses
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

__all__ = [
    "STAGE_NAMES",
    "capacity_sweep_stage_configs",
    "CapacitySweepStageReport",
    "CountCapacitySweepReport",
    "run_count_capacity_sweep_gate",
]

# Order matches docs/CODEX_TASKS_A1_R005_RETRY.md A1-R005D-005's own "Staged
# sweep" list exactly: "1. steps x2  2. steps x4 if needed  3. rank 16
# 4. arg_dim x2".
STAGE_NAMES: tuple[str, ...] = ("steps_x2", "steps_x4", "rank_16", "arg_dim_x2")


def capacity_sweep_stage_configs(
    base_config: CountCounterfactualGateConfig,
) -> tuple[tuple[str, CountCounterfactualGateConfig], ...]:
    """The four staged-sweep configs, each isolating exactly one knob
    relative to `base_config` (module docstring, "Staged sweep: one variable
    at a time"). `base_config` itself is expected to already be
    A1-R005D-004's own baseline (`configs/phase_a1_count_counterfactual_gate.
    yaml`'s `primitive_rank=8`/`primitive_train.steps=10000`/`arg_dim=16`),
    but this function does not assume any particular baseline values -- it
    only ever multiplies/replaces `base_config`'s own fields, so it stays
    correct if the baseline config file's numbers ever change.
    """
    baseline_steps = base_config.primitive_train.steps
    baseline_arg_dim = base_config.arg_dim
    stages = (
        (
            "steps_x2",
            dataclasses.replace(
                base_config,
                primitive_train=dataclasses.replace(
                    base_config.primitive_train, steps=baseline_steps * 2
                ),
            ),
        ),
        (
            "steps_x4",
            dataclasses.replace(
                base_config,
                primitive_train=dataclasses.replace(
                    base_config.primitive_train, steps=baseline_steps * 4
                ),
            ),
        ),
        ("rank_16", dataclasses.replace(base_config, primitive_rank=16)),
        ("arg_dim_x2", dataclasses.replace(base_config, arg_dim=baseline_arg_dim * 2)),
    )
    assert tuple(name for name, _ in stages) == STAGE_NAMES
    return stages


@dataclass(frozen=True)
class CapacitySweepStageReport:
    """One staged-sweep candidate's full 5-seed gate result."""

    name: str
    swept_field: str
    swept_value: int
    multi_seed: CountCounterfactualGateMultiSeedReport
    primitive_param_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "swept_field": self.swept_field,
            "swept_value": self.swept_value,
            "multi_seed": self.multi_seed.to_dict(),
            "primitive_param_count": self.primitive_param_count,
        }


_STAGE_SWEPT_FIELD: dict[str, str] = {
    "steps_x2": "primitive_train.steps",
    "steps_x4": "primitive_train.steps",
    "rank_16": "primitive_rank",
    "arg_dim_x2": "arg_dim",
}


def _swept_value(name: str, config: CountCounterfactualGateConfig) -> int:
    field = _STAGE_SWEPT_FIELD[name]
    if field == "primitive_train.steps":
        return config.primitive_train.steps
    if field == "primitive_rank":
        return config.primitive_rank
    if field == "arg_dim":
        return config.arg_dim
    raise KeyError(field)  # pragma: no cover - _STAGE_SWEPT_FIELD is exhaustive


@dataclass(frozen=True)
class CountCapacitySweepReport:
    """A1-R005D-005's overall verdict: the staged-sweep candidates actually
    run (stops at the first `passed=True`, or exhausts all four), and which
    one -- if any -- is the "smallest robust passing configuration"
    (module docstring, "Stopping early")."""

    stages: tuple[CapacitySweepStageReport, ...]
    selected_stage: str | None
    sweep_exhausted_without_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "selected_stage": self.selected_stage,
            "sweep_exhausted_without_pass": self.sweep_exhausted_without_pass,
        }


def run_count_capacity_sweep_gate(
    base_config: CountCounterfactualGateConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    run_dir: str | Path | None = None,
    correct_threshold: float = CORRECT_THRESHOLD,
    effectful_wrong_argument_ceiling: float = EFFECTFUL_WRONG_ARGUMENT_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
) -> CountCapacitySweepReport:
    """Run A1-R005D-005's staged sweep against `base_config` (expected to be
    A1-R005D-004's own baseline), stopping at the first stage whose full
    `seeds`-seed gate passes. Writes `run_dir/<stage_name>/seed_<n>/...` per
    stage (`run_count_counterfactual_gate_multi_seed`'s own artifact layout)
    when `run_dir` is given. The four threshold parameters default to
    A1-R005D-004's own numbers (same bar every stage is trying to clear);
    they exist mainly so tests can force a stage to pass or fail without
    depending on real training dynamics, matching `apc.evaluation.
    count_counterfactual_gate.run_count_counterfactual_gate_multi_seed`'s
    own convention.
    """
    run_dir_path = Path(run_dir) if run_dir is not None else None
    stage_reports: list[CapacitySweepStageReport] = []
    selected_stage: str | None = None

    for name, stage_config in capacity_sweep_stage_configs(base_config):
        stage_run_dir = run_dir_path / name if run_dir_path is not None else None
        multi_seed = run_count_counterfactual_gate_multi_seed(
            stage_config,
            seeds=seeds,
            run_dir=stage_run_dir,
            correct_threshold=correct_threshold,
            effectful_wrong_argument_ceiling=effectful_wrong_argument_ceiling,
            none_ceiling=none_ceiling,
            min_causal_gap=min_causal_gap,
        )
        stage_report = CapacitySweepStageReport(
            name=name,
            swept_field=_STAGE_SWEPT_FIELD[name],
            swept_value=_swept_value(name, stage_config),
            multi_seed=multi_seed,
            primitive_param_count=multi_seed.per_seed[0].primitive_param_count,
        )
        stage_reports.append(stage_report)
        if stage_run_dir is not None:
            (stage_run_dir / "report.json").write_text(
                json.dumps(stage_report.to_dict(), indent=2), encoding="utf-8"
            )
        if multi_seed.passed:
            selected_stage = name
            break

    return CountCapacitySweepReport(
        stages=tuple(stage_reports),
        selected_stage=selected_stage,
        sweep_exhausted_without_pass=selected_stage is None,
    )
