"""Re-analysis of the existing, frozen A1-R005 run (Task A1-R005D-001,
`docs/CODEX_TASKS_A1_R005_RETRY.md`: "Re-analyze existing R005 artifacts ...
expand diagnostics without retraining").

A1-R005 (`apc.evaluation.parameterized_primitive_gate`, STOP GATE, H2c)
failed (`docs/DECISIONS.md` ADR-0029): `runs/phase_a1_parameterized_
primitive_gate/` is frozen evidence of that failure and must not be
overwritten (`docs/EXPERIMENT_PLAN_A1_R005_RETRY.md` section 2, "Existing
result is frozen"). This module only *reads* that run directory (or any
other `parameterized_primitive_gate`-shaped run directory) and derives
additional diagnostics from it; it never trains a model, never runs a
forward pass through one, and never writes into the run directory it
reads from.

## What is and is not recoverable without a model

`apc.evaluation.parameterized_primitive_gate.ParameterizedPrimitiveGateReport`
persists only aggregate exact-match scores per arm/operation
(`per_operation_correct_exact_match` etc.) -- it never persists per-example
model predictions or token-level logits. Two consequences:

1. **Recoverable without any model** (this module computes both): per-
   operation/per-seed exact-match numbers already saved in each `seed_<n>/
   report.json` (this module just re-aggregates them into one table), plus
   `argument_effect_rate` -- `docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s
   "Wrong-argument control rule", `argument_effect = [target(correct_arg)
   != target(wrong_arg)]` -- and target-output-length statistics. Both are
   pure functions of the deterministic oracle interpreter
   (`apc.environments.primitive_call.PrimitiveCall.execute`) and the
   deterministic generator (`apc.environments.generator.TaskGenerator.
   generate_online`, "reproduces byte-identical examples ... independent of
   call order"), never of the trained Stable Core or primitive bank.
2. **Not recoverable without rerunning the trained model**: token accuracy
   and a wrong-argument exact-match score restricted to the effectful
   subset. Both need to know what the *model* predicted per example, and
   that was never persisted. Recomputing them would require reproducing
   the exact trained primitive bank (rerunning `_pretrain_frozen_stable_
   core` and `_train_primitives`), which is retraining in every sense this
   task is scoped to avoid. `reanalyze_seed`/`reanalyze_run` report these
   fields explicitly as `None` with `NOT_RECOVERABLE_REASON` attached
   rather than silently omitting them, so "raw and effectful wrong-argument
   scores [stay] clearly separated" (Task A1-R005D-001's acceptance
   criterion) instead of the gap being invisible.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apc.environments.generator import (
    Example,
    build_mixed_operation_generator,
    oracle_call_for_example,
)
from apc.environments.primitive_call import PrimitiveCall
from apc.evaluation.parameterized_primitive_gate import _wrong_argument_value

__all__ = [
    "NOT_RECOVERABLE_REASON",
    "SeedReanalysis",
    "RunReanalysis",
    "reconstruct_unseen_eval_examples",
    "compute_argument_effect",
    "reanalyze_seed",
    "reanalyze_run",
]

NOT_RECOVERABLE_REASON = (
    "not recoverable from existing run artifacts without rerunning the trained model: "
    "apc.evaluation.parameterized_primitive_gate persists only aggregate exact-match "
    "scores, never per-example model predictions, and Task A1-R005D-001 is explicitly "
    "scoped to expand diagnostics 'without retraining'"
)


def reconstruct_unseen_eval_examples(seed_config: dict[str, Any]) -> list[Example]:
    """Deterministically regenerate the unseen evaluation batch one seed of
    the original A1-R005 run used, touching no model and taking no gradient
    step.

    `apc.environments.generator.TaskGenerator.generate_online` is a pure
    function of `(seed, step, split)` plus the generator's own construction
    parameters. `apc.evaluation.parameterized_primitive_gate._pretrain_
    frozen_stable_core` builds its generator via `apc.evaluation.shared_
    core_generalization.train_shared_core`, which itself calls
    `apc.environments.generator.build_mixed_operation_generator(seed=config.
    seed, operation_names=config.operation_names, vocab_size=config.
    vocab_size, sequence_length_range=config.sequence_length_range,
    permute_symbols=config.permute_symbols)` with `permute_symbols` always
    `False` for this gate (`SharedCoreGateConfig(..., permute_symbols=False,
    ...)`); `run_parameterized_primitive_gate` then draws `unseen_examples =
    core.generator.generate_online(config.num_unseen_eval_examples, step=0,
    split="test")`. `seed_config` is `ParameterizedPrimitiveGateReport.
    config.to_dict()` as already saved in every `seed_<n>/report.json`, so
    replaying those same calls from it regenerates byte-identical examples.
    `reanalyze_seed` cross-checks this against the saved report's own
    `per_operation_eval_counts` and raises if they disagree.
    """
    generator = build_mixed_operation_generator(
        seed=seed_config["seed"],
        operation_names=tuple(seed_config["operation_names"]),
        vocab_size=seed_config["vocab_size"],
        sequence_length_range=tuple(seed_config["sequence_length_range"]),
        permute_symbols=False,
    )
    return generator.generate_online(
        seed_config["num_unseen_eval_examples"], step=0, split="test"
    )


def compute_argument_effect(example: Example) -> bool:
    """Whether A1-R005's own deterministic "Wrong argument" substitution
    (`apc.evaluation.parameterized_primitive_gate._wrong_argument_value`,
    "+1 modulo the argument's own valid domain") actually changes the
    ground-truth target for `example`.

    `docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "Wrong-argument control rule":
    `argument_effect = [target(correct_arg) != target(wrong_arg)]`. Purely a
    property of the oracle interpreter (`PrimitiveCall.execute`, which
    mirrors `apc.environments.operations.Operation.apply` exactly) applied
    to `example`'s own content -- never the trained model.
    """
    correct_call = oracle_call_for_example(example)
    wrong_call = PrimitiveCall(
        operation=correct_call.operation,
        arguments=_wrong_argument_value(correct_call, example),
    )
    wrong_target = wrong_call.execute(example.input_tokens, example.vocab_size)
    return wrong_target != example.target_tokens


@dataclass(frozen=True)
class SeedReanalysis:
    """A1-R005D-001's expanded diagnostics for one seed of an existing
    `parameterized_primitive_gate` run: the four arms' existing per-
    operation exact-match numbers (re-exposed, not recomputed) alongside
    `argument_effect_rate` and target-output-length statistics (newly
    derived, model-free) and explicit not-recoverable markers for what
    would require rerunning the trained model."""

    seed: int
    operation_names: tuple[str, ...]
    num_unseen_eval_examples: int
    per_operation_eval_counts: dict[str, int]
    per_operation_correct_exact_match: dict[str, float]
    per_operation_wrong_argument_exact_match: dict[str, float]
    per_operation_wrong_family_exact_match: dict[str, float]
    per_operation_none_exact_match: dict[str, float]
    per_operation_argument_effect_rate: dict[str, float]
    per_operation_target_output_length_mean: dict[str, float]
    per_operation_effectful_wrong_argument_exact_match: dict[str, float | None]
    per_operation_correct_token_accuracy: dict[str, float | None]
    causal_gap: float
    not_recoverable: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))


def _load_seed_report(run_dir: Path, seed: int) -> dict[str, Any]:
    path = run_dir / f"seed_{seed}" / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def reanalyze_seed(run_dir: str | Path, seed: int) -> SeedReanalysis:
    """Re-analyze one seed's already-saved `seed_<n>/report.json`. Read-only
    with respect to `run_dir`: never writes into it, never constructs or
    trains a model."""
    run_dir = Path(run_dir)
    report = _load_seed_report(run_dir, seed)
    seed_config = report["config"]
    operation_names = tuple(seed_config["operation_names"])

    examples = reconstruct_unseen_eval_examples(seed_config)
    if len(examples) != report["num_unseen_eval_examples"]:
        raise ValueError(
            f"reconstructed {len(examples)} unseen eval examples for seed {seed}, "
            f"expected {report['num_unseen_eval_examples']} from the saved report -- "
            "generator reconstruction does not match the original run"
        )

    effect_hits: dict[str, int] = dict.fromkeys(operation_names, 0)
    effect_totals: dict[str, int] = dict.fromkeys(operation_names, 0)
    length_totals: dict[str, int] = dict.fromkeys(operation_names, 0)
    for example in examples:
        operation = oracle_call_for_example(example).operation
        effect_totals[operation] += 1
        if compute_argument_effect(example):
            effect_hits[operation] += 1
        length_totals[operation] += len(example.target_tokens)

    per_operation_eval_counts = dict(effect_totals)
    if per_operation_eval_counts != report["per_operation_eval_counts"]:
        raise ValueError(
            f"reconstructed per-operation eval counts {per_operation_eval_counts} for "
            f"seed {seed} do not match the saved report's "
            f"{report['per_operation_eval_counts']} -- generator reconstruction does "
            "not match the original run"
        )

    per_operation_argument_effect_rate = {
        name: effect_hits[name] / effect_totals[name] for name in operation_names
    }
    per_operation_target_output_length_mean = {
        name: length_totals[name] / effect_totals[name] for name in operation_names
    }
    not_recoverable = {
        "per_operation_effectful_wrong_argument_exact_match": NOT_RECOVERABLE_REASON,
        "per_operation_correct_token_accuracy": NOT_RECOVERABLE_REASON,
    }

    return SeedReanalysis(
        seed=seed,
        operation_names=operation_names,
        num_unseen_eval_examples=report["num_unseen_eval_examples"],
        per_operation_eval_counts=per_operation_eval_counts,
        per_operation_correct_exact_match=dict(report["per_operation_correct_exact_match"]),
        per_operation_wrong_argument_exact_match=dict(
            report["per_operation_wrong_argument_exact_match"]
        ),
        per_operation_wrong_family_exact_match=dict(
            report["per_operation_wrong_family_exact_match"]
        ),
        per_operation_none_exact_match=dict(report["per_operation_none_exact_match"]),
        per_operation_argument_effect_rate=per_operation_argument_effect_rate,
        per_operation_target_output_length_mean=per_operation_target_output_length_mean,
        per_operation_effectful_wrong_argument_exact_match=dict.fromkeys(operation_names),
        per_operation_correct_token_accuracy=dict.fromkeys(operation_names),
        causal_gap=report["causal_gap"],
        not_recoverable=not_recoverable,
    )


@dataclass(frozen=True)
class RunReanalysis:
    """A1-R005D-001's verdict for a whole run: `reanalyze_seed` per seed
    plus cross-seed means, mirroring `apc.evaluation.parameterized_
    primitive_gate.ParameterizedPrimitiveGateMultiSeedReport`'s own
    aggregation shape (mean over seeds), but per-operation for every arm
    rather than only for Correct."""

    run_dir: str
    seeds: tuple[int, ...]
    operation_names: tuple[str, ...]
    per_seed: tuple[SeedReanalysis, ...]
    mean_per_operation_correct_exact_match: dict[str, float]
    mean_per_operation_wrong_argument_exact_match: dict[str, float]
    mean_per_operation_wrong_family_exact_match: dict[str, float]
    mean_per_operation_none_exact_match: dict[str, float]
    mean_per_operation_argument_effect_rate: dict[str, float]
    mean_per_operation_target_output_length_mean: dict[str, float]
    not_recoverable: dict[str, str]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "seeds": list(self.seeds),
            "operation_names": list(self.operation_names),
            "per_seed": [seed_report.to_dict() for seed_report in self.per_seed],
            "mean_per_operation_correct_exact_match": self.mean_per_operation_correct_exact_match,
            "mean_per_operation_wrong_argument_exact_match": (
                self.mean_per_operation_wrong_argument_exact_match
            ),
            "mean_per_operation_wrong_family_exact_match": (
                self.mean_per_operation_wrong_family_exact_match
            ),
            "mean_per_operation_none_exact_match": self.mean_per_operation_none_exact_match,
            "mean_per_operation_argument_effect_rate": self.mean_per_operation_argument_effect_rate,
            "mean_per_operation_target_output_length_mean": (
                self.mean_per_operation_target_output_length_mean
            ),
            "not_recoverable": self.not_recoverable,
            "notes": list(self.notes),
        }


_NOTES: tuple[str, ...] = (
    "raw wrong-argument exact match (mean_per_operation_wrong_argument_exact_match / "
    "per-seed per_operation_wrong_argument_exact_match) mixes effectful and "
    "non-effectful examples: on a non-effectful example the deterministic '+1 modulo "
    "domain' wrong argument happens to produce the same ground-truth target as the "
    "correct argument, so a model can score 'correct' there by coincidence regardless "
    "of whether it is causally using the argument at all.",
    "mean_per_operation_argument_effect_rate is the fraction of the unseen eval batch "
    "where the wrong argument actually changes the ground-truth target; it is a "
    "structural property of the generator/interpreter, not of the trained model, and "
    "is high for all four operations at this batch size (see run artifact) -- so the "
    "raw wrong-argument score above is not explained by a lack of effectful examples.",
    "the effectful-restricted wrong-argument exact match that would isolate causal "
    "argument sensitivity from this confound is listed under not_recoverable: the "
    "original run never persisted per-example model predictions, only aggregate "
    "exact-match scores, so it cannot be reconstructed without rerunning the trained "
    "model (out of scope for this no-retraining re-analysis).",
    "mean_per_operation_target_output_length_mean is the mean ground-truth target "
    "length per operation (1 for COUNT/BIND; up to sequence_length_range[1] for SHIFT; "
    "up to sequence_length_range[1] // 2 for SELECT) -- structural context for reading "
    "per-operation Correct differences, per ADR-0029 finding 2's exact-match "
    "compounding-penalty hypothesis.",
)


def reanalyze_run(run_dir: str | Path, seeds: Sequence[int]) -> RunReanalysis:
    """Re-analyze every seed in `seeds` of an existing `parameterized_
    primitive_gate`-shaped run directory and aggregate. Read-only:
    `run_dir` and everything under it are never modified."""
    if not seeds:
        raise ValueError("seeds must be non-empty")
    run_dir = Path(run_dir)
    per_seed = tuple(reanalyze_seed(run_dir, seed) for seed in seeds)

    operation_names = per_seed[0].operation_names
    for seed_report in per_seed[1:]:
        if seed_report.operation_names != operation_names:
            raise ValueError(
                "all seeds must share the same operation_names to aggregate; got "
                f"{seed_report.operation_names} for seed {seed_report.seed}, expected "
                f"{operation_names}"
            )

    def _mean(field_name: str) -> dict[str, float]:
        return {
            name: statistics.fmean(
                getattr(seed_report, field_name)[name] for seed_report in per_seed
            )
            for name in operation_names
        }

    return RunReanalysis(
        run_dir=str(run_dir),
        seeds=tuple(seed_report.seed for seed_report in per_seed),
        operation_names=operation_names,
        per_seed=per_seed,
        mean_per_operation_correct_exact_match=_mean("per_operation_correct_exact_match"),
        mean_per_operation_wrong_argument_exact_match=_mean(
            "per_operation_wrong_argument_exact_match"
        ),
        mean_per_operation_wrong_family_exact_match=_mean(
            "per_operation_wrong_family_exact_match"
        ),
        mean_per_operation_none_exact_match=_mean("per_operation_none_exact_match"),
        mean_per_operation_argument_effect_rate=_mean("per_operation_argument_effect_rate"),
        mean_per_operation_target_output_length_mean=_mean(
            "per_operation_target_output_length_mean"
        ),
        not_recoverable=per_seed[0].not_recoverable,
        notes=_NOTES,
    )
