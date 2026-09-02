"""Counterfactual COUNT gate (Phase A.1 Post-Correction Task A1-R005D-004,
`docs/CODEX_TASKS_A1_R005_RETRY.md`, STOP GATE).

`docs/DECISIONS.md` ADR-0030 (Task A1-R005D-001) found that A1-R005's own
"correct family + wrong argument" control was overwhelmingly *effectful* for
`COUNT` (argument_effect_rate 0.667 -- a wrong argument usually did change
the ground-truth output) yet `COUNT`'s raw Correct-minus-Wrong-argument gap
was still small in absolute terms relative to its own Correct score. Two
candidate explanations from `docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md`
survive that finding: F3 (additive conditioning is too weak to make the
residual actually depend on the argument) and F4 ("training does not force
argument use" -- under A1-R005's i.i.d. sampling, most training content was
seen with only one incidental argument value, so a learner could fit a
content-only shortcut without ever needing to read `PrimitiveCall.arguments`
at all). This task isolates F4 specifically, per `docs/
AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "Counterfactual training rule": "At
least one retry experiment must present the same content with multiple
different arguments in the same batch or tightly coupled training group. The
purpose is to make argument identity the only information that can explain
target differences." `docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md`
section 4/6 names `COUNT` as the operation to try this on first (single-token
output, so a failure here cannot be blamed on `SHIFT`/`SELECT`'s multi-token
exact-match compounding, ADR-0029 finding 2).

## Counterfactual groups, not i.i.d. examples

`generate_count_counterfactual_groups` replaces `apc.environments.generator.
TaskGenerator`'s ordinary "one random content, one random argument" sampling
with grouped sampling: for one randomly drawn content sequence, it searches
for several `COUNT.target` values whose *outputs* (`min(count(content,
target), vocab_size - 1)`) are pairwise distinct (`CountCounterfactualGroup`,
at least `MIN_GROUP_SIZE=2`, up to `group_size` -- default 3, matching the
design doc's own `a1 -> y1, a2 -> y2, a3 -> y3` illustration), then builds
one real `Example` per chosen target via `apc.environments.interpreter.
run_program` (so `oracle_call_for_example`, `apc.core.data.collate_batch`,
and every other `Example`-consuming function in this codebase work
unmodified on the result). Both primitive training *and* the unseen
evaluation batch are built this way, so the "present the same content with
multiple arguments" property holds throughout the whole gate, not merely as
an incidental property of some training batches.

Because the group construction actively *selects* targets for pairwise
distinct outputs (rather than drawing a target at random and hoping it
happens to differ), every "Wrong argument" pairing this module evaluates is
effectful *by construction*: `argument_effect_rate` is computed on the
unseen eval batch and reported, not assumed, but it is expected to equal (or
sit at) `1.0` rather than A1-R005's incidental 0.667. This is a deliberately
stronger construction than A1-R005D-001's post-hoc "raw vs. effectful"
split (`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "Wrong-argument control
rule") -- there is no raw/effectful distinction to draw here because the
generator never produces a non-effectful pairing in the first place.

## Frozen Stable Core: pretraining stays ordinary i.i.d.

Only the *primitive* training loop uses counterfactual groups. The Stable
Core is still pretrained task-blind (`include_task_spec=False`) on ordinary
i.i.d. `COUNT` examples via `apc.evaluation.shared_core_generalization.
train_shared_core`, exactly as `apc.evaluation.parameterized_primitive_gate`
pretrains its (four-operation) core -- the counterfactual-grouping fix is
specifically for the argument-conditioning path (F4), not for the
task-blind content representation, which A1-R001/A1-R003 already gate
separately. `operation_names=("COUNT",)` here (a single-operation stream) is
sufficient and appropriately minimal: A1-R005D-004's own scope is COUNT
only ("Work: COUNT only, ... one COUNT family").

## Why there is no "Wrong family" arm

`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 8's four-arm matrix
needs at least one *other* family to force instead of the correct one;
`apc.evaluation.parameterized_primitive_gate.ParameterizedPrimitiveGateConfig`
enforces `len(operation_names) >= 2` for exactly this reason. This gate
registers exactly one `apc.primitives.conditioning.ConditionedPrimitive`
family (`COUNT`) by design ("Work: ... one COUNT family") -- there is no
second family available to force, so this module measures three arms only:
**Correct**, **effectful Wrong argument**, and **None** (`docs/
CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-004's own "Evaluate Correct /
effectful Wrong argument / None"), plus the derived **causal gap** (`Correct
- max(effectful Wrong argument, None)`, matching every earlier Phase A.1
primitive gate's own formula, restricted to the two arms this gate actually
has).

## Filling in an acceptance number the task text does not restate

A1-R005D-004's own "Acceptance" list gives three numbered criteria (Correct
>= 0.90, effectful Wrong argument <= 0.30, causal gap >= 0.50) and does not
restate a `None` ceiling, unlike A1-R005's H2c (which explicitly lists
`None <= 0.30` as its own numbered arm). Its own "Work" item nonetheless
requires `None` to be *evaluated*, and `docs/exec-plans/active/
A1_R005_RETRY.md`'s standing "Scientific rule" states the outcome this whole
retry sequence is after: "Correct >> effectful Wrong argument, while Wrong
family and None remain low" -- `None` staying low is a standing requirement
of the retry program, not merely an A1-R005-specific number. Per `AGENTS.md`'s
decision log guidance ("if acceptance criteria are underspecified ... add a
concise ADR"), this gate reuses the same `<= 0.30` "materially low"
convention `apc.evaluation.parameterized_primitive_gate` already applied to
its own undeclared H2c thresholds (`NONE_CEILING` below) and includes
`none_passed` in the overall `passed` verdict, rather than leaving `None`
purely informational. See `docs/DECISIONS.md` for the ADR recording this.

## Scope discipline

Same restriction as every earlier Phase A.1 primitive gate: `apc.plastic`,
`apc.consolidation`, and `apc.meta` are never imported here; no `Router` is
imported or reachable (oracle routing only, via `apc.core.execution`'s
existing A1-R005 `ConditionedPrimitive` dispatch -- this task does not
change `apc.core.execution` at all). This gate constructs a fresh
single-primitive `PrimitiveBank` per run and discards it after measuring the
three arms; no composition, promotion to a persistent/`STABLE` bank, or
capacity/architecture change is made here (`docs/CODEX_TASKS_A1_R005_RETRY.md`
A1-R005D-005/A1-R005D-006 own that, explicitly gated on this task's own
"first hard STOP GATE" result, `docs/exec-plans/active/A1_R005_RETRY.md`
R005-M3).

### Update (Task A1-R005D-005): argument-path gradient-norm logging

`docs/CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-005 ("Minimal capacity/training
sweep") asks to log "argument-path gradient norms if easy" alongside the
existing learning curve. `_train_primitives` now additionally records
`argument_path_grad_norm` (the combined L2 norm of `ConditionedPrimitive.
argument_encoder` and `c_proj`'s gradients -- the design doc's `e_a`/`C_i`
path) and `content_path_grad_norm` (`a_proj`/`b_proj`'s gradients -- `A_i`/
`B_i`, inherited from `Primitive`) into the same `primitive_metrics.jsonl`
row, at the same `eval_every` cadence as `progress_correct_exact_match`.
Both are read after `nn.utils.clip_grad_norm_` has already rescaled every
bank parameter's gradient in place; since that clip applies one shared
scalar to the whole bank, the *ratio* between the two path norms this field
reports -- whether the argument path is receiving any training signal at
all, relative to the content path -- is identical whether read before or
after clipping, so reading after it (simpler: no need to snapshot grads
pre-clip) loses nothing this diagnostic needs. This is additive
instrumentation only: no change to the three-arm metrics, the acceptance
thresholds, or the already-recorded A1-R005D-004 STOP GATE result (`docs/
DECISIONS.md` ADR-0033) -- this module is reused unmodified in its
scientific behavior by `apc.evaluation.count_capacity_sweep_gate`
(A1-R005D-005).

### Update (Task A1-R005D-006): pluggable conditioning variant

`docs/DECISIONS.md` ADR-0034 (Task A1-R005D-005) points at `apc.primitives.
conditioning.ConditionedPrimitive`'s additive-conditioning formula itself
(F3) as the likely remaining bottleneck once capacity/training increases
were ruled out. A1-R005D-006 ("Conditioning architecture comparison")
compares that formula (V0) against `FiLMConditionedPrimitive` (V1) and
`BasisModulatedConditionedPrimitive` (V2) -- see `apc.primitives.
conditioning`'s own "Update" section. `_build_primitive_bank`, `_train_
primitives`, `run_count_counterfactual_gate`, and `run_count_counterfactual_
gate_multi_seed` each gained a keyword-only `variant: ConditioningVariant =
ConditioningVariant.ADDITIVE` parameter (`_build_primitive_bank`/`run_count_
counterfactual_gate[_multi_seed]` also gained `num_basis`, used only by V2)
so `apc.evaluation.count_conditioning_architecture_gate` (A1-R005D-006's own
comparison driver) can hold the counterfactual-group generation, frozen
Stable Core pretraining, and three-arm evaluation completely fixed and vary
only which `ArgumentConditionedPrimitive` subclass gets trained -- the exact
reuse-over-duplication precedent `apc.evaluation.count_capacity_sweep_gate`
(A1-R005D-005) already set for the capacity/training sweep. Every default
(`ConditioningVariant.ADDITIVE`) reproduces every pre-A1-R005D-006 call's
behavior byte-for-byte: `build_argument_conditioned_primitive(..., ADDITIVE,
...)` delegates straight to `build_conditioned_primitive`, so A1-R005D-004's
and A1-R005D-005's already-recorded results are untouched by this change.
The argument-path/content-path gradient-norm split (`_train_primitives`'s
A1-R005D-005 addition, above) is generalized the same way: `_argument_and_
content_path_params` resolves the right "argument path" projection
(`c_proj`/`film_proj`/`mix_proj`+`basis`) per variant instead of assuming
`ConditionedPrimitive.c_proj` specifically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX, collate_batch
from apc.core.execution import (
    evaluate_exact_match_no_primitive,
    evaluate_exact_match_with_oracle_calls,
    forward_logits_with_oracle_calls_trainable,
)
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SharedCoreTokens
from apc.environments.generator import Example, OracleMetadata, oracle_call_for_example
from apc.environments.interpreter import run_program
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec, operation_id
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.conditioning import (
    DEFAULT_ARG_DIM,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_BASIS_VECTORS,
    ArgumentConditionedPrimitive,
    BasisModulatedConditionedPrimitive,
    ConditionedPrimitive,
    ConditioningVariant,
    FiLMConditionedPrimitive,
    build_argument_conditioned_primitive,
)
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

__all__ = [
    "COUNT_OPERATION",
    "MIN_GROUP_SIZE",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "CORRECT_THRESHOLD",
    "EFFECTFUL_WRONG_ARGUMENT_CEILING",
    "NONE_CEILING",
    "MIN_CAUSAL_GAP",
    "CountCounterfactualGroup",
    "generate_count_counterfactual_groups",
    "CountCounterfactualGateConfig",
    "count_counterfactual_gate_config_from_dict",
    "FrozenStableCore",
    "CountCounterfactualGateReport",
    "run_count_counterfactual_gate",
    "CountCounterfactualGateMultiSeedReport",
    "run_count_counterfactual_gate_multi_seed",
]

COUNT_OPERATION = "COUNT"

# A counterfactual group needs at least one "wrong argument" partner besides
# the correct one.
MIN_GROUP_SIZE = 2
# docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md section 4's own
# illustration: "same content x + arg a1 -> y1, arg a2 -> y2, arg a3 -> y3".
DEFAULT_GROUP_SIZE = 3
# Empirically (vocab_size=10, sequence_length_range=(6, 10)), fewer than 2
# distinct COUNT outputs occurs for well under 0.1% of randomly drawn
# content; this bounds retries for that rare case without looping forever.
_MAX_CONTENT_RESAMPLE_ATTEMPTS = 500

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md: ">=5 seeds" for a gate claim.
MIN_GATE_SEEDS = 5

# docs/CODEX_TASKS_A1_R005_RETRY.md A1-R005D-004's own explicit numbers.
CORRECT_THRESHOLD = 0.90
EFFECTFUL_WRONG_ARGUMENT_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50
# Not restated by A1-R005D-004's own acceptance list; filled in from H2b/H2c's
# standing "materially low" convention (module docstring, "Filling in an
# acceptance number the task text does not restate").
NONE_CEILING = 0.30


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 48,
        "dropout": 0.0,
    }


def _derive_local_seed(seed: int, step: int, label: str) -> int:
    """Deterministic sub-seed for one `(seed, step, label)` triple,
    independent of `PYTHONHASHSEED` -- same construction as `apc.
    environments.generator._derive_seed`/`apc.evaluation.
    task_blind_content_gate._derive_local_seed`, kept local here since both
    are private to their own modules."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    """Combined L2 norm of `parameters`' current `.grad` tensors (`0.0` for
    a parameter with no gradient yet, e.g. before the first backward pass).
    """
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += float(p.grad.detach().norm(2).item()) ** 2
    return total_sq**0.5


def _argument_and_content_path_params(
    primitive: ArgumentConditionedPrimitive, variant: ConditioningVariant
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """The argument-path-vs-content-path split A1-R005D-005 introduced
    (`docs/DECISIONS.md` ADR-0034), generalized across `ConditioningVariant`s
    (A1-R005D-006): 'content path' is always the inherited `a_proj`/`b_proj`;
    'argument path' is always `argument_encoder` plus whichever projection
    turns `e_a` into the primitive's own conditioning signal (`c_proj` for
    V0/additive, `film_proj` for V1/FiLM, `mix_proj` + `basis` for
    V2/basis)."""
    content_path_params = list(primitive.a_proj.parameters()) + list(primitive.b_proj.parameters())
    argument_path_params = list(primitive.argument_encoder.parameters())
    if variant == ConditioningVariant.ADDITIVE:
        assert isinstance(primitive, ConditionedPrimitive)
        argument_path_params += list(primitive.c_proj.parameters())
    elif variant == ConditioningVariant.FILM:
        assert isinstance(primitive, FiLMConditionedPrimitive)
        argument_path_params += list(primitive.film_proj.parameters())
    elif variant == ConditioningVariant.BASIS:
        assert isinstance(primitive, BasisModulatedConditionedPrimitive)
        argument_path_params += [*primitive.mix_proj.parameters(), primitive.basis]
    else:
        raise ValueError(f"unknown ConditioningVariant: {variant!r}")
    return argument_path_params, content_path_params


def _count_output(content: Sequence[int], target: int, vocab_size: int) -> int:
    return min(sum(1 for token in content if token == target), vocab_size - 1)


def _build_count_example(
    input_tokens: tuple[int, ...], target: int, vocab_size: int, split: str
) -> Example:
    """One real `Example` for `COUNT(target)` applied to `input_tokens`,
    built the same way `apc.environments.generator.TaskGenerator._generate`
    builds a depth-1 example -- `apc.environments.interpreter.run_program`
    is the same ground-truth executor, just driven by an explicit `target`
    instead of `CountOp.sample_params`'s random draw -- so every other
    `Example`-consuming function in this codebase (`oracle_call_for_example`,
    `apc.core.data.collate_batch`, `apc.core.execution`'s oracle-routing
    entry points) works on the result unmodified."""
    program = Program(steps=(ProgramStep(operation=COUNT_OPERATION, params={"target": target}),))
    result = run_program(program, input_tokens, vocab_size)
    return Example(
        input_tokens=input_tokens,
        target_tokens=result.output_tokens,
        program=program,
        operation_graph=result.graph,
        category="known",
        split=split,
        vocab_size=vocab_size,
        task_spec=TaskSpec.from_program(program),
        oracle_metadata=OracleMetadata(label="K", primitive_operations=(COUNT_OPERATION,)),
        symbol_permutation=None,
    )


@dataclass(frozen=True)
class CountCounterfactualGroup:
    """One content sequence paired with >=`MIN_GROUP_SIZE` `COUNT.target`
    values whose outputs are pairwise distinct -- "same content x + arg a1
    -> y1, arg a2 -> y2, ..." (`docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md`
    section 4). `examples[i]` is the real `Example` for `targets[i]`; every
    `examples[i].target_tokens` differs from every other member's, checked
    eagerly here rather than merely assumed from the construction that
    produced them, so a future caller building a group by another route
    cannot silently violate the property this whole gate depends on."""

    input_tokens: tuple[int, ...]
    examples: tuple[Example, ...]
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.examples) != len(self.targets):
            raise ValueError(
                f"CountCounterfactualGroup requires len(examples) == len(targets), got "
                f"{len(self.examples)} and {len(self.targets)}"
            )
        if len(self.examples) < MIN_GROUP_SIZE:
            raise ValueError(
                f"CountCounterfactualGroup requires >= {MIN_GROUP_SIZE} members, got "
                f"{len(self.examples)}"
            )
        outputs = [example.target_tokens for example in self.examples]
        if len(set(outputs)) != len(outputs):
            raise ValueError(
                "CountCounterfactualGroup requires pairwise distinct outputs across its "
                f"members; got outputs {outputs} for targets {self.targets}"
            )

    def wrong_target_index(self, member_index: int) -> int:
        """The index of a group member other than `member_index`, guaranteed
        to have a different output (`__post_init__`'s own invariant) --
        deterministic "next member, wrapping around" choice used to build the
        "Wrong argument" arm."""
        return (member_index + 1) % len(self.examples)


def _generate_count_counterfactual_group(
    rng: random.Random,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    group_size: int,
    split: str,
) -> CountCounterfactualGroup:
    min_len, max_len = sequence_length_range
    for _ in range(_MAX_CONTENT_RESAMPLE_ATTEMPTS):
        length = rng.randint(min_len, max_len)
        input_tokens = tuple(rng.randrange(vocab_size) for _ in range(length))

        candidate_targets = list(range(vocab_size))
        rng.shuffle(candidate_targets)
        chosen_targets: list[int] = []
        seen_outputs: set[int] = set()
        for target in candidate_targets:
            output = _count_output(input_tokens, target, vocab_size)
            if output in seen_outputs:
                continue
            seen_outputs.add(output)
            chosen_targets.append(target)
            if len(chosen_targets) == group_size:
                break

        if len(chosen_targets) >= MIN_GROUP_SIZE:
            examples = tuple(
                _build_count_example(input_tokens, target, vocab_size, split)
                for target in chosen_targets
            )
            return CountCounterfactualGroup(
                input_tokens=input_tokens, examples=examples, targets=tuple(chosen_targets)
            )

    raise RuntimeError(
        f"failed to construct a COUNT counterfactual group with >= {MIN_GROUP_SIZE} pairwise "
        f"distinct outputs after {_MAX_CONTENT_RESAMPLE_ATTEMPTS} resamples (vocab_size="
        f"{vocab_size}, sequence_length_range={sequence_length_range}) -- this should be "
        "exceedingly rare; widen sequence_length_range or vocab_size if it recurs"
    )


def generate_count_counterfactual_groups(
    seed: int,
    n_groups: int,
    *,
    step: int,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[CountCounterfactualGroup, ...]:
    """Deterministic-by-`(seed, step, split)` counterfactual group batch,
    matching `apc.environments.generator.TaskGenerator.generate_online`'s
    determinism convention (one shared `random.Random` stream for the whole
    call, re-derived from `(seed, step, split)` rather than any mutable
    generator state) so replaying the same arguments reproduces the same
    groups.
    """
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if group_size < MIN_GROUP_SIZE:
        raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {group_size}")
    rng = random.Random(_derive_local_seed(seed, step, split))
    return tuple(
        _generate_count_counterfactual_group(
            rng, vocab_size, sequence_length_range, group_size, split
        )
        for _ in range(n_groups)
    )


def _flatten_groups(
    groups: Sequence[CountCounterfactualGroup],
) -> tuple[list[Example], dict[int, PrimitiveCall], dict[int, tuple[int, ...]]]:
    """Flatten `groups` into one example list plus two `id(example)`-keyed
    side tables: `wrong_call_by_id` (the "Wrong argument" `PrimitiveCall` to
    force for that example -- a different group member's target, per
    `CountCounterfactualGroup.wrong_target_index`) and
    `wrong_target_tokens_by_id` (that partner's own ground-truth output,
    used only to report `argument_effect_rate`). `id(...)` keys are safe
    here because every flattened `Example` stays alive for the caller's
    entire use of these tables (held in the returned list itself); `Example`
    cannot be used as a dict key directly since `ProgramStep.params` is a
    plain (unhashable) `dict`.
    """
    examples: list[Example] = []
    wrong_call_by_id: dict[int, PrimitiveCall] = {}
    wrong_target_tokens_by_id: dict[int, tuple[int, ...]] = {}
    for group in groups:
        for index, example in enumerate(group.examples):
            partner_index = group.wrong_target_index(index)
            wrong_call_by_id[id(example)] = PrimitiveCall(
                operation=COUNT_OPERATION, arguments={"target": group.targets[partner_index]}
            )
            wrong_target_tokens_by_id[id(example)] = group.examples[partner_index].target_tokens
            examples.append(example)
    return examples, wrong_call_by_id, wrong_target_tokens_by_id


@dataclass(frozen=True)
class CountCounterfactualGateConfig:
    """Explicit, serializable configuration for one seed's A1-R005D-004
    gate run. `primitive_rank`/`core_train`/`primitive_train` default to
    values distinct from `apc.evaluation.parameterized_primitive_gate`'s own
    dataclass defaults for the same reason that module's defaults differ
    from the library-wide `SharedCoreGateTrainConfig`/`PrimitiveTrainConfig`
    defaults: the *shipped config file* (`configs/
    phase_a1_count_counterfactual_gate.yaml`), not this dataclass, is what
    pins the actual "baseline rank/steps first" budget A1-R005D-004's own
    "Work" item asks for (A1-R005's own `configs/
    phase_a1_parameterized_primitive_gate.yaml` numbers, reused unchanged so
    the counterfactual-training-data change is this run's only variable
    relative to A1-R005).
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    primitive_rank: int = 8
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    primitive_train: PrimitiveTrainConfig = field(default_factory=PrimitiveTrainConfig)
    num_unseen_eval_groups: int = 1400
    min_unseen_eval_examples: int = 1024

    def __post_init__(self) -> None:
        if self.group_size < MIN_GROUP_SIZE:
            raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {self.group_size}")
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= the upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]}) so every COUNT.target "
                "value this config can generate stays inside the argument encoder's declared "
                "bucket range"
            )
        if self.num_unseen_eval_groups < 1:
            raise ValueError(
                f"num_unseen_eval_groups must be >= 1, got {self.num_unseen_eval_groups}"
            )
        if self.min_unseen_eval_examples < 1:
            raise ValueError(
                f"min_unseen_eval_examples must be >= 1, got {self.min_unseen_eval_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def count_counterfactual_gate_config_from_dict(
    raw: dict[str, Any],
) -> CountCounterfactualGateConfig:
    """Parse a `configs/phase_a1_count_counterfactual_gate.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = CountCounterfactualGateConfig()
    core_train_raw = raw.get("core_train")
    core_train = (
        dataclasses.replace(defaults.core_train, **core_train_raw)
        if core_train_raw
        else defaults.core_train
    )
    primitive_train_raw = raw.get("primitive_train")
    primitive_train = (
        dataclasses.replace(defaults.primitive_train, **primitive_train_raw)
        if primitive_train_raw
        else defaults.primitive_train
    )
    return CountCounterfactualGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        primitive_rank=raw.get("primitive_rank", defaults.primitive_rank),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        primitive_train=primitive_train,
        num_unseen_eval_groups=raw.get(
            "num_unseen_eval_groups", defaults.num_unseen_eval_groups
        ),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind on ordinary i.i.d. `COUNT`
    examples, then frozen: every parameter has `requires_grad=False`. Unlike
    `apc.evaluation.parameterized_primitive_gate.FrozenStableCore`, this does
    not carry a `TaskGenerator`: primitive training here draws from
    `generate_count_counterfactual_groups`, not the plain generator (module
    docstring, "Frozen Stable Core: pretraining stays ordinary i.i.d.")."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: CountCounterfactualGateConfig, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=(COUNT_OPERATION,),
        permute_symbols=False,
        include_task_spec=False,
        model=config.model,
        train=config.core_train,
    )
    trained = train_shared_core(core_config, metrics_path=metrics_path)
    model = trained.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenStableCore(
        model=model,
        tokens=trained.tokens,
        final_core_train_loss=trained.final_train_loss,
        device=trained.device,
    )


def _build_primitive_bank(
    d_model: int,
    rank: int,
    *,
    vocab_size: int,
    max_sequence_length: int,
    arg_dim: int,
    variant: ConditioningVariant = ConditioningVariant.ADDITIVE,
    num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
) -> PrimitiveBank:
    """Exactly one argument-conditioned `COUNT` primitive, keyed at
    `operation_id("COUNT")` (ADR-0024's convention) -- "one COUNT family"
    (module docstring, "Why there is no 'Wrong family' arm"). `variant`
    selects which `ArgumentConditionedPrimitive` subclass is built (Task
    A1-R005D-006); the default reproduces every pre-A1-R005D-006 caller's
    `ConditionedPrimitive` exactly."""
    bank = PrimitiveBank()
    primitive = build_argument_conditioned_primitive(
        operation_id(COUNT_OPERATION),
        COUNT_OPERATION,
        PrimitiveConfig(d_model=d_model, rank=rank),
        variant,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        arg_dim=arg_dim,
        num_basis=num_basis,
        status=PrimitiveStatus.CANDIDATE,
        metadata={"operation": COUNT_OPERATION, "conditioning_variant": variant.value},
    )
    bank.add_primitive(primitive)
    return bank


def _train_primitives(
    core: FrozenStableCore,
    bank: PrimitiveBank,
    config: CountCounterfactualGateConfig,
    metrics_path: str | Path | None = None,
    *,
    variant: ConditioningVariant = ConditioningVariant.ADDITIVE,
) -> tuple[float, int]:
    """Train `bank`'s one `ConditionedPrimitive` via oracle-forced routing
    through `core`'s frozen weights, on counterfactual groups
    (`generate_count_counterfactual_groups`) instead of `apc.evaluation.
    parameterized_primitive_gate._train_primitives`'s i.i.d. sampling -- the
    one functional difference this task's whole hypothesis (F4) is about.
    Each step draws `ceil(primitive_train.batch_size / group_size)` fresh
    groups and flattens every member into that step's batch, so the same
    content appears multiple times per step, each time paired with a
    different (correct) target -- content alone can never explain the
    target within a step, only content-plus-argument can (`docs/
    AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "Counterfactual training rule").

    Returns `(final_loss, total_examples_seen)` -- `total_examples_seen`
    replaces `apc.evaluation.parameterized_primitive_gate`'s
    `steps * batch_size` compute-budget proxy, since a step's actual example
    count here varies with how many members each drawn group happened to
    realize (`CountCounterfactualGroup`'s size is `>= MIN_GROUP_SIZE`, not
    always exactly `config.group_size`).
    """
    model, tokens, device = core.model, core.tokens, core.device
    bank.to(device)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.primitive_train.lr,
        weight_decay=config.primitive_train.weight_decay,
    )

    count_primitive = bank.get(operation_id(COUNT_OPERATION))
    assert isinstance(count_primitive, ArgumentConditionedPrimitive)
    argument_path_params, content_path_params = _argument_and_content_path_params(
        count_primitive, variant
    )
    # Re-anchor the shared global RNG stream after bank/optimizer
    # construction, per ADR-0016's pattern.
    set_seed(config.seed)

    groups_per_step = max(1, -(-config.primitive_train.batch_size // config.group_size))

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    total_examples_seen = 0
    try:
        for step in range(config.primitive_train.steps):
            groups = generate_count_counterfactual_groups(
                config.seed,
                groups_per_step,
                step=step,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, _, _ = _flatten_groups(groups)
            calls = [oracle_call_for_example(example) for example in examples]
            batch = collate_batch(examples, tokens, device=device, include_task_spec=False)
            total_examples_seen += len(examples)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_logits_with_oracle_calls_trainable(
                model, batch.input_ids, bank, calls
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if config.primitive_train.grad_clip > 0:
                nn.utils.clip_grad_norm_(bank.parameters(), config.primitive_train.grad_clip)
            optimizer.step()
            final_loss = float(loss.item())

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.primitive_train.eval_every == 0
                or step_number == config.primitive_train.steps
            ):
                progress_groups = generate_count_counterfactual_groups(
                    config.seed,
                    max(1, -(-config.primitive_train.progress_eval_examples // config.group_size)),
                    step=step,
                    split="val",
                    vocab_size=config.vocab_size,
                    sequence_length_range=config.sequence_length_range,
                    group_size=config.group_size,
                )
                progress_examples, _, _ = _flatten_groups(progress_groups)
                progress_exact_match, _, _ = evaluate_exact_match_with_oracle_calls(
                    model, progress_examples, tokens, bank, device=device
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "loss": final_loss,
                            "progress_correct_exact_match": progress_exact_match,
                            "argument_path_grad_norm": _grad_norm(argument_path_params),
                            "content_path_grad_norm": _grad_norm(content_path_params),
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    return final_loss, total_examples_seen


@dataclass(frozen=True)
class CountCounterfactualGateReport:
    """Everything observed while pretraining, freezing, training the one
    `COUNT` primitive on top of, and evaluating one seed's three-arm causal
    ablation (Correct / effectful Wrong argument / None)."""

    config: CountCounterfactualGateConfig
    core_steps_trained: int
    core_examples_seen: int
    final_core_train_loss: float
    primitive_steps_trained: int
    primitive_examples_seen: int
    final_primitive_train_loss: float
    correct_exact_match: float
    effectful_wrong_argument_exact_match: float
    none_exact_match: float
    causal_gap: float
    argument_effect_rate: float
    num_unseen_eval_groups: int
    num_unseen_eval_examples: int
    mean_eval_group_size: float
    bank_size: int
    core_param_count: int
    core_trainable_param_count: int
    primitive_param_count: int
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_count_counterfactual_gate(
    config: CountCounterfactualGateConfig,
    *,
    core_metrics_path: str | Path | None = None,
    primitive_metrics_path: str | Path | None = None,
    variant: ConditioningVariant = ConditioningVariant.ADDITIVE,
    num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
) -> CountCounterfactualGateReport:
    """Run one seed of the A1-R005D-004 gate end to end: pretrain and freeze
    a task-blind `COUNT`-only Stable Core, train one argument-conditioned
    `COUNT` primitive (`variant`, default `ConditioningVariant.ADDITIVE` --
    `ConditionedPrimitive`, byte-identical to every pre-A1-R005D-006 caller)
    on counterfactual groups, then evaluate Correct/effectful Wrong
    argument/None on a large, independently-drawn unseen batch of
    counterfactual groups."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, metrics_path=core_metrics_path)
    bank = _build_primitive_bank(
        core.model.config.d_model,
        config.primitive_rank,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
        variant=variant,
        num_basis=num_basis,
    )
    final_primitive_loss, primitive_examples_seen = _train_primitives(
        core, bank, config, metrics_path=primitive_metrics_path, variant=variant
    )

    unseen_groups = generate_count_counterfactual_groups(
        config.seed,
        config.num_unseen_eval_groups,
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )
    unseen_examples, wrong_call_by_id, wrong_target_tokens_by_id = _flatten_groups(unseen_groups)
    if len(unseen_examples) < config.min_unseen_eval_examples:
        raise ValueError(
            f"unseen eval batch realized only {len(unseen_examples)} examples from "
            f"{config.num_unseen_eval_groups} groups, below min_unseen_eval_examples="
            f"{config.min_unseen_eval_examples}; increase num_unseen_eval_groups"
        )

    def _wrong_argument_call_provider(example: Example) -> PrimitiveCall:
        return wrong_call_by_id[id(example)]

    correct_exact_match, _, _ = evaluate_exact_match_with_oracle_calls(
        core.model, unseen_examples, core.tokens, bank, device=core.device
    )
    effectful_wrong_argument_exact_match, _, _ = evaluate_exact_match_with_oracle_calls(
        core.model,
        unseen_examples,
        core.tokens,
        bank,
        _wrong_argument_call_provider,
        device=core.device,
    )
    none_exact_match, _ = evaluate_exact_match_no_primitive(
        core.model, unseen_examples, core.tokens, device=core.device
    )

    argument_effect_rate = statistics.fmean(
        float(example.target_tokens != wrong_target_tokens_by_id[id(example)])
        for example in unseen_examples
    )
    causal_gap = correct_exact_match - max(effectful_wrong_argument_exact_match, none_exact_match)

    return CountCounterfactualGateReport(
        config=config,
        core_steps_trained=config.core_train.steps,
        core_examples_seen=config.core_train.steps * config.core_train.batch_size,
        final_core_train_loss=core.final_core_train_loss,
        primitive_steps_trained=config.primitive_train.steps,
        primitive_examples_seen=primitive_examples_seen,
        final_primitive_train_loss=final_primitive_loss,
        correct_exact_match=correct_exact_match,
        effectful_wrong_argument_exact_match=effectful_wrong_argument_exact_match,
        none_exact_match=none_exact_match,
        causal_gap=causal_gap,
        argument_effect_rate=argument_effect_rate,
        num_unseen_eval_groups=len(unseen_groups),
        num_unseen_eval_examples=len(unseen_examples),
        mean_eval_group_size=len(unseen_examples) / len(unseen_groups),
        bank_size=len(bank),
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        primitive_param_count=bank.total_parameter_count(),
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class CountCounterfactualGateMultiSeedReport:
    """A1-R005D-004's verdict aggregated across seeds. `passed` is computed
    against the cross-seed *mean* of each arm, matching every other Phase
    A.1 gate's own convention."""

    seeds: tuple[int, ...]
    per_seed: tuple[CountCounterfactualGateReport, ...]
    mean_correct_exact_match: float
    stdev_correct_exact_match: float
    min_correct_exact_match: float
    max_correct_exact_match: float
    mean_effectful_wrong_argument_exact_match: float
    mean_none_exact_match: float
    mean_causal_gap: float
    mean_argument_effect_rate: float
    correct_threshold: float
    effectful_wrong_argument_ceiling: float
    none_ceiling: float
    min_causal_gap: float
    correct_passed: bool
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    family_count_passed: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "stdev_correct_exact_match": self.stdev_correct_exact_match,
            "min_correct_exact_match": self.min_correct_exact_match,
            "max_correct_exact_match": self.max_correct_exact_match,
            "mean_effectful_wrong_argument_exact_match": (
                self.mean_effectful_wrong_argument_exact_match
            ),
            "mean_none_exact_match": self.mean_none_exact_match,
            "mean_causal_gap": self.mean_causal_gap,
            "mean_argument_effect_rate": self.mean_argument_effect_rate,
            "correct_threshold": self.correct_threshold,
            "effectful_wrong_argument_ceiling": self.effectful_wrong_argument_ceiling,
            "none_ceiling": self.none_ceiling,
            "min_causal_gap": self.min_causal_gap,
            "correct_passed": self.correct_passed,
            "effectful_wrong_argument_passed": self.effectful_wrong_argument_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "family_count_passed": self.family_count_passed,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_count_counterfactual_gate_multi_seed(
    base_config: CountCounterfactualGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    correct_threshold: float = CORRECT_THRESHOLD,
    effectful_wrong_argument_ceiling: float = EFFECTFUL_WRONG_ARGUMENT_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
    variant: ConditioningVariant = ConditioningVariant.ADDITIVE,
    num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
) -> CountCounterfactualGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 gate. `variant`/`num_basis` (Task A1-R005D-006) select the
    `ConditioningVariant` trained for every seed; the default reproduces
    every pre-A1-R005D-006 caller's behavior exactly.
    """
    per_seed: list[CountCounterfactualGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        core_metrics_path = (
            run_dir_path / f"seed_{seed}" / "core_metrics.jsonl" if run_dir_path else None
        )
        primitive_metrics_path = (
            run_dir_path / f"seed_{seed}" / "primitive_metrics.jsonl" if run_dir_path else None
        )
        report = run_count_counterfactual_gate(
            config,
            core_metrics_path=core_metrics_path,
            primitive_metrics_path=primitive_metrics_path,
            variant=variant,
            num_basis=num_basis,
        )
        if run_dir_path is not None:
            seed_dir = run_dir_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    correct_values = [report.correct_exact_match for report in per_seed]
    mean_correct, stdev_correct, min_correct, max_correct = _summarize(correct_values)
    mean_effectful_wrong_argument = statistics.fmean(
        report.effectful_wrong_argument_exact_match for report in per_seed
    )
    mean_none = statistics.fmean(report.none_exact_match for report in per_seed)
    mean_causal_gap = mean_correct - max(mean_effectful_wrong_argument, mean_none)
    mean_argument_effect_rate = statistics.fmean(
        report.argument_effect_rate for report in per_seed
    )

    correct_passed = mean_correct >= correct_threshold
    effectful_wrong_argument_passed = (
        mean_effectful_wrong_argument <= effectful_wrong_argument_ceiling
    )
    none_passed = mean_none <= none_ceiling
    causal_gap_passed = mean_causal_gap >= min_causal_gap
    family_count_passed = all(report.bank_size == 1 for report in per_seed)
    passed = (
        correct_passed
        and effectful_wrong_argument_passed
        and none_passed
        and causal_gap_passed
        and family_count_passed
    )

    return CountCounterfactualGateMultiSeedReport(
        seeds=tuple(report.config.seed for report in per_seed),
        per_seed=tuple(per_seed),
        mean_correct_exact_match=mean_correct,
        stdev_correct_exact_match=stdev_correct,
        min_correct_exact_match=min_correct,
        max_correct_exact_match=max_correct,
        mean_effectful_wrong_argument_exact_match=mean_effectful_wrong_argument,
        mean_none_exact_match=mean_none,
        mean_causal_gap=mean_causal_gap,
        mean_argument_effect_rate=mean_argument_effect_rate,
        correct_threshold=correct_threshold,
        effectful_wrong_argument_ceiling=effectful_wrong_argument_ceiling,
        none_ceiling=none_ceiling,
        min_causal_gap=min_causal_gap,
        correct_passed=correct_passed,
        effectful_wrong_argument_passed=effectful_wrong_argument_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        family_count_passed=family_count_passed,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
