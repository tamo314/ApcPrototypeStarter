"""Counterfactual BIND gate (Phase A.1 Post-Correction Task A1-R005D-007,
`docs/CODEX_TASKS_A1_R005_RETRY.md`).

`docs/DECISIONS.md` ADR-0033/ADR-0034/ADR-0035 (Tasks A1-R005D-004/005/006)
established, for `COUNT`, that the counterfactual-training-group fix alone
(F4) does not close the causal gap, that additional capacity/training budget
does not either (F1/F2 ruled out), and that neither of two alternative
conditioning formulas (F3's FiLM/basis alternatives) reaches the acceptance
thresholds either -- COUNT's own STOP GATE remains failed at the baseline
architecture. A1-R005D-007 runs the exact same counterfactual-group protocol
(`docs/CODEX_TASKS_A1_R005_RETRY.md`'s own "Use same-content/multiple-key
groups with deliberately different outputs") against a *different*
parameterized operation, `BIND`, at the same baseline architecture (V0/
additive `apc.primitives.conditioning.ConditionedPrimitive`, `primitive_rank
=8`, `arg_dim=16`) A1-R005D-004 used -- this task does not repeat the
architecture comparison for `BIND` (A1-R005D-006 already answered that
question for the shared conditioning module; there is no `BIND`-specific
reason to expect a different formula ranking, and re-running it here would
duplicate, not extend, that STOP GATE's finding).

## Why `BIND` needs its own group generator, not `count_counterfactual_gate`'s

`apc.environments.operations.BindOp` differs from `CountOp` in two ways this
module's counterfactual-group construction must respect:

- **Structural validity**: `BindOp.is_valid_for_length` requires an *even*
  input length (`>= 2`) -- the content is read as `(key, value)` pairs
  (`sequence[0::2]`/`sequence[1::2]`). `_valid_bind_lengths` restricts
  `sequence_length_range` to the even lengths in it before any content is
  drawn, so this generator never constructs an odd-length `BIND` example
  (which `BindOp.apply` would still compute *something* for via its
  `range(0, len(sequence) - 1, 2)` loop, but which could never come from
  `BindOp.sample_params`/the ordinary generator in the first place).
- **Argument domain**: `CountOp.sample_params` draws `target` from the whole
  vocabulary (any value is a legal count target, including one that happens
  to appear zero times). `BindOp.sample_params` instead draws `query_key`
  only from keys *actually present* in the content (`sequence[0::2]`) --
  querying an absent key is not something the real operation/generator ever
  produces. This module's counterfactual candidates are therefore the
  content's own distinct present keys (shuffled, then filtered for pairwise-
  distinct outputs), not the full vocabulary range `generate_count_
  counterfactual_groups` searches.

Everything else -- frozen task-blind Stable Core pretraining, oracle-forced
primitive training through the frozen core, the three-arm evaluation
(Correct / effectful Wrong argument / None) plus derived causal gap, and the
multi-seed aggregation/acceptance-threshold machinery -- mirrors `apc.
evaluation.count_counterfactual_gate` structurally. Per that module's own
`FrozenStableCore` docstring precedent ("Unlike `apc.evaluation.
parameterized_primitive_gate.FrozenStableCore`, this does not carry a
`TaskGenerator`"), each operation-specific counterfactual gate module defines
its own small set of generic helpers rather than importing another
operation's module -- this module follows that same convention rather than
depending on `count_counterfactual_gate` for anything.

## No conditioning-variant/gradient-norm plumbing here

A1-R005D-005/006's capacity-sweep and conditioning-architecture-comparison
generalizations of `count_counterfactual_gate` are specific to *COUNT's own*
STOP GATE follow-up investigation; A1-R005D-007's own task text asks only for
the counterfactual protocol at the established baseline architecture, not a
repeat of either investigation for `BIND`. This module therefore builds a
plain `ConditionedPrimitive` (`apc.primitives.conditioning.
build_conditioned_primitive`, V0/additive) directly and does not log
argument-path/content-path gradient norms, matching A1-R005D-004's own
original (pre-A1-R005D-005) scope.

## Why there is no "Wrong family" arm

Same reasoning as `count_counterfactual_gate` (module docstring, "Why there
is no 'Wrong family' arm"): this gate registers exactly one primitive family
(`BIND`), so there is no second family available to force. Three arms only:
Correct, effectful Wrong argument, None, plus the derived causal gap.

## "One persistent BIND family across keys"

A1-R005D-007's fourth acceptance criterion is exactly `count_counterfactual_
gate`'s own `family_count_passed` check (`bank_size == 1` for every seed):
one `ConditionedPrimitive` instance is queried with every distinct
`query_key` a group produces, never one primitive per key value (`apc.
primitives.conditioning`'s "Persistent-capacity acceptance").

## Filling in the `None` ceiling

A1-R005D-007's own "Acceptance" list, like A1-R005D-004's, does not restate a
`None` ceiling. Reused unchanged from A1-R005D-004's own gap-filling
(`NONE_CEILING = 0.30`, `count_counterfactual_gate` module docstring's
"Filling in an acceptance number the task text does not restate") -- the
standing "materially low" convention, not a new decision.

## Scope discipline

Same restriction as every earlier Phase A.1 primitive gate: `apc.plastic`,
`apc.consolidation`, and `apc.meta` are never imported here; no `Router` is
imported or reachable (oracle routing only, via `apc.core.execution`'s
existing `ArgumentConditionedPrimitive` dispatch -- this task does not change
`apc.core.execution` at all). This gate constructs a fresh single-primitive
`PrimitiveBank` per run and discards it after measuring the three arms.
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
    build_conditioned_primitive,
)
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

__all__ = [
    "BIND_OPERATION",
    "MIN_GROUP_SIZE",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "CORRECT_THRESHOLD",
    "EFFECTFUL_WRONG_ARGUMENT_CEILING",
    "NONE_CEILING",
    "MIN_CAUSAL_GAP",
    "BindCounterfactualGroup",
    "generate_bind_counterfactual_groups",
    "BindCounterfactualGateConfig",
    "bind_counterfactual_gate_config_from_dict",
    "FrozenStableCore",
    "BindCounterfactualGateReport",
    "run_bind_counterfactual_gate",
    "BindCounterfactualGateMultiSeedReport",
    "run_bind_counterfactual_gate_multi_seed",
]

BIND_OPERATION = "BIND"

# A counterfactual group needs at least one "wrong argument" partner besides
# the correct one.
MIN_GROUP_SIZE = 2
# Matches count_counterfactual_gate's own DEFAULT_GROUP_SIZE convention
# (docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md section 4's "same
# content x + arg a1 -> y1, arg a2 -> y2, arg a3 -> y3" illustration).
DEFAULT_GROUP_SIZE = 3
# Same order-of-magnitude safety bound as count_counterfactual_gate's own
# _MAX_CONTENT_RESAMPLE_ATTEMPTS; BIND's candidate pool per draw (the
# content's own distinct present keys) is smaller than COUNT's (the whole
# vocabulary), so this generator resamples the whole content more often on
# average, but still overwhelmingly succeeds well within this bound.
_MAX_CONTENT_RESAMPLE_ATTEMPTS = 500

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md: ">=5 seeds" for a gate claim.
MIN_GATE_SEEDS = 5

# docs/CODEX_TASKS_A1_R005_RETRY.md A1-R005D-007's own explicit numbers
# (identical to A1-R005D-004's).
CORRECT_THRESHOLD = 0.90
EFFECTFUL_WRONG_ARGUMENT_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50
# Not restated by A1-R005D-007's own acceptance list; filled in from
# A1-R005D-004's own precedent (module docstring, "Filling in the None
# ceiling").
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
    count_counterfactual_gate._derive_local_seed`, kept local here since both
    are private to their own modules."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _valid_bind_lengths(sequence_length_range: tuple[int, int]) -> tuple[int, ...]:
    """The even lengths `>= 2` in `sequence_length_range` -- the only input
    lengths `apc.environments.operations.BindOp.is_valid_for_length` accepts
    (module docstring, "Why BIND needs its own group generator")."""
    min_len, max_len = sequence_length_range
    return tuple(
        length for length in range(min_len, max_len + 1) if length >= 2 and length % 2 == 0
    )


def _bind_output(content: Sequence[int], query_key: int) -> int:
    """`apc.environments.operations.BindOp.apply`'s lookup, reimplemented
    directly against a raw content sequence (not a full `params` dict) so
    candidate keys can be pre-filtered for pairwise-distinct outputs before
    committing to building a real `Example` -- same role as `count_
    counterfactual_gate._count_output`. Last matching pair wins, matching
    `BindOp`'s own `dict.update`-style semantics."""
    value = 0
    for i in range(0, len(content) - 1, 2):
        if content[i] == query_key:
            value = content[i + 1]
    return value


def _build_bind_example(
    input_tokens: tuple[int, ...], query_key: int, vocab_size: int, split: str
) -> Example:
    """One real `Example` for `BIND(query_key)` applied to `input_tokens`,
    built the same way `count_counterfactual_gate._build_count_example`
    builds a depth-1 example -- `apc.environments.interpreter.run_program` is
    the same ground-truth executor, just driven by an explicit `query_key`
    instead of `BindOp.sample_params`'s random draw."""
    program = Program(
        steps=(ProgramStep(operation=BIND_OPERATION, params={"query_key": query_key}),)
    )
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
        oracle_metadata=OracleMetadata(label="K", primitive_operations=(BIND_OPERATION,)),
        symbol_permutation=None,
    )


@dataclass(frozen=True)
class BindCounterfactualGroup:
    """One content sequence paired with >=`MIN_GROUP_SIZE` `BIND.query_key`
    values -- each drawn from a key actually present in the content
    (`content[0::2]`), matching `BindOp.sample_params` -- whose outputs are
    pairwise distinct. `examples[i]` is the real `Example` for
    `query_keys[i]`; every `examples[i].target_tokens` differs from every
    other member's, checked eagerly here rather than merely assumed from the
    construction that produced them."""

    input_tokens: tuple[int, ...]
    examples: tuple[Example, ...]
    query_keys: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.examples) != len(self.query_keys):
            raise ValueError(
                f"BindCounterfactualGroup requires len(examples) == len(query_keys), got "
                f"{len(self.examples)} and {len(self.query_keys)}"
            )
        if len(self.examples) < MIN_GROUP_SIZE:
            raise ValueError(
                f"BindCounterfactualGroup requires >= {MIN_GROUP_SIZE} members, got "
                f"{len(self.examples)}"
            )
        outputs = [example.target_tokens for example in self.examples]
        if len(set(outputs)) != len(outputs):
            raise ValueError(
                "BindCounterfactualGroup requires pairwise distinct outputs across its "
                f"members; got outputs {outputs} for query_keys {self.query_keys}"
            )

    def wrong_key_index(self, member_index: int) -> int:
        """The index of a group member other than `member_index`, guaranteed
        to have a different output (`__post_init__`'s own invariant) --
        deterministic "next member, wrapping around" choice used to build the
        "Wrong argument" arm."""
        return (member_index + 1) % len(self.examples)


def _generate_bind_counterfactual_group(
    rng: random.Random,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    group_size: int,
    split: str,
) -> BindCounterfactualGroup:
    valid_lengths = _valid_bind_lengths(sequence_length_range)
    if not valid_lengths:
        raise ValueError(
            f"sequence_length_range {sequence_length_range} contains no even length >= 2 -- "
            "BIND requires an even-length key/value input (apc.environments.operations."
            "BindOp.is_valid_for_length); widen sequence_length_range"
        )
    for _ in range(_MAX_CONTENT_RESAMPLE_ATTEMPTS):
        length = rng.choice(valid_lengths)
        input_tokens = tuple(rng.randrange(vocab_size) for _ in range(length))

        candidate_keys = list(dict.fromkeys(input_tokens[0::2]))
        rng.shuffle(candidate_keys)
        chosen_keys: list[int] = []
        seen_outputs: set[int] = set()
        for key in candidate_keys:
            output = _bind_output(input_tokens, key)
            if output in seen_outputs:
                continue
            seen_outputs.add(output)
            chosen_keys.append(key)
            if len(chosen_keys) == group_size:
                break

        if len(chosen_keys) >= MIN_GROUP_SIZE:
            examples = tuple(
                _build_bind_example(input_tokens, key, vocab_size, split) for key in chosen_keys
            )
            return BindCounterfactualGroup(
                input_tokens=input_tokens, examples=examples, query_keys=tuple(chosen_keys)
            )

    raise RuntimeError(
        f"failed to construct a BIND counterfactual group with >= {MIN_GROUP_SIZE} pairwise "
        f"distinct outputs after {_MAX_CONTENT_RESAMPLE_ATTEMPTS} resamples (vocab_size="
        f"{vocab_size}, sequence_length_range={sequence_length_range}) -- widen "
        "sequence_length_range or vocab_size if this recurs"
    )


def generate_bind_counterfactual_groups(
    seed: int,
    n_groups: int,
    *,
    step: int,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[BindCounterfactualGroup, ...]:
    """Deterministic-by-`(seed, step, split)` counterfactual group batch,
    matching `count_counterfactual_gate.generate_count_counterfactual_groups`'s
    determinism convention (one shared `random.Random` stream for the whole
    call, re-derived from `(seed, step, split)`) so replaying the same
    arguments reproduces the same groups.
    """
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if group_size < MIN_GROUP_SIZE:
        raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {group_size}")
    rng = random.Random(_derive_local_seed(seed, step, split))
    return tuple(
        _generate_bind_counterfactual_group(
            rng, vocab_size, sequence_length_range, group_size, split
        )
        for _ in range(n_groups)
    )


def _flatten_groups(
    groups: Sequence[BindCounterfactualGroup],
) -> tuple[list[Example], dict[int, PrimitiveCall], dict[int, tuple[int, ...]]]:
    """Flatten `groups` into one example list plus two `id(example)`-keyed
    side tables: `wrong_call_by_id` (the "Wrong argument" `PrimitiveCall` to
    force for that example -- a different group member's key, per
    `BindCounterfactualGroup.wrong_key_index`) and `wrong_target_tokens_by_id`
    (that partner's own ground-truth output, used only to report
    `argument_effect_rate`)."""
    examples: list[Example] = []
    wrong_call_by_id: dict[int, PrimitiveCall] = {}
    wrong_target_tokens_by_id: dict[int, tuple[int, ...]] = {}
    for group in groups:
        for index, example in enumerate(group.examples):
            partner_index = group.wrong_key_index(index)
            wrong_call_by_id[id(example)] = PrimitiveCall(
                operation=BIND_OPERATION,
                arguments={"query_key": group.query_keys[partner_index]},
            )
            wrong_target_tokens_by_id[id(example)] = group.examples[partner_index].target_tokens
            examples.append(example)
    return examples, wrong_call_by_id, wrong_target_tokens_by_id


@dataclass(frozen=True)
class BindCounterfactualGateConfig:
    """Explicit, serializable configuration for one seed's A1-R005D-007 gate
    run. `primitive_rank`/`arg_dim`/`core_train`/`primitive_train` default to
    the same baseline numbers `configs/phase_a1_count_counterfactual_gate.yaml`
    pinned for A1-R005D-004, per this task's own "same baseline architecture"
    scope (module docstring)."""

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
        if not _valid_bind_lengths(self.sequence_length_range):
            raise ValueError(
                f"sequence_length_range {self.sequence_length_range} contains no even length "
                ">= 2 -- BIND requires an even-length key/value input (apc.environments."
                "operations.BindOp.is_valid_for_length); widen sequence_length_range"
            )
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= the upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]}) so this config stays "
                "internally consistent (matching count_counterfactual_gate's own convention)"
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


def bind_counterfactual_gate_config_from_dict(
    raw: dict[str, Any],
) -> BindCounterfactualGateConfig:
    """Parse a `configs/phase_a1_bind_counterfactual_gate.yaml`-shaped dict,
    matching every other Phase A.1 gate's convention of filling in defaults
    for whatever the file omits."""
    defaults = BindCounterfactualGateConfig()
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
    return BindCounterfactualGateConfig(
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
    """A Stable Core pretrained task-blind on ordinary i.i.d. `BIND`
    examples, then frozen: every parameter has `requires_grad=False`. Not
    shared with `count_counterfactual_gate.FrozenStableCore` (or
    `parameterized_primitive_gate.FrozenStableCore`) -- each operation-
    specific gate module defines its own (module docstring)."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: BindCounterfactualGateConfig, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=(BIND_OPERATION,),
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
) -> PrimitiveBank:
    """Exactly one argument-conditioned `BIND` primitive (V0/additive,
    `ConditionedPrimitive`), keyed at `operation_id("BIND")` (ADR-0024's
    convention) -- "one BIND family" (module docstring, "Why there is no
    'Wrong family' arm")."""
    bank = PrimitiveBank()
    primitive = build_conditioned_primitive(
        operation_id(BIND_OPERATION),
        BIND_OPERATION,
        PrimitiveConfig(d_model=d_model, rank=rank),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        arg_dim=arg_dim,
        status=PrimitiveStatus.CANDIDATE,
        metadata={"operation": BIND_OPERATION},
    )
    bank.add_primitive(primitive)
    return bank


def _train_primitives(
    core: FrozenStableCore,
    bank: PrimitiveBank,
    config: BindCounterfactualGateConfig,
    metrics_path: str | Path | None = None,
) -> tuple[float, int]:
    """Train `bank`'s one `ConditionedPrimitive` via oracle-forced routing
    through `core`'s frozen weights, on counterfactual groups
    (`generate_bind_counterfactual_groups`) -- the same content appears
    multiple times per step, each time paired with a different (correct)
    key, so content alone can never explain the target within a step
    (`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "Counterfactual training
    rule").

    Returns `(final_loss, total_examples_seen)`.
    """
    model, tokens, device = core.model, core.tokens, core.device
    bank.to(device)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.primitive_train.lr,
        weight_decay=config.primitive_train.weight_decay,
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
            groups = generate_bind_counterfactual_groups(
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
                progress_groups = generate_bind_counterfactual_groups(
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
class BindCounterfactualGateReport:
    """Everything observed while pretraining, freezing, training the one
    `BIND` primitive on top of, and evaluating one seed's three-arm causal
    ablation (Correct / effectful Wrong argument / None)."""

    config: BindCounterfactualGateConfig
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


def run_bind_counterfactual_gate(
    config: BindCounterfactualGateConfig,
    *,
    core_metrics_path: str | Path | None = None,
    primitive_metrics_path: str | Path | None = None,
) -> BindCounterfactualGateReport:
    """Run one seed of the A1-R005D-007 gate end to end: pretrain and freeze
    a task-blind `BIND`-only Stable Core, train one argument-conditioned
    `ConditionedPrimitive` (V0/additive) on counterfactual groups, then
    evaluate Correct/effectful Wrong argument/None on a large,
    independently-drawn unseen batch of counterfactual groups."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, metrics_path=core_metrics_path)
    bank = _build_primitive_bank(
        core.model.config.d_model,
        config.primitive_rank,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    )
    final_primitive_loss, primitive_examples_seen = _train_primitives(
        core, bank, config, metrics_path=primitive_metrics_path
    )

    unseen_groups = generate_bind_counterfactual_groups(
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

    return BindCounterfactualGateReport(
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
class BindCounterfactualGateMultiSeedReport:
    """A1-R005D-007's verdict aggregated across seeds. `passed` is computed
    against the cross-seed *mean* of each arm, matching every other Phase
    A.1 gate's own convention."""

    seeds: tuple[int, ...]
    per_seed: tuple[BindCounterfactualGateReport, ...]
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


def run_bind_counterfactual_gate_multi_seed(
    base_config: BindCounterfactualGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    correct_threshold: float = CORRECT_THRESHOLD,
    effectful_wrong_argument_ceiling: float = EFFECTFUL_WRONG_ARGUMENT_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
) -> BindCounterfactualGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 gate.
    """
    per_seed: list[BindCounterfactualGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        core_metrics_path = (
            run_dir_path / f"seed_{seed}" / "core_metrics.jsonl" if run_dir_path else None
        )
        primitive_metrics_path = (
            run_dir_path / f"seed_{seed}" / "primitive_metrics.jsonl" if run_dir_path else None
        )
        report = run_bind_counterfactual_gate(
            config,
            core_metrics_path=core_metrics_path,
            primitive_metrics_path=primitive_metrics_path,
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

    return BindCounterfactualGateMultiSeedReport(
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
