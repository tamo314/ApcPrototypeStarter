"""Task-blind content encoder path gate (Phase A.1 Post-Correction Task
A1-R001, STOP GATE, H2a).

`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R001 / `docs/
EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md` section 2 (H2a): before any
primitive-causality claim can be trusted (A1-R003 onward), the primitive
input state (`h_content`) must be *invariant* to task specification for
identical content -- otherwise a "primitive" could still be routing
task-conditioned information around the primitive computation itself,
exactly the ambiguity `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`
section 1 raises.

## Why the previous factorization was not enough

`apc.core.model.DecoderOnlyTransformer.encode_split` (Task A1-005) reads
`task_state`/`content_state` as two renamed views of *one* causal forward
pass over `[BOS] [TASK_START] op arg... [TASK_END] input... [SEP]`. Per
`docs/DECISIONS.md` ADR-0022, a content position's hidden state in that
pass has, by construction, already attended over every task token before
it -- so `content_state` there is `f(task, content)`, not `f(content)`,
regardless of what the model learned. That was fine for A1-C005's probing
question ("is task information linearly decodable somewhere"), but it is
the wrong representation to hand to a primitive that is supposed to prove
it -- not merely "solve the task well" -- is doing the operation-specific
work.

## This module's fix

`apc.core.model.DecoderOnlyTransformer.encode_task_content_split` (this
task) runs `encode` twice through the *same* shared weights: once over a
task-only sequence (`apc.core.data.build_task_only_tokens`) and once over a
content-only sequence (`apc.core.data.build_content_only_tokens`), per
`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 3's "shared
weights, called twice" minimal design. `content_state` therefore cannot
depend on task specification for *any* weights, trained or not: there is no
token in the content-only sequence's own input that could carry task
information, so no amount of training could ever create a path from task to
content through this encoding. This is a structural guarantee, strictly
stronger than what any empirical leakage probe on a trained model could
show (docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md section 4: "this is
stronger than a leakage probe").

## Why this gate uses freshly-initialized weights, not a trained core

The invariant under test is architectural, not learned: it holds for *any*
`DecoderOnlyTransformer` weights, because it follows from which tokens are
in the content-only sequence, not from what the model learned to do with
them. Training the core first would not make the test more informative --
it would only make it slower and would not change what is being measured.
Demonstrating the invariant holds for several independently-initialized
seeds (`docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md` section 2's
"verified over >=5 seeds / generated batches") is therefore both sufficient
and appropriately minimal (`AGENTS.md`: "prefer the smallest implementation
that can falsify the current hypothesis"). Whether a *trained* task-blind
content representation is actually good enough to *support* primitive
execution is the next task's question (A1-R003's parameter-free oracle
primitive benchmark), not this one's.

## What this gate checks

For each of several procedurally generated "contents" (spanning all eight
`apc.environments.operations.KNOWN_OPERATION_NAMES`, Task A1-R001's "all 8
operation families covered" acceptance item):

1. **Token-level invariance.** `build_content_only_tokens` is recomputed
   after swapping the example's `task_spec` for a freshly sampled
   alternate (different operation and/or different arguments); the
   resulting token tuple must be byte-identical to the original, since the
   function's own signature has no `TaskSpec` parameter to read from.
2. **Representation-level invariance.** `encode_task_content_split` is run
   once per alternate task spec (task-only sequence varies; content-only
   sequence never does); `content_state` must be numerically identical
   (within `config.atol`) across every alternate, and must equal plain
   `model.encode(content_ids)` computed directly.
3. **No task token in content-encoder input.** No id in any content-only
   token sequence falls inside `apc.core.tokens.SharedCoreTokens`' task
   segment/operation/argument ranges.
4. **Padding invariance** (a real regression guard, not a tautology): the
   same content's `content_state`, computed once alone and once batched
   alongside other differently-lengthed contents with right-padding, must
   agree at the content's own (non-pad) positions.

`apc.primitives`, `apc.plastic`, `apc.consolidation`, and `apc.meta` are
never imported here, matching the scope discipline of every earlier
Phase A.1 Correction gate: this module tests the encoder path only, not
routing or primitive execution (A1-R003 onward).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from apc.core.data import build_content_only_tokens, build_task_only_tokens, pad_token_sequences
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.core.tokens import task_token_id_set as _task_token_id_set
from apc.core.train import resolve_device
from apc.environments.generator import Example, build_mixed_operation_generator
from apc.environments.operations import KNOWN_OPERATION_NAMES, get_operation
from apc.environments.task_spec import (
    TaskSpec,
    TaskStepSpec,
    default_argument_value_span,
    num_registered_operations,
)
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "ContentInvarianceRecord",
    "TaskBlindContentGateConfig",
    "task_blind_content_gate_config_from_dict",
    "TaskBlindContentGateReport",
    "TaskBlindContentGateMultiSeedReport",
    "run_task_blind_content_gate",
    "run_task_blind_content_gate_multi_seed",
]

# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md section 2 (H2a): "verified
# over >=5 seeds / generated batches".
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 48,
        "dropout": 0.0,
    }


def _derive_local_seed(seed: int, content_index: int, label: str) -> int:
    """Deterministic sub-seed for one `(content_index, label)` pair,
    independent of `PYTHONHASHSEED` -- same construction as `apc.
    environments.generator._derive_seed`, kept local here since that helper
    is private to its own module. Used both to seed one operation's
    dedicated content generator (`content_index=-1`, `label=operation
    name`) and to seed one alternate-argument sampler (`content_index` is
    the content's own index, `label` is the alternate operation name)."""
    digest = hashlib.sha256(f"{seed}:{content_index}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class TaskBlindContentGateConfig:
    """Explicit, serializable configuration for one seed's A1-R001 gate run."""

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES
    model: dict[str, Any] = field(default_factory=_default_model_config)
    num_contents_per_operation: int = 8
    min_alternate_task_specs_per_content: int = 2
    atol: float = 1e-5
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.num_contents_per_operation < 1:
            raise ValueError(
                f"num_contents_per_operation must be >= 1, got {self.num_contents_per_operation}"
            )
        if self.min_alternate_task_specs_per_content < 1:
            raise ValueError(
                "min_alternate_task_specs_per_content must be >= 1, got "
                f"{self.min_alternate_task_specs_per_content}"
            )
        if self.atol < 0:
            raise ValueError(f"atol must be >= 0, got {self.atol}")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def task_blind_content_gate_config_from_dict(raw: dict[str, Any]) -> TaskBlindContentGateConfig:
    """Parse a `configs/phase_a1_task_blind_content_gate.yaml`-shaped dict,
    matching the other Phase A.1 gates' convention of filling in defaults
    for whatever the file omits."""
    defaults = TaskBlindContentGateConfig()
    return TaskBlindContentGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        model=dict(raw.get("model", defaults.model)),
        num_contents_per_operation=raw.get(
            "num_contents_per_operation", defaults.num_contents_per_operation
        ),
        min_alternate_task_specs_per_content=raw.get(
            "min_alternate_task_specs_per_content", defaults.min_alternate_task_specs_per_content
        ),
        atol=raw.get("atol", defaults.atol),
        device=raw.get("device", defaults.device),
    )


def _build_tokens(config: TaskBlindContentGateConfig) -> SharedCoreTokens:
    return build_shared_core_tokens(
        config.vocab_size,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(config.vocab_size, config.sequence_length_range),
    )


@dataclass(frozen=True)
class ContentInvarianceRecord:
    """One generated content's invariance check across every alternate task
    specification tested against it."""

    content_index: int
    content_length: int
    original_operation: str
    alternate_operations: tuple[str, ...]
    max_absolute_difference: float
    token_level_invariant: bool
    task_token_leak_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_index": self.content_index,
            "content_length": self.content_length,
            "original_operation": self.original_operation,
            "alternate_operations": list(self.alternate_operations),
            "max_absolute_difference": self.max_absolute_difference,
            "token_level_invariant": self.token_level_invariant,
            "task_token_leak_ids": list(self.task_token_leak_ids),
        }


@dataclass(frozen=True)
class TaskBlindContentGateReport:
    """Everything observed while running one seed of the A1-R001 gate."""

    config: TaskBlindContentGateConfig
    num_contents_tested: int
    per_operation_content_counts: dict[str, int]
    per_operation_alternate_counts: dict[str, int]
    all_operations_covered: bool
    representation_max_absolute_difference: float
    representation_invariant: bool
    token_level_invariant: bool
    padding_max_absolute_difference: float
    padding_invariant: bool
    no_task_token_leak: bool
    passed: bool
    wall_clock_seconds: float
    device: str
    per_content: tuple[ContentInvarianceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "num_contents_tested": self.num_contents_tested,
            "per_operation_content_counts": self.per_operation_content_counts,
            "per_operation_alternate_counts": self.per_operation_alternate_counts,
            "all_operations_covered": self.all_operations_covered,
            "representation_max_absolute_difference": self.representation_max_absolute_difference,
            "representation_invariant": self.representation_invariant,
            "token_level_invariant": self.token_level_invariant,
            "padding_max_absolute_difference": self.padding_max_absolute_difference,
            "padding_invariant": self.padding_invariant,
            "no_task_token_leak": self.no_task_token_leak,
            "passed": self.passed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
            "per_content": [record.to_dict() for record in self.per_content],
        }


def run_task_blind_content_gate(config: TaskBlindContentGateConfig) -> TaskBlindContentGateReport:
    """Run one seed of the A1-R001 task-blind content encoder path gate.

    Builds a freshly-initialized (never trained -- see module docstring)
    `DecoderOnlyTransformer`, draws procedurally generated content spanning
    every operation in `config.operation_names`, and checks token-level and
    representation-level invariance of `content_state` to task
    specification, absence of task tokens from the content-encoder input,
    and invariance to batch padding.
    """
    start = time.perf_counter()
    set_seed(config.seed)
    device = resolve_device(config.device)

    tokens = _build_tokens(config)
    model_config = TransformerConfig(vocab_size=tokens.model_vocab_size, **config.model)
    model = DecoderOnlyTransformer(model_config).to(device)
    model.eval()
    # Re-anchor the shared global RNG stream after model construction
    # (ADR-0016 pattern) so generated content does not depend on the
    # model's own parameter count.
    set_seed(config.seed)

    # One dedicated single-operation generator per requested operation,
    # each drawing exactly `num_contents_per_operation` examples -- this
    # guarantees "all 8 operation families covered" by construction rather
    # than leaving per-operation coverage to chance under uniform sampling
    # from one combined mixed-operation stream (a small draw from the
    # latter can easily miss a rarer operation entirely).
    examples: list[Example] = []
    per_operation_content_counts: dict[str, int] = {}
    for name in config.operation_names:
        operation_generator = build_mixed_operation_generator(
            seed=_derive_local_seed(config.seed, -1, f"content_generator:{name}"),
            operation_names=(name,),
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        op_examples = operation_generator.generate_online(
            config.num_contents_per_operation, step=0, split="test"
        )
        examples.extend(op_examples)
        per_operation_content_counts[name] = len(op_examples)

    task_token_ids = _task_token_id_set(tokens)
    per_operation_alternate_counts: dict[str, int] = dict.fromkeys(config.operation_names, 0)
    per_content: list[ContentInvarianceRecord] = []
    singleton_content_states: list[torch.Tensor] = []
    content_id_sequences: list[tuple[int, ...]] = []
    representation_max_diff = 0.0

    with torch.no_grad():
        for index, example in enumerate(examples):
            assert example.task_spec is not None
            content_ids = build_content_only_tokens(example, tokens)
            content_id_sequences.append(content_ids)
            leak_ids = tuple(sorted({tid for tid in content_ids if tid in task_token_ids}))

            length = len(example.input_tokens)
            alternate_names = [
                name
                for name in config.operation_names
                if get_operation(name).is_valid_for_length(length)
            ]
            if len(alternate_names) < config.min_alternate_task_specs_per_content:
                raise ValueError(
                    f"content {index} (length {length}) admits only "
                    f"{len(alternate_names)} valid operation(s), below "
                    f"min_alternate_task_specs_per_content="
                    f"{config.min_alternate_task_specs_per_content}; widen "
                    "sequence_length_range or lower the threshold"
                )

            content_batch = torch.tensor([content_ids], dtype=torch.long, device=device)
            reference_content_state = model.encode(content_batch)
            singleton_content_states.append(reference_content_state.squeeze(0))

            content_token_level_ok = True
            content_max_diff = 0.0
            for alt_name in alternate_names:
                rng = random.Random(_derive_local_seed(config.seed, index, alt_name))
                params = get_operation(alt_name).sample_params(
                    rng, example.input_tokens, config.vocab_size
                )
                alt_step = TaskStepSpec(operation=alt_name, arguments=params)
                alt_task_spec = TaskSpec(steps=(alt_step,))

                # Token-level invariance: content-only tokens must be
                # unaffected by swapping the task spec attached to this
                # content (the acceptance criterion's literal wording).
                alt_example = dataclasses.replace(example, task_spec=alt_task_spec)
                alt_content_ids = build_content_only_tokens(alt_example, tokens)
                if alt_content_ids != content_ids:
                    content_token_level_ok = False

                task_ids = build_task_only_tokens(alt_task_spec, tokens)
                task_batch = torch.tensor([task_ids], dtype=torch.long, device=device)
                encoded = model.encode_task_content_split(task_batch, content_batch)

                diff = (encoded.content_state - reference_content_state).abs().max().item()
                content_max_diff = max(content_max_diff, diff)
                per_operation_alternate_counts[alt_name] += 1

            representation_max_diff = max(representation_max_diff, content_max_diff)
            per_content.append(
                ContentInvarianceRecord(
                    content_index=index,
                    content_length=length,
                    original_operation=example.task_spec.operation_sequence[0],
                    alternate_operations=tuple(alternate_names),
                    max_absolute_difference=content_max_diff,
                    token_level_invariant=content_token_level_ok,
                    task_token_leak_ids=leak_ids,
                )
            )

        # Padding invariance: re-encode every content in one right-padded
        # batch and compare each content's own (non-pad) positions against
        # its singleton encoding above.
        padded_batch = pad_token_sequences(content_id_sequences, tokens.pad, device)
        padded_content_states = model.encode(padded_batch)
        padding_max_diff = 0.0
        for index, content_ids in enumerate(content_id_sequences):
            real_length = len(content_ids)
            batched_slice = padded_content_states[index, :real_length, :]
            diff = (batched_slice - singleton_content_states[index]).abs().max().item()
            padding_max_diff = max(padding_max_diff, diff)

    token_level_invariant = all(record.token_level_invariant for record in per_content)
    no_task_token_leak = all(not record.task_token_leak_ids for record in per_content)
    representation_invariant = representation_max_diff <= config.atol
    padding_invariant = padding_max_diff <= config.atol
    all_operations_covered = all(
        per_operation_content_counts[name] > 0 or per_operation_alternate_counts[name] > 0
        for name in config.operation_names
    )
    passed = (
        token_level_invariant
        and no_task_token_leak
        and representation_invariant
        and padding_invariant
        and all_operations_covered
    )

    return TaskBlindContentGateReport(
        config=config,
        num_contents_tested=len(examples),
        per_operation_content_counts=per_operation_content_counts,
        per_operation_alternate_counts=per_operation_alternate_counts,
        all_operations_covered=all_operations_covered,
        representation_max_absolute_difference=representation_max_diff,
        representation_invariant=representation_invariant,
        token_level_invariant=token_level_invariant,
        padding_max_absolute_difference=padding_max_diff,
        padding_invariant=padding_invariant,
        no_task_token_leak=no_task_token_leak,
        passed=passed,
        wall_clock_seconds=time.perf_counter() - start,
        device=str(device),
        per_content=tuple(per_content),
    )


@dataclass(frozen=True)
class TaskBlindContentGateMultiSeedReport:
    """A1-R001's verdict aggregated across seeds. `passed` is a per-seed-
    then-ANDed verdict (every seed's own report must pass), matching `apc.
    evaluation.task_content_probes.TaskContentProbeMultiSeedReport`'s
    convention for a gate with no mean-based threshold."""

    seeds: tuple[int, ...]
    per_seed: tuple[TaskBlindContentGateReport, ...]
    max_representation_absolute_difference: float
    max_padding_absolute_difference: float
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "max_representation_absolute_difference": self.max_representation_absolute_difference,
            "max_padding_absolute_difference": self.max_padding_absolute_difference,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_task_blind_content_gate_multi_seed(
    base_config: TaskBlindContentGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> TaskBlindContentGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 multi-seed gate."""
    per_seed = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        report = run_task_blind_content_gate(config)
        if run_dir_path is not None:
            seed_dir = run_dir_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    max_repr_diff = max(report.representation_max_absolute_difference for report in per_seed)
    max_padding_diff = max(report.padding_max_absolute_difference for report in per_seed)

    return TaskBlindContentGateMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        max_representation_absolute_difference=max_repr_diff,
        max_padding_absolute_difference=max_padding_diff,
        passed=all(report.passed for report in per_seed),
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
