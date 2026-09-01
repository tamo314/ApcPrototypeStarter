"""Parameter-free oracle primitive benchmark (Phase A.1 Post-Correction Task
A1-R003, STOP GATE, H2b).

`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R003: for
`apc.environments.operations.DETERMINISTIC_OPERATION_NAMES` (`COPY`,
`NEGATE`, `COMPARE`, `ACCUMULATE` -- the four known operations with no hidden
per-instance parameter, `docs/DECISIONS.md` ADR-0017), pre-register one
neural primitive family per operation, freeze the Stable Core, train the
primitives, and measure the causal ablation matrix's three arms (`docs/
design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 8): **Correct** (the
oracle-forced correct primitive family), **Wrong** (an oracle-forced
*incorrect* family), and **None** (no primitive at all, `apc.core.execution.
evaluate_exact_match_no_primitive` -- A1-R002's own STOP GATE arm). A
primitive is causally supported only if Correct is high while Wrong and None
are materially lower (`AGENTS.md`'s causal primitive evidence rule).

## How the Stable Core gets "frozen"

`AGENTS.md`'s Stable Core role permits it to provide "task-independent
content encoding, representation transport, decoding infrastructure" but
forbids it from "perform[ing] the operation-specific transformation before
primitive execution." A Stable Core frozen at random initialization would
satisfy the second half trivially but not the first: `apc.core.model.
DecoderOnlyTransformer.decode` is a `LayerNorm`-normalized hidden state
dot-producted against (tied) token embeddings, so a primitive's rank-limited
residual delta needs a working content-encoding/decoding scaffold under it
to have anything to build on -- with a genuinely untrained core, a Correct/
Wrong/None result would be uninterpretable either way (`AGENTS.md`: "If
Correct, Wrong, and None are all low, suspect content representation or
primitive capacity").

This gate resolves that by **pretraining then freezing**: `_pretrain_frozen_
stable_core` reuses `apc.evaluation.shared_core_generalization.
train_shared_core` (the exact same procedure Task A1-R002's decoder leakage
gate already reuses for its own "None" arm, ADR-0026) with
`include_task_spec=False` and `operation_names` restricted to
`DETERMINISTIC_OPERATION_NAMES` -- the Stable Core trains on the online
mixed stream of all four operations, but the task segment is never part of
its input, so it structurally cannot know *which* operation a given example
requires (mirroring A1-C004/A1-R002's own non-identifiability finding,
ADR-0017/ADR-0020/ADR-0021). This gives the core genuine pressure to learn
useful token/content/position representation and decoding infrastructure
(including cross-position aggregation via its causal self-attention layers,
which parameter-free operations like `ACCUMULATE`/`COMPARE` need) without
ever being able to resolve the operation-specific transformation itself --
exactly the role `AGENTS.md` assigns it. Every one of its parameters is then
frozen (`requires_grad_(False)`) before any primitive is constructed or
trained. `apc.evaluation.shared_core_generalization`'s own explicit-task,
high-performing variant (Task A1-C004) is never touched here, matching
`AGENTS.md`'s "preserve the earlier high-performing shared-core solver ...
as a baseline, not as the causal primitive path."

## Training primitives through the frozen core

`apc.core.execution.forward_logits_with_oracle_calls_trainable` (this task's
addition to that module) is used instead of the no-grad A1-C007 evaluation
entry point: it runs the frozen `encode`, then oracle-forces exactly the
example's own correct primitive (`apc.environments.generator.
oracle_call_for_example`) via `apply_bank_with_oracle_calls_trainable`, then
`decode`s -- with a gradient path back into the selected primitive's
`a_proj`/`b_proj` weights and nowhere else (the frozen core's own output
carries no grad; an unselected primitive family is never touched for that
example). The optimizer step is scoped to `bank.parameters()` only. Training
data and the loss shape (token-level cross-entropy over the answer span,
`apc.core.data.collate_batch`) mirror `train_shared_core` exactly, so the
primitive-training loop is directly comparable to the core-pretraining loop
that precedes it.

## Correct / Wrong / None

- **Correct**: `apc.core.execution.evaluate_exact_match_with_oracle_calls`
  with the default `oracle_call_for_example` provider -- the example's own
  correct operation, exactly A1-C007's existing evaluation entry point,
  unmodified.
- **Wrong**: the same function with a substituted `oracle_call_provider`
  (A1-C007 Work item 3's own injection point, `tests/test_oracle_routing.py::
  test_evaluate_exact_match_with_oracle_calls_uses_the_injected_provider`)
  that forces a *different* operation's family, chosen via a fixed
  derangement over `config.operation_names` (`_default_wrong_operation_map`,
  every operation maps to a different one, deterministic and reproducible --
  see that function's docstring). Every current `DETERMINISTIC_OPERATION_
  NAMES` member is parameter-free (`required_argument_names == frozenset()`),
  so constructing `PrimitiveCall(operation=wrong_name)` for any example's
  content is always well-formed regardless of that example's real operation.
- **None**: `apc.core.execution.evaluate_exact_match_no_primitive` -- no
  bank, router, or workspace anywhere in the call graph, reusing A1-R002's
  own "None" arm entry point unmodified.

## Scope discipline

Same restriction as every earlier Phase A.1 Correction/Post-Correction gate:
`apc.plastic`, `apc.consolidation`, and `apc.meta` are never imported here.
`apc.primitives` is imported (unlike A1-R001/A1-R002) because this is the
first Post-Correction gate whose entire point is primitive execution; no
`Router` is imported or reachable anywhere in this module (oracle routing
only, per A1-C007's "the learned path never receives oracle calls").
Composition (`apc.environments.primitive_call`'s multi-call chains),
argument-conditioned primitives (`SHIFT`/`SELECT`/`COUNT`/`BIND`, Task
A1-R004/A1-R005), and any promotion of these trained primitives to a
persistent/STABLE bank are all out of scope -- this gate constructs a fresh
`PrimitiveBank` of `CANDIDATE`-status primitives per run and discards it
after measuring the three arms.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Callable, Sequence
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
from apc.environments.generator import Example, TaskGenerator, oracle_call_for_example
from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES, get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import operation_id
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "CORRECT_THRESHOLD",
    "WRONG_CEILING",
    "NONE_CEILING",
    "MIN_CAUSAL_GAP",
    "PrimitiveTrainConfig",
    "ParameterFreePrimitiveGateConfig",
    "parameter_free_primitive_gate_config_from_dict",
    "FrozenStableCore",
    "ParameterFreePrimitiveGateReport",
    "run_parameter_free_primitive_gate",
    "ParameterFreePrimitiveGateMultiSeedReport",
    "run_parameter_free_primitive_gate_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md: ">=5 seeds" for a gate claim.
MIN_GATE_SEEDS = 5

# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md section 3 (H2b).
CORRECT_THRESHOLD = 0.95
WRONG_CEILING = 0.30
NONE_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 48,
        "dropout": 0.0,
    }


@dataclass(frozen=True)
class PrimitiveTrainConfig:
    """Explicit, serializable optimization budget for training the
    primitive bank on top of one seed's frozen Stable Core."""

    steps: int = 8000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 1000
    progress_eval_examples: int = 128

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if self.progress_eval_examples < 1:
            raise ValueError(
                f"progress_eval_examples must be >= 1, got {self.progress_eval_examples}"
            )


@dataclass(frozen=True)
class ParameterFreePrimitiveGateConfig:
    """Explicit, serializable configuration for one seed's gate run.

    `operation_names` defaults to `DETERMINISTIC_OPERATION_NAMES` (A1-R003's
    own scope); every entry must be parameter-free (`Operation.
    required_argument_names` empty) so a "Wrong family" oracle call is always
    constructible with no arguments regardless of which example it is forced
    onto -- see module docstring's "Wrong" arm.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = DETERMINISTIC_OPERATION_NAMES
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    primitive_rank: int = 8
    primitive_train: PrimitiveTrainConfig = field(default_factory=PrimitiveTrainConfig)
    num_unseen_eval_examples: int = 2048
    min_examples_per_operation: int = 32

    def __post_init__(self) -> None:
        if len(self.operation_names) < 2:
            raise ValueError(
                "operation_names must have >= 2 entries: a 'Wrong family' control "
                "requires at least one other operation to force instead of the correct one"
            )
        non_parameter_free = [
            name
            for name in self.operation_names
            if get_operation(name).required_argument_names
        ]
        if non_parameter_free:
            raise ValueError(
                "ParameterFreePrimitiveGateConfig.operation_names must all be "
                f"parameter-free (Operation.required_argument_names == frozenset()); "
                f"got parameterized operation(s): {non_parameter_free}. Task A1-R003 is "
                "scoped to DETERMINISTIC_OPERATION_NAMES; parameterized primitives are "
                "Task A1-R004/A1-R005."
            )
        if self.num_unseen_eval_examples < 1:
            raise ValueError(
                f"num_unseen_eval_examples must be >= 1, got {self.num_unseen_eval_examples}"
            )
        if self.min_examples_per_operation < 1:
            raise ValueError(
                f"min_examples_per_operation must be >= 1, got {self.min_examples_per_operation}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def parameter_free_primitive_gate_config_from_dict(
    raw: dict[str, Any],
) -> ParameterFreePrimitiveGateConfig:
    """Parse a `configs/phase_a1_parameter_free_primitive_gate.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = ParameterFreePrimitiveGateConfig()
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
    return ParameterFreePrimitiveGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        primitive_rank=raw.get("primitive_rank", defaults.primitive_rank),
        primitive_train=primitive_train,
        num_unseen_eval_examples=raw.get(
            "num_unseen_eval_examples", defaults.num_unseen_eval_examples
        ),
        min_examples_per_operation=raw.get(
            "min_examples_per_operation", defaults.min_examples_per_operation
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind (see module docstring) and then
    frozen: every parameter has `requires_grad=False`. Carries its own
    `TaskGenerator` so primitive training draws from the identical stream
    (same seed, same operation pool, same vocabulary) the core itself was
    pretrained on."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    generator: TaskGenerator
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: ParameterFreePrimitiveGateConfig, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    """Pretrain one `DecoderOnlyTransformer` via `train_shared_core` with no
    task segment ever visible (`include_task_spec=False`), restricted to
    `config.operation_names`, then freeze every parameter -- see module
    docstring's "How the Stable Core gets frozen"."""
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=config.operation_names,
        permute_symbols=False,
        include_task_spec=False,
        model=config.model,
        train=config.core_train,
        num_unseen_eval_examples=config.num_unseen_eval_examples,
        min_examples_per_operation=config.min_examples_per_operation,
    )
    trained = train_shared_core(core_config, metrics_path=metrics_path)
    model = trained.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenStableCore(
        model=model,
        tokens=trained.tokens,
        generator=trained.generator,
        final_core_train_loss=trained.final_train_loss,
        device=trained.device,
    )


def _build_primitive_bank(
    operation_names: Sequence[str], d_model: int, rank: int
) -> PrimitiveBank:
    """One `Primitive`, keyed at `operation_id(name)` (ADR-0024's
    convention -- "an experiment that registers one bank primitive per
    operation at operation_id(name) gets oracle routing for free"), per
    entry in `operation_names`. Left in `CANDIDATE` status and fully
    trainable (`Primitive.__init__` does not freeze by default) -- unlike
    ADR-0004's "STABLE primitives are frozen" rule, these primitives are the
    thing this gate trains."""
    bank = PrimitiveBank()
    for name in operation_names:
        primitive = Primitive(
            operation_id(name),
            PrimitiveConfig(d_model=d_model, rank=rank),
            status=PrimitiveStatus.CANDIDATE,
            metadata={"operation": name},
        )
        bank.add_primitive(primitive)
    return bank


def _default_wrong_operation_map(operation_names: Sequence[str]) -> dict[str, str]:
    """A fixed derangement over `operation_names` (a cyclic shift): every
    operation maps to a different one, deterministic and reproducible across
    runs/seeds, used to force the causal ablation matrix's "Wrong family"
    arm (module docstring)."""
    names = list(operation_names)
    n = len(names)
    return {names[i]: names[(i + 1) % n] for i in range(n)}


def _wrong_call_provider(wrong_operation_map: dict[str, str]) -> Callable[[Example], PrimitiveCall]:
    def provider(example: Example) -> PrimitiveCall:
        correct = oracle_call_for_example(example)
        return PrimitiveCall(operation=wrong_operation_map[correct.operation])

    return provider


def _train_primitives(
    core: FrozenStableCore,
    bank: PrimitiveBank,
    config: ParameterFreePrimitiveGateConfig,
    metrics_path: str | Path | None = None,
) -> float:
    """Train `bank`'s primitives (only) via oracle-forced routing through
    `core`'s frozen weights -- see module docstring's "Training primitives
    through the frozen core"."""
    model, tokens, generator, device = core.model, core.tokens, core.generator, core.device
    bank.to(device)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.primitive_train.lr,
        weight_decay=config.primitive_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after bank/optimizer
    # construction, per ADR-0016's pattern.
    set_seed(config.seed)

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    try:
        for step in range(config.primitive_train.steps):
            examples = generator.generate_online(
                config.primitive_train.batch_size, step=step, split="train"
            )
            calls = [oracle_call_for_example(example) for example in examples]
            batch = collate_batch(examples, tokens, device=device, include_task_spec=False)

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
                progress_examples = generator.generate_online(
                    config.primitive_train.progress_eval_examples, step=step, split="val"
                )
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

    return final_loss


def _operation_of(example: Example) -> str:
    assert example.task_spec is not None
    (operation,) = example.task_spec.operation_sequence
    return operation


def _per_operation_exact_match(
    examples: Sequence[Example],
    predictions: Sequence[tuple[int, ...]],
    operation_names: Sequence[str],
    min_examples_per_operation: int,
) -> tuple[dict[str, float], dict[str, int]]:
    matches: dict[str, int] = dict.fromkeys(operation_names, 0)
    totals: dict[str, int] = dict.fromkeys(operation_names, 0)
    for example, prediction in zip(examples, predictions, strict=True):
        operation = _operation_of(example)
        totals[operation] += 1
        if prediction == example.target_tokens:
            matches[operation] += 1

    missing = [name for name in operation_names if totals[name] < min_examples_per_operation]
    if missing:
        raise ValueError(
            f"unseen eval batch has fewer than {min_examples_per_operation} examples for "
            f"operation(s) {missing}; increase num_unseen_eval_examples"
        )

    per_operation_exact_match = {name: matches[name] / totals[name] for name in operation_names}
    return per_operation_exact_match, totals


@dataclass(frozen=True)
class ParameterFreePrimitiveGateReport:
    """Everything observed while pretraining, freezing, training primitives
    on top of, and evaluating the causal ablation matrix for one seed."""

    config: ParameterFreePrimitiveGateConfig
    core_steps_trained: int
    core_examples_seen: int
    final_core_train_loss: float
    primitive_steps_trained: int
    primitive_examples_seen: int
    final_primitive_train_loss: float
    correct_exact_match: float
    wrong_exact_match: float
    none_exact_match: float
    causal_gap: float
    per_operation_correct_exact_match: dict[str, float]
    per_operation_wrong_exact_match: dict[str, float]
    per_operation_none_exact_match: dict[str, float]
    per_operation_eval_counts: dict[str, int]
    num_unseen_eval_examples: int
    wrong_operation_map: dict[str, str]
    core_param_count: int
    core_trainable_param_count: int
    primitive_param_count: int
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_parameter_free_primitive_gate(
    config: ParameterFreePrimitiveGateConfig,
    *,
    core_metrics_path: str | Path | None = None,
    primitive_metrics_path: str | Path | None = None,
    wrong_operation_map: dict[str, str] | None = None,
) -> ParameterFreePrimitiveGateReport:
    """Run one seed of the A1-R003 gate end to end: pretrain and freeze a
    Stable Core, train one primitive per `config.operation_names`, then
    evaluate Correct/Wrong/None on a large, independently-drawn unseen batch.
    """
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, metrics_path=core_metrics_path)
    bank = _build_primitive_bank(
        config.operation_names, core.model.config.d_model, config.primitive_rank
    )
    final_primitive_loss = _train_primitives(
        core, bank, config, metrics_path=primitive_metrics_path
    )

    resolved_wrong_map = wrong_operation_map or _default_wrong_operation_map(
        config.operation_names
    )

    unseen_examples = core.generator.generate_online(
        config.num_unseen_eval_examples, step=0, split="test"
    )

    correct_exact_match, correct_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model, unseen_examples, core.tokens, bank, device=core.device
    )
    wrong_exact_match, wrong_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model,
        unseen_examples,
        core.tokens,
        bank,
        _wrong_call_provider(resolved_wrong_map),
        device=core.device,
    )
    none_exact_match, none_predictions = evaluate_exact_match_no_primitive(
        core.model, unseen_examples, core.tokens, device=core.device
    )

    min_per_op = config.min_examples_per_operation
    per_operation_correct, per_operation_counts = _per_operation_exact_match(
        unseen_examples, correct_predictions, config.operation_names, min_per_op
    )
    per_operation_wrong, _ = _per_operation_exact_match(
        unseen_examples, wrong_predictions, config.operation_names, min_per_op
    )
    per_operation_none, _ = _per_operation_exact_match(
        unseen_examples, none_predictions, config.operation_names, min_per_op
    )

    causal_gap = correct_exact_match - max(wrong_exact_match, none_exact_match)

    return ParameterFreePrimitiveGateReport(
        config=config,
        core_steps_trained=config.core_train.steps,
        core_examples_seen=config.core_train.steps * config.core_train.batch_size,
        final_core_train_loss=core.final_core_train_loss,
        primitive_steps_trained=config.primitive_train.steps,
        primitive_examples_seen=config.primitive_train.steps * config.primitive_train.batch_size,
        final_primitive_train_loss=final_primitive_loss,
        correct_exact_match=correct_exact_match,
        wrong_exact_match=wrong_exact_match,
        none_exact_match=none_exact_match,
        causal_gap=causal_gap,
        per_operation_correct_exact_match=per_operation_correct,
        per_operation_wrong_exact_match=per_operation_wrong,
        per_operation_none_exact_match=per_operation_none,
        per_operation_eval_counts=per_operation_counts,
        num_unseen_eval_examples=len(unseen_examples),
        wrong_operation_map=resolved_wrong_map,
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
class ParameterFreePrimitiveGateMultiSeedReport:
    """A1-R003's verdict aggregated across seeds. `passed` is computed
    against the cross-seed *mean* of each arm, matching every other Phase
    A.1 gate's own convention (e.g. `SharedCoreGateMultiSeedReport`)."""

    seeds: tuple[int, ...]
    per_seed: tuple[ParameterFreePrimitiveGateReport, ...]
    operation_names: tuple[str, ...]
    mean_correct_exact_match: float
    stdev_correct_exact_match: float
    min_correct_exact_match: float
    max_correct_exact_match: float
    mean_wrong_exact_match: float
    mean_none_exact_match: float
    mean_causal_gap: float
    per_operation_mean_correct_exact_match: dict[str, float]
    correct_threshold: float
    wrong_ceiling: float
    none_ceiling: float
    min_causal_gap: float
    correct_passed: bool
    wrong_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "operation_names": list(self.operation_names),
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "stdev_correct_exact_match": self.stdev_correct_exact_match,
            "min_correct_exact_match": self.min_correct_exact_match,
            "max_correct_exact_match": self.max_correct_exact_match,
            "mean_wrong_exact_match": self.mean_wrong_exact_match,
            "mean_none_exact_match": self.mean_none_exact_match,
            "mean_causal_gap": self.mean_causal_gap,
            "per_operation_mean_correct_exact_match": self.per_operation_mean_correct_exact_match,
            "correct_threshold": self.correct_threshold,
            "wrong_ceiling": self.wrong_ceiling,
            "none_ceiling": self.none_ceiling,
            "min_causal_gap": self.min_causal_gap,
            "correct_passed": self.correct_passed,
            "wrong_passed": self.wrong_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_parameter_free_primitive_gate_multi_seed(
    base_config: ParameterFreePrimitiveGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    correct_threshold: float = CORRECT_THRESHOLD,
    wrong_ceiling: float = WRONG_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
) -> ParameterFreePrimitiveGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 gate.
    """
    per_seed: list[ParameterFreePrimitiveGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        core_metrics_path = (
            run_dir_path / f"seed_{seed}" / "core_metrics.jsonl" if run_dir_path else None
        )
        primitive_metrics_path = (
            run_dir_path / f"seed_{seed}" / "primitive_metrics.jsonl" if run_dir_path else None
        )
        report = run_parameter_free_primitive_gate(
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
    mean_wrong = statistics.fmean(report.wrong_exact_match for report in per_seed)
    mean_none = statistics.fmean(report.none_exact_match for report in per_seed)
    mean_causal_gap = mean_correct - max(mean_wrong, mean_none)

    per_operation_mean_correct: dict[str, float] = {}
    for name in base_config.operation_names:
        op_values = [report.per_operation_correct_exact_match[name] for report in per_seed]
        per_operation_mean_correct[name] = statistics.fmean(op_values)

    correct_passed = mean_correct >= correct_threshold
    wrong_passed = mean_wrong <= wrong_ceiling
    none_passed = mean_none <= none_ceiling
    causal_gap_passed = mean_causal_gap >= min_causal_gap
    passed = correct_passed and wrong_passed and none_passed and causal_gap_passed

    return ParameterFreePrimitiveGateMultiSeedReport(
        seeds=tuple(report.config.seed for report in per_seed),
        per_seed=tuple(per_seed),
        operation_names=tuple(base_config.operation_names),
        mean_correct_exact_match=mean_correct,
        stdev_correct_exact_match=stdev_correct,
        min_correct_exact_match=min_correct,
        max_correct_exact_match=max_correct,
        mean_wrong_exact_match=mean_wrong,
        mean_none_exact_match=mean_none,
        mean_causal_gap=mean_causal_gap,
        per_operation_mean_correct_exact_match=per_operation_mean_correct,
        correct_threshold=correct_threshold,
        wrong_ceiling=wrong_ceiling,
        none_ceiling=none_ceiling,
        min_causal_gap=min_causal_gap,
        correct_passed=correct_passed,
        wrong_passed=wrong_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
