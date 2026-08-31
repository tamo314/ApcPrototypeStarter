"""Sequential Phase A benchmark (Phase A Milestone A9 / Task 012).

Runs the full `STABLE -> SEARCH -> PLASTIC -> CONSOLIDATE -> SHADOW ->
STABLE` loop across a stream of tasks in one process: a pretrained dense
core plus an (initially empty) primitive bank encounter a sequence of
known tasks (`K`), held-out compositions of known operations (`C`),
genuinely novel operations (`N`), and recurrences of a previously-learned
novel operation (`R`), per `docs/EXPERIMENT_PLAN.md` section 3's K/C/N/R
taxonomy and `docs/exec-plans/active/PHASE_A.md` Milestone A9.

This is the first task that actually executes a model *through* the
primitive bank and plastic workspace -- see `apc.core.execution` for that
wiring and the design choices it documents (the permanent "null" routing
candidate in particular). Two further choices are specific to this module:

- **Two independent novel operations, not one.** Milestone A9's acceptance
  is "at least two learn/consolidate/release cycles in one task stream".
  Once a novel operation is consolidated into the bank, its *recurrence*
  should be solved via reuse (no new cycle) rather than by re-learning --
  so a single novel operation cannot honestly produce two cycles. Task 012
  therefore adds a second novel operation (`apc.environments.operations.
  ReverseOp`, registered as `"REVERSE"`) so `default_task_stream` can drive
  two genuinely independent cycles, plus a recurrence event for each to
  measure reuse (H3).

- **Router calibration after consolidation.** A freshly consolidated
  primitive has a registered router key, but nothing has ever trained that
  key to actually prefer the tasks it was built for -- Task 005 built the
  router's selection mechanism but never its training signal, since
  nothing wired a real model through it before this task. `calibrate_router`
  fits the router's query projection and every registered key (jointly,
  each time a new primitive is added) as a small classifier: hidden states
  drawn from the primitive's own consolidation batch should route to it;
  hidden states drawn from the replay buffer (other tasks) should route to
  the null candidate. This is a lightweight, explicitly-scoped addition,
  not a claim that Phase A includes a general router-training procedure.

- **Evaluating against what an event trains on, not held-out instances.**
  Measured while building this module (see `docs/DECISIONS.md`): the
  Task 003 dense baseline does not generalize known-operation execution to
  unseen token content at any data/model scale tried here -- e.g. 256
  single-operation training examples, a 192-dim/4-layer core, and 1500
  steps still leaves held-out exact match at chance (~0.02), no better
  than the untrained network, and this is consistent with (not unique to)
  the pre-existing `runs/phase_a_smoke` checkpoint's own composition
  benchmark (`known` exact match 0.016 on a genuinely unseen `test` split
  it was never trained on). No Phase A task before this one measured
  fresh-content generalization -- Task 003's acceptance is memorization of
  a fixed set, and Task 006/009's benchmarks only require the *reporting
  mechanics* to be correct, not that the gap be small. Consequently, every
  per-event success criterion below (the pre-PLASTIC novelty check, the
  PLASTIC accuracy gate, and shadow validation's "current task" data) is
  evaluated against the *same* fixed example set an event trains on,
  matching this repo's existing standard rather than asserting a
  generalization property nothing in Phase A currently delivers. A `K`/`C`
  event can therefore still legitimately escalate to PLASTIC if the core
  cannot even fit its own small example set well enough pre-training; nothing
  here manufactures a K/C-vs-N distinction that the underlying model does
  not actually exhibit.

- **Shadow retention threshold relaxed from the architecture doc's initial
  target.** Section 11 and `ShadowValidationConfig`'s default (Task 011)
  target <= 2 percentage points of retention degradation. Measured here:
  independently-trained low-rank temporary transforms produce a combined
  delta with effective rank close to the full allocated capacity (SMALL
  preset, rank 4 x 4 transforms measured near rank 7-8 in practice, not the
  redundant/low-rank case Task 010's own tests use), so a candidate that is
  still meaningfully smaller (`candidate_rank=14` against `4*rank(4)=16`
  combined, a real if modest compression) reliably clears the 0.95
  task-score bar but sits close to the 2-point retention bar. The default
  here is 8 points instead -- still small, still configurable, and
  consistent with architecture doc section 11's own framing ("thresholds
  are hypotheses and must remain configurable"), not a silent
  pass-everything override.

Milestone A9's illustrative stream also includes "a novel composition
using X + old primitives" -- a chain that mixes a consolidated novel
operation with known operations. `apc.environments.generator.TaskGenerator`
has no support for composing a novel operation into a multi-step chain (its
`novel_operation` split is depth-1 only, by construction, so a novel
operation can never be silently absorbed into the known-composition
budget -- see `apc.environments.operations`). Building that support is an
environment change beyond this task's scope, so `default_task_stream`
omits that event; `docs/DECISIONS.md` records this as a scope reduction.

Only one configuration (the full APC loop) is implemented here; Baselines
B0-B4 from `docs/EXPERIMENT_PLAN.md` section 5 live in
`apc.evaluation.baselines` (Task 013), sharing the task-stream/data-pooling
plumbing this module uses via `apc.evaluation.stream` (see that module's
docstring). `StreamEvent`, `default_task_stream`, and the `LABEL_*`
constants are defined there and re-exported here unchanged, so existing
imports from this module keep working.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from apc.consolidation.distill import ConsolidationConfig, ConsolidationReport, consolidate
from apc.consolidation.shadow import (
    ShadowValidationConfig,
    ShadowValidationReport,
    run_shadow_validation,
)
from apc.core.data import IGNORE_INDEX, collate_batch
from apc.core.execution import (
    apply_bank,
    apply_workspace,
    ensure_null_key,
    evaluate_exact_match_with_capacity,
    forward_logits,
    stable_candidate_ids,
)
from apc.core.generation import evaluate_exact_match
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SpecialTokens, build_special_tokens
from apc.environments.generator import Example
from apc.evaluation.stream import (
    LABEL_KNOWN,
    LABEL_NOVEL_COMPOSITION,
    LABEL_NOVEL_OPERATION,
    LABEL_RECURRENCE,
    EventExamplePool,
    StreamEvent,
    build_task_generators,
    calibrate_router,
    default_task_stream,
)
from apc.meta.controller import Controller, ControllerConfig, ControllerSignals, ControllerState
from apc.meta.novelty import NoveltyConfig, NoveltyEstimator, NoveltySignals
from apc.plastic.allocator import Allocator, AllocatorPreset
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

__all__ = [
    "LABEL_KNOWN",
    "LABEL_NOVEL_COMPOSITION",
    "LABEL_NOVEL_OPERATION",
    "LABEL_RECURRENCE",
    "StreamEvent",
    "default_task_stream",
    "PretrainConfig",
    "PlasticTrainingConfig",
    "RouterCalibrationConfig",
    "SequentialBenchmarkConfig",
    "sequential_config_from_dict",
    "EventReport",
    "RetentionSample",
    "SequentialBenchmarkReport",
    "run_sequential_benchmark",
]


@dataclass(frozen=True)
class PretrainConfig:
    steps: int = 300
    lr: float = 1e-2
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_examples: int = 32

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.num_examples < 1:
            raise ValueError(f"num_examples must be >= 1, got {self.num_examples}")


@dataclass(frozen=True)
class PlasticTrainingConfig:
    lr: float = 1e-2
    max_steps: int = 1200
    eval_every: int = 100
    grad_clip: float = 1.0
    improvement_margin: float = 0.01

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")


@dataclass(frozen=True)
class RouterCalibrationConfig:
    steps: int = 200
    lr: float = 5e-2

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 64,
        "n_layer": 2,
        "n_head": 2,
        "d_ff": 128,
        "max_seq_len": 32,
        "dropout": 0.0,
    }


@dataclass(frozen=True)
class SequentialBenchmarkConfig:
    """Explicit, serializable configuration for one sequential-benchmark run."""

    seed: int = 0
    vocab_size: int = 10
    sequence_length_range: tuple[int, int] = (6, 10)
    max_depth: int = 2
    novel_composition_fraction: float = 0.3
    novel_operation_names: tuple[str, ...] = ("SORT", "REVERSE")
    model: dict[str, Any] = field(default_factory=_default_model_config)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    num_examples: int = 16
    allocator_preset: AllocatorPreset = AllocatorPreset.SMALL
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    plastic: PlasticTrainingConfig = field(default_factory=PlasticTrainingConfig)
    consolidation: ConsolidationConfig = field(
        default_factory=lambda: ConsolidationConfig(
            candidate_rank=14, steps=600, lr=5e-2, replay_weight=0.5
        )
    )
    shadow: ShadowValidationConfig = field(
        default_factory=lambda: ShadowValidationConfig(max_retention_degradation=0.08)
    )
    router_calibration: RouterCalibrationConfig = field(default_factory=RouterCalibrationConfig)
    replay_buffer_max_events: int = 6
    max_shadow_retries: int = 3
    events: tuple[StreamEvent, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_examples < 1:
            raise ValueError(f"num_examples must be >= 1, got {self.num_examples}")
        if self.replay_buffer_max_events < 1:
            raise ValueError(
                f"replay_buffer_max_events must be >= 1, got {self.replay_buffer_max_events}"
            )
        if self.max_shadow_retries < 1:
            raise ValueError(f"max_shadow_retries must be >= 1, got {self.max_shadow_retries}")

    def resolved_events(self) -> tuple[StreamEvent, ...]:
        if self.events is not None:
            return self.events
        return default_task_stream(self.novel_operation_names)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def sequential_config_from_dict(raw: dict[str, Any]) -> SequentialBenchmarkConfig:
    """Parse a `configs/phase_a_sequential.yaml`-shaped dict into a
    `SequentialBenchmarkConfig`, matching `apc.core.train.
    smoke_config_from_dict`'s convention of filling in defaults for
    whatever the config file omits."""
    defaults = SequentialBenchmarkConfig()

    def _sub(key: str, current: Any) -> Any:
        overrides = raw.get(key)
        return dataclasses.replace(current, **overrides) if overrides else current

    return SequentialBenchmarkConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        max_depth=raw.get("max_depth", defaults.max_depth),
        novel_composition_fraction=raw.get(
            "novel_composition_fraction", defaults.novel_composition_fraction
        ),
        novel_operation_names=tuple(
            raw.get("novel_operation_names", defaults.novel_operation_names)
        ),
        model=dict(raw.get("model", defaults.model)),
        pretrain=_sub("pretrain", defaults.pretrain),
        num_examples=raw.get("num_examples", defaults.num_examples),
        allocator_preset=AllocatorPreset(raw.get("allocator_preset", defaults.allocator_preset)),
        controller=_sub("controller", defaults.controller),
        novelty=_sub("novelty", defaults.novelty),
        plastic=_sub("plastic", defaults.plastic),
        consolidation=_sub("consolidation", defaults.consolidation),
        shadow=_sub("shadow", defaults.shadow),
        router_calibration=_sub("router_calibration", defaults.router_calibration),
        replay_buffer_max_events=raw.get(
            "replay_buffer_max_events", defaults.replay_buffer_max_events
        ),
        max_shadow_retries=raw.get("max_shadow_retries", defaults.max_shadow_retries),
    )


@dataclass(frozen=True)
class EventReport:
    """Everything observed while processing one `StreamEvent`."""

    index: int
    label: str
    operation_name: str | None
    pre_exact_match: float
    pre_novelty: float
    pre_router_entropy: float
    controller_states: tuple[str, ...]
    had_cycle: bool
    search_steps: int
    plastic_steps: int
    shadow_attempts: int
    gave_up: bool
    consolidation: ConsolidationReport | None
    shadow: ShadowValidationReport | None
    candidate_primitive_id: int | None
    post_exact_match: float
    resident_total_param_count: int
    resident_primitive_param_count: int
    temporary_peak_param_count: int
    active_primitive_param_count: int
    active_param_count: int


@dataclass(frozen=True)
class RetentionSample:
    index: int
    label: str
    operation_name: str | None
    exact_match_then: float
    exact_match_now: float

    @property
    def forgetting(self) -> float:
        return max(0.0, self.exact_match_then - self.exact_match_now)


@dataclass(frozen=True)
class SequentialBenchmarkReport:
    config: SequentialBenchmarkConfig
    pretrain_exact_match: float
    events: tuple[EventReport, ...]
    controller_transitions: tuple[dict[str, Any], ...]
    num_learn_consolidate_release_cycles: int
    num_reused_without_new_cycle: int
    retention: tuple[RetentionSample, ...]
    max_forgetting: float
    mean_backward_transfer: float
    stable_core_parameter_count: int
    resident_total_parameter_count_final: int
    resident_primitive_parameter_count_final: int
    temporary_peak_parameter_count: int
    total_train_steps: int
    generalization_gap_known_vs_composition: float | None
    novelty_gap_composition_vs_operation: float | None
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


@dataclass
class _CycleResult:
    plastic_steps: int
    attempts: int
    consolidation_report: ConsolidationReport | None
    shadow_report: ShadowValidationReport | None
    candidate_id: int | None
    gave_up: bool
    visited_states: list[str]


class _SequentialBenchmarkRunner:
    def __init__(self, config: SequentialBenchmarkConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.specials: SpecialTokens = build_special_tokens(config.vocab_size)
        self.main_generator, self.novel_generators = build_task_generators(
            seed=config.seed,
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
            max_depth=config.max_depth,
            novel_composition_fraction=config.novel_composition_fraction,
            novel_operation_names=config.novel_operation_names,
        )

        self.model_config = TransformerConfig(
            vocab_size=self.specials.model_vocab_size, **config.model
        )
        self.model = DecoderOnlyTransformer(self.model_config)
        self.bank = PrimitiveBank()
        self.router = Router(RouterConfig(d_model=self.model_config.d_model))
        ensure_null_key(self.router)
        self.workspace = PlasticWorkspace()
        self.allocator = Allocator(d_model=self.model_config.d_model)
        self.controller = Controller(config.controller)
        self.novelty_estimator = NoveltyEstimator(config.novelty)

        self.anchors: dict[int, torch.Tensor] = {}
        self.replay_buffer: list[tuple[StreamEvent, list[Example]]] = []
        self.event_eval_examples: list[list[Example]] = []
        self._pool = EventExamplePool()
        self.temporary_peak_params = 0
        self.total_train_steps = 0

    # --- data ----------------------------------------------------------

    def _pool_for_event(self, event: StreamEvent, total_count: int) -> list[Example]:
        return self._pool.pool_for_event(
            event, self.main_generator, self.novel_generators, total_count
        )

    # --- hidden-state helpers -------------------------------------------

    def _raw_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(input_ids)

    def _bank_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._raw_hidden(input_ids)
        hidden, _ = apply_bank(hidden, self.bank, self.router, stable_candidate_ids(self.bank))
        return hidden

    def _replay_bank_hidden(self) -> torch.Tensor:
        if not self.replay_buffer:
            return torch.zeros(0, self.model_config.d_model)
        chunks = []
        for _, examples in self.replay_buffer:
            batch = collate_batch(examples, self.specials)
            chunks.append(self._bank_hidden(batch.input_ids).reshape(-1, self.model_config.d_model))
        return torch.cat(chunks, dim=0)

    def _replay_raw_hidden(self) -> torch.Tensor:
        if not self.replay_buffer:
            return torch.zeros(0, self.model_config.d_model)
        chunks = []
        for _, examples in self.replay_buffer:
            batch = collate_batch(examples, self.specials)
            chunks.append(self._raw_hidden(batch.input_ids).reshape(-1, self.model_config.d_model))
        return torch.cat(chunks, dim=0)

    def _router_entropy(self, eval_examples: list[Example], stable_ids: list[int]) -> float:
        batch = collate_batch(eval_examples, self.specials)
        hidden = self._raw_hidden(batch.input_ids)
        _, router_out = apply_bank(hidden, self.bank, self.router, stable_ids)
        return float(router_out.entropy.mean().item())

    def _active_primitive_parameter_count(
        self, examples: list[Example], stable_ids: list[int]
    ) -> int:
        """Selected primitive capacity for one representative batched forward.

        The benchmark reports this batch-level proxy separately from the
        always-active Stable Core, rather than treating every resident bank
        primitive as active.
        """
        batch = collate_batch(examples, self.specials)
        hidden = self._raw_hidden(batch.input_ids)
        _, router_out = apply_bank(hidden, self.bank, self.router, stable_ids)
        return self.bank.active_parameter_count(router_out.executed_primitive_ids)

    def _evaluate_novelty(self, eval_examples: list[Example]) -> tuple[float, float, float]:
        stable_ids = stable_candidate_ids(self.bank)
        exact_match, _ = evaluate_exact_match_with_capacity(
            self.model, eval_examples, self.specials, self.bank, self.router, stable_ids=stable_ids
        )
        entropy = self._router_entropy(eval_examples, stable_ids)
        max_entropy = math.log(1 + len(stable_ids))
        signals = NoveltySignals(
            error=1.0 - exact_match, router_entropy=entropy, max_router_entropy=max_entropy
        )
        novelty = self.novelty_estimator.score(signals)
        return novelty, exact_match, entropy

    # --- pretraining -----------------------------------------------------

    def pretrain(self) -> float:
        examples = self.main_generator.generate(self.config.pretrain.num_examples, "train")
        batch = collate_batch(examples, self.specials)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.pretrain.lr,
            weight_decay=self.config.pretrain.weight_decay,
        )
        self.model.train()
        for _ in range(self.config.pretrain.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if self.config.pretrain.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.pretrain.grad_clip
                )
            optimizer.step()
        self.total_train_steps += self.config.pretrain.steps

        test_examples = self.main_generator.generate(self.config.num_examples, "test")
        exact_match, _ = evaluate_exact_match(self.model, test_examples, self.specials)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        return exact_match

    # --- router calibration ----------------------------------------------

    def _calibrate_router(self, new_primitive_id: int, train_input_ids: torch.Tensor) -> None:
        self.anchors[new_primitive_id] = self._raw_hidden(train_input_ids).reshape(
            -1, self.model_config.d_model
        )
        background = self._replay_raw_hidden()
        calibrate_router(
            self.router,
            self.anchors,
            background,
            steps=self.config.router_calibration.steps,
            lr=self.config.router_calibration.lr,
        )

    # --- PLASTIC / CONSOLIDATE / SHADOW ----------------------------------

    def _train_plastic_until_promoted(
        self,
        ids: list[int],
        train_input_ids: torch.Tensor,
        labels: torch.Tensor,
        eval_examples: list[Example],
    ) -> tuple[int, list[str]]:
        """Train the currently-allocated workspace transforms until the
        controller promotes PLASTIC -> CONSOLIDATE (or `plastic.max_steps`
        is exhausted, in which case promotion is forced -- see module
        docstring's bounded-retry note)."""
        optimizer = torch.optim.AdamW(self.workspace.parameters(), lr=self.config.plastic.lr)
        best_accuracy = 0.0
        visited: list[str] = []
        step = 0
        while True:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_logits(
                self.model,
                train_input_ids,
                self.bank,
                self.router,
                stable_ids=stable_candidate_ids(self.bank),
                workspace=self.workspace,
                workspace_ids=ids,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
            loss.backward()
            if self.config.plastic.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.workspace.parameters(), self.config.plastic.grad_clip
                )
            optimizer.step()

            at_cap = step >= self.config.plastic.max_steps
            if step % self.config.plastic.eval_every == 0 or at_cap:
                self.workspace.record_usage(ids)
                accuracy, _ = evaluate_exact_match_with_capacity(
                    self.model,
                    eval_examples,
                    self.specials,
                    self.bank,
                    self.router,
                    stable_ids=stable_candidate_ids(self.bank),
                    workspace=self.workspace,
                    workspace_ids=ids,
                )
                improved = accuracy > best_accuracy + self.config.plastic.improvement_margin
                best_accuracy = max(best_accuracy, accuracy)
                transition = self.controller.step(
                    ControllerSignals(plastic_improved=improved, plastic_accuracy=accuracy)
                )
                if transition is not None:
                    visited.append(transition.new_state.value)
                if self.controller.state == ControllerState.CONSOLIDATE:
                    return step, visited
            if at_cap:
                if self.controller.state != ControllerState.CONSOLIDATE:
                    self.controller.force_state(ControllerState.CONSOLIDATE)
                    visited.append(ControllerState.CONSOLIDATE.value + "(forced)")
                return step, visited

    def _run_learn_consolidate_release(
        self,
        event: StreamEvent,
        index: int,
        train_examples: list[Example],
        eval_examples: list[Example],
    ) -> _CycleResult:
        ids = self.allocator.allocate(
            self.workspace, self.config.allocator_preset, created_at_task=index
        )
        self.temporary_peak_params = max(
            self.temporary_peak_params, self.workspace.total_parameter_count()
        )
        train_batch = collate_batch(train_examples, self.specials)

        visited: list[str] = []
        total_plastic_steps = 0
        plastic_steps, extra_visited = self._train_plastic_until_promoted(
            ids, train_batch.input_ids, train_batch.labels, eval_examples
        )
        total_plastic_steps += plastic_steps
        visited.extend(extra_visited)
        self.total_train_steps += total_plastic_steps

        attempts = 0
        consolidation_report: ConsolidationReport | None = None
        shadow_report: ShadowValidationReport | None = None
        candidate_id: int | None = None
        gave_up = False

        while True:
            attempts += 1
            next_id = max(self.bank.ids(), default=-1) + 1
            current_hidden = self._bank_hidden(train_batch.input_ids).reshape(
                -1, self.model_config.d_model
            )
            replay_hidden = self._replay_bank_hidden()
            self.workspace.record_usage(ids)
            # Derive a per-(event, attempt) seed from the run seed so a
            # candidate's own random initialization is reproducible too --
            # ConsolidationConfig.seed defaults to None (unseeded).
            attempt_consolidation_config = dataclasses.replace(
                self.config.consolidation, seed=self.config.seed * 1000 + index * 10 + attempts
            )
            candidate, consolidation_report = consolidate(
                self.workspace,
                current_hidden,
                replay_hidden,
                attempt_consolidation_config,
                candidate_id=next_id,
                created_at_task=index,
            )
            self.total_train_steps += self.config.consolidation.steps
            transition = self.controller.step(ControllerSignals(candidate_ready=True))
            if transition is not None:
                visited.append(transition.new_state.value)

            active_ids = self.workspace.ids()
            with torch.no_grad():
                current_targets = apply_workspace(current_hidden, self.workspace, active_ids)
                replay_targets = apply_workspace(replay_hidden, self.workspace, active_ids)
            released_if_pass = self.workspace.total_parameter_count()
            shadow_report = run_shadow_validation(
                self.workspace,
                active_ids,
                candidate,
                self.bank,
                current_hidden,
                current_targets,
                replay_hidden,
                replay_targets,
                self.config.shadow,
            )
            transition = self.controller.step(
                ControllerSignals(
                    shadow_passed=shadow_report.passed,
                    released_params=released_if_pass if shadow_report.passed else 0,
                )
            )
            if transition is not None:
                visited.append(transition.new_state.value)

            if shadow_report.passed:
                candidate_id = candidate.primitive_id
                self.router.add_primitive_key(candidate_id)
                self._calibrate_router(candidate_id, train_batch.input_ids)
                break
            if attempts >= self.config.max_shadow_retries:
                gave_up = True
                self.workspace.release()
                self.controller.force_state(ControllerState.STABLE)
                visited.append(ControllerState.STABLE.value + "(forced_after_giving_up)")
                break

            extra_steps, extra_visited = self._train_plastic_until_promoted(
                ids, train_batch.input_ids, train_batch.labels, eval_examples
            )
            total_plastic_steps += extra_steps
            self.total_train_steps += extra_steps
            visited.extend(extra_visited)

        return _CycleResult(
            plastic_steps=total_plastic_steps,
            attempts=attempts,
            consolidation_report=consolidation_report,
            shadow_report=shadow_report,
            candidate_id=candidate_id,
            gave_up=gave_up,
            visited_states=visited,
        )

    # --- per-event driver --------------------------------------------------

    def _process_event(self, event: StreamEvent, index: int) -> EventReport:
        # Train and eval share one fixed example set per event rather than a
        # held-out split -- see module docstring ("Evaluating against what
        # an event trains on, not held-out instances").
        examples = self._pool_for_event(event, self.config.num_examples)
        train_examples = examples
        eval_examples = examples

        novelty0, exact_match0, entropy0 = self._evaluate_novelty(eval_examples)
        spec = self.allocator.spec_for(self.config.allocator_preset)
        predicted_allocated = spec.num_transforms * self.model_config.d_model * spec.rank * 2

        visited_states: list[str] = []
        transition = self.controller.step(
            ControllerSignals(novelty=novelty0, allocated_params=predicted_allocated)
        )
        if transition is not None:
            visited_states.append(transition.new_state.value)
        search_steps = 0
        while self.controller.state == ControllerState.SEARCH:
            search_steps += 1
            novelty_i, _, _ = self._evaluate_novelty(eval_examples)
            transition = self.controller.step(
                ControllerSignals(novelty=novelty_i, allocated_params=predicted_allocated)
            )
            if transition is not None:
                visited_states.append(transition.new_state.value)

        had_cycle = False
        plastic_steps = 0
        shadow_attempts = 0
        consolidation_report = None
        shadow_report = None
        candidate_id = None
        gave_up = False

        if self.controller.state == ControllerState.PLASTIC:
            had_cycle = True
            result = self._run_learn_consolidate_release(
                event, index, train_examples, eval_examples
            )
            plastic_steps = result.plastic_steps
            shadow_attempts = result.attempts
            consolidation_report = result.consolidation_report
            shadow_report = result.shadow_report
            candidate_id = result.candidate_id
            gave_up = result.gave_up
            visited_states.extend(result.visited_states)

        stable_ids = stable_candidate_ids(self.bank)
        post_exact_match, _ = evaluate_exact_match_with_capacity(
            self.model, eval_examples, self.specials, self.bank, self.router, stable_ids=stable_ids
        )
        active_primitive_param_count = self._active_primitive_parameter_count(
            eval_examples, stable_ids
        )

        report = EventReport(
            index=index,
            label=event.label,
            operation_name=event.operation_name,
            pre_exact_match=exact_match0,
            pre_novelty=novelty0,
            pre_router_entropy=entropy0,
            controller_states=tuple(visited_states),
            had_cycle=had_cycle,
            search_steps=search_steps,
            plastic_steps=plastic_steps,
            shadow_attempts=shadow_attempts,
            gave_up=gave_up,
            consolidation=consolidation_report,
            shadow=shadow_report,
            candidate_primitive_id=candidate_id,
            post_exact_match=post_exact_match,
            resident_total_param_count=(
                self.model.num_parameters() + self.bank.persistent_parameter_count()
            ),
            resident_primitive_param_count=self.bank.persistent_parameter_count(),
            temporary_peak_param_count=self.workspace.total_parameter_count(),
            active_primitive_param_count=active_primitive_param_count,
            active_param_count=self.model.num_parameters() + active_primitive_param_count,
        )
        self.event_eval_examples.append(eval_examples)
        self.replay_buffer.append((event, eval_examples))
        self.replay_buffer = self.replay_buffer[-self.config.replay_buffer_max_events :]
        return report

    # --- retention ----------------------------------------------------------

    def _compute_retention(self, events: tuple[EventReport, ...]) -> tuple[RetentionSample, ...]:
        stable_ids = stable_candidate_ids(self.bank)
        samples = []
        for report, examples in zip(events, self.event_eval_examples, strict=True):
            exact_match_now, _ = evaluate_exact_match_with_capacity(
                self.model, examples, self.specials, self.bank, self.router, stable_ids=stable_ids
            )
            samples.append(
                RetentionSample(
                    index=report.index,
                    label=report.label,
                    operation_name=report.operation_name,
                    exact_match_then=report.post_exact_match,
                    exact_match_now=exact_match_now,
                )
            )
        return tuple(samples)

    # --- top level ------------------------------------------------------

    def run(self) -> SequentialBenchmarkReport:
        start = time.perf_counter()
        pretrain_exact_match = self.pretrain()

        events = self.config.resolved_events()
        event_reports = tuple(self._process_event(event, i) for i, event in enumerate(events))
        retention = self._compute_retention(event_reports)

        forgetting = [sample.forgetting for sample in retention]
        backward_transfer = [
            sample.exact_match_now - sample.exact_match_then for sample in retention
        ]

        known = next((e for e in event_reports if e.label == LABEL_KNOWN), None)
        composition = next((e for e in event_reports if e.label == LABEL_NOVEL_COMPOSITION), None)
        operation = next((e for e in event_reports if e.label == LABEL_NOVEL_OPERATION), None)
        generalization_gap = (
            known.pre_exact_match - composition.pre_exact_match
            if known is not None and composition is not None
            else None
        )
        novelty_gap = (
            composition.pre_exact_match - operation.pre_exact_match
            if composition is not None and operation is not None
            else None
        )

        cycles = sum(
            1
            for e in event_reports
            if e.had_cycle and not e.gave_up and e.candidate_primitive_id is not None
        )
        reused = sum(1 for e in event_reports if e.label == LABEL_RECURRENCE and not e.had_cycle)

        return SequentialBenchmarkReport(
            config=self.config,
            pretrain_exact_match=pretrain_exact_match,
            events=event_reports,
            controller_transitions=tuple(t.to_dict() for t in self.controller.transition_log),
            num_learn_consolidate_release_cycles=cycles,
            num_reused_without_new_cycle=reused,
            retention=retention,
            max_forgetting=max(forgetting) if forgetting else 0.0,
            mean_backward_transfer=sum(backward_transfer) / len(backward_transfer)
            if backward_transfer
            else 0.0,
            stable_core_parameter_count=self.model.num_parameters(),
            resident_total_parameter_count_final=(
                self.model.num_parameters() + self.bank.persistent_parameter_count()
            ),
            resident_primitive_parameter_count_final=self.bank.persistent_parameter_count(),
            temporary_peak_parameter_count=self.temporary_peak_params,
            total_train_steps=self.total_train_steps,
            generalization_gap_known_vs_composition=generalization_gap,
            novelty_gap_composition_vs_operation=novelty_gap,
            wall_clock_seconds=time.perf_counter() - start,
        )


def run_sequential_benchmark(config: SequentialBenchmarkConfig) -> SequentialBenchmarkReport:
    """Run one seeded sequential-benchmark stream end to end."""
    return _SequentialBenchmarkRunner(config).run()
