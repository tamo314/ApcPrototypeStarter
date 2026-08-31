"""Baselines B0-B4 for the Phase A sequential benchmark (Task 013).

`docs/EXPERIMENT_PLAN.md` section 5 defines five configurations to compare
under the same task stream:

    B0 Fixed dense    -- a capacity-matched Transformer, no primitives.
    B1 Fixed sparse   -- stable core + fixed primitive bank + router, no expansion.
    B2 Grow-only      -- dynamic plastic capacity becomes permanent; no consolidation/pruning.
    B3 Grow + replay  -- dynamic expansion with replay but no primitive compression.
    B4 APC            -- the full finite-state loop.

B4 is `apc.evaluation.sequential_benchmark.run_sequential_benchmark`,
already built in Task 012. This module adds B0-B3, sharing as much as
Task 012's own machinery as makes sense for each baseline's definition:

- **Shared data.** `BaselineConfig` wraps the exact same
  `SequentialBenchmarkConfig` B4 runs under (seed, vocab, model size,
  pretrain budget, per-event example count, task stream). Every baseline
  builds its `TaskGenerator`s and pools per-event examples via
  `apc.evaluation.stream.build_task_generators` /
  `EventExamplePool` -- the same functions B4 uses -- so B0-B4 draw
  identical data given the same config and event-processing order
  (`TaskGenerator` is deterministic by seed; see that module's docstring).
- **Shared evaluation.** B0 (no bank at all) evaluates with
  `apc.core.generation.evaluate_exact_match`; B1 (uses a real
  `PrimitiveBank`/`Router`) evaluates with
  `apc.core.execution.evaluate_exact_match_with_capacity`, the same
  function B4 uses. B2/B3 need their own small greedy-decode loop (see
  below) since their execution path does not fit either abstraction.
- **Shared budgets.** Every baseline's per-event training loop stops via
  `apc.evaluation.stream.train_until_plateau`, reusing
  `SequentialBenchmarkConfig.plastic`'s step budget and
  `SequentialBenchmarkConfig.controller`'s plateau-patience/min-accuracy
  fields -- the same numbers that gate B4's own PLASTIC-to-CONSOLIDATE
  promotion (`apc.meta.controller.Controller._decide_plastic`), so a
  baseline's per-event compute is capped and stopped on the same
  criterion, not an arbitrarily different one.

Design choices specific to this module (recorded here rather than hidden,
per AGENTS.md workflow):

- **B1's fixed bank is populated once, unconditionally, before the
  stream.** "Stable core + fixed primitive bank + router, no expansion"
  does not specify where the fixed bank's content comes from. An empty
  fixed bank would make B1 behave identically to B0 (routing over zero
  real candidates is a no-op, see `apc.core.execution.apply_bank`), which
  is not an informative baseline. Instead, `B1Runner` runs exactly one
  learn -> consolidate -> shadow cycle on a held-out slice of the
  known-operation ("train" split) data immediately after pretraining --
  reusing `apc.consolidation.distill.consolidate`,
  `apc.consolidation.shadow.run_shadow_validation`, and
  `apc.evaluation.stream.calibrate_router`, the exact same functions B4's
  learn/consolidate/release cycle uses -- then freezes bank and router for
  the rest of the run: the controller/novelty machinery is never invoked
  again regardless of event label. If that one shadow validation fails,
  B1 proceeds with whatever ended up in the bank (possibly still empty);
  this is reported honestly rather than retried, since B1's whole point is
  a system that cannot adapt further.
- **B2/B3 do not use a `PrimitiveBank`/`Router` at all.** A "grow-only"
  baseline in the continual-learning literature (e.g. Progressive Neural
  Networks) adds a new, *unconditionally active* block of capacity per
  task -- there is no sparse gating decision to make, since gating is
  exactly the mechanism APC's router/consolidation exists to make
  learnable at small bank sizes (see `apc.core.execution`'s module
  docstring on the null-routing-candidate problem). Building a
  discriminative router for capacity that is never compressed or
  functionally consolidated would reintroduce that same
  hard-to-discriminate-at-small-scale problem into a baseline whose
  purpose is to be architecturally simple. So B2/B3 keep a plain growing
  `list[Primitive]` (`self.grown`), summed unconditionally into the hidden
  state every forward pass via `apc.core.execution.sum_primitive_deltas`
  (the same delta-combination primitive `apply_bank`/`apply_workspace`
  use, just without router selection) -- this also makes "active
  parameters per inference step equals persistent parameters" an honest,
  reportable property that contrasts directly with B4's sparse routing.
- **B2/B3 grow for every event, unconditionally.** Milestone A9's
  controller-driven SEARCH/PLASTIC decision is APC-specific machinery;
  "grow-only" baselines in the literature grow for every new task
  precisely because they have no novelty-detection mechanism to decide
  otherwise. Growing unconditionally is what makes the "bounded persistent
  growth" comparison (Experiment Plan H4) meaningful: B2/B3's persistent
  parameter count should visibly outgrow B4's over the same stream.
- **B3's replay weight is a new, baseline-only knob
  (`BaselineConfig.replay_weight`), not a reuse of
  `ConsolidationConfig.replay_weight`.** The latter scales a
  consolidation-distillation loss (candidate learning to reproduce a
  teacher's delta); B3 has no distillation step at all -- it adds a
  cross-entropy replay loss straight onto the raw task loss its newly
  grown transforms are trained with. Reusing the same config field for a
  different loss would be a silent semantic overload, so a small dedicated
  field is added instead.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from apc.consolidation.distill import consolidate
from apc.consolidation.shadow import run_shadow_validation
from apc.core.data import IGNORE_INDEX, Batch, collate_batch
from apc.core.execution import (
    apply_bank,
    apply_workspace,
    ensure_null_key,
    evaluate_exact_match_with_capacity,
    forward_logits,
    stable_candidate_ids,
    sum_primitive_deltas,
)
from apc.core.generation import evaluate_exact_match
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example
from apc.evaluation.sequential_benchmark import (
    SequentialBenchmarkConfig,
    SequentialBenchmarkReport,
    sequential_config_from_dict,
)
from apc.evaluation.sequential_benchmark import (
    run_sequential_benchmark as _run_apc,
)
from apc.evaluation.stream import (
    EventExamplePool,
    StreamEvent,
    build_task_generators,
    calibrate_router,
    train_until_plateau,
)
from apc.plastic.allocator import Allocator
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

BASELINE_NAMES: tuple[str, ...] = ("B0", "B1", "B2", "B3", "B4")


@dataclass(frozen=True)
class BaselineConfig:
    """Everything B0-B3 need, sharing every data/model/budget knob with B4
    via one embedded `SequentialBenchmarkConfig` (see module docstring)."""

    sequential: SequentialBenchmarkConfig = field(default_factory=SequentialBenchmarkConfig)
    replay_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.replay_weight < 0:
            raise ValueError(f"replay_weight must be >= 0, got {self.replay_weight}")

    def to_dict(self) -> dict[str, Any]:
        return {"sequential": self.sequential.to_dict(), "replay_weight": self.replay_weight}


def baseline_config_from_dict(raw: dict[str, Any]) -> BaselineConfig:
    """Parse the same config-file shape `sequential_config_from_dict` reads,
    plus an optional top-level `baseline_replay_weight` key (B3 only)."""
    sequential = sequential_config_from_dict(raw)
    return BaselineConfig(
        sequential=sequential,
        replay_weight=raw.get("baseline_replay_weight", BaselineConfig().replay_weight),
    )


@dataclass(frozen=True)
class BaselineEventReport:
    """Everything observed while a baseline processes one `StreamEvent`."""

    index: int
    label: str
    operation_name: str | None
    pre_exact_match: float
    post_exact_match: float
    train_steps: int
    resident_total_param_count: int
    resident_primitive_param_count: int
    active_primitive_param_count: int
    active_param_count: int


@dataclass(frozen=True)
class BaselineRetentionSample:
    index: int
    label: str
    operation_name: str | None
    exact_match_then: float
    exact_match_now: float

    @property
    def forgetting(self) -> float:
        return max(0.0, self.exact_match_then - self.exact_match_now)


@dataclass(frozen=True)
class BaselineReport:
    baseline: str
    config: BaselineConfig
    pretrain_exact_match: float
    events: tuple[BaselineEventReport, ...]
    retention: tuple[BaselineRetentionSample, ...]
    max_forgetting: float
    mean_backward_transfer: float
    stable_core_parameter_count: int
    resident_total_parameter_count_final: int
    resident_primitive_parameter_count_final: int
    total_train_steps: int
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


class _BaseRunner:
    """Shared pretrain/event-loop/retention driver; subclasses supply how
    capacity is represented and how a forward pass is evaluated."""

    name: str
    freeze_core_after_pretrain: bool = True

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        seq = config.sequential
        set_seed(seq.seed)
        self.specials = build_special_tokens(seq.vocab_size)
        self.main_generator, self.novel_generators = build_task_generators(
            seed=seq.seed,
            vocab_size=seq.vocab_size,
            sequence_length_range=seq.sequence_length_range,
            max_depth=seq.max_depth,
            novel_composition_fraction=seq.novel_composition_fraction,
            novel_operation_names=seq.novel_operation_names,
        )
        self.pool = EventExamplePool()
        self.model_config = TransformerConfig(
            vocab_size=self.specials.model_vocab_size, **seq.model
        )
        self.model = DecoderOnlyTransformer(self.model_config)
        self.event_eval_examples: list[list[Example]] = []
        self.total_train_steps = 0

        # Re-anchor the shared global RNG stream now that setup-phase
        # construction (model, ...) has run -- see ADR-0016. Subclasses
        # that construct additional RNG-consuming state after calling
        # `super().__init__` (e.g. `B1Runner`'s bank/router) must reseed
        # again at the end of their own `__init__` for the same reason.
        set_seed(seq.seed)

    def pretrain(self) -> float:
        seq = self.config.sequential
        examples = self.main_generator.generate(seq.pretrain.num_examples, "train")
        batch = collate_batch(examples, self.specials)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=seq.pretrain.lr,
            weight_decay=seq.pretrain.weight_decay,
        )
        self.model.train()
        for _ in range(seq.pretrain.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if seq.pretrain.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), seq.pretrain.grad_clip)
            optimizer.step()
        self.total_train_steps += seq.pretrain.steps

        test_examples = self.main_generator.generate(seq.num_examples, "test")
        exact_match, _ = evaluate_exact_match(self.model, test_examples, self.specials)

        if self.freeze_core_after_pretrain:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        return exact_match

    def after_pretrain(self) -> None:
        """Hook for a baseline that needs one-time setup after pretraining
        but before the stream starts (B1's seed-and-freeze cycle)."""

    def pool_for_event(self, event: StreamEvent) -> list[Example]:
        return self.pool.pool_for_event(
            event, self.main_generator, self.novel_generators, self.config.sequential.num_examples
        )

    def evaluate(self, examples: list[Example]) -> float:
        raise NotImplementedError

    def process_event(self, event: StreamEvent, index: int) -> BaselineEventReport:
        raise NotImplementedError

    def resident_primitive_param_count(self) -> int:
        raise NotImplementedError

    def resident_total_param_count(self) -> int:
        return self.model.num_parameters() + self.resident_primitive_param_count()

    def active_primitive_param_count(self, examples: list[Example]) -> int:
        raise NotImplementedError

    def active_param_count(self, examples: list[Example]) -> int:
        """Stable-Core plus actually executed primitive capacity."""
        return self.model.num_parameters() + self.active_primitive_param_count(examples)

    def compute_retention(
        self, events: tuple[BaselineEventReport, ...]
    ) -> tuple[BaselineRetentionSample, ...]:
        samples = []
        for report, examples in zip(events, self.event_eval_examples, strict=True):
            exact_match_now = self.evaluate(examples)
            samples.append(
                BaselineRetentionSample(
                    index=report.index,
                    label=report.label,
                    operation_name=report.operation_name,
                    exact_match_then=report.post_exact_match,
                    exact_match_now=exact_match_now,
                )
            )
        return tuple(samples)

    def run(self) -> BaselineReport:
        start = time.perf_counter()
        pretrain_exact_match = self.pretrain()
        self.after_pretrain()

        events = self.config.sequential.resolved_events()
        event_reports = tuple(self.process_event(event, i) for i, event in enumerate(events))
        retention = self.compute_retention(event_reports)

        forgetting = [s.forgetting for s in retention]
        backward_transfer = [s.exact_match_now - s.exact_match_then for s in retention]

        return BaselineReport(
            baseline=self.name,
            config=self.config,
            pretrain_exact_match=pretrain_exact_match,
            events=event_reports,
            retention=retention,
            max_forgetting=max(forgetting) if forgetting else 0.0,
            mean_backward_transfer=(sum(backward_transfer) / len(backward_transfer))
            if backward_transfer
            else 0.0,
            stable_core_parameter_count=self.model.num_parameters(),
            resident_total_parameter_count_final=self.resident_total_param_count(),
            resident_primitive_parameter_count_final=self.resident_primitive_param_count(),
            total_train_steps=self.total_train_steps,
            wall_clock_seconds=time.perf_counter() - start,
        )

    def _plateau(
        self, step_fn: Callable[[], None], eval_fn: Callable[[], float]
    ) -> tuple[int, float]:
        seq = self.config.sequential
        return train_until_plateau(
            step_fn,
            eval_fn,
            max_steps=seq.plastic.max_steps,
            eval_every=seq.plastic.eval_every,
            improvement_margin=seq.plastic.improvement_margin,
            plateau_patience=seq.controller.plastic_plateau_patience,
            min_accuracy=seq.controller.plastic_min_accuracy,
        )


class B0Runner(_BaseRunner):
    """Fixed dense: no primitives at all, continual fine-tuning of the
    same dense core for every event (the core is never frozen)."""

    name = "B0"
    freeze_core_after_pretrain = False

    def evaluate(self, examples: list[Example]) -> float:
        exact_match, _ = evaluate_exact_match(self.model, examples, self.specials)
        return exact_match

    def resident_primitive_param_count(self) -> int:
        return 0

    def active_primitive_param_count(self, examples: list[Example]) -> int:
        return 0

    def process_event(self, event: StreamEvent, index: int) -> BaselineEventReport:
        examples = self.pool_for_event(event)
        pre = self.evaluate(examples)
        batch = collate_batch(examples, self.specials)
        plastic = self.config.sequential.plastic
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=plastic.lr)

        def step_fn() -> None:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if plastic.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), plastic.grad_clip)
            optimizer.step()

        def eval_fn() -> float:
            return self.evaluate(examples)

        steps, post = self._plateau(step_fn, eval_fn)
        self.total_train_steps += steps
        self.event_eval_examples.append(examples)
        active_primitive_param_count = self.active_primitive_param_count(examples)
        return BaselineEventReport(
            index=index,
            label=event.label,
            operation_name=event.operation_name,
            pre_exact_match=pre,
            post_exact_match=post,
            train_steps=steps,
            resident_total_param_count=self.resident_total_param_count(),
            resident_primitive_param_count=self.resident_primitive_param_count(),
            active_primitive_param_count=active_primitive_param_count,
            active_param_count=self.model.num_parameters() + active_primitive_param_count,
        )


class B1Runner(_BaseRunner):
    """Fixed sparse: stable core + a bank/router populated by exactly one
    learn -> consolidate -> shadow cycle before the stream, then frozen."""

    name = "B1"

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__(config)
        self.bank = PrimitiveBank()
        self.router = Router(RouterConfig(d_model=self.model_config.d_model))
        ensure_null_key(self.router)
        set_seed(config.sequential.seed)  # re-anchor after this class's own construction

    def evaluate(self, examples: list[Example]) -> float:
        stable_ids = stable_candidate_ids(self.bank)
        exact_match, _ = evaluate_exact_match_with_capacity(
            self.model, examples, self.specials, self.bank, self.router, stable_ids=stable_ids
        )
        return exact_match

    def resident_primitive_param_count(self) -> int:
        return self.bank.persistent_parameter_count()

    def active_primitive_param_count(self, examples: list[Example]) -> int:
        stable_ids = stable_candidate_ids(self.bank)
        batch = collate_batch(examples, self.specials)
        with torch.no_grad():
            hidden = self.model.encode(batch.input_ids)
        _, router_out = apply_bank(hidden, self.bank, self.router, stable_ids)
        return self.bank.active_parameter_count(router_out.executed_primitive_ids)

    def after_pretrain(self) -> None:
        seq = self.config.sequential
        seed_examples = self.main_generator.generate(seq.pretrain.num_examples, "train")
        batch = collate_batch(seed_examples, self.specials)

        allocator = Allocator(d_model=self.model_config.d_model)
        workspace = PlasticWorkspace()
        ids = allocator.allocate(workspace, seq.allocator_preset, created_at_task=-1)
        optimizer = torch.optim.AdamW(workspace.parameters(), lr=seq.plastic.lr)

        def step_fn() -> None:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_logits(
                self.model,
                batch.input_ids,
                self.bank,
                self.router,
                workspace=workspace,
                workspace_ids=ids,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if seq.plastic.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(workspace.parameters(), seq.plastic.grad_clip)
            optimizer.step()

        def eval_fn() -> float:
            workspace.record_usage(ids)
            accuracy, _ = evaluate_exact_match_with_capacity(
                self.model,
                seed_examples,
                self.specials,
                self.bank,
                self.router,
                workspace=workspace,
                workspace_ids=ids,
            )
            return accuracy

        steps, _ = self._plateau(step_fn, eval_fn)
        self.total_train_steps += steps

        with torch.no_grad():
            current_hidden = self.model.encode(batch.input_ids).reshape(
                -1, self.model_config.d_model
            )
        replay_hidden = torch.zeros(0, self.model_config.d_model)

        consolidation_config = dataclasses.replace(seq.consolidation, seed=seq.seed * 1000 - 1)
        candidate, _ = consolidate(
            workspace,
            current_hidden,
            replay_hidden,
            consolidation_config,
            candidate_id=0,
            created_at_task=-1,
        )
        self.total_train_steps += seq.consolidation.steps

        active_ids = workspace.ids()
        with torch.no_grad():
            current_targets = apply_workspace(current_hidden, workspace, active_ids)
            replay_targets = apply_workspace(replay_hidden, workspace, active_ids)
        shadow_report = run_shadow_validation(
            workspace,
            active_ids,
            candidate,
            self.bank,
            current_hidden,
            current_targets,
            replay_hidden,
            replay_targets,
            seq.shadow,
        )
        self.seed_shadow_passed = shadow_report.passed
        if shadow_report.passed:
            self.router.add_primitive_key(candidate.primitive_id)
            calibrate_router(
                self.router,
                {candidate.primitive_id: current_hidden},
                replay_hidden,
                steps=seq.router_calibration.steps,
                lr=seq.router_calibration.lr,
            )

    def process_event(self, event: StreamEvent, index: int) -> BaselineEventReport:
        examples = self.pool_for_event(event)
        exact_match = self.evaluate(examples)
        self.event_eval_examples.append(examples)
        active_primitive_param_count = self.active_primitive_param_count(examples)
        return BaselineEventReport(
            index=index,
            label=event.label,
            operation_name=event.operation_name,
            pre_exact_match=exact_match,
            post_exact_match=exact_match,
            train_steps=0,
            resident_total_param_count=self.resident_total_param_count(),
            resident_primitive_param_count=self.resident_primitive_param_count(),
            active_primitive_param_count=active_primitive_param_count,
            active_param_count=self.model.num_parameters() + active_primitive_param_count,
        )


class _GrowRunner(_BaseRunner):
    """Shared machinery for B2 (grow-only) and B3 (grow + replay): every
    event allocates a fresh batch of transforms, trains them, then freezes
    and permanently keeps them -- no consolidation, no router, see module
    docstring."""

    use_replay: bool = False

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__(config)
        self.allocator = Allocator(d_model=self.model_config.d_model)
        self.grown: list[Primitive] = []
        self.replay_buffer: list[Example] = []

    def _hidden(
        self, input_ids: torch.Tensor, trainable: list[Primitive] | None = None
    ) -> torch.Tensor:
        hidden = self.model.encode(input_ids)
        if self.grown:
            with torch.no_grad():
                hidden = hidden + sum_primitive_deltas(self.grown, hidden)
        if trainable:
            hidden = hidden + sum_primitive_deltas(trainable, hidden)
        return hidden

    def _logits(
        self, input_ids: torch.Tensor, trainable: list[Primitive] | None = None
    ) -> torch.Tensor:
        return self.model.decode(self._hidden(input_ids, trainable))

    @torch.no_grad()
    def _generate_greedy(
        self,
        prompt_ids: torch.Tensor,
        eos_id: int,
        max_new_tokens: int,
        extra: list[Primitive] | None = None,
    ) -> tuple[int, ...]:
        self.model.eval()
        generated = prompt_ids
        new_tokens: list[int] = []
        for _ in range(max_new_tokens):
            logits = self._logits(generated, extra)
            next_id = int(logits[0, -1, :].argmax(dim=-1).item())
            new_tokens.append(next_id)
            if next_id == eos_id:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], dtype=torch.long, device=generated.device)],
                dim=1,
            )
        if new_tokens and new_tokens[-1] == eos_id:
            new_tokens = new_tokens[:-1]
        return tuple(new_tokens)

    def evaluate(self, examples: list[Example], extra: list[Primitive] | None = None) -> float:
        """`extra` lets a caller mid-PLASTIC-equivalent-training evaluate
        with the currently-training (not yet frozen/grown) transforms
        included -- without it, evaluation would only ever see `self.grown`
        (the *previous* event's permanent capacity), so a training loop's
        own plateau-detection eval would never reflect what it just
        trained."""
        correct = 0
        for example in examples:
            prompt = (self.specials.bos,) + example.input_tokens + (self.specials.sep,)
            prompt_ids = torch.tensor([prompt], dtype=torch.long)
            max_new_tokens = len(example.target_tokens) + 2
            prediction = self._generate_greedy(prompt_ids, self.specials.eos, max_new_tokens, extra)
            if prediction == example.target_tokens:
                correct += 1
        return correct / len(examples)

    def resident_primitive_param_count(self) -> int:
        return sum(t.num_parameters() for t in self.grown)

    def active_primitive_param_count(self, examples: list[Example]) -> int:
        # Every grown transform is summed unconditionally into every
        # forward pass (no routing) -- see module docstring.
        return self.resident_primitive_param_count()

    def _replay_batch(self) -> Batch | None:
        if not self.replay_buffer:
            return None
        return collate_batch(self.replay_buffer, self.specials)

    def process_event(self, event: StreamEvent, index: int) -> BaselineEventReport:
        seq = self.config.sequential
        examples = self.pool_for_event(event)
        pre = self.evaluate(examples)
        batch = collate_batch(examples, self.specials)

        workspace = PlasticWorkspace()
        ids = self.allocator.allocate(workspace, seq.allocator_preset, created_at_task=index)
        transforms = workspace.get_many(ids)
        optimizer = torch.optim.AdamW(workspace.parameters(), lr=seq.plastic.lr)

        replay_batch = self._replay_batch() if self.use_replay else None

        def step_fn() -> None:
            optimizer.zero_grad(set_to_none=True)
            logits = self._logits(batch.input_ids, transforms)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if replay_batch is not None:
                replay_logits = self._logits(replay_batch.input_ids, transforms)
                replay_loss = F.cross_entropy(
                    replay_logits.reshape(-1, replay_logits.size(-1)),
                    replay_batch.labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss = loss + self.config.replay_weight * replay_loss
            loss.backward()
            if seq.plastic.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(workspace.parameters(), seq.plastic.grad_clip)
            optimizer.step()

        def eval_fn() -> float:
            return self.evaluate(examples, transforms)

        steps, post = self._plateau(step_fn, eval_fn)
        self.total_train_steps += steps

        for t in transforms:
            t.freeze()
        self.grown.extend(transforms)

        self.event_eval_examples.append(examples)
        self.replay_buffer.extend(examples)
        max_replay = seq.replay_buffer_max_events * seq.num_examples
        self.replay_buffer = self.replay_buffer[-max_replay:]

        active_primitive_param_count = self.active_primitive_param_count(examples)
        return BaselineEventReport(
            index=index,
            label=event.label,
            operation_name=event.operation_name,
            pre_exact_match=pre,
            post_exact_match=post,
            train_steps=steps,
            resident_total_param_count=self.resident_total_param_count(),
            resident_primitive_param_count=self.resident_primitive_param_count(),
            active_primitive_param_count=active_primitive_param_count,
            active_param_count=self.model.num_parameters() + active_primitive_param_count,
        )


class B2Runner(_GrowRunner):
    """Grow-only: dynamic plastic capacity becomes permanent, no replay."""

    name = "B2"
    use_replay = False


class B3Runner(_GrowRunner):
    """Grow + replay: same growth as B2, plus a replay loss term while
    training each event's new capacity."""

    name = "B3"
    use_replay = True


_RUNNERS: dict[str, type[_BaseRunner]] = {
    "B0": B0Runner,
    "B1": B1Runner,
    "B2": B2Runner,
    "B3": B3Runner,
}


def run_baseline(name: str, config: BaselineConfig) -> BaselineReport:
    """Run one of B0-B3 end to end. For B4, call
    `apc.evaluation.sequential_benchmark.run_sequential_benchmark(config.sequential)`
    directly (its own, richer report type doesn't fit `BaselineReport`)."""
    if name not in _RUNNERS:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {sorted(_RUNNERS)} (or 'B4')")
    return _RUNNERS[name](config).run()


def run_all_baselines(
    config: BaselineConfig,
) -> dict[str, BaselineReport | SequentialBenchmarkReport]:
    """Run B0-B3 plus B4 under one shared `BaselineConfig`, returning
    `{"B0": BaselineReport, ..., "B3": BaselineReport, "B4":
    SequentialBenchmarkReport}` for comparison reporting/plotting."""
    results: dict[str, BaselineReport | SequentialBenchmarkReport] = {
        name: run_baseline(name, config) for name in _RUNNERS
    }
    results["B4"] = _run_apc(config.sequential)
    return results


def all_baselines_to_dict(
    results: dict[str, BaselineReport | SequentialBenchmarkReport],
) -> dict[str, Any]:
    return {name: report.to_dict() for name, report in results.items()}
