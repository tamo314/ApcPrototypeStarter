"""Frozen `h_content` information audit (Phase A.1 diagnostic Task
A1-R005E-002, `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`).

`docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md` opened this diagnostic phase
after A1-R005's retry (D-001 through D-008, `docs/results/
PHASE_A1_R005_RETRY_D001_D008_SUMMARY.md`) found that all four parameterized
primitives (`SHIFT`/`SELECT`/`COUNT`/`BIND`) fail the causal ablation matrix
under a shared low-rank additive-conditioning architecture, without
distinguishing whether the bottleneck is the frozen task-blind content
representation those primitives read (`h_content`) or the primitive/operator
computation class itself. This module is `D-E1` / `A1-R005E-002`: measure,
per operation, how much raw information the frozen `h_content` the retry's
own primitives actually consumed still contains -- before any operator-class
question (`A1-R005E-003` onward) is asked.

## Diagnostic only

Per `docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md` D-E1 and `docs/
CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` A1-R005E-002's own "Acceptance:
diagnostic only; explicitly state what information is and is not
recoverable" -- unlike every earlier Phase A.1 primitive gate, this module
has no STOP GATE `passed` verdict. `*_DESIRABLE_THRESHOLD` constants below
are the experiment plan's own "suggested diagnostic thresholds" surfaced as
informational `*_meets_threshold` booleans per report, not a pass/fail gate
that blocks downstream work.

## Four checkpoints, not one

`docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` A1-R005E-002 asks to reuse "the
same relevant frozen checkpoints" the retry actually trained primitives
against. Unlike `apc.evaluation.parameterized_primitive_gate` (A1-R005's
original, since-retried gate), which pretrained *one* Stable Core shared
across all four operations, the retry's own counterfactual gates
(`apc.evaluation.count_counterfactual_gate`/`bind_counterfactual_gate`/
`sequence_counterfactual_gate`) each pretrained a *dedicated*,
single-operation task-blind Stable Core -- so "the same checkpoint" for
`COUNT` is not the same weights as "the same checkpoint" for `BIND`. This
module reproduces that: one frozen `DecoderOnlyTransformer` per
`config.operation_names` entry, each pretrained via `apc.evaluation.
shared_core_generalization.train_shared_core` with `include_task_spec=False`
and `operation_names=(that one operation,)`, using the identical model
architecture and `core_train` budget the retry's own configs pinned
(`d_model=192, n_layer=4, n_head=4, d_ff=768, max_seq_len=48`;
`steps=30000, batch_size=128, lr=3e-4, weight_decay=0.01, grad_clip=1.0`).
Given the same `seed`, this reproduces the retry's own checkpoints
bit-for-bit (this codebase never serializes Stable Core weights to disk --
every Phase A.1 gate treats "same config + same seed" as "same checkpoint",
matching `apc.evaluation.count_counterfactual_gate`'s own precedent), so no
literal weight file needs to exist for this module to test the retry's own
representation. `apc.primitives`/`apc.plastic`/`apc.consolidation`/`apc.meta`
are never imported here (no primitive is trained or evaluated by this
module) -- only `h_content = model.encode(...)` is read, per the task's own
"Do not alter Stable Core."

## What `h_content[j]` means here

Every probe below reads `content_state = model.encode(content_only_ids)`
where `content_only_ids` is exactly `apc.core.data.build_content_only_tokens`
`= (bos,) + input_tokens + (sep,)` -- the *same* content-only prompt
`apc.core.execution.evaluate_exact_match_no_primitive`/
`generate_greedy_with_oracle_call`'s first decoding step feeds through
`model.encode` (this codebase's causal primitive path never runs
`encode_split`/`encode_task_content_split` for these task-blind-pretrained
cores; see `apc.core.execution` module docstring, "Decoder-input audit").
Content position `j` (0-indexed within `input_tokens`) is
`content_state[:, 1 + j, :]` -- offset by 1 for `[BOS]`.

## Probes (per `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` A1-R005E-002)

1. **Token identity**: per-position linear probe, `h_content[j] -> token id
   at position j`. Almost tautologically recoverable for *this* position
   (the state was computed from that very token via the residual stream),
   but not free -- see design-docs/REPRESENTATION_OPERATOR_ISOLATION.md
   section 4, "reconstruction is not enough" -- reported as a baseline floor
   the other probes are compared against, not as a hard test.
2. **Absolute position**: per-position linear probe, `h_content[j] ->
   position j` (0-indexed within content, matching where a `SHIFT`/`SELECT`
   primitive would need to address relative to). Learned absolute position
   embeddings are added at the input, so this too is expected near ceiling
   unless training somehow discarded them.
3. **Full content-sequence reconstruction**: unlike probes 1-2, this reads a
   *single* vector per example -- `content_state` at the `[SEP]` position,
   i.e. exactly the state `evaluate_exact_match_no_primitive`'s first
   decoding step (and every oracle-forced primitive call's first step)
   conditions on to predict the first output token. This is a categorically
   harder and more operationally relevant question than probes 1-2: those
   ask "does *this* position's own state still carry its own token" (a
   position that, causally, has attended to nothing after itself);
   reconstructing the *entire* input from the one state at `[SEP]` (which
   has attended over every content position) asks whether that single
   downstream-facing vector -- not the whole per-position sequence -- still
   carries the whole input, which is what a compact argument-conditioned
   primitive reading a *pooled or terminal* summary (rather than attending
   freely over every position, the way an oracle latent operator in
   A1-R005E-003 is allowed to) would actually have available.
4. **BIND key/value role probe** (`operation == "BIND"` only): per-position
   linear probe, `h_content[j] -> even/odd position parity` (`apc.
   environments.operations.BindOp`: keys at even positions, values at odd
   positions). Directly load-bearing for BIND's associative-retrieval
   computation (`docs/DECISIONS.md` ADR-0036's "position-targeted attention"
   candidate explanation for BIND's near-zero causal gap).
5. **Adjacency/pair probe** (`operation == "BIND"` only, optional per the
   task text): per-key-position linear probe, `h_content[key position] ->
   token id of that key's own paired value (key position + 1)`. Tests
   whether a single position's frozen state carries its *neighbor's*
   identity without attending to it again -- the specific structural
   information a keyed-retrieval operator (A1-R005E-003's BIND oracle
   latent operator) needs to be built on top of, distinct from role probe 4
   (which only asks "am I a key or a value", not "what is my partner").

## Filling in an undeclared threshold

`docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md` D-E1 states desirable numbers
for token identity (`>=0.98`), position (`>=0.98`), and full reconstruction
(`>=0.95` sequence exact) but not for the optional role/pair probes. Per
`AGENTS.md`'s decision log guidance, `ROLE_PROBE_DESIRABLE_THRESHOLD`/
`PAIR_PROBE_DESIRABLE_THRESHOLD` reuse the same `0.95` "near ceiling"
convention as the reconstruction probe: both are coarser readouts (a binary
role label; a single-token partner identity) than reconstructing the whole
sequence, so if the frozen representation is not the bottleneck they should
be at least as recoverable. See `docs/DECISIONS.md` for the ADR recording
this.

## Seed policy

`docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md` section 8 states seed policy
only for `D-E3`/`D-E5` ("development: 1-2 seeds; decision evidence: >=5
seeds"), not for `D-E1`. This module follows `apc.evaluation.
task_content_probes`' own precedent for an analogous "no minimum seed count
stated" diagnostic: 3 seeds (`DEFAULT_SEEDS`) as a modest robustness check
beyond a single run, without the cost (4 dedicated 30000-step Stable Cores
per seed) of a full 5-seed sweep.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import collate_content_only_batch
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SharedCoreTokens
from apc.environments.generator import Example, TaskGenerator
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.evaluation.task_content_probes import ProbeTrainConfig
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "BIND_ROLE_OPERATIONS",
    "TOKEN_IDENTITY_DESIRABLE_THRESHOLD",
    "POSITION_DESIRABLE_THRESHOLD",
    "RECONSTRUCTION_EXACT_MATCH_DESIRABLE_THRESHOLD",
    "ROLE_PROBE_DESIRABLE_THRESHOLD",
    "PAIR_PROBE_DESIRABLE_THRESHOLD",
    "RepresentationAuditConfig",
    "representation_audit_config_from_dict",
    "FrozenStableCore",
    "OperationRepresentationAuditReport",
    "RepresentationAuditReport",
    "run_representation_audit",
    "OperationRepresentationAuditSummary",
    "RepresentationAuditMultiSeedReport",
    "run_representation_audit_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# apc.environments.operations.BindOp is the only currently registered
# operation with a well-defined key/value role per content position.
BIND_ROLE_OPERATIONS: tuple[str, ...] = ("BIND",)

# docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md D-E1's own "suggested
# diagnostic thresholds" -- informational only (module docstring, "Diagnostic
# only"), not an APC/STOP-GATE acceptance criterion.
TOKEN_IDENTITY_DESIRABLE_THRESHOLD = 0.98
POSITION_DESIRABLE_THRESHOLD = 0.98
RECONSTRUCTION_EXACT_MATCH_DESIRABLE_THRESHOLD = 0.95
# Not stated by D-E1's own text; filled in from the same convention (module
# docstring, "Filling in an undeclared threshold").
ROLE_PROBE_DESIRABLE_THRESHOLD = 0.95
PAIR_PROBE_DESIRABLE_THRESHOLD = 0.95


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
class RepresentationAuditConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-002
    audit run, covering every operation in `operation_names`.

    `model`/`core_train` default to the retry's own shared baseline budget
    (module docstring, "Four checkpoints, not one") so a config file that
    omits them reproduces the retry's checkpoints exactly.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    probe_train: ProbeTrainConfig = field(default_factory=ProbeTrainConfig)
    num_probe_train_examples: int = 4096
    num_probe_eval_examples: int = 2048

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.num_probe_train_examples < 1:
            raise ValueError(
                f"num_probe_train_examples must be >= 1, got {self.num_probe_train_examples}"
            )
        if self.num_probe_eval_examples < 1:
            raise ValueError(
                f"num_probe_eval_examples must be >= 1, got {self.num_probe_eval_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def representation_audit_config_from_dict(raw: dict[str, Any]) -> RepresentationAuditConfig:
    """Parse a `configs/phase_a1_representation_audit.yaml`-shaped dict,
    matching every other Phase A.1 gate's convention of filling in defaults
    for whatever the file omits."""
    defaults = RepresentationAuditConfig()
    core_train_raw = raw.get("core_train")
    core_train = (
        dataclasses.replace(defaults.core_train, **core_train_raw)
        if core_train_raw
        else defaults.core_train
    )
    probe_train_raw = raw.get("probe_train")
    probe_train = (
        dataclasses.replace(defaults.probe_train, **probe_train_raw)
        if probe_train_raw
        else defaults.probe_train
    )
    return RepresentationAuditConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        probe_train=probe_train,
        num_probe_train_examples=raw.get(
            "num_probe_train_examples", defaults.num_probe_train_examples
        ),
        num_probe_eval_examples=raw.get(
            "num_probe_eval_examples", defaults.num_probe_eval_examples
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind on one operation's ordinary i.i.d.
    stream, then frozen: every parameter has `requires_grad=False`. Carries
    its own `TaskGenerator` (module docstring, "Four checkpoints, not one")
    so probe-train/probe-eval batches are drawn from the identical stream the
    core itself was pretrained on, matching `apc.evaluation.
    task_content_probes.run_task_content_probes`'s own convention."""

    operation: str
    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    generator: TaskGenerator
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: RepresentationAuditConfig, operation: str, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    """Pretrain one `DecoderOnlyTransformer` via `train_shared_core` with no
    task segment ever visible (`include_task_spec=False`), restricted to a
    single operation, then freeze every parameter -- see module docstring's
    "Four checkpoints, not one"."""
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=(operation,),
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
        operation=operation,
        model=model,
        tokens=trained.tokens,
        generator=trained.generator,
        final_core_train_loss=trained.final_train_loss,
        device=trained.device,
    )


@dataclass
class _ExtractedRepresentation:
    """Frozen `h_content` features pulled from one batch of examples,
    gathered into the tensors every probe below trains/evaluates on. Plain
    (non-frozen) dataclass -- purely an internal transport object between
    `_extract_representation` and the probe-fitting functions."""

    content_features: torch.Tensor  # [sum(content_len), d_model] -- h_content[j], probes 1/2/4.
    content_token_labels: torch.Tensor  # [sum(content_len)] long -- token id at j. Probe 1.
    content_position_labels: torch.Tensor  # [sum(content_len)] long -- position j. Probe 2.
    summary_features: torch.Tensor  # [N, d_model] -- h_content at [SEP]. Probe 3.
    reconstruction_targets: torch.Tensor  # [N, max_content_length] long -- 0 outside valid.
    reconstruction_mask: torch.Tensor  # [N, max_content_length] float -- 1.0 at valid positions.
    role_labels: torch.Tensor | None  # [sum(content_len)] long, 0=key/1=value. Probe 4.
    key_features: torch.Tensor | None  # [num_keys, d_model] -- h_content at each key. Probe 5.
    key_pair_targets: torch.Tensor | None  # [num_keys] long -- paired value's token id. Probe 5.


def _extract_representation(
    core: FrozenStableCore,
    examples: list[Example],
    *,
    max_content_length: int,
    include_roles: bool,
) -> _ExtractedRepresentation:
    """One frozen forward pass (`model.encode`, `torch.no_grad()`) over
    `examples`' content-only prompts, gathered into the fixed-size tensors
    every probe below trains/evaluates on. `core.model` is never put into
    `.train()` mode and never receives a gradient here."""
    model, tokens, device = core.model, core.tokens, core.device
    content_batch = collate_content_only_batch(examples, tokens, device=device)
    model.eval()
    with torch.no_grad():
        content_state = model.encode(content_batch)

    content_feature_rows: list[torch.Tensor] = []
    content_token_rows: list[torch.Tensor] = []
    content_position_rows: list[torch.Tensor] = []
    summary_rows: list[torch.Tensor] = []
    reconstruction_targets = torch.zeros(
        len(examples), max_content_length, dtype=torch.long, device=device
    )
    reconstruction_mask = torch.zeros(len(examples), max_content_length, device=device)
    role_rows: list[torch.Tensor] = []
    key_feature_rows: list[torch.Tensor] = []
    key_pair_target_rows: list[int] = []

    for row, example in enumerate(examples):
        length = len(example.input_tokens)
        if length > max_content_length:
            raise ValueError(
                f"example content length {length} exceeds max_content_length="
                f"{max_content_length}; widen sequence_length_range[1]"
            )
        positions = content_state[row, 1 : 1 + length, :]
        content_feature_rows.append(positions)
        token_ids = torch.tensor(example.input_tokens, dtype=torch.long, device=device)
        content_token_rows.append(token_ids)
        content_position_rows.append(torch.arange(length, device=device))
        summary_rows.append(content_state[row, 1 + length, :])
        reconstruction_targets[row, :length] = token_ids
        reconstruction_mask[row, :length] = 1.0

        if include_roles:
            if length % 2 != 0:
                raise ValueError(
                    "BIND role/pair probes require even-length content "
                    f"(apc.environments.operations.BindOp), got length={length}"
                )
            role_rows.append(
                torch.tensor([j % 2 for j in range(length)], dtype=torch.long, device=device)
            )
            for j in range(0, length, 2):
                key_feature_rows.append(positions[j])
                key_pair_target_rows.append(int(example.input_tokens[j + 1]))

    return _ExtractedRepresentation(
        content_features=torch.cat(content_feature_rows, dim=0),
        content_token_labels=torch.cat(content_token_rows, dim=0),
        content_position_labels=torch.cat(content_position_rows, dim=0),
        summary_features=torch.stack(summary_rows, dim=0),
        reconstruction_targets=reconstruction_targets,
        reconstruction_mask=reconstruction_mask,
        role_labels=torch.cat(role_rows, dim=0) if include_roles else None,
        key_features=torch.stack(key_feature_rows, dim=0) if include_roles else None,
        key_pair_targets=(
            torch.tensor(key_pair_target_rows, dtype=torch.long, device=device)
            if include_roles
            else None
        ),
    )


def _fit_classification_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    num_classes: int,
    train_config: ProbeTrainConfig,
    device: torch.device,
) -> float:
    """Fit one `nn.Linear(d_model, num_classes)` by full-batch AdamW
    cross-entropy on `(train_features, train_labels)`; return top-1 accuracy
    on the disjoint `(eval_features, eval_labels)`. Identical shape to
    `apc.evaluation.task_content_probes._fit_classification_probe`, kept as
    its own copy here per this codebase's "each gate module is independently
    reviewable end to end" convention (see e.g. `apc.evaluation.
    parameterized_primitive_gate` module docstring)."""
    d_model = train_features.shape[-1]
    probe = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    probe.train()
    for _ in range(train_config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        predictions = probe(eval_features).argmax(dim=-1)
        return (predictions == eval_labels).to(torch.float32).mean().item()


def _fit_reconstruction_probe(
    train_summary: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    eval_summary: torch.Tensor,
    eval_targets: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    max_content_length: int,
    vocab_size: int,
    train_config: ProbeTrainConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Fit one `nn.Linear(d_model, max_content_length * vocab_size)` --
    reshaped to `[max_content_length, vocab_size]` per-position logits -- by
    full-batch AdamW cross-entropy (masked to each example's own valid
    positions) on `(train_summary, train_targets, train_mask)`. Returns
    `(token_accuracy, exact_match)` on the disjoint eval set: masked
    per-position top-1 accuracy, and the fraction of examples whose entire
    valid span (all positions `< length`) is reconstructed exactly."""
    d_model = train_summary.shape[-1]
    head = nn.Linear(d_model, max_content_length * vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    head.train()
    for _ in range(train_config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_summary).view(-1, max_content_length, vocab_size)
        per_position_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), train_targets.reshape(-1), reduction="none"
        ).view(-1, max_content_length)
        loss = (per_position_loss * train_mask).sum() / train_mask.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        eval_logits = head(eval_summary).view(-1, max_content_length, vocab_size)
        predictions = eval_logits.argmax(dim=-1)
        correct = (predictions == eval_targets).to(torch.float32) * eval_mask
        token_accuracy = (correct.sum() / eval_mask.sum().clamp_min(1.0)).item()
        mismatches = ((predictions != eval_targets).to(torch.float32) * eval_mask).sum(dim=-1)
        exact_match = (mismatches == 0).to(torch.float32).mean().item()
    return token_accuracy, exact_match


@dataclass(frozen=True)
class OperationRepresentationAuditReport:
    """Everything observed while pretraining, freezing, and probing one
    operation's frozen Stable Core for one seed."""

    operation: str
    core_final_train_loss: float
    core_param_count: int
    core_trainable_param_count: int
    num_probe_train_examples: int
    num_probe_eval_examples: int
    token_identity_accuracy: float
    token_identity_meets_threshold: bool
    position_accuracy: float
    position_meets_threshold: bool
    reconstruction_token_accuracy: float
    reconstruction_exact_match: float
    reconstruction_meets_threshold: bool
    role_accuracy: float | None
    role_meets_threshold: bool | None
    pair_accuracy: float | None
    pair_meets_threshold: bool | None
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def _run_operation_audit(
    config: RepresentationAuditConfig,
    operation: str,
    *,
    core_metrics_path: str | Path | None = None,
) -> OperationRepresentationAuditReport:
    """Run the full A1-R005E-002 audit for one `(config.seed, operation)`
    pair: pretrain and freeze that operation's dedicated Stable Core, draw
    fresh probe-train/probe-eval batches from its own generator, and fit
    every probe described in the module docstring."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, operation, metrics_path=core_metrics_path)

    # Distinct, far-apart online steps from the core's own train("train")/
    # progress-eval("val") calls, and from each other -- see apc.evaluation.
    # task_content_probes.run_task_content_probes for the identical
    # (seed, step, split) determinism argument.
    probe_train_examples = core.generator.generate_online(
        config.num_probe_train_examples, step=10_000, split="test"
    )
    probe_eval_examples = core.generator.generate_online(
        config.num_probe_eval_examples, step=20_000, split="test"
    )

    max_content_length = config.sequence_length_range[1]
    include_roles = operation in BIND_ROLE_OPERATIONS

    train_features = _extract_representation(
        core,
        probe_train_examples,
        max_content_length=max_content_length,
        include_roles=include_roles,
    )
    eval_features = _extract_representation(
        core,
        probe_eval_examples,
        max_content_length=max_content_length,
        include_roles=include_roles,
    )

    # Re-anchor the shared global RNG stream before fitting probes (ADR-0016
    # pattern), matching apc.evaluation.task_content_probes.
    set_seed(config.seed)

    token_identity_accuracy = _fit_classification_probe(
        train_features.content_features,
        train_features.content_token_labels,
        eval_features.content_features,
        eval_features.content_token_labels,
        config.vocab_size,
        config.probe_train,
        core.device,
    )
    position_accuracy = _fit_classification_probe(
        train_features.content_features,
        train_features.content_position_labels,
        eval_features.content_features,
        eval_features.content_position_labels,
        max_content_length,
        config.probe_train,
        core.device,
    )
    reconstruction_token_accuracy, reconstruction_exact_match = _fit_reconstruction_probe(
        train_features.summary_features,
        train_features.reconstruction_targets,
        train_features.reconstruction_mask,
        eval_features.summary_features,
        eval_features.reconstruction_targets,
        eval_features.reconstruction_mask,
        max_content_length=max_content_length,
        vocab_size=config.vocab_size,
        train_config=config.probe_train,
        device=core.device,
    )

    role_accuracy: float | None = None
    pair_accuracy: float | None = None
    if include_roles:
        assert train_features.role_labels is not None
        assert eval_features.role_labels is not None
        role_accuracy = _fit_classification_probe(
            train_features.content_features,
            train_features.role_labels,
            eval_features.content_features,
            eval_features.role_labels,
            2,
            config.probe_train,
            core.device,
        )
        assert train_features.key_features is not None
        assert train_features.key_pair_targets is not None
        assert eval_features.key_features is not None
        assert eval_features.key_pair_targets is not None
        pair_accuracy = _fit_classification_probe(
            train_features.key_features,
            train_features.key_pair_targets,
            eval_features.key_features,
            eval_features.key_pair_targets,
            config.vocab_size,
            config.probe_train,
            core.device,
        )

    return OperationRepresentationAuditReport(
        operation=operation,
        core_final_train_loss=core.final_core_train_loss,
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        num_probe_train_examples=len(probe_train_examples),
        num_probe_eval_examples=len(probe_eval_examples),
        token_identity_accuracy=token_identity_accuracy,
        token_identity_meets_threshold=(
            token_identity_accuracy >= TOKEN_IDENTITY_DESIRABLE_THRESHOLD
        ),
        position_accuracy=position_accuracy,
        position_meets_threshold=position_accuracy >= POSITION_DESIRABLE_THRESHOLD,
        reconstruction_token_accuracy=reconstruction_token_accuracy,
        reconstruction_exact_match=reconstruction_exact_match,
        reconstruction_meets_threshold=(
            reconstruction_exact_match >= RECONSTRUCTION_EXACT_MATCH_DESIRABLE_THRESHOLD
        ),
        role_accuracy=role_accuracy,
        role_meets_threshold=(
            None if role_accuracy is None else role_accuracy >= ROLE_PROBE_DESIRABLE_THRESHOLD
        ),
        pair_accuracy=pair_accuracy,
        pair_meets_threshold=(
            None if pair_accuracy is None else pair_accuracy >= PAIR_PROBE_DESIRABLE_THRESHOLD
        ),
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


@dataclass(frozen=True)
class RepresentationAuditReport:
    """Everything observed for one seed, across every operation in
    `config.operation_names`."""

    config: RepresentationAuditConfig
    per_operation: dict[str, OperationRepresentationAuditReport]
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "per_operation": {
                name: report.to_dict() for name, report in self.per_operation.items()
            },
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_representation_audit(
    config: RepresentationAuditConfig,
    *,
    core_metrics_dir: str | Path | None = None,
) -> RepresentationAuditReport:
    """Run one seed of the A1-R005E-002 audit: for every operation in
    `config.operation_names`, pretrain and freeze its own dedicated Stable
    Core and fit every probe described in the module docstring."""
    start = time.perf_counter()
    per_operation: dict[str, OperationRepresentationAuditReport] = {}
    device_str = "cpu"
    metrics_dir_path = Path(core_metrics_dir) if core_metrics_dir is not None else None
    for operation in config.operation_names:
        metrics_path = (
            metrics_dir_path / f"{operation}_core_metrics.jsonl" if metrics_dir_path else None
        )
        report = _run_operation_audit(config, operation, core_metrics_path=metrics_path)
        per_operation[operation] = report
        device_str = report.device
    return RepresentationAuditReport(
        config=config,
        per_operation=per_operation,
        wall_clock_seconds=time.perf_counter() - start,
        device=device_str,
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class OperationRepresentationAuditSummary:
    """One operation's per-probe means aggregated across seeds."""

    operation: str
    mean_token_identity_accuracy: float
    stdev_token_identity_accuracy: float
    mean_position_accuracy: float
    stdev_position_accuracy: float
    mean_reconstruction_token_accuracy: float
    mean_reconstruction_exact_match: float
    stdev_reconstruction_exact_match: float
    mean_role_accuracy: float | None
    mean_pair_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RepresentationAuditMultiSeedReport:
    """A1-R005E-002's full result across seeds. There is no `passed` verdict
    (module docstring, "Diagnostic only") -- `docs/results/
    A1_R005E_DIAGNOSTIC_RESULT.md` (A1-R005E-008) is where these numbers are
    interpreted against `docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md`."""

    seeds: tuple[int, ...]
    per_seed: tuple[RepresentationAuditReport, ...]
    per_operation_summary: dict[str, OperationRepresentationAuditSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "per_operation_summary": {
                name: summary.to_dict() for name, summary in self.per_operation_summary.items()
            },
        }


def run_representation_audit_multi_seed(
    base_config: RepresentationAuditConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> RepresentationAuditMultiSeedReport:
    """Run one audit seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation."""
    per_seed: list[RepresentationAuditReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        report = run_representation_audit(config, core_metrics_dir=seed_dir)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    per_operation_summary: dict[str, OperationRepresentationAuditSummary] = {}
    for operation in base_config.operation_names:
        op_reports = [report.per_operation[operation] for report in per_seed]
        token_mean, token_stdev, _, _ = _summarize(
            [r.token_identity_accuracy for r in op_reports]
        )
        position_mean, position_stdev, _, _ = _summarize(
            [r.position_accuracy for r in op_reports]
        )
        recon_token_mean, _, _, _ = _summarize(
            [r.reconstruction_token_accuracy for r in op_reports]
        )
        recon_exact_mean, recon_exact_stdev, _, _ = _summarize(
            [r.reconstruction_exact_match for r in op_reports]
        )
        role_values = [r.role_accuracy for r in op_reports if r.role_accuracy is not None]
        pair_values = [r.pair_accuracy for r in op_reports if r.pair_accuracy is not None]
        per_operation_summary[operation] = OperationRepresentationAuditSummary(
            operation=operation,
            mean_token_identity_accuracy=token_mean,
            stdev_token_identity_accuracy=token_stdev,
            mean_position_accuracy=position_mean,
            stdev_position_accuracy=position_stdev,
            mean_reconstruction_token_accuracy=recon_token_mean,
            mean_reconstruction_exact_match=recon_exact_mean,
            stdev_reconstruction_exact_match=recon_exact_stdev,
            mean_role_accuracy=statistics.fmean(role_values) if role_values else None,
            mean_pair_accuracy=statistics.fmean(pair_values) if pair_values else None,
        )

    return RepresentationAuditMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
    )
