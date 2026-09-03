"""Oracle latent operator benchmark (Phase A.1 diagnostic Task
A1-R005E-003, `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`).

`docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md` section 6 ("Oracle
latent operators") and `docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md` D-E2
motivate this module: A1-R005E-002's representation audit (`apc.evaluation.
representation_audit`, ADR-0039) found that `SHIFT`/`SELECT` retain strong
per-position token/position information (`0.94-1.00`) in the frozen
task-blind `h_content`, but that a *single pooled summary vector* (the state
at `[SEP]`, exactly what a pointwise argument-conditioned primitive reads)
cannot reconstruct the full input for any of the four operations
(`0.00-0.02` sequence exact match). This leaves open whether the *full
per-position* `h_content` sequence -- not the one compressed summary -- is
already sufficient, given perfect (oracle) knowledge of which position(s) to
read, to decode the correct target. This module (`D-E2`) answers exactly
that question, before any question of *learned* operator capacity
(`A1-R005E-004`) is asked: "determine whether perfect addressing on frozen
hidden states enables strong decoding" (A1-R005E-003's own goal text).

## Zero new training for SHIFT/SELECT/BIND: reuse the model's own decode head

`apc.core.model.DecoderOnlyTransformer` is trained end to end (module
docstring: "trained autoregressively on `[BOS] input... [SEP] target...
[EOS]`") with a full next-token cross-entropy loss over *every* position,
including the content region itself -- not only the answer span. Consequently
`decode(content_state[:, k, :])` (content-only prompt padded as
`[BOS, input_0, ..., input_{L-1}, SEP]`, so padded index `k` for
`k` in `[0, L-1]` is the state that has attended through exactly
`[BOS, input_0, ..., input_{k-1}]`) is already trained, by ordinary
pretraining alone, to approximate `input_tokens[k]` as its argmax next-token
prediction -- with no primitive, no routing, and no extra training of any
kind. Concretely: padded index `0` is `[BOS]` itself, whose state predicts
`input_tokens[0]`; padded index `k >= 1` is `input_tokens[k-1]`, whose state
(having just attended through that token) predicts `input_tokens[k]`. So:

    `predict_input_token_state(content_state, k) = content_state[:, k, :]`
    approximates `input_tokens[k]` via `model.decode(...)`, for `k` in
    `[0, L-1]`.

`SHIFT`, `SELECT`, and `BIND` all produce a target that is a *literal copy*
of one or more input tokens at oracle-known source positions (`ShiftOp`:
`target[t] = input[(amount + t) % L]`; `SelectOp`: `target[t] =
input[indices[t]]`; `BindOp`: `target[0] = input[key_position + 1]`, the
value paired with the queried key, "last match wins" on a repeated key). The
oracle latent operator for each of these three operations is therefore
exactly "permute which content position's hidden state feeds `decode` for
each output slot, using the oracle's own knowledge of the source-position
mapping" (`REPRESENTATION_OPERATOR_ISOLATION.md`: "Permute hidden-state
positions by oracle `amount` ... gather ... by oracle ordered `indices` ...
locate queried key; gather/read associated value hidden state" -- read
literally). No parameter here is fit to any data at all: `_source_positions_
for_example`/`_oracle_decode_predictions` below are pure indexing plus the
one already-frozen `model.decode` call.

The oracle is only ever allowed to look at raw content symbols to *find*
these positions (`example.input_tokens`, matching `AGENTS_A1_R005E_DIAGNOSTIC_
ADDENDUM.md`'s "Oracle operator rule": "Oracle operators may use ground-truth
argument semantics and exact content positions for diagnosis") -- the actual
prediction at each addressed position always comes from `model.decode`,
never from `example.target_tokens` itself.

## COUNT is different in kind: a small diagnostic readout, not zero training

`CountOp`'s target (`min(count(content, target), vocab_size - 1)`) is not a
copy of any single input token's identity -- it is a scalar aggregate over
however many positions match the query. There is no padded index whose
ordinary next-token pretraining target is "the number of matches", so
"permute a hidden state into the decode slot" has no analogue here.
`REPRESENTATION_OPERATOR_ISOLATION.md` section 6 explicitly anticipates this
by offering two options for `COUNT` specifically ("aggregate matched
positions with a documented deterministic or small diagnostic count
readout") where `SHIFT`/`SELECT`/`BIND`'s text has only one. This module
takes the second option: oracle-locate every matched position from raw
symbols (`_count_match_positions`), sum-pool (not mean-pool -- see below)
`h_content` at those positions into one vector, and fit a small
`nn.Linear(d_model, vocab_size)` readout (`ProbeTrainConfig`, the same
"lightweight linear probe" budget `apc.evaluation.task_content_probes`/
`representation_audit` already use) from that pooled vector to the count
token, on a disjoint train/eval split. This is still squarely inside this
diagnostic phase's own rules (`AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`:
"Modules introduced here may intentionally be too large/expensive to be APC
primitives ... upper-bound probes, not candidate production architecture")
and is a categorically smaller commitment than `A1-R005E-004`'s later
learned high-capacity *operator* (this module's readout never sees the
argument beyond using it, via the oracle, to select which positions to pool
-- there is no argument-conditioning path inside the readout itself, unlike
a real `ConditionedPrimitive`).

Sum-pool, not mean-pool: a query target can match zero positions in a given
example (a real, common outcome at `vocab_size=10`, `sequence_length_range=
(6, 10)` -- `target_tokens[0] == 0` is a legitimate label, not an edge case
to exclude). Mean-pooling an empty set of matched states is undefined
(division by zero); sum-pooling an empty set is exactly the zero vector, a
well-defined input the readout can learn to map to the "0 matches" class
like any other.

Per-operation matched-position feature convention: this module gathers
`content_state` at each matched position using the *same* "state at the
token itself" indexing `apc.evaluation.representation_audit` uses for its
own token/position probes (`content_state[:, 1 + i, :]` for 0-indexed
content position `i`) -- unlike the "state *before* the token" indexing
`predict_input_token_state` uses above for `SHIFT`/`SELECT`/`BIND`. This is
a deliberate, not arbitrary, difference: `SHIFT`/`SELECT`/`BIND`'s indexing
is dictated by what `decode` was already pretrained to predict from a given
position (there is only one correct choice); `COUNT`'s readout is a *new*
head being fit from scratch, so there is no pretrained-target constraint
picking an offset for it, and "the state that has just seen the matching
token" is the more natural reading of "the position's own identity/evidence"
for an aggregation task -- matching `representation_audit`'s own
`content_features` convention for exactly that reason.

## Acceptance targets (unlike A1-R005E-002, this task states numeric targets)

`docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` A1-R005E-003's own "Acceptance
targets": `SHIFT`/`SELECT`/`BIND` exact match `>= 0.90` and token accuracy
`>= 0.98`; `COUNT` exact match `>= 0.90` (no token-accuracy figure is
restated for `COUNT`, though it is numerically identical to exact match for
this operation since `CountOp.output_length` is always `1` -- `BIND`'s
output is also always length `1`, so its own exact-match/token-accuracy
numbers are likewise identical by construction; the two-metric bar as
written still applies to it since the task text names it alongside
`SHIFT`/`SELECT`). Because explicit numeric targets are stated here (unlike
A1-R005E-002's "diagnostic only, no pass/fail gate"), this module computes
`*_passed` verdicts, matching every other Phase A.1 primitive gate's own
convention (mean-across-seeds, per `docs/DECISIONS.md` ADR-0033's "the
gate's own `passed` computation uses the cross-seed mean" precedent) rather
than A1-R005E-002's own no-verdict shape.

Per A1-R005E-003's own "Interpret failure as representation/decoder/
interface evidence, not learned-routing failure": no `Router`, `PrimitiveBank`,
`Primitive`, `PlasticWorkspace`, or `apc.consolidation`/`apc.meta` module is
imported anywhere in this file -- there is no learned routing decision
anywhere in this benchmark's call graph to blame a failure on.

## Four checkpoints, not one; seed policy

Same precedent as `apc.evaluation.representation_audit` (module docstring,
"Four checkpoints, not one"; see also that module's "Seed policy" section):
one dedicated, task-blind, single-operation frozen Stable Core is pretrained
per `(seed, operation)` pair via `apc.evaluation.shared_core_generalization.
train_shared_core` with `include_task_spec=False`, at the identical
`model`/`core_train` budget the retry's own counterfactual gates and
A1-R005E-002 used -- so a config file that leaves `model`/`core_train` at
their retry-matching values genuinely tests "the same relevant frozen
checkpoints" A1-R005E-002 already audited, per this task's own "determine
whether perfect addressing on frozen hidden states enables strong decoding"
(the frozen hidden states in question). `docs/EXPERIMENT_PLAN_A1_R005E_
DIAGNOSTIC.md` section 8 states a seed policy only for `D-E3`/`D-E5`
("development: 1-2 seeds; decision evidence: >=5 seeds"), not `D-E2`; this
module reuses A1-R005E-002's own fill-in (`DEFAULT_SEEDS = (0, 1, 2)`) for
the same reason that module gave (a modest robustness check beyond a single
run, without a full 5-seed sweep's cost).
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
from apc.environments.generator import Example, TaskGenerator, oracle_call_for_example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES
from apc.environments.primitive_call import PrimitiveCall
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
    "COUNT_READOUT_OPERATIONS",
    "TOKEN_ACCURACY_GATED_OPERATIONS",
    "EXACT_MATCH_THRESHOLD",
    "TOKEN_ACCURACY_THRESHOLD",
    "OracleLatentOperatorConfig",
    "oracle_latent_operator_config_from_dict",
    "FrozenStableCore",
    "OperationOracleLatentOperatorReport",
    "OracleLatentOperatorReport",
    "run_oracle_latent_operator_benchmark",
    "OperationOracleLatentOperatorSummary",
    "OracleLatentOperatorMultiSeedReport",
    "run_oracle_latent_operator_benchmark_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# apc.environments.operations.CountOp is the only registered operation whose
# oracle latent operator uses a trained readout instead of pure re-routed
# decode() -- see module docstring, "COUNT is different in kind".
COUNT_READOUT_OPERATIONS: tuple[str, ...] = ("COUNT",)

# docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md A1-R005E-003's own "Acceptance
# targets" gate token accuracy for SHIFT/SELECT/BIND explicitly; COUNT's own
# target output is always length 1 (CountOp.output_length), so its token
# accuracy is numerically identical to its exact match by construction and
# is reported but not separately gated, matching the task text's own list.
TOKEN_ACCURACY_GATED_OPERATIONS: tuple[str, ...] = ("SHIFT", "SELECT", "BIND")

EXACT_MATCH_THRESHOLD = 0.90
TOKEN_ACCURACY_THRESHOLD = 0.98


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
class OracleLatentOperatorConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-003
    benchmark run, covering every operation in `operation_names`.

    `model`/`core_train` default to the generic `SharedCoreGateTrainConfig`
    budget; a config file that wants to reproduce the retry's own
    checkpoints (module docstring, "Four checkpoints, not one") pins the
    same `steps=30000, batch_size=128, lr=3e-4, ...` values A1-R005E-002's
    own config file does.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    num_eval_examples: int = 2048
    count_readout_train: ProbeTrainConfig = field(default_factory=ProbeTrainConfig)
    num_count_readout_train_examples: int = 4096

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.num_eval_examples < 1:
            raise ValueError(f"num_eval_examples must be >= 1, got {self.num_eval_examples}")
        if self.num_count_readout_train_examples < 1:
            raise ValueError(
                "num_count_readout_train_examples must be >= 1, got "
                f"{self.num_count_readout_train_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def oracle_latent_operator_config_from_dict(raw: dict[str, Any]) -> OracleLatentOperatorConfig:
    """Parse a `configs/phase_a1_oracle_latent_operator_benchmark.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = OracleLatentOperatorConfig()
    core_train_raw = raw.get("core_train")
    core_train = (
        dataclasses.replace(defaults.core_train, **core_train_raw)
        if core_train_raw
        else defaults.core_train
    )
    count_readout_train_raw = raw.get("count_readout_train")
    count_readout_train = (
        dataclasses.replace(defaults.count_readout_train, **count_readout_train_raw)
        if count_readout_train_raw
        else defaults.count_readout_train
    )
    return OracleLatentOperatorConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        count_readout_train=count_readout_train,
        num_count_readout_train_examples=raw.get(
            "num_count_readout_train_examples", defaults.num_count_readout_train_examples
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind on one operation's ordinary i.i.d.
    stream, then frozen: every parameter has `requires_grad=False`. Carries
    its own `TaskGenerator` (module docstring, "Four checkpoints, not one")
    so eval/readout-train batches are drawn from the identical stream the
    core itself was pretrained on."""

    operation: str
    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    generator: TaskGenerator
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: OracleLatentOperatorConfig, operation: str, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    """Pretrain one `DecoderOnlyTransformer` via `train_shared_core` with no
    task segment ever visible (`include_task_spec=False`), restricted to a
    single operation, then freeze every parameter -- see module docstring's
    "Four checkpoints, not one". Deliberately its own copy rather than a
    cross-import of `apc.evaluation.representation_audit`'s identically-named
    helper, matching every Phase A.1 gate module's "independently reviewable
    end to end" convention (see e.g. `apc.evaluation.count_counterfactual_
    gate`'s own `_pretrain_frozen_stable_core`)."""
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


def _shift_source_positions(length: int, amount: int) -> list[int]:
    """`ShiftOp.apply`'s own rotation (`sequence[amount:] + sequence[:amount]`)
    read as a source-position mapping: `target[t] = input[(amount + t) %
    length]`."""
    amount = amount % length
    return [(amount + t) % length for t in range(length)]


def _select_source_positions(indices: Sequence[int]) -> list[int]:
    """`SelectOp.apply`'s own gather (`sequence[i] for i in indices`) read as
    a source-position mapping: `target[t] = input[indices[t]]`."""
    return list(indices)


def _bind_value_position(input_tokens: tuple[int, ...], query_key: int) -> int:
    """`BindOp.apply`'s own "last matching pair wins" search
    (`apc.environments.operations.BindOp.apply`), read as a single
    source-position mapping: the content position of the *value* paired
    with the last (highest-index) key position equal to `query_key`."""
    key_position: int | None = None
    for i in range(0, len(input_tokens) - 1, 2):
        if input_tokens[i] == query_key:
            key_position = i
    if key_position is None:
        raise ValueError(
            f"query_key {query_key} not found among BIND key positions in "
            f"{input_tokens!r} -- BindOp.sample_params guarantees a present "
            "key, so this indicates a mismatched oracle call"
        )
    return key_position + 1


def _count_match_positions(input_tokens: tuple[int, ...], target: int) -> list[int]:
    """`CountOp.apply`'s own match predicate (`token == target`), read as
    the set of content positions the oracle readout pools over."""
    return [i for i, token in enumerate(input_tokens) if token == target]


def _source_positions_for_example(
    operation: str, example: Example, call: PrimitiveCall
) -> list[int]:
    """The oracle content-position mapping for one `SHIFT`/`SELECT`/`BIND`
    example: `result[t]` is the content position whose `model.decode`
    prediction should equal `example.target_tokens[t]` (module docstring,
    "Zero new training for SHIFT/SELECT/BIND")."""
    length = len(example.input_tokens)
    if operation == "SHIFT":
        return _shift_source_positions(length, call.arguments["amount"])
    if operation == "SELECT":
        return _select_source_positions(call.arguments["indices"])
    if operation == "BIND":
        return [_bind_value_position(example.input_tokens, call.arguments["query_key"])]
    raise ValueError(
        f"_source_positions_for_example does not support operation {operation!r} "
        "-- COUNT uses the dedicated trained-readout path instead"
    )


def _oracle_decode_predictions(
    model: DecoderOnlyTransformer,
    content_state: torch.Tensor,
    row_source_positions: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Gather `content_state[row, position, :]` for every `(row, position)`
    pair implied by `row_source_positions` in one batched call, run the
    frozen `model.decode` once over all of them, and regroup the argmax
    predictions back per row -- the batched form of "permute hidden-state
    positions by oracle addressing, then decode" for `SHIFT`/`SELECT`/`BIND`.
    No gradient is tracked and no parameter is fit anywhere in this
    function."""
    flat_rows: list[int] = []
    flat_positions: list[int] = []
    lengths: list[int] = []
    for row, positions in enumerate(row_source_positions):
        lengths.append(len(positions))
        flat_rows.extend([row] * len(positions))
        flat_positions.extend(positions)

    if not flat_rows:
        return [[] for _ in row_source_positions]

    row_tensor = torch.tensor(flat_rows, dtype=torch.long, device=content_state.device)
    position_tensor = torch.tensor(flat_positions, dtype=torch.long, device=content_state.device)
    with torch.no_grad():
        gathered = content_state[row_tensor, position_tensor, :]
        logits = model.decode(gathered)
        predictions = logits.argmax(dim=-1).tolist()

    result: list[list[int]] = []
    cursor = 0
    for length in lengths:
        result.append(predictions[cursor : cursor + length])
        cursor += length
    return result


def _evaluate_oracle_decode(
    model: DecoderOnlyTransformer,
    tokens: SharedCoreTokens,
    examples: Sequence[Example],
    operation: str,
    *,
    device: torch.device | str,
) -> tuple[float, float]:
    """Run the zero-training oracle latent operator (`SHIFT`/`SELECT`/`BIND`)
    over `examples` in one batched forward pass and return `(exact_match,
    token_accuracy)`."""
    with torch.no_grad():
        content_state = model.encode(collate_content_only_batch(examples, tokens, device=device))

    calls = [oracle_call_for_example(example) for example in examples]
    row_source_positions = [
        _source_positions_for_example(operation, example, call)
        for example, call in zip(examples, calls, strict=True)
    ]
    predictions = _oracle_decode_predictions(model, content_state, row_source_positions)

    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    for prediction, example in zip(predictions, examples, strict=True):
        target = example.target_tokens
        total_tokens += len(target)
        correct_tokens += sum(
            1 for p, t in zip(prediction, target, strict=True) if p == t
        )
        if tuple(prediction) == target:
            exact_matches += 1

    return exact_matches / len(examples), correct_tokens / total_tokens


def _count_summary_feature(
    content_state_row: torch.Tensor, match_positions: Sequence[int]
) -> torch.Tensor:
    """Sum-pool `content_state_row` (one example's `[seq_len, d_model]`
    per-position states) at each 0-indexed content position in
    `match_positions`, using the "state at the token itself" offset
    (`content_state_row[1 + i]`, module docstring's "Per-operation
    matched-position feature convention"). Returns the zero vector when
    `match_positions` is empty (module docstring, "Sum-pool, not
    mean-pool")."""
    if not match_positions:
        return torch.zeros(content_state_row.shape[-1], device=content_state_row.device)
    gathered = content_state_row[[1 + i for i in match_positions], :]
    return gathered.sum(dim=0)


def _fit_and_evaluate_count_readout(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    num_classes: int,
    train_config: ProbeTrainConfig,
    device: torch.device,
) -> float:
    """Fit one `nn.Linear(d_model, num_classes)` by full-batch AdamW
    cross-entropy on `(train_features, train_labels)`; return exact-match
    accuracy on the disjoint `(eval_features, eval_labels)`. Same shape as
    `apc.evaluation.representation_audit._fit_classification_probe`, kept as
    its own copy per this codebase's "each gate module is independently
    reviewable end to end" convention."""
    d_model = train_features.shape[-1]
    readout = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        readout.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    readout.train()
    for _ in range(train_config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(readout(train_features), train_labels)
        loss.backward()
        optimizer.step()
    readout.eval()
    with torch.no_grad():
        predictions = readout(eval_features).argmax(dim=-1)
        return (predictions == eval_labels).to(torch.float32).mean().item()


def _run_count_oracle_benchmark(
    config: OracleLatentOperatorConfig,
    core: FrozenStableCore,
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
) -> float:
    """Oracle-locate match positions from raw symbols, sum-pool their
    `h_content`, and fit/evaluate the small diagnostic readout described in
    the module docstring's "COUNT is different in kind"."""
    device = core.device
    with torch.no_grad():
        train_content_state = core.model.encode(
            collate_content_only_batch(train_examples, core.tokens, device=device)
        )
        eval_content_state = core.model.encode(
            collate_content_only_batch(eval_examples, core.tokens, device=device)
        )

    def _features_and_labels(
        examples: Sequence[Example], content_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls = [oracle_call_for_example(example) for example in examples]
        features = torch.stack(
            [
                _count_summary_feature(
                    content_state[row],
                    _count_match_positions(example.input_tokens, call.arguments["target"]),
                )
                for row, (example, call) in enumerate(zip(examples, calls, strict=True))
            ]
        )
        labels = torch.tensor(
            [example.target_tokens[0] for example in examples], dtype=torch.long, device=device
        )
        return features, labels

    train_features, train_labels = _features_and_labels(train_examples, train_content_state)
    eval_features, eval_labels = _features_and_labels(eval_examples, eval_content_state)

    # Re-anchor the shared global RNG stream before fitting the readout
    # (ADR-0016 pattern), matching apc.evaluation.representation_audit.
    set_seed(config.seed)

    return _fit_and_evaluate_count_readout(
        train_features,
        train_labels,
        eval_features,
        eval_labels,
        config.vocab_size,
        config.count_readout_train,
        device,
    )


@dataclass(frozen=True)
class OperationOracleLatentOperatorReport:
    """Everything observed while pretraining, freezing, and benchmarking one
    operation's frozen Stable Core for one seed."""

    operation: str
    core_final_train_loss: float
    core_param_count: int
    core_trainable_param_count: int
    num_eval_examples: int
    exact_match: float
    token_accuracy: float
    exact_match_passed: bool
    token_accuracy_passed: bool | None
    passed: bool
    readout_trained: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def _run_operation_benchmark(
    config: OracleLatentOperatorConfig,
    operation: str,
    *,
    core_metrics_path: str | Path | None = None,
) -> OperationOracleLatentOperatorReport:
    """Run the full A1-R005E-003 benchmark for one `(config.seed,
    operation)` pair: pretrain and freeze that operation's dedicated Stable
    Core, then evaluate its oracle latent operator (module docstring)."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, operation, metrics_path=core_metrics_path)

    readout_trained = operation in COUNT_READOUT_OPERATIONS
    if readout_trained:
        train_examples = core.generator.generate_online(
            config.num_count_readout_train_examples, step=10_000, split="test"
        )
        eval_examples = core.generator.generate_online(
            config.num_eval_examples, step=20_000, split="test"
        )
        exact_match = _run_count_oracle_benchmark(config, core, train_examples, eval_examples)
        # CountOp.output_length is always 1, so token accuracy and exact
        # match coincide by construction -- see module docstring.
        token_accuracy = exact_match
    else:
        eval_examples = core.generator.generate_online(
            config.num_eval_examples, step=20_000, split="test"
        )
        exact_match, token_accuracy = _evaluate_oracle_decode(
            core.model, core.tokens, eval_examples, operation, device=core.device
        )

    exact_match_passed = exact_match >= EXACT_MATCH_THRESHOLD
    token_accuracy_passed = (
        token_accuracy >= TOKEN_ACCURACY_THRESHOLD
        if operation in TOKEN_ACCURACY_GATED_OPERATIONS
        else None
    )
    passed = exact_match_passed and (
        token_accuracy_passed if token_accuracy_passed is not None else True
    )

    return OperationOracleLatentOperatorReport(
        operation=operation,
        core_final_train_loss=core.final_core_train_loss,
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        num_eval_examples=len(eval_examples),
        exact_match=exact_match,
        token_accuracy=token_accuracy,
        exact_match_passed=exact_match_passed,
        token_accuracy_passed=token_accuracy_passed,
        passed=passed,
        readout_trained=readout_trained,
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


@dataclass(frozen=True)
class OracleLatentOperatorReport:
    """Everything observed for one seed, across every operation in
    `config.operation_names`."""

    config: OracleLatentOperatorConfig
    per_operation: dict[str, OperationOracleLatentOperatorReport]
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


def run_oracle_latent_operator_benchmark(
    config: OracleLatentOperatorConfig,
    *,
    core_metrics_dir: str | Path | None = None,
) -> OracleLatentOperatorReport:
    """Run one seed of the A1-R005E-003 benchmark: for every operation in
    `config.operation_names`, pretrain and freeze its own dedicated Stable
    Core and evaluate its oracle latent operator."""
    start = time.perf_counter()
    per_operation: dict[str, OperationOracleLatentOperatorReport] = {}
    device_str = "cpu"
    metrics_dir_path = Path(core_metrics_dir) if core_metrics_dir is not None else None
    for operation in config.operation_names:
        metrics_path = (
            metrics_dir_path / f"{operation}_core_metrics.jsonl" if metrics_dir_path else None
        )
        report = _run_operation_benchmark(config, operation, core_metrics_path=metrics_path)
        per_operation[operation] = report
        device_str = report.device
    return OracleLatentOperatorReport(
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
class OperationOracleLatentOperatorSummary:
    """One operation's exact-match/token-accuracy means aggregated across
    seeds, with pass/fail verdicts computed on the cross-seed mean (`docs/
    DECISIONS.md` ADR-0033's precedent)."""

    operation: str
    mean_exact_match: float
    stdev_exact_match: float
    min_exact_match: float
    max_exact_match: float
    mean_token_accuracy: float
    stdev_token_accuracy: float
    exact_match_passed: bool
    token_accuracy_passed: bool | None
    passed: bool
    readout_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class OracleLatentOperatorMultiSeedReport:
    """A1-R005E-003's full result across seeds, with an overall `passed`
    verdict (all operations in `seeds`' shared `operation_names` pass their
    own targets) -- unlike A1-R005E-002, this task states explicit numeric
    acceptance targets (module docstring)."""

    seeds: tuple[int, ...]
    per_seed: tuple[OracleLatentOperatorReport, ...]
    per_operation_summary: dict[str, OperationOracleLatentOperatorSummary]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "per_operation_summary": {
                name: summary.to_dict() for name, summary in self.per_operation_summary.items()
            },
            "passed": self.passed,
        }


def run_oracle_latent_operator_benchmark_multi_seed(
    base_config: OracleLatentOperatorConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> OracleLatentOperatorMultiSeedReport:
    """Run one benchmark seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation."""
    per_seed: list[OracleLatentOperatorReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        report = run_oracle_latent_operator_benchmark(config, core_metrics_dir=seed_dir)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    per_operation_summary: dict[str, OperationOracleLatentOperatorSummary] = {}
    for operation in base_config.operation_names:
        op_reports = [report.per_operation[operation] for report in per_seed]
        exact_mean, exact_stdev, exact_min, exact_max = _summarize(
            [r.exact_match for r in op_reports]
        )
        token_mean, token_stdev, _, _ = _summarize([r.token_accuracy for r in op_reports])
        exact_match_passed = exact_mean >= EXACT_MATCH_THRESHOLD
        token_accuracy_passed = (
            token_mean >= TOKEN_ACCURACY_THRESHOLD
            if operation in TOKEN_ACCURACY_GATED_OPERATIONS
            else None
        )
        passed = exact_match_passed and (
            token_accuracy_passed if token_accuracy_passed is not None else True
        )
        per_operation_summary[operation] = OperationOracleLatentOperatorSummary(
            operation=operation,
            mean_exact_match=exact_mean,
            stdev_exact_match=exact_stdev,
            min_exact_match=exact_min,
            max_exact_match=exact_max,
            mean_token_accuracy=token_mean,
            stdev_token_accuracy=token_stdev,
            exact_match_passed=exact_match_passed,
            token_accuracy_passed=token_accuracy_passed,
            passed=passed,
            readout_trained=operation in COUNT_READOUT_OPERATIONS,
        )

    overall_passed = all(summary.passed for summary in per_operation_summary.values())

    return OracleLatentOperatorMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
        passed=overall_passed,
    )
