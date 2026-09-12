"""Unified Oracle Causal Benchmark (Phase A.1 Post-Diagnostic Task A1-B002, STOP GATE).

Evaluates all 8 canonical operations (4 parameter-free: COPY, REVERSE, SORT, NEGATE;
4 parameterized: SELECT, COUNT, BIND, SHIFT) using oracle selection over a single
frozen shared task-blind Stable Core ($h_{\\text{content}} = f(\\text{content})$).

Acceptance Criteria (ADR-0046 / CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md):
- Multi-seed benchmark (5 seeds: 0, 1, 2, 3, 4) on unseen evaluations.
- Overall Correct exact match >= 0.90.
- Parameter-free operations Correct exact match >= 0.95.
- Parameterized operations:
  - SELECT, COUNT, BIND, SHIFT Correct exact match >= 0.90.
  - Causal gap >= 0.50.
  - None arm <= B_natural + 0.05.
- Task-blind maximum absolute difference <= 10^-5.
- Strict sparse execution: unselected primitives receive zero forward calls.
- STOP GATE.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    PARAMETERIZED_OPERATION_NAMES,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    MIN_GROUP_SIZE,
    _correct_argument_value,
    _default_model_config,
    _flatten_groups,
    _labels_for_examples,
    generate_compact_operator_counterfactual_groups,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedContentEncoder,
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
    encode_content,
)
from apc.evaluation.shared_encoder_mixed_operation_training_gate import (
    SharedMixedTrainingConfig,
    _train_shared,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.conditioning import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
)
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveBase,
    PrimitiveStatus,
    ReverseRelativePrimitive,
    ReverseRelativePrimitiveConfig,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
)
from apc.utils.seed import set_seed

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5
TASK_BLIND_ATOL = 1e-5

PARAMETER_FREE_OPERATIONS: Final[tuple[str, ...]] = ("COPY", "REVERSE", "SORT", "NEGATE")
# Canonical parameterized operations: SELECT, COUNT, BIND, SHIFT
PARAMETERIZED_OPERATIONS: Final[tuple[str, ...]] = PARAMETERIZED_OPERATION_NAMES
ALL_CANONICAL_OPERATIONS: Final[tuple[str, ...]] = (
    *PARAMETERIZED_OPERATIONS,
    *PARAMETER_FREE_OPERATIONS,
)

# Natural base rate ceilings (ADR-0044, ADR-0046: None <= B_natural + 0.05)
# Note: For SHIFT, unconditioned execution defaults to shift=0; in cyclic shift
# with sequence length L in [6, 10], P(true_shift = 0) = 1/L ~ 0.125 - 0.167.
NATURAL_BASELINES: Final[dict[str, float]] = {
    "SELECT": 0.05,
    "COUNT": 0.35,
    "BIND": 0.35,
    "SHIFT": 0.15,
    "COPY": 0.01,
    "REVERSE": 0.01,
    "SORT": 0.01,
    "NEGATE": 0.01,
}

# Fixed derangements for Wrong Family testing
WRONG_FAMILY_MAP: Final[dict[str, str]] = {
    # Parameterized derangement
    "SELECT": "COUNT",
    "COUNT": "BIND",
    "BIND": "SHIFT",
    "SHIFT": "SELECT",
    # Parameter-free derangement
    "COPY": "REVERSE",
    "REVERSE": "SORT",
    "SORT": "NEGATE",
    "NEGATE": "COPY",
}

DEFAULT_WRONG_FAMILY_ARGUMENTS: Final[dict[str, Any]] = {
    "SELECT": (0,),
    "COUNT": 0,
    "BIND": 0,
    "SHIFT": 0,
    "COPY": None,
    "REVERSE": None,
    "SORT": None,
    "NEGATE": None,
}

_EVAL_BATCH_SIZE = 128


def _derive_local_seed(master_seed: int, step: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class UnifiedBenchmarkConfig:
    """Explicit configuration for Task A1-B002 Unified Oracle Causal Benchmark."""

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    device: str = "auto"

    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
    arg_dim: int = 16
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH

    # Training steps for primitives over frozen core
    parameterized_train_steps: int = 12000
    parameter_free_train_steps: int = 4000
    operator_lr: float = 0.0008
    operator_weight_decay: float = 0.0001
    operator_grad_clip: float = 1.0

    # Core pretraining fallback (if checkpoint missing)
    core_train_steps: int = 16000
    core_lr: float = 0.0003
    core_weight_decay: float = 0.0001
    shared_encoder_checkpoint: str | None = None

    num_unseen_eval_groups: int = 1400  # For parameterized (4,200 examples)
    num_unseen_eval_examples: int = 1024  # For parameter-free
    min_unseen_eval_examples: int = 1024

    def __post_init__(self) -> None:
        if self.group_size < MIN_GROUP_SIZE:
            raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {self.group_size}")
        if self.d_operator < 1:
            raise ValueError(f"d_operator must be >= 1, got {self.d_operator}")
        if self.n_operator_head < 1:
            raise ValueError(f"n_operator_head must be >= 1, got {self.n_operator_head}")
        if self.d_operator % self.n_operator_head != 0:
            raise ValueError(
                f"d_operator ({self.d_operator}) must be divisible by n_operator_head "
                f"({self.n_operator_head})"
            )
        if self.num_unseen_eval_examples < self.min_unseen_eval_examples:
            raise ValueError("num_unseen_eval_examples below minimum")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def unified_benchmark_config_from_dict(raw: dict[str, Any]) -> UnifiedBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = UnifiedBenchmarkConfig()
    return UnifiedBenchmarkConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        device=raw.get("device", defaults.device),
        d_operator=raw.get("d_operator", defaults.d_operator),
        n_operator_head=raw.get("n_operator_head", defaults.n_operator_head),
        d_operator_ff=raw.get("d_operator_ff", defaults.d_operator_ff),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        parameterized_train_steps=raw.get(
            "parameterized_train_steps", defaults.parameterized_train_steps
        ),
        parameter_free_train_steps=raw.get(
            "parameter_free_train_steps", defaults.parameter_free_train_steps
        ),
        operator_lr=raw.get("operator_lr", defaults.operator_lr),
        operator_weight_decay=raw.get("operator_weight_decay", defaults.operator_weight_decay),
        operator_grad_clip=raw.get("operator_grad_clip", defaults.operator_grad_clip),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        core_lr=raw.get("core_lr", defaults.core_lr),
        core_weight_decay=raw.get("core_weight_decay", defaults.core_weight_decay),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
        num_unseen_eval_groups=raw.get(
            "num_unseen_eval_groups", defaults.num_unseen_eval_groups
        ),
        num_unseen_eval_examples=raw.get(
            "num_unseen_eval_examples", defaults.num_unseen_eval_examples
        ),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


def _get_or_train_frozen_shared_core(
    config: UnifiedBenchmarkConfig, seed_dir: Path | None = None
) -> SharedContentEncoder:
    """Obtain or pretrain the frozen shared task-blind encoder for this seed."""
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=PARAMETERIZED_OPERATION_NAMES,
        group_size=config.group_size,
        model=config.model,
        device=config.device,
        d_operator=config.d_operator,
        n_operator_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        arg_dim=config.arg_dim,
        max_sequence_length=config.max_sequence_length,
    )

    ckpt_path: Path | None = None
    if config.shared_encoder_checkpoint:
        cand = Path(config.shared_encoder_checkpoint)
        if cand.is_file():
            ckpt_path = cand
    if ckpt_path is None and seed_dir is not None:
        cand = seed_dir / "shared_encoder.pt"
        if cand.is_file():
            ckpt_path = cand
    if ckpt_path is None:
        cand = (
            Path("runs")
            / "phase_a1_shift_compact_structural_probe"
            / f"seed_{config.seed}"
            / "shared_encoder.pt"
        )
        if cand.is_file():
            ckpt_path = cand

    core: SharedContentEncoder | None = None
    if ckpt_path is not None:
        try:
            arch = build_shared_encoder_architecture(arch_cfg)
            state_dict = torch.load(ckpt_path, map_location=arch.core.device, weights_only=True)
            arch.core.model.load_state_dict(state_dict)
            core = arch.core
        except RuntimeError:
            core = None

    if core is None:
        mixed_cfg = SharedMixedTrainingConfig(
            seed=config.seed,
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
            operation_names=PARAMETERIZED_OPERATION_NAMES,
            group_size=config.group_size,
            model=config.model,
            device=config.device,
            d_operator=config.d_operator,
            n_operator_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            arg_dim=config.arg_dim,
            max_sequence_length=config.max_sequence_length,
            joint_train=type(SharedMixedTrainingConfig().joint_train)(
                steps=config.core_train_steps,
                lr=config.core_lr,
                weight_decay=config.core_weight_decay,
            ),
        )
        arch = build_shared_encoder_architecture(mixed_cfg.to_architecture_config())
        _train_shared(arch, mixed_cfg)
        core = arch.core
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(core.model.state_dict(), seed_dir / "shared_encoder.pt")

    core.model.eval()
    for param in core.model.parameters():
        param.requires_grad_(False)

    return core


def _generate_parameter_free_examples(
    seed: int,
    n: int,
    *,
    operation: str,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
) -> list[Example]:
    """Deterministically generate parameter-free examples."""
    import random

    rng = random.Random(_derive_local_seed(seed, 0, f"{split}:{operation}"))
    op = get_operation(operation)
    examples: list[Example] = []
    for _ in range(n):
        seq_len = rng.randint(sequence_length_range[0], sequence_length_range[1])
        seq = tuple(rng.randrange(vocab_size) for _ in range(seq_len))
        params = op.sample_params(rng, seq, vocab_size)
        step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(step,))
        res = run_program(prog, seq, vocab_size)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="known",
            split=split,
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(label="K", primitive_operations=(operation,)),
        )
        examples.append(ex)
    return examples


def _build_heterogeneous_bank(
    config: UnifiedBenchmarkConfig,
) -> tuple[PrimitiveBank, dict[str, int]]:
    """Construct a PrimitiveBank populated with all 8 canonical primitives."""
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}

    # 1. SELECT (CrossPositionPrimitive)
    p_select = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SELECT",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["SELECT"] = p_select.primitive_id

    # 2. COUNT (CrossPositionPrimitive)
    p_count = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="COUNT",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["COUNT"] = p_count.primitive_id

    # 3. BIND (CrossPositionPrimitive)
    p_bind = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="BIND",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["BIND"] = p_bind.primitive_id

    # 4. SHIFT (ShiftRelativePrimitive)
    p_shift = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["SHIFT"] = p_shift.primitive_id

    # 5. COPY (CrossPositionPrimitive, parameter-free)
    p_copy = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="COPY",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["COPY"] = p_copy.primitive_id

    # 6. REVERSE (ReverseRelativePrimitive)
    p_reverse = bank.new_reverse_relative_primitive(
        ReverseRelativePrimitiveConfig(
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["REVERSE"] = p_reverse.primitive_id

    # 7. SORT (CrossPositionPrimitive, parameter-free)
    p_sort = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SORT",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["SORT"] = p_sort.primitive_id

    # 8. NEGATE (CrossPositionPrimitive, parameter-free)
    p_negate = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="NEGATE",
            d_model=config.model["d_model"],
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        ),
        status=PrimitiveStatus.STABLE,
    )
    op_to_id["NEGATE"] = p_negate.primitive_id

    return bank, op_to_id


def _train_single_primitive(
    core: SharedContentEncoder,
    primitive: PrimitiveBase,
    config: UnifiedBenchmarkConfig,
    operation: str,
    *,
    steps: int,
) -> float:
    """Train a single primitive on top of the frozen shared core."""
    import random

    device = core.device
    primitive.train()
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=config.operator_lr,
        weight_decay=config.operator_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    final_loss = 0.0
    is_parameterized = operation in PARAMETERIZED_OPERATIONS
    groups_per_step = max(1, -(-32 // config.group_size))

    for step in range(1, steps + 1):
        if is_parameterized:
            groups = generate_compact_operator_counterfactual_groups(
                config.seed,
                groups_per_step,
                operation=operation,
                step=step,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, _, _ = _flatten_groups(groups)
            argument_values: Sequence[Any] | None = [
                _correct_argument_value(operation, ex) for ex in examples
            ]
        else:
            step_rng = random.Random(_derive_local_seed(config.seed, step, f"train:{operation}"))
            op_obj = get_operation(operation)
            examples = []
            for _ in range(32):
                seq_len = step_rng.randint(
                    config.sequence_length_range[0], config.sequence_length_range[1]
                )
                seq = tuple(step_rng.randrange(config.vocab_size) for _ in range(seq_len))
                params = op_obj.sample_params(step_rng, seq, config.vocab_size)
                p_step = ProgramStep(operation=operation, params=params)
                prog = Program(steps=(p_step,))
                res = run_program(prog, seq, config.vocab_size)
                examples.append(
                    Example(
                        input_tokens=seq,
                        target_tokens=res.output_tokens,
                        program=prog,
                        operation_graph=res.graph,
                        category="known",
                        split="train",
                        vocab_size=config.vocab_size,
                        task_spec=TaskSpec.from_program(prog),
                        oracle_metadata=OracleMetadata(
                            label="K", primitive_operations=(operation,)
                        ),
                    )
                )
            argument_values = None

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(operation).output_length(length) for length in content_lengths
        ]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        optimizer.zero_grad(set_to_none=True)
        if isinstance(
            primitive,
            (
                CrossPositionPrimitive,
                ShiftRelativePrimitive,
                ReverseRelativePrimitive,
                ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
            ),
        ):
            logits = primitive(h, content_lengths, output_lengths, argument_values)
        else:
            logits = primitive(h)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        if config.operator_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(primitive.parameters(), config.operator_grad_clip)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.item())

    primitive.eval()
    return final_loss


def _make_arg_provider_correct(target_op: str) -> Callable[[Example], Any]:
    def _provider(ex: Example) -> Any:
        return _correct_argument_value(target_op, ex)

    return _provider


def _make_arg_provider_map(arg_map: dict[int, Any]) -> Callable[[Example], Any]:
    def _provider(ex: Example) -> Any:
        return arg_map[id(ex)]

    return _provider


def _make_arg_provider_const(val: Any) -> Callable[[Example], Any]:
    def _provider(ex: Example) -> Any:
        return val

    return _provider


def _evaluate_primitive_arm(
    core: SharedContentEncoder,
    primitive: PrimitiveBase,
    examples: Sequence[Example],
    eval_operation: str,
    *,
    argument_provider: Callable[[Example], Any] | None = None,
    primitive_operation: str | None = None,
) -> tuple[float, float]:
    """Evaluate exact match and token accuracy for one arm."""
    primitive.eval()
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    device = core.device
    prim_op: str = str(
        primitive_operation or getattr(primitive, "operation", eval_operation) or eval_operation
    )

    with torch.no_grad():
        for start in range(0, len(examples), _EVAL_BATCH_SIZE):
            chunk = examples[start : start + _EVAL_BATCH_SIZE]
            content_lengths = [len(example.input_tokens) for example in chunk]
            output_lengths = [
                get_operation(prim_op).output_length(length) for length in content_lengths
            ]
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

            argument_values = (
                None
                if argument_provider is None
                else [argument_provider(example) for example in chunk]
            )
            if isinstance(
                primitive,
                (
                    CrossPositionPrimitive,
                    ShiftRelativePrimitive,
                    ReverseRelativePrimitive,
                    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
                ),
            ):
                logits = primitive(h, content_lengths, output_lengths, argument_values)
            else:
                logits = primitive(h)

            predictions = logits.argmax(dim=-1)
            for row, (example, n) in enumerate(zip(chunk, output_lengths, strict=True)):
                target = example.target_tokens
                prediction = tuple(predictions[row, :n].tolist())
                total_tokens += len(target)
                min_len = min(len(prediction), len(target))
                correct_tokens += sum(
                    1
                    for p, t in zip(prediction[:min_len], target[:min_len], strict=True)
                    if p == t
                )
                if len(prediction) == len(target) and prediction == target:
                    exact_matches += 1

    return exact_matches / len(examples), correct_tokens / total_tokens


@dataclass(frozen=True)
class OperationBenchmarkResult:
    """Benchmark metrics for a single operation."""

    operation: str
    is_parameterized: bool
    primitive_id: int
    primitive_class: str
    parameter_count: int
    final_loss: float
    correct_exact_match: float
    correct_token_accuracy: float
    wrong_argument_exact_match: float | None
    wrong_family_exact_match: float
    none_exact_match: float
    exact_match_causal_gap: float
    natural_baseline: float
    none_passed: bool
    correct_passed: bool
    causal_gap_passed: bool
    passed: bool


@dataclass(frozen=True)
class UnifiedBenchmarkReport:
    """Report for one seed of Task A1-B002."""

    config: UnifiedBenchmarkConfig
    seed: int
    core_param_count: int
    total_bank_params: int
    results_by_operation: dict[str, OperationBenchmarkResult]
    mean_correct_exact_match: float
    parameter_free_correct_exact_match: float
    parameterized_correct_exact_match: float
    task_blind_max_abs_diff: float
    task_blind_passed: bool
    sparse_execution_passed: bool
    all_operations_passed: bool
    passed: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "total_bank_params": self.total_bank_params,
            "results_by_operation": {
                k: dataclasses.asdict(v) for k, v in self.results_by_operation.items()
            },
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "parameter_free_correct_exact_match": self.parameter_free_correct_exact_match,
            "parameterized_correct_exact_match": self.parameterized_correct_exact_match,
            "task_blind_max_abs_diff": self.task_blind_max_abs_diff,
            "task_blind_passed": self.task_blind_passed,
            "sparse_execution_passed": self.sparse_execution_passed,
            "all_operations_passed": self.all_operations_passed,
            "passed": self.passed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_unified_oracle_causal_benchmark(
    config: UnifiedBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> UnifiedBenchmarkReport:
    """Run one seed of Task A1-B002."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    # 1. Obtain frozen shared Core
    core = _get_or_train_frozen_shared_core(config, seed_dir=seed_dir)
    device = str(core.device)

    # 2. Build PrimitiveBank
    set_seed(config.seed)
    bank, op_to_id = _build_heterogeneous_bank(config)
    bank.to(core.device)

    # 3. Train each primitive over the frozen core
    losses: dict[str, float] = {}
    for op in ALL_CANONICAL_OPERATIONS:
        pid = op_to_id[op]
        prim = bank.get(pid)
        steps = (
            config.parameterized_train_steps
            if op in PARAMETERIZED_OPERATIONS
            else config.parameter_free_train_steps
        )
        loss = _train_single_primitive(core, prim, config, op, steps=steps)
        losses[op] = loss

    # 4. Freeze bank
    bank.freeze_all()

    # 5. Evaluate all 8 operations
    results: dict[str, OperationBenchmarkResult] = {}
    for op in ALL_CANONICAL_OPERATIONS:
        pid = op_to_id[op]
        prim = bank.get(pid)
        is_param = op in PARAMETERIZED_OPERATIONS
        wrong_op = WRONG_FAMILY_MAP[op]
        wrong_pid = op_to_id[wrong_op]
        wrong_prim = bank.get(wrong_pid)
        b_natural = NATURAL_BASELINES[op]

        if is_param:
            groups = generate_compact_operator_counterfactual_groups(
                config.seed,
                config.num_unseen_eval_groups,
                operation=op,
                step=0,
                split="test",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, wrong_arg_map, _ = _flatten_groups(groups)

            # Correct Arm
            corr_exact, corr_tok = _evaluate_primitive_arm(
                core,
                prim,
                examples,
                op,
                argument_provider=_make_arg_provider_correct(op),
            )
            # Wrong Argument Arm
            wr_arg_exact, _ = _evaluate_primitive_arm(
                core,
                prim,
                examples,
                op,
                argument_provider=_make_arg_provider_map(wrong_arg_map),
            )
            # Wrong Family Arm
            wr_op = WRONG_FAMILY_MAP[op]
            wr_fam_exact, _ = _evaluate_primitive_arm(
                core,
                wrong_prim,
                examples,
                op,
                argument_provider=_make_arg_provider_const(DEFAULT_WRONG_FAMILY_ARGUMENTS[wr_op]),
                primitive_operation=wr_op,
            )
            # None Arm
            none_exact, _ = _evaluate_primitive_arm(
                core, prim, examples, op, argument_provider=None
            )

            causal_gap = corr_exact - max(wr_arg_exact, wr_fam_exact, none_exact)
            correct_passed = corr_exact >= 0.90
            causal_gap_passed = causal_gap >= 0.50
            none_passed = none_exact <= (b_natural + 0.05)
            op_passed = correct_passed and causal_gap_passed and none_passed

            results[op] = OperationBenchmarkResult(
                operation=op,
                is_parameterized=True,
                primitive_id=pid,
                primitive_class=prim.__class__.__name__,
                parameter_count=prim.num_parameters(),
                final_loss=losses[op],
                correct_exact_match=corr_exact,
                correct_token_accuracy=corr_tok,
                wrong_argument_exact_match=wr_arg_exact,
                wrong_family_exact_match=wr_fam_exact,
                none_exact_match=none_exact,
                exact_match_causal_gap=causal_gap,
                natural_baseline=b_natural,
                none_passed=none_passed,
                correct_passed=correct_passed,
                causal_gap_passed=causal_gap_passed,
                passed=op_passed,
            )
            print(
                f"  [Seed {config.seed}] {op:<7s}: Correct={corr_exact:.4f} (Tok={corr_tok:.4f}), "
                f"WrArg={wr_arg_exact:.4f}, WrFam={wr_fam_exact:.4f}, None={none_exact:.4f}, "
                f"Gap={causal_gap:.4f} -> {'PASS' if op_passed else 'FAIL'}"
            )
        else:
            examples = _generate_parameter_free_examples(
                config.seed,
                config.num_unseen_eval_examples,
                operation=op,
                split="test",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            # Correct Arm
            corr_exact, corr_tok = _evaluate_primitive_arm(
                core, prim, examples, op, argument_provider=None
            )
            # Wrong Family Arm
            wr_op = WRONG_FAMILY_MAP[op]
            wr_fam_exact, _ = _evaluate_primitive_arm(
                core,
                wrong_prim,
                examples,
                op,
                argument_provider=None,
                primitive_operation=wr_op,
            )
            # None Arm: evaluate unconditioned / identity baseline (disabled primitive)
            prim.enabled = False
            none_exact, _ = _evaluate_primitive_arm(
                core, prim, examples, op, argument_provider=None
            )
            prim.enabled = True

            causal_gap = corr_exact - max(wr_fam_exact, none_exact)
            correct_passed = corr_exact >= 0.95
            causal_gap_passed = causal_gap >= 0.50
            none_passed = none_exact <= (b_natural + 0.05)
            op_passed = correct_passed and causal_gap_passed and none_passed

            results[op] = OperationBenchmarkResult(
                operation=op,
                is_parameterized=False,
                primitive_id=pid,
                primitive_class=prim.__class__.__name__,
                parameter_count=prim.num_parameters(),
                final_loss=losses[op],
                correct_exact_match=corr_exact,
                correct_token_accuracy=corr_tok,
                wrong_argument_exact_match=None,
                wrong_family_exact_match=wr_fam_exact,
                none_exact_match=none_exact,
                exact_match_causal_gap=causal_gap,
                natural_baseline=b_natural,
                none_passed=none_passed,
                correct_passed=correct_passed,
                causal_gap_passed=causal_gap_passed,
                passed=op_passed,
            )
            print(
                f"  [Seed {config.seed}] {op:<7s}: Correct={corr_exact:.4f} (Tok={corr_tok:.4f}), "
                f"WrFam={wr_fam_exact:.4f}, None={none_exact:.4f}, "
                f"Gap={causal_gap:.4f} -> {'PASS' if op_passed else 'FAIL'}"
            )

    # 6. Strict Sparse Execution Verification
    bank.reset_all_forward_call_counts()
    sample_ex = _generate_parameter_free_examples(
        config.seed, 1, operation="COPY", split="test"
    )
    _evaluate_primitive_arm(
        core, bank.get(op_to_id["COPY"]), sample_ex, "COPY", argument_provider=None
    )
    counts = bank.forward_call_counts()
    sparse_passed = (counts[op_to_id["COPY"]] >= 1) and all(
        counts[pid] == 0 for op, pid in op_to_id.items() if op != "COPY"
    )

    # 7. Task-Blind Invariance Verification
    invariance_group = generate_compact_operator_counterfactual_groups(
        config.seed,
        1,
        operation="SELECT",
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )[0]
    core.model.eval()
    with torch.no_grad():
        feat, _ = encode_content(core, list(invariance_group.examples))
    ref = feat[0:1]
    task_blind_max_abs_diff = float((feat - ref).abs().max().item())
    task_blind_passed = task_blind_max_abs_diff <= TASK_BLIND_ATOL

    # Summary aggregations
    all_exacts = [r.correct_exact_match for r in results.values()]
    param_free_exacts = [
        r.correct_exact_match for r in results.values() if not r.is_parameterized
    ]
    param_exacts = [r.correct_exact_match for r in results.values() if r.is_parameterized]

    mean_correct = float(statistics.mean(all_exacts))
    mean_param_free = float(statistics.mean(param_free_exacts))
    mean_param = float(statistics.mean(param_exacts))

    all_op_passed = all(r.passed for r in results.values())
    overall_passed = (
        all_op_passed
        and (mean_correct >= 0.90)
        and (mean_param_free >= 0.95)
        and (mean_param >= 0.90)
        and task_blind_passed
        and sparse_passed
    )

    wall_clock = time.perf_counter() - start_time
    print(
        f"[Seed {config.seed}] Summary: Overall={mean_correct:.4f}, PF={mean_param_free:.4f}, "
        f"P={mean_param:.4f}, Blind={task_blind_passed}, Sparse={sparse_passed} "
        f"({wall_clock:.1f}s) -> {'PASS' if overall_passed else 'FAIL'}"
    )

    return UnifiedBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=sum(p.numel() for p in core.model.parameters()),
        total_bank_params=bank.total_parameter_count(),
        results_by_operation=results,
        mean_correct_exact_match=mean_correct,
        parameter_free_correct_exact_match=mean_param_free,
        parameterized_correct_exact_match=mean_param,
        task_blind_max_abs_diff=task_blind_max_abs_diff,
        task_blind_passed=task_blind_passed,
        sparse_execution_passed=sparse_passed,
        all_operations_passed=all_op_passed,
        passed=overall_passed,
        wall_clock_seconds=wall_clock,
        device=device,
    )


@dataclass(frozen=True)
class UnifiedBenchmarkMultiSeedReport:
    """Aggregated report across >= 5 seeds for Task A1-B002."""

    seeds: list[int]
    per_seed_reports: list[UnifiedBenchmarkReport]
    mean_overall_correct: float
    mean_parameter_free_correct: float
    mean_parameterized_correct: float
    per_operation_means: dict[str, dict[str, float]]
    meets_seed_policy: bool
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "mean_overall_correct": self.mean_overall_correct,
            "mean_parameter_free_correct": self.mean_parameter_free_correct,
            "mean_parameterized_correct": self.mean_parameterized_correct,
            "per_operation_means": self.per_operation_means,
            "meets_seed_policy": self.meets_seed_policy,
            "overall_passed": self.overall_passed,
            "per_seed_reports": [r.to_dict() for r in self.per_seed_reports],
        }


def unified_benchmark_report_from_dict(d: dict[str, Any]) -> UnifiedBenchmarkReport:
    """Deserialize UnifiedBenchmarkReport from dictionary."""
    config = unified_benchmark_config_from_dict(d["config"])
    results = {
        op: OperationBenchmarkResult(**res_dict)
        for op, res_dict in d["results_by_operation"].items()
    }
    return UnifiedBenchmarkReport(
        config=config,
        seed=int(d["seed"]),
        core_param_count=int(d["core_param_count"]),
        total_bank_params=int(d["total_bank_params"]),
        results_by_operation=results,
        mean_correct_exact_match=float(d["mean_correct_exact_match"]),
        parameter_free_correct_exact_match=float(d["parameter_free_correct_exact_match"]),
        parameterized_correct_exact_match=float(d["parameterized_correct_exact_match"]),
        task_blind_max_abs_diff=float(d["task_blind_max_abs_diff"]),
        task_blind_passed=bool(d["task_blind_passed"]),
        sparse_execution_passed=bool(d["sparse_execution_passed"]),
        all_operations_passed=bool(d["all_operations_passed"]),
        passed=bool(d["passed"]),
        wall_clock_seconds=float(d["wall_clock_seconds"]),
        device=str(d["device"]),
    )


def run_unified_oracle_causal_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Callable[[int], UnifiedBenchmarkConfig],
    *,
    output_dir: Path | None = None,
) -> UnifiedBenchmarkMultiSeedReport:
    """Run multi-seed benchmark across seeds."""
    seed_list = list(seeds)
    reports: list[UnifiedBenchmarkReport] = []

    for s in seed_list:
        print(f"\n==================== Benchmark Seed {s} ====================")
        cfg = config_factory(s)
        s_dir = (output_dir / f"seed_{s}") if output_dir else None
        if s_dir and (s_dir / "report.json").is_file():
            print(f"Loading existing report for Seed {s} from {s_dir / 'report.json'}")
            with (s_dir / "report.json").open("r", encoding="utf-8") as f:
                rep = unified_benchmark_report_from_dict(json.load(f))
        else:
            rep = run_unified_oracle_causal_benchmark(cfg, seed_dir=s_dir)
            if s_dir:
                s_dir.mkdir(parents=True, exist_ok=True)
                with (s_dir / "report.json").open("w", encoding="utf-8") as f:
                    json.dump(rep.to_dict(), f, indent=2)
        reports.append(rep)

    mean_overall = float(statistics.mean(r.mean_correct_exact_match for r in reports))
    mean_pf = float(statistics.mean(r.parameter_free_correct_exact_match for r in reports))
    mean_p = float(statistics.mean(r.parameterized_correct_exact_match for r in reports))

    op_means: dict[str, dict[str, float]] = {}
    for op in ALL_CANONICAL_OPERATIONS:
        op_means[op] = {
            "correct_exact_match": float(
                statistics.mean(
                    r.results_by_operation[op].correct_exact_match for r in reports
                )
            ),
            "exact_match_causal_gap": float(
                statistics.mean(
                    r.results_by_operation[op].exact_match_causal_gap for r in reports
                )
            ),
            "none_exact_match": float(
                statistics.mean(
                    r.results_by_operation[op].none_exact_match for r in reports
                )
            ),
            "wrong_family_exact_match": float(
                statistics.mean(
                    r.results_by_operation[op].wrong_family_exact_match for r in reports
                )
            ),
        }

    meets_policy = len(seed_list) >= MIN_GATE_SEEDS
    overall_passed = (
        meets_policy
        and all(r.passed for r in reports)
        and (mean_overall >= 0.90)
        and (mean_pf >= 0.95)
        and (mean_p >= 0.90)
    )

    summary_rep = UnifiedBenchmarkMultiSeedReport(
        seeds=seed_list,
        per_seed_reports=reports,
        mean_overall_correct=mean_overall,
        mean_parameter_free_correct=mean_pf,
        mean_parameterized_correct=mean_p,
        per_operation_means=op_means,
        meets_seed_policy=meets_policy,
        overall_passed=overall_passed,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_rep.to_dict(), f, indent=2)

    return summary_rep
