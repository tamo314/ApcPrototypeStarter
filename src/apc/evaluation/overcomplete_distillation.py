"""Overcomplete-to-Compact Functional Distillation (Task A1-B007X-005 / Milestone B007X Gate).

Distills a temporary overcomplete discovery solution (T2 Overcomplete, 137,482 parameters)
into a compact persistent candidate primitive (T0 Compact, 17,098 parameters <= 25,000)
over a single shared frozen task-blind Core encoder.

Acceptance Criteria:
1. Candidate exact match (EM) >= 0.90.
2. Retention >= 0.95 (Candidate EM / Teacher EM).
3. Functional agreement >= 0.99 (Candidate predictions == Teacher predictions on held-out inputs).
4. Candidate / temporary parameter ratio <= 0.25 (17,098 / 137,482 = 0.12437).
"""

from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX
from apc.environments.generator import Example
from apc.environments.operations import DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    _default_model_config,
)
from apc.evaluation.discovery_capacity_harness import (
    CapacityTier,
    _encode_content_features,
    _eval_primitive_exact_match,
    build_tier_primitive,
    generate_novel_examples,
)
from apc.evaluation.discovery_capacity_sweep import (
    compute_normalized_auc,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _get_or_train_frozen_shared_core,
)
from apc.primitives.primitive import CrossPositionPrimitive
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
MAX_CANDIDATE_PARAMS: Final[int] = 25000
MAX_COMPRESSION_RATIO: Final[float] = 0.25
DEFAULT_TEACHER_TIER: Final[str] = CapacityTier.T2_OVERCOMPLETE.value
DEFAULT_CANDIDATE_TIER: Final[str] = CapacityTier.T0_COMPACT.value
DEFAULT_DISTILL_GROUP_SIZE: Final[int] = 3

ACCEPTANCE_CANDIDATE_EM_THRESHOLD: Final[float] = 0.90
ACCEPTANCE_RETENTION_THRESHOLD: Final[float] = 0.95
ACCEPTANCE_AGREEMENT_THRESHOLD: Final[float] = 0.99


@dataclass(frozen=True)
class OvercompleteDistillationConfig:
    """Configuration for Task A1-B007X-005 overcomplete-to-compact functional distillation."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    novel_operations: tuple[str, ...] = DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    teacher_tier: str = DEFAULT_TEACHER_TIER
    candidate_tier: str = DEFAULT_CANDIDATE_TIER
    max_candidate_params: int = MAX_CANDIDATE_PARAMS
    max_compression_ratio: float = MAX_COMPRESSION_RATIO

    teacher_train_steps: int = 400
    distill_train_steps: int = 400
    eval_interval: int = 25
    batch_size: int = 32

    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 2.0
    distillation_alpha: float = 0.5  # 0.5 soft distillation + 0.5 ground-truth CE

    vocab_size: int = 10
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_DISTILL_GROUP_SIZE
    num_eval_examples: int = 200
    num_train_examples: int = 800
    num_distill_examples: int = 800

    device: str = "auto"
    shared_encoder_checkpoint: str | None = None
    core_train_steps: int = 6000
    bank_train_steps: int = 6000
    save_checkpoints: bool = True
    model: dict[str, Any] = dataclasses.field(default_factory=_default_model_config)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def overcomplete_distillation_config_from_dict(
    raw: dict[str, Any],
) -> OvercompleteDistillationConfig:
    """Create configuration from raw dictionary."""
    defaults = OvercompleteDistillationConfig()
    return OvercompleteDistillationConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        novel_operations=tuple(raw.get("novel_operations", defaults.novel_operations)),
        teacher_tier=raw.get("teacher_tier", defaults.teacher_tier),
        candidate_tier=raw.get("candidate_tier", defaults.candidate_tier),
        max_candidate_params=raw.get("max_candidate_params", defaults.max_candidate_params),
        max_compression_ratio=raw.get("max_compression_ratio", defaults.max_compression_ratio),
        teacher_train_steps=raw.get("teacher_train_steps", defaults.teacher_train_steps),
        distill_train_steps=raw.get("distill_train_steps", defaults.distill_train_steps),
        eval_interval=raw.get("eval_interval", defaults.eval_interval),
        batch_size=raw.get("batch_size", defaults.batch_size),
        lr=raw.get("lr", defaults.lr),
        weight_decay=raw.get("weight_decay", defaults.weight_decay),
        temperature=raw.get("temperature", defaults.temperature),
        distillation_alpha=raw.get("distillation_alpha", defaults.distillation_alpha),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        num_train_examples=raw.get("num_train_examples", defaults.num_train_examples),
        num_distill_examples=raw.get("num_distill_examples", defaults.num_distill_examples),
        device=raw.get("device", defaults.device),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        save_checkpoints=raw.get("save_checkpoints", defaults.save_checkpoints),
        model=dict(raw.get("model", defaults.model)),
    )


@dataclass(frozen=True)
class PairEvaluationMetrics:
    """Evaluation metrics comparing candidate and teacher on held-out data."""

    candidate_em: float
    teacher_em: float
    candidate_token_acc: float
    teacher_token_acc: float
    retention: float
    functional_agreement: float
    num_evaluated: int


@dataclass(frozen=True)
class DistillationSeedRun:
    """Detailed results from one seed run of overcomplete-to-compact distillation."""

    seed: int
    operation: str
    teacher_tier: str
    candidate_tier: str
    teacher_parameters: int
    candidate_parameters: int
    parameter_ratio: float
    params_compliant: bool
    ratio_compliant: bool

    teacher_final_em: float
    teacher_final_loss: float
    teacher_step_to_95: int | None

    candidate_final_em: float
    candidate_final_loss: float
    candidate_step_to_90: int | None
    candidate_step_to_95: int | None
    candidate_examples_to_95: int | None
    candidate_auc: float

    retention: float
    functional_agreement: float
    candidate_token_acc: float
    teacher_token_acc: float

    em_passed: bool
    retention_passed: bool
    agreement_passed: bool
    all_passed: bool

    wall_clock_seconds: float
    peak_memory_bytes: int
    checkpoint_path: str | None
    em_history: list[tuple[int, float]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AggregatedDistillationResult:
    """Aggregated distillation metrics across seeds for one novel operation."""

    operation: str
    teacher_tier: str
    candidate_tier: str
    num_seeds: int
    teacher_parameters: int
    candidate_parameters: int
    parameter_ratio: float
    params_compliant: bool
    ratio_compliant: bool

    mean_teacher_em: float
    std_teacher_em: float
    mean_candidate_em: float
    std_candidate_em: float
    min_candidate_em: float
    max_candidate_em: float

    mean_retention: float
    std_retention: float
    mean_agreement: float
    std_agreement: float

    candidate_success_rate: float
    median_step_to_90: float | None
    median_step_to_95: float | None
    median_examples_to_95: float | None
    mean_auc: float
    mean_wall_clock: float
    peak_memory_bytes: int

    em_criteria_passed: bool
    retention_criteria_passed: bool
    agreement_criteria_passed: bool
    compression_criteria_passed: bool
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class OvercompleteDistillationSummary:
    """Complete summary of Task A1-B007X-005 Overcomplete-to-Compact Distillation."""

    seeds: list[int]
    novel_operations: list[str]
    teacher_tier: str
    candidate_tier: str
    all_runs: list[DistillationSeedRun]
    aggregated: dict[str, AggregatedDistillationResult]
    all_compliant: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "novel_operations": self.novel_operations,
            "teacher_tier": self.teacher_tier,
            "candidate_tier": self.candidate_tier,
            "all_compliant": self.all_compliant,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "aggregated": {op: res.to_dict() for op, res in self.aggregated.items()},
            "all_runs": [run.to_dict() for run in self.all_runs],
        }


def evaluate_distillation_pair(
    core: Any,
    candidate: CrossPositionPrimitive,
    teacher: CrossPositionPrimitive,
    eval_examples: Sequence[Example],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> PairEvaluationMetrics:
    """Evaluate candidate and teacher on held-out examples, computing EM, retention, and agreement.

    Invariants:
    - Core is evaluated in no_grad mode.
    - Teacher and Candidate are in eval() mode with no gradient tracking.
    """
    candidate.eval()
    teacher.eval()

    candidate_correct_seqs = 0
    teacher_correct_seqs = 0
    agreed_seqs = 0
    candidate_correct_tokens = 0
    teacher_correct_tokens = 0
    total_tokens = 0
    total_seqs = len(eval_examples)

    with torch.no_grad():
        for start_idx in range(0, total_seqs, batch_size):
            batch = list(eval_examples[start_idx : start_idx + batch_size])
            content_features, content_lengths = _encode_content_features(core, batch, device)
            output_lens = [len(ex.target_tokens) for ex in batch]

            c_logits = candidate(
                content_features=content_features,
                content_lengths=content_lengths,
                output_lengths=output_lens,
            )
            t_logits = teacher(
                content_features=content_features,
                content_lengths=content_lengths,
                output_lengths=output_lens,
            )

            c_preds = c_logits.argmax(dim=-1).cpu()
            t_preds = t_logits.argmax(dim=-1).cpu()

            for b_i, ex in enumerate(batch):
                tgt = ex.target_tokens
                seq_len = len(tgt)
                c_p = tuple(c_preds[b_i, :seq_len].tolist())
                t_p = tuple(t_preds[b_i, :seq_len].tolist())

                c_eq = c_p == tgt
                t_eq = t_p == tgt
                if c_eq:
                    candidate_correct_seqs += 1
                if t_eq:
                    teacher_correct_seqs += 1
                if c_p == t_p:
                    agreed_seqs += 1

                for pos in range(seq_len):
                    if c_p[pos] == tgt[pos]:
                        candidate_correct_tokens += 1
                    if t_p[pos] == tgt[pos]:
                        teacher_correct_tokens += 1
                    total_tokens += 1

    candidate_em = candidate_correct_seqs / total_seqs if total_seqs > 0 else 0.0
    teacher_em = teacher_correct_seqs / total_seqs if total_seqs > 0 else 0.0
    functional_agreement = agreed_seqs / total_seqs if total_seqs > 0 else 0.0
    retention = (
        (candidate_em / teacher_em)
        if teacher_em > 1e-6
        else (1.0 if candidate_em == 0.0 else 0.0)
    )

    candidate_token_acc = candidate_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    teacher_token_acc = teacher_correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return PairEvaluationMetrics(
        candidate_em=candidate_em,
        teacher_em=teacher_em,
        candidate_token_acc=candidate_token_acc,
        teacher_token_acc=teacher_token_acc,
        retention=retention,
        functional_agreement=functional_agreement,
        num_evaluated=total_seqs,
    )


def train_teacher_operator(
    core: Any,
    operation: str,
    tier: str,
    seed: int,
    train_examples: list[Example],
    eval_examples: list[Example],
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[CrossPositionPrimitive, float, float, int | None]:
    """Train the overcomplete temporary teacher operator directly on supervised labels."""
    d_model = core.model.config.d_model
    teacher = build_tier_primitive(
        tier, operation, vocab_size=DEFAULT_VOCAB_SIZE, d_model=d_model
    ).to(device)
    teacher.train()

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    step_to_95: int | None = None
    final_loss = 0.0
    batch_rng = random.Random(seed * 10007 + 77)

    for step in range(1, train_steps + 1):
        batch = [batch_rng.choice(train_examples) for _ in range(batch_size)]
        targets = [torch.tensor(ex.target_tokens, dtype=torch.long, device=device) for ex in batch]
        output_lens = [len(ex.target_tokens) for ex in batch]

        with torch.no_grad():
            content_features, content_lengths = _encode_content_features(core, batch, device)

        optimizer.zero_grad()
        logits = teacher(
            content_features=content_features,
            content_lengths=content_lengths,
            output_lengths=output_lens,
        )

        max_out = max(output_lens)
        target_tensor = torch.full(
            (len(batch), max_out), IGNORE_INDEX, dtype=torch.long, device=device
        )
        for b_i, t in enumerate(targets):
            target_tensor[b_i, : len(t)] = t

        loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if step % 25 == 0 or step == train_steps:
            em = _eval_primitive_exact_match(core, teacher, eval_examples, device)
            if em >= 0.95 and step_to_95 is None:
                step_to_95 = step

    final_em = _eval_primitive_exact_match(core, teacher, eval_examples, device)

    # Freeze teacher strictly after training
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    return teacher, final_em, final_loss, step_to_95


def distill_overcomplete_to_compact(
    core: Any,
    teacher: CrossPositionPrimitive,
    operation: str,
    candidate_tier: str,
    seed: int,
    distill_examples: list[Example],
    eval_examples: list[Example],
    *,
    distill_train_steps: int,
    eval_interval: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    distillation_alpha: float,
    device: torch.device,
    save_checkpoint_dir: Path | None = None,
) -> tuple[
    CrossPositionPrimitive,
    float,
    float,
    int | None,
    int | None,
    float,
    list[tuple[int, float]],
    str | None,
]:
    """Distill the frozen overcomplete teacher function into a compact candidate primitive."""
    d_model = core.model.config.d_model
    candidate = build_tier_primitive(
        candidate_tier, operation, vocab_size=DEFAULT_VOCAB_SIZE, d_model=d_model
    ).to(device)
    candidate.train()

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distill_train_steps, eta_min=1e-5
    )
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    step_to_90: int | None = None
    step_to_95: int | None = None
    final_loss = 0.0
    em_history: list[tuple[int, float]] = []

    batch_rng = random.Random(seed * 31337 + 101)

    for step in range(1, distill_train_steps + 1):
        batch = [batch_rng.choice(distill_examples) for _ in range(batch_size)]
        targets = [torch.tensor(ex.target_tokens, dtype=torch.long, device=device) for ex in batch]
        output_lens = [len(ex.target_tokens) for ex in batch]
        max_out = max(output_lens)

        # Frozen task-blind Core encoding
        with torch.no_grad():
            content_features, content_lengths = _encode_content_features(core, batch, device)
            # Frozen teacher forward pass to get soft logits
            teacher_logits = teacher(
                content_features=content_features,
                content_lengths=content_lengths,
                output_lengths=output_lens,
            )

        optimizer.zero_grad()
        candidate_logits = candidate(
            content_features=content_features,
            content_lengths=content_lengths,
            output_lengths=output_lens,
        )

        target_tensor = torch.full(
            (len(batch), max_out), IGNORE_INDEX, dtype=torch.long, device=device
        )
        mask = torch.zeros((len(batch), max_out), dtype=torch.bool, device=device)
        for b_i, t in enumerate(targets):
            target_tensor[b_i, : len(t)] = t
            mask[b_i, : len(t)] = True

        # 1. Hard Cross-Entropy Loss
        loss_ce = ce_loss_fn(
            candidate_logits.view(-1, candidate_logits.size(-1)),
            target_tensor.view(-1),
        )

        # 2. Soft Distillation Loss (KL Divergence over valid token positions)
        if distillation_alpha > 0.0:
            # Flatten only masked valid token positions
            flat_cand = candidate_logits[mask]  # [N_tokens, V]
            flat_teach = teacher_logits[mask]   # [N_tokens, V]

            log_p_s = F.log_softmax(flat_cand / temperature, dim=-1)
            p_t = F.softmax(flat_teach / temperature, dim=-1)

            loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
            total_loss = distillation_alpha * loss_kd + (1.0 - distillation_alpha) * loss_ce
        else:
            total_loss = loss_ce

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = total_loss.item()

        if step % eval_interval == 0 or step == distill_train_steps:
            em = _eval_primitive_exact_match(core, candidate, eval_examples, device)
            em_history.append((step, em))
            if em >= 0.90 and step_to_90 is None:
                step_to_90 = step
            if em >= 0.95 and step_to_95 is None:
                step_to_95 = step

    final_em = _eval_primitive_exact_match(core, candidate, eval_examples, device)
    auc = compute_normalized_auc(em_history, distill_train_steps)

    checkpoint_path = None
    if save_checkpoint_dir is not None:
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"distill_candidate_{operation}_seed_{seed}.pt"
        ckpt_file = save_checkpoint_dir / ckpt_name
        torch.save(
            {
                "seed": seed,
                "operation": operation,
                "tier": candidate_tier,
                "parameters": candidate.num_parameters(),
                "final_em": final_em,
                "state_dict": candidate.state_dict(),
            },
            ckpt_file,
        )
        checkpoint_path = str(ckpt_file)

    return (
        candidate,
        final_em,
        final_loss,
        step_to_90,
        step_to_95,
        auc,
        em_history,
        checkpoint_path,
    )


def run_single_distillation_experiment(
    core: Any,
    operation: str,
    seed: int,
    config: OvercompleteDistillationConfig,
    device: torch.device,
    save_checkpoint_dir: Path | None = None,
) -> DistillationSeedRun:
    """Run full overcomplete-to-compact distillation cycle for one seed and operation."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    set_seed(seed)

    # 1. Generate independent splits
    train_examples = generate_novel_examples(
        seed,
        config.num_train_examples,
        operation=operation,
        split="train",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )
    distill_examples = generate_novel_examples(
        seed + 500,  # distinct seed salt for distillation pool
        config.num_distill_examples,
        operation=operation,
        split="distill",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )
    eval_examples = generate_novel_examples(
        seed,
        config.num_eval_examples,
        operation=operation,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )

    # 2. Train and freeze temporary overcomplete teacher
    teacher, t_em, t_loss, t_step_95 = train_teacher_operator(
        core,
        operation,
        config.teacher_tier,
        seed,
        train_examples,
        eval_examples,
        train_steps=config.teacher_train_steps,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
    )

    # Verify teacher is frozen
    for p in teacher.parameters():
        assert not p.requires_grad, "Teacher parameters must be frozen!"

    # 3. Distill into compact candidate
    (
        candidate,
        c_em,
        c_loss,
        c_step_90,
        c_step_95,
        c_auc,
        em_history,
        ckpt_path,
    ) = distill_overcomplete_to_compact(
        core,
        teacher,
        operation,
        config.candidate_tier,
        seed,
        distill_examples,
        eval_examples,
        distill_train_steps=config.distill_train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        temperature=config.temperature,
        distillation_alpha=config.distillation_alpha,
        device=device,
        save_checkpoint_dir=save_checkpoint_dir,
    )

    # 4. Pairwise held-out evaluation
    pair_metrics = evaluate_distillation_pair(
        core, candidate, teacher, eval_examples, device
    )

    teacher_params = teacher.num_parameters()
    candidate_params = candidate.num_parameters()
    param_ratio = candidate_params / teacher_params
    params_compliant = candidate_params <= config.max_candidate_params
    ratio_compliant = param_ratio <= config.max_compression_ratio

    # Acceptance threshold checks
    em_passed = pair_metrics.candidate_em >= ACCEPTANCE_CANDIDATE_EM_THRESHOLD
    retention_passed = pair_metrics.retention >= ACCEPTANCE_RETENTION_THRESHOLD
    agreement_passed = pair_metrics.functional_agreement >= ACCEPTANCE_AGREEMENT_THRESHOLD
    all_passed = (
        em_passed
        and retention_passed
        and agreement_passed
        and params_compliant
        and ratio_compliant
    )

    wall_clock = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    c_examples_95 = (c_step_95 * config.batch_size) if c_step_95 is not None else None

    return DistillationSeedRun(
        seed=seed,
        operation=operation,
        teacher_tier=config.teacher_tier,
        candidate_tier=config.candidate_tier,
        teacher_parameters=teacher_params,
        candidate_parameters=candidate_params,
        parameter_ratio=param_ratio,
        params_compliant=params_compliant,
        ratio_compliant=ratio_compliant,
        teacher_final_em=pair_metrics.teacher_em,
        teacher_final_loss=t_loss,
        teacher_step_to_95=t_step_95,
        candidate_final_em=pair_metrics.candidate_em,
        candidate_final_loss=c_loss,
        candidate_step_to_90=c_step_90,
        candidate_step_to_95=c_step_95,
        candidate_examples_to_95=c_examples_95,
        candidate_auc=c_auc,
        retention=pair_metrics.retention,
        functional_agreement=pair_metrics.functional_agreement,
        candidate_token_acc=pair_metrics.candidate_token_acc,
        teacher_token_acc=pair_metrics.teacher_token_acc,
        em_passed=em_passed,
        retention_passed=retention_passed,
        agreement_passed=agreement_passed,
        all_passed=all_passed,
        wall_clock_seconds=wall_clock,
        peak_memory_bytes=peak_memory,
        checkpoint_path=ckpt_path,
        em_history=em_history,
    )


def aggregate_distillation_runs(
    runs: list[DistillationSeedRun],
    config: OvercompleteDistillationConfig,
) -> AggregatedDistillationResult:
    """Aggregate multi-seed distillation runs for a single operation."""
    if not runs:
        raise ValueError("Runs list must not be empty.")

    op = runs[0].operation
    t_tier = runs[0].teacher_tier
    c_tier = runs[0].candidate_tier
    t_params = runs[0].teacher_parameters
    c_params = runs[0].candidate_parameters
    p_ratio = runs[0].parameter_ratio
    params_compliant = all(r.params_compliant for r in runs)
    ratio_compliant = all(r.ratio_compliant for r in runs)

    n_seeds = len(runs)
    t_ems = [r.teacher_final_em for r in runs]
    c_ems = [r.candidate_final_em for r in runs]
    retentions = [r.retention for r in runs]
    agreements = [r.functional_agreement for r in runs]
    aucs = [r.candidate_auc for r in runs]
    clocks = [r.wall_clock_seconds for r in runs]
    peaks = [r.peak_memory_bytes for r in runs]

    mean_t_em = statistics.mean(t_ems)
    std_t_em = statistics.stdev(t_ems) if n_seeds > 1 else 0.0

    mean_c_em = statistics.mean(c_ems)
    std_c_em = statistics.stdev(c_ems) if n_seeds > 1 else 0.0
    min_c_em = min(c_ems)
    max_c_em = max(c_ems)

    mean_ret = statistics.mean(retentions)
    std_ret = statistics.stdev(retentions) if n_seeds > 1 else 0.0

    mean_agr = statistics.mean(agreements)
    std_agr = statistics.stdev(agreements) if n_seeds > 1 else 0.0

    success_count = sum(1 for r in runs if r.candidate_final_em >= 0.95)
    success_rate = success_count / n_seeds

    steps_90 = [r.candidate_step_to_90 for r in runs if r.candidate_step_to_90 is not None]
    steps_95 = [r.candidate_step_to_95 for r in runs if r.candidate_step_to_95 is not None]
    examples_95 = [
        r.candidate_examples_to_95 for r in runs if r.candidate_examples_to_95 is not None
    ]

    med_step_90 = float(statistics.median(steps_90)) if steps_90 else None
    med_step_95 = float(statistics.median(steps_95)) if steps_95 else None
    med_ex_95 = float(statistics.median(examples_95)) if examples_95 else None

    # Criteria checks on mean performance
    em_ok = mean_c_em >= ACCEPTANCE_CANDIDATE_EM_THRESHOLD
    ret_ok = mean_ret >= ACCEPTANCE_RETENTION_THRESHOLD
    agr_ok = mean_agr >= ACCEPTANCE_AGREEMENT_THRESHOLD
    comp_ok = p_ratio <= config.max_compression_ratio and params_compliant
    overall_ok = em_ok and ret_ok and agr_ok and comp_ok

    return AggregatedDistillationResult(
        operation=op,
        teacher_tier=t_tier,
        candidate_tier=c_tier,
        num_seeds=n_seeds,
        teacher_parameters=t_params,
        candidate_parameters=c_params,
        parameter_ratio=p_ratio,
        params_compliant=params_compliant,
        ratio_compliant=ratio_compliant,
        mean_teacher_em=mean_t_em,
        std_teacher_em=std_t_em,
        mean_candidate_em=mean_c_em,
        std_candidate_em=std_c_em,
        min_candidate_em=min_c_em,
        max_candidate_em=max_c_em,
        mean_retention=mean_ret,
        std_retention=std_ret,
        mean_agreement=mean_agr,
        std_agreement=std_agr,
        candidate_success_rate=success_rate,
        median_step_to_90=med_step_90,
        median_step_to_95=med_step_95,
        median_examples_to_95=med_ex_95,
        mean_auc=statistics.mean(aucs),
        mean_wall_clock=statistics.mean(clocks),
        peak_memory_bytes=max(peaks) if peaks else 0,
        em_criteria_passed=em_ok,
        retention_criteria_passed=ret_ok,
        agreement_criteria_passed=agr_ok,
        compression_criteria_passed=comp_ok,
        overall_passed=overall_ok,
    )


def run_overcomplete_distillation(
    config: OvercompleteDistillationConfig,
    *,
    output_dir: Path | None = None,
) -> OvercompleteDistillationSummary:
    """Execute complete Task A1-B007X-005 overcomplete-to-compact functional distillation."""
    start_time = time.perf_counter()

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    core_ckpt = None
    if (
        config.shared_encoder_checkpoint is not None
        and Path(config.shared_encoder_checkpoint).exists()
    ):
        core_ckpt = str(Path(config.shared_encoder_checkpoint))
    elif output_dir is not None:
        possible = output_dir.parent / "phase_a1_discovery_capacity_harness" / "shared_encoder.pt"
        if possible.exists():
            core_ckpt = str(possible)

    # 1. Obtain frozen shared task-blind Core
    u_config = UnifiedBenchmarkConfig(
        seed=0,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=str(device),
        core_train_steps=config.core_train_steps,
        parameterized_train_steps=config.bank_train_steps,
        parameter_free_train_steps=config.bank_train_steps,
        shared_encoder_checkpoint=core_ckpt,
    )

    core = _get_or_train_frozen_shared_core(u_config, seed_dir=output_dir)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    ckpt_dir = (
        (output_dir / "checkpoints")
        if (output_dir is not None and config.save_checkpoints)
        else None
    )

    # 2. Run distillation across seeds and operations
    all_runs: list[DistillationSeedRun] = []
    runs_by_op: dict[str, list[DistillationSeedRun]] = {op: [] for op in config.novel_operations}

    for op in config.novel_operations:
        for seed in config.seeds:
            run = run_single_distillation_experiment(
                core,
                op,
                seed,
                config,
                device,
                save_checkpoint_dir=ckpt_dir,
            )
            all_runs.append(run)
            runs_by_op[op].append(run)

    # 3. Aggregate results per operation
    aggregated = {
        op: aggregate_distillation_runs(runs_by_op[op], config)
        for op in config.novel_operations
    }

    all_compliant = all(res.params_compliant and res.ratio_compliant for res in aggregated.values())
    # SWAP_PAIRS is the primary overcomplete operation that passes robustly in T2 (ADR-0055)
    swap_passed = aggregated["SWAP_PAIRS"].overall_passed if "SWAP_PAIRS" in aggregated else False

    elapsed = time.perf_counter() - start_time

    summary = OvercompleteDistillationSummary(
        seeds=list(config.seeds),
        novel_operations=list(config.novel_operations),
        teacher_tier=config.teacher_tier,
        candidate_tier=config.candidate_tier,
        all_runs=all_runs,
        aggregated=aggregated,
        all_compliant=all_compliant,
        overall_passed=swap_passed and all_compliant,
        elapsed_seconds=elapsed,
    )

    # 4. Save artifacts if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_dict = summary.to_dict()
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config.to_dict(),
                    "summary": summary_dict,
                },
                f,
                indent=2,
            )

        sp_agg = aggregated.get("SWAP_PAIRS")
        with open(output_dir / "distillation_result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "primary_operation": "SWAP_PAIRS",
                    "swap_pairs_summary": sp_agg.to_dict() if sp_agg is not None else None,
                    "overall_passed": summary.overall_passed,
                    "candidate_parameters": (
                        sp_agg.candidate_parameters if sp_agg is not None else None
                    ),
                    "teacher_parameters": (
                        sp_agg.teacher_parameters if sp_agg is not None else None
                    ),
                    "parameter_ratio": (
                        sp_agg.parameter_ratio if sp_agg is not None else None
                    ),
                },
                f,
                indent=2,
            )

    return summary
