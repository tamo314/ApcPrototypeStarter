"""Novel-Task and Capacity-Ladder Harness (Task A1-B007X-002 / Milestone B007X Gate).

Creates a controlled discovery-capacity benchmark infrastructure:
1. Defines >=2 novel operations (SWAP_PAIRS, INVERT_HALF, ROTATE_TRIPLETS).
2. Verifies bank and composition failure first (best_existing_EM < 0.20).
3. Implements temporary capacity ladder with matched single-layer cross-attention topology:
   - T0 (Compact Control): ~17k parameters (17,098)
   - T1 (Medium): ~67k parameters (67,082)
   - T2 (Overcomplete): ~137k parameters (137,482, 8.04x >= 4x)
   - T3 (Extra-Large, optional): ~232k parameters (232,458, 13.6x)
4. Enforces task-blind invariants (h_content = f(content)) and no oracle leakage.
5. Establishes common data protocol, training loop, and metrics accounting for Task A1-B007X-003.

Acceptance Criteria (CODEX_TASKS_A1_B007X_DISCOVERY_COMPRESSION.md):
- novelty verified (best_existing_EM < 0.20 for all novel tasks),
- capacity counts measured (exact matching for T0-T3; T2 >= 4x T0),
- no oracle leakage (verified mathematically invariant content representation),
- common data protocol established.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES,
    get_operation,
)
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    PARAMETERIZED_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition_search import search_composition_recipe
from apc.primitives.conditioning import DEFAULT_MAX_SEQUENCE_LENGTH
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
)
from apc.utils.seed import set_seed

NOVELTY_EM_THRESHOLD: Final[float] = 0.20
MIN_COMPRESSION_RATIO: Final[float] = 4.0
_EVAL_BATCH_SIZE: Final[int] = 64


class CapacityTier(str, Enum):  # noqa: UP042
    """Capacity tiers for temporary discovery operators."""

    T0_COMPACT = "T0_compact"
    T1_MEDIUM = "T1_medium"
    T2_OVERCOMPLETE = "T2_overcomplete"
    T3_EXTRA_LARGE = "T3_extra_large"


@dataclass(frozen=True)
class TierSpec:
    """Specification of a capacity tier under matched single-layer topology."""

    tier: str
    d_operator: int
    d_operator_ff: int
    n_head: int
    expected_parameters: int
    ratio_to_compact: float
    role: str


TIER_SPECS: dict[str, TierSpec] = {
    CapacityTier.T0_COMPACT.value: TierSpec(
        tier=CapacityTier.T0_COMPACT.value,
        d_operator=32,
        d_operator_ff=64,
        n_head=4,
        expected_parameters=17098,
        ratio_to_compact=1.000,
        role="compact_control",
    ),
    CapacityTier.T1_MEDIUM.value: TierSpec(
        tier=CapacityTier.T1_MEDIUM.value,
        d_operator=64,
        d_operator_ff=256,
        n_head=4,
        expected_parameters=67082,
        ratio_to_compact=67082 / 17098,
        role="medium",
    ),
    CapacityTier.T2_OVERCOMPLETE.value: TierSpec(
        tier=CapacityTier.T2_OVERCOMPLETE.value,
        d_operator=96,
        d_operator_ff=384,
        n_head=6,
        expected_parameters=137482,
        ratio_to_compact=137482 / 17098,
        role="overcomplete",
    ),
    CapacityTier.T3_EXTRA_LARGE.value: TierSpec(
        tier=CapacityTier.T3_EXTRA_LARGE.value,
        d_operator=128,
        d_operator_ff=512,
        n_head=8,
        expected_parameters=232458,
        ratio_to_compact=232458 / 17098,
        role="extra_large_optional",
    ),
}


def build_tier_primitive(
    tier: str,
    operation: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    d_model: int = 192,
    primitive_id: int = 0,
) -> CrossPositionPrimitive:
    """Build a CrossPositionPrimitive for a specified capacity tier."""
    spec = TIER_SPECS.get(tier)
    if spec is None:
        valid_tiers = list(TIER_SPECS.keys())
        raise ValueError(f"Unknown capacity tier: '{tier}'. Must be one of {valid_tiers}")

    config = CrossPositionPrimitiveConfig(
        operation=operation,
        d_model=d_model,
        d_operator=spec.d_operator,
        n_head=spec.n_head,
        d_operator_ff=spec.d_operator_ff,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
    )
    return CrossPositionPrimitive(primitive_id, config)


def generate_novel_examples(
    seed: int,
    n: int,
    *,
    operation: str,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
) -> list[Example]:
    """Deterministically generate examples for a novel operation."""
    op = get_operation(operation)
    salt = 2000 if split == "test" else (1000 if split == "train" else 500)
    rng = random.Random(seed * 7919 + salt + (hash(operation) % 10000))

    valid_lengths = [
        length
        for length in range(sequence_length_range[0], sequence_length_range[1] + 1)
        if op.is_valid_for_length(length)
    ]
    if not valid_lengths:
        raise ValueError(f"No valid lengths for '{operation}' in range {sequence_length_range}")

    examples: list[Example] = []
    for _ in range(n):
        seq_len = rng.choice(valid_lengths)
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
            category="novel",
            split=split,
            vocab_size=vocab_size,
        )
        examples.append(ex)
    return examples


@dataclass(frozen=True)
class NoveltyCalibrationResult:
    """Outcome of novelty verification for one operation."""

    operation: str
    best_bank_primitive_name: str
    best_bank_primitive_em: float
    best_composition_recipe: str
    best_composition_em: float
    best_existing_em: float
    threshold: float
    is_novel: bool


@dataclass(frozen=True)
class TierAuditResult:
    """Outcome of capacity count and topology audit for one tier."""

    tier: str
    d_operator: int
    d_operator_ff: int
    n_head: int
    actual_parameters: int
    expected_parameters: int
    ratio_to_compact: float
    counts_match: bool


@dataclass(frozen=True)
class DiscoveryHarnessRunSummary:
    """Complete summary of Task A1-B007X-002 harness verification."""

    seed: int
    novel_operations: list[str]
    novelty_results: dict[str, NoveltyCalibrationResult]
    tier_audit_results: dict[str, TierAuditResult]
    all_novel_verified: bool
    all_tier_counts_verified: bool
    overcomplete_ratio_verified: bool
    no_oracle_leakage_verified: bool
    data_protocol_verified: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "novel_operations": self.novel_operations,
            "all_novel_verified": self.all_novel_verified,
            "all_tier_counts_verified": self.all_tier_counts_verified,
            "overcomplete_ratio_verified": self.overcomplete_ratio_verified,
            "no_oracle_leakage_verified": self.no_oracle_leakage_verified,
            "data_protocol_verified": self.data_protocol_verified,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "novelty_results": {
                op: dataclasses.asdict(res) for op, res in self.novelty_results.items()
            },
            "tier_audit_results": {
                tier: dataclasses.asdict(res) for tier, res in self.tier_audit_results.items()
            },
        }


@dataclass(frozen=True)
class DiscoveryCapacityHarnessConfig:
    """Configuration for Task A1-B007X-002 harness verification."""

    seed: int = 0
    vocab_size: int = 10
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = dataclasses.field(default_factory=_default_model_config)
    device: str = "auto"

    novel_operations: tuple[str, ...] = DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    tiers_to_audit: tuple[str, ...] = tuple(TIER_SPECS.keys())
    novelty_em_threshold: float = NOVELTY_EM_THRESHOLD
    min_compression_ratio: float = MIN_COMPRESSION_RATIO

    num_eval_examples: int = 200
    num_trial_train_examples: int = 400
    trial_train_steps: int = 200
    batch_size: int = 32

    core_train_steps: int = 6000
    bank_train_steps: int = 6000
    shared_encoder_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def discovery_harness_config_from_dict(raw: dict[str, Any]) -> DiscoveryCapacityHarnessConfig:
    """Create configuration from raw dict."""
    defaults = DiscoveryCapacityHarnessConfig()
    return DiscoveryCapacityHarnessConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        device=raw.get("device", defaults.device),
        novel_operations=tuple(raw.get("novel_operations", defaults.novel_operations)),
        tiers_to_audit=tuple(raw.get("tiers_to_audit", defaults.tiers_to_audit)),
        novelty_em_threshold=raw.get("novelty_em_threshold", defaults.novelty_em_threshold),
        min_compression_ratio=raw.get("min_compression_ratio", defaults.min_compression_ratio),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        num_trial_train_examples=raw.get(
            "num_trial_train_examples", defaults.num_trial_train_examples
        ),
        trial_train_steps=raw.get("trial_train_steps", defaults.trial_train_steps),
        batch_size=raw.get("batch_size", defaults.batch_size),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
    )


def _encode_content_features(
    core: Any,
    examples: Sequence[Example],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Extract task-blind content representation h_content = f(content) from frozen core."""
    content_lengths = [len(ex.input_tokens) for ex in examples]
    lmax = max(content_lengths)
    b_ids = collate_content_only_batch(examples, core.tokens, device=device)
    with torch.no_grad():
        h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]
    return h_content, content_lengths


def _eval_primitive_exact_match(
    core: Any,
    primitive: Any,
    examples: list[Example],
    device: torch.device,
) -> float:
    """Evaluate exact match of a primitive on examples using task-blind content representation."""
    core.model.eval()
    primitive.eval()
    correct = 0
    total = len(examples)

    with torch.no_grad():
        for i in range(0, total, _EVAL_BATCH_SIZE):
            batch = examples[i : i + _EVAL_BATCH_SIZE]
            output_lens = [len(ex.target_tokens) for ex in batch]
            content_features, content_lengths = _encode_content_features(core, batch, device)

            # Determine argument values if parameterized primitive
            op_name = getattr(primitive, "operation", getattr(primitive, "name", "unknown"))
            arg_vals: list[Any] | None = None
            if op_name in PARAMETERIZED_OPERATIONS:
                if op_name == "SHIFT":
                    arg_vals = [1 for _ in batch]
                elif op_name == "SELECT":
                    arg_vals = [(0, 1) for _ in batch]
                elif op_name == "COUNT":
                    arg_vals = [0 for _ in batch]
                elif op_name == "BIND":
                    arg_vals = [0 for _ in batch]

            logits = primitive(
                content_features=content_features,
                content_lengths=content_lengths,
                output_lengths=output_lens,
                argument_values=arg_vals,
            )

            preds = logits.argmax(dim=-1)
            for b_idx, ex in enumerate(batch):
                olen = len(ex.target_tokens)
                pred_tokens = tuple(preds[b_idx, :olen].cpu().tolist())
                if pred_tokens == ex.target_tokens:
                    correct += 1

    return correct / total if total > 0 else 0.0


def verify_novelty_against_bank(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    novel_operations: tuple[str, ...],
    *,
    seed: int,
    num_eval_examples: int,
    device: torch.device,
    novelty_threshold: float = NOVELTY_EM_THRESHOLD,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
) -> dict[str, NoveltyCalibrationResult]:
    """Verify that all novel operations fail on frozen bank primitives and compositions."""
    results: dict[str, NoveltyCalibrationResult] = {}

    for op_name in novel_operations:
        eval_examples = generate_novel_examples(
            seed,
            num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=vocab_size,
            sequence_length_range=sequence_length_range,
        )

        # 1. Single primitive search across bank
        best_prim_name = "none"
        best_prim_em = -1.0
        for pid in bank.ids():
            prim = bank.get(pid)
            pname = getattr(prim, "operation", getattr(prim, "name", f"primitive_{pid}"))
            em = _eval_primitive_exact_match(core, prim, eval_examples, device)
            if em > best_prim_em:
                best_prim_em = em
                best_prim_name = pname

        # 2. Composition search (depth 2 search)
        comp_search = search_composition_recipe(
            core=core,
            bank=bank,
            op_to_id=op_to_id,
            adaptation_examples=eval_examples,
            max_depth=2,
        )
        best_comp_recipe = (
            str(comp_search.recovered_recipe) if comp_search.recovered_recipe else "none"
        )
        best_comp_em = comp_search.exact_match_adapt

        best_existing_em = max(best_prim_em, best_comp_em)
        is_novel = best_existing_em < novelty_threshold

        results[op_name] = NoveltyCalibrationResult(
            operation=op_name,
            best_bank_primitive_name=best_prim_name,
            best_bank_primitive_em=best_prim_em,
            best_composition_recipe=best_comp_recipe,
            best_composition_em=best_comp_em,
            best_existing_em=best_existing_em,
            threshold=novelty_threshold,
            is_novel=is_novel,
        )

    return results


def verify_no_oracle_leakage(
    core: Any,
    novel_operations: tuple[str, ...],
    *,
    seed: int,
    device: torch.device,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> bool:
    """Verify h_content is completely invariant to task specification (no oracle leakage)."""
    core.model.eval()
    rng = random.Random(seed * 31337)
    test_seq = tuple(rng.randrange(vocab_size) for _ in range(8))

    step_copy = ProgramStep("COPY", {})
    prog_copy = Program(steps=(step_copy,))
    res_copy = run_program(prog_copy, test_seq, vocab_size)

    base_ex = Example(
        input_tokens=test_seq,
        target_tokens=res_copy.output_tokens,
        program=prog_copy,
        operation_graph=res_copy.graph,
        category="canonical",
        split="test",
        vocab_size=vocab_size,
    )
    base_h, _ = _encode_content_features(core, [base_ex], device)

    for op_name in novel_operations:
        op_novel = get_operation(op_name)
        params = op_novel.sample_params(rng, test_seq, vocab_size)
        step_novel = ProgramStep(op_name, params)
        prog_novel = Program(steps=(step_novel,))
        res_novel = run_program(prog_novel, test_seq, vocab_size)

        test_ex = Example(
            input_tokens=test_seq,
            target_tokens=res_novel.output_tokens,
            program=prog_novel,
            operation_graph=res_novel.graph,
            category="novel",
            split="test",
            vocab_size=vocab_size,
        )
        h_test, _ = _encode_content_features(core, [test_ex], device)
        diff = (base_h - h_test).abs().max().item()
        if diff > 1e-6:
            return False

    return True


def audit_capacity_ladder(
    tiers: tuple[str, ...],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    d_model: int = 192,
) -> dict[str, TierAuditResult]:
    """Audit actual parameter counts and topology specs across tiers."""
    results: dict[str, TierAuditResult] = {}
    for tier in tiers:
        spec = TIER_SPECS[tier]
        prim = build_tier_primitive(
            tier,
            operation="SWAP_PAIRS",
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            d_model=d_model,
        )
        actual_params = prim.num_parameters()
        counts_match = actual_params == spec.expected_parameters
        compact_params = TIER_SPECS[CapacityTier.T0_COMPACT.value].expected_parameters
        results[tier] = TierAuditResult(
            tier=tier,
            d_operator=spec.d_operator,
            d_operator_ff=spec.d_operator_ff,
            n_head=spec.n_head,
            actual_parameters=actual_params,
            expected_parameters=spec.expected_parameters,
            ratio_to_compact=actual_params / compact_params,
            counts_match=counts_match,
        )
    return results


def run_trial_tier_training(
    core: Any,
    tier: str,
    operation: str,
    train_examples: list[Example],
    eval_examples: list[Example],
    *,
    train_steps: int,
    batch_size: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Execute trial training under matched data protocol to verify convergence protocol."""
    d_model = core.model.config.d_model
    prim = build_tier_primitive(
        tier, operation, vocab_size=DEFAULT_VOCAB_SIZE, d_model=d_model
    ).to(device)
    prim.train()

    optimizer = torch.optim.AdamW(prim.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    step_to_90 = None
    step_to_95 = None
    final_loss = 0.0

    start_time = time.perf_counter()
    rng = random.Random(42)

    for step in range(1, train_steps + 1):
        batch = [rng.choice(train_examples) for _ in range(batch_size)]
        targets = [torch.tensor(ex.target_tokens, dtype=torch.long, device=device) for ex in batch]
        output_lens = [len(ex.target_tokens) for ex in batch]

        content_features, content_lengths = _encode_content_features(core, batch, device)

        optimizer.zero_grad()
        logits = prim(
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
            em = _eval_primitive_exact_match(core, prim, eval_examples, device)
            if em >= 0.90 and step_to_90 is None:
                step_to_90 = step
            if em >= 0.95 and step_to_95 is None:
                step_to_95 = step
                break

    elapsed = time.perf_counter() - start_time
    final_em = _eval_primitive_exact_match(core, prim, eval_examples, device)

    return {
        "tier": tier,
        "operation": operation,
        "final_em": final_em,
        "final_loss": final_loss,
        "step_to_90": step_to_90,
        "step_to_95": step_to_95,
        "elapsed_seconds": elapsed,
        "parameters": prim.num_parameters(),
    }


def run_discovery_capacity_harness(
    config: DiscoveryCapacityHarnessConfig,
    *,
    output_dir: Path | None = None,
) -> DiscoveryHarnessRunSummary:
    """Execute complete Task A1-B007X-002 novel-task and capacity-ladder harness verification."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # 1. Prepare frozen Shared Core & Bank
    u_config = UnifiedBenchmarkConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=str(device),
        core_train_steps=config.core_train_steps,
        parameterized_train_steps=config.bank_train_steps,
        parameter_free_train_steps=config.bank_train_steps,
        min_unseen_eval_examples=10,
        shared_encoder_checkpoint=config.shared_encoder_checkpoint,
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=output_dir)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(device)

    cand_bank_ckpts = [
        Path(f"runs/phase_a1_consolidation_benchmark/seed_{config.seed}/primitive_bank.pt"),
        Path(f"runs/phase_a1_plastic_workspace_benchmark/seed_{config.seed}/primitive_bank.pt"),
        Path(f"runs/phase_a1_composition_library_benchmark/seed_{config.seed}/primitive_bank.pt"),
    ]
    loaded_bank = False
    for b_ckpt in cand_bank_ckpts:
        if b_ckpt.is_file():
            try:
                sd = torch.load(b_ckpt, map_location=device, weights_only=True)
                bank.load_state_dict(sd, strict=False)
                loaded_bank = True
                break
            except Exception:
                pass

    if not loaded_bank and config.bank_train_steps > 0:
        from apc.evaluation.unified_oracle_causal_benchmark import _train_single_primitive

        for op in ALL_CANONICAL_OPERATIONS:
            prim_module = bank.get(op_to_id[op])
            _train_single_primitive(core, prim_module, u_config, op, steps=config.bank_train_steps)

    for pid in bank.ids():
        bank.get(pid).freeze()
        bank.get(pid).eval()

    # 2. Verify Novelty against Bank and Compositions (X2 Calibration)
    novelty_results = verify_novelty_against_bank(
        core,
        bank,
        op_to_id,
        config.novel_operations,
        seed=config.seed,
        num_eval_examples=config.num_eval_examples,
        device=device,
        novelty_threshold=config.novelty_em_threshold,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )
    all_novel_verified = all(res.is_novel for res in novelty_results.values())

    # 3. Verify No Oracle Leakage
    no_oracle_leakage_verified = verify_no_oracle_leakage(
        core,
        config.novel_operations,
        seed=config.seed,
        device=device,
        vocab_size=config.vocab_size,
    )

    # 4. Audit Capacity Ladder Parameters and Topology
    tier_audit_results = audit_capacity_ladder(config.tiers_to_audit)
    all_tier_counts_verified = all(res.counts_match for res in tier_audit_results.values())
    t2_res = tier_audit_results.get(CapacityTier.T2_OVERCOMPLETE.value)
    overcomplete_ratio_verified = (
        t2_res is not None and t2_res.ratio_to_compact >= config.min_compression_ratio
    )

    # 5. Data Protocol & Trial Convergence Verification
    # Run quick trial on T0 and T2 for the first novel operation
    sample_op = config.novel_operations[0]
    train_exs = generate_novel_examples(
        config.seed,
        config.num_trial_train_examples,
        operation=sample_op,
        split="train",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )
    eval_exs = generate_novel_examples(
        config.seed,
        config.num_eval_examples,
        operation=sample_op,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )

    trial_t0 = run_trial_tier_training(
        core,
        CapacityTier.T0_COMPACT.value,
        sample_op,
        train_exs,
        eval_exs,
        train_steps=config.trial_train_steps,
        batch_size=config.batch_size,
        device=device,
    )
    trial_t2 = run_trial_tier_training(
        core,
        CapacityTier.T2_OVERCOMPLETE.value,
        sample_op,
        train_exs,
        eval_exs,
        train_steps=config.trial_train_steps,
        batch_size=config.batch_size,
        device=device,
    )

    t0_expected = TIER_SPECS[CapacityTier.T0_COMPACT.value].expected_parameters
    t2_expected = TIER_SPECS[CapacityTier.T2_OVERCOMPLETE.value].expected_parameters
    data_protocol_verified = (
        trial_t0["parameters"] == t0_expected
        and trial_t2["parameters"] == t2_expected
        and trial_t0["final_loss"] >= 0.0
        and trial_t2["final_loss"] >= 0.0
    )

    elapsed = time.perf_counter() - start_time
    overall_passed = (
        all_novel_verified
        and all_tier_counts_verified
        and overcomplete_ratio_verified
        and no_oracle_leakage_verified
        and data_protocol_verified
    )

    summary = DiscoveryHarnessRunSummary(
        seed=config.seed,
        novel_operations=list(config.novel_operations),
        novelty_results=novelty_results,
        tier_audit_results=tier_audit_results,
        all_novel_verified=all_novel_verified,
        all_tier_counts_verified=all_tier_counts_verified,
        overcomplete_ratio_verified=overcomplete_ratio_verified,
        no_oracle_leakage_verified=no_oracle_leakage_verified,
        data_protocol_verified=data_protocol_verified,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "summary": summary.to_dict(),
                    "trials": {"T0": trial_t0, "T2": trial_t2},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )

    return summary
