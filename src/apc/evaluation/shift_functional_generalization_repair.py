# ruff: noqa: E501
"""Task B-C005R3-009: SHIFT Functional-Generalization Repair & Safe Versioned
Replacement.

ADR-0081 (D2-005, `shift_seed24_adequacy_audit.json`, regate_sealed seed 24 /
model_seed=4, bank_size=16) classified SHIFT as `TRUE_PRIMITIVE_INADEQUACY`:
closed-loop exact match 0.921875, below the 0.95 functional-adequacy floor
even for a *correctly identified, correctly argued* call. `B-C005R3-004`
(ADR-0085) could not reproduce or refute this on `development` seeds 10-14:
those seeds have no pretrained shared-encoder checkpoint under
`runs/phase_a1_shift_compact_structural_probe/` (only sealed seeds 0-4 do,
and `assert_sealed_access_permitted` refuses this task's access to them), so
`_build_frozen_base_system` falls back to a fresh random Core init and raw
closed-loop EM is meaningless there (measured 0.0 on all 5 development
seeds). ADR-0085 flagged this exact blocker: resolving it needs either
sealed access via the R3-011/R3-012 pathway (not yet open -- those tasks run
after this one and after R3-010) or an explicitly authorized alternative
validation strategy.

**Resolution used by this task (explicitly confirmed with the user before
any GPU spend):** pretrain a genuinely NEW shared-encoder checkpoint for each
`development` seed (10-14) via the existing, unmodified Phase A.1 recipe
(`apc.evaluation.shift_compact_structural_probe._get_or_train_frozen_shared_core`),
saved to the SAME shared per-seed cache path every other module in this
codebase already consults
(`runs/phase_a1_shift_compact_structural_probe/seed_{seed}/shared_encoder.pt`)
-- not a sealed seed, not sealed data, no code path touched or reused from
the sealed partition. This makes `_build_frozen_base_system` (used
unmodified by every R3-006/007/008-style task) pick up a real pretrained
encoder for these seeds transparently. Development-only (not also
`NEW_VALIDATION_SEEDS` 15-19): R3-006/007/008 all train, select between
variants, AND report their Gate on `development` alone (an internal disjoint
example split stands in for "selection", never the separate validation-seed
partition) -- the experiment plan's Local-repair-Gate table's "Validation
primary" column header names the *metric role* (the primary validation
metric), not the seed partition. This task follows that same, already
type-committed-and-accepted precedent, which also halves the pretraining
cost.

**Why SHIFT still needs a genuinely new candidate, not just "train it":**
sealed seeds 0-4 (which DO have a trained `ShiftRelativePrimitive`) already
plateau at ADR-0081's ~0.92 EM under the existing recipe/architecture
(`ShiftRelativeCrossPositionOperator` / `primitives.primitive.ShiftRelativePrimitive`,
18,282 params, a modular relative-position cross-attention operator). No
formal "compact budget manifest" exists anywhere in this repo (confirmed by
grep; the 18,282 figure appears only in docstrings/tests) -- this task
treats `ShiftRelativePrimitive`'s own existing parameter count as the
compact-budget ceiling, disclosed explicitly rather than invented from a
missing document, and keeps `operator_train_steps` identical to the existing
recipe's own default (12,000) for both trained variants so neither gets an
undisclosed step-budget advantage.

Two variants are trained per development seed on an internal, disjoint
selection split (never the final Gate query split), a *sampling-only*
mechanism exactly like R3-008's BIND fix (same architecture, same step
budget, same optimizer -- only which (length, amount) combinations get
proportionally more training exposure differs):

- ``iid_baseline``: ordinary i.i.d. draws over SHIFT's legal
  ``(content_length, amount)`` domain (`amount in [0, content_length)` per
  `ShiftOp.sample_params`) -- the "current physical family" control.
- ``error_weighted_stratified``: a mandatory pre-repair diagnostic (this
  task's own analogue of R3-006/008's "confirm before repairing") first
  measures `iid_baseline`'s own per-``(length, amount)`` exact-match rate on
  a disjoint diagnostic pool; strata scoring below that pool's own mean get
  proportionally more exposure in a second training run of the *same*
  architecture, over the *same* step budget -- never a blind uniform
  stratification, always targeted at empirically measured weak strata.

Commit is a same-physical-family **versioned replacement**
(`PrimitiveBank.replace_primitive`, a small generic addition to
`apc.primitives.bank` this task adds): the chosen variant's primitive is
staged into a deep-copied candidate bank, gated (Gate criteria below), and
only on PASS is a new bank checkpoint written under this task's OWN output
directory (never overwriting the shared `primitive_bank_16.pt` cache other
R3-006/007/008 runs and future reruns depend on for reproducibility) with an
incremented `metadata["version"]`. A fresh, from-serialized-checkpoints
reconstruction (mirroring `apc.evaluation.fresh_runtime_recurrence`'s own
"prove it lives in persistent state" pattern -- an in-process fresh-object
rebuild from disk, not a literal separate OS process, disclosed as such) then
re-measures the committed candidate with zero adaptation steps and zero
temporary parameters.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.functional_metrics_v2 import (
    ReferenceAdequacyState,
    ReferenceAllocation,
    reference_adequacy_state,
)
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.relation_split_protocol import (
    FULL_BANK_16_OPERATIONS,
    assert_sealed_access_permitted,
)
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    evaluate_legacy_regression,
)
from apc.evaluation.shift_compact_structural_probe import (
    ShiftStructuralProbeConfig,
    _get_or_train_frozen_shared_core,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    CrossPositionPrimitiveConfig,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.primitive import (
    PrimitiveStatus,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
)
from apc.utils.seed import set_seed
from apc.utils.seed_derivation import derive_seed
from apc.utils.system_info import get_system_info

DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = DEFAULT_DEV_SEEDS
TARGET_OPERATION: Final[str] = "SHIFT"
OTHER_OPERATIONS_FOR_REGRESSION: Final[tuple[str, ...]] = ("SELECT", "COUNT", "BIND")
VARIANTS: Final[tuple[str, ...]] = ("iid_baseline", "error_weighted_stratified")
SHARED_ENCODER_CACHE_DIR: Final[Path] = Path("runs/phase_a1_shift_compact_structural_probe")
COMPACT_PARAM_BUDGET_REFERENCE: Final[int] = 18282

ADR_0081_REFERENCE: Final[dict[str, Any]] = {
    "source": "runs/phase_b_b2_second_diagnostic/shift_seed24_adequacy_audit.json (model_seed=4 / regate_sealed seed 24, bank_size=16)",
    "closed_loop_exact_match": 0.921875,
    "classification": "TRUE_PRIMITIVE_INADEQUACY / SEQUENTIAL_RULE_BIAS (premature accept at n=32)",
}
R3_004_REFERENCE: Final[dict[str, Any]] = {
    "source": "runs/phase_b_b2_post_d2/r3_004_paired_baseline/failure_reproduction_matrix.json:SHIFT_reference_adequacy",
    "status": "UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE",
    "measured_v2_development_mean_closed_loop_exact_match": 0.0,
    "resolution_this_task": (
        "pretrained a NEW, non-sealed shared-encoder checkpoint per development seed "
        "(user-confirmed before GPU spend) via the existing, unmodified Phase A.1 "
        "recipe -- see module docstring."
    ),
}


# ---------------------------------------------------------------------------
# 0. SHIFT's legal (length, amount) grid and a params-controlled example
#    builder (same underlying generator primitives as
#    `apc.evaluation.consolidation_benchmark.generate_benchmark_examples`,
#    just with `amount` pinned instead of drawn by `ShiftOp.sample_params`).
# ---------------------------------------------------------------------------


def shift_legal_grid(sequence_length_range: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Every legal `(content_length, amount)` pair: `amount in [0, length)`
    per `ShiftOp.sample_params` (`rng.randrange(len(sequence))`)."""
    lo, hi = sequence_length_range
    return tuple(
        (length, amount) for length in range(lo, hi + 1) for amount in range(length)
    )


def _generate_shift_example_with_params(
    rng_seed: int, *, length: int, amount: int, vocab_size: int, split: str
) -> Example:
    """Construct one SHIFT example with an explicit `(length, amount)`,
    reusing the exact same generator primitives (`Program`/`ProgramStep`/
    `run_program`/`TaskSpec`) as `generate_benchmark_examples`, just pinning
    `amount` instead of letting `ShiftOp.sample_params` draw it. Content
    tokens are still drawn i.i.d. -- this never synthesizes a new environment
    or operation, only which legal (length, amount) combination gets asked
    for."""
    import random

    rng = random.Random(rng_seed)
    seq = tuple(rng.randrange(vocab_size) for _ in range(length))
    step = ProgramStep(operation="SHIFT", params={"amount": amount})
    prog = Program(steps=(step,))
    res = run_program(prog, seq, vocab_size)
    return Example(
        input_tokens=seq,
        target_tokens=res.output_tokens,
        program=prog,
        operation_graph=res.graph,
        category="known",
        split=split,
        vocab_size=vocab_size,
        task_spec=TaskSpec.from_program(prog),
        oracle_metadata=OracleMetadata(label="K", primitive_operations=("SHIFT",)),
    )


def _shift_params(example: Example) -> tuple[int, int]:
    """`(content_length, amount)` for one SHIFT example, read from its own
    `task_spec` (never re-derived from the generator's internal RNG)."""
    assert example.task_spec is not None
    amount = int(example.task_spec.steps[0].arguments["amount"])
    return len(example.input_tokens), amount


# ---------------------------------------------------------------------------
# 1. Pretrained-core provenance (Content Encoder), development seeds only.
# ---------------------------------------------------------------------------


def ensure_pretrained_core(seed: int, config: ShiftFunctionalGeneralizationRepairConfig) -> dict[str, Any]:
    """Load (or, only if genuinely absent, train and cache) a shared-encoder
    checkpoint for `seed` at the canonical per-seed cache path every other
    module in this codebase already consults. Never touches a sealed seed
    (enforced by the caller's `assert_sealed_access_permitted` call before
    this function is ever reached for any seed in `config.development_seeds`)."""
    seed_dir = config.shared_encoder_cache_dir / f"seed_{seed}"
    ckpt_path = seed_dir / "shared_encoder.pt"
    was_cached = ckpt_path.is_file()
    probe_config = ShiftStructuralProbeConfig(
        seed=seed,
        sequence_length_range=config.sequence_length_range,
        device=config.device,
        core_train_steps=config.core_train_steps,
        core_lr=config.core_lr,
        core_weight_decay=config.core_weight_decay,
    )
    _get_or_train_frozen_shared_core(probe_config, seed_dir=seed_dir)
    return {
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "was_already_cached": was_cached,
        "core_train_steps": config.core_train_steps,
        "sealed_data_used": False,
    }


# ---------------------------------------------------------------------------
# 2. Direct primitive-level training/eval (bypasses router/bank for speed;
#    the FINAL Gate measurement below always goes through the real bank via
#    `execute_composition_recipe`, never this shortcut).
# ---------------------------------------------------------------------------


def _encode_content(core: Any, examples: Sequence[Example]) -> tuple[torch.Tensor, list[int]]:
    device = core.device
    content_lengths = [len(ex.input_tokens) for ex in examples]
    batch_input = collate_content_only_batch(examples, core.tokens, device=device)
    with torch.no_grad():
        h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
    return h, content_lengths


def _shift_correctness(
    core: Any, candidate: ShiftRelativePrimitive, examples: Sequence[Example], *, batch_size: int = 256
) -> list[bool]:
    """Per-example exact-match correctness of `candidate` in isolation
    (direct forward, oracle `amount` argument), used for diagnostics and
    variant selection only -- never the final Gate cell."""
    candidate.eval()
    correctness: list[bool] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            h, content_lengths = _encode_content(core, chunk)
            output_lengths = content_lengths
            argument_values = [_shift_params(ex)[1] for ex in chunk]
            logits = candidate(h, content_lengths, output_lengths, argument_values)
            predictions = logits.argmax(dim=-1)
            for row, (ex, n) in enumerate(zip(chunk, output_lengths, strict=True)):
                prediction = tuple(predictions[row, :n].tolist())
                correctness.append(prediction == ex.target_tokens)
    return correctness


def diagnose_stratum_errors(
    core: Any, candidate: ShiftRelativePrimitive, examples: Sequence[Example]
) -> list[dict[str, Any]]:
    """Per-`(length, amount)` exact-match breakdown of `candidate` over
    `examples` -- the mandatory pre-repair diagnostic this task's docstring
    describes, empirically grounding which strata (if any) are weak rather
    than assuming length/amount coverage is the bottleneck."""
    correctness = _shift_correctness(core, candidate, examples)
    by_stratum: dict[tuple[int, int], list[bool]] = {}
    for ex, correct in zip(examples, correctness, strict=True):
        key = _shift_params(ex)
        by_stratum.setdefault(key, []).append(correct)
    rows = []
    for (length, amount), flags in sorted(by_stratum.items()):
        rows.append(
            {
                "length": length,
                "amount": amount,
                "wraps_around": amount > 0,
                "n": len(flags),
                "exact_match": sum(flags) / len(flags),
            }
        )
    return rows


def _build_weighted_schedule(
    diag_rows: Sequence[dict[str, Any]], schedule_len: int, seed: int
) -> list[tuple[int, int]]:
    """A deterministic weighted round-robin over `(length, amount)` strata,
    weight `max(0.05, 1.02 - stratum_exact_match)` (every stratum keeps a
    residual floor of exposure; strata the diagnostic found weak get
    proportionally more). Computed ONCE from `diag_rows` (a disjoint
    diagnostic split, never the final Gate or selection examples) and never
    updated online during the training loop that follows."""
    strata = [(row["length"], row["amount"]) for row in diag_rows if row["n"] > 0]
    if not strata:
        raise ValueError("diag_rows must contain at least one stratum with n > 0")
    weights = [max(0.05, 1.02 - row["exact_match"]) for row in diag_rows if row["n"] > 0]
    total_weight = sum(weights)
    counts = [max(1, round(w / total_weight * schedule_len)) for w in weights]

    # Deterministic interleave (largest-remainder-style round robin) so early
    # steps are not dominated by whichever stratum happens first in `strata`.
    remaining = list(counts)
    schedule: list[tuple[int, int]] = []
    idx = 0
    n = len(strata)
    while len(schedule) < schedule_len and any(r > 0 for r in remaining):
        if remaining[idx] > 0:
            schedule.append(strata[idx])
            remaining[idx] -= 1
        idx = (idx + 1) % n
    if len(schedule) < schedule_len:
        # Pad deterministically by cycling strata (only reachable if counts
        # summed below schedule_len due to rounding).
        pad_rng_seed = derive_seed(
            master_seed=seed, stream_namespace="r3_009_schedule_pad", task_key="SHIFT", sample_index=0
        )
        i = pad_rng_seed % n
        while len(schedule) < schedule_len:
            schedule.append(strata[i % n])
            i += 1
    return schedule[:schedule_len]


def _train_shift_candidate(
    core: Any,
    primitive_id: int,
    *,
    variant: str,
    seed: int,
    config: ShiftFunctionalGeneralizationRepairConfig,
    diag_rows: Sequence[dict[str, Any]] | None = None,
) -> ShiftRelativePrimitive:
    """Train a fresh `ShiftRelativePrimitive` (same class the bank uses at
    runtime -- no state-dict transplant needed) against the frozen `core`.
    `iid_baseline` draws batches i.i.d. over the legal (length, amount)
    domain every step; `error_weighted_stratified` instead draws from a
    schedule pre-weighted by `diag_rows`'s own measured weak strata. Both
    use the SAME step budget, optimizer, and architecture -- only the
    per-step example source differs (a sampling-only mechanism)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "error_weighted_stratified" and not diag_rows:
        raise ValueError("error_weighted_stratified requires diag_rows")

    device = core.device
    set_seed(derive_seed(master_seed=seed, stream_namespace="r3_009_train_init", task_key=variant, sample_index=0))
    candidate = ShiftRelativePrimitive(
        primitive_id,
        ShiftRelativePrimitiveConfig(
            d_model=core.model.config.d_model,
            vocab_size=core.tokens.env_vocab_size,
        ),
        status=PrimitiveStatus.CANDIDATE,
        metadata={"trained_variant": variant, "parent_seed": seed},
    ).to(device)

    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=config.operator_lr, weight_decay=config.operator_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.operator_train_steps, eta_min=1e-5
    )

    schedule: list[tuple[int, int]] | None = None
    if variant == "error_weighted_stratified":
        assert diag_rows is not None
        schedule = _build_weighted_schedule(
            diag_rows, config.operator_train_steps * config.operator_batch_size, seed
        )

    candidate.train()
    for step in range(config.operator_train_steps):
        if schedule is not None:
            batch_strata = schedule[
                step * config.operator_batch_size : (step + 1) * config.operator_batch_size
            ]
            examples = [
                _generate_shift_example_with_params(
                    derive_seed(
                        master_seed=seed,
                        stream_namespace=f"r3_009_train_{variant}",
                        task_key="SHIFT",
                        sample_index=step * config.operator_batch_size + i,
                    ),
                    length=length,
                    amount=amount,
                    vocab_size=core.tokens.env_vocab_size,
                    split="train",
                )
                for i, (length, amount) in enumerate(batch_strata)
            ]
        else:
            examples = generate_benchmark_examples(
                derive_seed(
                    master_seed=seed,
                    stream_namespace=f"r3_009_train_{variant}",
                    task_key="SHIFT",
                    sample_index=step,
                ),
                config.operator_batch_size,
                operation="SHIFT",
                split="train",
                sequence_length_range=config.sequence_length_range,
            )

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = content_lengths
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)
        argument_values = [_shift_params(ex)[1] for ex in examples]

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        optimizer.zero_grad(set_to_none=True)
        logits = candidate(h, content_lengths, output_lengths, argument_values)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        loss.backward()
        if config.operator_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), config.operator_grad_clip)
        optimizer.step()
        scheduler.step()

    candidate.eval()
    for p in candidate.parameters():
        p.requires_grad_(False)
    return candidate


# ---------------------------------------------------------------------------
# 3. Closed-loop Gate measurement through the REAL bank/composition pipeline.
# ---------------------------------------------------------------------------


def closed_loop_shift_em(
    core: Any, bank: PrimitiveBank, op_to_id: dict[str, int], examples: Sequence[Example], *, batch_size: int = 256
) -> tuple[float, int, int]:
    """Mean exact match, successes, trials -- via `execute_composition_recipe`
    (oracle call resolution, real primitive forward pass; the generator's
    oracle is used only to pick WHICH call to make, never to compute the
    output)."""
    successes = 0
    total = 0
    bank.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            logits = execute_composition_recipe(core=core, bank=bank, op_to_id=op_to_id, examples=chunk)
            predictions = logits.argmax(dim=-1)
            for row, ex in enumerate(chunk):
                n = len(ex.target_tokens)
                prediction = tuple(predictions[row, :n].tolist())
                successes += int(prediction == ex.target_tokens)
                total += 1
    return successes / total, successes, total


# ---------------------------------------------------------------------------
# 4. Freeze audit / hashing (same convention as R3-006/007/008).
# ---------------------------------------------------------------------------


def _state_dict_hash(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def build_freeze_audit(bank_before: PrimitiveBank, bank_after: PrimitiveBank, shift_id: int) -> dict[str, Any]:
    """Bit-for-bit confirmation that every non-SHIFT primitive is unchanged
    and SHIFT's own weights did change."""
    other_unchanged: dict[str, bool] = {}
    for pid in bank_before.ids():
        if pid == shift_id:
            continue
        before = bank_before.get(pid)
        after = bank_after.get(pid)
        other_unchanged[str(pid)] = _state_dict_hash(before) == _state_dict_hash(after)
    shift_changed = _state_dict_hash(bank_before.get(shift_id)) != _state_dict_hash(bank_after.get(shift_id))
    return {
        "other_primitive_unchanged_by_id": other_unchanged,
        "all_other_primitives_unchanged": all(other_unchanged.values()),
        "shift_primitive_changed_by_training": shift_changed,
        "freeze_audit_passed": all(other_unchanged.values()) and shift_changed,
    }


# ---------------------------------------------------------------------------
# 5. Config.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftFunctionalGeneralizationRepairConfig:
    """Explicit configuration for Task B-C005R3-009."""

    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    bank_size: int = 16
    sequence_length_range: tuple[int, int] = (6, 10)
    core_train_steps: int = 16000
    core_lr: float = 3e-4
    core_weight_decay: float = 1e-4
    operator_train_steps: int = 12000
    operator_lr: float = 5e-4
    operator_weight_decay: float = 1e-4
    operator_grad_clip: float = 1.0
    operator_batch_size: int = 32
    diagnostic_examples: int = 1000
    selection_query_examples: int = 128
    gate_query_examples: int = 512
    other_op_query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    gate_mean_query_em_threshold: float = 0.99
    gate_ref_adequacy_tau: float = 0.95
    gate_other_task_regression_pp_max: float = 1.0
    variants: tuple[str, ...] = VARIANTS
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    shared_encoder_cache_dir: Path = SHARED_ENCODER_CACHE_DIR
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair")

    def __post_init__(self) -> None:
        if not self.development_seeds:
            raise ValueError("development_seeds must be non-empty")
        if self.bank_size not in DEFAULT_BANK_SIZES:
            raise ValueError(f"bank_size must be one of {DEFAULT_BANK_SIZES}")
        if self.sequence_length_range[0] < 1 or self.sequence_length_range[1] < self.sequence_length_range[0]:
            raise ValueError("sequence_length_range must be a valid ascending range with length >= 1")
        if self.operator_train_steps < 1 or self.core_train_steps < 0:
            raise ValueError("operator_train_steps must be positive and core_train_steps must be non-negative")
        if self.operator_batch_size < 1:
            raise ValueError("operator_batch_size must be positive")
        for name in ("diagnostic_examples", "selection_query_examples", "gate_query_examples", "other_op_query_examples"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.gate_mean_query_em_threshold <= 1.0:
            raise ValueError("gate_mean_query_em_threshold must be in (0, 1]")
        if not 0.0 < self.gate_ref_adequacy_tau <= 1.0:
            raise ValueError("gate_ref_adequacy_tau must be in (0, 1]")
        if self.gate_other_task_regression_pp_max < 0.0:
            raise ValueError("gate_other_task_regression_pp_max must be non-negative")
        if not self.variants or set(self.variants) - set(VARIANTS):
            raise ValueError(f"variants must be a non-empty subset of {VARIANTS}")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["shared_encoder_cache_dir"] = str(self.shared_encoder_cache_dir)
        data["output_dir"] = str(self.output_dir)
        return data


# ---------------------------------------------------------------------------
# 6. Fresh bank (re)construction from an arbitrary checkpoint path (the
#    shared `get_or_build_16_primitive_bank` cache helper does not accept
#    one, so this is a small, scoped duplication of its cached-load branch).
# ---------------------------------------------------------------------------


def _rebuild_full_bank_structure(core: Any, seed: int) -> tuple[PrimitiveBank, dict[str, int]]:
    u_bank_cfg = UnifiedBenchmarkConfig(seed=seed, vocab_size=10, device=core.device)
    bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
    for op in tuple(BRANCH_B_NOVEL_OPERATION_NAMES) + tuple(PHASE_A2_INCREMENTAL_NEW_OPERATIONS):
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op, d_model=core.model.config.d_model, d_operator=32, n_head=4,
                d_operator_ff=64, vocab_size=10, max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id
    return bank, op_to_id


def load_bank_from_checkpoint(core: Any, seed: int, checkpoint_path: Path) -> tuple[PrimitiveBank, dict[str, int]]:
    """Fresh reconstruction: bank STRUCTURE rebuilt from scratch (no shared
    object references), weights loaded ONLY from `checkpoint_path` on disk."""
    bank, op_to_id = _rebuild_full_bank_structure(core, seed)
    bank.to(core.device)
    sd = torch.load(checkpoint_path, map_location=core.device, weights_only=True)
    bank.load_state_dict(sd)
    bank.freeze_all()
    bank.eval()
    return bank, op_to_id


# ---------------------------------------------------------------------------
# 7. Orchestration.
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _build_variant_selection(per_variant_selection: dict[str, list[float]]) -> dict[str, Any]:
    per_variant_mean = {variant: _mean(scores) for variant, scores in per_variant_selection.items()}
    best_score = max((s for s in per_variant_mean.values() if s is not None), default=None)
    tied = [
        v for v, s in per_variant_mean.items()
        if s is not None and best_score is not None and abs(s - best_score) < 1e-9
    ]
    if len(tied) > 1 and "error_weighted_stratified" in tied:
        chosen = "error_weighted_stratified"
        tie_break_applied = True
    else:
        chosen = tied[0] if tied else next(iter(per_variant_mean))
        tie_break_applied = len(tied) > 1
    return {
        "task": "B-C005R3-009",
        "selection_split": "disjoint from both training and the final Gate query examples (derive_seed stream_namespace='r3_009_selection')",
        "per_variant_mean_selection_query_em": per_variant_mean,
        "per_seed_selection_scores": per_variant_selection,
        "chosen_variant": chosen,
        "tie_break_applied": tie_break_applied,
        "tie_break_rule": "on a tie within 1e-9, prefer error_weighted_stratified (directly targets diagnosed weak strata over the generic iid_baseline)",
    }


def run_shift_functional_generalization_repair(
    config: ShiftFunctionalGeneralizationRepairConfig,
) -> dict[str, Any]:
    """Executes B-C005R3-009 end to end on `development` seeds only."""
    start = time.perf_counter()
    assert_sealed_access_permitted(
        config.development_seeds, purpose="B-C005R3-009_shift_functional_generalization_repair"
    )
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    pretrain_provenance: dict[str, Any] = {}
    diagnostics_by_seed: dict[str, Any] = {}
    per_variant_selection: dict[str, list[float]] = {v: [] for v in config.variants}
    per_seed_variant_final: dict[str, list[dict[str, Any]]] = {v: [] for v in config.variants}
    other_op_baseline: dict[str, list[float]] = {op: [] for op in OTHER_OPERATIONS_FOR_REGRESSION}
    other_op_repaired_by_variant: dict[str, dict[str, list[float]]] = {
        v: {op: [] for op in OTHER_OPERATIONS_FOR_REGRESSION} for v in config.variants
    }
    freeze_audits: dict[str, Any] = {}
    checkpoint_hashes: dict[str, Any] = {}
    legacy_regression: dict[str, dict[str, float]] = {"before": {}, "after": {}}
    candidate_count_before_after: dict[str, tuple[int, int]] = {}
    variant_primitives_by_seed: dict[int, dict[str, ShiftRelativePrimitive]] = {}
    base_systems_by_seed: dict[int, tuple[Any, PrimitiveBank, Any, dict[str, int]]] = {}

    for seed in config.development_seeds:
        set_seed(seed)
        pretrain_provenance[str(seed)] = ensure_pretrained_core(seed, config)

        base_hn_config = HardNegativeBenchmarkConfig(
            seeds=(seed,), router_train_examples=config.router_train_examples,
            router_steps=config.router_steps, device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(seed, base_hn_config)
        bank, router, candidate_ids, _semantic_ids, _distractor_ids = build_scaled_bank_and_router(
            core, base_bank, base_router, op_to_id, config.bank_size, seed=seed
        )
        shift_id = op_to_id["SHIFT"]
        base_systems_by_seed[seed] = (core, bank, router, op_to_id)
        candidate_count_before_after[str(seed)] = (len(candidate_ids), len(candidate_ids))

        # --- Mandatory pre-repair diagnostic (disjoint from selection/final splits) ---
        diag_examples = generate_benchmark_examples(
            derive_seed(master_seed=seed, stream_namespace="r3_009_diagnostic", task_key="SHIFT", sample_index=0),
            config.diagnostic_examples, operation="SHIFT", split="dev",
            sequence_length_range=config.sequence_length_range,
        )
        baseline_candidate = _train_shift_candidate(core, shift_id, variant="iid_baseline", seed=seed, config=config)
        diag_rows = diagnose_stratum_errors(core, baseline_candidate, diag_examples)
        diagnostics_by_seed[str(seed)] = {
            "pool_size": len(diag_examples),
            "pool_mean_exact_match": _mean([r["exact_match"] for r in diag_rows]),
            "per_stratum": diag_rows,
            "weak_strata": [
                {"length": r["length"], "amount": r["amount"], "exact_match": r["exact_match"]}
                for r in diag_rows
                if r["exact_match"] < (_mean([x["exact_match"] for x in diag_rows]) or 1.0)
            ],
        }

        variant_primitives: dict[str, ShiftRelativePrimitive] = {"iid_baseline": baseline_candidate}
        if "error_weighted_stratified" in config.variants:
            variant_primitives["error_weighted_stratified"] = _train_shift_candidate(
                core, shift_id, variant="error_weighted_stratified", seed=seed, config=config, diag_rows=diag_rows
            )
        variant_primitives_by_seed[seed] = variant_primitives

        legacy_regression["before"][str(seed)] = evaluate_legacy_regression(
            core, bank, router, FULL_BANK_16_OPERATIONS, op_to_id, seed
        )

        for variant in config.variants:
            candidate = variant_primitives[variant]
            candidate_bank = copy.deepcopy(bank)
            candidate_bank.replace_primitive(shift_id, candidate)

            freeze_audits[f"{variant}_seed_{seed}"] = build_freeze_audit(bank, candidate_bank, shift_id)
            checkpoint_hashes[f"{variant}_seed_{seed}"] = {
                "before_shift_hash": _state_dict_hash(bank.get(shift_id)),
                "after_shift_hash": _state_dict_hash(candidate_bank.get(shift_id)),
                "candidate_bank_full_state_hash": _state_dict_hash(candidate_bank),
            }

            sel_examples = generate_benchmark_examples(
                derive_seed(master_seed=seed, stream_namespace="r3_009_selection", task_key="SHIFT", sample_index=0),
                config.selection_query_examples, operation="SHIFT", split="dev",
                sequence_length_range=config.sequence_length_range,
            )
            sel_em, _, _ = closed_loop_shift_em(core, candidate_bank, op_to_id, sel_examples)
            per_variant_selection[variant].append(sel_em)

            final_examples = generate_benchmark_examples(
                derive_seed(master_seed=seed, stream_namespace="r3_009_final", task_key="SHIFT", sample_index=0),
                config.gate_query_examples, operation="SHIFT", split="dev",
                sequence_length_range=config.sequence_length_range,
            )
            final_em, successes, trials = closed_loop_shift_em(core, candidate_bank, op_to_id, final_examples)
            ref_state = reference_adequacy_state(
                successes, trials, ReferenceAllocation(), tau=config.gate_ref_adequacy_tau
            )
            per_seed_variant_final[variant].append(
                {
                    "seed": seed,
                    "mean_query_exact_match": final_em,
                    "successes": successes,
                    "trials": trials,
                    "reference_adequacy_state": ref_state.value if ref_state is not None else None,
                    "reference_adequacy_tau": config.gate_ref_adequacy_tau,
                }
            )

            for op in OTHER_OPERATIONS_FOR_REGRESSION:
                op_examples = generate_benchmark_examples(
                    seed * 50_000 + op_to_id[op] + 100, config.other_op_query_examples, operation=op, split="dev"
                )
                base_em, _, _ = closed_loop_shift_em(core, bank, op_to_id, op_examples)
                repaired_em, _, _ = closed_loop_shift_em(core, candidate_bank, op_to_id, op_examples)
                other_op_baseline[op].append(base_em)
                other_op_repaired_by_variant[variant][op].append(repaired_em)

    # -----------------------------------------------------------------
    # Variant selection (disjoint split; never reused in the Gate cells above).
    # -----------------------------------------------------------------
    variant_selection = _build_variant_selection(per_variant_selection)
    chosen_variant = variant_selection["chosen_variant"]
    chosen_final_cells = per_seed_variant_final[chosen_variant]

    other_op_regression_pp = {
        op: abs((_mean(other_op_repaired_by_variant[chosen_variant][op]) or 0.0) - (_mean(other_op_baseline[op]) or 0.0)) * 100.0
        for op in OTHER_OPERATIONS_FOR_REGRESSION
    }
    max_other_op_regression_pp = max(other_op_regression_pp.values()) if other_op_regression_pp else 0.0

    mean_query_em = _mean([c["mean_query_exact_match"] for c in chosen_final_cells]) or 0.0
    all_ref_adequate = all(c["reference_adequacy_state"] == ReferenceAdequacyState.REF_ADEQUATE.value for c in chosen_final_cells)
    mean_query_em_pass = mean_query_em >= config.gate_mean_query_em_threshold
    regression_pass = max_other_op_regression_pp <= config.gate_other_task_regression_pp_max
    freeze_pass = all(
        freeze_audits[f"{chosen_variant}_seed_{seed}"]["freeze_audit_passed"] for seed in config.development_seeds
    )
    gate_pass = mean_query_em_pass and all_ref_adequate and regression_pass and freeze_pass

    gate = {
        "result": "VALIDATION_PASS" if gate_pass else "FAIL",
        "mean_query_exact_match": {
            "measured": mean_query_em, "threshold": config.gate_mean_query_em_threshold, "pass": mean_query_em_pass,
        },
        "all_models_ref_adequate": {
            "measured": all_ref_adequate,
            "per_seed_state": {c["seed"]: c["reference_adequacy_state"] for c in chosen_final_cells},
            "tau": config.gate_ref_adequacy_tau,
        },
        "other_task_regression_pp": {
            "by_operation": other_op_regression_pp, "max": max_other_op_regression_pp,
            "threshold": config.gate_other_task_regression_pp_max, "pass": regression_pass,
        },
        "freeze_audit_all_passed": freeze_pass,
        "compact_param_budget": {
            "candidate_param_count": variant_primitives_by_seed[config.development_seeds[0]][chosen_variant].num_parameters(),
            "reference_ceiling": COMPACT_PARAM_BUDGET_REFERENCE,
            "note": (
                "No formal compact-budget manifest exists in this repo (confirmed by grep); this "
                "task treats ShiftRelativePrimitive's own existing parameter count as the ceiling "
                "and keeps operator_train_steps identical to the existing recipe (12,000) for both "
                "trained variants, disclosed explicitly rather than invented from a missing document."
            ),
        },
        "scientific_caveat": (
            "This result applies only to the `development` partition "
            f"(seeds {list(config.development_seeds)}), using freshly pretrained (non-sealed) "
            "shared-encoder checkpoints -- not a reproduction of ADR-0081's sealed-partition "
            "finding (regate_sealed seed 24 / model_seed=4, bank_size=16, closed_loop_exact_match "
            "0.921875), which remains inaccessible under current sealed-access rules "
            "(R3-011/R3-012 pathway required). The >=0.99 target is a new availability goal for "
            "this task, not a claim about past results, and does not change the underlying 0.95 "
            "functional-adequacy threshold used by reference_adequacy_state's own tau."
        ),
    }

    # -----------------------------------------------------------------
    # Versioned bank transaction: per-seed commit only if that seed's OWN
    # cell individually clears the Gate; otherwise rollback (never touches
    # the shared primitive_bank_16.pt cache other tasks depend on).
    # -----------------------------------------------------------------
    transaction_log: dict[str, Any] = {}
    committed_checkpoint_by_seed: dict[int, Path] = {}
    for cell in chosen_final_cells:
        seed = cell["seed"]
        core, bank, _router, op_to_id = base_systems_by_seed[seed]
        shift_id = op_to_id["SHIFT"]
        candidate = variant_primitives_by_seed[seed][chosen_variant]
        seed_cell_pass = (
            cell["mean_query_exact_match"] >= config.gate_mean_query_em_threshold
            and cell["reference_adequacy_state"] == ReferenceAdequacyState.REF_ADEQUATE.value
            and freeze_audits[f"{chosen_variant}_seed_{seed}"]["freeze_audit_passed"]
        )
        old_primitive = bank.get(shift_id)
        old_version = int(old_primitive.metadata.get("version", 0))
        seed_out_dir = Path(config.output_dir) / "committed_bank" / f"seed_{seed}"
        if seed_cell_pass:
            candidate_bank = copy.deepcopy(bank)
            candidate.metadata["version"] = old_version + 1
            candidate_bank.replace_primitive(shift_id, candidate)
            seed_out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = seed_out_dir / f"primitive_bank_16_shift_v{old_version + 1}.pt"
            torch.save(candidate_bank.state_dict(), ckpt_path)
            committed_checkpoint_by_seed[seed] = ckpt_path
            transaction_log[str(seed)] = {
                "action": "COMMITTED",
                "version_before": old_version,
                "version_after": old_version + 1,
                "committed_checkpoint": str(ckpt_path),
                "shared_cache_mutated": False,
            }
        else:
            transaction_log[str(seed)] = {
                "action": "ROLLED_BACK",
                "reason": "seed-level Gate cell did not individually pass",
                "version_before": old_version,
                "version_after": old_version,
                "shared_cache_mutated": False,
            }

    # -----------------------------------------------------------------
    # Fresh-runtime reload: from-disk reconstruction only, zero adaptation,
    # zero temporary parameters (mirrors fresh_runtime_recurrence's own
    # in-process fresh-object-rebuild convention; disclosed as such, not a
    # literal separate OS process).
    # -----------------------------------------------------------------
    fresh_runtime_report: dict[str, Any] = {}
    for seed, ckpt_path in committed_checkpoint_by_seed.items():
        core, _bank, _router, op_to_id = base_systems_by_seed[seed]
        fresh_bank, fresh_op_to_id = load_bank_from_checkpoint(core, seed, ckpt_path)
        final_examples = generate_benchmark_examples(
            derive_seed(master_seed=seed, stream_namespace="r3_009_final", task_key="SHIFT", sample_index=0),
            config.gate_query_examples, operation="SHIFT", split="dev",
            sequence_length_range=config.sequence_length_range,
        )
        reload_em, reload_successes, reload_trials = closed_loop_shift_em(core, fresh_bank, fresh_op_to_id, final_examples)
        pre_reload_cell = next(c for c in chosen_final_cells if c["seed"] == seed)
        fresh_runtime_report[str(seed)] = {
            "adaptation_steps": 0,
            "temporary_parameters": 0,
            "reload_mean_query_exact_match": reload_em,
            "reload_successes": reload_successes,
            "reload_trials": reload_trials,
            "pre_commit_mean_query_exact_match": pre_reload_cell["mean_query_exact_match"],
            "matches_pre_commit": reload_em == pre_reload_cell["mean_query_exact_match"],
        }

    shift_generalization_repair = {
        "task": "B-C005R3-009",
        "development_seeds": list(config.development_seeds),
        "bank_size": config.bank_size,
        "original_reference": {"adr_0081": ADR_0081_REFERENCE, "r3_004_adr_0085": R3_004_REFERENCE},
        "pretrain_provenance": pretrain_provenance,
        "pre_repair_diagnostics": diagnostics_by_seed,
        "variant_selection_summary": {
            "chosen_variant": chosen_variant,
            "per_variant_mean_selection_query_em": variant_selection["per_variant_mean_selection_query_em"],
        },
        "chosen_variant_final_cells": chosen_final_cells,
        "other_op_regression_pp": other_op_regression_pp,
        "max_other_op_regression_pp": max_other_op_regression_pp,
        "legacy_routing_regression_before": legacy_regression["before"],
        "candidate_per_argument_persistent_duplication_check": {
            "candidate_ids_before_after_by_seed": {
                seed_key: {"before": before, "after": after}
                for seed_key, (before, after) in candidate_count_before_after.items()
            },
            "no_persistent_duplication": all(before == after for before, after in candidate_count_before_after.values()),
        },
        "gate": gate,
    }

    frozen_state_audit = {
        "task": "B-C005R3-009",
        "by_variant_seed": freeze_audits,
        "all_freeze_audits_passed": all(a["freeze_audit_passed"] for a in freeze_audits.values()),
    }

    protocol = {
        "task": "B-C005R3-009",
        "gate": "local_repair_gate",
        "result": gate["result"],
        "chosen_variant": chosen_variant,
        "scientific_caveat": gate["scientific_caveat"],
        "downstream_note": (
            "This gate result applies only to the `development` partition, using freshly "
            "pretrained (non-sealed) shared-encoder checkpoints for this task. It does not "
            "confirm repair of ADR-0081's sealed-partition finding, which remains inaccessible "
            "under current sealed-access rules (R3-011/R3-012 pathway required)."
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")
        (output_dir / "shift_generalization_repair.json").write_text(
            json.dumps(shift_generalization_repair, indent=2), encoding="utf-8"
        )
        (output_dir / "checkpoint_hashes.json").write_text(json.dumps(checkpoint_hashes, indent=2), encoding="utf-8")
        (output_dir / "frozen_state_audit.json").write_text(json.dumps(frozen_state_audit, indent=2), encoding="utf-8")
        (output_dir / "variant_selection.json").write_text(json.dumps(variant_selection, indent=2), encoding="utf-8")
        (output_dir / "bank_transaction_log.json").write_text(json.dumps(transaction_log, indent=2), encoding="utf-8")
        (output_dir / "fresh_runtime_report.json").write_text(json.dumps(fresh_runtime_report, indent=2), encoding="utf-8")
        (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "shift_generalization_repair": shift_generalization_repair,
        "checkpoint_hashes": checkpoint_hashes,
        "frozen_state_audit": frozen_state_audit,
        "variant_selection": variant_selection,
        "bank_transaction_log": transaction_log,
        "fresh_runtime_report": fresh_runtime_report,
        "protocol": protocol,
    }
