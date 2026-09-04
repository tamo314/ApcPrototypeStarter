"""SHIFT compact structural probe (Phase A.1 diagnostic Task A1-R005E-S006,
`docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

## What this tests

In A1-R005E-006A and A1-R005E-S002, the standard compact cross-position
operator (`CompactCrossPositionOperator`) reached ~100% on SELECT, COUNT, and
BIND, but plateaued at ~52% exact match on SHIFT under both specialized and
shared encoders (ADR-0043, ADR-0044). ADR-0042 diagnosed this shortfall not as
a representation limit or multi-task interference, but as an architectural
mismatch: computing modular position arithmetic `(slot + shift) mod L` in a
single dot-product cross-attention layer over standard learned absolute
position embeddings lacks the necessary inductive bias.

Task A1-R005E-S006 tests whether equipping the primitive-scale operator with
an explicit modular relative-position attention bias
(`ShiftRelativeCrossPositionOperator`) allows SHIFT to achieve the target
accuracy (Correct exact >= 0.90, token accuracy >= 0.98) on top of the
**frozen shared task-blind representation**.

```text
content -> Frozen Shared Encoder -> h_content
                                        | (keys/values)
(slot, shift) -> Modular RelPos Bias -> Cross-Attention -> FFN -> Readout -> output
```

## Controls

Per `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`:
1. Keep the successful shared encoder fixed/frozen (`requires_grad=False`).
2. Compare against:
   - Shared compact baseline (S002 SHIFT): Correct exact 0.5193, gap 0.5192
   - E-006A SHIFT (unshared joint compact): Correct exact 0.5194, gap 0.5192
   - E-004 SHIFT (frozen high-capacity upper bound): Correct exact 0.6180, gap 0.6180
   - E-005 SHIFT (frozen standard compact): Correct exact 0.0380, gap 0.0380
3. Primitive-scale size: ~18,282 parameters (only +128 parameters relative to
   the standard 18,154 compact operator, ~133x smaller than E-004's 2.44M operator).
4. Preserve the three-arm causal ablation: Correct, effectful Wrong argument,
   and None arms on held-out unseen counterfactual groups across >= 5 seeds.
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
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    CORRECT_EXACT_MATCH_THRESHOLD,
    DEFAULT_GROUP_SIZE,
    EFFECTFUL_WRONG_ARGUMENT_CEILING,
    MIN_CAUSAL_GAP,
    MIN_GROUP_SIZE,
    NONE_CEILING,
    TOKEN_ACCURACY_THRESHOLD,
    CompactOperatorGroup,
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
from apc.primitives.conditioning import (
    DEFAULT_ARG_DIM,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    default_argument_encoder,
)
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "TASK_BLIND_ATOL",
    "ShiftRelativeCrossPositionOperator",
    "ShiftStructuralProbeConfig",
    "shift_structural_probe_config_from_dict",
    "ShiftStructuralProbeReport",
    "run_shift_compact_structural_probe",
    "ShiftStructuralProbeMultiSeedReport",
    "run_shift_compact_structural_probe_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5
TASK_BLIND_ATOL = 1e-5
_EVAL_BATCH_SIZE = 256


def _derive_local_seed(seed: int, step: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class ShiftRelativeCrossPositionOperator(nn.Module):
    """Primitive-scale SHIFT operator equipped with modular relative-position
    attention bias.

    Preserves the footprint of `CompactCrossPositionOperator` ($d_{operator}=32$,
    $n_{head}=4$, $d_{ff}=64$, single cross-attention block, small FFN, linear
    readout), adding a learned modular relative-position bias table
    `rel_pos_bias` of shape `[max_sequence_length, n_head]`:

    $$\\delta(i, j, a, L) = (j - i - a) \\pmod L$$
    $$A_{h, i, j} = (q_{h, i}^\\top k_{h, j}) / \\sqrt{d_h} + \\text{rel\\_bias}[\\delta, h]$$

    Parameters: 18,282 (primitive scale, +128 parameters over 18,154 standard).
    """

    def __init__(
        self,
        *,
        d_model: int = 192,
        d_operator: int = 32,
        n_head: int = 4,
        d_operator_ff: int = 64,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        arg_dim: int = DEFAULT_ARG_DIM,
    ) -> None:
        super().__init__()
        if d_operator % n_head != 0:
            raise ValueError(f"d_operator ({d_operator}) must be divisible by n_head ({n_head})")
        self.operation = "SHIFT"
        self.d_operator = d_operator
        self.n_head = n_head
        self.max_sequence_length = max_sequence_length

        self.content_in_proj = nn.Linear(d_model, d_operator)
        self.content_position_embedding = nn.Embedding(max_sequence_length, d_operator)
        self.answer_query_embedding = nn.Embedding(max_sequence_length, d_operator)
        self.arg_encoder = default_argument_encoder(
            "SHIFT",
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
        )
        self.arg_proj = nn.Linear(arg_dim, d_operator)

        # Learned modular relative position bias: [max_sequence_length, n_head]
        self.rel_pos_bias = nn.Embedding(max_sequence_length, n_head)
        nn.init.zeros_(self.rel_pos_bias.weight)

        self.cross_attn = nn.MultiheadAttention(d_operator, n_head, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_operator)
        self.ffn = nn.Sequential(
            nn.Linear(d_operator, d_operator_ff),
            nn.GELU(),
            nn.Linear(d_operator_ff, d_operator),
        )
        self.ffn_norm = nn.LayerNorm(d_operator)
        self.readout = nn.Linear(d_operator, vocab_size)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None,
    ) -> torch.Tensor:
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        kv = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
            shifts = torch.zeros(batch, 1, 1, dtype=torch.long, device=device)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)
            shift_list = [int(v) for v in argument_values]
            shifts = torch.tensor(shift_list, device=device).view(batch, 1, 1)

        query = query_slots + arg_token

        # Vectorized modular relative position displacement:
        # disp[b, i, j] = (j - i - a) % L
        c_lens = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
        s_idx = torch.arange(out_max, device=device).view(1, out_max, 1)
        p_idx = torch.arange(lmax, device=device).view(1, 1, lmax)
        disp = (p_idx - s_idx - shifts) % c_lens
        attn_bias = self.rel_pos_bias(disp).permute(0, 3, 1, 2)

        # Pad mask: True where key is beyond this example's content length
        pad_mask = (p_idx >= c_lens).unsqueeze(1).expand(-1, self.n_head, out_max, -1)
        attn_mask = torch.where(
            pad_mask,
            torch.tensor(float("-inf"), device=device),
            attn_bias,
        ).reshape(batch * self.n_head, out_max, lmax)

        attn_out, _ = self.cross_attn(query, kv, kv, attn_mask=attn_mask, need_weights=False)
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)


@dataclass(frozen=True)
class ShiftStructuralProbeConfig:
    """Explicit, serializable configuration for Task A1-R005E-S006 SHIFT compact
    structural probe."""

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    device: str = "auto"
    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    core_train_steps: int = 16000
    core_lr: float = 3e-4
    core_weight_decay: float = 1e-4
    operator_train_steps: int = 12000
    operator_lr: float = 5e-4
    operator_weight_decay: float = 1e-4
    operator_grad_clip: float = 1.0
    shared_encoder_checkpoint: str | None = None
    num_unseen_eval_groups: int = 1400
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
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]})"
            )
        if self.core_train_steps < 0:
            raise ValueError(f"core_train_steps must be >= 0, got {self.core_train_steps}")
        if self.operator_train_steps < 1:
            raise ValueError(f"operator_train_steps must be >= 1, got {self.operator_train_steps}")
        if self.num_unseen_eval_groups < 1:
            raise ValueError(
                f"num_unseen_eval_groups must be >= 1, got {self.num_unseen_eval_groups}"
            )
        if self.min_unseen_eval_examples < 1:
            raise ValueError(
                f"min_unseen_eval_examples must be >= 1, got {self.min_unseen_eval_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def shift_structural_probe_config_from_dict(raw: dict[str, Any]) -> ShiftStructuralProbeConfig:
    defaults = ShiftStructuralProbeConfig()
    return ShiftStructuralProbeConfig(
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
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        core_lr=raw.get("core_lr", defaults.core_lr),
        core_weight_decay=raw.get("core_weight_decay", defaults.core_weight_decay),
        operator_train_steps=raw.get("operator_train_steps", defaults.operator_train_steps),
        operator_lr=raw.get("operator_lr", defaults.operator_lr),
        operator_weight_decay=raw.get("operator_weight_decay", defaults.operator_weight_decay),
        operator_grad_clip=raw.get("operator_grad_clip", defaults.operator_grad_clip),
        shared_encoder_checkpoint=raw.get(
            "shared_encoder_checkpoint", defaults.shared_encoder_checkpoint
        ),
        num_unseen_eval_groups=raw.get("num_unseen_eval_groups", defaults.num_unseen_eval_groups),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


def _get_or_train_frozen_shared_core(
    config: ShiftStructuralProbeConfig,
    seed_dir: Path | None = None,
) -> SharedContentEncoder:
    """Obtain the frozen shared task-blind content encoder.

    Loads from `config.shared_encoder_checkpoint` or `seed_dir /
    shared_encoder.pt` if available; otherwise trains via balanced
    mixed-operation training across all 4 operations (S002 protocol) for
    `config.core_train_steps` and saves the checkpoint.
    """
    ckpt_path = None
    if config.shared_encoder_checkpoint is not None:
        p = Path(config.shared_encoder_checkpoint)
        if p.exists():
            ckpt_path = p
    elif seed_dir is not None:
        p = seed_dir / "shared_encoder.pt"
        if p.exists():
            ckpt_path = p

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

    if ckpt_path is not None:
        arch = build_shared_encoder_architecture(arch_cfg)
        state_dict = torch.load(ckpt_path, map_location=arch.core.device)
        arch.core.model.load_state_dict(state_dict)
        core = arch.core
    else:
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

    # Hard freeze: evaluation mode, no gradients to the encoder
    core.model.eval()
    for param in core.model.parameters():
        param.requires_grad_(False)

    return core


def _task_blind_invariance_max_abs_diff(
    core: SharedContentEncoder, group: CompactOperatorGroup
) -> float:
    core.model.eval()
    with torch.no_grad():
        content_features, _ = encode_content(core, list(group.examples))
    reference = content_features[0:1]
    return float((content_features - reference).abs().max().item())


def _evaluate_arm(
    core: SharedContentEncoder,
    operator: ShiftRelativeCrossPositionOperator,
    examples: Sequence[Example],
    *,
    argument_provider: Callable[[Example], Any] | None,
) -> tuple[float, float]:
    operator.eval()
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    device = core.device

    with torch.no_grad():
        for start in range(0, len(examples), _EVAL_BATCH_SIZE):
            chunk = examples[start : start + _EVAL_BATCH_SIZE]
            content_lengths = [len(example.input_tokens) for example in chunk]
            output_lengths = [
                get_operation("SHIFT").output_length(length) for length in content_lengths
            ]
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

            argument_values = (
                None
                if argument_provider is None
                else [argument_provider(example) for example in chunk]
            )
            logits = operator(h, content_lengths, output_lengths, argument_values)
            predictions = logits.argmax(dim=-1)

            for row, (example, n) in enumerate(zip(chunk, output_lengths, strict=True)):
                target = example.target_tokens
                prediction = tuple(predictions[row, :n].tolist())
                total_tokens += len(target)
                correct_tokens += sum(
                    1 for p, t in zip(prediction, target, strict=True) if p == t
                )
                if prediction == target:
                    exact_matches += 1

    return exact_matches / len(examples), correct_tokens / total_tokens


@dataclass(frozen=True)
class ShiftStructuralProbeReport:
    """Everything observed for one seed of Task A1-R005E-S006."""

    config: ShiftStructuralProbeConfig
    operator_param_count: int
    core_param_count: int
    steps_trained: int
    final_operator_train_loss: float
    correct_exact_match: float
    correct_token_accuracy: float
    effectful_wrong_argument_exact_match: float
    effectful_wrong_argument_token_accuracy: float
    none_exact_match: float
    none_token_accuracy: float
    exact_match_causal_gap: float
    token_accuracy_causal_gap: float
    argument_effect_rate: float
    task_blind_max_abs_diff: float
    task_blind_invariant_passed: bool
    num_unseen_eval_groups: int
    num_unseen_eval_examples: int
    correct_exact_match_passed: bool
    token_accuracy_passed: bool
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "operator_param_count": self.operator_param_count,
            "core_param_count": self.core_param_count,
            "steps_trained": self.steps_trained,
            "final_operator_train_loss": self.final_operator_train_loss,
            "correct_exact_match": self.correct_exact_match,
            "correct_token_accuracy": self.correct_token_accuracy,
            "effectful_wrong_argument_exact_match": self.effectful_wrong_argument_exact_match,
            "effectful_wrong_argument_token_accuracy": self.effectful_wrong_argument_token_accuracy,
            "none_exact_match": self.none_exact_match,
            "none_token_accuracy": self.none_token_accuracy,
            "exact_match_causal_gap": self.exact_match_causal_gap,
            "token_accuracy_causal_gap": self.token_accuracy_causal_gap,
            "argument_effect_rate": self.argument_effect_rate,
            "task_blind_max_abs_diff": self.task_blind_max_abs_diff,
            "task_blind_invariant_passed": self.task_blind_invariant_passed,
            "num_unseen_eval_groups": self.num_unseen_eval_groups,
            "num_unseen_eval_examples": self.num_unseen_eval_examples,
            "correct_exact_match_passed": self.correct_exact_match_passed,
            "token_accuracy_passed": self.token_accuracy_passed,
            "effectful_wrong_argument_passed": self.effectful_wrong_argument_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "passed": self.passed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_shift_compact_structural_probe(
    config: ShiftStructuralProbeConfig,
    *,
    seed_dir: Path | None = None,
    metrics_path: str | Path | None = None,
) -> ShiftStructuralProbeReport:
    """Run one seed of Task A1-R005E-S006."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    # 1. Obtain frozen shared core
    core = _get_or_train_frozen_shared_core(config, seed_dir=seed_dir)
    device = core.device

    # 2. Build ShiftRelativeCrossPositionOperator
    set_seed(config.seed)
    operator = ShiftRelativeCrossPositionOperator(
        d_model=core.model.config.d_model,
        d_operator=config.d_operator,
        n_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    ).to(device)

    # 3. Train operator with cosine decay
    optimizer = torch.optim.AdamW(
        operator.parameters(),
        lr=config.operator_lr,
        weight_decay=config.operator_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.operator_train_steps, eta_min=1e-5
    )

    metrics_file = None
    if metrics_path is not None:
        p = Path(metrics_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = p.open("w", encoding="utf-8")

    groups_per_step = max(1, -(-32 // config.group_size))
    final_loss = 0.0

    operator.train()
    try:
        for step in range(1, config.operator_train_steps + 1):
            groups = generate_compact_operator_counterfactual_groups(
                config.seed,
                groups_per_step,
                operation="SHIFT",
                step=step,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, _, _ = _flatten_groups(groups)
            argument_values = [_correct_argument_value("SHIFT", ex) for ex in examples]
            content_lengths = [len(ex.input_tokens) for ex in examples]
            output_lengths = [
                get_operation("SHIFT").output_length(length) for length in content_lengths
            ]
            out_max = max(output_lengths)
            labels = _labels_for_examples(examples, output_lengths, out_max, device)

            with torch.no_grad():
                batch_input = collate_content_only_batch(examples, core.tokens, device=device)
                h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

            optimizer.zero_grad(set_to_none=True)
            logits = operator(h, content_lengths, output_lengths, argument_values)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
            loss.backward()
            if config.operator_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(operator.parameters(), config.operator_grad_clip)
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.item())

            should_log = step % 2000 == 0 or step == config.operator_train_steps
            if metrics_file is not None and should_log:
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": final_loss,
                            "lr": float(scheduler.get_last_lr()[0]),
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    # 4. Evaluate on large unseen test set
    unseen_groups = generate_compact_operator_counterfactual_groups(
        config.seed,
        config.num_unseen_eval_groups,
        operation="SHIFT",
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )
    unseen_examples, wrong_argument_by_id, wrong_target_tokens_by_id = _flatten_groups(
        unseen_groups
    )
    if len(unseen_examples) < config.min_unseen_eval_examples:
        raise ValueError(
            f"unseen eval realized only {len(unseen_examples)} examples from "
            f"{config.num_unseen_eval_groups} groups; increase num_unseen_eval_groups"
        )

    correct_exact_match, correct_token_accuracy = _evaluate_arm(
        core,
        operator,
        unseen_examples,
        argument_provider=lambda ex: _correct_argument_value("SHIFT", ex),
    )
    wrong_exact_match, wrong_token_accuracy = _evaluate_arm(
        core,
        operator,
        unseen_examples,
        argument_provider=lambda ex: wrong_argument_by_id[id(ex)],
    )
    none_exact_match, none_token_accuracy = _evaluate_arm(
        core,
        operator,
        unseen_examples,
        argument_provider=None,
    )

    argument_effect_rate = statistics.fmean(
        float(ex.target_tokens != wrong_target_tokens_by_id[id(ex)]) for ex in unseen_examples
    )
    exact_match_causal_gap = correct_exact_match - max(wrong_exact_match, none_exact_match)
    token_accuracy_causal_gap = correct_token_accuracy - max(
        wrong_token_accuracy, none_token_accuracy
    )

    task_blind_max_abs_diff = _task_blind_invariance_max_abs_diff(core, unseen_groups[0])
    task_blind_invariant_passed = task_blind_max_abs_diff <= TASK_BLIND_ATOL

    correct_exact_match_passed = correct_exact_match >= CORRECT_EXACT_MATCH_THRESHOLD
    token_accuracy_passed = correct_token_accuracy >= TOKEN_ACCURACY_THRESHOLD
    effectful_wrong_argument_passed = wrong_exact_match <= EFFECTFUL_WRONG_ARGUMENT_CEILING
    none_passed = none_exact_match <= NONE_CEILING
    causal_gap_passed = exact_match_causal_gap >= MIN_CAUSAL_GAP
    passed = (
        correct_exact_match_passed
        and token_accuracy_passed
        and effectful_wrong_argument_passed
        and none_passed
        and causal_gap_passed
        and task_blind_invariant_passed
    )

    operator_param_count = sum(p.numel() for p in operator.parameters())
    core_param_count = core.model.num_parameters()

    return ShiftStructuralProbeReport(
        config=config,
        operator_param_count=operator_param_count,
        core_param_count=core_param_count,
        steps_trained=config.operator_train_steps,
        final_operator_train_loss=final_loss,
        correct_exact_match=correct_exact_match,
        correct_token_accuracy=correct_token_accuracy,
        effectful_wrong_argument_exact_match=wrong_exact_match,
        effectful_wrong_argument_token_accuracy=wrong_token_accuracy,
        none_exact_match=none_exact_match,
        none_token_accuracy=none_token_accuracy,
        exact_match_causal_gap=exact_match_causal_gap,
        token_accuracy_causal_gap=token_accuracy_causal_gap,
        argument_effect_rate=argument_effect_rate,
        task_blind_max_abs_diff=task_blind_max_abs_diff,
        task_blind_invariant_passed=task_blind_invariant_passed,
        num_unseen_eval_groups=len(unseen_groups),
        num_unseen_eval_examples=len(unseen_examples),
        correct_exact_match_passed=correct_exact_match_passed,
        token_accuracy_passed=token_accuracy_passed,
        effectful_wrong_argument_passed=effectful_wrong_argument_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        passed=passed,
        wall_clock_seconds=time.perf_counter() - start_time,
        device=str(device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_val = statistics.fmean(values)
    stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_val, stdev_val, min(values), max(values)


@dataclass(frozen=True)
class ShiftStructuralProbeMultiSeedReport:
    """Multi-seed summary and baseline comparison for Task A1-R005E-S006."""

    seeds: tuple[int, ...]
    per_seed: tuple[ShiftStructuralProbeReport, ...]
    operator_param_count: int
    core_param_count: int
    mean_correct_exact_match: float
    stdev_correct_exact_match: float
    min_correct_exact_match: float
    max_correct_exact_match: float
    mean_correct_token_accuracy: float
    mean_wrong_exact_match: float
    mean_wrong_token_accuracy: float
    mean_none_exact_match: float
    mean_none_token_accuracy: float
    mean_exact_match_causal_gap: float
    mean_token_accuracy_causal_gap: float
    mean_argument_effect_rate: float
    max_task_blind_max_abs_diff: float
    task_blind_invariant_passed: bool
    correct_exact_match_passed: bool
    token_accuracy_passed: bool
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool
    meets_seed_policy: bool
    # Comparison baselines
    s002_shift_correct: float = 0.5193
    s002_shift_gap: float = 0.5192
    e006a_shift_correct: float = 0.5194
    e006a_shift_gap: float = 0.5192
    e004_shift_correct: float = 0.6180
    e004_shift_gap: float = 0.6180
    e005_shift_correct: float = 0.0380
    e005_shift_gap: float = 0.0380

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "operator_param_count": self.operator_param_count,
            "core_param_count": self.core_param_count,
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "stdev_correct_exact_match": self.stdev_correct_exact_match,
            "min_correct_exact_match": self.min_correct_exact_match,
            "max_correct_exact_match": self.max_correct_exact_match,
            "mean_correct_token_accuracy": self.mean_correct_token_accuracy,
            "mean_wrong_exact_match": self.mean_wrong_exact_match,
            "mean_wrong_token_accuracy": self.mean_wrong_token_accuracy,
            "mean_none_exact_match": self.mean_none_exact_match,
            "mean_none_token_accuracy": self.mean_none_token_accuracy,
            "mean_exact_match_causal_gap": self.mean_exact_match_causal_gap,
            "mean_token_accuracy_causal_gap": self.mean_token_accuracy_causal_gap,
            "mean_argument_effect_rate": self.mean_argument_effect_rate,
            "max_task_blind_max_abs_diff": self.max_task_blind_max_abs_diff,
            "task_blind_invariant_passed": self.task_blind_invariant_passed,
            "correct_exact_match_passed": self.correct_exact_match_passed,
            "token_accuracy_passed": self.token_accuracy_passed,
            "effectful_wrong_argument_passed": self.effectful_wrong_argument_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
            "baselines": {
                "s002_shift_correct": self.s002_shift_correct,
                "s002_shift_gap": self.s002_shift_gap,
                "e006a_shift_correct": self.e006a_shift_correct,
                "e006a_shift_gap": self.e006a_shift_gap,
                "e004_shift_correct": self.e004_shift_correct,
                "e004_shift_gap": self.e004_shift_gap,
                "e005_shift_correct": self.e005_shift_correct,
                "e005_shift_gap": self.e005_shift_gap,
            },
        }


def run_shift_compact_structural_probe_multi_seed(
    base_config: ShiftStructuralProbeConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> ShiftStructuralProbeMultiSeedReport:
    per_seed: list[ShiftStructuralProbeReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None

    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        metrics_path = seed_dir / "shift_probe_metrics.jsonl" if seed_dir else None
        report = run_shift_compact_structural_probe(
            config, seed_dir=seed_dir, metrics_path=metrics_path
        )
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    mean_correct, stdev_correct, min_correct, max_correct = _summarize(
        [r.correct_exact_match for r in per_seed]
    )
    mean_correct_tok = statistics.fmean(r.correct_token_accuracy for r in per_seed)
    mean_wrong = statistics.fmean(r.effectful_wrong_argument_exact_match for r in per_seed)
    mean_wrong_tok = statistics.fmean(r.effectful_wrong_argument_token_accuracy for r in per_seed)
    mean_none = statistics.fmean(r.none_exact_match for r in per_seed)
    mean_none_tok = statistics.fmean(r.none_token_accuracy for r in per_seed)
    mean_exact_gap = mean_correct - max(mean_wrong, mean_none)
    mean_tok_gap = mean_correct_tok - max(mean_wrong_tok, mean_none_tok)
    mean_arg_effect = statistics.fmean(r.argument_effect_rate for r in per_seed)
    max_invariance_diff = max(r.task_blind_max_abs_diff for r in per_seed)
    all_invariance_passed = all(r.task_blind_invariant_passed for r in per_seed)

    correct_exact_match_passed = mean_correct >= CORRECT_EXACT_MATCH_THRESHOLD
    token_accuracy_passed = mean_correct_tok >= TOKEN_ACCURACY_THRESHOLD
    effectful_wrong_argument_passed = mean_wrong <= EFFECTFUL_WRONG_ARGUMENT_CEILING
    none_passed = mean_none <= NONE_CEILING
    causal_gap_passed = mean_exact_gap >= MIN_CAUSAL_GAP
    overall_passed = (
        correct_exact_match_passed
        and token_accuracy_passed
        and effectful_wrong_argument_passed
        and none_passed
        and causal_gap_passed
        and all_invariance_passed
    )

    return ShiftStructuralProbeMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        operator_param_count=per_seed[0].operator_param_count,
        core_param_count=per_seed[0].core_param_count,
        mean_correct_exact_match=mean_correct,
        stdev_correct_exact_match=stdev_correct,
        min_correct_exact_match=min_correct,
        max_correct_exact_match=max_correct,
        mean_correct_token_accuracy=mean_correct_tok,
        mean_wrong_exact_match=mean_wrong,
        mean_wrong_token_accuracy=mean_wrong_tok,
        mean_none_exact_match=mean_none,
        mean_none_token_accuracy=mean_none_tok,
        mean_exact_match_causal_gap=mean_exact_gap,
        mean_token_accuracy_causal_gap=mean_tok_gap,
        mean_argument_effect_rate=mean_arg_effect,
        max_task_blind_max_abs_diff=max_invariance_diff,
        task_blind_invariant_passed=all_invariance_passed,
        correct_exact_match_passed=correct_exact_match_passed,
        token_accuracy_passed=token_accuracy_passed,
        effectful_wrong_argument_passed=effectful_wrong_argument_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        passed=overall_passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
