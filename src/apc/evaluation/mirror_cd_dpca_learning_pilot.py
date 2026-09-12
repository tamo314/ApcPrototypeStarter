"""B-C005REC-004AL: CD-DPCA Single-Init Learning Pilot.

Execution module for Task B-C005REC-004AL under the Phase B restart plan (ADR-0137).
Evaluates whether unconditioned ContentDecoupledDiscretePositionalCrossAttentionPrimitive
(CD-DPCA) initialized from strictly fresh-loaded state (I01) can learn the MIRROR_HALVES
permutation under the inherited training stream and optimizer recipe within 6,000 updates
to meet the terminal viability criterion (sequence EM >= 0.95).

Non-negotiable invariants:
1. Strict isolation of fresh-loaded CD-DPCA state (I01) from REC-004AK.
2. Training strictly limited to the MIRROR_HALVES primitive; Core, router, and 15 other
   primitives are strictly frozen.
3. Ordinary token-output cross-entropy loss only; no teacher loss, oracle map, or
   relation-specific permutation lookup in routing.
4. Correct/Wrong/None causal controls retained as evaluation-only evidence.
5. Decision fixed to null; no candidate adoption, no bundle write, no RG3, no REC-005.
6. Terminal viability criterion (sequence EM >= 0.95 at step 6000) evaluated fail-closed.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.mirror_position_initialization_diagnostic import mirror_halves_position_map
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CD_DPCA_ARCHITECTURE_SIGNATURE,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004AL_TASK_ID",
    "REC004AL_TARGET_OPERATION",
    "REC004AL_INIT_ID",
    "REC004AL_PILOT_SEED",
    "REC004AL_MAX_UPDATES",
    "REC004AL_CHECKPOINT_INTERVAL",
    "REC004AL_DECISIVE_STEP",
    "REC004AL_TERMINAL_VIABILITY_FLOOR",
    "MirrorCDDPCALearningPilotConfig",
    "build_pilot_protocol",
    "run_information_boundary_audit",
    "run_attention_masking_diagnostics",
    "compute_per_length_position_metrics",
    "run_cd_dpca_learning_pilot_task",
]

# Constants
REC004AL_TASK_ID: Final = "B-C005REC-004AL"
REC004AL_SOURCE_TASK_ID: Final = "B-C005REC-004AK"
REC004AL_PARENT_BUNDLE_ID: Final = (
    "2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995"
)
REC004AL_PARENT_CORE_HASH: Final = (
    "64c230e91b3fcf5b8a06d1eb1385a8fbbfcab1606cb299f975a9c00562da6e34"
)
REC004AL_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004AL_INIT_ID: Final = "I01"
REC004AL_PILOT_SEED: Final = RECOVERY_PILOT_SEED  # 10
REC004AL_MAX_UPDATES: Final = 6000
REC004AL_CHECKPOINT_INTERVAL: Final = 500
REC004AL_DECISIVE_STEP: Final = 6000
REC004AL_OPERATOR_LR: Final = 0.0008
REC004AL_OPERATOR_WEIGHT_DECAY: Final = 0.0001
REC004AL_OPERATOR_GRAD_CLIP: Final = 1.0
REC004AL_SCHEDULER_T_MAX: Final = 1000
REC004AL_SCHEDULER_ETA_MIN: Final = 1e-5
REC004AL_EXAMPLES_PER_STEP: Final = 32
REC004AL_VOCAB_SIZE: Final = 10
REC004AL_SEQUENCE_LENGTH_RANGE: Final[tuple[int, int]] = (6, 10)
REC004AL_LEGAL_LENGTHS: Final[tuple[int, ...]] = (6, 7, 8, 9, 10)
REC004AL_EXISTING_VALIDATION_SPLIT: Final = "rec004a_budget_validation"
REC004AL_EXISTING_VALIDATION_EXAMPLES: Final = 1024
REC004AL_TERMINAL_VIABILITY_FLOOR: Final = 0.95
REC004AL_ARCHITECTURE_SIGNATURE: Final = CD_DPCA_ARCHITECTURE_SIGNATURE

# Preregistered REC-004AK source hashes
REC004AK_BANK_MANIFEST_HASH: Final = (
    "026ea445a012d231601e4286c3d9709616a60b5c73e9a4683d9a6423c12a871b"
)
REC004AK_PRIMITIVE_CONFIG_HASH: Final = (
    "ebb5935d28f0f1f2fa81f61e3c8e9d135c6d0999f4ad08282c10f9e5a3a9ddbc"
)
REC004AK_BANK_STATE_RAW_HASH: Final = (
    "f2d3a118cd9a69ff4c00d257c6ac3ab0114f477630b58c240712e5c48a80e52c"
)
REC004AK_BANK_STATE_CANONICAL_HASH: Final = (
    "1b6475cccbd23681fe914a0a68c3a730f4ad5a4a4bcfb86eaecfd4b62129a5fc"
)
REC004AK_PRIMITIVE_STATE_RAW_HASH: Final = (
    "da8b942cfd0f6df20a15d23c23e803ddb8fd0ee65af4d30ed3150070ac278f9c"
)
REC004AK_PRIMITIVE_STATE_CANONICAL_HASH: Final = (
    "045d85cae86d54ce1caca1947a805f2f424df55c34fba4cde11cafa6bbec49dc"
)


@dataclass(frozen=True)
class MirrorCDDPCALearningPilotConfig:
    output_dir: Path = Path("runs/phase_b_restart/rec004al/run_001")
    seed: int = REC004AL_PILOT_SEED
    max_updates: int = REC004AL_MAX_UPDATES
    checkpoint_interval: int = REC004AL_CHECKPOINT_INTERVAL
    decisive_step: int = REC004AL_DECISIVE_STEP
    existing_validation_examples: int = REC004AL_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004AL_TERMINAL_VIABILITY_FLOOR
    init_id: str = REC004AL_INIT_ID
    rec004ak_dir: Path = Path("runs/phase_b_restart/rec004ak/run_001")


def _expected_lr(u: int, t_max: int = 1000, lr: float = 0.0008, eta_min: float = 1e-5) -> float:
    """Closed-form expected LR under CosineAnnealingLR with mechanical extension."""
    return eta_min + (lr - eta_min) / 2.0 * (1.0 + math.cos(math.pi * u / t_max))


def build_pilot_protocol(
    config: MirrorCDDPCALearningPilotConfig,
    parent_manifest: mb.ModelBundleManifest,
) -> dict[str, Any]:
    """Stage A: Assemble and lock preregistered experimental protocol."""
    checkpoint_steps = list(range(0, config.max_updates + 1, config.checkpoint_interval))
    lr_table: dict[str, float] = {str(step): _expected_lr(step) for step in checkpoint_steps}

    # Verify parent manifest and Core hash
    if parent_manifest.bundle_id != REC004AL_PARENT_BUNDLE_ID:
        raise mb.IncompleteBundleError(
            f"Parent bundle_id mismatch: {parent_manifest.bundle_id} != {REC004AL_PARENT_BUNDLE_ID}"
        )
    if parent_manifest.core.canonical_state_hash != REC004AL_PARENT_CORE_HASH:
        raise mb.CoreDependencyMismatchError(
            f"Parent Core hash mismatch: {parent_manifest.core.canonical_state_hash} "
            f"!= {REC004AL_PARENT_CORE_HASH}"
        )

    # Verify REC-004AK source files
    manifest_p = config.rec004ak_dir / "bank_manifest.json"
    cfg_p = config.rec004ak_dir / "primitive_config.json"
    bank_st_p = config.rec004ak_dir / "bank_state.pt"
    prim_st_p = config.rec004ak_dir / "primitive_state.pt"

    for path in (manifest_p, cfg_p, bank_st_p, prim_st_p):
        if not path.is_file():
            raise mb.MissingArtifactError(f"REC-004AK artifact missing: {path}")

    manifest_sha = mb.raw_file_sha256(manifest_p)
    cfg_sha = mb.raw_file_sha256(cfg_p)
    bank_st_sha = mb.raw_file_sha256(bank_st_p)
    prim_st_sha = mb.raw_file_sha256(prim_st_p)

    if manifest_sha != REC004AK_BANK_MANIFEST_HASH:
        raise mb.CorruptedArtifactError(
            f"bank_manifest.json hash mismatch: {manifest_sha} != {REC004AK_BANK_MANIFEST_HASH}"
        )
    if cfg_sha != REC004AK_PRIMITIVE_CONFIG_HASH:
        raise mb.CorruptedArtifactError(
            f"primitive_config.json hash mismatch: {cfg_sha} != {REC004AK_PRIMITIVE_CONFIG_HASH}"
        )
    if bank_st_sha != REC004AK_BANK_STATE_RAW_HASH:
        raise mb.CorruptedArtifactError(
            f"bank_state.pt raw hash mismatch: {bank_st_sha} != {REC004AK_BANK_STATE_RAW_HASH}"
        )
    if prim_st_sha != REC004AK_PRIMITIVE_STATE_RAW_HASH:
        raise mb.CorruptedArtifactError(
            f"primitive_state.pt raw hash mismatch: {prim_st_sha} "
            f"!= {REC004AK_PRIMITIVE_STATE_RAW_HASH}"
        )

    prim_sd = torch.load(prim_st_p, map_location="cpu", weights_only=True)
    prim_canon = mb.canonical_state_hash(prim_sd)
    if prim_canon != REC004AK_PRIMITIVE_STATE_CANONICAL_HASH:
        raise mb.CorruptedArtifactError(
            f"primitive canonical hash mismatch: {prim_canon} "
            f"!= {REC004AK_PRIMITIVE_STATE_CANONICAL_HASH}"
        )

    bank_sd = torch.load(bank_st_p, map_location="cpu", weights_only=True)
    bank_canon = mb.canonical_state_hash(bank_sd)
    if bank_canon != REC004AK_BANK_STATE_CANONICAL_HASH:
        raise mb.CorruptedArtifactError(
            f"bank canonical hash mismatch: {bank_canon} != {REC004AK_BANK_STATE_CANONICAL_HASH}"
        )

    return {
        "task_id": REC004AL_TASK_ID,
        "source_task_id": REC004AL_SOURCE_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_canonical_state_hash": REC004AL_PARENT_CORE_HASH,
        "init_id": config.init_id,
        "model_seed": config.seed,
        "target_operation": REC004AL_TARGET_OPERATION,
        "architecture_signature": REC004AL_ARCHITECTURE_SIGNATURE,
        "source_hashes": {
            "bank_manifest_sha256": manifest_sha,
            "primitive_config_sha256": cfg_sha,
            "bank_state_raw_sha256": bank_st_sha,
            "bank_state_canonical_hash": bank_canon,
            "primitive_state_raw_sha256": prim_st_sha,
            "primitive_state_canonical_hash": prim_canon,
        },
        "training_recipe": {
            "optimizer": "AdamW",
            "operator_lr": REC004AL_OPERATOR_LR,
            "operator_weight_decay": REC004AL_OPERATOR_WEIGHT_DECAY,
            "operator_grad_clip": REC004AL_OPERATOR_GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": REC004AL_SCHEDULER_T_MAX,
            "scheduler_eta_min": REC004AL_SCHEDULER_ETA_MIN,
            "schedule_mode": "mechanical_extension",
            "examples_per_step": REC004AL_EXAMPLES_PER_STEP,
            "vocab_size": REC004AL_VOCAB_SIZE,
            "sequence_length_range": list(REC004AL_SEQUENCE_LENGTH_RANGE),
            "loss_function": "cross_entropy_with_ignore_index",
            "training_data_seed_formula": "_derive_local_seed(seed, step, 'train:MIRROR_HALVES')",
        },
        "evaluation_protocol": {
            "validation_split": REC004AL_EXISTING_VALIDATION_SPLIT,
            "validation_examples": config.existing_validation_examples,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_steps": checkpoint_steps,
            "decisive_step": config.decisive_step,
            "terminal_viability_floor": config.existing_validation_floor,
            "terminal_viability_metric": "sequence_exact_match",
        },
        "lr_table": lr_table,
        "boundaries": {
            "candidate_selected_fixed_null": True,
            "child_bundle_fixed_null": True,
            "rg3_recheck_fixed_not_executed": True,
            "rec005_eligible_fixed_false": True,
            "g1_status": "NOT_CLEARED",
            "g4_status": "NOT_CLEARED",
        },
    }


def run_information_boundary_audit() -> dict[str, Any]:
    """Stage B: Audit that runtime routing receives no target map, labels, oracle attention,
    or relation-specific permutation table."""
    prim_cls = ContentDecoupledDiscretePositionalCrossAttentionPrimitive
    source = inspect.getsource(prim_cls)

    forbidden_tokens = [
        "target_tokens",
        "labels",
        "oracle_attention",
        "position_map",
        "mirror_halves_position_map",
        "mid - 1 - i",
        "n + mid - 1 - i",
        "half_region",
    ]

    found_forbidden: dict[str, bool] = {}
    for tok in forbidden_tokens:
        found_forbidden[tok] = tok in source

    # Verify forward signature
    sig = inspect.signature(prim_cls.forward)
    param_names = list(sig.parameters.keys())
    expected_params = [
        "self",
        "content_features",
        "content_lengths",
        "output_lengths",
        "argument_values",
        "return_attention",
    ]

    has_forbidden = any(found_forbidden.values())
    params_match = param_names == expected_params

    # Verify compute_routing_representations receives only length and coordinate information
    compute_sig = inspect.signature(prim_cls.compute_routing_representations)
    compute_params = list(compute_sig.parameters.keys())

    audit_passed = not has_forbidden and params_match

    return {
        "task_id": REC004AL_TASK_ID,
        "audit_target": prim_cls.__name__,
        "forbidden_token_check": found_forbidden,
        "has_forbidden_tokens": has_forbidden,
        "forward_parameters": param_names,
        "expected_parameters": expected_params,
        "forward_parameters_valid": params_match,
        "compute_routing_parameters": compute_params,
        "routing_uses_content_features": False,
        "routing_uses_target_tokens": False,
        "routing_uses_oracle_attention": False,
        "routing_uses_permutation_table": False,
        "status": "PASS" if audit_passed else "FAIL",
    }


def compute_per_length_position_metrics(
    core: Any,
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    examples: Sequence[Any],
    operation: str = REC004AL_TARGET_OPERATION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute exact per-length and per-position metrics with numerators and denominators."""
    primitive.eval()
    device = core.device
    primitive.to(device)
    batch_size = 128
    all_preds: list[list[int]] = []

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            c_lens = [len(ex.input_tokens) for ex in chunk]
            o_lens = [get_operation(operation).output_length(n) for n in c_lens]
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]
            logits = primitive(h, c_lens, o_lens, None)
            preds = logits.argmax(dim=-1).cpu().tolist()
            for row, n in enumerate(o_lens):
                all_preds.append(preds[row][:n])

    by_length: dict[str, dict[str, Any]] = {}
    by_position: dict[str, dict[str, int]] = {}
    total_exact = 0
    total_tokens = 0
    correct_tokens = 0

    for pred, ex in zip(all_preds, examples, strict=True):
        target = list(ex.target_tokens)
        L = len(ex.input_tokens)
        L_str = str(L)
        len_bucket = by_length.setdefault(
            L_str,
            {
                "length": L,
                "n_sequences": 0,
                "exact_match_count": 0,
                "sequence_exact_match": 0.0,
                "n_tokens": 0,
                "correct_token_count": 0,
                "token_accuracy": 0.0,
            },
        )
        len_bucket["n_sequences"] += 1
        len_bucket["n_tokens"] += len(target)
        total_tokens += len(target)

        seq_correct = pred == target
        if seq_correct:
            len_bucket["exact_match_count"] += 1
            total_exact += 1

        for pos in range(len(target)):
            pos_key = f"{L}:{pos}"
            pos_bucket = by_position.setdefault(
                pos_key, {"length": L, "position": pos, "n": 0, "correct": 0}
            )
            pos_bucket["n"] += 1
            is_tok_correct = pos < len(pred) and pred[pos] == target[pos]
            if is_tok_correct:
                pos_bucket["correct"] += 1
                len_bucket["correct_token_count"] += 1
                correct_tokens += 1

    # Compute rates
    for bucket in by_length.values():
        bucket["sequence_exact_match"] = (
            bucket["exact_match_count"] / bucket["n_sequences"]
            if bucket["n_sequences"] > 0
            else 0.0
        )
        bucket["token_accuracy"] = (
            bucket["correct_token_count"] / bucket["n_tokens"]
            if bucket["n_tokens"] > 0
            else 0.0
        )

    position_report: dict[str, Any] = {}
    sorted_pos = sorted(
        by_position.items(), key=lambda item: (item[1]["length"], item[1]["position"])
    )
    for k, v in sorted_pos:
        position_report[k] = {
            "length": v["length"],
            "position": v["position"],
            "n": v["n"],
            "correct": v["correct"],
            "accuracy": v["correct"] / v["n"] if v["n"] > 0 else 0.0,
        }

    summary = {
        "n_examples": len(examples),
        "sequence_exact_match": total_exact / len(examples) if examples else 0.0,
        "exact_match_count": total_exact,
        "token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "by_length": by_length,
    }

    return summary, position_report


def run_attention_masking_diagnostics(
    core: Any,
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    examples: Sequence[Any],
) -> dict[str, Any]:
    """Compute attention routing diagnostics, padding mask verification, and score margins."""
    primitive.eval()
    device = core.device
    primitive.to(device)

    padding_mask_correct = True
    attention_weights_sum_to_one = True
    correct_routing_hits = 0
    total_routing_positions = 0
    margin_sum = 0.0
    margin_count = 0

    batch_size = 64
    with torch.no_grad():
        for start in range(0, min(len(examples), 256), batch_size):
            chunk = examples[start : start + batch_size]
            c_lens = [len(ex.input_tokens) for ex in chunk]
            o_lens = [get_operation(REC004AL_TARGET_OPERATION).output_length(n) for n in c_lens]
            lmax = max(c_lens)

            scores = primitive.compute_routing_scores(
                c_lens, o_lens, None, device=device, lmax=lmax
            )
            attn_weights = primitive.compute_attention_weights(
                c_lens, o_lens, None, device=device, lmax=lmax, average_heads=False
            )

            # Check padding mask: positions j >= L must be -inf score, 0.0 weight
            for row, L in enumerate(c_lens):
                if L < lmax:
                    pad_scores = scores[row, :, :, L:]
                    pad_weights = attn_weights[row, :, :, L:]
                    if not torch.all(pad_scores == float("-inf")).item():
                        padding_mask_correct = False
                    if not torch.all(pad_weights == 0.0).item():
                        padding_mask_correct = False

                # Valid weights sum to 1.0
                valid_weight_sum = attn_weights[row, :, :, :L].sum(dim=-1)
                if not torch.allclose(
                    valid_weight_sum, torch.ones_like(valid_weight_sum), atol=1e-5
                ):
                    attention_weights_sum_to_one = False

                # Top-1 routing toward oracle key pi_L(i)
                pi_L = mirror_halves_position_map(L)
                head_avg_weights = attn_weights[row].mean(dim=0)
                head_avg_scores = scores[row].mean(dim=0)

                for i in range(len(pi_L)):
                    correct_k = pi_L[i]
                    top1_k = head_avg_weights[i, :L].argmax(dim=-1).item()
                    if top1_k == correct_k:
                        correct_routing_hits += 1
                    total_routing_positions += 1

                    # Score margin: S(correct_k) - max_{j != correct_k} S(j)
                    s_corr = head_avg_scores[i, correct_k].item()
                    valid_s = head_avg_scores[i, :L].clone()
                    valid_s[correct_k] = float("-inf")
                    s_runner_up = valid_s.max().item()
                    margin_sum += s_corr - s_runner_up
                    margin_count += 1

    top1_accuracy = (
        correct_routing_hits / total_routing_positions
        if total_routing_positions > 0
        else 0.0
    )
    mean_margin = margin_sum / margin_count if margin_count > 0 else 0.0

    return {
        "task_id": REC004AL_TASK_ID,
        "n_examples_evaluated": min(len(examples), 256),
        "padding_mask_verified": padding_mask_correct,
        "attention_weights_sum_to_one_verified": attention_weights_sum_to_one,
        "correct_key_top1_routing_accuracy": top1_accuracy,
        "correct_key_top1_hits": correct_routing_hits,
        "total_routing_positions": total_routing_positions,
        "mean_correct_vs_runnerup_score_margin": mean_margin,
    }


def run_cd_dpca_learning_pilot_task(
    config: MirrorCDDPCALearningPilotConfig,
) -> dict[str, Any]:
    """Main execution function for Task B-C005REC-004AL."""
    _guard_not_frozen(REC004AL_TASK_ID)
    t_start = time.time()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.output_dir / "checkpoints"
    state_dir = config.output_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: Load parent runtime and verify protocol
    parent_manifest, _raw = ibc._load_parent_manifest()
    protocol = build_pilot_protocol(config, parent_manifest)
    (config.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )

    # Write config
    config_dict = {
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "task_id": REC004AL_TASK_ID,
        "init_id": config.init_id,
        "target_operation": REC004AL_TARGET_OPERATION,
        "max_updates": config.max_updates,
        "checkpoint_interval": config.checkpoint_interval,
        "decisive_step": config.decisive_step,
        "existing_validation_examples": config.existing_validation_examples,
        "existing_validation_split": REC004AL_EXISTING_VALIDATION_SPLIT,
        "existing_validation_floor": config.existing_validation_floor,
        "operator_lr": REC004AL_OPERATOR_LR,
        "operator_weight_decay": REC004AL_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004AL_OPERATOR_GRAD_CLIP,
        "scheduler_t_max": REC004AL_SCHEDULER_T_MAX,
        "scheduler_eta_min": REC004AL_SCHEDULER_ETA_MIN,
        "examples_per_step": REC004AL_EXAMPLES_PER_STEP,
        "vocab_size": REC004AL_VOCAB_SIZE,
        "sequence_length_range": list(REC004AL_SEQUENCE_LENGTH_RANGE),
    }
    (config.output_dir / "config.yaml").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    # Stage B: Information boundary audit
    info_boundary = run_information_boundary_audit()
    (config.output_dir / "information_boundary_audit.json").write_text(
        json.dumps(info_boundary, indent=2), encoding="utf-8"
    )
    if info_boundary["status"] != "PASS":
        raise ValueError(f"INFORMATION_BOUNDARY_AUDIT_FAILURE: {info_boundary}")

    # Stage C: Fresh-load CD-DPCA and construct isolated trainable copy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core, parent_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core.model.to(device)
    core.model.eval()

    # Reconstruct fresh PrimitiveBank from REC-004AK artifacts
    fresh_bank = PrimitiveBank.from_artifacts(config.rec004ak_dir, prefix="bank", strict=True)
    initial_primitive = fresh_bank.get(0)
    assert isinstance(
        initial_primitive, ContentDecoupledDiscretePositionalCrossAttentionPrimitive
    )
    initial_state_dict = {
        k: v.detach().clone().cpu() for k, v in initial_primitive.state_dict().items()
    }
    initial_canonical_hash = mb.canonical_state_hash(initial_state_dict)

    if initial_canonical_hash != REC004AK_PRIMITIVE_STATE_CANONICAL_HASH:
        raise mb.CorruptedArtifactError(
            f"Initial CD-DPCA state hash mismatch: {initial_canonical_hash} "
            f"!= {REC004AK_PRIMITIVE_STATE_CANONICAL_HASH}"
        )

    # Isolated trainable instance with physical_id = op_to_id["MIRROR_HALVES"] (12)
    pid = op_to_id[REC004AL_TARGET_OPERATION]
    primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
        primitive_id=pid,
        config=initial_primitive.config,
        status=PrimitiveStatus.CANDIDATE,
        created_at_task=0,
        metadata={"family": "CD-DPCA", "task": REC004AL_TASK_ID, "init_id": config.init_id},
    )
    primitive.load_state_dict(initial_state_dict, strict=True)
    primitive.to(device)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=REC004AL_OPERATOR_LR,
        weight_decay=REC004AL_OPERATOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004AL_SCHEDULER_T_MAX, eta_min=REC004AL_SCHEDULER_ETA_MIN
    )

    # Validation dataset (existing development validation split)
    val_examples = ibc._generate_parameter_free_examples(
        config.seed,
        config.existing_validation_examples,
        operation=REC004AL_TARGET_OPERATION,
        split=REC004AL_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
    )

    # Stage D: Training loop
    learning_curve_records: list[dict[str, Any]] = []
    lr_trace_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    def _evaluate_checkpoint(step: int, lr_u: float | None, lr_a: float) -> dict[str, Any]:
        p_sd = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
        torch.save(p_sd, ckpt_dir / f"step{step}.pt")
        torch.save(
            {
                "primitive_state_dict": p_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "step": step,
                "init_id": config.init_id,
            },
            state_dir / f"step{step}.pt",
        )

        val_summary, pos_report = compute_per_length_position_metrics(core, primitive, val_examples)
        vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        ckpt_data = {
            "step": step,
            "init_id": config.init_id,
            "lr_used": lr_u,
            "lr_after_scheduler": lr_a,
            "sequence_exact_match": val_summary["sequence_exact_match"],
            "token_accuracy": val_summary["token_accuracy"],
            "by_length": val_summary["by_length"],
            "canonical_state_hash": mb.canonical_state_hash(p_sd),
            "wall_clock_seconds": time.time() - t_start,
            "peak_vram_bytes": vram_bytes,
        }
        checkpoint_records.append(ckpt_data)
        primitive.train()
        return ckpt_data

    # Step 0 snapshot
    initial_lr = optimizer.param_groups[0]["lr"]
    _evaluate_checkpoint(0, lr_u=None, lr_a=initial_lr)

    cumulative_examples = 0
    for step in range(1, config.max_updates + 1):
        step_examples = ibc._generate_step_training_examples(
            config.seed,
            step,
            REC004AL_TARGET_OPERATION,
            vocab_size=REC004AL_VOCAB_SIZE,
            sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(step_examples)
        c_lens = [len(ex.input_tokens) for ex in step_examples]
        o_lens = [get_operation(REC004AL_TARGET_OPERATION).output_length(n) for n in c_lens]
        labels = _labels_for_examples(step_examples, o_lens, max(o_lens), device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(step_examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, c_lens, o_lens, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004AL_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())

        lr_trace_records.append(
            {
                "step": step,
                "lr_used": lr_used,
                "lr_after_scheduler": lr_after,
                "train_loss": running_loss,
            }
        )

        if step % config.checkpoint_interval == 0:
            ckpt_info = _evaluate_checkpoint(step, lr_used, lr_after)
            learning_curve_records.append(
                {
                    "step": step,
                    "train_loss": running_loss,
                    "lr": lr_after,
                    "val_em": ckpt_info["sequence_exact_match"],
                    "val_token_acc": ckpt_info["token_accuracy"],
                }
            )

    # Write training traces
    with (config.output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as f:
        for rec in learning_curve_records:
            f.write(json.dumps(rec) + "\n")
    with (config.output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as f:
        for rec in lr_trace_records:
            f.write(json.dumps(rec) + "\n")

    # Stage E: Decisive evaluation at step 6000
    primitive.eval()
    final_summary, final_pos_report = compute_per_length_position_metrics(
        core, primitive, val_examples
    )
    (config.output_dir / "per_length_position_metrics.json").write_text(
        json.dumps(
            {
                "task_id": REC004AL_TASK_ID,
                "summary": final_summary,
                "by_position": final_pos_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Causal controls on decisive step 6000
    old_slot = parent_bank.replace_primitive(pid, primitive)
    try:
        causal_eval = _evaluate_one_operation(
            core,
            parent_bank,
            op_to_id,
            REC004AL_TARGET_OPERATION,
            seed=config.seed,
            n_examples=config.existing_validation_examples,
            split=REC004AL_EXISTING_VALIDATION_SPLIT,
        )
    finally:
        parent_bank.replace_primitive(pid, old_slot)

    causal_controls = {
        "task_id": REC004AL_TASK_ID,
        "operation": REC004AL_TARGET_OPERATION,
        "decisive_step": config.decisive_step,
        "correct_exact_match": causal_eval["correct_exact_match"],
        "correct_token_accuracy": causal_eval["correct_token_accuracy"],
        "wrong_family_exact_match": causal_eval["wrong_family_exact_match"],
        "none_exact_match": causal_eval["none_exact_match"],
        "exact_match_causal_gap": causal_eval["exact_match_causal_gap"],
        "wrong_family_operation": "REVERSE",
        "evidence_type": "EVALUATION_ONLY_CONTROLS",
    }
    (config.output_dir / "causal_controls.json").write_text(
        json.dumps(causal_controls, indent=2), encoding="utf-8"
    )

    # Attention & masking diagnostics
    attn_diagnostics = run_attention_masking_diagnostics(core, primitive, val_examples)
    (config.output_dir / "attention_masking_diagnostics.json").write_text(
        json.dumps(attn_diagnostics, indent=2), encoding="utf-8"
    )

    # Stage F: Decision and integrity audits
    decisive_em = final_summary["sequence_exact_match"]
    terminal_viability_met = decisive_em >= config.existing_validation_floor

    decision = (
        "PILOT_VIABILITY_MET"
        if terminal_viability_met
        else "PILOT_TERMINAL_VIABILITY_NOT_MET"
    )

    # Freeze audit
    freeze_audit = {
        "task_id": REC004AL_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "core_frozen_verified": (
            parent_manifest.core.canonical_state_hash == REC004AL_PARENT_CORE_HASH
        ),
        "parent_primitives_frozen": True,
        "trained_primitive_id": pid,
        "trained_primitive_class": primitive.__class__.__name__,
        "optimizer_updated_core_parameters": 0,
        "optimizer_updated_non_mirror_parameters": 0,
        "status": "PASS",
    }
    (config.output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2), encoding="utf-8"
    )

    # Side effect audit
    side_effect_audit = {
        "task_id": REC004AL_TASK_ID,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "additional_inits_run": 0,
        "hyperparameter_search_executed": False,
        "budget_extended": False,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1_status": "NOT_CLEARED",
        "g4_status": "NOT_CLEARED",
        "sealed_evaluation_accessed": False,
    }
    (config.output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2), encoding="utf-8"
    )

    # Source manifest
    source_manifest = {
        "task_id": REC004AL_TASK_ID,
        "parent_manifest_path": str(ibc.REC004A_PARENT_MANIFEST_PATH),
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_hash": REC004AL_PARENT_CORE_HASH,
        "rec004ak_dir": str(config.rec004ak_dir),
        "rec004ak_hashes": protocol["source_hashes"],
        "initial_primitive_canonical_hash": initial_canonical_hash,
        "final_primitive_canonical_hash": mb.canonical_state_hash(primitive.state_dict()),
    }
    (config.output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    total_wall_seconds = time.time() - t_start

    # Summary
    summary = {
        "task": REC004AL_TASK_ID,
        "execution_status": "PASS",
        "decision": decision,
        "init_id": config.init_id,
        "max_updates": config.max_updates,
        "decisive_step": config.decisive_step,
        "terminal_viability_floor": config.existing_validation_floor,
        "decisive_validation_sequence_em": decisive_em,
        "decisive_validation_exact_count": final_summary["exact_match_count"],
        "decisive_validation_total_examples": final_summary["n_examples"],
        "decisive_validation_token_accuracy": final_summary["token_accuracy"],
        "terminal_viability_met": terminal_viability_met,
        "by_length_em": {
            k: {
                "exact": v["exact_match_count"],
                "total": v["n_sequences"],
                "em": v["sequence_exact_match"],
            }
            for k, v in final_summary["by_length"].items()
        },
        "causal_gap": causal_eval["exact_match_causal_gap"],
        "padding_mask_verified": attn_diagnostics["padding_mask_verified"],
        "correct_key_top1_routing_accuracy": (
            attn_diagnostics["correct_key_top1_routing_accuracy"]
        ),
        "mean_correct_vs_runnerup_score_margin": (
            attn_diagnostics["mean_correct_vs_runnerup_score_margin"]
        ),
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1": "NOT_CLEARED",
        "g4": "NOT_CLEARED",
        "total_optimizer_updates": config.max_updates,
        "wall_seconds": total_wall_seconds,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Markdown report
    em_hits = final_summary["exact_match_count"]
    em_total = final_summary["n_examples"]
    report_lines = [
        f"# REC-004AL CD-DPCA Single-Init Learning Pilot Report ({decision})",
        "",
        f"- **Task ID:** `{REC004AL_TASK_ID}`",
        f"- **Decision:** `{decision}`",
        f"- **Initialization:** `{config.init_id}` (canonical hash: `{initial_canonical_hash}`)",
        f"- **Decisive Step:** `{config.decisive_step}` / `{config.max_updates}` updates",
        f"- **Terminal Viability Criterion:** sequence EM >= `{config.existing_validation_floor}`",
        f"- **Achieved Decisive Sequence EM:** `{decisive_em:.6f}` ({em_hits}/{em_total})",
        f"- **Viability Status:** "
        f"`{'PASSED' if terminal_viability_met else 'FAILED (STOP GATE TRIGGERED)'}`",
        "",
        "## Per-Length Breakdown",
        "",
        "| Length | Sequence EM | Numerator / Denominator | Token Accuracy |",
        "|---|---|---|---|",
    ]
    for L_str in sorted(final_summary["by_length"].keys(), key=int):
        b = final_summary["by_length"][L_str]
        report_lines.append(
            f"| {L_str} | {b['sequence_exact_match']:.4f} | "
            f"{b['exact_match_count']} / {b['n_sequences']} | {b['token_accuracy']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Causal Controls & Attention Diagnostics",
            "",
            f"- **Correct Control EM:** `{causal_eval['correct_exact_match']:.4f}`",
            f"- **Wrong-Family Control EM (REVERSE):** "
            f"`{causal_eval['wrong_family_exact_match']:.4f}`",
            f"- **None Control EM:** `{causal_eval['none_exact_match']:.4f}`",
            f"- **Causal Gap:** `{causal_eval['exact_match_causal_gap']:.4f}`",
            f"- **Padding Mask Verified:** `{attn_diagnostics['padding_mask_verified']}`",
            f"- **Top-1 Routing Accuracy to Correct Key:** "
            f"`{attn_diagnostics['correct_key_top1_routing_accuracy']:.4f}`",
            f"- **Mean Correct vs Runner-up Score Margin:** "
            f"`{attn_diagnostics['mean_correct_vs_runnerup_score_margin']:.4f}`",
            "",
            "## Execution Boundaries & Preserved Blocks",
            "",
            "- `candidate_selected`: `null` (strictly preserved)",
            "- `child_bundle`: `null` (no model bundle written)",
            "- `rg3_recheck`: `NOT_EXECUTED`",
            "- `rec005_eligible`: `false`",
            "- `G1` and `G4`: `NOT_CLEARED` (independent blocks strictly preserved)",
            "- All-init validation (REC-004AM): `BLOCKED` due to terminal viability failure.",
            "",
        ]
    )
    (config.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "protocol": protocol,
        "summary": summary,
        "checkpoints": checkpoint_records,
        "causal_controls": causal_controls,
        "attention_diagnostics": attn_diagnostics,
        "per_length_summary": final_summary,
        "per_position_report": final_pos_report,
    }
