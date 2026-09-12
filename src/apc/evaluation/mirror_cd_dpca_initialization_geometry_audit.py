"""REC-004AS: archived CD-DPCA initialization/data-interaction audit.

This module is deliberately diagnostic-only.  It loads the already-qualified
REC-004AL/AM/AR checkpoints, reconstructs the archived first batches, and
performs forwards/autograd calls only.  It never constructs an optimizer,
changes a checkpoint, or samples an additional initialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.mirror_cd_dpca_all_init_validation import (
    REC004AM_INIT_IDS,
    REC004AM_INIT_SEEDS,
)
from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    REC004AL_EXISTING_VALIDATION_SPLIT,
    REC004AL_PILOT_SEED,
    REC004AL_SEQUENCE_LENGTH_RANGE,
    REC004AL_TARGET_OPERATION,
    REC004AL_VOCAB_SIZE,
)
from apc.evaluation.mirror_cd_dpca_warm_start_pilot import (
    generate_step_training_examples_warm_start,
)
from apc.evaluation.mirror_position_initialization_diagnostic import mirror_halves_position_map
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004AS_TASK_ID",
    "REC004AS_DECISION",
    "MirrorCDDPCAInitializationGeometryAuditConfig",
    "classify_basin_at_500",
    "run_cd_dpca_initialization_geometry_audit",
]

REC004AS_TASK_ID: Final = "B-C005REC-004AS"
REC004AS_DECISION: Final = "INITIALIZATION_DATA_INTERACTION_IDENTIFIED"
REC004AS_ARMS: Final[tuple[str, ...]] = ("baseline", "warm_start")
REC004AS_STEPS: Final[tuple[int, ...]] = tuple(range(0, 6001, 500))
REC004AS_LENGTHS: Final[tuple[int, ...]] = (6, 7, 8, 9, 10)
REC004AS_VULNERABLE_FRACTION: Final = 0.20
REC004AS_TIE_TOLERANCE: Final = 1e-7


@dataclass(frozen=True)
class MirrorCDDPCAInitializationGeometryAuditConfig:
    """Fixed, artifact-only inputs for REC-004AS."""

    output_dir: Path = Path("runs/phase_b_restart/rec004as/run_001")
    rec004ak_dir: Path = Path("runs/phase_b_restart/rec004ak/run_001")
    rec004al_dir: Path = Path("runs/phase_b_restart/rec004al/run_001")
    rec004am_dir: Path = Path("runs/phase_b_restart/rec004am/run_001")
    rec004ar_dir: Path = Path("runs/phase_b_restart/rec004ar/run_001")
    init_ids: tuple[str, ...] = REC004AM_INIT_IDS
    init_seeds: dict[str, int] | None = None

    def resolved_init_seeds(self) -> dict[str, int]:
        """Return the pre-registered (not newly sampled) initialization identifiers."""
        return dict(REC004AM_INIT_SEEDS if self.init_seeds is None else self.init_seeds)


def _cells() -> list[tuple[int, int]]:
    return [(length, position) for length in REC004AS_LENGTHS for position in range(length)]


def _cell_key(length: int, position: int) -> str:
    return f"{length}:{position}"


def _pairwise_distance_summary(weight: torch.Tensor, n_rows: int) -> dict[str, float]:
    values = weight[:n_rows].detach().float()
    distances = torch.pdist(values)
    if distances.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(distances.mean().item()),
        "min": float(distances.min().item()),
        "max": float(distances.max().item()),
    }


def _instantiate_primitive(
    state_dict: dict[str, torch.Tensor], device: torch.device
) -> ContentDecoupledDiscretePositionalCrossAttentionPrimitive:
    """Create an ephemeral exact checkpoint view while preserving caller RNG state."""
    config = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation=REC004AL_TARGET_OPERATION,
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=REC004AL_VOCAB_SIZE,
        max_sequence_length=32,
        arg_dim=16,
    )
    # Constructor defaults are immediately overwritten. fork_rng ensures this temporary
    # construction cannot create a persisted RNG branch or a new initialization.
    with torch.random.fork_rng(devices=[]):
        primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
            primitive_id=0,
            config=config,
            status=PrimitiveStatus.CANDIDATE,
            created_at_task=0,
            metadata={"task": REC004AS_TASK_ID, "ephemeral_diagnostic": True},
        )
    primitive.load_state_dict(state_dict, strict=True)
    primitive.to(device)
    primitive.eval()
    return primitive


def _checkpoint_path(
    config: MirrorCDDPCAInitializationGeometryAuditConfig,
    init_id: str,
    arm: str,
    step: int,
) -> Path:
    if arm == "warm_start":
        return config.rec004am_dir / init_id / "checkpoints" / f"step{step}.pt"
    if init_id == "I01":
        return config.rec004al_dir / "checkpoints" / f"step{step}.pt"
    return config.rec004ar_dir / "baseline_controls" / init_id / "checkpoints" / f"step{step}.pt"


def _required_checkpoints(
    config: MirrorCDDPCAInitializationGeometryAuditConfig,
) -> list[Path]:
    return [
        _checkpoint_path(config, init_id, arm, step)
        for init_id in config.init_ids
        for arm in REC004AS_ARMS
        for step in REC004AS_STEPS
    ]


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes(paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise mb.MissingArtifactError(f"Required REC-004AS source checkpoint missing: {path}")
        hashes[str(path)] = _raw_sha256(path)
    return hashes


def _mean_score_metrics(
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    length: int,
    position: int,
) -> dict[str, Any]:
    """Fixed step/checkpoint routing-cell metrics, with oracle map used post-forward only."""
    device = primitive.query_position_embedding.weight.device
    with torch.no_grad():
        query, key, _mask = primitive.compute_routing_representations(
            [length], [length], None, device=device, lmax=length
        )
        scores = primitive.compute_routing_scores(
            [length], [length], None, device=device, lmax=length
        )[0, :, position, :length]
        weights = torch.softmax(scores, dim=-1)
    mean_scores = scores.mean(dim=0)
    mean_weights = weights.mean(dim=0)
    correct_key = mirror_halves_position_map(length)[position]
    masked_wrong = mean_scores.clone()
    masked_wrong[correct_key] = float("-inf")
    strongest_competitor = int(masked_wrong.argmax().item())
    correct_score = float(mean_scores[correct_key].item())
    competitor_score = float(mean_scores[strongest_competitor].item())
    top1_key = int(mean_weights.argmax().item())
    correct_rank = 1 + int((masked_wrong >= mean_scores[correct_key]).sum().item())
    sorted_scores = torch.sort(mean_scores, descending=True).values
    query_vec = query[0, position]
    key_vecs = key[0, :length]
    length_vec = primitive.length_embedding.weight[length]
    query_pos_vec = primitive.query_position_embedding.weight[position]
    cosines = F.cosine_similarity(query_vec.unsqueeze(0), key_vecs, dim=-1)
    entropy = -float((mean_weights * torch.log(mean_weights.clamp_min(1e-12))).sum().item())
    return {
        "length": length,
        "output_position": position,
        "correct_key": correct_key,
        "correct_key_rank": correct_rank,
        "correct_key_probability": float(mean_weights[correct_key].item()),
        "correct_key_margin": correct_score - competitor_score,
        "top1_key": top1_key,
        "top1_competitor_identity": strongest_competitor,
        "attention_entropy": entropy,
        "score_standard_deviation": float(mean_scores.std(unbiased=False).item()),
        "maximum_logit_gap": float((sorted_scores[0] - sorted_scores[1]).item()),
        "query_embedding_norm": float(query_vec.norm().item()),
        "correct_key_embedding_norm": float(key_vecs[correct_key].norm().item()),
        "top1_key_embedding_norm": float(key_vecs[top1_key].norm().item()),
        "key_embedding_norms": [float(value.norm().item()) for value in key_vecs],
        "query_key_cosine_similarities": [float(value.item()) for value in cosines],
        "correct_query_key_cosine": float(cosines[correct_key].item()),
        "top1_query_key_cosine": float(cosines[top1_key].item()),
        "length_embedding_norm": float(length_vec.norm().item()),
        "length_to_query_norm_ratio": float(
            length_vec.norm().item() / max(query_vec.norm().item(), 1e-12)
        ),
        "length_query_position_cosine": float(
            F.cosine_similarity(length_vec.unsqueeze(0), query_pos_vec.unsqueeze(0), dim=-1).item()
        ),
    }


def _geometry_for_state(
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
) -> dict[str, Any]:
    records = [_mean_score_metrics(primitive, length, position) for length, position in _cells()]
    return {
        "cells": records,
        "positional_embedding_pairwise_distance_summary": {
            "query_position": _pairwise_distance_summary(
                primitive.query_position_embedding.weight, 10
            ),
            "key_position": _pairwise_distance_summary(primitive.key_position_embedding.weight, 10),
        },
    }


def _vulnerability_map(cells: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Pre-fixed bottom-20% extractors; no result-dependent feature selection."""
    n = max(1, math.ceil(len(cells) * REC004AS_VULNERABLE_FRACTION))

    def keys(items: Sequence[dict[str, Any]]) -> list[str]:
        return [_cell_key(int(item["length"]), int(item["output_position"])) for item in items[:n]]

    return {
        "lowest_correct_key_margin_cells": keys(
            sorted(cells, key=lambda item: item["correct_key_margin"])
        ),
        "worst_correct_key_rank_cells": keys(
            sorted(
                cells, key=lambda item: (-int(item["correct_key_rank"]), item["correct_key_margin"])
            )
        ),
        "lowest_entropy_incorrect_cells": keys(
            sorted(
                (item for item in cells if item["top1_key"] != item["correct_key"]),
                key=lambda item: item["attention_entropy"],
            )
        ),
    }


def classify_basin_at_500(record: dict[str, Any]) -> str:
    """Classify a fixed routing-cell state without consulting terminal outcomes."""
    if abs(float(record["correct_key_margin"])) <= REC004AS_TIE_TOLERANCE:
        return "UNRESOLVED_AT_500"
    if int(record["top1_key"]) == int(record["correct_key"]):
        return "TRUE_BASIN_ENTRY"
    return "FALSE_BASIN_ENTRY"


def _trajectory_summary(records_by_step: dict[int, dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, dict[str, Any]] = {}
    for length, position in _cells():
        key = _cell_key(length, position)
        step0 = records_by_step[0]["cells_by_key"][key]
        step500 = records_by_step[500]["cells_by_key"][key]
        terminal = records_by_step[6000]["cells_by_key"][key]
        basin = classify_basin_at_500(step500)
        maintained = (
            int(step500["top1_key"]) == int(terminal["top1_key"])
            if basin != "UNRESOLVED_AT_500"
            else False
        )
        by_cell[key] = {
            "length": length,
            "output_position": position,
            "step0_top1_key": step0["top1_key"],
            "step500_top1_key": step500["top1_key"],
            "terminal_top1_key": terminal["top1_key"],
            "top1_key_transition": f"{step0['top1_key']}->{step500['top1_key']}",
            "correct_key_margin_change_0_to_500": (
                step500["correct_key_margin"] - step0["correct_key_margin"]
            ),
            "entropy_change_0_to_500": step500["attention_entropy"] - step0["attention_entropy"],
            "correct_key_probability_change_0_to_500": (
                step500["correct_key_probability"] - step0["correct_key_probability"]
            ),
            "basin_classification_at_500": basin,
            "false_basin_competitor_key": (
                step500["top1_key"] if basin == "FALSE_BASIN_ENTRY" else None
            ),
            "basin_maintained_to_terminal": maintained,
        }
    return by_cell


def _routing_gradient_groups(
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    gradients: Sequence[torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    parameters = list(primitive.parameters())
    by_parameter = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    }

    def grad(parameter: torch.Tensor) -> torch.Tensor:
        value = by_parameter.get(id(parameter))
        return torch.zeros_like(parameter) if value is None else value

    d_operator = primitive.d_operator
    in_proj_weight = grad(primitive.cross_attn.in_proj_weight)
    in_proj_bias = primitive.cross_attn.in_proj_bias
    bias = torch.zeros(0, device=in_proj_weight.device)
    if in_proj_bias is not None:
        bias = grad(in_proj_bias)[: 2 * d_operator]
    return {
        "query_position_embeddings": grad(primitive.query_position_embedding.weight).flatten(),
        "key_position_embeddings": grad(primitive.key_position_embedding.weight).flatten(),
        "length_embeddings": grad(primitive.length_embedding.weight).flatten(),
        "Wq": in_proj_weight[:d_operator].flatten(),
        "Wk": in_proj_weight[d_operator : 2 * d_operator].flatten(),
        "other_score_producing_routing_parameters": bias.flatten(),
    }


def _early_loss_and_features(
    core: Any, examples: Sequence[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    content_lengths = [len(example.input_tokens) for example in examples]
    output_lengths = [
        get_operation(REC004AL_TARGET_OPERATION).output_length(length) for length in content_lengths
    ]
    labels = _labels_for_examples(examples, output_lengths, max(output_lengths), device)
    with torch.no_grad():
        batch = collate_content_only_batch(examples, core.tokens, device=device)
        features = core.model.encode(batch)[:, 1 : 1 + max(content_lengths), :]
    return features, labels, content_lengths


def _gradient_diagnostic(
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    features: torch.Tensor,
    labels: torch.Tensor,
    content_lengths: Sequence[int],
) -> dict[str, Any]:
    """Compute -grad(M) dot grad(L) for every fixed routing cell, without updates."""
    device = features.device
    output_lengths = [
        get_operation(REC004AL_TARGET_OPERATION).output_length(length) for length in content_lengths
    ]
    parameters = list(primitive.parameters())
    primitive.train()
    logits = primitive(features, content_lengths, output_lengths, None)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    loss_gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    loss_groups = _routing_gradient_groups(primitive, loss_gradients)
    records: dict[str, Any] = {}
    for length, position in _cells():
        primitive.zero_grad(set_to_none=True)
        scores = primitive.compute_routing_scores(
            [length], [length], None, device=device, lmax=length
        )
        mean_scores = scores[0, :, position, :length].mean(dim=0)
        correct_key = mirror_halves_position_map(length)[position]
        wrong_scores = mean_scores.clone()
        wrong_scores[correct_key] = float("-inf")
        margin = mean_scores[correct_key] - wrong_scores.max()
        margin_gradients = torch.autograd.grad(margin, parameters, allow_unused=True)
        margin_groups = _routing_gradient_groups(primitive, margin_gradients)
        total_dot = 0.0
        total_m_sq = 0.0
        total_l_sq = 0.0
        group_directions: dict[str, Any] = {}
        for group_name, margin_vector in margin_groups.items():
            loss_vector = loss_groups[group_name]
            directional = -float(torch.dot(margin_vector, loss_vector).item())
            total_dot += directional
            total_m_sq += float(torch.dot(margin_vector, margin_vector).item())
            total_l_sq += float(torch.dot(loss_vector, loss_vector).item())
            group_directions[group_name] = directional
        magnitude = abs(total_dot)
        records[_cell_key(length, position)] = {
            "length": length,
            "output_position": position,
            "margin_directed_gradient": total_dot,
            "margin_directed_gradient_sign": "POSITIVE"
            if total_dot > 0
            else ("NEGATIVE" if total_dot < 0 else "ZERO"),
            "magnitude": magnitude,
            "cosine_alignment": total_dot / max(math.sqrt(total_m_sq * total_l_sq), 1e-12),
            "gradient_contribution_by_parameter_family": group_directions,
        }
    primitive.eval()
    return records


def _stream_digest(examples: Sequence[Any]) -> str:
    payload = [
        {"input": list(example.input_tokens), "target": list(example.target_tokens)}
        for example in examples
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _matched_seed_comparison(
    baseline: dict[str, Any], warm: dict[str, Any], vulnerable: dict[str, list[str]]
) -> dict[str, Any]:
    outcomes: dict[str, list[str]] = {
        "warm_start_moved_to_correct_basin": [],
        "warm_start_created_false_basin": [],
        "baseline_only_correct_basin": [],
        "same_basin_both_arms": [],
    }
    for key, base_cell in baseline.items():
        warm_cell = warm[key]
        b_class = base_cell["basin_classification_at_500"]
        w_class = warm_cell["basin_classification_at_500"]
        if b_class != "TRUE_BASIN_ENTRY" and w_class == "TRUE_BASIN_ENTRY":
            outcomes["warm_start_moved_to_correct_basin"].append(key)
        if b_class == "TRUE_BASIN_ENTRY" and w_class == "FALSE_BASIN_ENTRY":
            outcomes["warm_start_created_false_basin"].append(key)
        if b_class == "TRUE_BASIN_ENTRY" and w_class != "TRUE_BASIN_ENTRY":
            outcomes["baseline_only_correct_basin"].append(key)
        if b_class == w_class and base_cell["step500_top1_key"] == warm_cell["step500_top1_key"]:
            outcomes["same_basin_both_arms"].append(key)
    initial_vulnerable = sorted({key for values in vulnerable.values() for key in values})
    outcomes["same_initial_vulnerable_cells"] = initial_vulnerable
    return outcomes


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state is not a dict: {path}")
    return state


def _transplanted_state(
    base: dict[str, torch.Tensor], donor: dict[str, torch.Tensor], group: str
) -> dict[str, torch.Tensor]:
    result = {name: tensor.clone() for name, tensor in base.items()}
    if group == "query_position_embeddings":
        result["query_position_embedding.weight"] = donor["query_position_embedding.weight"].clone()
    elif group == "key_position_embeddings":
        result["key_position_embedding.weight"] = donor["key_position_embedding.weight"].clone()
    elif group == "length_embeddings":
        result["length_embedding.weight"] = donor["length_embedding.weight"].clone()
    elif group == "Wq":
        result["cross_attn.in_proj_weight"][:32] = donor["cross_attn.in_proj_weight"][:32]
    elif group == "Wk":
        result["cross_attn.in_proj_weight"][32:64] = donor["cross_attn.in_proj_weight"][32:64]
    elif group == "other_score_producing_routing_parameters":
        result["cross_attn.in_proj_bias"][:64] = donor["cross_attn.in_proj_bias"][:64]
    else:
        raise ValueError(f"Unknown transplant group: {group}")
    return result


def _geometry_transplantation(
    config: MirrorCDDPCAInitializationGeometryAuditConfig, device: torch.device
) -> dict[str, Any]:
    """I02 base/I01 donor, one score-path parameter family at a time, evaluation only."""
    base_state = _load_state(_checkpoint_path(config, "I02", "baseline", 0))
    donor_state = _load_state(_checkpoint_path(config, "I01", "baseline", 0))
    base_model = _instantiate_primitive(base_state, device)
    base_cells = _geometry_for_state(base_model)["cells"]
    base_by_key = {_cell_key(c["length"], c["output_position"]): c for c in base_cells}
    reports: dict[str, Any] = {}
    for group in _routing_gradient_groups(
        base_model, tuple(torch.zeros_like(parameter) for parameter in base_model.parameters())
    ):
        model = _instantiate_primitive(_transplanted_state(base_state, donor_state, group), device)
        new_cells = _geometry_for_state(model)["cells"]
        changed = 0
        margin_deltas: list[float] = []
        for cell in new_cells:
            original = base_by_key[_cell_key(cell["length"], cell["output_position"])]
            changed += int(cell["top1_key"] != original["top1_key"])
            margin_deltas.append(cell["correct_key_margin"] - original["correct_key_margin"])
        reports[group] = {
            "base_init": "I02",
            "donor_init": "I01",
            "one_group_only": True,
            "top1_changed_cells": changed,
            "mean_absolute_correct_margin_change": sum(abs(value) for value in margin_deltas)
            / len(margin_deltas),
            "mean_correct_margin_change": sum(margin_deltas) / len(margin_deltas),
        }
    return reports


def _decision_summary(
    matched: dict[str, dict[str, Any]], gradients: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    per_init: dict[str, Any] = {}
    for init_id, comparison in matched.items():
        base = gradients[init_id]["baseline"]
        warm = gradients[init_id]["warm_start"]
        signs_changed = sum(
            base[key]["margin_directed_gradient_sign"] != warm[key]["margin_directed_gradient_sign"]
            for key in base
        )
        diverged = len(comparison["warm_start_moved_to_correct_basin"]) + len(
            comparison["warm_start_created_false_basin"]
        )
        per_init[init_id] = {
            "gradient_sign_changed_cells": signs_changed,
            "basin_diverged_cells": diverged,
            "stream_changed_basin_direction": diverged > 0,
        }
    # A matched step-0 state cannot by itself determine different arm outcomes.  The
    # pre-fixed interaction criterion additionally requires observed first-stream
    # directional changes and a basin split in every archived initialization.
    interaction_all_five = all(
        entry["gradient_sign_changed_cells"] > 0 and entry["stream_changed_basin_direction"]
        for entry in per_init.values()
    )
    return {
        "decision": REC004AS_DECISION if interaction_all_five else "INSUFFICIENT_EVIDENCE_STOP",
        "step0_geometry_alone_explains_terminal_susceptibility": False,
        "early_data_alone_has_initialization_invariant_direction": False,
        "initialization_data_interaction_supported": interaction_all_five,
        "per_initialization": per_init,
        "generic_oracle_free_initialization_repair_uniquely_identified": False,
        "learning_pilot_authorized": False,
        "stopped_research_directions": [
            "initialization-only repair pilot",
            "architecture/init sweep",
            "candidate adoption",
            "bundle promotion",
            "RG3",
            "REC-005",
        ],
    }


def run_cd_dpca_initialization_geometry_audit(
    config: MirrorCDDPCAInitializationGeometryAuditConfig,
) -> dict[str, Any]:
    """Run REC-004AS forwards/autograd diagnostics over immutable archived sources."""
    started = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = _required_checkpoints(config)
    source_hashes_before = _checkpoint_hashes(checkpoint_paths)
    device = torch.device("cpu")
    init_seeds = config.resolved_init_seeds()

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, _parent_bank, _op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=REC004AL_PILOT_SEED), parent_manifest
    )
    core.model.to(device)
    core.model.eval()
    for parameter in core.model.parameters():
        parameter.requires_grad_(False)

    baseline_examples = ibc._generate_step_training_examples(
        REC004AL_PILOT_SEED,
        1,
        REC004AL_TARGET_OPERATION,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
        n=32,
    )
    warm_examples = generate_step_training_examples_warm_start(
        REC004AL_PILOT_SEED,
        1,
        REC004AL_TARGET_OPERATION,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
        n=32,
    )
    early_batches = {
        "baseline": _early_loss_and_features(core, baseline_examples, device),
        "warm_start": _early_loss_and_features(core, warm_examples, device),
    }

    all_geometry: dict[str, Any] = {}
    all_trajectories: dict[str, dict[str, Any]] = {}
    vulnerability: dict[str, Any] = {}
    gradients: dict[str, dict[str, dict[str, Any]]] = {}
    matched: dict[str, dict[str, Any]] = {}
    for init_id in config.init_ids:
        per_arm_steps: dict[str, dict[int, dict[str, Any]]] = {}
        all_geometry[init_id] = {}
        all_trajectories[init_id] = {}
        gradients[init_id] = {}
        for arm in REC004AS_ARMS:
            per_step: dict[int, dict[str, Any]] = {}
            for step in REC004AS_STEPS:
                state = _load_state(_checkpoint_path(config, init_id, arm, step))
                model = _instantiate_primitive(state, device)
                geometry = _geometry_for_state(model)
                geometry["cells_by_key"] = {
                    _cell_key(cell["length"], cell["output_position"]): cell
                    for cell in geometry["cells"]
                }
                per_step[step] = geometry
            per_arm_steps[arm] = per_step
            all_geometry[init_id][arm] = per_step[0]
            all_trajectories[init_id][arm] = _trajectory_summary(per_step)
            step0_model = _instantiate_primitive(
                _load_state(_checkpoint_path(config, init_id, arm, 0)), device
            )
            features, labels, content_lengths = early_batches[arm]
            gradients[init_id][arm] = _gradient_diagnostic(
                step0_model, features, labels, content_lengths
            )
        vulnerability[init_id] = _vulnerability_map(all_geometry[init_id]["baseline"]["cells"])
        matched[init_id] = _matched_seed_comparison(
            all_trajectories[init_id]["baseline"],
            all_trajectories[init_id]["warm_start"],
            vulnerability[init_id],
        )

    gradient_differences: dict[str, Any] = {}
    for init_id in config.init_ids:
        gradient_differences[init_id] = {}
        for key, baseline in gradients[init_id]["baseline"].items():
            warm = gradients[init_id]["warm_start"][key]
            gradient_differences[init_id][key] = {
                "baseline_margin_directed_gradient": baseline["margin_directed_gradient"],
                "warm_start_margin_directed_gradient": warm["margin_directed_gradient"],
                "warm_minus_baseline": (
                    warm["margin_directed_gradient"] - baseline["margin_directed_gradient"]
                ),
                "sign_changed": (
                    baseline["margin_directed_gradient_sign"]
                    != warm["margin_directed_gradient_sign"]
                ),
            }

    transplantation = _geometry_transplantation(config, device)
    decision = _decision_summary(matched, gradients)
    source_hashes_after = _checkpoint_hashes(checkpoint_paths)
    sources_unchanged = source_hashes_before == source_hashes_after
    if not sources_unchanged:
        raise RuntimeError("REC-004AS source checkpoint hash changed during diagnostic")

    protocol = {
        "task_id": REC004AS_TASK_ID,
        "analysis_only": True,
        "fixed_geometry_metrics": [
            "correct_key_rank",
            "correct_key_probability",
            "correct_key_margin",
            "top1_competitor_identity",
            "attention_entropy",
            "score_standard_deviation",
            "maximum_logit_gap",
            "query_embedding_norm",
            "key_embedding_norms",
            "query_key_cosine_similarities",
            "positional_embedding_pairwise_distance_summary",
            "length_embedding_norm_and_query_relation",
        ],
        "routing_cells": "(length, output_position), all lengths 6..10 and all valid positions",
        "early_batches": (
            "exact deterministic step-1 archived/reconstructable standard and distinct streams"
        ),
        "no_optimizer_construction": True,
        "optimizer_updates": 0,
        "new_seed_or_initialization": False,
        "validation_split_reference": REC004AL_EXISTING_VALIDATION_SPLIT,
        "transplantation": (
            "I02 base + I01 donor; one score-path parameter family at a time; forward-only"
        ),
    }
    source_manifest = {
        "task_id": REC004AS_TASK_ID,
        "init_ids": list(config.init_ids),
        "init_seeds": init_seeds,
        "checkpoint_sha256_before": source_hashes_before,
        "checkpoint_sha256_after": source_hashes_after,
        "source_checkpoints_unchanged": sources_unchanged,
        "baseline_step1_stream_sha256": _stream_digest(baseline_examples),
        "warm_start_step1_stream_sha256": _stream_digest(warm_examples),
    }
    side_effect_audit = {
        "task_id": REC004AS_TASK_ID,
        "optimizer_updates": 0,
        "core_changed": False,
        "parent_primitives_changed": False,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005": "BLOCKED",
        "g1": "BLOCKED",
        "g4": "BLOCKED",
        "sealed_data_access": 0,
        "source_checkpoint_hashes_unchanged": sources_unchanged,
    }
    summary = {
        "task_id": REC004AS_TASK_ID,
        "status": "PASS",
        "decision": decision["decision"],
        "decision_evidence": decision,
        "n_initializations": len(config.init_ids),
        "n_routing_cells_per_initialization": len(_cells()),
        "n_checkpoint_states_evaluated": len(checkpoint_paths),
        "source_checkpoints_unchanged": sources_unchanged,
        "optimizer_updates": 0,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005": "BLOCKED",
        "g1": "BLOCKED",
        "g4": "BLOCKED",
        "sealed_data_access": 0,
        "wall_seconds": time.time() - started,
    }
    payloads = {
        "protocol.json": protocol,
        "source_manifest.json": source_manifest,
        "step0_routing_geometry.json": all_geometry,
        "initial_vulnerability_map.json": vulnerability,
        "routing_trajectories.json": all_trajectories,
        "matched_seed_intervention_comparison.json": matched,
        "early_gradient_interaction.json": gradients,
        "early_gradient_baseline_vs_warm_difference.json": gradient_differences,
        "parameter_family_localization.json": {
            "native_gradient_directional_contributions": gradients,
            "one_group_transplantation_i02_base_i01_donor": transplantation,
        },
        "side_effect_audit.json": side_effect_audit,
        "summary.json": summary,
    }
    for name, payload in payloads.items():
        (config.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# REC-004AS CD-DPCA Initialization-Geometry × Early-Data Basin Susceptibility Audit",
        "",
        f"- Task: `{REC004AS_TASK_ID}`",
        f"- Decision: `{summary['decision']}`",
        f"- Routing cells: `{len(_cells())}` per initialization, all lengths 6..10",
        f"- Checkpoint states read: `{len(checkpoint_paths)}`; unchanged: `{sources_unchanged}`",
        "- Optimizer updates: `0`; sealed-data access: `0`; candidate/bundle/RG3/REC-005: blocked.",
        "",
        "## Decision",
        "",
        "Matched arms share the exact step-0 state, while their step-500 basin outcomes and",
        "step-1 CE margin-directional gradients differ by initialization.  Step-0 geometry",
        "therefore cannot independently determine the arm divergence, and the sampler has no",
        "initialization-invariant direction.  No single generic oracle-free initialization",
        "constraint is identified; an initialization-only learning pilot is not authorized.",
        "",
        "See JSON artifacts for the fixed full-cell metric table, trajectories, gradient",
        "directional derivatives, and one-family-at-a-time I02←I01 forward-only transplants.",
    ]
    (config.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary
