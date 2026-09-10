"""B-C005REC-004O: I03 score-only continuation from the complete 6000 state.

This is deliberately a single-init, no-adoption repair pilot.  It preserves
the historical AdamW/Scheduler/RNG continuation contract while applying a
fail-closed row-wise guard to the fused Q/K/V tensor: Q/K and position bias
may update, V and every O1 downstream parameter may not.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as rec004g
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_ffn_anchored_downstream_interaction_audit as rec004l
from apc.evaluation import mirror_ffn_value_path_leave_one_out_necessity_audit as rec004n
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.evaluation.unified_oracle_causal_benchmark import _derive_local_seed
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004O_TASK_ID",
    "REC004O_SOURCE_STEP",
    "REC004O_TARGET_STEP",
    "MirrorScoreOnlyContinuationPilotConfig",
    "build_score_only_datasets",
    "build_forward_graph_partition",
    "run_joint_control_replay",
    "run_mirror_score_only_continuation_pilot_task",
]

REC004O_TASK_ID: Final = "B-C005REC-004O"
REC004O_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_I03_SCORE_ONLY_CONTINUATION_PILOT.md"
)
REC004O_INIT_ID: Final = "I03"
REC004O_ARM: Final = rec004g.REC004G_ARM
REC004O_SOURCE_STEP: Final = rec004g.REC004G_SOURCE_STEP
REC004O_TARGET_STEP: Final = rec004g.REC004G_TARGET_CUMULATIVE_STEP
REC004O_CONTROL_REPLAY_STEP: Final = 6500
REC004O_CHECKPOINT_INTERVAL: Final = rec004g.REC004G_CHECKPOINT_INTERVAL
REC004O_VALIDATION_SPLIT: Final = "score_only_repair_validation_v1"
REC004O_CONFIRMATION_SPLIT: Final = "score_only_length10_confirmation_v1"
REC004O_VALIDATION_EXAMPLES: Final = 1024
REC004O_CONFIRMATION_EXAMPLES: Final = 512
REC004O_CONFIRMATION_LENGTH: Final = 10
REC004O_EM_FLOOR: Final = 0.95
REC004O_CHUNK_SIZE: Final = 128

_FULLY_FROZEN_PREFIXES: Final[tuple[str, ...]] = (
    "content_in_proj.",
    "content_position_embedding.",
    "cross_attn.out_proj.",
    "answer_query_embedding.",
    "attn_norm.",
    "ffn.",
    "ffn_norm.",
    "readout.",
)
_POSITION_BIAS_PREFIXES: Final[tuple[str, ...]] = (
    "position_bias_hidden.",
    "position_bias_out.",
)
_FUSED_QKV_NAMES: Final[tuple[str, ...]] = (
    "cross_attn.in_proj_weight",
    "cross_attn.in_proj_bias",
)


class SelectiveFreezeUnsafe(RuntimeError):
    """Raised before continuing when a row-level freeze contract is not provable."""


@dataclass(frozen=True)
class MirrorScoreOnlyContinuationPilotConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004o/run_001")
    seed: int = RECOVERY_PILOT_SEED
    checkpoint_interval: int = REC004O_CHECKPOINT_INTERVAL


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _state_equal(left: Any, right: Any, path: str = "state") -> tuple[bool, str | None]:
    """Exact recursive comparison that also identifies the first mismatch."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            (True, None)
            if torch.equal(left.detach().cpu(), right.detach().cpu())
            else (False, path)
        )
    if type(left) is not type(right):
        return False, f"{path}:type({type(left).__name__}!={type(right).__name__})"
    if isinstance(left, dict):
        if set(left) != set(right):
            return False, f"{path}:keys"
        for key in sorted(left, key=str):
            ok, detail = _state_equal(left[key], right[key], f"{path}.{key}")
            if not ok:
                return ok, detail
        return True, None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False, f"{path}:length"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            ok, detail = _state_equal(a, b, f"{path}[{index}]")
            if not ok:
                return ok, detail
        return True, None
    return (True, None) if left == right else (False, path)


def _state_fingerprint(value: Any) -> str:
    """Stable hash for a nested state dict, including exact tensor bytes."""
    h = hashlib.sha256()

    def _visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(b"tensor:")
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            # AdamW's per-parameter ``step`` is a scalar tensor.  Reshape it
            # before byte-viewing: PyTorch disallows a dtype-changing view of
            # a 0-D tensor even though its byte representation is well-defined.
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict:")
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                _visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b"list:")
            for child in item:
                _visit(child)
        else:
            h.update(repr(item).encode())

    _visit(value)
    return h.hexdigest()


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _slice_rows(tensor: torch.Tensor) -> slice:
    if tensor.ndim < 1 or tensor.shape[0] % 3 != 0:
        raise SelectiveFreezeUnsafe("FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: invalid fused QKV shape")
    row_count = tensor.shape[0] // 3
    return slice(2 * row_count, 3 * row_count)


def _digest_examples(examples: list[Example]) -> set[str]:
    return {traj_audit._digest_example(example) for example in examples}


def _build_dataset(
    *, seed: int, split: str, n_examples: int, protected: set[str], fixed_length: int | None
) -> tuple[list[Example], dict[str, Any]]:
    rng = random.Random(_derive_local_seed(seed, 0, f"{split}:{rec004g.REC004G_TARGET_OPERATION}"))
    op_name = rec004g.REC004G_TARGET_OPERATION
    op = get_operation(op_name)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    local_protected = set(protected)
    total_draws = 0
    for slot in range(n_examples):
        rejected: list[str] = []
        while True:
            total_draws += 1
            length = fixed_length or rng.randint(*rec004g.REC004G_SEQUENCE_LENGTH_RANGE)
            sequence = tuple(rng.randrange(rec004g.REC004G_VOCAB_SIZE) for _ in range(length))
            params = op.sample_params(rng, sequence, rec004g.REC004G_VOCAB_SIZE)
            program = Program(steps=(ProgramStep(operation=op_name, params=params),))
            result = run_program(program, sequence, rec004g.REC004G_VOCAB_SIZE)
            digest = traj_audit._digest(sequence, result.output_tokens)
            if digest not in local_protected:
                break
            rejected.append(digest)
        local_protected.add(digest)
        if rejected:
            substitutions.append(
                {"slot": slot, "rejected_digests": rejected, "accepted_digest": digest}
            )
        examples.append(
            Example(
                input_tokens=sequence,
                target_tokens=result.output_tokens,
                program=program,
                operation_graph=result.graph,
                category="known",
                split=split,
                vocab_size=rec004g.REC004G_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(program),
                oracle_metadata=OracleMetadata(label="K", primitive_operations=(op_name,)),
            )
        )
    detail = {
        "split": split,
        "n": len(examples),
        "fixed_length": fixed_length,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "frozen_dataset_digest_sha256": rec004l._dataset_digest(examples),
    }
    return examples, detail


def build_score_only_datasets(seed: int) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Lock both new data sets before any model forward.

    REC-004N's protected registry already includes all prior REC-004A–N
    validation/reference/query/diagnostic additions.  Its base registry uses
    the full 1–18000 stream, a strict superset of this task's required 1–12000.
    """
    prior_datasets, prior_details = rec004n.build_stage_datasets(seed)
    protected, protected_counts = traj_audit.build_protected_digest_registry(seed)
    for examples in prior_datasets.values():
        protected |= _digest_examples(examples)
    validation, validation_detail = _build_dataset(
        seed=seed,
        split=REC004O_VALIDATION_SPLIT,
        n_examples=REC004O_VALIDATION_EXAMPLES,
        protected=protected,
        fixed_length=None,
    )
    protected_after_validation = protected | _digest_examples(validation)
    confirmation, confirmation_detail = _build_dataset(
        seed=seed,
        split=REC004O_CONFIRMATION_SPLIT,
        n_examples=REC004O_CONFIRMATION_EXAMPLES,
        protected=protected_after_validation,
        fixed_length=REC004O_CONFIRMATION_LENGTH,
    )
    validation_digests = _digest_examples(validation)
    confirmation_digests = _digest_examples(confirmation)
    overlap: dict[str, Any] = {
        "validation_with_prior": sorted(validation_digests & protected),
        "confirmation_with_prior": sorted(confirmation_digests & protected),
        "new_sets_mutual": sorted(validation_digests & confirmation_digests),
    }
    overlap["disjoint"] = not any(overlap.values())
    details = {
        REC004O_VALIDATION_SPLIT: validation_detail,
        REC004O_CONFIRMATION_SPLIT: confirmation_detail,
        "protected_registry": {
            "base_source_counts": protected_counts,
            "prior_rec004n_dataset_details": prior_details,
            "protected_digest_count_before_new_sets": len(protected),
            "training_stream_contract": "steps_1_to_18000 (strict superset of required 1_to_12000)",
        },
        "cross_dataset_disjointness": overlap,
    }
    if not overlap["disjoint"]:
        raise SelectiveFreezeUnsafe("dataset digest collision after pre-output construction")
    return {REC004O_VALIDATION_SPLIT: validation, REC004O_CONFIRMATION_SPLIT: confirmation}, details


def build_forward_graph_partition(primitive: Any) -> dict[str, Any]:
    """Classify actual parameter objects, failing on every unaccounted key.

    The identifier set comes from `named_parameters()` of the loaded primitive,
    not a checkpoint-name guess.  The fixed module roles are then checked
    exhaustively against the real runtime parameter inventory.
    """
    names = [name for name, _parameter in primitive.named_parameters()]
    state_keys = list(primitive.state_dict().keys())
    if set(names) != set(state_keys):
        raise SelectiveFreezeUnsafe(
            "FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: named parameter/state_dict inventory differs"
        )
    groups: dict[str, list[str]] = {
        "frozen_downstream": [],
        "score_position_bias": [],
        "fused_qk": [],
        "fused_v": [],
    }
    for name in names:
        if name in _FUSED_QKV_NAMES:
            parameter = dict(primitive.named_parameters())[name]
            _slice_rows(parameter)
            groups["fused_qk"].append(name)
            groups["fused_v"].append(name)
        elif name.startswith(_POSITION_BIAS_PREFIXES):
            groups["score_position_bias"].append(name)
        elif name.startswith(_FULLY_FROZEN_PREFIXES):
            groups["frozen_downstream"].append(name)
        else:
            raise SelectiveFreezeUnsafe(
                f"FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: no forward-graph role for {name}"
            )
    covered = set(groups["frozen_downstream"] + groups["score_position_bias"] + groups["fused_qk"])
    if covered != set(names) or not groups["fused_qk"] or not groups["score_position_bias"]:
        raise SelectiveFreezeUnsafe("FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: incomplete score partition")
    return {
        "parameter_names_from_loaded_runtime": names,
        "state_dict_keys_from_loaded_runtime": state_keys,
        "groups": groups,
        "trainable_tensor_keys": groups["fused_qk"] + groups["score_position_bias"],
        "frozen_whole_tensor_keys": groups["frozen_downstream"],
        "fused_qkv_row_order": "Q, K, V = consecutive thirds on dim=0",
    }


def _forward_backward_trace(core: Any, primitive: Any, examples: list[Example]) -> dict[str, Any]:
    """Real gradient trace used to reject a nominal-but-dead partition."""
    prior_requires = {
        name: parameter.requires_grad for name, parameter in primitive.named_parameters()
    }
    try:
        primitive.train()
        primitive.zero_grad(set_to_none=True)
        for _name, parameter in primitive.named_parameters():
            parameter.requires_grad_(True)
        chunk = examples[: min(16, len(examples))]
        lengths = [len(example.input_tokens) for example in chunk]
        output_lengths = [len(example.target_tokens) for example in chunk]
        batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
        with torch.no_grad():
            h_content = core.model.encode(batch)[:, 1 : 1 + max(lengths), :]
        labels = _labels_for_examples(chunk, output_lengths, max(output_lengths), core.device)
        loss = F.cross_entropy(
            primitive(h_content, lengths, output_lengths, None).reshape(
                -1, rec004g.REC004G_VOCAB_SIZE
            ),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        gradients = {
            name: {
                "has_gradient": parameter.grad is not None,
                "l1": float(parameter.grad.abs().sum().item())
                if parameter.grad is not None
                else None,
            }
            for name, parameter in primitive.named_parameters()
        }
        if not all(row["has_gradient"] for row in gradients.values()):
            raise SelectiveFreezeUnsafe(
                "FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: parameter absent from "
                "real forward/backward trace"
            )
        return {"loss": float(loss.item()), "gradients": gradients}
    finally:
        primitive.zero_grad(set_to_none=True)
        for name, parameter in primitive.named_parameters():
            parameter.requires_grad_(prior_requires[name])


def _build_resumed_runtime(
    core: Any, training_state: dict[str, Any]
) -> tuple[Any, torch.optim.AdamW, Any]:
    primitive = mpbr._new_arm_primitive(core, REC004O_ARM)
    primitive.to(core.device)
    primitive.load_state_dict(
        {
            key: value.to(core.device)
            for key, value in training_state["primitive_state_dict"].items()
        },
        strict=True,
    )
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=rec004g.REC004G_OPERATOR_LR,
        weight_decay=rec004g.REC004G_OPERATOR_WEIGHT_DECAY,
    )
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=rec004g.REC004G_T_MAX,
        eta_min=rec004g.REC004G_SCHEDULER_ETA_MIN,
    )
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    if scheduler.last_epoch != REC004O_SOURCE_STEP:
        raise SelectiveFreezeUnsafe(
            "FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: unexpected scheduler source counter"
        )
    return primitive, optimizer, scheduler


def _training_state(
    primitive: Any, optimizer: Any, scheduler: Any, device: torch.device, step: int
) -> dict[str, Any]:
    return {
        "primitive_state_dict": {
            key: value.detach().clone().cpu() for key, value in primitive.state_dict().items()
        },
        "optimizer_state_dict": _clone_state(optimizer.state_dict()),
        "scheduler_state_dict": _clone_state(scheduler.state_dict()),
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None,
        "cumulative_updates": step,
        "init_id": REC004O_INIT_ID,
        "arm": REC004O_ARM,
    }


def _one_recipe_update(
    core: Any, primitive: Any, optimizer: Any, scheduler: Any, step: int
) -> float:
    examples = ibc._generate_step_training_examples(
        RECOVERY_PILOT_SEED,
        step,
        rec004g.REC004G_TARGET_OPERATION,
        vocab_size=rec004g.REC004G_VOCAB_SIZE,
        sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
    )
    lengths = [len(example.input_tokens) for example in examples]
    output_lengths = [
        get_operation(rec004g.REC004G_TARGET_OPERATION).output_length(length) for length in lengths
    ]
    labels = _labels_for_examples(examples, output_lengths, max(output_lengths), core.device)
    with torch.no_grad():
        batch = collate_content_only_batch(examples, core.tokens, device=core.device)
        h_content = core.model.encode(batch)[:, 1 : 1 + max(lengths), :]
    optimizer.zero_grad(set_to_none=True)
    logits = primitive(h_content, lengths, output_lengths, None)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(primitive.parameters(), rec004g.REC004G_OPERATOR_GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    if not math.isfinite(float(loss.item())):
        raise SelectiveFreezeUnsafe(f"non-finite joint-control loss at step {step}")
    return float(loss.item())


def _rec004g_state_path(step: int) -> Path:
    return (
        Path("runs/phase_b_b2_model_bundle_recovery/rec004g/run_001")
        / REC004O_INIT_ID
        / REC004O_ARM
        / "training_states"
        / f"step{step}.pt"
    )


def run_joint_control_replay(core: Any, source_state: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the historical all-parameter I03 6000→6500 continuation exactly."""
    expected_path = _rec004g_state_path(REC004O_CONTROL_REPLAY_STEP)
    if not expected_path.is_file():
        return {"status": "JOINT_CONTROL_REPLAY_MISMATCH", "reason": f"missing {expected_path}"}
    expected = torch.load(expected_path, map_location="cpu", weights_only=False)
    primitive, optimizer, scheduler = _build_resumed_runtime(core, source_state)
    primitive.train()
    for parameter in primitive.parameters():
        parameter.requires_grad_(True)
    rec004g._restore_rng_state(source_state, core.device)
    for step in range(REC004O_SOURCE_STEP + 1, REC004O_CONTROL_REPLAY_STEP + 1):
        _one_recipe_update(core, primitive, optimizer, scheduler, step)
    actual = _training_state(
        primitive, optimizer, scheduler, core.device, REC004O_CONTROL_REPLAY_STEP
    )
    matches, first_difference = _state_equal(actual, expected)
    return {
        "status": "VERIFIED" if matches else "JOINT_CONTROL_REPLAY_MISMATCH",
        "expected_path": str(expected_path),
        "first_difference": first_difference,
        "actual_state_fingerprint": _state_fingerprint(actual),
        "expected_state_fingerprint": _state_fingerprint(expected),
        "primitive_hash": mb.canonical_state_hash(actual["primitive_state_dict"]),
        "expected_primitive_hash": mb.canonical_state_hash(expected["primitive_state_dict"]),
    }


class _FreezeGuard:
    def __init__(self, primitive: Any, optimizer: Any, partition: dict[str, Any]) -> None:
        self.params = dict(primitive.named_parameters())
        self.optimizer = optimizer
        self.partition = partition
        self.fused_names = [name for name in partition["groups"]["fused_qk"] if name in self.params]
        self.full_frozen_names = partition["frozen_whole_tensor_keys"]
        self.frozen_parameter_source = {
            name: self.params[name].detach().clone() for name in self.full_frozen_names
        }
        self.frozen_optimizer_source = {
            name: _clone_state(optimizer.state[self.params[name]])
            for name in self.full_frozen_names
        }
        self.v_parameter_source = {
            name: self.params[name].detach().clone()[_slice_rows(self.params[name])]
            for name in self.fused_names
        }
        self.v_optimizer_source = {
            name: _clone_state(optimizer.state[self.params[name]]) for name in self.fused_names
        }
        self.violations: list[dict[str, Any]] = []
        for name in self.full_frozen_names:
            self.params[name].requires_grad_(False)
        for name in self.fused_names + partition["groups"]["score_position_bias"]:
            self.params[name].requires_grad_(True)

    def zero_v_gradients(self) -> dict[str, float]:
        l1: dict[str, float] = {}
        for name in self.fused_names:
            parameter = self.params[name]
            if parameter.grad is None:
                raise SelectiveFreezeUnsafe(
                    "FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: missing fused QKV gradient"
                )
            v_rows = _slice_rows(parameter)
            l1[name] = float(parameter.grad[v_rows].abs().sum().item())
            parameter.grad[v_rows].zero_()
        return l1

    def restore_and_verify(self, step: int) -> None:
        with torch.no_grad():
            for name in self.fused_names:
                parameter = self.params[name]
                parameter[_slice_rows(parameter)].copy_(self.v_parameter_source[name])
                current = self.optimizer.state[parameter]
                source = self.v_optimizer_source[name]
                for key, value in current.items():
                    if (
                        isinstance(value, torch.Tensor)
                        and key in source
                        and isinstance(source[key], torch.Tensor)
                    ):
                        if value.shape == parameter.shape:
                            value[_slice_rows(parameter)].copy_(source[key][_slice_rows(parameter)])
        for name in self.full_frozen_names:
            parameter = self.params[name]
            if not torch.equal(parameter.detach(), self.frozen_parameter_source[name]):
                self.violations.append({"step": step, "kind": "frozen_parameter", "name": name})
            equal, detail = _state_equal(
                self.optimizer.state[parameter], self.frozen_optimizer_source[name]
            )
            if not equal:
                self.violations.append(
                    {"step": step, "kind": "frozen_optimizer_state", "name": name, "detail": detail}
                )
        for name in self.fused_names:
            parameter = self.params[name]
            v_rows = _slice_rows(parameter)
            if not torch.equal(parameter.detach()[v_rows], self.v_parameter_source[name]):
                self.violations.append({"step": step, "kind": "v_parameter_row", "name": name})
            source = self.v_optimizer_source[name]
            for key, source_value in source.items():
                current_value = self.optimizer.state[parameter].get(key)
                if isinstance(source_value, torch.Tensor) and source_value.shape == parameter.shape:
                    if not isinstance(current_value, torch.Tensor) or not torch.equal(
                        current_value[v_rows], source_value[v_rows]
                    ):
                        self.violations.append(
                            {
                                "step": step,
                                "kind": "v_optimizer_row",
                                "name": name,
                                "state_key": key,
                            }
                        )
        if self.violations:
            raise SelectiveFreezeUnsafe("FUSED_QKV_SELECTIVE_FREEZE_UNSAFE")

    def audit(self) -> dict[str, Any]:
        return {
            "frozen_whole_tensor_keys": self.full_frozen_names,
            "fused_v_tensor_keys": self.fused_names,
            "v_row_contract": (
                "parameter and row-shaped AdamW state tensors bit-identical; "
                "shared scalar step may advance with Q/K"
            ),
            "violations": self.violations,
            "passed": not self.violations,
        }


def _score_only_train(
    core: Any,
    source_state: dict[str, Any],
    partition: dict[str, Any],
    config: MirrorScoreOnlyContinuationPilotConfig,
    output_dir: Path,
) -> dict[str, Any]:
    primitive, optimizer, scheduler = _build_resumed_runtime(core, source_state)
    primitive.train()
    guard = _FreezeGuard(primitive, optimizer, partition)
    rec004g._restore_rng_state(source_state, core.device)
    run_dir = output_dir / REC004O_INIT_ID / "SCORE_ONLY_QK_POSITION_BIAS"
    state_dir = run_dir / "training_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    lr_trace: list[dict[str, Any]] = []
    gradient_l1: dict[str, float] = {name: 0.0 for name in partition["trainable_tensor_keys"]}
    v_gradient_l1 = {name: 0.0 for name in guard.fused_names}
    start = time.time()
    for step in range(REC004O_SOURCE_STEP + 1, REC004O_TARGET_STEP + 1):
        examples = ibc._generate_step_training_examples(
            config.seed,
            step,
            rec004g.REC004G_TARGET_OPERATION,
            vocab_size=rec004g.REC004G_VOCAB_SIZE,
            sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        lengths = [len(example.input_tokens) for example in examples]
        output_lengths = [
            get_operation(rec004g.REC004G_TARGET_OPERATION).output_length(length)
            for length in lengths
        ]
        labels = _labels_for_examples(examples, output_lengths, max(output_lengths), core.device)
        with torch.no_grad():
            batch = collate_content_only_batch(examples, core.tokens, device=core.device)
            h_content = core.model.encode(batch)[:, 1 : 1 + max(lengths), :]
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            primitive(h_content, lengths, output_lengths, None).reshape(
                -1, rec004g.REC004G_VOCAB_SIZE
            ),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        for name in gradient_l1:
            grad = dict(primitive.named_parameters())[name].grad
            if grad is not None:
                gradient_l1[name] += float(grad.abs().sum().item())
        v_before_zero = guard.zero_v_gradients()
        for name, value in v_before_zero.items():
            v_gradient_l1[name] += value
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), rec004g.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        guard.restore_and_verify(step)
        scheduler.step()
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise SelectiveFreezeUnsafe(
                f"FUSED_QKV_SELECTIVE_FREEZE_UNSAFE: non-finite loss step {step}"
            )
        lr_trace.append(
            {
                "step": step,
                "lr_used": lr_used,
                "lr_after_scheduler": float(scheduler.get_last_lr()[0]),
                "loss": loss_value,
            }
        )
        if step % config.checkpoint_interval == 0:
            torch.save(
                _training_state(primitive, optimizer, scheduler, core.device, step),
                state_dir / f"step{step}.pt",
            )
    primitive.eval()
    final_state = _training_state(primitive, optimizer, scheduler, core.device, REC004O_TARGET_STEP)
    return {
        "primitive": primitive,
        "final_state": final_state,
        "lr_trace": lr_trace,
        "gradient_l1": gradient_l1,
        "v_gradient_l1_before_mask": v_gradient_l1,
        "freeze_audit": guard.audit(),
        "wall_clock_seconds": time.time() - start,
        "new_optimizer_updates": REC004O_TARGET_STEP - REC004O_SOURCE_STEP,
    }


def _evaluate_new_datasets(
    core: Any, primitive: Any, datasets: dict[str, list[Example]]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    operation = get_operation(rec004g.REC004G_TARGET_OPERATION)
    for name, examples in datasets.items():
        by_length: dict[int, list[Example]] = {}
        for example in examples:
            by_length.setdefault(len(example.input_tokens), []).append(example)
        total = 0
        j0_correct = 0
        o1_correct = 0
        per_length: dict[str, Any] = {}
        for length, length_examples in sorted(by_length.items()):
            length_j0 = 0
            length_o1 = 0
            for start in range(0, len(length_examples), REC004O_CHUNK_SIZE):
                chunk = length_examples[start : start + REC004O_CHUNK_SIZE]
                lengths = [length] * len(chunk)
                output_lengths = [operation.output_length(length)] * len(chunk)
                batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
                with torch.no_grad():
                    h_content = core.model.encode(batch)[:, 1 : 1 + length, :]
                    j0 = rec004j._run_j0_decomposition(
                        primitive, h_content, lengths, output_lengths
                    )
                    o1 = oracle_probe.run_oracle_forward(
                        primitive, h_content, lengths, output_lengths
                    )
                j0_preds = resid_audit._predict_from_logits(j0["logits"], output_lengths)
                o1_preds = resid_audit._predict_from_logits(o1["logits"], output_lengths)
                for j0_prediction, o1_prediction, example in zip(
                    j0_preds, o1_preds, chunk, strict=True
                ):
                    target = tuple(example.target_tokens)
                    length_j0 += int(tuple(j0_prediction) == target)
                    length_o1 += int(tuple(o1_prediction) == target)
            count = len(length_examples)
            total += count
            j0_correct += length_j0
            o1_correct += length_o1
            per_length[str(length)] = {
                "n": count,
                "j0_sequence_exact_match": length_j0 / count,
                "oracle_sequence_exact_match": length_o1 / count,
            }
        results[name] = {
            "n": total,
            "j0_sequence_exact_match": j0_correct / total,
            "oracle_sequence_exact_match": o1_correct / total,
            "per_length": per_length,
        }
    return results


def _load_historical_primitive(core: Any, step: int, source: str) -> Any:
    if source == "source":
        state = rec004g.load_source_training_state(REC004O_INIT_ID)
    else:
        state_path = _rec004g_state_path(step)
        if not state_path.is_file():
            raise SelectiveFreezeUnsafe(f"missing historical joint control {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    primitive = mpbr._new_arm_primitive(core, REC004O_ARM)
    primitive.to(core.device)
    primitive.load_state_dict(
        {key: value.to(core.device) for key, value in state["primitive_state_dict"].items()},
        strict=True,
    )
    primitive.eval()
    return primitive


def _decision(
    score: dict[str, Any],
    source: dict[str, Any],
    historical_joint: dict[str, Any],
    frozen_exact: bool,
    score_updated: bool,
) -> dict[str, Any]:
    validation = score[REC004O_VALIDATION_SPLIT]
    confirmation = score[REC004O_CONFIRMATION_SPLIT]
    validation_len10 = validation["per_length"].get("10", {}).get("j0_sequence_exact_match")
    success = all(
        (
            validation["j0_sequence_exact_match"] >= REC004O_EM_FLOOR,
            validation_len10 is not None and validation_len10 >= REC004O_EM_FLOOR,
            confirmation["j0_sequence_exact_match"] >= REC004O_EM_FLOOR,
            validation["oracle_sequence_exact_match"] >= REC004O_EM_FLOOR,
            confirmation["oracle_sequence_exact_match"] >= REC004O_EM_FLOOR,
            frozen_exact,
        )
    )
    score_mean = (
        validation["j0_sequence_exact_match"] + confirmation["j0_sequence_exact_match"]
    ) / 2
    joint_mean = (
        historical_joint[REC004O_VALIDATION_SPLIT]["j0_sequence_exact_match"]
        + historical_joint[REC004O_CONFIRMATION_SPLIT]["j0_sequence_exact_match"]
    ) / 2
    source_mean = (
        source[REC004O_VALIDATION_SPLIT]["j0_sequence_exact_match"]
        + source[REC004O_CONFIRMATION_SPLIT]["j0_sequence_exact_match"]
    ) / 2
    if success:
        label = "SCORE_ONLY_CONTINUATION_SUPPORTED_ON_I03_PILOT"
    elif score_mean > joint_mean:
        label = "SCORE_ONLY_IMPROVEMENT_INSUFFICIENT"
    else:
        label = "SCORE_ONLY_CONTINUATION_NOT_SUPPORTED"
    tags = []
    if score_updated and score_mean <= source_mean:
        tags.append("FROZEN_CONTENT_PREP_LIMITS_SCORE_LEARNING")
    return {
        "label": label,
        "tags": tags,
        "floor": REC004O_EM_FLOOR,
        "all_acceptance_conditions": success,
        "frozen_downstream_exact": frozen_exact,
        "score_mean_j0": score_mean,
        "historical_joint_mean_j0": joint_mean,
        "source_mean_j0": source_mean,
        "endpoint_only": True,
        "candidate_selected": False,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
    }


def _render_report(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    return (
        "\n".join(
            [
                f"# {REC004O_TASK_ID} — I03 score-only continuation pilot",
                "",
                f"Result: `{decision.get('label', report.get('result_label', 'NOT_EXECUTED'))}`",
                "",
                "This single-init mechanism pilot does not select a candidate, "
                "publish a bundle, or run RG3.",
                f"Joint-control replay: `{report.get('joint_control_replay', {}).get('status')}`.",
                f"Frozen downstream exact: `{decision.get('frozen_downstream_exact')}`.",
                f"New optimizer updates: "
                f"`{report.get('cost_accounting', {}).get('new_optimizer_updates')}`.",
                "",
                "New datasets were locked before any model output and are "
                "development-exposed, not sealed/RG3 data.",
            ]
        )
        + "\n"
    )


def run_mirror_score_only_continuation_pilot_task(
    config: MirrorScoreOnlyContinuationPilotConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_score_only_continuation_pilot_task")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.seed != RECOVERY_PILOT_SEED:
        raise ValueError(f"{REC004O_TASK_ID} is fixed to I03's recovery seed {RECOVERY_PILOT_SEED}")
    if not REC004O_CONTRACT_FILE.is_file():
        return {
            "implementation_status": "STOPPED",
            "result_label": "AUTHORIZATION_ARTIFACT_MISSING",
        }
    start = time.time()
    before_cache = _snapshot_forbidden_cache_hashes(config.seed)
    source_paths = {
        "source_6000": rec004g._rec004d_training_state_path(REC004O_INIT_ID),
        "joint_control_6500": _rec004g_state_path(REC004O_CONTROL_REPLAY_STEP),
        "joint_control_12000": _rec004g_state_path(REC004O_TARGET_STEP),
    }
    source_hashes_before = {
        key: mb.raw_file_sha256(path) for key, path in source_paths.items() if path.is_file()
    }
    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    _write_json(
        output_dir / "config.yaml",
        {
            "seed": config.seed,
            "output_dir": str(output_dir),
            "checkpoint_interval": config.checkpoint_interval,
        },
    )
    _write_json(output_dir / "system.json", get_system_info(seed=config.seed))

    # Dataset manifests are deliberately the first task artifacts involving data;
    # no model forward is allowed before this pre-output lock completes.
    datasets, dataset_details = build_score_only_datasets(config.seed)
    _write_json(output_dir / "dataset_manifest.json", dataset_details)

    source_state = rec004g.load_source_training_state(REC004O_INIT_ID)
    trace_primitive, _trace_optimizer, _trace_scheduler = _build_resumed_runtime(core, source_state)
    partition = build_forward_graph_partition(trace_primitive)
    trace = _forward_backward_trace(core, trace_primitive, datasets[REC004O_VALIDATION_SPLIT])
    partition["real_forward_backward_trace"] = trace
    _write_json(output_dir / "forward_graph_partition.json", partition)

    try:
        control = run_joint_control_replay(core, source_state)
    except Exception as error:  # fail closed; this gate must never permit a fallback run
        control = {
            "status": "JOINT_CONTROL_REPLAY_MISMATCH",
            "reason": f"replay exception: {type(error).__name__}: {error}",
        }
    _write_json(output_dir / "joint_control_replay.json", control)
    if control["status"] != "VERIFIED":
        stop_report = {
            "implementation_status": "STOPPED",
            "result_label": "JOINT_CONTROL_REPLAY_MISMATCH",
            "joint_control_replay": control,
        }
        _write_json(output_dir / "summary.json", stop_report)
        (output_dir / "report.md").write_text(_render_report(stop_report), encoding="utf-8")
        return stop_report

    try:
        outcome = _score_only_train(core, source_state, partition, config, output_dir)
    except SelectiveFreezeUnsafe as error:
        stop_report = {
            "implementation_status": "STOPPED",
            "result_label": "FUSED_QKV_SELECTIVE_FREEZE_UNSAFE",
            "reason": str(error),
            "joint_control_replay": control,
        }
        _write_json(output_dir / "summary.json", stop_report)
        (output_dir / "report.md").write_text(_render_report(stop_report), encoding="utf-8")
        return stop_report
    (output_dir / "lr_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcome["lr_trace"]), encoding="utf-8"
    )
    _write_json(output_dir / "freeze_audit.json", outcome["freeze_audit"])

    source_primitive = _load_historical_primitive(core, REC004O_SOURCE_STEP, "source")
    joint_primitive = _load_historical_primitive(core, REC004O_TARGET_STEP, "joint")
    score_metrics = _evaluate_new_datasets(core, outcome["primitive"], datasets)
    source_metrics = _evaluate_new_datasets(core, source_primitive, datasets)
    joint_metrics = _evaluate_new_datasets(core, joint_primitive, datasets)
    _write_json(
        output_dir / "endpoint_metrics.json",
        {
            "score_only": score_metrics,
            "source_6000": source_metrics,
            "historical_joint_12000": joint_metrics,
        },
    )

    source_parameter_hash = mb.canonical_state_hash(source_state["primitive_state_dict"])
    final_parameter_hash = mb.canonical_state_hash(outcome["final_state"]["primitive_state_dict"])
    trainable_keys = partition["trainable_tensor_keys"]
    score_updated = any(
        not torch.equal(
            outcome["final_state"]["primitive_state_dict"][key],
            source_state["primitive_state_dict"][key],
        )
        for key in trainable_keys
    )
    frozen_exact = bool(outcome["freeze_audit"]["passed"])
    decision = _decision(score_metrics, source_metrics, joint_metrics, frozen_exact, score_updated)
    source_hashes_after = {
        key: mb.raw_file_sha256(path) for key, path in source_paths.items() if path.is_file()
    }
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    protected_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    after_cache = _snapshot_forbidden_cache_hashes(config.seed)
    report: dict[str, Any] = {
        "task_id": REC004O_TASK_ID,
        "implementation_status": "COMPLETED",
        "result_label": decision["label"],
        "decision": decision,
        "joint_control_replay": control,
        "forward_graph_partition": {
            key: value for key, value in partition.items() if key != "real_forward_backward_trace"
        },
        "source_primitive_hash": source_parameter_hash,
        "score_only_terminal_primitive_hash": final_parameter_hash,
        "score_path_updated": score_updated,
        "freeze_audit": outcome["freeze_audit"],
        "source_artifacts_unchanged": source_hashes_before == source_hashes_after,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "protected_operations_unchanged": protected_before == protected_after,
        "shared_cache_unchanged": before_cache == after_cache,
        "candidate_selected": False,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "cost_accounting": {
            "wall_clock_seconds": time.time() - start,
            "new_optimizer_updates": outcome["new_optimizer_updates"],
            "expected_new_optimizer_updates": REC004O_TARGET_STEP - REC004O_SOURCE_STEP,
            "score_gradient_l1": outcome["gradient_l1"],
            "v_gradient_l1_before_mask": outcome["v_gradient_l1_before_mask"],
        },
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
    return report
