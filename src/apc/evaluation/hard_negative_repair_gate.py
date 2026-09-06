"""B-C005G sealed hard-negative re-gate for the repaired B2 mechanism.

The module intentionally separates three seed roles: the historical B-C005
partition (never accepted here), the R1/R2 development partition used to fit
the already-selected recipe, and a new sealed evaluation partition used only
for the final matrix.  A serialized protocol hash is written before the first
cell runs; changing the protocol in the same run directory is rejected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.evaluation.adequacy_repair_benchmark import (
    AdequacyRepairConfig,
    CellAdequacyDiagnostic,
    evaluate_adequacy_policy_cell,
)
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_LEVELS,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    SEALED_GATE_SEEDS,
    CellRepairDiagnostic,
    RetrievalRepairConfig,
    evaluate_repair_cell,
    train_repaired_router_and_scorer,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

# These seeds were designated before this implementation's first B-C005G run.
# They are disjoint from both the original B-C005 sealed partition (0--4) and
# the R1/R2 development partition (10--14).
DEFAULT_REGATE_SEEDS: Final[tuple[int, ...]] = (20, 21, 22, 23, 24)


@dataclass(frozen=True)
class HardNegativeRegateConfig:
    """Frozen B-C005G protocol, including repaired R2 and sequential verifier."""

    sealed_seeds: tuple[int, ...] = DEFAULT_REGATE_SEEDS
    development_seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    support_examples: int = 128
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    adequacy_exact_match_threshold: float = 0.95
    confidence_level: float = 0.95
    initial_support: int = 32
    support_increment: int = 32
    max_support: int = 128
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None
    expected_protocol_hash: str | None = None

    def __post_init__(self) -> None:
        if len(self.sealed_seeds) < 1 or len(self.development_seeds) < 1:
            raise ValueError("sealed_seeds and development_seeds must be non-empty")
        if len(self.sealed_seeds) != len(self.development_seeds):
            raise ValueError("sealed_seeds and development_seeds must have equal length")
        if set(self.sealed_seeds) & set(self.development_seeds):
            raise ValueError("sealed and development partitions must be disjoint")
        if set(self.sealed_seeds) & SEALED_GATE_SEEDS:
            raise ValueError("original B-C005 sealed seeds may not be used for B-C005G")
        if set(self.development_seeds) & SEALED_GATE_SEEDS:
            raise ValueError("original B-C005 sealed seeds may not be used for training")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if self.support_examples < self.max_support:
            raise ValueError("support_examples must be >= max_support")
        if self.query_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["levels"] = [level.value for level in self.levels]
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result

    def protocol_payload(self) -> dict[str, Any]:
        """Return the immutable, output-location-independent seal payload."""
        result = self.to_dict()
        result.pop("output_dir")
        result.pop("expected_protocol_hash")
        return result

    def protocol_hash(self) -> str:
        """Return a stable SHA-256 hash for the sealed execution protocol."""
        payload = json.dumps(self.protocol_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegateCellDiagnostic:
    """One B-C005G matrix cell, retaining both call and family-level metrics."""

    sealed_seed: int
    development_seed: int
    bank_size: int
    level: str
    target_operation: str
    physical_primitive_top1: float
    physical_primitive_topk: float
    argument_accuracy: float
    primitive_call_top1: float
    primitive_call_topk: float
    candidate_rank: float
    score_margin: float
    closed_loop_exact_match: float
    false_functional_acceptance_rate: float
    false_plastic_rate: float
    mean_support_consumed: float
    selected_forward_calls: int
    unselected_forward_calls: int
    sparse_execution_passed: bool
    router_unchanged: bool
    primitive_functions_unchanged: bool
    evaluation_metadata_leakage: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _snapshot_parameters(router: Any, bank: Any) -> dict[str, torch.Tensor]:
    """Copy state whose mutation would invalidate a sealed evaluation cell."""
    result = {
        f"router/{name}": value.detach().clone()
        for name, value in router.state_dict().items()
    }
    for primitive_id in bank.ids():
        for name, value in bank.get(primitive_id).state_dict().items():
            result[f"primitive/{primitive_id}/{name}"] = value.detach().clone()
    return result


def _is_unchanged(snapshot: dict[str, torch.Tensor], router: Any, bank: Any) -> tuple[bool, bool]:
    """Check router and primitive tensors independently against a snapshot."""
    current = _snapshot_parameters(router, bank)
    router_ok = all(
        name in current and torch.equal(value, current[name])
        for name, value in snapshot.items()
        if name.startswith("router/")
    )
    primitive_ok = all(
        name in current and torch.equal(value, current[name])
        for name, value in snapshot.items()
        if name.startswith("primitive/")
    )
    return router_ok, primitive_ok


def _write_or_validate_seal(config: HardNegativeRegateConfig) -> str:
    """Persist the first protocol seal or reject a changed protocol thereafter."""
    protocol_hash = config.protocol_hash()
    if config.expected_protocol_hash is not None and config.expected_protocol_hash != protocol_hash:
        raise ValueError("expected_protocol_hash does not match the supplied B-C005G protocol")
    if config.output_dir is None:
        return protocol_hash
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seal_path = config.output_dir / "protocol.json"
    seal = {
        "task_id": "B-C005G",
        "protocol_hash": protocol_hash,
        "protocol": config.protocol_payload(),
    }
    if seal_path.exists():
        prior = json.loads(seal_path.read_text(encoding="utf-8"))
        if prior.get("protocol_hash") != protocol_hash or prior.get("protocol") != seal["protocol"]:
            raise ValueError(
                "B-C005G protocol is already sealed; post-seal configuration mutation rejected"
            )
    else:
        seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    return protocol_hash


def _mean(cells: Sequence[RegateCellDiagnostic], field: str) -> float:
    return statistics.fmean(float(getattr(cell, field)) for cell in cells)


def _full_matrix(config: HardNegativeRegateConfig) -> bool:
    return (
        len(config.sealed_seeds) >= 5
        and set(config.bank_sizes) == set(DEFAULT_BANK_SIZES)
        and set(config.levels) == set(DEFAULT_LEVELS)
        and set(config.target_operations) == set(DEFAULT_TARGET_OPERATIONS)
    )


def _frozen_r2_config(
    config: HardNegativeRegateConfig, development_seed: int
) -> RetrievalRepairConfig:
    return RetrievalRepairConfig(
        seeds=(development_seed,),
        bank_sizes=config.bank_sizes,
        levels=config.levels,
        target_operations=config.target_operations,
        support_examples=config.initial_support,
        query_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        arg_lambda=config.arg_lambda,
        adequacy_exact_match_threshold=config.adequacy_exact_match_threshold,
        deterministic_algorithms=config.deterministic_algorithms,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _adequacy_config(config: HardNegativeRegateConfig) -> AdequacyRepairConfig:
    return AdequacyRepairConfig(
        seeds=config.development_seeds,
        bank_sizes=config.bank_sizes,
        levels=config.levels,
        target_operations=config.target_operations,
        policies=("sequential",),
        support_examples=config.support_examples,
        query_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        arg_lambda=config.arg_lambda,
        adequacy_exact_match_threshold=config.adequacy_exact_match_threshold,
        confidence_level=config.confidence_level,
        initial_support=config.initial_support,
        support_increment=config.support_increment,
        max_support=config.max_support,
        deterministic_algorithms=config.deterministic_algorithms,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _criteria(
    config: HardNegativeRegateConfig, cells: Sequence[RegateCellDiagnostic]
) -> dict[str, dict[str, Any]]:
    n128 = [cell for cell in cells if cell.bank_size == 128]
    level_thresholds = {
        HardNegativeLevel.L0_ORTHOGONAL.value: 0.98,
        HardNegativeLevel.L1_RANDOM_SCORE_SPACE.value: 0.98,
        HardNegativeLevel.L2_NEAR_NEIGHBOR.value: 0.98,
        HardNegativeLevel.L3_SEMANTICALLY_RELATED.value: 0.95,
        HardNegativeLevel.L4_CONFUSABLE_FAMILY.value: 0.90,
    }
    result: dict[str, dict[str, Any]] = {
        "required_matrix": {
            "target": "5 seeds x N={16,32,64,128} x L0-L4 x 4 targets",
            "measured": len(cells),
            "passed": _full_matrix(config) and len(cells) == 400,
        },
    }
    for level, threshold in level_thresholds.items():
        selected = [cell for cell in n128 if cell.level == level]
        top1 = _mean(selected, "primitive_call_top1") if selected else 0.0
        topk = _mean(selected, "primitive_call_topk") if selected else 0.0
        result[f"{level}_full_primitive_call_top1"] = {
            "target": f">= {threshold:.2f}", "measured": top1, "passed": top1 >= threshold
        }
        result[f"{level}_top_k_inclusion"] = {
            "target": ">= 0.99", "measured": topk, "passed": topk >= 0.99
        }
    l4 = [cell for cell in n128 if cell.level == HardNegativeLevel.L4_CONFUSABLE_FAMILY.value]
    l4_phys = _mean(l4, "physical_primitive_top1") if l4 else 0.0
    l4_arg = _mean(l4, "argument_accuracy") if l4 else 0.0
    result["l4_physical_primitive_family_top1"] = {
        "target": ">= 0.98", "measured": l4_phys, "passed": l4_phys >= 0.98
    }
    result["l4_argument_accuracy"] = {
        "target": ">= 0.95", "measured": l4_arg, "passed": l4_arg >= 0.95
    }
    result["wrong_functional_acceptance"] = {
        "target": "<= 0.01",
        "measured": max((cell.false_functional_acceptance_rate for cell in cells), default=1.0),
        "passed": all(cell.false_functional_acceptance_rate <= 0.01 for cell in cells),
    }
    result["closed_loop_exact_match"] = {
        "target": ">= 0.95",
        "measured": _mean(cells, "closed_loop_exact_match") if cells else 0.0,
        "passed": bool(cells) and _mean(cells, "closed_loop_exact_match") >= 0.95,
    }
    result["false_plastic"] = {
        "target": "<= 0.02",
        "measured": max((cell.false_plastic_rate for cell in cells), default=1.0),
        "passed": all(cell.false_plastic_rate <= 0.02 for cell in cells),
    }
    result["zero_unselected_primitive_calls"] = {
        "target": "== 0",
        "measured": sum(cell.unselected_forward_calls for cell in cells),
        "passed": all(cell.unselected_forward_calls == 0 for cell in cells),
    }
    unchanged = all(
        cell.router_unchanged and cell.primitive_functions_unchanged for cell in cells
    )
    result["router_and_primitives_unchanged"] = {
        "target": "True",
        "measured": unchanged,
        "passed": unchanged,
    }
    result["evaluation_metadata_leakage"] = {
        "target": "== 0",
        "measured": sum(cell.evaluation_metadata_leakage for cell in cells),
        "passed": all(cell.evaluation_metadata_leakage == 0 for cell in cells),
    }
    return result


def run_hard_negative_regate(config: HardNegativeRegateConfig) -> dict[str, Any]:
    """Run B-C005G once on its sealed partition with the frozen R2 mechanism."""
    protocol_hash = _write_or_validate_seal(config)
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    adequacy_config = _adequacy_config(config)
    cells: list[RegateCellDiagnostic] = []

    for sealed_seed, development_seed in zip(
        config.sealed_seeds, config.development_seeds, strict=True
    ):
        set_seed(sealed_seed, deterministic_algorithms=config.deterministic_algorithms)
        # The Phase A.2 base checkpoint is a pre-existing frozen artifact.  No
        # hard-negative example from either old or new sealed partition enters R2 training.
        model_seed = sealed_seed % 5
        base_config = HardNegativeBenchmarkConfig(
            seeds=(model_seed,),
            bank_sizes=config.bank_sizes,
            levels=config.levels,
            target_operations=config.target_operations,
            num_eval_examples=config.query_examples,
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            top_k=config.top_k,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)

        dev_train: dict[str, list[Any]] = {}
        for operation, primitive_id in op_to_id.items():
            dev_train[operation] = list(
                generate_benchmark_examples(
                    development_seed * 50_000 + primitive_id * 100 + 7,
                    config.router_train_examples,
                    operation=operation,
                    split="train",
                )
            )

        for bank_size in config.bank_sizes:
            bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
                core, base_bank, base_router, op_to_id, bank_size, seed=sealed_seed
            )
            operation_by_id = {
                primitive_id: operation for operation, primitive_id in op_to_id.items()
            }
            operation_by_id.update({primitive_id: "SWAP_ENDS" for primitive_id in distractor_ids})
            repaired_router, argument_scorer = train_repaired_router_and_scorer(
                core,
                router,
                candidate_ids,
                operation_by_id,
                dev_train,
                config=_frozen_r2_config(config, development_seed),
                condition="R2",
                device=core.device,
            )
            repaired_router.eval()
            for parameter in repaired_router.parameters():
                parameter.requires_grad_(False)
            if argument_scorer is not None:
                argument_scorer.eval()
                for parameter in argument_scorer.parameters():
                    parameter.requires_grad_(False)
            snapshot = _snapshot_parameters(repaired_router, bank)

            for operation in config.target_operations:
                target_id = op_to_id[operation]
                support = generate_benchmark_examples(
                    sealed_seed * 1_000_000 + bank_size * 100 + target_id,
                    config.support_examples,
                    operation=operation,
                    split="test",
                )
                query = generate_benchmark_examples(
                    sealed_seed * 1_000_000 + bank_size * 100 + target_id + 1,
                    config.query_examples,
                    operation=operation,
                    split="test",
                )
                for level in config.levels:
                    retrieval: CellRepairDiagnostic = evaluate_repair_cell(
                        core=core,
                        bank=bank,
                        router=repaired_router,
                        argument_scorer=argument_scorer,
                        candidate_ids=candidate_ids,
                        operation_by_id=operation_by_id,
                        target_operation=operation,
                        target_id=target_id,
                        level=level,
                        support_examples=support[: config.initial_support],
                        query_examples=query,
                        seed=sealed_seed,
                        top_k=config.top_k,
                        adequacy_threshold=config.adequacy_exact_match_threshold,
                        arg_lambda=config.arg_lambda,
                        condition="R2_frozen",
                        bank_size=bank_size,
                    )
                    safety: CellAdequacyDiagnostic = evaluate_adequacy_policy_cell(
                        core=core,
                        bank=bank,
                        router=repaired_router,
                        argument_scorer=argument_scorer,
                        candidate_ids=candidate_ids,
                        operation_by_id=operation_by_id,
                        target_operation=operation,
                        target_id=target_id,
                        level=level,
                        support_examples=support,
                        query_examples=query,
                        seed=sealed_seed,
                        policy="sequential",
                        config=adequacy_config,
                    )
                    router_ok, primitives_ok = _is_unchanged(snapshot, repaired_router, bank)
                    metadata_leakage = int(
                        not (retrieval.leak_audit_passed and safety.leak_audit_passed)
                    )
                    cells.append(
                        RegateCellDiagnostic(
                            sealed_seed=sealed_seed,
                            development_seed=development_seed,
                            bank_size=bank_size,
                            level=level.value,
                            target_operation=operation,
                            physical_primitive_top1=retrieval.physical_primitive_top1,
                            physical_primitive_topk=retrieval.physical_primitive_topk,
                            argument_accuracy=retrieval.argument_accuracy,
                            primitive_call_top1=retrieval.primitive_call_top1,
                            primitive_call_topk=retrieval.primitive_call_topk,
                            candidate_rank=retrieval.candidate_rank,
                            score_margin=retrieval.score_margin,
                            closed_loop_exact_match=safety.closed_loop_exact_match,
                            false_functional_acceptance_rate=safety.false_functional_acceptance_rate,
                            false_plastic_rate=safety.false_plastic_rate,
                            mean_support_consumed=safety.mean_support_consumed,
                            selected_forward_calls=safety.selected_forward_calls,
                            unselected_forward_calls=safety.unselected_forward_calls,
                            sparse_execution_passed=safety.sparse_execution_passed,
                            router_unchanged=router_ok,
                            primitive_functions_unchanged=primitives_ok,
                            evaluation_metadata_leakage=metadata_leakage,
                        )
                    )

    curves: dict[str, dict[str, float]] = {}
    for bank_size in config.bank_sizes:
        for level in config.levels:
            selected = [
                cell
                for cell in cells
                if cell.bank_size == bank_size and cell.level == level.value
            ]
            curves[f"N{bank_size}/{level.value}"] = {
                "full_primitive_call_top1": _mean(selected, "primitive_call_top1"),
                "top_k_inclusion": _mean(selected, "primitive_call_topk"),
                "physical_primitive_top1": _mean(selected, "physical_primitive_top1"),
                "argument_accuracy": _mean(selected, "argument_accuracy"),
                "positive_negative_margin": _mean(selected, "score_margin"),
                "closed_loop_exact_match": _mean(selected, "closed_loop_exact_match"),
            }
    criteria = _criteria(config, cells)
    all_passed = all(item["passed"] for item in criteria.values())
    report = {
        "task_id": "B-C005G",
        "protocol_hash": protocol_hash,
        "config": config.to_dict(),
        "development_partition": list(config.development_seeds),
        "sealed_evaluation_partition": list(config.sealed_seeds),
        "matrix": [cell.to_dict() for cell in cells],
        "scaling_curves": curves,
        "criteria": criteria,
        "all_criteria_passed": all_passed,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if config.output_dir is not None:
        import yaml

        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (config.output_dir / "system.json").write_text(
            json.dumps(get_system_info(), indent=2), encoding="utf-8"
        )
        (config.output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(cell.to_dict()) + "\n" for cell in cells), encoding="utf-8"
        )
        (config.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        lines = [
            "# B-C005G — New Sealed Hard-Negative B2 Re-Gate",
            "",
            f"**Status:** {'PASS' if all_passed else 'FAIL'}",
            f"**Protocol hash:** `{protocol_hash}`",
            f"**Development partition:** {list(config.development_seeds)}",
            f"**New sealed evaluation partition:** {list(config.sealed_seeds)}",
            "",
            "## Acceptance criteria",
            "",
            "| Criterion | Target | Measured | Result |",
            "|---|---|---:|:---:|",
        ]
        for name, value in criteria.items():
            measured = value["measured"]
            measured_text = f"{measured:.4f}" if isinstance(measured, float) else str(measured)
            result = "PASS" if value["passed"] else "FAIL"
            lines.append(f"| {name} | {value['target']} | {measured_text} | {result} |")
        lines.extend(
            [
                "",
                "## Scaling curves",
                "",
                "| Cell | Call top-1 | Top-k | Family top-1 | Arg acc. | Margin | Closed-loop EM |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, value in curves.items():
            lines.append(
                f"| {name} | {value['full_primitive_call_top1']:.4f} | "
                f"{value['top_k_inclusion']:.4f} | {value['physical_primitive_top1']:.4f} | "
                f"{value['argument_accuracy']:.4f} | {value['positive_negative_margin']:.4f} | "
                f"{value['closed_loop_exact_match']:.4f} |"
            )
        (config.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
