"""Explicit one-task dispatcher for the Phase B B2 post-D2 repair series.

Per `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md` Section 0: exactly one
named task runs per invocation, and there is no `--all` or implicit
next-task execution. Only `B-C005R3-001` through `B-C005R3-010` are
implemented so far; every other task ID is rejected until it is explicitly
implemented and wired in here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.bind_argument_scorer_repair import (
    BindArgumentScorerRepairConfig,
    run_bind_argument_scorer_repair,
)
from apc.evaluation.count_bind_key_scoring_repair import (
    CountBindKeyScoringConfig,
    run_count_bind_key_scoring_repair,
)
from apc.evaluation.functional_metrics_v2 import (
    FunctionalMetricsV2Config,
    run_functional_metrics_v2_protocol,
)
from apc.evaluation.paired_baseline_repair import (
    PairedBaselineConfig,
    run_paired_baseline_repair,
)
from apc.evaluation.paired_integration_regression import (
    PairedIntegrationRegressionConfig,
    run_paired_integration_regression,
)
from apc.evaluation.post_d2_repair_benchmark import (
    PostD2ReproducibilityConfig,
    run_post_d2_reproducibility_task,
)
from apc.evaluation.relation_split_protocol import (
    RelationSplitProtocolConfig,
    run_relation_split_protocol,
)
from apc.evaluation.safe_bounded_verification import (
    SafeBoundedVerificationConfig,
    run_safe_bounded_verification,
)
from apc.evaluation.select_argument_encoding_repair import (
    SelectArgumentEncodingConfig,
    run_select_argument_encoding_repair,
)
from apc.evaluation.shift_functional_generalization_repair import (
    ShiftFunctionalGeneralizationRepairConfig,
    run_shift_functional_generalization_repair,
)
from apc.meta.phase_b_protocol import HardNegativeLevel

_IMPLEMENTED_TASKS = (
    "B-C005R3-001",
    "B-C005R3-002",
    "B-C005R3-003",
    "B-C005R3-004",
    "B-C005R3-005",
    "B-C005R3-006",
    "B-C005R3-007",
    "B-C005R3-008",
    "B-C005R3-009",
    "B-C005R3-010",
)


def _load_r3_001_config(config_path: Path) -> PostD2ReproducibilityConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = PostD2ReproducibilityConfig()
    variants_raw = raw.get(
        "pythonhashseed_variants",
        ["<unset>" if v is None else v for v in defaults.pythonhashseed_variants],
    )
    variants = tuple(None if v in (None, "<unset>") else str(v) for v in variants_raw)
    return PostD2ReproducibilityConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        operations=tuple(raw.get("operations", defaults.operations)),
        splits=tuple(raw.get("splits", defaults.splits)),
        n_examples=raw.get("n_examples", defaults.n_examples),
        pythonhashseed_variants=variants,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_002_config(config_path: Path) -> RelationSplitProtocolConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = RelationSplitProtocolConfig()
    return RelationSplitProtocolConfig(
        development_representativeness_bank_size=raw.get(
            "development_representativeness_bank_size",
            defaults.development_representativeness_bank_size,
        ),
        development_representativeness_query_examples=raw.get(
            "development_representativeness_query_examples",
            defaults.development_representativeness_query_examples,
        ),
        run_empirical_exposure_probe=raw.get(
            "run_empirical_exposure_probe", defaults.run_empirical_exposure_probe
        ),
        empirical_probe_seed=raw.get("empirical_probe_seed", defaults.empirical_probe_seed),
        device=raw.get("device", defaults.device),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_003_config(config_path: Path) -> FunctionalMetricsV2Config:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = FunctionalMetricsV2Config()
    return FunctionalMetricsV2Config(
        tau=raw.get("tau", defaults.tau),
        max_candidates=raw.get("max_candidates", defaults.max_candidates),
        looks=tuple(raw.get("looks", defaults.looks)),
        alpha_accept_episode=raw.get("alpha_accept_episode", defaults.alpha_accept_episode),
        alpha_reject_episode=raw.get("alpha_reject_episode", defaults.alpha_reject_episode),
        alpha_ref_episode=raw.get("alpha_ref_episode", defaults.alpha_ref_episode),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_005_config(config_path: Path) -> SafeBoundedVerificationConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = SafeBoundedVerificationConfig()
    return SafeBoundedVerificationConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        target_operations=tuple(raw.get("target_operations", defaults.target_operations)),
        verification_examples=raw.get("verification_examples", defaults.verification_examples),
        reference_examples=raw.get("reference_examples", defaults.reference_examples),
        tau=raw.get("tau", defaults.tau),
        looks=tuple(raw.get("looks", defaults.looks)),
        alpha_accept_episode=raw.get("alpha_accept_episode", defaults.alpha_accept_episode),
        alpha_reject_episode=raw.get("alpha_reject_episode", defaults.alpha_reject_episode),
        alpha_ref_episode=raw.get("alpha_ref_episode", defaults.alpha_ref_episode),
        legacy_max_support=raw.get("legacy_max_support", defaults.legacy_max_support),
        fixed_ns=tuple(raw.get("fixed_ns", defaults.fixed_ns)),
        bernoulli_episodes_per_p=raw.get(
            "bernoulli_episodes_per_p", defaults.bernoulli_episodes_per_p
        ),
        availability_min_accept_rate=raw.get(
            "availability_min_accept_rate", defaults.availability_min_accept_rate
        ),
        composition_check_examples=raw.get(
            "composition_check_examples", defaults.composition_check_examples
        ),
        composition_max_depth=raw.get("composition_max_depth", defaults.composition_max_depth),
        composition_beam_width=raw.get("composition_beam_width", defaults.composition_beam_width),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_004_config(config_path: Path) -> PairedBaselineConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = PairedBaselineConfig()
    return PairedBaselineConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        retrieval_target_operations=tuple(
            raw.get("retrieval_target_operations", defaults.retrieval_target_operations)
        ),
        support_examples=raw.get("support_examples", defaults.support_examples),
        query_examples=raw.get("query_examples", defaults.query_examples),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        top_k=raw.get("top_k", defaults.top_k),
        ranking_margin=raw.get("ranking_margin", defaults.ranking_margin),
        ranking_beta=raw.get("ranking_beta", defaults.ranking_beta),
        arg_lambda=raw.get("arg_lambda", defaults.arg_lambda),
        adequacy_exact_match_threshold=raw.get(
            "adequacy_exact_match_threshold", defaults.adequacy_exact_match_threshold
        ),
        shift_operation=raw.get("shift_operation", defaults.shift_operation),
        shift_reference_examples=raw.get(
            "shift_reference_examples", defaults.shift_reference_examples
        ),
        select_bind_target_operation=raw.get(
            "select_bind_target_operation", defaults.select_bind_target_operation
        ),
        probe_train_examples=raw.get("probe_train_examples", defaults.probe_train_examples),
        probe_eval_examples=raw.get("probe_eval_examples", defaults.probe_eval_examples),
        probe_steps=raw.get("probe_steps", defaults.probe_steps),
        probe_lr=raw.get("probe_lr", defaults.probe_lr),
        shuffled_chance_tolerance=raw.get(
            "shuffled_chance_tolerance", defaults.shuffled_chance_tolerance
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_006_config(config_path: Path) -> CountBindKeyScoringConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = CountBindKeyScoringConfig()
    return CountBindKeyScoringConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        support_examples=raw.get("support_examples", defaults.support_examples),
        query_examples=raw.get("query_examples", defaults.query_examples),
        selection_query_examples=raw.get(
            "selection_query_examples", defaults.selection_query_examples
        ),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        ranking_margin=raw.get("ranking_margin", defaults.ranking_margin),
        ranking_beta=raw.get("ranking_beta", defaults.ranking_beta),
        top_k=raw.get("top_k", defaults.top_k),
        adequacy_exact_match_threshold=raw.get(
            "adequacy_exact_match_threshold", defaults.adequacy_exact_match_threshold
        ),
        gate_top1_threshold=raw.get("gate_top1_threshold", defaults.gate_top1_threshold),
        gate_topk_threshold=raw.get("gate_topk_threshold", defaults.gate_topk_threshold),
        gate_legacy_regression_pp_max=raw.get(
            "gate_legacy_regression_pp_max", defaults.gate_legacy_regression_pp_max
        ),
        variants=tuple(raw.get("variants", defaults.variants)),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_007_config(config_path: Path) -> SelectArgumentEncodingConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = SelectArgumentEncodingConfig()
    return SelectArgumentEncodingConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        support_examples=raw.get("support_examples", defaults.support_examples),
        query_examples=raw.get("query_examples", defaults.query_examples),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        ranking_margin=raw.get("ranking_margin", defaults.ranking_margin),
        ranking_beta=raw.get("ranking_beta", defaults.ranking_beta),
        arg_lambda=raw.get("arg_lambda", defaults.arg_lambda),
        top_k=raw.get("top_k", defaults.top_k),
        adequacy_exact_match_threshold=raw.get(
            "adequacy_exact_match_threshold", defaults.adequacy_exact_match_threshold
        ),
        gate_full_argument_accuracy_threshold=raw.get(
            "gate_full_argument_accuracy_threshold", defaults.gate_full_argument_accuracy_threshold
        ),
        gate_full_call_top1_threshold=raw.get(
            "gate_full_call_top1_threshold", defaults.gate_full_call_top1_threshold
        ),
        gate_other_op_regression_pp_max=raw.get(
            "gate_other_op_regression_pp_max", defaults.gate_other_op_regression_pp_max
        ),
        cardinality_stress_lengths=tuple(
            raw.get("cardinality_stress_lengths", defaults.cardinality_stress_lengths)
        ),
        cardinality_stress_examples=raw.get(
            "cardinality_stress_examples", defaults.cardinality_stress_examples
        ),
        rare_value_probe_examples=raw.get(
            "rare_value_probe_examples", defaults.rare_value_probe_examples
        ),
        rare_value_query_examples=raw.get(
            "rare_value_query_examples", defaults.rare_value_query_examples
        ),
        rare_value_percentile=raw.get("rare_value_percentile", defaults.rare_value_percentile),
        rare_value_min_subset_size=raw.get(
            "rare_value_min_subset_size", defaults.rare_value_min_subset_size
        ),
        dilution_cardinalities=tuple(
            raw.get("dilution_cardinalities", defaults.dilution_cardinalities)
        ),
        dilution_arg_vocab_size=raw.get(
            "dilution_arg_vocab_size", defaults.dilution_arg_vocab_size
        ),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_008_config(config_path: Path) -> BindArgumentScorerRepairConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = BindArgumentScorerRepairConfig()
    return BindArgumentScorerRepairConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        support_examples=raw.get("support_examples", defaults.support_examples),
        query_examples=raw.get("query_examples", defaults.query_examples),
        selection_support_examples=raw.get(
            "selection_support_examples", defaults.selection_support_examples
        ),
        selection_query_examples=raw.get(
            "selection_query_examples", defaults.selection_query_examples
        ),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        ranking_margin=raw.get("ranking_margin", defaults.ranking_margin),
        ranking_beta=raw.get("ranking_beta", defaults.ranking_beta),
        arg_lambda=raw.get("arg_lambda", defaults.arg_lambda),
        top_k=raw.get("top_k", defaults.top_k),
        adequacy_exact_match_threshold=raw.get(
            "adequacy_exact_match_threshold", defaults.adequacy_exact_match_threshold
        ),
        repair_pool_examples=raw.get("repair_pool_examples", defaults.repair_pool_examples),
        repair_training_budget=raw.get(
            "repair_training_budget", defaults.repair_training_budget
        ),
        repair_steps=raw.get("repair_steps", defaults.repair_steps),
        repair_lr=raw.get("repair_lr", defaults.repair_lr),
        gate_argument_accuracy_threshold=raw.get(
            "gate_argument_accuracy_threshold", defaults.gate_argument_accuracy_threshold
        ),
        gate_full_call_top1_threshold=raw.get(
            "gate_full_call_top1_threshold", defaults.gate_full_call_top1_threshold
        ),
        gate_family_top1_threshold=raw.get(
            "gate_family_top1_threshold", defaults.gate_family_top1_threshold
        ),
        gate_topk_threshold=raw.get("gate_topk_threshold", defaults.gate_topk_threshold),
        gate_other_op_regression_pp_max=raw.get(
            "gate_other_op_regression_pp_max", defaults.gate_other_op_regression_pp_max
        ),
        variants=tuple(raw.get("variants", defaults.variants)),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_009_config(config_path: Path) -> ShiftFunctionalGeneralizationRepairConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = ShiftFunctionalGeneralizationRepairConfig()
    return ShiftFunctionalGeneralizationRepairConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        core_lr=raw.get("core_lr", defaults.core_lr),
        core_weight_decay=raw.get("core_weight_decay", defaults.core_weight_decay),
        operator_train_steps=raw.get("operator_train_steps", defaults.operator_train_steps),
        operator_lr=raw.get("operator_lr", defaults.operator_lr),
        operator_weight_decay=raw.get("operator_weight_decay", defaults.operator_weight_decay),
        operator_grad_clip=raw.get("operator_grad_clip", defaults.operator_grad_clip),
        operator_batch_size=raw.get("operator_batch_size", defaults.operator_batch_size),
        diagnostic_examples=raw.get("diagnostic_examples", defaults.diagnostic_examples),
        selection_query_examples=raw.get(
            "selection_query_examples", defaults.selection_query_examples
        ),
        gate_query_examples=raw.get("gate_query_examples", defaults.gate_query_examples),
        other_op_query_examples=raw.get(
            "other_op_query_examples", defaults.other_op_query_examples
        ),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        gate_mean_query_em_threshold=raw.get(
            "gate_mean_query_em_threshold", defaults.gate_mean_query_em_threshold
        ),
        gate_ref_adequacy_tau=raw.get("gate_ref_adequacy_tau", defaults.gate_ref_adequacy_tau),
        gate_other_task_regression_pp_max=raw.get(
            "gate_other_task_regression_pp_max", defaults.gate_other_task_regression_pp_max
        ),
        variants=tuple(raw.get("variants", defaults.variants)),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        bank_checkpoint_dir=Path(raw.get("bank_checkpoint_dir", defaults.bank_checkpoint_dir)),
        shared_encoder_cache_dir=Path(
            raw.get("shared_encoder_cache_dir", defaults.shared_encoder_cache_dir)
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_010_config(config_path: Path) -> PairedIntegrationRegressionConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = PairedIntegrationRegressionConfig()
    nominal_levels_raw = raw.get(
        "nominal_levels", [level.value for level in defaults.nominal_levels]
    )
    nominal_levels = tuple(HardNegativeLevel.from_str(value) for value in nominal_levels_raw)
    return PairedIntegrationRegressionConfig(
        development_seeds=tuple(raw.get("development_seeds", defaults.development_seeds)),
        bank_size=raw.get("bank_size", defaults.bank_size),
        nominal_target_operations=tuple(
            raw.get("nominal_target_operations", defaults.nominal_target_operations)
        ),
        nominal_levels=nominal_levels,
        support_examples=raw.get("support_examples", defaults.support_examples),
        query_examples=raw.get("query_examples", defaults.query_examples),
        shift_query_examples=raw.get("shift_query_examples", defaults.shift_query_examples),
        safety_verification_examples=raw.get(
            "safety_verification_examples", defaults.safety_verification_examples
        ),
        safety_reference_examples=raw.get(
            "safety_reference_examples", defaults.safety_reference_examples
        ),
        legacy_deterministic_eval_examples=raw.get(
            "legacy_deterministic_eval_examples", defaults.legacy_deterministic_eval_examples
        ),
        router_train_examples=raw.get("router_train_examples", defaults.router_train_examples),
        router_steps=raw.get("router_steps", defaults.router_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        top_k=raw.get("top_k", defaults.top_k),
        ranking_margin=raw.get("ranking_margin", defaults.ranking_margin),
        ranking_beta=raw.get("ranking_beta", defaults.ranking_beta),
        arg_lambda=raw.get("arg_lambda", defaults.arg_lambda),
        count_bind_variant=raw.get("count_bind_variant", defaults.count_bind_variant),
        count_bind_router_steps=raw.get(
            "count_bind_router_steps", defaults.count_bind_router_steps
        ),
        count_bind_router_lr=raw.get("count_bind_router_lr", defaults.count_bind_router_lr),
        count_bind_ranking_margin=raw.get(
            "count_bind_ranking_margin", defaults.count_bind_ranking_margin
        ),
        count_bind_ranking_beta=raw.get(
            "count_bind_ranking_beta", defaults.count_bind_ranking_beta
        ),
        bind_head_variant=raw.get("bind_head_variant", defaults.bind_head_variant),
        bind_head_pool_examples=raw.get(
            "bind_head_pool_examples", defaults.bind_head_pool_examples
        ),
        bind_head_training_budget=raw.get(
            "bind_head_training_budget", defaults.bind_head_training_budget
        ),
        bind_head_steps=raw.get("bind_head_steps", defaults.bind_head_steps),
        bind_head_lr=raw.get("bind_head_lr", defaults.bind_head_lr),
        tau=raw.get("tau", defaults.tau),
        looks=tuple(raw.get("looks", defaults.looks)),
        alpha_accept_episode=raw.get("alpha_accept_episode", defaults.alpha_accept_episode),
        alpha_reject_episode=raw.get("alpha_reject_episode", defaults.alpha_reject_episode),
        alpha_ref_episode=raw.get("alpha_ref_episode", defaults.alpha_ref_episode),
        legacy_max_support=raw.get("legacy_max_support", defaults.legacy_max_support),
        nominal_l0l2_top1_threshold=raw.get(
            "nominal_l0l2_top1_threshold", defaults.nominal_l0l2_top1_threshold
        ),
        nominal_l3_top1_threshold=raw.get(
            "nominal_l3_top1_threshold", defaults.nominal_l3_top1_threshold
        ),
        nominal_l4_top1_threshold=raw.get(
            "nominal_l4_top1_threshold", defaults.nominal_l4_top1_threshold
        ),
        nominal_top5_threshold=raw.get("nominal_top5_threshold", defaults.nominal_top5_threshold),
        nominal_l4_family_top1_threshold=raw.get(
            "nominal_l4_family_top1_threshold", defaults.nominal_l4_family_top1_threshold
        ),
        nominal_argument_accuracy_threshold=raw.get(
            "nominal_argument_accuracy_threshold", defaults.nominal_argument_accuracy_threshold
        ),
        shift_mean_query_em_threshold=raw.get(
            "shift_mean_query_em_threshold", defaults.shift_mean_query_em_threshold
        ),
        safety_rate_budget=raw.get("safety_rate_budget", defaults.safety_rate_budget),
        legacy_mean_regression_pp_max=raw.get(
            "legacy_mean_regression_pp_max", defaults.legacy_mean_regression_pp_max
        ),
        legacy_worst_regression_pp_max=raw.get(
            "legacy_worst_regression_pp_max", defaults.legacy_worst_regression_pp_max
        ),
        device=raw.get("device", defaults.device),
        deterministic_algorithms=raw.get(
            "deterministic_algorithms", defaults.deterministic_algorithms
        ),
        bank_checkpoint_dir=Path(raw.get("bank_checkpoint_dir", defaults.bank_checkpoint_dir)),
        r3009_bank_transaction_log=Path(
            raw.get("r3009_bank_transaction_log", defaults.r3009_bank_transaction_log)
        ),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B2 post-D2 repair -- explicit single-task dispatch"
    )
    parser.add_argument(
        "--task",
        required=True,
        help=f"Exact task ID to run. Implemented: {', '.join(_IMPLEMENTED_TASKS)}. "
        "No --all and no implicit next-task execution.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to this task's own configs/phase_b_b2_post_d2_<task>.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.task not in _IMPLEMENTED_TASKS:
        print(
            f"ERROR: task {args.task!r} is not implemented by this dispatcher. "
            f"Only {_IMPLEMENTED_TASKS} run today; every later task in "
            "docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md requires its own "
            "explicit user instruction and its own implementation before it "
            "can be dispatched here.",
            file=sys.stderr,
        )
        return 2

    if args.task == "B-C005R3-001":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_reproducibility.yaml")
        config = _load_r3_001_config(config_path)
        if args.output_dir is not None:
            config = PostD2ReproducibilityConfig(
                seeds=config.seeds,
                operations=config.operations,
                splits=config.splits,
                n_examples=config.n_examples,
                pythonhashseed_variants=config.pythonhashseed_variants,
                output_dir=args.output_dir,
            )
        report = run_post_d2_reproducibility_task(config)
        next_blocked = "B-C005R3-002 onward"
    elif args.task == "B-C005R3-002":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_relation_split.yaml")
        r3_002_config = _load_r3_002_config(config_path)
        if args.output_dir is not None:
            r3_002_config = RelationSplitProtocolConfig(
                development_representativeness_bank_size=r3_002_config.development_representativeness_bank_size,
                development_representativeness_query_examples=r3_002_config.development_representativeness_query_examples,
                run_empirical_exposure_probe=r3_002_config.run_empirical_exposure_probe,
                empirical_probe_seed=r3_002_config.empirical_probe_seed,
                device=r3_002_config.device,
                output_dir=args.output_dir,
            )
        report = run_relation_split_protocol(r3_002_config)
        next_blocked = "B-C005R3-003 onward"
    elif args.task == "B-C005R3-003":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_functional_metrics_v2.yaml")
        r3_003_config = _load_r3_003_config(config_path)
        if args.output_dir is not None:
            r3_003_config = FunctionalMetricsV2Config(
                tau=r3_003_config.tau,
                max_candidates=r3_003_config.max_candidates,
                looks=r3_003_config.looks,
                alpha_accept_episode=r3_003_config.alpha_accept_episode,
                alpha_reject_episode=r3_003_config.alpha_reject_episode,
                alpha_ref_episode=r3_003_config.alpha_ref_episode,
                output_dir=args.output_dir,
            )
        report = run_functional_metrics_v2_protocol(r3_003_config)
        next_blocked = "B-C005R3-004 onward"
    elif args.task == "B-C005R3-004":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_paired_baseline.yaml")
        r3_004_config = _load_r3_004_config(config_path)
        if args.output_dir is not None:
            r3_004_config = dataclasses.replace(r3_004_config, output_dir=args.output_dir)
        report = run_paired_baseline_repair(r3_004_config)
        next_blocked = "B-C005R3-005 onward"
    elif args.task == "B-C005R3-005":
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_safe_bounded_verification.yaml"
        )
        r3_005_config = _load_r3_005_config(config_path)
        if args.output_dir is not None:
            r3_005_config = dataclasses.replace(r3_005_config, output_dir=args.output_dir)
        report = run_safe_bounded_verification(r3_005_config)
        next_blocked = "B-C005R3-006 onward"
    elif args.task == "B-C005R3-006":
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_count_bind_key_scoring_repair.yaml"
        )
        r3_006_config = _load_r3_006_config(config_path)
        if args.output_dir is not None:
            r3_006_config = dataclasses.replace(r3_006_config, output_dir=args.output_dir)
        report = run_count_bind_key_scoring_repair(r3_006_config)
        next_blocked = "B-C005R3-007 onward"
    elif args.task == "B-C005R3-007":
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_select_argument_encoding_repair.yaml"
        )
        r3_007_config = _load_r3_007_config(config_path)
        if args.output_dir is not None:
            r3_007_config = dataclasses.replace(r3_007_config, output_dir=args.output_dir)
        report = run_select_argument_encoding_repair(r3_007_config)
        next_blocked = "B-C005R3-008 onward"
    elif args.task == "B-C005R3-008":
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_bind_argument_scorer_repair.yaml"
        )
        r3_008_config = _load_r3_008_config(config_path)
        if args.output_dir is not None:
            r3_008_config = dataclasses.replace(r3_008_config, output_dir=args.output_dir)
        report = run_bind_argument_scorer_repair(r3_008_config)
        next_blocked = "B-C005R3-009 onward"
    elif args.task == "B-C005R3-009":
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_shift_functional_generalization_repair.yaml"
        )
        r3_009_config = _load_r3_009_config(config_path)
        if args.output_dir is not None:
            r3_009_config = dataclasses.replace(r3_009_config, output_dir=args.output_dir)
        report = run_shift_functional_generalization_repair(r3_009_config)
        next_blocked = "B-C005R3-010 onward"
    else:  # B-C005R3-010
        config_path = args.config or Path(
            "configs/phase_b_b2_post_d2_paired_integration_regression.yaml"
        )
        r3_010_config = _load_r3_010_config(config_path)
        if args.output_dir is not None:
            r3_010_config = dataclasses.replace(r3_010_config, output_dir=args.output_dir)
        report = run_paired_integration_regression(r3_010_config)
        next_blocked = "B-C005R3-011 onward"

    print(json.dumps(report["protocol"], indent=2))
    print(
        f"STOP: only {args.task} was executed. {next_blocked} and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    result = report["protocol"]["result"]
    passing_results = (
        "INFRASTRUCTURE_OR_PROTOCOL_PASS",
        "COMPARISON_ESTABLISHED",
        "G3_PASS",
        "VALIDATION_PASS",
        "DEVELOPMENT_INTEGRATION_PASS",
    )
    return 0 if result in passing_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
