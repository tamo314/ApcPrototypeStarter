"""Explicit one-task dispatcher for the Phase B B2 model bundle recovery
series (Task B-C005REC-001..008).

Per `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` section 0 and
`docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md`: exactly one named
task runs per invocation, and there is no `--all` or implicit next-task
execution. B-C005REC-001 was a documentation/audit-only task (no `src/`
code, no dispatcher entry -- see ADR-0092); only B-C005REC-002 onward has a
dispatcher entry, and only once each is actually implemented. B-C005REC-003
(ADR-0094) fixes the full 16-primitive build DAG and preregistered recovery
protocol on CPU tiny fixtures + real-registry structural checks -- it starts
no GPU training and no 5-model run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.incremental_budget_calibration import (
    REC004A_RECHECK_QUERY_EXAMPLES,
    REC004A_STEP_LADDER,
    REC004A_VALIDATION_EXAMPLES,
    REC004A_VALIDATION_FLOOR,
    IncrementalBudgetCalibrationConfig,
    run_incremental_budget_calibration_task,
)
from apc.evaluation.mirror_attention_clamp_causal_replay import (
    MirrorAttentionClampCausalReplayConfig,
    run_mirror_attention_clamp_causal_replay_task,
)
from apc.evaluation.mirror_attention_score_credit_assignment_audit import (
    MirrorAttentionScoreCreditAssignmentAuditConfig,
    run_mirror_attention_score_credit_assignment_audit_task,
)
from apc.evaluation.mirror_budget_extension import (
    MirrorBudgetExtensionConfig,
    run_mirror_budget_extension_task,
)
from apc.evaluation.mirror_contamination_free_checkpoint_trajectory_audit import (
    MirrorContaminationFreeCheckpointTrajectoryAuditConfig,
    run_mirror_contamination_free_checkpoint_trajectory_audit_task,
)
from apc.evaluation.mirror_content_prep_release_pilot import (
    MirrorContentPrepReleasePilotConfig,
    run_mirror_content_prep_release_pilot_task,
)
from apc.evaluation.mirror_cross_position_cross_length_score_gradient_interference_audit import (
    MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig,
    run_mirror_cross_position_cross_length_score_gradient_interference_audit_task,
)
from apc.evaluation.mirror_cvof_trust_region_pilot import (
    MirrorCVOFTrustRegionPilotConfig,
    run_cvof_trust_region_pilot_task,
)
from apc.evaluation.mirror_dense_trajectory_transition_audit import (
    MirrorDenseTrajectoryTransitionAuditConfig,
    run_mirror_dense_trajectory_transition_audit_task,
)
from apc.evaluation.mirror_downstream_freeze_causal_replay import (
    MirrorDownstreamFreezeCausalReplayConfig,
    run_mirror_downstream_freeze_causal_replay_task,
)
from apc.evaluation.mirror_ffn_anchored_downstream_interaction_audit import (
    MirrorFfnAnchoredDownstreamInteractionAuditConfig,
    run_mirror_ffn_anchored_downstream_interaction_audit_task,
)
from apc.evaluation.mirror_ffn_value_path_leave_one_out_necessity_audit import (
    MirrorFfnValuePathLeaveOneOutNecessityAuditConfig,
    run_mirror_ffn_value_path_leave_one_out_necessity_audit_task,
)
from apc.evaluation.mirror_ffn_value_path_subcomponent_attribution import (
    MirrorFfnValuePathSubcomponentAttributionConfig,
    run_mirror_ffn_value_path_subcomponent_attribution_task,
)
from apc.evaluation.mirror_late_progress_conditional_extension import (
    MirrorLateProgressConditionalExtensionConfig,
    run_mirror_late_progress_conditional_extension_task,
)
from apc.evaluation.mirror_late_stage_attention_bottleneck_revalidation import (
    MirrorLateStageAttentionBottleneckRevalidationConfig,
    run_mirror_late_stage_attention_bottleneck_revalidation_task,
)
from apc.evaluation.mirror_normal_cvof_protection_pilot import (
    MirrorNormalCVOFProtectionPilotConfig,
    run_mirror_normal_cvof_protection_pilot_task,
)
from apc.evaluation.mirror_oracle_attention_substitution_probe import (
    OracleAttentionSubstitutionProbeConfig,
    run_oracle_attention_substitution_probe_task,
)
from apc.evaluation.mirror_position_bias_repair import (
    REC004D_CHECKPOINT_INTERVAL,
    REC004D_EXISTING_VALIDATION_EXAMPLES,
    REC004D_EXISTING_VALIDATION_FLOOR,
    REC004D_MAX_UPDATES_PER_RUN,
    REC004D_RECHECK_QUERY_EXAMPLES,
    MirrorPositionBiasRepairConfig,
    run_mirror_position_bias_repair_task,
)
from apc.evaluation.mirror_position_initialization_diagnostic import (
    REC004C_CHECKPOINT_INTERVAL,
    REC004C_DESCRIPTIVE_FLOOR,
    REC004C_MAX_UPDATES_PER_INIT,
    REC004C_SCHEDULE_VALIDATION_EXAMPLES,
    MirrorPositionInitializationDiagnosticConfig,
    run_mirror_position_initialization_diagnostic_task,
)
from apc.evaluation.mirror_position_score_residual_audit import (
    MirrorPositionScoreResidualAuditConfig,
    run_mirror_position_score_residual_audit_task,
)
from apc.evaluation.mirror_schedule_comparison import (
    REC004B_CHECKPOINT_INTERVAL,
    REC004B_MAX_UPDATES,
    REC004B_RECHECK_QUERY_EXAMPLES,
    REC004B_SCHEDULE_VALIDATION_EXAMPLES,
    REC004B_SCHEDULE_VALIDATION_FLOOR,
    MirrorScheduleComparisonConfig,
    run_mirror_schedule_comparison_task,
)
from apc.evaluation.mirror_score_function_finite_step_dynamics_audit import (
    MirrorScoreFunctionFiniteStepDynamicsAuditConfig,
    run_mirror_score_function_finite_step_dynamics_audit_task,
)
from apc.evaluation.mirror_score_only_continuation_pilot import (
    MirrorScoreOnlyContinuationPilotConfig,
    run_mirror_score_only_continuation_pilot_task,
)
from apc.evaluation.mirror_temporal_mechanism_rollback_audit import (
    MirrorTemporalMechanismRollbackAuditConfig,
    run_mirror_temporal_mechanism_rollback_audit_task,
)
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    ModelBundleContractConfig,
    PilotRestoreBuildConfig,
    RecoveryBuildPlanConfig,
    run_model_bundle_contract_task,
    run_pilot_restore_build_task,
    run_recovery_build_plan_task,
)

_IMPLEMENTED_TASKS = (
    "B-C005REC-002",
    "B-C005REC-003",
    "B-C005REC-004",
    "B-C005REC-004A",
    "B-C005REC-004B",
    "B-C005REC-004C",
    "B-C005REC-004D",
    "B-C005REC-004E",
    "B-C005REC-004F",
    "B-C005REC-004G",
    "B-C005REC-004H",
    "B-C005REC-004I",
    "B-C005REC-004J",
    "B-C005REC-004K",
    "B-C005REC-004L",
    "B-C005REC-004M",
    "B-C005REC-004N",
    "B-C005REC-004O",
    "B-C005REC-004P",
    "B-C005REC-004Q",
    "B-C005REC-004R",
    "B-C005REC-004S",
    "B-C005REC-004T",
    "B-C005REC-004U",
    "B-C005REC-004V",
    "B-C005REC-004W",
    "B-C005REC-004X",
)


def _load_rec002_config(config_path: Path) -> ModelBundleContractConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = ModelBundleContractConfig()
    real_core_source = raw.get("real_core_source", defaults.real_core_source)
    return ModelBundleContractConfig(
        real_core_source=Path(real_core_source) if real_core_source else None,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_rec003_config(config_path: Path) -> RecoveryBuildPlanConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = RecoveryBuildPlanConfig()
    seeds = raw.get("seeds", list(defaults.seeds))
    return RecoveryBuildPlanConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seeds=tuple(seeds),
    )


def _load_rec004_config(config_path: Path) -> PilotRestoreBuildConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = PilotRestoreBuildConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004's pilot seed is pre-registered as {RECOVERY_PILOT_SEED}; "
            f"config requested seed={seed}"
        )
    return PilotRestoreBuildConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        direct_query_examples_per_operation=raw.get(
            "direct_query_examples_per_operation", defaults.direct_query_examples_per_operation
        ),
        non_shift_floor=raw.get("non_shift_floor", defaults.non_shift_floor),
    )


def _load_rec004a_config(config_path: Path) -> IncrementalBudgetCalibrationConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = IncrementalBudgetCalibrationConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004A's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004); config requested seed={seed}"
        )
    return IncrementalBudgetCalibrationConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        step_ladder=tuple(raw.get("step_ladder", REC004A_STEP_LADDER)),
        validation_examples=raw.get("validation_examples", REC004A_VALIDATION_EXAMPLES),
        validation_floor=raw.get("validation_floor", REC004A_VALIDATION_FLOOR),
        recheck_query_examples=raw.get("recheck_query_examples", REC004A_RECHECK_QUERY_EXAMPLES),
    )


def _load_rec004b_config(config_path: Path) -> MirrorScheduleComparisonConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorScheduleComparisonConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004B's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A); config requested seed={seed}"
        )
    return MirrorScheduleComparisonConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        schedule_validation_examples=raw.get(
            "schedule_validation_examples", REC004B_SCHEDULE_VALIDATION_EXAMPLES
        ),
        schedule_validation_floor=raw.get(
            "schedule_validation_floor", REC004B_SCHEDULE_VALIDATION_FLOOR
        ),
        recheck_query_examples=raw.get("recheck_query_examples", REC004B_RECHECK_QUERY_EXAMPLES),
        checkpoint_interval=raw.get("checkpoint_interval", REC004B_CHECKPOINT_INTERVAL),
        max_updates=raw.get("max_updates", REC004B_MAX_UPDATES),
    )


def _load_rec004c_config(config_path: Path) -> MirrorPositionInitializationDiagnosticConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorPositionInitializationDiagnosticConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004C's Core seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A/REC-004B); config requested seed={seed}"
        )
    return MirrorPositionInitializationDiagnosticConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        schedule_validation_examples=raw.get(
            "schedule_validation_examples", REC004C_SCHEDULE_VALIDATION_EXAMPLES
        ),
        checkpoint_interval=raw.get("checkpoint_interval", REC004C_CHECKPOINT_INTERVAL),
        max_updates_per_init=raw.get("max_updates_per_init", REC004C_MAX_UPDATES_PER_INIT),
        descriptive_floor=raw.get("descriptive_floor", REC004C_DESCRIPTIVE_FLOOR),
    )


def _load_rec004d_config(config_path: Path) -> MirrorPositionBiasRepairConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorPositionBiasRepairConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004D's Core seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A/REC-004C); config requested seed={seed}"
        )
    return MirrorPositionBiasRepairConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        existing_validation_examples=raw.get(
            "existing_validation_examples", REC004D_EXISTING_VALIDATION_EXAMPLES
        ),
        existing_validation_floor=raw.get(
            "existing_validation_floor", REC004D_EXISTING_VALIDATION_FLOOR
        ),
        recheck_query_examples=raw.get("recheck_query_examples", REC004D_RECHECK_QUERY_EXAMPLES),
        checkpoint_interval=raw.get("checkpoint_interval", REC004D_CHECKPOINT_INTERVAL),
        max_updates_per_run=raw.get("max_updates_per_run", REC004D_MAX_UPDATES_PER_RUN),
    )


def _load_rec004e_config(config_path: Path) -> MirrorPositionScoreResidualAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorPositionScoreResidualAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004E reads REC-004D artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorPositionScoreResidualAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004f_config(config_path: Path) -> OracleAttentionSubstitutionProbeConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = OracleAttentionSubstitutionProbeConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004F reads REC-004D/REC-004E artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return OracleAttentionSubstitutionProbeConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004g_config(config_path: Path) -> MirrorBudgetExtensionConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorBudgetExtensionConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004G reads REC-004D artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorBudgetExtensionConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004h_config(config_path: Path) -> MirrorLateProgressConditionalExtensionConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorLateProgressConditionalExtensionConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004H reads REC-004G artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorLateProgressConditionalExtensionConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004i_config(
    config_path: Path,
) -> MirrorContaminationFreeCheckpointTrajectoryAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorContaminationFreeCheckpointTrajectoryAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004I reads REC-004D/G/H artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorContaminationFreeCheckpointTrajectoryAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004j_config(
    config_path: Path,
) -> MirrorLateStageAttentionBottleneckRevalidationConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorLateStageAttentionBottleneckRevalidationConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004J reads REC-004H artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorLateStageAttentionBottleneckRevalidationConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004k_config(
    config_path: Path,
) -> MirrorTemporalMechanismRollbackAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorTemporalMechanismRollbackAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004K reads REC-004D/H/J artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorTemporalMechanismRollbackAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004l_config(
    config_path: Path,
) -> MirrorFfnAnchoredDownstreamInteractionAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorFfnAnchoredDownstreamInteractionAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004L reads REC-004D/H/K artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorFfnAnchoredDownstreamInteractionAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004m_config(
    config_path: Path,
) -> MirrorFfnValuePathSubcomponentAttributionConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorFfnValuePathSubcomponentAttributionConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004M reads REC-004D/H/K/L artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorFfnValuePathSubcomponentAttributionConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004n_config(
    config_path: Path,
) -> MirrorFfnValuePathLeaveOneOutNecessityAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorFfnValuePathLeaveOneOutNecessityAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004N reads REC-004D/H/K/L/M artifacts pre-registered under "
            f"seed {RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorFfnValuePathLeaveOneOutNecessityAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004o_config(config_path: Path) -> MirrorScoreOnlyContinuationPilotConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorScoreOnlyContinuationPilotConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004O is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorScoreOnlyContinuationPilotConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        checkpoint_interval=raw.get("checkpoint_interval", defaults.checkpoint_interval),
    )


def _load_rec004p_config(config_path: Path) -> MirrorContentPrepReleasePilotConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorContentPrepReleasePilotConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004P is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorContentPrepReleasePilotConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        checkpoint_interval=raw.get("checkpoint_interval", defaults.checkpoint_interval),
    )


def _load_rec004q_config(config_path: Path) -> MirrorAttentionScoreCreditAssignmentAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorAttentionScoreCreditAssignmentAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004Q is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorAttentionScoreCreditAssignmentAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
    )


def _load_rec004r_config(
    config_path: Path,
) -> MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004R is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)), seed=seed
    )


def _load_rec004s_config(
    config_path: Path,
) -> MirrorScoreFunctionFiniteStepDynamicsAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorScoreFunctionFiniteStepDynamicsAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004S is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorScoreFunctionFiniteStepDynamicsAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)), seed=seed
    )


def _load_rec004t_config(
    config_path: Path,
) -> MirrorDenseTrajectoryTransitionAuditConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorDenseTrajectoryTransitionAuditConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004T is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorDenseTrajectoryTransitionAuditConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        oracle_em_threshold=raw.get("oracle_em_threshold", defaults.oracle_em_threshold),
        max_replay_window_updates=raw.get(
            "max_replay_window_updates", defaults.max_replay_window_updates
        ),
        full_probe_step_interval=raw.get(
            "full_probe_step_interval", defaults.full_probe_step_interval
        ),
        sentinel_subset_per_dataset=raw.get(
            "sentinel_subset_per_dataset", defaults.sentinel_subset_per_dataset
        ),
    )


def _load_rec004u_config(
    config_path: Path,
) -> MirrorAttentionClampCausalReplayConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorAttentionClampCausalReplayConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004U is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorAttentionClampCausalReplayConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        oracle_em_threshold=raw.get("oracle_em_threshold", defaults.oracle_em_threshold),
        max_replay_window_updates=raw.get(
            "max_replay_window_updates", defaults.max_replay_window_updates
        ),
        full_probe_step_interval=raw.get(
            "full_probe_step_interval", defaults.full_probe_step_interval
        ),
        sentinel_subset_per_dataset=raw.get(
            "sentinel_subset_per_dataset", defaults.sentinel_subset_per_dataset
        ),
    )


def _load_rec004v_config(
    config_path: Path,
) -> MirrorDownstreamFreezeCausalReplayConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorDownstreamFreezeCausalReplayConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004V is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorDownstreamFreezeCausalReplayConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        oracle_em_threshold=raw.get("oracle_em_threshold", defaults.oracle_em_threshold),
        max_replay_window_updates=raw.get(
            "max_replay_window_updates", defaults.max_replay_window_updates
        ),
        full_probe_step_interval=raw.get(
            "full_probe_step_interval", defaults.full_probe_step_interval
        ),
        sentinel_subset_per_dataset=raw.get(
            "sentinel_subset_per_dataset", defaults.sentinel_subset_per_dataset
        ),
        parity_updates=raw.get("parity_updates", defaults.parity_updates),
    )


def _load_rec004w_config(
    config_path: Path,
) -> MirrorNormalCVOFProtectionPilotConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorNormalCVOFProtectionPilotConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004W is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorNormalCVOFProtectionPilotConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        oracle_em_threshold=raw.get("oracle_em_threshold", defaults.oracle_em_threshold),
        j0_delta_floor=raw.get("j0_delta_floor", defaults.j0_delta_floor),
        max_replay_window_updates=raw.get(
            "max_replay_window_updates", defaults.max_replay_window_updates
        ),
        full_probe_step_interval=raw.get(
            "full_probe_step_interval", defaults.full_probe_step_interval
        ),
        sentinel_subset_per_dataset=raw.get(
            "sentinel_subset_per_dataset", defaults.sentinel_subset_per_dataset
        ),
        parity_updates=raw.get("parity_updates", defaults.parity_updates),
    )


def _load_rec004x_config(
    config_path: Path,
) -> MirrorCVOFTrustRegionPilotConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorCVOFTrustRegionPilotConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004X is fixed to I03's pre-registered seed "
            f"{RECOVERY_PILOT_SEED}; config requested seed={seed}"
        )
    return MirrorCVOFTrustRegionPilotConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        oracle_em_threshold=raw.get("oracle_em_threshold", defaults.oracle_em_threshold),
        j0_delta_floor=raw.get("j0_delta_floor", defaults.j0_delta_floor),
        j0_hard_freeze_delta_floor=raw.get(
            "j0_hard_freeze_delta_floor", defaults.j0_hard_freeze_delta_floor
        ),
        max_replay_window_updates=raw.get(
            "max_replay_window_updates", defaults.max_replay_window_updates
        ),
        full_probe_step_interval=raw.get(
            "full_probe_step_interval", defaults.full_probe_step_interval
        ),
        sentinel_subset_per_dataset=raw.get(
            "sentinel_subset_per_dataset", defaults.sentinel_subset_per_dataset
        ),
        parity_updates=raw.get("parity_updates", defaults.parity_updates),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B2 model bundle recovery -- explicit single-task dispatch"
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
        help="Defaults to this task's own configs/phase_b_b2_model_bundle_recovery_<task>.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.task not in _IMPLEMENTED_TASKS:
        print(
            f"ERROR: task {args.task!r} is not implemented by this dispatcher. "
            f"Only {_IMPLEMENTED_TASKS} run today; every later task in "
            "docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md requires its own "
            "explicit user instruction and its own implementation before it "
            "can be dispatched here.",
            file=sys.stderr,
        )
        return 2

    if args.task == "B-C005REC-002":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec002.yaml")
        rec002_config = _load_rec002_config(config_path)
        if args.output_dir is not None:
            rec002_config = dataclasses.replace(rec002_config, output_dir=args.output_dir)
        report = run_model_bundle_contract_task(rec002_config)
        expected_result = "RG1_PASS"
    elif args.task == "B-C005REC-003":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec003.yaml")
        rec003_config = _load_rec003_config(config_path)
        if args.output_dir is not None:
            rec003_config = dataclasses.replace(rec003_config, output_dir=args.output_dir)
        report = run_recovery_build_plan_task(rec003_config)
        expected_result = "RG2_PASS"
    elif args.task == "B-C005REC-004":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004.yaml")
        rec004_config = _load_rec004_config(config_path)
        if args.output_dir is not None:
            rec004_config = dataclasses.replace(rec004_config, output_dir=args.output_dir)
        report = run_pilot_restore_build_task(rec004_config)
        expected_result = "RG3_PASS"
    elif args.task == "B-C005REC-004A":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004a.yaml")
        rec004a_config = _load_rec004a_config(config_path)
        if args.output_dir is not None:
            rec004a_config = dataclasses.replace(rec004a_config, output_dir=args.output_dir)
        rec004a_report = run_incremental_budget_calibration_task(rec004a_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004a_report["implementation_status"],
                    "calibration_status": rec004a_report["calibration_status"],
                    "rg3_recheck": rec004a_report["rg3_recheck"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004A was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004a_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1
    elif args.task == "B-C005REC-004B":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004b.yaml")
        rec004b_config = _load_rec004b_config(config_path)
        if args.output_dir is not None:
            rec004b_config = dataclasses.replace(rec004b_config, output_dir=args.output_dir)
        rec004b_report = run_mirror_schedule_comparison_task(rec004b_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004b_report["implementation_status"],
                    "paired_comparison_status": rec004b_report["paired_comparison_status"],
                    "mirror_candidate_status": rec004b_report["mirror_candidate_status"],
                    "rg3_recheck": rec004b_report["rg3_recheck"],
                    "rec005_eligible": rec004b_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004B was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004b_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1
    elif args.task == "B-C005REC-004C":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004c.yaml")
        rec004c_config = _load_rec004c_config(config_path)
        if args.output_dir is not None:
            rec004c_config = dataclasses.replace(rec004c_config, output_dir=args.output_dir)
        rec004c_report = run_mirror_position_initialization_diagnostic_task(rec004c_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004c_report["implementation_status"],
                    "source_audit_status": rec004c_report["source_audit_status"],
                    "execution_contract_status": rec004c_report["execution_contract_status"],
                    "initialization_experiment_status": (
                        rec004c_report["initialization_experiment_status"]
                    ),
                    "selected_init": rec004c_report["selected_init"],
                    "child_bundle": rec004c_report["child_bundle"],
                    "rg3_recheck": rec004c_report["rg3_recheck"],
                    "rec005_eligible": rec004c_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004C was executed (diagnostic only -- no candidate "
            "selected, no child bundle, no RG3 recheck). B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004c_report["execution_contract_status"] == "PASS" else 1
    elif args.task == "B-C005REC-004D":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004d.yaml")
        rec004d_config = _load_rec004d_config(config_path)
        if args.output_dir is not None:
            rec004d_config = dataclasses.replace(rec004d_config, output_dir=args.output_dir)
        rec004d_report = run_mirror_position_bias_repair_task(rec004d_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004d_report["implementation_status"],
                    "source_audit_status": rec004d_report["source_audit_status"],
                    "candidate_status": rec004d_report["candidate_status"],
                    "rg3_recheck": rec004d_report["rg3_recheck"],
                    "rec005_eligible": rec004d_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004D was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004d_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1
    elif args.task == "B-C005REC-004E":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004e.yaml")
        rec004e_config = _load_rec004e_config(config_path)
        if args.output_dir is not None:
            rec004e_config = dataclasses.replace(rec004e_config, output_dir=args.output_dir)
        rec004e_report = run_mirror_position_score_residual_audit_task(rec004e_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004e_report["implementation_status"],
                    "source_replay_status": rec004e_report["source_replay_status"],
                    "position_grid_status": rec004e_report["position_grid_status"],
                    "score_observation_status": rec004e_report["score_observation_status"],
                    "intervention_status": rec004e_report["intervention_status"],
                    "residual_diagnosis": rec004e_report["residual_diagnosis"],
                    "next_repair_contract": rec004e_report["next_repair_contract"],
                    "rg3_recheck": rec004e_report["rg3_recheck"],
                    "rec005_eligible": rec004e_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004E was executed (diagnostic + at-most-one proposed "
            "repair contract -- no implementation, no training, no candidate, no child "
            "bundle, no RG3 recheck). B-C005REC-005 onward and B-C006/Task Inference "
            "remain blocked pending an explicit next user instruction."
        )
        return 0 if rec004e_report["implementation_status"] == "COMPLETE" else 1
    elif args.task == "B-C005REC-004F":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004f.yaml")
        rec004f_config = _load_rec004f_config(config_path)
        if args.output_dir is not None:
            rec004f_config = dataclasses.replace(rec004f_config, output_dir=args.output_dir)
        rec004f_report = run_oracle_attention_substitution_probe_task(rec004f_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004f_report["implementation_status"],
                    "probe_status": rec004f_report["probe_status"],
                    "diagnosis": rec004f_report["diagnosis"],
                    "rg3_recheck": rec004f_report["rg3_recheck"],
                    "rec005_eligible": rec004f_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004F was executed (the single diagnostic probe REC-004E's "
            "own next_repair_contract.md proposed -- no implementation, no training, no "
            "candidate, no child bundle, no RG3 recheck). B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004f_report["implementation_status"] == "COMPLETE" else 1
    elif args.task == "B-C005REC-004G":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004g.yaml")
        rec004g_config = _load_rec004g_config(config_path)
        if args.output_dir is not None:
            rec004g_config = dataclasses.replace(rec004g_config, output_dir=args.output_dir)
        rec004g_report = run_mirror_budget_extension_task(rec004g_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004g_report.get("implementation_status"),
                    "source_replay_status": rec004g_report.get("source_replay_status"),
                    "extension_training_status": rec004g_report.get("extension_training_status"),
                    "candidate_status_at_target_step": rec004g_report.get(
                        "candidate_status_at_target_step"
                    ),
                    "rg3_recheck": rec004g_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004G was executed (a limited budget-extension "
            "experiment on REC-004D's own saved P/I01-I05 checkpoints -- no candidate "
            "selected, no child bundle, no RG3 recheck even if the floor is cleared). "
            "B-C005REC-005 (the real five-model cohort) onward, B-C005R3-011, "
            "B-C006, and Task Inference remain blocked pending an explicit next "
            "user instruction."
        )
        return 0 if rec004g_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004H":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004h.yaml")
        rec004h_config = _load_rec004h_config(config_path)
        if args.output_dir is not None:
            rec004h_config = dataclasses.replace(rec004h_config, output_dir=args.output_dir)
        rec004h_report = run_mirror_late_progress_conditional_extension_task(rec004h_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004h_report.get("implementation_status"),
                    "source_replay_status": rec004h_report.get("source_replay_status"),
                    "late_audit_status": rec004h_report.get("late_audit_status"),
                    "continuation_decision": rec004h_report.get("continuation_decision"),
                    "extension_status": rec004h_report.get("extension_status"),
                    "new_optimizer_updates": rec004h_report.get("new_optimizer_updates"),
                    "terminal_floor_status": rec004h_report.get("terminal_floor_status"),
                    "rg3_recheck": rec004h_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004H was executed (a length-10 late-progress audit "
            "plus, only if every below-floor init showed confirmed late-stage "
            "progress, a conditional extension of all 5 P/I01-I05 inits from "
            "REC-004G's saved step=12000 states to a common step=18000 -- no "
            "candidate selected, no child bundle, no RG3 recheck even if all 5 clear "
            "the floor at step=18000). B-C005REC-005 (the real five-model cohort) "
            "onward, B-C005R3-011, B-C006, and Task Inference remain blocked pending "
            "an explicit next user instruction."
        )
        return 0 if rec004h_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004I":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004i.yaml")
        rec004i_config = _load_rec004i_config(config_path)
        if args.output_dir is not None:
            rec004i_config = dataclasses.replace(rec004i_config, output_dir=args.output_dir)
        rec004i_report = run_mirror_contamination_free_checkpoint_trajectory_audit_task(
            rec004i_config
        )
        print(
            json.dumps(
                {
                    "implementation_status": rec004i_report.get("implementation_status"),
                    "source_replay_status": rec004i_report.get("source_replay_status"),
                    "checkpoints_evaluated": rec004i_report.get("checkpoints_evaluated"),
                    "new_optimizer_updates": rec004i_report.get("new_optimizer_updates"),
                    "pattern_classification": rec004i_report.get("pattern_classification"),
                    "rg3_recheck": rec004i_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004I was executed (a forward-only, zero-new-update "
            "trajectory audit of every saved P/I01-I05 checkpoint step=6000-18000 "
            "against a newly built, contamination-checked clean_selection_validation_v2 "
            "set -- no training, no candidate selected, no child bundle, no RG3 "
            "recheck). B-C005REC-005 (the real five-model cohort) onward, "
            "B-C005R3-011, B-C006, and Task Inference remain blocked pending an "
            "explicit next user instruction."
        )
        return 0 if rec004i_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004J":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004j.yaml")
        rec004j_config = _load_rec004j_config(config_path)
        if args.output_dir is not None:
            rec004j_config = dataclasses.replace(rec004j_config, output_dir=args.output_dir)
        rec004j_report = run_mirror_late_stage_attention_bottleneck_revalidation_task(
            rec004j_config
        )
        print(
            json.dumps(
                {
                    "implementation_status": rec004j_report.get("implementation_status"),
                    "checkpoint_source_replay_status": rec004j_report.get(
                        "checkpoint_source_replay_status"
                    ),
                    "clean_v2_source_replay_status": rec004j_report.get(
                        "clean_v2_source_replay_status"
                    ),
                    "i03_17500_decision": rec004j_report.get("i03_17500_decision"),
                    "i05_collapse_classification": rec004j_report.get(
                        "i05_collapse_classification"
                    ),
                    "new_optimizer_updates": rec004j_report.get("new_optimizer_updates"),
                    "rg3_recheck": rec004j_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004J was executed (a forward-only, zero-new-update "
            "revalidation of REC-004F's oracle-attention-substitution finding at "
            "step=17500/18000 for I03/I04/I05, on REC-004I's own clean_selection_"
            "validation_v2 length-10 subset plus a new development-exposed "
            "length10_mechanism_probe_v1 set -- no training, no candidate selected, "
            "no child bundle, no RG3 recheck). B-C005REC-005 (the real five-model "
            "cohort) onward, B-C005R3-011, B-C006, and Task Inference remain blocked "
            "pending an explicit next user instruction."
        )
        return 0 if rec004j_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004K":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004k.yaml")
        rec004k_config = _load_rec004k_config(config_path)
        if args.output_dir is not None:
            rec004k_config = dataclasses.replace(rec004k_config, output_dir=args.output_dir)
        rec004k_report = run_mirror_temporal_mechanism_rollback_audit_task(rec004k_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004k_report.get("implementation_status"),
                    "checkpoint_source_replay_status": rec004k_report.get(
                        "checkpoint_source_replay_status"
                    ),
                    "rec004j_cross_check_status": rec004k_report.get("rec004j_cross_check_status"),
                    "stage_a_temporal_decision": rec004k_report.get("stage_a_temporal_decision"),
                    "stage_c_status": rec004k_report.get("stage_c_status"),
                    "component_decomposition_parity_status": rec004k_report.get(
                        "component_decomposition_parity_status"
                    ),
                    "i03_component_decision": rec004k_report.get("i03_component_decision"),
                    "new_optimizer_updates": rec004k_report.get("new_optimizer_updates"),
                    "rg3_recheck": rec004k_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004K was executed (a forward-only, zero-new-update "
            "matched-data temporal recheck of I03's oracle-attention finding at "
            "step=6000 against REC-004J's own two length-10 datasets, plus -- only if "
            "that recheck confirmed a genuine temporal mechanism shift -- a same-init "
            "component rollback audit from step=17500 to step=6000 on an in-memory, "
            "evaluation-only merged primitive -- no training, no candidate selected, "
            "no child bundle, no RG3 recheck, no I03 checkpoint file ever modified). "
            "B-C005REC-005 (the real five-model cohort) onward, B-C005R3-011, "
            "B-C006, and Task Inference remain blocked pending an explicit next "
            "user instruction."
        )
        return 0 if rec004k_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004L":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004l.yaml")
        rec004l_config = _load_rec004l_config(config_path)
        if args.output_dir is not None:
            rec004l_config = dataclasses.replace(rec004l_config, output_dir=args.output_dir)
        rec004l_report = run_mirror_ffn_anchored_downstream_interaction_audit_task(rec004l_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004l_report.get("implementation_status"),
                    "forward_graph_invariance_status": rec004l_report.get(
                        "forward_graph_invariance_status"
                    ),
                    "cross_check_status": rec004l_report.get("cross_check_status"),
                    "component_decomposition_parity_status": rec004l_report.get(
                        "component_decomposition_parity_status"
                    ),
                    "ffn_interaction_decision": rec004l_report.get("ffn_interaction_decision"),
                    "new_optimizer_updates": rec004l_report.get("new_optimizer_updates"),
                    "rg3_recheck": rec004l_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004L was executed (a forward-only, zero-new-update "
            "FFN-anchored downstream interaction audit -- fixing FFN_BLOCK as the base "
            "and rolling it back jointly with exactly one partner component at a time "
            "from I03@17500 to I03@6000, on an in-memory, evaluation-only merged "
            "primitive, across REC-004K's own two datasets plus one new disjoint "
            "development-exposed set -- no training, no candidate selected, no child "
            "bundle, no RG3 recheck, no I03 checkpoint file ever modified, no "
            "3-component or exhaustive combination search). B-C005REC-005 (the real "
            "five-model cohort) onward, B-C005R3-011, B-C006, and Task Inference "
            "remain blocked pending an explicit next user instruction."
        )
        return 0 if rec004l_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004M":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004m.yaml")
        rec004m_config = _load_rec004m_config(config_path)
        if args.output_dir is not None:
            rec004m_config = dataclasses.replace(rec004m_config, output_dir=args.output_dir)
        rec004m_report = run_mirror_ffn_value_path_subcomponent_attribution_task(rec004m_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004m_report.get("implementation_status"),
                    "forward_graph_invariance_status": rec004m_report.get(
                        "forward_graph_invariance_status"
                    ),
                    "qk_rollback_invariance_status": rec004m_report.get(
                        "qk_rollback_invariance_status"
                    ),
                    "cross_check_status": rec004m_report.get("cross_check_status"),
                    "decomposition_parity_status": rec004m_report.get(
                        "decomposition_parity_status"
                    ),
                    "value_path_subcomponent_decision": rec004m_report.get(
                        "value_path_subcomponent_decision"
                    ),
                    "safety_check_status": rec004m_report.get("safety_check_status"),
                    "new_optimizer_updates": rec004m_report.get("new_optimizer_updates"),
                    "rg3_recheck": rec004m_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004M was executed (a forward-only, zero-new-update "
            "attribution audit that further splits REC-004K/L's own VALUE_OUTPROJ "
            "group into CONTENT_PREP/V_PROJECTION/ATTN_OUT_PROJ/SCORE_PROJECTION_"
            "CONTROL by real state-dict key and, for the fused Q/K/V tensor, by "
            "row-slice -- rolling FFN_BLOCK back jointly with exactly one such "
            "subcomponent at a time from I03@17500 to I03@6000, on an in-memory, "
            "evaluation-only merged primitive, across four datasets -- plus a "
            "conditional safety check on I04/I05's own trajectories if the result "
            "localizes to one subcomponent. No training, no candidate selected, no "
            "child bundle, no RG3 recheck, no checkpoint file ever modified, no "
            "combination beyond the fixed set. B-C005REC-005 (the real five-model "
            "cohort) onward, B-C005R3-011, B-C006, and Task Inference remain blocked "
            "pending an explicit next user instruction."
        )
        return 0 if rec004m_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004N":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004n.yaml")
        rec004n_config = _load_rec004n_config(config_path)
        if args.output_dir is not None:
            rec004n_config = dataclasses.replace(rec004n_config, output_dir=args.output_dir)
        rec004n_report = run_mirror_ffn_value_path_leave_one_out_necessity_audit_task(
            rec004n_config
        )
        print(
            json.dumps(
                {
                    "implementation_status": rec004n_report.get("implementation_status"),
                    "source_replay_status": rec004n_report.get("source_replay_status"),
                    "qk_rollback_invariance_status": rec004n_report.get(
                        "qk_rollback_invariance_status"
                    ),
                    "j0_attention_score_path_invariant": rec004n_report.get(
                        "j0_attention_score_path_invariant"
                    ),
                    "value_path_necessity_decision": rec004n_report.get(
                        "value_path_necessity_decision"
                    ),
                    "value_path_necessity_tags": rec004n_report.get("value_path_necessity_tags"),
                    "safety_check_status": rec004n_report.get("safety_check_status"),
                    "new_optimizer_updates": rec004n_report.get("new_optimizer_updates"),
                    "rg3_recheck": rec004n_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004N was executed (a forward-only, zero-new-update "
            "leave-one-out necessity audit -- starting from the already-established "
            "sufficient set FFN_BLOCK+CONTENT_PREP+V_PROJECTION+ATTN_OUT_PROJ and "
            "removing exactly one VALUE_OUTPROJ subcomponent at a time, on an "
            "in-memory, evaluation-only merged primitive, across five datasets, plus "
            "a J0 (real, non-oracle) attention-equivalence check for F_VO against "
            "I03@17500 and a conditional I04/I05 safety check if F_VO passes. No "
            "training, no candidate selected, no child bundle, no RG3 recheck, no "
            "checkpoint file ever modified, no combination beyond the fixed set. "
            "B-C005REC-005 (the real five-model cohort) onward, B-C005R3-011, "
            "B-C006, and Task Inference remain blocked pending an explicit next "
            "user instruction."
        )
        return 0 if rec004n_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004O":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004o.yaml")
        rec004o_config = _load_rec004o_config(config_path)
        if args.output_dir is not None:
            rec004o_config = dataclasses.replace(rec004o_config, output_dir=args.output_dir)
        rec004o_report = run_mirror_score_only_continuation_pilot_task(rec004o_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004o_report.get("implementation_status"),
                    "result_label": rec004o_report.get("result_label"),
                    "joint_control_replay": rec004o_report.get("joint_control_replay", {}).get(
                        "status"
                    ),
                    "frozen_downstream_exact": rec004o_report.get("decision", {}).get(
                        "frozen_downstream_exact"
                    ),
                    "new_optimizer_updates": rec004o_report.get("cost_accounting", {}).get(
                        "new_optimizer_updates"
                    ),
                    "rg3_recheck": rec004o_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004O was executed (a single-init I03 score-only "
            "continuation pilot with a mandatory historical 500-step joint replay and "
            "a fail-closed fused-QKV V-row freeze contract). No candidate is selected, "
            "no child bundle is built, and no RG3/REC-005/sealed evaluation runs."
        )
        return 0 if rec004o_report.get("implementation_status") == "COMPLETED" else 1
    elif args.task == "B-C005REC-004P":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004p.yaml")
        rec004p_config = _load_rec004p_config(config_path)
        if args.output_dir is not None:
            rec004p_config = dataclasses.replace(rec004p_config, output_dir=args.output_dir)
        rec004p_report = run_mirror_content_prep_release_pilot_task(rec004p_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004p_report.get("implementation_status"),
                    "result_label": rec004p_report.get("result_label"),
                    "source_replay": rec004p_report.get("source_replay", {}).get("status"),
                    "new_optimizer_updates": rec004p_report.get("cost_accounting", {}).get(
                        "new_optimizer_updates"
                    ),
                    "rg3_recheck": rec004p_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004P was executed (the single-init I03 "
            "CONTENT_PREP-release score/value compatibility pilot). No candidate is "
            "selected, no child bundle is built, and no RG3/REC-005/sealed evaluation runs."
        )
        return 0 if rec004p_report.get("implementation_status") == "COMPLETED" else 1
    elif args.task == "B-C005REC-004Q":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004q.yaml")
        rec004q_config = _load_rec004q_config(config_path)
        if args.output_dir is not None:
            rec004q_config = dataclasses.replace(rec004q_config, output_dir=args.output_dir)
        rec004q_report = run_mirror_attention_score_credit_assignment_audit_task(rec004q_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004q_report.get("implementation_status"),
                    "result_label": rec004q_report.get("result_label"),
                    "rec004p_metric_audit": rec004q_report.get("rec004p_metric_audit"),
                    "new_optimizer_updates": rec004q_report.get("cost_accounting", {}).get(
                        "new_optimizer_updates"
                    ),
                    "rg3_recheck": rec004q_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004Q was executed (a read-only I03 attention-score "
            "credit-assignment and AdamW-state diagnostic). No optimizer step, training, "
            "candidate selection, child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004q_report.get("implementation_status") == "COMPLETED" else 1
    elif args.task == "B-C005REC-004R":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004r.yaml")
        rec004r_config = _load_rec004r_config(config_path)
        if args.output_dir is not None:
            rec004r_config = dataclasses.replace(rec004r_config, output_dir=args.output_dir)
        rec004r_report = (
            run_mirror_cross_position_cross_length_score_gradient_interference_audit_task(
                rec004r_config
            )
        )
        print(
            json.dumps(
                {
                    "implementation_status": rec004r_report.get("implementation_status"),
                    "result_label": rec004r_report.get("result_label"),
                    "new_optimizer_updates": rec004r_report.get("cost_accounting", {}).get(
                        "new_optimizer_updates"
                    ),
                    "rg3_recheck": rec004r_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004R was executed (a read-only I03 shared-score "
            "cross-position/cross-length gradient aggregation diagnostic). No optimizer "
            "step, repair training, candidate selection, child bundle, RG3/REC-005, or "
            "sealed evaluation ran."
        )
        return 0 if rec004r_report.get("implementation_status") == "COMPLETED" else 1
    elif args.task == "B-C005REC-004S":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004s.yaml")
        rec004s_config = _load_rec004s_config(config_path)
        if args.output_dir is not None:
            rec004s_config = dataclasses.replace(rec004s_config, output_dir=args.output_dir)
        rec004s_report = run_mirror_score_function_finite_step_dynamics_audit_task(rec004s_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004s_report.get("implementation_status"),
                    "result_label": rec004s_report.get("result_label"),
                    "new_optimizer_updates": rec004s_report.get("cost_accounting", {}).get(
                        "new_optimizer_updates"
                    ),
                    "rg3_recheck": rec004s_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004S was executed (a read-only I03 finite-step score-function "
            "dynamics vs local linear prediction audit). No optimizer step, repair training, "
            "candidate selection, child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004s_report.get("implementation_status") == "COMPLETED" else 1
    elif args.task == "B-C005REC-004T":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004t.yaml")
        rec004t_config = _load_rec004t_config(config_path)
        if args.output_dir is not None:
            rec004t_config = dataclasses.replace(rec004t_config, output_dir=args.output_dir)
        rec004t_report = run_mirror_dense_trajectory_transition_audit_task(rec004t_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004t_report.get("implementation_status"),
                    "coarse_localization": rec004t_report.get("coarse_localization"),
                    "historical_dense_replay": rec004t_report.get("historical_dense_replay"),
                    "trajectory_diagnosis": rec004t_report.get("trajectory_diagnosis"),
                    "transition_window": rec004t_report.get("transition_window"),
                    "temporal_peaks": rec004t_report.get("temporal_peaks"),
                    "diagnostic_replay_optimizer_updates": rec004t_report.get(
                        "diagnostic_replay_optimizer_updates"
                    ),
                    "new_candidate_training_updates": rec004t_report.get(
                        "new_candidate_training_updates"
                    ),
                    "rg3_recheck": rec004t_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004T was executed (an I03 dense trajectory transition replay "
            "and oracle-compatibility onset audit). No candidate training, candidate selection, "
            "child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004t_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004U":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004u.yaml")
        rec004u_config = _load_rec004u_config(config_path)
        if args.output_dir is not None:
            rec004u_config = dataclasses.replace(rec004u_config, output_dir=args.output_dir)
        rec004u_report = run_mirror_attention_clamp_causal_replay_task(rec004u_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004u_report.get("implementation_status"),
                    "historical_control_source": rec004u_report.get("historical_control_source"),
                    "attention_clamp_initial_parity": rec004u_report.get(
                        "attention_clamp_initial_parity"
                    ),
                    "counterfactual_optimizer_updates": rec004u_report.get(
                        "counterfactual_optimizer_updates"
                    ),
                    "new_candidate_training_updates": rec004u_report.get(
                        "new_candidate_training_updates"
                    ),
                    "causal_diagnosis": rec004u_report.get("causal_diagnosis"),
                    "T_oracle_loss_historical": rec004u_report.get("T_oracle_loss_historical"),
                    "T_oracle_loss_clamp": rec004u_report.get("T_oracle_loss_clamp"),
                    "rg3_recheck": rec004u_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004U was executed (an I03 pre-transition attention-clamp "
            "causal replay & score-stability repair gate). No candidate training, "
            "candidate selection, child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004u_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004V":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004v.yaml")
        rec004v_config = _load_rec004v_config(config_path)
        if args.output_dir is not None:
            rec004v_config = dataclasses.replace(rec004v_config, output_dir=args.output_dir)
        rec004v_report = run_mirror_downstream_freeze_causal_replay_task(rec004v_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004v_report.get("implementation_status"),
                    "baseline_parity": rec004v_report.get("baseline_parity"),
                    "counterfactual_optimizer_updates": rec004v_report.get(
                        "counterfactual_optimizer_updates"
                    ),
                    "new_candidate_training_updates": rec004v_report.get(
                        "new_candidate_training_updates"
                    ),
                    "downstream_causal_diagnosis": rec004v_report.get(
                        "downstream_causal_diagnosis"
                    ),
                    "decision_case": rec004v_report.get("decision_case"),
                    "strong_single_groups": rec004v_report.get("strong_single_groups"),
                    "cvof_strong": rec004v_report.get("cvof_strong"),
                    "rg3_recheck": rec004v_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004V was executed (an I03 attention-clamped downstream "
            "freeze necessity replay). No candidate training, candidate selection, "
            "child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004v_report.get("implementation_status") == "COMPLETE" else 1
    elif args.task == "B-C005REC-004W":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004w.yaml")
        rec004w_config = _load_rec004w_config(config_path)
        if args.output_dir is not None:
            rec004w_config = dataclasses.replace(rec004w_config, output_dir=args.output_dir)
        rec004w_report = run_mirror_normal_cvof_protection_pilot_task(rec004w_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004w_report.get("implementation_status"),
                    "historical_source": rec004w_report.get("historical_source"),
                    "initial_parity": rec004w_report.get("initial_parity"),
                    "stage_a_parity": rec004w_report.get("stage_a_parity"),
                    "intervention_optimizer_updates": rec004w_report.get(
                        "intervention_optimizer_updates"
                    ),
                    "new_candidate_training_updates": rec004w_report.get(
                        "new_candidate_training_updates"
                    ),
                    "result_label": rec004w_report.get("result_label"),
                    "success_gate": rec004w_report.get("success_gate"),
                    "fresh_normal_validation_overall_j0_em": rec004w_report.get(
                        "fresh_normal_validation_overall_j0_em"
                    ),
                    "fresh_normal_validation_j0_delta_vs_historical": rec004w_report.get(
                        "fresh_normal_validation_j0_delta_vs_historical"
                    ),
                    "fresh_length10_confirmation_j0_em": rec004w_report.get(
                        "fresh_length10_confirmation_j0_em"
                    ),
                    "fresh_length10_confirmation_j0_delta_vs_historical": rec004w_report.get(
                        "fresh_length10_confirmation_j0_delta_vs_historical"
                    ),
                    "o1_compatibility_all_length10": rec004w_report.get(
                        "o1_compatibility_all_length10"
                    ),
                    "rg3_recheck": rec004w_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004W was executed (an I03 normal-attention CVOF "
            "protection continuation pilot). No candidate training, candidate selection, "
            "child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004w_report.get("implementation_status") == "COMPLETE" else 1
    else:  # B-C005REC-004X
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004x.yaml")
        rec004x_config = _load_rec004x_config(config_path)
        if args.output_dir is not None:
            rec004x_config = dataclasses.replace(rec004x_config, output_dir=args.output_dir)
        rec004x_report = run_cvof_trust_region_pilot_task(rec004x_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004x_report.get("implementation_status"),
                    "radius_calibration": rec004x_report.get("radius_calibration"),
                    "trust_region_active": rec004x_report.get("trust_region_active"),
                    "intervention_optimizer_updates": rec004x_report.get(
                        "intervention_optimizer_updates"
                    ),
                    "new_candidate_training_updates": rec004x_report.get(
                        "new_candidate_training_updates"
                    ),
                    "result_label": rec004x_report.get("result_label"),
                    "ablation_tags": rec004x_report.get("ablation_tags"),
                    "metrics": rec004x_report.get("metrics"),
                    "rg3_recheck": rec004x_report.get("rg3_recheck"),
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004X was executed (an I03 pre-transition CVOF "
            "trust-region plasticity pilot). No candidate training, candidate selection, "
            "child bundle, RG3/REC-005, or sealed evaluation ran."
        )
        return 0 if rec004x_report.get("implementation_status") == "COMPLETE" else 1

    next_blocked = {
        "B-C005REC-002": "B-C005REC-003 onward",
        "B-C005REC-003": "B-C005REC-004 onward",
        "B-C005REC-004": "B-C005REC-005 onward",
    }[args.task]

    print(json.dumps(report["protocol"], indent=2))
    print(
        f"STOP: only {args.task} was executed. {next_blocked} and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    result = report["protocol"]["result"]
    return 0 if result == expected_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
