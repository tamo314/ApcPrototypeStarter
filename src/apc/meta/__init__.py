"""Novelty estimation, adequacy evidence, and controller policies."""

from apc.meta.adequacy import (
    AdequacyEvidence,
    AdequacyEvidenceConfig,
    compute_adequacy_evidence,
    extract_task_representations,
)
from apc.meta.episode_log import (
    ControllerAction,
    EpisodeLogger,
    EpisodeRecord,
)
from apc.meta.learned_controller import (
    AdequacyClassifier,
    AdequacyControllerConfig,
    ControllerEvaluationMetrics,
    ControllerPrediction,
    LearnedAdequacyController,
    build_default_trained_controller,
    compute_auroc,
    evaluate_controller_metrics,
    train_adequacy_classifier,
)

__all__ = [
    "AdequacyClassifier",
    "AdequacyControllerConfig",
    "AdequacyEvidence",
    "AdequacyEvidenceConfig",
    "ControllerAction",
    "ControllerEvaluationMetrics",
    "ControllerPrediction",
    "EpisodeLogger",
    "EpisodeRecord",
    "LearnedAdequacyController",
    "build_default_trained_controller",
    "compute_adequacy_evidence",
    "compute_auroc",
    "evaluate_controller_metrics",
    "extract_task_representations",
    "train_adequacy_classifier",
]
