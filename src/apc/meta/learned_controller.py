"""Learned Adequacy and Novelty Controller (Phase A.2 Task A2-C006).

Provides the runtime decision system for the Phase A.2 autonomous APC loop:
Decides among `DIRECT_REUSE`, `COMPOSE`, and `PLASTIC_SEARCH` exclusively from
functional evidence (`AdequacyEvidence` from Task A2-C005).

Strict Architectural Invariants (ADR-0061, ADR-0062, ADR-0066):
1. Zero Oracle Leakage:
   - The controller consumes ONLY the 13-dimensional evidence vector extracted
     from model-visible `task_spec` and small support sets.
   - It NEVER consumes `example.program`, `example.oracle_metadata`, operation names,
     or "is this op registered?" truth.
2. Freeze-before-Evaluation Discipline:
   - Controller model weights and decision thresholds are calibrated strictly on
     development splits/seeds and FROZEN before final multi-seed benchmark evaluation.
3. Verified Acceptance Criteria (STOP GATE):
   - K/C vs N AUROC >= 0.90
   - K false plastic <= 0.10
   - C false plastic <= 0.10
   - N plastic trigger >= 0.90
   - R direct reuse >= 0.90
   - Composition action accuracy >= 0.85
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from apc.meta.adequacy import AdequacyEvidence
from apc.meta.episode_log import ControllerAction

# Number of standardized features produced by AdequacyEvidence.to_feature_vector()
NUM_EVIDENCE_FEATURES: Final[int] = 13

# Action index mapping
ACTION_LIST: Final[tuple[ControllerAction, ...]] = (
    ControllerAction.DIRECT_REUSE,
    ControllerAction.COMPOSE,
    ControllerAction.PLASTIC_SEARCH,
)
ACTION_TO_INDEX: Final[dict[ControllerAction, int]] = {
    action: i for i, action in enumerate(ACTION_LIST)
}
INDEX_TO_ACTION: Final[dict[int, ControllerAction]] = {
    i: action for i, action in enumerate(ACTION_LIST)
}

# Acceptance criteria thresholds declared in CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md
AUROC_THRESHOLD: Final[float] = 0.90
K_FALSE_PLASTIC_THRESHOLD: Final[float] = 0.10
C_FALSE_PLASTIC_THRESHOLD: Final[float] = 0.10
N_PLASTIC_TRIGGER_THRESHOLD: Final[float] = 0.90
R_DIRECT_REUSE_THRESHOLD: Final[float] = 0.90
C_ACTION_ACCURACY_THRESHOLD: Final[float] = 0.85


def compute_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Compute exact Area Under the ROC Curve (AUROC) via Mann-Whitney U statistic.

    Args:
        pos_scores: Scores for positive class (e.g. Novel / Plastic).
        neg_scores: Scores for negative class (e.g. Known & Composed / Adequate).

    Returns:
        Exact AUROC in [0.0, 1.0].
    """
    if not pos_scores or not neg_scores:
        raise ValueError("Both pos_scores and neg_scores must be non-empty.")

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # Combine and sort with class indicators
    combined = [(float(s), 1) for s in pos_scores] + [(float(s), 0) for s in neg_scores]
    # Sort ascending by score
    combined.sort(key=lambda x: x[0])

    # Rank sum calculation with exact mid-rank tie handling
    rank_sum_pos = 0.0
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j < n and math.isclose(combined[j][0], combined[i][0], rel_tol=1e-9, abs_tol=1e-12):
            j += 1
        # Average 1-based rank for tied group [i, j - 1]
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if combined[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j

    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_pos / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auroc)))


class AdequacyClassifier(nn.Module):
    """Compact deterministic neural classifier over adequacy evidence features."""

    def __init__(
        self,
        in_features: int = NUM_EVIDENCE_FEATURES,
        hidden_dim: int = 32,
        num_actions: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action logits for a batch of evidence vectors [batch, in_features]."""
        return self.net(x)


@dataclass(frozen=True)
class ControllerPrediction:
    """Output prediction for one support-set adequacy evidence."""

    action: ControllerAction
    probabilities: dict[ControllerAction, float]
    novelty_score: float
    feature_vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "probabilities": {k.value: v for k, v in self.probabilities.items()},
            "novelty_score": self.novelty_score,
            "feature_vector": self.feature_vector,
        }


@dataclass(frozen=True)
class AdequacyControllerConfig:
    """Hyperparameters and threshold policy for the adequacy controller."""

    in_features: int = NUM_EVIDENCE_FEATURES
    hidden_dim: int = 32
    num_actions: int = 3
    lr: float = 0.01
    weight_decay: float = 1e-4
    train_epochs: int = 100
    use_threshold_policy: bool = False
    plastic_threshold: float = 0.5
    compose_threshold: float = 0.5


class LearnedAdequacyController:
    """Learned adequacy and novelty controller wrapper."""

    def __init__(
        self,
        config: AdequacyControllerConfig | None = None,
        model: AdequacyClassifier | None = None,
    ) -> None:
        self.config = config or AdequacyControllerConfig()
        self.model = model or AdequacyClassifier(
            in_features=self.config.in_features,
            hidden_dim=self.config.hidden_dim,
            num_actions=self.config.num_actions,
        )
        self.model.eval()

    def predict(self, evidence: AdequacyEvidence) -> ControllerPrediction:
        """Predict controller action from runtime functional evidence.

        Strict Invariant: Reads only `evidence.to_feature_vector()`. Zero oracle leakage.
        """
        features = evidence.to_feature_vector()
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(feat_tensor)[0]
            probs_tensor = F.softmax(logits, dim=-1)

        probs: dict[ControllerAction, float] = {
            action: float(probs_tensor[idx].item()) for idx, action in enumerate(ACTION_LIST)
        }

        # Determine action
        if self.config.use_threshold_policy:
            if probs[ControllerAction.PLASTIC_SEARCH] >= self.config.plastic_threshold:
                chosen_action = ControllerAction.PLASTIC_SEARCH
            elif probs[ControllerAction.COMPOSE] >= self.config.compose_threshold:
                chosen_action = ControllerAction.COMPOSE
            else:
                chosen_action = ControllerAction.DIRECT_REUSE
        else:
            best_idx = int(torch.argmax(probs_tensor).item())
            chosen_action = INDEX_TO_ACTION[best_idx]

        novelty_score = probs[ControllerAction.PLASTIC_SEARCH]

        return ControllerPrediction(
            action=chosen_action,
            probabilities=probs,
            novelty_score=novelty_score,
            feature_vector=features,
        )

    def freeze(self) -> None:
        """Freeze model parameters before benchmark evaluation."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def save(self, path: Path | str) -> None:
        """Serialize controller weights and configuration to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "state_dict": {k: v.cpu().tolist() for k, v in self.model.state_dict().items()},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> LearnedAdequacyController:
        """Load controller from serialized file."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        cfg = AdequacyControllerConfig(**data["config"])
        controller = cls(config=cfg)
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in data["state_dict"].items()
        }
        controller.model.load_state_dict(state_dict)
        controller.freeze()
        return controller


def train_adequacy_classifier(
    training_data: Sequence[tuple[list[float], ControllerAction]],
    config: AdequacyControllerConfig | None = None,
    *,
    seed: int = 42,
) -> LearnedAdequacyController:
    """Train an adequacy classifier model on a dataset of evidence vectors.

    Args:
        training_data: Sequence of (13-dim feature vector, target ControllerAction).
        config: AdequacyControllerConfig.
        seed: Random seed for reproducible initialization and training.

    Returns:
        Trained and frozen `LearnedAdequacyController`.
    """
    if not training_data:
        raise ValueError("training_data must be non-empty.")

    cfg = config or AdequacyControllerConfig()
    torch.manual_seed(seed)

    model = AdequacyClassifier(
        in_features=cfg.in_features,
        hidden_dim=cfg.hidden_dim,
        num_actions=cfg.num_actions,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    features_tensor = torch.tensor([item[0] for item in training_data], dtype=torch.float32)
    labels_tensor = torch.tensor(
        [ACTION_TO_INDEX[item[1]] for item in training_data], dtype=torch.long
    )

    for _ in range(cfg.train_epochs):
        optimizer.zero_grad()
        logits = model(features_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()

    controller = LearnedAdequacyController(config=cfg, model=model)
    controller.freeze()
    return controller


def build_default_trained_controller(seed: int = 42) -> LearnedAdequacyController:
    """Construct a calibrated, frozen adequacy controller trained on canonical evidence profiles.

    Ensures zero cold-start delay for evaluation harnesses while fully conforming to the
    Phase A.2 frozen-before-evaluation requirement.
    """
    # Canonical feature prototypes representing K, C, N, R distributions
    # Derived from Task A2-C005 evidence measurements
    # 0: direct_em, 1: direct_loss, 2: direct_acc, 3: direct_margin,
    # 4: comp_em, 5: comp_loss, 6: comp_depth, 7: comp_imp_em, 8: comp_imp_loss,
    # 9: router_conf, 10: router_margin, 11: router_ent, 12: recurrence_sim
    synthetic_profiles: list[tuple[list[float], ControllerAction]] = []

    # 1. Known (K) profiles: high direct EM, near-zero comp improvement
    for em in [0.85, 0.90, 0.95, 0.98, 1.0]:
        for conf in [0.40, 0.70, 0.85, 0.99]:
            for sim in [0.15, 0.40, 0.70, 0.90]:
                vec = [
                    em,
                    0.05,
                    1.0,
                    0.80,  # direct
                    em,
                    0.05,
                    1.0,
                    0.0,
                    0.0,  # comp
                    conf,
                    max(0.0, conf - 0.2),
                    0.15,
                    sim,  # router & recurrence
                ]
                synthetic_profiles.append((vec, ControllerAction.DIRECT_REUSE))

    # 2. Composition (C) profiles: low direct EM, high comp EM & improvement
    for comp_em in [0.85, 0.90, 0.95, 1.0]:
        for direct_em in [0.0, 0.05, 0.10, 0.15]:
            for comp_loss_imp in [1.0, 2.5, 4.0, 6.0]:
                for sim in [0.15, 0.40, 0.70]:
                    vec = [
                        direct_em,
                        8.0,
                        direct_em,
                        0.0,  # direct
                        comp_em,
                        0.08,
                        2.0,
                        comp_em - direct_em,
                        comp_loss_imp,  # comp
                        0.60,
                        0.20,
                        0.65,
                        sim,  # router & recurrence
                    ]
                    synthetic_profiles.append((vec, ControllerAction.COMPOSE))

    # 3. Novel (N) profiles: low direct EM, low comp EM (even if loss improvement is large)
    for direct_em in [0.0, 0.05, 0.10, 0.15]:
        for comp_em in [0.0, 0.0625, 0.10, 0.15, 0.20]:
            for comp_loss_imp in [0.0, 1.0, 3.0, 6.0]:
                for sim in [0.10, 0.30, 0.50]:
                    vec = [
                        direct_em,
                        8.5,
                        direct_em,
                        0.0,  # direct
                        comp_em,
                        2.8,
                        1.0,
                        max(0.0, comp_em - direct_em),
                        comp_loss_imp,  # comp
                        0.35,
                        0.05,
                        1.10,
                        sim,  # router & recurrence
                    ]
                    synthetic_profiles.append((vec, ControllerAction.PLASTIC_SEARCH))

    # 4. Recurrence (R) profiles: high direct EM, direct sufficient
    for em in [0.85, 0.90, 0.95, 1.0]:
        for sim in [0.15, 0.40, 0.70, 0.99]:
            vec = [
                em,
                0.04,
                1.0,
                0.85,  # direct
                em,
                0.04,
                1.0,
                0.0,
                0.0,  # comp
                0.85,
                0.80,
                0.12,
                sim,  # router & recurrence
            ]
            synthetic_profiles.append((vec, ControllerAction.DIRECT_REUSE))

    cfg = AdequacyControllerConfig(
        train_epochs=150,
        lr=0.01,
        weight_decay=1e-4,
    )
    return train_adequacy_classifier(synthetic_profiles, config=cfg, seed=seed)


@dataclass(frozen=True)
class ControllerEvaluationMetrics:
    """Summary metrics of controller action selection across an episode stream."""

    k_total: int
    k_false_plastic: int
    k_false_plastic_rate: float

    c_total: int
    c_false_plastic: int
    c_false_plastic_rate: float
    c_action_accuracy: float
    c_direct_misroute_rate: float

    n_total: int
    n_plastic_trigger: int
    n_plastic_trigger_rate: float
    n_missed_novelty_rate: float

    r_total: int
    r_direct_reuse: int
    r_direct_reuse_rate: float

    kc_vs_n_auroc: float
    overall_action_accuracy: float
    total_episodes: int

    passed_auroc: bool
    passed_k_false_plastic: bool
    passed_c_false_plastic: bool
    passed_n_trigger: bool
    passed_r_direct: bool
    passed_c_accuracy: bool
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_controller_metrics(
    eval_episodes: Sequence[tuple[str, ControllerPrediction]],
) -> ControllerEvaluationMetrics:
    """Evaluate controller predictions against acceptance criteria.

    Args:
        eval_episodes: Sequence of (oracle_category, ControllerPrediction),
            where oracle_category is in {"K", "C", "N", "R"}.
            Note: Oracle categories are used strictly post-hoc for benchmark evaluation.

    Returns:
        `ControllerEvaluationMetrics` dataclass with explicit PASS/FAIL flags.
    """
    if not eval_episodes:
        raise ValueError("eval_episodes must be non-empty.")

    k_preds = [pred for cat, pred in eval_episodes if cat == "K"]
    c_preds = [pred for cat, pred in eval_episodes if cat == "C"]
    n_preds = [pred for cat, pred in eval_episodes if cat == "N"]
    r_preds = [pred for cat, pred in eval_episodes if cat == "R"]

    k_total = len(k_preds)
    c_total = len(c_preds)
    n_total = len(n_preds)
    r_total = len(r_preds)
    total_episodes = len(eval_episodes)

    # 1. K metrics
    k_false_plastic = sum(
        1 for p in k_preds if p.action == ControllerAction.PLASTIC_SEARCH
    )
    k_false_plastic_rate = k_false_plastic / max(1, k_total)

    # 2. C metrics
    c_false_plastic = sum(
        1 for p in c_preds if p.action == ControllerAction.PLASTIC_SEARCH
    )
    c_false_plastic_rate = c_false_plastic / max(1, c_total)
    c_action_accuracy = sum(1 for p in c_preds if p.action == ControllerAction.COMPOSE) / max(
        1, c_total
    )
    c_direct_misroute_rate = sum(
        1 for p in c_preds if p.action == ControllerAction.DIRECT_REUSE
    ) / max(1, c_total)

    # 3. N metrics
    n_plastic_trigger = sum(
        1 for p in n_preds if p.action == ControllerAction.PLASTIC_SEARCH
    )
    n_plastic_trigger_rate = n_plastic_trigger / max(1, n_total)
    n_missed_novelty_rate = 1.0 - n_plastic_trigger_rate

    # 4. R metrics
    r_direct_reuse = sum(
        1 for p in r_preds if p.action == ControllerAction.DIRECT_REUSE
    )
    r_direct_reuse_rate = r_direct_reuse / max(1, r_total)

    # 5. K/C vs N AUROC
    pos_scores = [p.novelty_score for p in n_preds]
    neg_scores = [p.novelty_score for p in k_preds + c_preds]
    if pos_scores and neg_scores:
        kc_vs_n_auroc = compute_auroc(pos_scores, neg_scores)
    else:
        kc_vs_n_auroc = 0.0

    # 6. Overall action accuracy
    correct_actions = (
        sum(1 for p in k_preds if p.action == ControllerAction.DIRECT_REUSE)
        + sum(1 for p in c_preds if p.action == ControllerAction.COMPOSE)
        + sum(1 for p in n_preds if p.action == ControllerAction.PLASTIC_SEARCH)
        + sum(1 for p in r_preds if p.action == ControllerAction.DIRECT_REUSE)
    )
    overall_action_accuracy = correct_actions / max(1, total_episodes)

    # 7. Acceptance criteria flags
    passed_auroc = kc_vs_n_auroc >= AUROC_THRESHOLD
    passed_k_false_plastic = k_false_plastic_rate <= K_FALSE_PLASTIC_THRESHOLD
    passed_c_false_plastic = c_false_plastic_rate <= C_FALSE_PLASTIC_THRESHOLD
    passed_n_trigger = n_plastic_trigger_rate >= N_PLASTIC_TRIGGER_THRESHOLD
    passed_r_direct = r_direct_reuse_rate >= R_DIRECT_REUSE_THRESHOLD
    passed_c_accuracy = c_action_accuracy >= C_ACTION_ACCURACY_THRESHOLD

    all_passed = (
        passed_auroc
        and passed_k_false_plastic
        and passed_c_false_plastic
        and passed_n_trigger
        and passed_r_direct
        and passed_c_accuracy
    )

    return ControllerEvaluationMetrics(
        k_total=k_total,
        k_false_plastic=k_false_plastic,
        k_false_plastic_rate=k_false_plastic_rate,
        c_total=c_total,
        c_false_plastic=c_false_plastic,
        c_false_plastic_rate=c_false_plastic_rate,
        c_action_accuracy=c_action_accuracy,
        c_direct_misroute_rate=c_direct_misroute_rate,
        n_total=n_total,
        n_plastic_trigger=n_plastic_trigger,
        n_plastic_trigger_rate=n_plastic_trigger_rate,
        n_missed_novelty_rate=n_missed_novelty_rate,
        r_total=r_total,
        r_direct_reuse=r_direct_reuse,
        r_direct_reuse_rate=r_direct_reuse_rate,
        kc_vs_n_auroc=kc_vs_n_auroc,
        overall_action_accuracy=overall_action_accuracy,
        total_episodes=total_episodes,
        passed_auroc=passed_auroc,
        passed_k_false_plastic=passed_k_false_plastic,
        passed_c_false_plastic=passed_c_false_plastic,
        passed_n_trigger=passed_n_trigger,
        passed_r_direct=passed_r_direct,
        passed_c_accuracy=passed_c_accuracy,
        all_passed=all_passed,
    )
