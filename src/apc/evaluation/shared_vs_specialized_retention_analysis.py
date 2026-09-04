"""Shared-vs-specialized retention analysis (Phase A.1 diagnostic Task
A1-R005E-S003, `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

## What this measures

Task A1-R005E-S002 trained exactly one shared task-blind content encoder across
all four parameterized operations (SHIFT, SELECT, COUNT, BIND). Task
A1-R005E-006A trained one task-blind content encoder per operation (specialized).

This analysis quantifies how much of E-006A's representation-accessibility gain
survives when all four operations share one content encoder:

    R_shared_correct(op) = SharedCorrect(op) / max(eps, E006A_Correct(op))
    R_shared_gap(op) = SharedGap(op) / max(eps, E006A_Gap(op))

Per `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` section 4:

  - `R_shared >= 0.90`: Strong shared support (shared representation retains
    most specialized benefit)
  - `0.70 <= R_shared < 0.90`: Partial sharing / mixed
  - `R_shared < 0.70`: Strong specialization dependence

Per `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` section 6, strong
shared-representation support requires per operation:
  - `R_shared_correct >= 0.90`
  - `R_shared_gap >= 0.90`
  - strong argument causality (effectful Wrong <= 0.30, effect rate ~ 1.0)
  - no severe seed collapse
and globally:
  - at least 3/4 operations satisfy those ratios,
  - the remaining failure is mechanistically interpretable.

It also compares the shared representation against historical controls:
  - E-005: Frozen task-blind core + compact cross-position operator (`C00`)
  - E-004: Frozen task-blind core + high-capacity operator (`C01`)
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES

__all__ = [
    "STRONG_SHARED_THRESHOLD",
    "MIXED_SHARED_THRESHOLD",
    "R_SHARED_EPS",
    "DEFAULT_S002_SUMMARY_PATH",
    "DEFAULT_E006A_SUMMARY_PATH",
    "DEFAULT_E005_SUMMARY_PATH",
    "DEFAULT_E004_SUMMARY_PATH",
    "RetentionBand",
    "RetentionConfig",
    "retention_config_from_dict",
    "calculate_retention",
    "classify_retention_band",
    "OperationRetentionSummary",
    "SharedVsSpecializedRetentionReport",
    "load_summary",
    "analyze_shared_vs_specialized_retention",
    "format_evidence_table_markdown",
]

# Classification thresholds from SHARED_QUERYABLE_REPRESENTATION.md section 4
# and EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md section 6.
STRONG_SHARED_THRESHOLD = 0.90
MIXED_SHARED_THRESHOLD = 0.70
R_SHARED_EPS = 1e-6

# Minimum gate seeds required for decision evidence
MIN_GATE_SEEDS = 5

# Canonical artifact paths
DEFAULT_S002_SUMMARY_PATH = Path(
    "runs/phase_a1_shared_encoder_mixed_operation_training_gate/summary.json"
)
DEFAULT_E006A_SUMMARY_PATH = Path(
    "runs/phase_a1_joint_representation_compact_operator_probe/summary.json"
)
DEFAULT_E005_SUMMARY_PATH = Path(
    "runs/phase_a1_compact_cross_position_operator_probe/summary.json"
)
DEFAULT_E004_SUMMARY_PATH = Path(
    "runs/phase_a1_frozen_high_capacity_operator_benchmark/summary.json"
)

RetentionBand = Literal["strong", "mixed", "specialized"]


def calculate_retention(
    shared_val: float, specialized_val: float, *, eps: float = R_SHARED_EPS
) -> float:
    """`R_shared = shared_val / max(eps, specialized_val)`."""
    return float(shared_val / max(eps, specialized_val))


def classify_retention_band(ratio: float) -> RetentionBand:
    """Classify retention ratio into qualitative band."""
    if ratio >= STRONG_SHARED_THRESHOLD:
        return "strong"
    if ratio >= MIXED_SHARED_THRESHOLD:
        return "mixed"
    return "specialized"


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a gate summary JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Summary file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_op_data(summary: dict[str, Any], operation: str) -> dict[str, Any]:
    per_op = summary.get("per_operation_summary", {})
    if operation not in per_op:
        raise KeyError(f"Operation {operation!r} not found in summary per_operation_summary")
    return dict(per_op[operation])


@dataclass(frozen=True)
class RetentionConfig:
    """Configuration for retention analysis."""

    s002_summary_path: str = str(DEFAULT_S002_SUMMARY_PATH)
    e006a_summary_path: str = str(DEFAULT_E006A_SUMMARY_PATH)
    e005_summary_path: str = str(DEFAULT_E005_SUMMARY_PATH)
    e004_summary_path: str = str(DEFAULT_E004_SUMMARY_PATH)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    eps: float = R_SHARED_EPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "s002_summary_path": self.s002_summary_path,
            "e006a_summary_path": self.e006a_summary_path,
            "e005_summary_path": self.e005_summary_path,
            "e004_summary_path": self.e004_summary_path,
            "operation_names": list(self.operation_names),
            "eps": self.eps,
        }


def retention_config_from_dict(raw: dict[str, Any]) -> RetentionConfig:
    """Parse retention config from dictionary."""
    defaults = RetentionConfig()
    return RetentionConfig(
        s002_summary_path=str(raw.get("s002_summary_path", defaults.s002_summary_path)),
        e006a_summary_path=str(raw.get("e006a_summary_path", defaults.e006a_summary_path)),
        e005_summary_path=str(raw.get("e005_summary_path", defaults.e005_summary_path)),
        e004_summary_path=str(raw.get("e004_summary_path", defaults.e004_summary_path)),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        eps=float(raw.get("eps", defaults.eps)),
    )


@dataclass(frozen=True)
class OperationRetentionSummary:
    """Per-operation retention and comparison metrics."""

    operation: str

    # S002 shared encoder metrics
    s002_correct_exact_match: float
    s002_stdev_correct_exact_match: float
    s002_min_correct_exact_match: float
    s002_max_correct_exact_match: float
    s002_correct_token_accuracy: float
    s002_wrong_exact_match: float
    s002_none_exact_match: float
    s002_causal_gap: float
    s002_token_accuracy_causal_gap: float
    s002_task_blind_invariant_passed: bool
    s002_argument_effect_rate: float
    s002_operator_param_count: int

    # E-006A specialized encoder metrics
    e006a_correct_exact_match: float
    e006a_stdev_correct_exact_match: float
    e006a_correct_token_accuracy: float
    e006a_wrong_exact_match: float
    e006a_none_exact_match: float
    e006a_causal_gap: float
    e006a_token_accuracy_causal_gap: float

    # Historical baseline controls
    e005_c00_correct_exact_match: float
    e005_c00_causal_gap: float
    e004_c01_correct_exact_match: float
    e004_c01_causal_gap: float

    # Retention scores (S002 vs E-006A)
    r_shared_correct: float
    r_shared_gap: float
    r_shared_token_accuracy: float
    r_shared_token_gap: float

    # Bands
    correct_retention_band: RetentionBand
    gap_retention_band: RetentionBand
    overall_retention_band: RetentionBand

    # Historical comparisons
    s002_vs_e005_correct_ratio: float
    s002_vs_e005_gap_ratio: float
    s002_vs_e004_correct_percentage: float
    s002_vs_e004_gap_percentage: float

    # Seed variance comparison (E-006A stdev / S002 stdev)
    variance_reduction_ratio: float

    # Section 6 criteria evaluation
    r_shared_correct_passed: bool
    r_shared_gap_passed: bool
    strong_argument_causality_passed: bool
    no_severe_seed_collapse_passed: bool
    passed_strong_shared_support: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SharedVsSpecializedRetentionReport:
    """Full retention report across operations."""

    config: RetentionConfig
    per_operation_summary: dict[str, OperationRetentionSummary]
    s002_seeds: list[int]
    e006a_seeds: list[int]
    e005_seeds: list[int]
    e004_seeds: list[int]
    num_operations_strong: int
    num_operations_mixed: int
    num_operations_specialized: int
    global_strong_support: bool
    meets_seed_policy: bool
    evidence_table_markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "per_operation_summary": {
                op: s.to_dict() for op, s in self.per_operation_summary.items()
            },
            "s002_seeds": self.s002_seeds,
            "e006a_seeds": self.e006a_seeds,
            "e005_seeds": self.e005_seeds,
            "e004_seeds": self.e004_seeds,
            "num_operations_strong": self.num_operations_strong,
            "num_operations_mixed": self.num_operations_mixed,
            "num_operations_specialized": self.num_operations_specialized,
            "global_strong_support": self.global_strong_support,
            "meets_seed_policy": self.meets_seed_policy,
            "evidence_table_markdown": self.evidence_table_markdown,
        }


def format_evidence_table_markdown(
    per_operation: dict[str, OperationRetentionSummary],
) -> str:
    """Generate Markdown evidence table per A1-R005E-S003 acceptance criteria."""
    header = (
        "| Operation | S002 Correct | E-006A Correct | `R_shared_correct` | S002 Gap "
        "| E-006A Gap | `R_shared_gap` | vs E-005 (C00) | vs E-004 (C01) | Band | Support |"
    )
    lines = [
        header,
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op, s in per_operation.items():
        support_str = "PASS" if s.passed_strong_shared_support else "FAIL"
        r_correct_pct = f"{s.r_shared_correct * 100:.2f}%"
        r_gap_pct = f"{s.r_shared_gap * 100:.2f}%"
        vs_e005 = f"{s.s002_vs_e005_correct_ratio:.2f}x"
        vs_e004 = f"{s.s002_vs_e004_correct_percentage:.1f}%"
        row = (
            f"| {op} | {s.s002_correct_exact_match:.4f} | {s.e006a_correct_exact_match:.4f} | "
            f"**{r_correct_pct}** | {s.s002_causal_gap:.4f} | {s.e006a_causal_gap:.4f} | "
            f"**{r_gap_pct}** | {vs_e005} | {vs_e004} | {s.overall_retention_band} | "
            f"**{support_str}** |"
        )
        lines.append(row)
    return "\n".join(lines)


def analyze_shared_vs_specialized_retention(
    config: RetentionConfig,
) -> SharedVsSpecializedRetentionReport:
    """Perform retention analysis across all configured operations."""
    s002 = load_summary(config.s002_summary_path)
    e006a = load_summary(config.e006a_summary_path)
    e005 = load_summary(config.e005_summary_path)
    e004 = load_summary(config.e004_summary_path)

    s002_seeds = list(s002.get("seeds", []))
    e006a_seeds = list(e006a.get("seeds", []))
    e005_seeds = list(e005.get("seeds", []))
    e004_seeds = list(e004.get("seeds", []))

    meets_seed_policy = (
        len(s002_seeds) >= MIN_GATE_SEEDS
        and len(e006a_seeds) >= MIN_GATE_SEEDS
        and len(e005_seeds) >= MIN_GATE_SEEDS
        and len(e004_seeds) >= MIN_GATE_SEEDS
    )

    per_operation: dict[str, OperationRetentionSummary] = {}
    num_strong = 0
    num_mixed = 0
    num_specialized = 0

    for op in config.operation_names:
        s002_op = _extract_op_data(s002, op)
        e006a_op = _extract_op_data(e006a, op)
        e005_op = _extract_op_data(e005, op)
        e004_op = _extract_op_data(e004, op)

        # S002 values
        s_correct = float(s002_op["mean_correct_exact_match"])
        s_stdev = float(s002_op.get("stdev_correct_exact_match", 0.0))
        s_min = float(s002_op.get("min_correct_exact_match", s_correct))
        s_max = float(s002_op.get("max_correct_exact_match", s_correct))
        s_tok_acc = float(s002_op.get("mean_correct_token_accuracy", s_correct))
        s_wrong = float(s002_op["mean_effectful_wrong_argument_exact_match"])
        s_none = float(s002_op["mean_none_exact_match"])
        s_gap = float(s002_op["mean_exact_match_causal_gap"])
        s_tok_gap = float(s002_op.get("mean_token_accuracy_causal_gap", s_gap))
        s_task_blind_pass = bool(s002_op.get("task_blind_invariant_passed", True))
        s_effect_rate = float(s002_op.get("mean_argument_effect_rate", 1.0))
        s_op_params = int(s002_op.get("operator_param_count", 0))

        # E-006A values
        e6_correct = float(e006a_op["mean_correct_exact_match"])
        e6_stdev = float(e006a_op.get("stdev_correct_exact_match", 0.0))
        e6_tok_acc = float(e006a_op.get("mean_correct_token_accuracy", e6_correct))
        e6_wrong = float(e006a_op["mean_effectful_wrong_argument_exact_match"])
        e6_none = float(e006a_op["mean_none_exact_match"])
        e6_gap = float(e006a_op["mean_exact_match_causal_gap"])
        e6_tok_gap = float(e006a_op.get("mean_token_accuracy_causal_gap", e6_gap))

        # E-005 (C00)
        e5_c00_correct = float(e005_op["mean_correct_exact_match"])
        e5_c00_gap = float(e005_op["mean_exact_match_causal_gap"])

        # E-004 (C01)
        e4_c01_correct = float(e004_op["mean_correct_exact_match"])
        e4_c01_gap = float(e004_op["mean_exact_match_causal_gap"])

        # Retention calculations
        r_correct = calculate_retention(s_correct, e6_correct, eps=config.eps)
        r_gap = calculate_retention(s_gap, e6_gap, eps=config.eps)
        r_tok_acc = calculate_retention(s_tok_acc, e6_tok_acc, eps=config.eps)
        r_tok_gap = calculate_retention(s_tok_gap, e6_tok_gap, eps=config.eps)

        band_correct = classify_retention_band(r_correct)
        band_gap = classify_retention_band(r_gap)
        # Overall band is the conservative minimum between correct and gap bands
        if band_correct == "strong" and band_gap == "strong":
            overall_band: RetentionBand = "strong"
            num_strong += 1
        elif band_correct == "specialized" or band_gap == "specialized":
            overall_band = "specialized"
            num_specialized += 1
        else:
            overall_band = "mixed"
            num_mixed += 1

        # Historical ratios
        vs_e005_correct = calculate_retention(s_correct, e5_c00_correct, eps=config.eps)
        vs_e005_gap = calculate_retention(s_gap, e5_c00_gap, eps=config.eps)
        vs_e004_correct_pct = calculate_retention(s_correct, e4_c01_correct, eps=config.eps) * 100.0
        vs_e004_gap_pct = calculate_retention(s_gap, e4_c01_gap, eps=config.eps) * 100.0

        # Variance reduction ratio (ratio > 1 means S002 has lower variance than E-006A)
        var_red = float(e6_stdev / max(config.eps, s_stdev))

        # Section 6 criteria checks
        r_correct_pass = r_correct >= STRONG_SHARED_THRESHOLD
        r_gap_pass = r_gap >= STRONG_SHARED_THRESHOLD
        causality_pass = s_wrong <= 0.30 and s_effect_rate >= 0.99
        no_collapse_pass = s_min >= 0.10
        support_pass = (
            r_correct_pass
            and r_gap_pass
            and causality_pass
            and no_collapse_pass
            and s_task_blind_pass
        )

        per_operation[op] = OperationRetentionSummary(
            operation=op,
            s002_correct_exact_match=s_correct,
            s002_stdev_correct_exact_match=s_stdev,
            s002_min_correct_exact_match=s_min,
            s002_max_correct_exact_match=s_max,
            s002_correct_token_accuracy=s_tok_acc,
            s002_wrong_exact_match=s_wrong,
            s002_none_exact_match=s_none,
            s002_causal_gap=s_gap,
            s002_token_accuracy_causal_gap=s_tok_gap,
            s002_task_blind_invariant_passed=s_task_blind_pass,
            s002_argument_effect_rate=s_effect_rate,
            s002_operator_param_count=s_op_params,
            e006a_correct_exact_match=e6_correct,
            e006a_stdev_correct_exact_match=e6_stdev,
            e006a_correct_token_accuracy=e6_tok_acc,
            e006a_wrong_exact_match=e6_wrong,
            e006a_none_exact_match=e6_none,
            e006a_causal_gap=e6_gap,
            e006a_token_accuracy_causal_gap=e6_tok_gap,
            e005_c00_correct_exact_match=e5_c00_correct,
            e005_c00_causal_gap=e5_c00_gap,
            e004_c01_correct_exact_match=e4_c01_correct,
            e004_c01_causal_gap=e4_c01_gap,
            r_shared_correct=r_correct,
            r_shared_gap=r_gap,
            r_shared_token_accuracy=r_tok_acc,
            r_shared_token_gap=r_tok_gap,
            correct_retention_band=band_correct,
            gap_retention_band=band_gap,
            overall_retention_band=overall_band,
            s002_vs_e005_correct_ratio=vs_e005_correct,
            s002_vs_e005_gap_ratio=vs_e005_gap,
            s002_vs_e004_correct_percentage=vs_e004_correct_pct,
            s002_vs_e004_gap_percentage=vs_e004_gap_pct,
            variance_reduction_ratio=var_red,
            r_shared_correct_passed=r_correct_pass,
            r_shared_gap_passed=r_gap_pass,
            strong_argument_causality_passed=causality_pass,
            no_severe_seed_collapse_passed=no_collapse_pass,
            passed_strong_shared_support=support_pass,
        )

    # Global strong support: at least 3/4 operations satisfy strong support
    num_ops = len(config.operation_names)
    threshold_ops = 3 if num_ops >= 4 else num_ops
    global_strong_support = num_strong >= threshold_ops

    table_md = format_evidence_table_markdown(per_operation)

    return SharedVsSpecializedRetentionReport(
        config=config,
        per_operation_summary=per_operation,
        s002_seeds=s002_seeds,
        e006a_seeds=e006a_seeds,
        e005_seeds=e005_seeds,
        e004_seeds=e004_seeds,
        num_operations_strong=num_strong,
        num_operations_mixed=num_mixed,
        num_operations_specialized=num_specialized,
        global_strong_support=global_strong_support,
        meets_seed_policy=meets_seed_policy,
        evidence_table_markdown=table_md,
    )
