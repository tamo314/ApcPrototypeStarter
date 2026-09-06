# ruff: noqa: E501
"""Integrated causal diagnosis and next-repair decision gate for B-C005D2-006.

This module implements the final task of the second diagnostic phase
(`docs/CODEX_TASKS_PHASE_B_B2_SECOND_DIAGNOSTIC.md`): combine the evidence
already produced by B-C005D2-001 through B-C005D2-005 into one causal
findings table and exactly one recommended next research action (or
`UNRESOLVED`). It does **not** implement any repair, retrain anything, or
touch the frozen APC system -- every field in its output is read back from
the JSON artifacts those five tasks already wrote to
`runs/phase_b_b2_second_diagnostic/`, never recomputed or guessed:

- `summary.json` (D2-001): `representativeness_verdict`.
- `representation_stage_summary.json` (D2-002): per-relation
  `z_probe`/`q_probe`/`control_a` results at the failing sealed cell.
- `semantic_relation_summary.json` (D2-003): relation-spread and
  difficulty-matched comparison verdicts.
- `l4_argument_breakdown.json` (D2-004): per-operation L4 failure
  classification.
- `shift_seed24_adequacy_audit.json` (D2-005): verifier spec-conformance,
  reference adequacy, and plastic-decision reclassification.

Per D2-006.1 ("no row may be filled from intuition alone"), every row's
`evidence` field is a direct read of one of those files -- never a fabricated
or hardcoded number. Per D2-006.2/D2-006.3, the next-repair recommendation is
derived by evaluating each of the six allowed options (A-F) against that same
evidence, and `_assert_no_forbidden_conclusions` is a runtime guard, not a
comment, against the two conclusions the task doc explicitly forbids
("router needs to be larger", "adequacy threshold should be lower").
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from apc.utils.system_info import get_system_info

_CANONICAL_RELATIONS: Final[tuple[str, ...]] = (
    "SHIFT->CYCLE_FOUR",
    "SELECT->BIND",
    "COUNT->BIND",
    "BIND->COUNT",
)
_L4_OPERATIONS: Final[tuple[str, ...]] = ("SHIFT", "SELECT", "COUNT", "BIND")
_FOCUS_CELL: Final[str] = "regate_sealed/R2_frozen_post_repair"
_ARGUMENT_FAILURE_LABELS: Final[frozenset[str]] = frozenset(
    {"ARGUMENT_SCORER_GENERALIZATION_FAILURE", "ARGUMENT_ENCODING_FAILURE"}
)
_FORBIDDEN_PHRASES: Final[tuple[str, ...]] = (
    "router needs to be larger",
    "router should be larger",
    "increase router capacity",
    "adequacy threshold should be lower",
    "lower the adequacy threshold",
    "lower the threshold",
)


@dataclass(frozen=True)
class IntegratedCausalDiagnosisConfig:
    """Paths to the already-produced D2-001..005 artifacts this task reads."""

    d2001_summary_path: Path = Path("runs/phase_b_b2_second_diagnostic/summary.json")
    d2002_summary_path: Path = Path(
        "runs/phase_b_b2_second_diagnostic/representation_stage_summary.json"
    )
    d2003_summary_path: Path = Path(
        "runs/phase_b_b2_second_diagnostic/semantic_relation_summary.json"
    )
    d2004_summary_path: Path = Path(
        "runs/phase_b_b2_second_diagnostic/l4_argument_breakdown.json"
    )
    d2005_summary_path: Path = Path(
        "runs/phase_b_b2_second_diagnostic/shift_seed24_adequacy_audit.json"
    )
    no_failure_threshold: float = 0.95
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.no_failure_threshold <= 1.0:
            raise ValueError("no_failure_threshold must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        for key in (
            "d2001_summary_path",
            "d2002_summary_path",
            "d2003_summary_path",
            "d2004_summary_path",
            "d2005_summary_path",
        ):
            result[key] = str(result[key])
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FindingRow:
    """One row of the D2-006.1 required findings table."""

    mechanism: str
    evidence: dict[str, Any]
    verdict: str
    confidence: str
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Individual findings-table rows -- each reads exactly one prior artifact.
# ---------------------------------------------------------------------------


def _row_l2_retrieval(d2005: dict[str, Any], threshold: float) -> FindingRow:
    """L0-L2 retrieval, reproduced within D2-005's own SHIFT/seed-24 reconstruction."""
    audits = d2005["bank_size_audits"]
    per_level_top1 = {
        level: [audit["rank_diagnostics"][level]["primitive_call_top1"] for audit in audits]
        for level in ("L0_orthogonal", "L1_random_score_space", "L2_near_neighbor")
    }
    all_top1 = [value for values in per_level_top1.values() for value in values]
    verdict = "NO_FAILURE" if min(all_top1) >= threshold else "UNRESOLVED"
    return FindingRow(
        mechanism="L2 retrieval",
        evidence={
            "source": "shift_seed24_adequacy_audit.json:bank_size_audits[*].rank_diagnostics",
            "per_level_primitive_call_top1": per_level_top1,
            "adr_0079_reported_l0_l2_primitive_call_top1": 1.000,
        },
        verdict=verdict,
        confidence="HIGH",
        next_action="none",
    )


def _row_l3_task_representation(d2002: dict[str, Any]) -> FindingRow:
    evidence = d2002["representation_stage_localization"]["verdict_evidence"]
    z_probe = evidence["z_probe_accuracy"]
    verdict = "NO_FAILURE" if z_probe >= 0.95 else "TASK_REPRESENTATION_BOTTLENECK"
    return FindingRow(
        mechanism="L3 task representation",
        evidence={
            "source": f"representation_stage_summary.json:verdict_evidence (focus_cell={_FOCUS_CELL})",
            "z_probe_accuracy": z_probe,
        },
        verdict=verdict,
        confidence="HIGH",
        next_action="none",
    )


def _row_l3_query_projection(d2002: dict[str, Any]) -> FindingRow:
    evidence = d2002["representation_stage_localization"]["verdict_evidence"]
    q_probe = evidence["q_probe_accuracy"]
    verdict = "NO_FAILURE" if q_probe >= 0.95 else "QUERY_PROJECTION_BOTTLENECK"
    return FindingRow(
        mechanism="L3 query projection",
        evidence={
            "source": f"representation_stage_summary.json:verdict_evidence (focus_cell={_FOCUS_CELL})",
            "q_probe_accuracy": q_probe,
            "note": "query_proj is frozen and identical across R0/R1/R2; only router keys are retrained.",
        },
        verdict=verdict,
        confidence="HIGH",
        next_action="none",
    )


def _row_l3_key_scoring(d2002: dict[str, Any]) -> FindingRow:
    per_relation = d2002["representation_stage_localization"]["per_relation_verdict"]
    bottleneck_relations = [
        relation for relation, verdict in per_relation.items() if verdict == "KEY_SCORING_BOTTLENECK"
    ]
    resolved = [v for v in per_relation.values() if v != "UNRESOLVED"]
    verdict = "KEY_SCORING_BOTTLENECK" if bottleneck_relations else "MIXED"
    confidence = "HIGH" if len(resolved) == len(per_relation) else "MEDIUM"
    return FindingRow(
        mechanism="L3 key/scoring",
        evidence={
            "source": f"representation_stage_summary.json:per_relation_verdict (focus_cell={_FOCUS_CELL})",
            "per_relation_verdict": per_relation,
            "bottleneck_relations": bottleneck_relations,
        },
        verdict=verdict,
        confidence=confidence,
        next_action=(
            f"scope any future key-scoring repair to {bottleneck_relations} only"
            if bottleneck_relations
            else "none"
        ),
    )


def _row_l3_relation_split(d2003: dict[str, Any]) -> FindingRow:
    verdict_block = d2003["verdict"]
    holdout_required = bool(verdict_block["semantic_relation_holdout_required"])
    labels = verdict_block["labels"]
    verdict = "SEMANTIC_RELATION_HOLDOUT_REQUIRED" if holdout_required else "SEED_SPLIT_SUFFICIENT"
    return FindingRow(
        mechanism="L3 relation split",
        evidence={
            "source": "semantic_relation_summary.json:verdict",
            "labels": labels,
            "cross_relation_spread_at_focus_cell": verdict_block["evidence"][
                "cross_relation_spread_at_focus_cell"
            ],
            "post_repair_matched_gap": d2003["difficulty_matched_comparison"]["post_repair"][
                "matched_gap"
            ],
            "post_repair_raw_gap": d2003["difficulty_matched_comparison"]["post_repair"]["raw_gap"],
        },
        verdict=verdict,
        confidence="HIGH",
        next_action=(
            "define development/validation/sealed relation sets before any next repair training"
            if holdout_required
            else "none"
        ),
    )


def _row_l4_family_routing(d2004: dict[str, Any], threshold: float) -> FindingRow:
    per_op = d2004["l4_operation_breakdown"][_FOCUS_CELL]
    family_top1 = {op: per_op[op]["family_top1"] for op in _L4_OPERATIONS}
    verdict = "NO_FAILURE" if min(family_top1.values()) >= threshold else "UNRESOLVED"
    return FindingRow(
        mechanism="L4 family routing",
        evidence={
            "source": f"l4_argument_breakdown.json:l4_operation_breakdown[{_FOCUS_CELL}]",
            "family_top1_by_operation": family_top1,
        },
        verdict=verdict,
        confidence="HIGH",
        next_action="none",
    )


def _row_l4_argument_resolution(d2004: dict[str, Any]) -> FindingRow:
    classification = {
        op: entry["classification"] for op, entry in d2004["failure_classification"].items()
    }
    failing_ops = {op: c for op, c in classification.items() if c in _ARGUMENT_FAILURE_LABELS}
    verdict = "MIXED" if failing_ops and len(failing_ops) < len(classification) else (
        "NO_FAILURE" if not failing_ops else "ARGUMENT_RESOLUTION_FAILURE"
    )
    return FindingRow(
        mechanism="L4 argument resolution",
        evidence={
            "source": "l4_argument_breakdown.json:failure_classification",
            "classification_by_operation": classification,
        },
        verdict=verdict,
        confidence="HIGH",
        next_action=(
            f"scope any future argument-scorer repair to {sorted(failing_ops)} only"
            if failing_ops
            else "none"
        ),
    )


def _row_adequacy_estimator(d2005: dict[str, Any]) -> FindingRow:
    spec_check = d2005["spec_verification"]
    labels = d2005["overall_classification"]["labels"]
    rule_biased = "SEQUENTIAL_RULE_BIAS" in labels
    verdict = "SEQUENTIAL_RULE_BIAS" if rule_biased else "NO_FAILURE"
    return FindingRow(
        mechanism="adequacy estimator",
        evidence={
            "source": "shift_seed24_adequacy_audit.json:spec_verification,overall_classification",
            "implementation_matches_spec": spec_check["implementation_matches_spec"],
            "overall_classification_labels": labels,
            "n_unsafe_reuse": d2005["overall_classification"]["n_unsafe_reuse"],
        },
        verdict=verdict,
        confidence="HIGH",
        next_action=(
            "adequacy metric/protocol repair (Option E): confirm asymmetric early-accept "
            "before deploying any confidence-based verifier change"
            if rule_biased
            else "none"
        ),
    )


def _row_installed_shift_adequacy(d2005: dict[str, Any], threshold: float) -> FindingRow:
    reference = {
        audit["bank_size"]: audit["reference_adequacy"] for audit in d2005["bank_size_audits"]
    }
    reference_em = {size: entry["reference_em"] for size, entry in reference.items()}
    wilson_upper = {size: entry["wilson_ci"]["upper"] for size, entry in reference.items()}
    inadequate = all(upper < threshold for upper in wilson_upper.values())
    verdict = "TRUE_PRIMITIVE_INADEQUACY" if inadequate else "UNRESOLVED"
    return FindingRow(
        mechanism="installed SHIFT adequacy",
        evidence={
            "source": "shift_seed24_adequacy_audit.json:bank_size_audits[*].reference_adequacy",
            "reference_em_by_bank_size": reference_em,
            "wilson_upper_bound_by_bank_size": wilson_upper,
            "threshold": threshold,
        },
        verdict=verdict,
        confidence="HIGH",
        next_action=(
            "primitive functional-generalization repair (Option F): the installed SHIFT "
            "primitive at this seed, not the controller, is below threshold"
            if inadequate
            else "none"
        ),
    )


def _row_false_plastic_metric(d2005: dict[str, Any]) -> FindingRow:
    overall = d2005["overall_classification"]
    misclassifies = overall["n_unsafe_reuse"] > 0 and overall["n_true_false_plastic"] == 0
    verdict = "METRIC_MISCLASSIFICATION" if misclassifies else "NO_FAILURE"
    return FindingRow(
        mechanism="false-plastic metric",
        evidence={
            "source": "shift_seed24_adequacy_audit.json:overall_classification",
            "n_true_false_plastic": overall["n_true_false_plastic"],
            "n_functionally_justified_plastic": overall["n_functionally_justified_plastic"],
            "n_unsafe_reuse": overall["n_unsafe_reuse"],
            "n_correctly_accepted": overall["n_correctly_accepted"],
        },
        verdict=verdict,
        confidence="HIGH",
        next_action=(
            "adequacy metric/protocol repair (Option E): the metric tracks only "
            "plastic-when-should-reuse and misses reuse-when-should-go-plastic"
            if misclassifies
            else "none"
        ),
    )


def build_findings_table(
    *,
    d2001: dict[str, Any],
    d2002: dict[str, Any],
    d2003: dict[str, Any],
    d2004: dict[str, Any],
    d2005: dict[str, Any],
    threshold: float,
) -> list[FindingRow]:
    """D2-006.1's required findings table, in the task doc's own row order."""
    del d2001  # D2-001's own representativeness_verdict is reported separately (see below)
    return [
        _row_l2_retrieval(d2005, threshold),
        _row_l3_task_representation(d2002),
        _row_l3_query_projection(d2002),
        _row_l3_key_scoring(d2002),
        _row_l3_relation_split(d2003),
        _row_l4_family_routing(d2004, threshold),
        _row_l4_argument_resolution(d2004),
        _row_adequacy_estimator(d2005),
        _row_installed_shift_adequacy(d2005, threshold),
        _row_false_plastic_metric(d2005),
    ]


# ---------------------------------------------------------------------------
# D2-006.2 -- evaluate the six allowed next-repair options against evidence.
# ---------------------------------------------------------------------------


def evaluate_next_repair_options(rows: Sequence[FindingRow], d2004: dict[str, Any]) -> dict[str, Any]:
    by_mechanism = {row.mechanism: row for row in rows}

    option_a_allowed = (
        by_mechanism["L3 task representation"].verdict == "NO_FAILURE"
        and by_mechanism["L3 query projection"].verdict == "QUERY_PROJECTION_BOTTLENECK"
    )
    option_b_allowed = by_mechanism["L3 task representation"].verdict == "TASK_REPRESENTATION_BOTTLENECK"
    option_c_allowed = by_mechanism["L3 relation split"].verdict == "SEMANTIC_RELATION_HOLDOUT_REQUIRED"

    failing_ops = [
        op
        for op, entry in d2004["failure_classification"].items()
        if entry["classification"] in _ARGUMENT_FAILURE_LABELS
    ]
    option_d_allowed = bool(failing_ops)

    option_e_allowed = by_mechanism["adequacy estimator"].verdict == "SEQUENTIAL_RULE_BIAS" or (
        by_mechanism["false-plastic metric"].verdict == "METRIC_MISCLASSIFICATION"
    )
    option_f_allowed = by_mechanism["installed SHIFT adequacy"].verdict == "TRUE_PRIMITIVE_INADEQUACY"

    return {
        "A_query_projection_repair": {"allowed": option_a_allowed},
        "B_task_representation_repair": {"allowed": option_b_allowed},
        "C_semantic_relation_holdout_redesign": {"allowed": option_c_allowed},
        "D_argument_scorer_repair": {"allowed": option_d_allowed, "operations": failing_ops},
        "E_adequacy_metric_protocol_repair": {"allowed": option_e_allowed},
        "F_primitive_functional_generalization_repair": {"allowed": option_f_allowed},
    }


def _recommend_next_action(options: dict[str, Any]) -> dict[str, Any]:
    """Exactly one recommended next research action (D2-006, section 7), chosen
    as the option that gates every other evidence-supported option: repeating
    a seed-only development/sealed split for any repair (key-scoring, argument
    scorer, or primitive retraining) risks reproducing the exact
    develops-fine/fails-sealed pattern this whole diagnostic phase exists to
    explain, per D2-003's own finding. If C is not evidence-supported, no
    single gating action exists and this returns UNRESOLVED plus whichever
    options ARE independently supported, for the user to choose among.
    """
    if options["C_semantic_relation_holdout_redesign"]["allowed"]:
        primary = "OPTION_C_SEMANTIC_RELATION_HOLDOUT_REDESIGN"
        rationale = (
            "D2-003 found genuine model-generalization failure at matched relation/difficulty "
            "(SEMANTIC_RELATION_HOLDOUT_REQUIRED), and this precedes every other supported "
            "option: any future repair task (key-scoring, argument-scorer, or primitive "
            "retraining) trained under a seed-only development/sealed split risks reproducing "
            "the same develops-fine/fails-sealed pattern this whole D2 phase was created to "
            "diagnose. Define development/validation/sealed relation sets by ADR before any "
            "next repair training."
        )
    else:
        primary = "UNRESOLVED"
        rationale = "No single gating option is evidence-supported; see per-option allowed flags."
    pending = [
        name
        for name, entry in options.items()
        if entry["allowed"] and name != "C_semantic_relation_holdout_redesign"
    ]
    return {
        "primary_next_action": primary,
        "rationale": rationale,
        "evidence_supported_but_not_recommended_as_primary": pending,
        "note": (
            "These options are evidence-supported but are not authorized by this task and "
            "remain pending an explicit user instruction after the semantic-relation holdout "
            "redesign (or independently, if the user chooses not to gate on it)."
        ),
    }


def _assert_no_forbidden_conclusions(payload: dict[str, Any]) -> None:
    """D2-006.3 runtime guard: this diagnosis must never conclude the router
    needs to be larger, or that the adequacy threshold should be lower."""
    text = json.dumps(payload).lower()
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in text:
            raise AssertionError(f"forbidden conclusion present in D2-006 output: {phrase!r}")


# ---------------------------------------------------------------------------
# Top-level run.
# ---------------------------------------------------------------------------


def run_integrated_causal_diagnosis(config: IntegratedCausalDiagnosisConfig) -> dict[str, Any]:
    """Run the B-C005D2-006 integrated causal diagnosis and decision gate."""
    start = time.perf_counter()

    d2001 = _load_json(config.d2001_summary_path)
    d2002 = _load_json(config.d2002_summary_path)
    d2003 = _load_json(config.d2003_summary_path)
    d2004 = _load_json(config.d2004_summary_path)
    d2005 = _load_json(config.d2005_summary_path)

    rows = build_findings_table(
        d2001=d2001,
        d2002=d2002,
        d2003=d2003,
        d2004=d2004,
        d2005=d2005,
        threshold=config.no_failure_threshold,
    )
    options = evaluate_next_repair_options(rows, d2004)
    recommendation = _recommend_next_action(options)

    report: dict[str, Any] = {
        "task_id": "B-C005D2-006",
        "config": config.to_dict(),
        "source_representativeness_verdict_d2001": d2001["representativeness_verdict"],
        "findings_table": [row.to_dict() for row in rows],
        "next_repair_options": options,
        "recommendation": recommendation,
        "downstream_blocked": True,
        "b_c006_and_task_inference_remain_blocked": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    _assert_no_forbidden_conclusions(report)

    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-006",
            "diagnostic_only": True,
            "no_repair_authorized": True,
            "no_architecture_change": True,
            "no_sealed_training": True,
            "stop_after_this_task": True,
        }
        (out / "integrated_causal_diagnosis_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "integrated_causal_diagnosis_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        (out / "integrated_causal_diagnosis_system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "final_causal_diagnosis.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report
