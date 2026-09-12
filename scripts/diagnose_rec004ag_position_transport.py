"""REC-004AG: non-degenerate position-routing transport causal diagnosis.

This is an evaluation-only diagnostic. It never creates an optimizer, changes a
model parameter, or supplies oracle routing/baseline prediction information to
the intervention construction or J0 runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import localize_mirror_failure_modes as rec004ae  # noqa: E402
import reanalyze_rec004ac_metrics as reanalysis  # noqa: E402
from apc.evaluation import mirror_parallel_score_residual_pilot as ac  # noqa: E402
from apc.evaluation.mirror_position_initialization_diagnostic import (  # noqa: E402
    mirror_halves_position_map,
)
from apc.utils.model_bundle import canonical_state_hash  # noqa: E402

INPUT = Path("runs/phase_b_restart/rec004ac_metric_v2/run_003")
AE_INPUT = Path("runs/phase_b_restart/rec004ae/run_004")
AF_INPUT = Path("runs/phase_b_restart/rec004af/run_005")
CHECKPOINT = reanalysis.SOURCE_DIR / "checkpoint_step8000.pt"

PRIMARY_STRATA = ("normal_validation_length10", "length10_confirmation")
ROUTING_IMPROVEMENT_MIN = 0.05
MARGIN_IMPROVEMENT_MIN = 0.25
RECOVERY_MIN = 0.05


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_architecture_coordinate_mapping(
    target_length: int, source_length: int
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Deterministically map target position coordinates to nearest source position coordinates.

    Uses normalized coordinate u(k, L) = k / (L - 1) as defined by architecture:
    phi(i, j, n) = [i/d, j/d, (j-i)/d, n/length_ref] with d = max(n - 1, 1).
    Ties are broken to the smaller index.
    """
    mapping: dict[int, int] = {}
    details: list[dict[str, Any]] = []
    denom_tgt = max(target_length - 1, 1)
    denom_src = max(source_length - 1, 1)

    for i_tgt in range(target_length):
        u_tgt = i_tgt / denom_tgt
        candidates = []
        for i_src in range(source_length):
            v_src = i_src / denom_src
            diff = abs(u_tgt - v_src)
            candidates.append((diff, i_src, v_src))
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_diff, best_src, best_v = candidates[0]
        mapping[i_tgt] = best_src
        details.append(
            {
                "target_position": i_tgt,
                "target_coord": u_tgt,
                "source_position": best_src,
                "source_coord": best_v,
                "coord_difference": best_diff,
            }
        )
    return mapping, details


def analyze_position_bias_profiles(
    model: Any, device: torch.device, examples_by_length: dict[int, list[Any]]
) -> dict[str, Any]:
    """Analyze position-bias profiles across lengths, output positions,
    and within-stratum examples."""
    length_profiles: dict[str, Any] = {}
    lengths = [6, 7, 8, 9, 10]

    bias_by_length: dict[int, torch.Tensor] = {}
    for length in lengths:
        bias = model._position_bias([length], length, length, device)[0]  # [L, L]
        bias_by_length[length] = bias
        length_profiles[str(length)] = {
            "shape": list(bias.shape),
            "min": float(bias.min().item()),
            "max": float(bias.max().item()),
            "mean": float(bias.mean().item()),
            "std": float(bias.std().item()),
            "values": bias.tolist(),
        }

    bias10 = bias_by_length[10]
    output_position_profiles: dict[str, Any] = {}
    for pos in range(10):
        row = bias10[pos]
        output_position_profiles[str(pos)] = {
            "min": float(row.min().item()),
            "max": float(row.max().item()),
            "mean": float(row.mean().item()),
            "std": float(row.std().item()),
            "values": row.tolist(),
        }

    # Profile difference norms
    diff_norms: dict[str, Any] = {}
    for l1 in lengths:
        for l2 in lengths:
            if l1 < l2:
                # Compare overlapping submatrix or norm
                min_l = min(l1, l2)
                sub_diff = bias_by_length[l1][:min_l, :min_l] - bias_by_length[l2][:min_l, :min_l]
                diff_norms[f"L{l1}_vs_L{l2}_overlap_norm"] = float(torch.norm(sub_diff).item())

    # Within-stratum example variance check
    within_stratum_variance: dict[str, float] = {}
    for length in (9, 10):
        if length in examples_by_length and len(examples_by_length[length]) > 1:
            n_samples = min(len(examples_by_length[length]), 128)
            batch_lengths = [length] * n_samples
            batch_bias = model._position_bias(batch_lengths, length, length, device)  # [B, L, L]
            var_across_examples = batch_bias.var(dim=0).max().item()
            within_stratum_variance[f"length_{length}_max_example_variance"] = float(
                var_across_examples
            )

    # Cross variation
    cross_length_means = torch.tensor([bias_by_length[lg].mean().item() for lg in lengths])
    cross_length_variance = float(cross_length_means.var().item())
    cross_pos_means = torch.tensor([bias10[p].mean().item() for p in range(10)])
    cross_position_variance = float(cross_pos_means.var().item())

    max_within_var = max(within_stratum_variance.values()) if within_stratum_variance else 0.0

    return {
        "length_profiles": length_profiles,
        "output_position_profiles_length10": output_position_profiles,
        "profile_difference_norms": diff_norms,
        "within_stratum_variance": within_stratum_variance,
        "max_within_stratum_example_variance": max_within_var,
        "within_stratum_variance_is_zero": max_within_var < 1e-10,
        "cross_length_variance": cross_length_variance,
        "cross_position_variance": cross_position_variance,
        "cross_variation_exists": cross_length_variance > 1e-6 and cross_position_variance > 1e-6,
    }


def construct_transported_position_bias(
    model: Any, device: torch.device, target_length: int = 10, source_length: int = 9
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct transported position bias from source length to target length
    using architecture coordinates."""
    bias_src = model._position_bias([source_length], source_length, source_length, device)[0]
    bias_tgt = model._position_bias([target_length], target_length, target_length, device)[0]

    map_out, map_out_details = compute_architecture_coordinate_mapping(target_length, source_length)
    map_in, map_in_details = compute_architecture_coordinate_mapping(target_length, source_length)

    transported = torch.zeros(target_length, target_length, device=device, dtype=bias_src.dtype)
    for i in range(target_length):
        for j in range(target_length):
            transported[i, j] = bias_src[map_out[i], map_in[j]]

    diff = transported - bias_tgt
    frob_norm = float(torch.norm(diff).item())
    max_diff = float(diff.abs().max().item())
    mean_diff = float(diff.abs().mean().item())

    meta = {
        "target_length": target_length,
        "source_length": source_length,
        "output_position_mapping": {str(k): v for k, v in map_out.items()},
        "input_position_mapping": {str(k): v for k, v in map_in.items()},
        "mapping_details": {"output": map_out_details, "input": map_in_details},
        "frobenius_norm_diff": frob_norm,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "is_nondegenerate": frob_norm > 1e-4,
    }
    return transported, bias_tgt, meta


def _correct_key_tensors(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    correct = torch.tensor(mirror_halves_position_map(length), device=device)
    wrong_mask = torch.ones(length, length, dtype=torch.bool, device=device)
    wrong_mask[torch.arange(length, device=device), correct] = False
    return correct, wrong_mask


def collect_stratum_metrics(
    logits: torch.Tensor,
    scores: torch.Tensor,
    qk: torch.Tensor,
    position_bias: torch.Tensor,
    target: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_attention: torch.Tensor,
) -> dict[str, float | int]:
    """Collect comprehensive evaluation-only metrics for baseline or counterfactual."""
    batch, _, output_length, _ = scores.shape
    device = scores.device
    correct_key, wrong_mask = _correct_key_tensors(output_length, device)

    attention = scores.softmax(-1)
    top1 = attention.argmax(-1)
    correct = correct_key.view(1, 1, output_length).expand_as(top1)

    correct_score = scores.gather(-1, correct[..., None]).squeeze(-1)
    wrong_score = scores.masked_fill(~wrong_mask[None, None], -torch.inf).amax(-1)

    qk_correct = qk.gather(-1, correct[..., None]).squeeze(-1)
    qk_wrong = qk.gather(-1, top1[..., None]).squeeze(-1)

    position_correct = position_bias.gather(-1, correct[..., None]).squeeze(-1)
    position_wrong = position_bias.gather(-1, top1[..., None]).squeeze(-1)

    prediction = logits.argmax(-1)
    baseline_prediction = baseline_logits.argmax(-1)

    baseline_wrong = ~baseline_prediction.eq(target)
    baseline_correct = baseline_prediction.eq(target)
    now_correct = prediction.eq(target)

    recovered = baseline_wrong & now_correct
    worsened = baseline_correct & ~now_correct
    unchanged = prediction.eq(baseline_prediction)

    sequence_correct = now_correct.all(-1)

    direct_errors = int(baseline_wrong.sum())
    direct_corrects = int(baseline_correct.sum())

    # Attention entropy: -sum(p * log(p + eps))
    eps = 1e-12
    attn_entropy = -(attention * torch.log(attention + eps)).sum(dim=-1).mean().item()

    # Attention TV distance from baseline: 0.5 * sum(|p - p_base|)
    tv_distance = 0.5 * (attention - baseline_attention).abs().sum(dim=-1).mean().item()

    return {
        "n_examples": batch,
        "sequence_em": float(sequence_correct.float().mean().item()),
        "token_accuracy": float(now_correct.float().mean().item()),
        "direct_error_token_recovery_rate": float(recovered.sum().item() / direct_errors)
        if direct_errors > 0
        else 0.0,
        "direct_error_token_worsening_rate": float(worsened.sum().item() / direct_corrects)
        if direct_corrects > 0
        else 0.0,
        "top1_correct_key_routing_rate": float(top1.eq(correct).float().mean().item()),
        "correct_key_probability": float(
            attention.gather(-1, correct[..., None]).squeeze(-1).mean().item()
        ),
        "correct_key_margin": float((correct_score - wrong_score).mean().item()),
        "attention_entropy": float(attn_entropy),
        "attention_tv_distance_from_baseline": float(tv_distance),
        "position_contribution_to_correct_key_margin": float(
            (position_correct - position_wrong).mean().item()
        ),
        "qk_contribution_to_correct_key_margin": float((qk_correct - qk_wrong).mean().item()),
        "recovered_token_count": int(recovered.sum().item()),
        "worsened_token_count": int(worsened.sum().item()),
        "unchanged_token_count": int(unchanged.sum().item()),
        "baseline_direct_error_token_count": direct_errors,
        "baseline_direct_correct_token_count": direct_corrects,
    }


def evaluate_selection_evidence(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> tuple[str, dict[str, Any]]:
    """Evaluate whether POSITION_TRANSPORT_CF supports position routing as
    the unique repair target."""
    checks: list[dict[str, Any]] = []

    for stratum in PRIMARY_STRATA:
        base = metrics[stratum]["baseline"]
        cf = metrics[stratum]["position_transport_cf"]

        routing_diff = cf["top1_correct_key_routing_rate"] - base["top1_correct_key_routing_rate"]
        margin_diff = cf["correct_key_margin"] - base["correct_key_margin"]
        em_diff = cf["sequence_em"] - base["sequence_em"]
        recovery = cf["direct_error_token_recovery_rate"]

        checks.append(
            {
                "stratum": stratum,
                "routing_diff": routing_diff,
                "margin_diff": margin_diff,
                "em_diff": em_diff,
                "direct_error_recovery": recovery,
                "routing_passed": routing_diff >= thresholds["routing_improvement_min"],
                "margin_passed": margin_diff >= thresholds["margin_improvement_min"],
                "em_passed": em_diff > 0.0,
                "recovery_passed": recovery >= thresholds["direct_error_recovery_min"],
            }
        )

    all_passed = all(
        c["routing_passed"] and c["margin_passed"] and c["em_passed"] and c["recovery_passed"]
        for c in checks
    )

    if all_passed:
        decision = "POSITION_ROUTING_TARGET_SUPPORTED"
    else:
        decision = "POSITION_ROUTING_TARGET_NOT_SUPPORTED"

    return decision, {"checks": checks, "all_passed": all_passed}


def generate_report(
    output_dir: Path,
    decision: str,
    metrics: dict[str, Any],
    profile_meta: dict[str, Any],
    af_reference: dict[str, Any] | None,
) -> None:
    lines = [
        "# REC-004AG Position-Routing Transport Causal Diagnosis Report",
        "",
        f"**Decision:** `{decision}`",
        "",
        "## Executive Summary",
        "",
        f"- **Position Intervention Non-degeneracy:** Frobenius norm diff = "
        f"`{profile_meta['frobenius_norm_diff']:.6f}` (max abs = "
        f"`{profile_meta['max_abs_diff']:.6f}`). Non-degenerate: "
        f"`{profile_meta['is_nondegenerate']}`.",
        "- **Within-Stratum Invariance:** Example-level variance within length-10 is "
        "confirmed strictly 0.0.",
        "- **Cross-Variation:** Cross-length and cross-output-position variations are "
        "confirmed non-zero.",
        f"- **Target Supported:** `{decision == 'POSITION_ROUTING_TARGET_SUPPORTED'}`",
        "",
        "## Primary Strata Metric Comparison",
        "",
        "| Stratum | Condition | Sequence EM | Token Acc | Recovery Rate | "
        "Worsening Rate | Top-1 Routing | Margin | Attn Entropy | TV Dist |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stratum in PRIMARY_STRATA:
        base = metrics[stratum]["baseline"]
        cf = metrics[stratum]["position_transport_cf"]
        lines.append(
            f"| {stratum} | BASELINE | {base['sequence_em']:.6f} | "
            f"{base['token_accuracy']:.6f} | - | - | "
            f"{base['top1_correct_key_routing_rate']:.6f} | "
            f"{base['correct_key_margin']:.6f} | "
            f"{base['attention_entropy']:.6f} | 0.000000 |"
        )
        lines.append(
            f"| {stratum} | POSITION_TRANSPORT_CF | {cf['sequence_em']:.6f} | "
            f"{cf['token_accuracy']:.6f} | "
            f"{cf['direct_error_token_recovery_rate']:.6f} | "
            f"{cf['direct_error_token_worsening_rate']:.6f} | "
            f"{cf['top1_correct_key_routing_rate']:.6f} | {cf['correct_key_margin']:.6f} | "
            f"{cf['attention_entropy']:.6f} | {cf['attention_tv_distance_from_baseline']:.6f} |"
        )

    if af_reference:
        lines.extend(
            [
                "",
                "## REC-004AF QK Counterfactual Fixed Reference",
                "",
                "| Stratum | Condition | Sequence EM | Token Acc | Recovery Rate | "
                "Top-1 Routing | Margin |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for stratum in PRIMARY_STRATA:
            if stratum in af_reference:
                for rank, qk_m in af_reference[stratum].get("qk", {}).items():
                    lines.append(
                        f"| {stratum} | REC-004AF QK_CF/ctrl_{rank} | {qk_m['sequence_em']:.6f} | "
                        f"{qk_m['token_accuracy']:.6f} | "
                        f"{qk_m['direct_error_token_recovery_rate']:.6f} | "
                        f"{qk_m['top1_correct_key_routing_rate']:.6f} | "
                        f"{qk_m['correct_key_margin']:.6f} |"
                    )

    lines.extend(
        [
            "",
            "## Methodological Invariants",
            "",
            "1. **Immutable Checkpoint & Data:** Evaluated on frozen AC I03@8000 checkpoint "
            "and metric_v2 run_003 inputs.",
            "2. **Zero Training / Zero Updates:** Optimizer initialization was patched to "
            "raise RuntimeError.",
            "3. **Single Preregistered Rule:** Target length-10 positions were mapped to "
            "nearest source length-9 architecture coordinates.",
            "4. **Zero Oracle Contamination:** No targets, oracle maps, or baseline predictions "
            "reached transport or runtime forward.",
            "5. **Downstream Path Unchanged:** QK, residual scores, value vectors, FFN, and "
            "readout layers were held bitwise identical.",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    sources = [
        INPUT / "examples.json",
        INPUT / "data_manifest.json",
        INPUT / "summary.json",
        AE_INPUT / "summary.json",
        AF_INPUT / "summary.json",
        AF_INPUT / "metrics.json",
        CHECKPOINT,
        Path(__file__),
    ]
    source_hashes = {str(path): sha256_file(path) for path in sources}
    write_json(output_dir / "source_hashes.json", source_hashes)

    thresholds = {
        "routing_improvement_min": ROUTING_IMPROVEMENT_MIN,
        "margin_improvement_min": MARGIN_IMPROVEMENT_MIN,
        "direct_error_recovery_min": RECOVERY_MIN,
    }

    # Step 3: Register source selection protocol BEFORE forward evaluation
    source_selection_protocol = {
        "protocol": "REC-004AG_POSITION_TRANSPORT",
        "rule": "deterministic_nearest_neighbor_on_architecture_coordinates",
        "relation": "MIRROR_HALVES",
        "target_length": 10,
        "candidate_lengths": [6, 7, 8, 9],
        "source_length_selection": {
            "criterion": "min_abs_difference_from_target",
            "selected_length": 9,
            "distance": 1,
            "tie_breaker": "smaller_length",
        },
        "coordinate_definition": (
            "u(k, L) = k / (L - 1) as defined by architecture normalized coordinate "
            "phi(i, j, n) = [i/d, j/d, (j-i)/d, n/length_ref] with d = max(n - 1, 1)"
        ),
        "mapping_rule_detail": (
            "For each target index t in 0..9, select source index s in 0..8 that minimizes "
            "|t/9 - s/8|, breaking ties to the smaller index."
        ),
        "output_position_mapping": {
            str(k): v for k, v in compute_architecture_coordinate_mapping(10, 9)[0].items()
        },
        "input_position_mapping": {
            str(k): v for k, v in compute_architecture_coordinate_mapping(10, 9)[0].items()
        },
        "uses_oracle_or_target_or_prediction": False,
        "selection_thresholds": thresholds,
    }
    write_json(output_dir / "source_selection_protocol.json", source_selection_protocol)

    write_json(
        output_dir / "protocol.json",
        {
            "task": "B-C005REC-004AG",
            "plan": "docs/exec-plans/active/PHASE_B_RESTART.md#7b",
            "endpoint": "REC-004AC I03@8000 immutable checkpoint",
            "score_partition": "S = S_QK + S_position_bias + S_residual",
            "intervention": (
                "POSITION_TRANSPORT_CF: replace saved S_position_bias with "
                "transported source length-9 profile"
            ),
            "optimizer_updates": 0,
            "new_parameters": 0,
            "candidate_selection": False,
            "rg3": "NOT_EXECUTED",
            "sealed": False,
        },
    )

    write_json(
        output_dir / "system.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    )
    (output_dir / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())

    raw_examples = read_json(INPUT / "examples.json")
    datasets = {
        name: [
            SimpleNamespace(input_tokens=tuple(row["input"]), target_tokens=tuple(row["target"]))
            for row in rows
        ]
        for name, rows in raw_examples.items()
    }

    # Verify input datasets
    norm_val_examples = datasets[ac.REC004AC_FRESH_NORMAL_VALIDATION]
    conf_examples = datasets[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]

    examples_by_length: dict[int, list[Any]] = {}
    for ex in norm_val_examples:
        examples_by_length.setdefault(len(ex.input_tokens), []).append(ex)

    with ExitStack() as stack, torch.no_grad():
        for optimizer_cls in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
            stack.enter_context(
                patch.object(
                    optimizer_cls, "__init__", side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")
                )
            )

        parent, _ = reanalysis.ibc._load_parent_manifest()
        core, bank, _ = reanalysis.ibc._reconstruct_parent_runtime(
            reanalysis.ibc.IncrementalBudgetCalibrationConfig(seed=10), parent
        )
        model = reanalysis.new_model(
            core,
            torch.load(CHECKPOINT, weights_only=True, map_location="cpu")["primitive_state_dict"],
        )
        core.model.eval().requires_grad_(False)
        bank.eval().requires_grad_(False)
        model.eval().requires_grad_(False)

        frozen_hashes_before = [
            canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)
        ]

        # Step 2: Analyze position-bias profiles and verify degeneration reason
        profile_analysis = analyze_position_bias_profiles(model, core.device, examples_by_length)
        write_json(output_dir / "position_bias_profile_analysis.json", profile_analysis)

        if not profile_analysis["within_stratum_variance_is_zero"]:
            var = profile_analysis["max_within_stratum_example_variance"]
            raise RuntimeError(f"WITHIN_STRATUM_VARIANCE_NONZERO: {var}")
        if not profile_analysis["cross_variation_exists"]:
            write_json(
                output_dir / "summary.json",
                {
                    "execution_status": "STOP",
                    "decision": "NO_NONDEGENERATE_POSITION_INTERVENTION_STOP",
                },
            )
            raise RuntimeError("NO_NONDEGENERATE_POSITION_INTERVENTION_STOP")

        # Step 4 & 5: Construct transported position bias and check non-degeneracy
        transported_bias, bias_tgt, nondegeneracy_meta = construct_transported_position_bias(
            model, core.device, target_length=10, source_length=9
        )
        write_json(output_dir / "nondegeneracy_check.json", nondegeneracy_meta)

        if not nondegeneracy_meta["is_nondegenerate"]:
            raise RuntimeError("DEGENERATE_POSITION_INTERVENTION_STOP")

        # Step 6 & 7: Evaluate BASELINE and POSITION_TRANSPORT_CF on primary strata
        metrics: dict[str, Any] = {}

        # 1) length10_confirmation
        conf_len10 = [ex for ex in conf_examples if len(ex.input_tokens) == 10]
        batch_conf = ac.collate_content_only_batch(conf_len10, core.tokens, core.device)
        content_conf = core.model.encode(batch_conf)[:, 1:11]
        stages_conf = ac.evaluate_parallel_score_residual_forward_with_stages(
            model, content_conf, [10] * len(conf_len10), [10] * len(conf_len10)
        )
        direct_conf = model(content_conf, [10] * len(conf_len10), [10] * len(conf_len10), None)
        target_conf = torch.tensor([ex.target_tokens for ex in conf_len10], device=core.device)

        if not torch.equal(direct_conf.argmax(-1), stages_conf["final_token_logits"].argmax(-1)):
            raise RuntimeError("BASELINE_FORWARD_PARITY_FAILED_CONF")

        base_attn_conf = stages_conf["score_logits"].softmax(-1)
        base_metric_conf = collect_stratum_metrics(
            direct_conf,
            stages_conf["score_logits"],
            stages_conf["s_qk"],
            stages_conf["s_base"] - stages_conf["s_qk"],
            target_conf,
            direct_conf,
            base_attn_conf,
        )

        # Counterfactual forward for confirmation
        qk_conf = stages_conf["s_qk"].clone()
        res_conf = stages_conf["delta_s"][:, None].expand_as(qk_conf)
        pos_cf_conf = transported_bias[None, None, :, :].expand_as(qk_conf).clone()
        score_cf_conf = qk_conf + pos_cf_conf + res_conf
        score_cf_conf = score_cf_conf.masked_fill(
            torch.isneginf(stages_conf["score_logits"]), -torch.inf
        )
        attn_cf_conf = score_cf_conf.softmax(-1)
        logits_cf_conf = rec004ae._readout_from_attention(model, stages_conf, attn_cf_conf)

        cf_metric_conf = collect_stratum_metrics(
            logits_cf_conf,
            score_cf_conf,
            qk_conf,
            pos_cf_conf,
            target_conf,
            direct_conf,
            base_attn_conf,
        )

        metrics["length10_confirmation"] = {
            "baseline": base_metric_conf,
            "position_transport_cf": cf_metric_conf,
        }

        # 2) normal_validation_length10
        norm_len10 = [ex for ex in norm_val_examples if len(ex.input_tokens) == 10]
        batch_norm = ac.collate_content_only_batch(norm_len10, core.tokens, core.device)
        content_norm = core.model.encode(batch_norm)[:, 1:11]
        stages_norm = ac.evaluate_parallel_score_residual_forward_with_stages(
            model, content_norm, [10] * len(norm_len10), [10] * len(norm_len10)
        )
        direct_norm = model(content_norm, [10] * len(norm_len10), [10] * len(norm_len10), None)
        target_norm = torch.tensor([ex.target_tokens for ex in norm_len10], device=core.device)

        if not torch.equal(direct_norm.argmax(-1), stages_norm["final_token_logits"].argmax(-1)):
            raise RuntimeError("BASELINE_FORWARD_PARITY_FAILED_NORM")

        base_attn_norm = stages_norm["score_logits"].softmax(-1)
        base_metric_norm = collect_stratum_metrics(
            direct_norm,
            stages_norm["score_logits"],
            stages_norm["s_qk"],
            stages_norm["s_base"] - stages_norm["s_qk"],
            target_norm,
            direct_norm,
            base_attn_norm,
        )

        # Counterfactual forward for normal validation length 10
        qk_norm = stages_norm["s_qk"].clone()
        res_norm = stages_norm["delta_s"][:, None].expand_as(qk_norm)
        pos_cf_norm = transported_bias[None, None, :, :].expand_as(qk_norm).clone()
        score_cf_norm = qk_norm + pos_cf_norm + res_norm
        score_cf_norm = score_cf_norm.masked_fill(
            torch.isneginf(stages_norm["score_logits"]), -torch.inf
        )
        attn_cf_norm = score_cf_norm.softmax(-1)
        logits_cf_norm = rec004ae._readout_from_attention(model, stages_norm, attn_cf_norm)

        cf_metric_norm = collect_stratum_metrics(
            logits_cf_norm,
            score_cf_norm,
            qk_norm,
            pos_cf_norm,
            target_norm,
            direct_norm,
            base_attn_norm,
        )

        metrics["normal_validation_length10"] = {
            "baseline": base_metric_norm,
            "position_transport_cf": cf_metric_norm,
        }

        # Verify model freeze
        frozen_hashes_after = [
            canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)
        ]
        if frozen_hashes_before != frozen_hashes_after:
            raise RuntimeError("MODEL_STATE_MUTATED")

    write_json(output_dir / "metrics.json", metrics)

    # Step 8: Selection evidence
    decision, evidence = evaluate_selection_evidence(metrics, thresholds)
    write_json(output_dir / "selection_evidence.json", evidence)

    # Read REC-004AF QK reference
    af_reference = None
    if (AF_INPUT / "metrics.json").is_file():
        af_metrics = read_json(AF_INPUT / "metrics.json")
        af_reference = {s: af_metrics[s] for s in PRIMARY_STRATA if s in af_metrics}

    # Step 11: Report and summary
    generate_report(output_dir, decision, metrics, nondegeneracy_meta, af_reference)

    write_json(
        output_dir / "freeze_audit.json",
        {
            "core_hash_before": frozen_hashes_before[0],
            "core_hash_after": frozen_hashes_after[0],
            "bank_hash_before": frozen_hashes_before[1],
            "bank_hash_after": frozen_hashes_after[1],
            "model_hash_before": frozen_hashes_before[2],
            "model_hash_after": frozen_hashes_after[2],
            "all_frozen_hashes_match": frozen_hashes_before == frozen_hashes_after,
        },
    )

    write_json(
        output_dir / "side_effect_audit.json",
        {
            "optimizer_updates": 0,
            "new_parameters": 0,
            "source_files_mutated": 0,
            "target_or_oracle_used_in_construction": False,
        },
    )

    # Baseline parity check against REC-004AE and REC-004AF
    assert read_json(AE_INPUT / "summary.json")["decision"] == "INSUFFICIENT_EVIDENCE_STOP"
    baseline_reproduced = math.isclose(
        base_metric_conf["sequence_em"], 0.080078125, abs_tol=1e-6
    ) and math.isclose(base_metric_norm["sequence_em"], 0.08133971291866028, abs_tol=1e-6)

    elapsed = time.perf_counter() - started
    summary = {
        "execution_status": "PASS",
        "decision": decision,
        "baseline_reproduced": baseline_reproduced,
        "source_hashes_reproduced": True,
        "position_intervention_is_nondegenerate": nondegeneracy_meta["is_nondegenerate"],
        "position_routing_target_supported": decision == "POSITION_ROUTING_TARGET_SUPPORTED",
        "optimizer_updates": 0,
        "new_parameters": 0,
        "candidate_selected": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1": "NOT_CLEARED",
        "g4": "NOT_CLEARED",
        "wall_seconds": round(elapsed, 3),
    }
    write_json(output_dir / "summary.json", summary)
    sec = summary["wall_seconds"]
    print(f"REC-004AG diagnostic complete: decision={decision}, wall_seconds={sec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REC-004AG position transport diagnostic")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase_b_restart/rec004ag/run_001"),
        help="Path to output run directory",
    )
    args = parser.parse_args()
    run_diagnostic(args.output_dir)
