"""REC-004AE: fixed-checkpoint MIRROR failure-mode localization; no training."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import reanalyze_rec004ac_metrics as reanalysis
from apc.evaluation import mirror_parallel_score_residual_pilot as ac
from apc.evaluation.mirror_position_initialization_diagnostic import mirror_halves_position_map
from apc.utils.model_bundle import canonical_state_hash

INPUT = Path("runs/phase_b_restart/rec004ac_metric_v2/run_003")
CHECKPOINT = reanalysis.SOURCE_DIR / "checkpoint_step8000.pt"
ABLATIONS = ("baseline", "without_position_bias", "without_qk")


def classify_tokens(
    direct: torch.Tensor, oracle: torch.Tensor, target: torch.Tensor
) -> dict[str, int]:
    """Partition output tokens by whether fixed oracle attention recovers J0."""
    direct_ok = direct.argmax(-1).eq(target)
    oracle_ok = oracle.argmax(-1).eq(target)
    return {
        "DIRECT_CORRECT": int(direct_ok.sum()),
        "ROUTING_ERROR_ORACLE_RECOVERS": int((~direct_ok & oracle_ok).sum()),
        "DOWNSTREAM_ERROR_PERSISTS_UNDER_ORACLE": int((~direct_ok & ~oracle_ok).sum()),
        "DIRECT_AND_ORACLE_CORRECT": int((direct_ok & oracle_ok).sum()),
    }


def _readout_from_attention(
    model: Any, stages: dict[str, Any], attention: torch.Tensor
) -> torch.Tensor:
    """Apply the exact saved value, output, FFN, and readout path to attention."""
    batch, heads, outputs, _ = attention.shape
    dim = model.d_operator
    value = stages["v_in"]
    wv = model.cross_attn.in_proj_weight.chunk(3, dim=0)[2]
    bv = (
        None if model.cross_attn.in_proj_bias is None else model.cross_attn.in_proj_bias.chunk(3)[2]
    )
    value = torch.nn.functional.linear(value, wv, bv)
    value = value.view(batch, -1, heads, dim // heads).transpose(1, 2)
    attended = (attention @ value).transpose(1, 2).reshape(batch, outputs, dim)
    query = model.answer_query_embedding(torch.arange(outputs, device=attention.device))[None]
    hidden = model.attn_norm(query + model.cross_attn.out_proj(attended))
    hidden = model.ffn_norm(hidden + model.ffn(hidden))
    return model.readout(hidden)


def ablation_scores(stages: dict[str, Any], name: str) -> torch.Tensor:
    """Return one of the preregistered component-removal diagnostic score matrices."""
    if name == "baseline":
        scores = stages["s_total"]
    elif name == "without_position_bias":
        scores = stages["s_total"] - (stages["s_base"] - stages["s_qk"])
    elif name == "without_qk":
        scores = stages["s_total"] - stages["s_qk"]
    else:
        raise ValueError(f"unknown ablation: {name}")
    return scores.masked_fill(torch.isneginf(stages["score_logits"]), -torch.inf)


def _empty_modes() -> dict[str, int]:
    return {
        "DIRECT_CORRECT": 0,
        "ROUTING_ERROR_ORACLE_RECOVERS": 0,
        "DOWNSTREAM_ERROR_PERSISTS_UNDER_ORACLE": 0,
        "DIRECT_AND_ORACLE_CORRECT": 0,
    }


def _component_table(stages: dict[str, Any], length: int) -> list[dict[str, Any]]:
    scores = stages["score_logits"]
    mapping = mirror_halves_position_map(length)
    rows: list[dict[str, Any]] = []
    for position, correct_key in enumerate(mapping):
        total = scores[:, :, position, :length]
        wrong_mask = torch.ones(length, dtype=torch.bool, device=scores.device)
        wrong_mask[correct_key] = False
        wrong = total.masked_fill(~wrong_mask[None, None], -torch.inf)
        top_wrong = wrong.argmax(-1)

        def gather(value: torch.Tensor, keys: torch.Tensor, pos: int = position) -> torch.Tensor:
            return value[:, :, pos, :length].gather(-1, keys[..., None]).squeeze(-1)

        correct = torch.full_like(top_wrong, correct_key)
        qk_delta = gather(stages["s_qk"], correct) - gather(stages["s_qk"], top_wrong)
        bias = stages["s_base"] - stages["s_qk"]
        bias_delta = gather(bias, correct) - gather(bias, top_wrong)
        residual = stages["delta_s"][:, None].expand_as(stages["s_base"])
        residual_delta = gather(residual, correct) - gather(residual, top_wrong)
        total_delta = gather(stages["s_total"], correct) - gather(stages["s_total"], top_wrong)
        correct_total = total.gather(-1, correct[..., None]).squeeze(-1)
        ranks = (total >= correct_total[..., None]).sum(-1)
        counts = torch.bincount(top_wrong.flatten(), minlength=length)
        rows.append(
            {
                "output_position": position,
                "correct_key": correct_key,
                "modal_top_wrong_key": int(counts.argmax()),
                "modal_top_wrong_key_fraction": float(counts.max() / top_wrong.numel()),
                "correct_key_rank_mean": float(ranks.float().mean()),
                "correct_key_top1_fraction": float((ranks == 1).float().mean()),
                "total_margin_mean": float(total_delta.mean()),
                "qk_margin_contribution_mean": float(qk_delta.mean()),
                "position_bias_margin_contribution_mean": float(bias_delta.mean()),
                "score_residual_margin_contribution_mean": float(residual_delta.mean()),
            }
        )
    return rows


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write, read, sha = reanalysis.write, reanalysis.read, reanalysis.sha
    source_paths = [
        INPUT / name for name in ("examples.json", "data_manifest.json", "summary.json")
    ]
    source_paths += [CHECKPOINT, Path(__file__)]
    before = {str(path): sha(path) for path in source_paths}
    write(output / "source_hashes.json", before)
    write(
        output / "protocol.json",
        {
            "task": "B-C005REC-004AE",
            "plan": "docs/exec-plans/active/PHASE_B_RESTART.md#7",
            "endpoint": "REC-004AC I03@8000",
            "input_run": str(INPUT),
            "development_only": True,
            "sealed": False,
            "optimizer_updates": 0,
            "new_parameters": 0,
            "candidate_selection": False,
            "rg3": "NOT_EXECUTED",
            "fixed_ablations": list(ABLATIONS),
            "oracle_is_evaluation_only": True,
            "localization_criteria": {
                "oracle_persistent_error_max": 0.01,
                "direct_error_oracle_recovery_min": 0.95,
                "same_failure_pattern_required_on": ["length10_confirmation", "normal_length10"],
                "ablation_must_uniquely_separate_component": True,
            },
        },
    )
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write(
        output / "system.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
    )
    assert read(INPUT / "summary.json")["status"] == "PASS"
    raw = read(INPUT / "examples.json")
    manifest = read(INPUT / "data_manifest.json")
    datasets = {
        name: [
            SimpleNamespace(input_tokens=tuple(row["input"]), target_tokens=tuple(row["target"]))
            for row in rows
        ]
        for name, rows in raw.items()
    }
    digests = {
        name: ac._dataset_digest(ac._digest_examples(rows)) for name, rows in datasets.items()
    }
    write(
        output / "data_manifest.json",
        {
            "input_sha256": sha(INPUT / "examples.json"),
            "counts": {name: len(rows) for name, rows in datasets.items()},
            "dataset_digests": digests,
            "source_manifest_matches": manifest
            == read(reanalysis.SOURCE_DIR / "fresh_validation_manifest.json"),
            "sealed": False,
        },
    )
    assert manifest == read(reanalysis.SOURCE_DIR / "fresh_validation_manifest.json")
    results: dict[str, Any] = {}
    position_tables: dict[str, list[dict[str, Any]]] = {}
    try:
        with ExitStack() as stack, torch.no_grad():
            for optimizer in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
                stack.enter_context(
                    patch.object(
                        optimizer, "__init__", side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")
                    )
                )
            parent, _ = reanalysis.ibc._load_parent_manifest()
            core, bank, _ = reanalysis.ibc._reconstruct_parent_runtime(
                reanalysis.ibc.IncrementalBudgetCalibrationConfig(seed=10), parent
            )
            model = reanalysis.new_model(
                core,
                torch.load(CHECKPOINT, weights_only=True, map_location="cpu")[
                    "primitive_state_dict"
                ],
            )
            core.model.eval().requires_grad_(False)
            bank.eval().requires_grad_(False)
            model.eval().requires_grad_(False)
            frozen = [canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)]
            torch.cuda.reset_peak_memory_stats()
            for name, examples in datasets.items():
                record = {
                    "n_examples": len(examples),
                    "sequence": {key: 0 for key in ABLATIONS},
                    "tokens": _empty_modes(),
                    "positions": {},
                    "by_length": {},
                    "oracle_sequence_correct": 0,
                }
                for length in sorted({len(example.input_tokens) for example in examples}):
                    subset = [
                        example for example in examples if len(example.input_tokens) == length
                    ]
                    stratum = {
                        "n_examples": len(subset),
                        "tokens": _empty_modes(),
                        "oracle_sequence_correct": 0,
                    }
                    for start in range(0, len(subset), 128):
                        chunk = subset[start : start + 128]
                        batch = ac.collate_content_only_batch(chunk, core.tokens, core.device)
                        content = core.model.encode(batch)[:, 1 : 1 + length]
                        stages = ac.evaluate_parallel_score_residual_forward_with_stages(
                            model, content, [length] * len(chunk), [length] * len(chunk)
                        )
                        direct = model(content, [length] * len(chunk), [length] * len(chunk), None)
                        target = torch.tensor(
                            [example.target_tokens for example in chunk], device=core.device
                        )
                        assert torch.equal(
                            direct.argmax(-1), stages["final_token_logits"].argmax(-1)
                        )
                        oracle_stages = ac.evaluate_parallel_score_residual_forward_with_stages(
                            model,
                            content,
                            [length] * len(chunk),
                            [length] * len(chunk),
                            oracle_attention=True,
                        )
                        oracle = oracle_stages["final_token_logits"]
                        modes = classify_tokens(direct, oracle, target)
                        for mode, count in modes.items():
                            record["tokens"][mode] += count
                            stratum["tokens"][mode] += count
                        record["oracle_sequence_correct"] += int(
                            (oracle.argmax(-1) == target).all(-1).sum()
                        )
                        stratum["oracle_sequence_correct"] += int(
                            (oracle.argmax(-1) == target).all(-1).sum()
                        )
                        for ablation in ABLATIONS:
                            scores = ablation_scores(stages, ablation)
                            logits = _readout_from_attention(model, stages, scores.softmax(-1))
                            if ablation == "baseline":
                                assert torch.equal(logits.argmax(-1), direct.argmax(-1))
                            record["sequence"][ablation] += int(
                                (logits.argmax(-1) == target).all(-1).sum()
                            )
                        if length == 10:
                            table = _component_table(stages, length)
                            for row in table:
                                pos = str(row["output_position"])
                                aggregate = record["positions"].setdefault(
                                    pos, {"n_batches": 0, "rows": []}
                                )
                                aggregate["n_batches"] += 1
                                aggregate["rows"].append(row)
                    stratum_token_total = len(subset) * length
                    stratum["token_total"] = stratum_token_total
                    stratum["oracle_persistent_error_rate"] = (
                        stratum["tokens"]["DOWNSTREAM_ERROR_PERSISTS_UNDER_ORACLE"]
                        / stratum_token_total
                    )
                    stratum_direct_errors = (
                        stratum_token_total - stratum["tokens"]["DIRECT_CORRECT"]
                    )
                    stratum["direct_error_oracle_recovery_rate"] = (
                        stratum["tokens"]["ROUTING_ERROR_ORACLE_RECOVERS"] / stratum_direct_errors
                        if stratum_direct_errors
                        else 1.0
                    )
                    stratum["oracle_sequence_em"] = stratum["oracle_sequence_correct"] / len(subset)
                    record["by_length"][str(length)] = stratum
                token_total = len(examples) * max(
                    len(example.target_tokens) for example in examples
                )
                record["token_total"] = token_total
                record["oracle_persistent_error_rate"] = (
                    record["tokens"]["DOWNSTREAM_ERROR_PERSISTS_UNDER_ORACLE"] / token_total
                )
                direct_errors = token_total - record["tokens"]["DIRECT_CORRECT"]
                record["direct_error_oracle_recovery_rate"] = (
                    record["tokens"]["ROUTING_ERROR_ORACLE_RECOVERS"] / direct_errors
                    if direct_errors
                    else 1.0
                )
                record["oracle_sequence_em"] = record["oracle_sequence_correct"] / len(examples)
                record["sequence_em"] = {
                    key: value / len(examples) for key, value in record["sequence"].items()
                }
                if record["positions"]:
                    collapsed: list[dict[str, Any]] = []
                    for pos, aggregate in record["positions"].items():
                        rows = aggregate["rows"]
                        fields = [
                            key
                            for key in rows[0]
                            if key not in {"output_position", "correct_key", "modal_top_wrong_key"}
                        ]
                        collapsed.append(
                            {
                                "output_position": int(pos),
                                "correct_key": rows[0]["correct_key"],
                                "modal_top_wrong_key": rows[0]["modal_top_wrong_key"],
                                **{
                                    key: sum(row[key] for row in rows) / len(rows) for key in fields
                                },
                            }
                        )
                    position_tables[name] = collapsed
                    del record["positions"]
                results[name] = record
                write(output / "failure_modes.json", results)
            assert frozen == [
                canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)
            ]
            baseline = read(INPUT / "summary.json")
            normal = results[ac.REC004AC_FRESH_NORMAL_VALIDATION]
            length10 = results[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]
            assert normal["sequence_em"]["baseline"] == baseline["normal_j0_em"]
            assert length10["sequence_em"]["baseline"] == baseline["length10"]["j0_sequence_em"]
            write(output / "position_components.json", position_tables)
            normal_length10 = normal["by_length"]["10"]
            routing_recovery = all(
                item["oracle_persistent_error_rate"] <= 0.01
                and item["direct_error_oracle_recovery_rate"] >= 0.95
                for item in (normal_length10, length10)
            )
            ablation_delta = {
                name: results[name]["sequence_em"]["without_position_bias"]
                - results[name]["sequence_em"]["without_qk"]
                for name in (
                    ac.REC004AC_FRESH_NORMAL_VALIDATION,
                    ac.REC004AC_FRESH_LENGTH10_CONFIRMATION,
                )
            }
            same_direction = all(value > 0 for value in ablation_delta.values()) or all(
                value < 0 for value in ablation_delta.values()
            )
            position_pattern_consistent = all(
                (left["qk_margin_contribution_mean"] * right["qk_margin_contribution_mean"] > 0)
                and (
                    left["position_bias_margin_contribution_mean"]
                    * right["position_bias_margin_contribution_mean"]
                    > 0
                )
                for left, right in zip(
                    position_tables[ac.REC004AC_FRESH_NORMAL_VALIDATION],
                    position_tables[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION],
                    strict=True,
                )
            )
            decision = (
                "ROUTING_LOCALIZED"
                if (routing_recovery and position_pattern_consistent and same_direction)
                else "INSUFFICIENT_EVIDENCE_STOP"
            )
            summary = {
                "execution_status": "PASS",
                "decision": decision,
                "routing_recovery_criteria": routing_recovery,
                "position_pattern_consistent": position_pattern_consistent,
                "ablation_direction_consistent": same_direction,
                "ablation_position_bias_minus_qk_sequence_em": ablation_delta,
                "frozen_state_hashes": frozen,
                "optimizer_updates": 0,
                "candidate_selected": None,
                "rg3": "NOT_EXECUTED",
                "rec005_eligible": False,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "wall_seconds": time.perf_counter() - started,
                "next_intervention": (
                    "LEARNED_POSITION_ONLY_ROUTING_WITH_QK_ROLE_REMOVAL"
                    if decision == "ROUTING_LOCALIZED"
                    else None
                ),
            }
            write(output / "summary.json", summary)
            lines = [
                "# REC-004AE MIRROR failure-mode localization",
                "",
                f"Decision: **{decision}**.",
                "",
                "| Dataset | Baseline EM | No position-bias EM | No QK EM | Oracle EM | "
                "Oracle-persistent token error | Direct-error oracle recovery |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for name, item in results.items():
                lines.append(
                    f"| {name} | {item['sequence_em']['baseline']:.6f} | "
                    f"{item['sequence_em']['without_position_bias']:.6f} | "
                    f"{item['sequence_em']['without_qk']:.6f} | "
                    f"{item['oracle_sequence_em']:.6f} | "
                    f"{item['oracle_persistent_error_rate']:.6f} | "
                    f"{item['direct_error_oracle_recovery_rate']:.6f} |"
                )
            lines += [
                "",
                "All tensors were frozen; no optimizer was constructed. "
                "Oracle attention is evaluation-only.",
                "The QK/position-bias removals are fixed diagnostic ablations, "
                "not coefficient search or candidate repair.",
                "RG3 and sealed evaluation were not executed.",
            ]
            (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        write(
            output / "summary.json",
            {
                "execution_status": "FAIL",
                "reason": repr(exc),
                "decision": "INSUFFICIENT_EVIDENCE_STOP",
                "rg3": "NOT_EXECUTED",
                "rec005_eligible": False,
            },
        )
        raise
    finally:
        after = {str(path): sha(path) for path in source_paths}
        write(output / "side_effect_audit.json", {"sources_unchanged": before == after})
        if before != after:
            raise RuntimeError("SOURCE_ARTIFACT_MUTATED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)
