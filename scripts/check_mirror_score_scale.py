"""Preregistered fixed-endpoint score-scale precheck; no training or selection."""

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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import reanalyze_rec004ac_metrics as reanalysis
from apc.evaluation import mirror_parallel_score_residual_pilot as ac
from apc.evaluation.mirror_attention_metrics import position_statistics
from apc.utils.model_bundle import canonical_state_hash

INPUT = Path("runs/phase_b_restart/rec004ac_metric_v2/run_002")
CONDITIONS = {
    "baseline": (1.0, 1.0),
    "temperature": (0.125, 1.0),
    "amplification": (1.0, 1000.0),
    "combined": (0.125, 1000.0),
}


def replay(
    model: Any,
    stages: dict[str, Any],
    alpha: float,
    beta: float,
    *,
    forced_attention: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Change only global score scales and replay the unchanged value/readout path."""
    scores = alpha * stages["s_base"] + beta * stages["delta_s"][:, None]
    scores = scores.masked_fill(torch.isneginf(stages["score_logits"]), -torch.inf)
    probabilities = scores.softmax(-1) if forced_attention is None else forced_attention
    batch, heads, outputs, inputs = scores.shape
    dim = model.d_operator
    weight = model.cross_attn.in_proj_weight[2 * dim :]
    bias = model.cross_attn.in_proj_bias
    value = F.linear(stages["v_in"], weight, None if bias is None else bias[2 * dim :])
    value = value.view(batch, inputs, heads, dim // heads).transpose(1, 2)
    attended = (probabilities @ value).transpose(1, 2).reshape(batch, outputs, dim)
    query = model.answer_query_embedding(torch.arange(outputs, device=scores.device))[None]
    hidden = model.attn_norm(query + model.cross_attn.out_proj(attended))
    hidden = model.ffn_norm(hidden + model.ffn(hidden))
    return model.readout(hidden), scores, probabilities


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write, read, sha = reanalysis.write, reanalysis.read, reanalysis.sha
    source_paths = [
        INPUT / name
        for name in (
            "examples.json",
            "data_manifest.json",
            "summary.json",
            "supplemental_validation.json",
        )
    ] + [
        reanalysis.SOURCE_DIR / "checkpoint_step8000.pt",
        Path("docs/exec-plans/active/PHASE_B_RESTART.md"),
        Path(__file__),
    ]
    source_paths += [Path(p) for p in read(INPUT / "source_hashes.json")]
    source_paths = list(dict.fromkeys(source_paths))
    before = {str(p): sha(p) for p in source_paths}
    write(output / "source_hashes.json", before)
    write(
        output / "protocol.json",
        {
            "plan": "docs/exec-plans/active/PHASE_B_RESTART.md#6",
            "conditions": CONDITIONS,
            "task": "MIRROR score-scale precheck",
            "init": "I03",
            "model_seed": 10,
            "endpoint": 8000,
            "new_parameters": 0,
            "optimizer_updates": 0,
            "selection": False,
            "sealed": False,
            "normal_and_length10_em_floor": 0.95,
            "oracle_floor": 0.95,
            "baseline_logit_tolerance": 1e-4,
            "baseline_prediction_tolerance": 0,
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
            "cpu_threads": torch.get_num_threads(),
        },
    )
    assert read(INPUT / "summary.json")["status"] == "PASS"
    assert (
        sha(INPUT / "examples.json")
        == read(INPUT / "supplemental_validation.json")["examples_sha256"]
    )
    data = read(INPUT / "examples.json")
    manifest = read(INPUT / "data_manifest.json")
    assert manifest == read(reanalysis.SOURCE_DIR / "fresh_validation_manifest.json")
    datasets = {
        name: [
            SimpleNamespace(input_tokens=tuple(e["input"]), target_tokens=tuple(e["target"]))
            for e in values
        ]
        for name, values in data.items()
    }
    digests = {
        name: ac._dataset_digest(ac._digest_examples(values)) for name, values in datasets.items()
    }
    for name, details in manifest["continuity_splits"].items():
        assert digests[name] == details["dataset_digest"]
    # Fresh splits' exact byte identity is certified by metric_v2's saved-input hash.
    write(
        output / "data_manifest.json",
        {
            "input_sha256": sha(INPUT / "examples.json"),
            "dataset_digests": digests,
            "counts": {n: len(v) for n, v in datasets.items()},
            "development_exposed": True,
            "sealed": False,
        },
    )
    parity_max = 0.0
    results: dict[str, dict[str, Any]] = {name: {} for name in CONDITIONS}
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
                torch.load(
                    reanalysis.SOURCE_DIR / "checkpoint_step8000.pt",
                    weights_only=True,
                    map_location="cpu",
                )["primitive_state_dict"],
            )
            core.model.eval().requires_grad_(False)
            bank.eval().requires_grad_(False)
            frozen = [canonical_state_hash(m.state_dict()) for m in (core.model, bank, model)]
            torch.cuda.reset_peak_memory_stats()
            for name, examples in datasets.items():
                counts = {
                    condition: {
                        "j0": 0,
                        "o1": 0,
                        "o1_p4": 0,
                        "n10": 0,
                        "p4_ranks": [],
                        "p4_margins": [],
                    }
                    for condition in CONDITIONS
                }
                for length in sorted({len(e.input_tokens) for e in examples}):
                    subset = [e for e in examples if len(e.input_tokens) == length]
                    for start in range(0, len(subset), 128):
                        chunk = subset[start : start + 128]
                        lens = [length] * len(chunk)
                        batch = ac.collate_content_only_batch(chunk, core.tokens, core.device)
                        h = core.model.encode(batch)[:, 1 : 1 + length]
                        stages = ac.evaluate_parallel_score_residual_forward_with_stages(
                            model, h, lens, lens
                        )
                        direct = model(h, lens, lens, None)
                        target = torch.tensor([e.target_tokens for e in chunk], device=core.device)
                        oracle = (
                            ac.evaluate_parallel_score_residual_forward_with_stages(
                                model, h, lens, lens, oracle_attention=True
                            )
                            if length == 10
                            else None
                        )
                        for condition, (alpha, beta) in CONDITIONS.items():
                            logits, scores, probabilities = replay(model, stages, alpha, beta)
                            if condition == "baseline":
                                difference = float((logits - direct).abs().max())
                                parity_max = max(parity_max, difference)
                                assert difference <= 1e-4, (name, difference)
                                assert torch.equal(logits.argmax(-1), direct.argmax(-1))
                            rec = counts[condition]
                            rec["j0"] += int((logits.argmax(-1) == target).all(-1).sum())
                            if oracle is not None:
                                o1, _, _ = replay(
                                    model,
                                    stages,
                                    alpha,
                                    beta,
                                    forced_attention=oracle["attn_probs"],
                                )
                                assert torch.equal(
                                    o1.argmax(-1), oracle["final_token_logits"].argmax(-1)
                                )
                                rec["o1"] += int((o1.argmax(-1) == target).all(-1).sum())
                                rec["o1_p4"] += int((o1.argmax(-1)[:, 4] == target[:, 4]).sum())
                                rec["n10"] += len(chunk)
                                stats = position_statistics(scores, probabilities, lens, 4)
                                rec["p4_ranks"].extend(stats["rank"].flatten().tolist())
                                rec["p4_margins"].extend(stats["margin"].flatten().tolist())
                for condition, rec in counts.items():
                    ranks = torch.tensor(rec.pop("p4_ranks"), dtype=torch.float32)
                    margins = torch.tensor(rec.pop("p4_margins"), dtype=torch.float32)
                    results[condition][name] = {
                        "n": len(examples),
                        "j0_em": rec["j0"] / len(examples),
                        "n10": rec["n10"],
                        "o1_em": rec["o1"] / rec["n10"],
                        "o1_p4_acc": rec["o1_p4"] / rec["n10"],
                        "p4_head_rank_mean": float(ranks.mean()),
                        "p4_head_top1": float((ranks <= 1).float().mean()),
                        "p4_head_margin_median": float(margins.median()),
                    }
                write(output / "metrics.json", results)
                print(
                    f"Evaluated {name}: {len(examples)} examples x 4 fixed conditions", flush=True
                )
            assert frozen == [
                canonical_state_hash(m.state_dict()) for m in (core.model, bank, model)
            ]
            baseline = read(INPUT / "summary.json")
            assert (
                results["baseline"][ac.REC004AC_FRESH_NORMAL_VALIDATION]["j0_em"]
                == baseline["normal_j0_em"]
            )
            assert (
                results["baseline"][ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]["j0_em"]
                == (baseline["length10"]["j0_sequence_em"])
            )
            gates = {
                condition: (
                    metrics[ac.REC004AC_FRESH_NORMAL_VALIDATION]["j0_em"] >= 0.95
                    and metrics[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]["j0_em"] >= 0.95
                    and all(m["o1_em"] >= 0.95 and m["o1_p4_acc"] >= 0.95 for m in metrics.values())
                )
                for condition, metrics in results.items()
            }
            write(
                output / "summary.json",
                {
                    "execution_status": "PASS",
                    "research_gate": "PASS" if any(gates.values()) else "FAIL_STOP",
                    "conditions": gates,
                    "baseline_max_logit_difference": parity_max,
                    "historical_em_parity": True,
                    "o1_prediction_parity": True,
                    "frozen_state_hashes": frozen,
                    "optimizer_updates": 0,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "wall_seconds": time.perf_counter() - started,
                    "parameter_accounting": reanalysis.cvof_parameter_accounting(model),
                    "precheck_actual_trainable_parameters": 0,
                    "core_resident_parameters": sum(
                        p.numel() for p in core.model.parameters()
                    ),
                    "parent_bank_resident_parameters": sum(p.numel() for p in bank.parameters()),
                    "parent_bank_active_parameters": 0,
                    "execution_scope": (
                        "frozen Core plus separate AC primitive; parent bank not executed"
                    ),
                    "rg3": "NOT_EXECUTED",
                    "candidate_selected": None,
                    "rec005_eligible": False,
                },
            )
            summary = read(output / "summary.json")
            lines = ["# MIRROR fixed score-scale precheck", "",
                     f"Execution: PASS. Research gate: {summary['research_gate']}.", "",
                     "| Condition | alpha | beta | Normal EM | Length-10 EM | Minimum O1 EM |",
                     "|---|---:|---:|---:|---:|---:|"]
            for condition, (alpha, beta) in CONDITIONS.items():
                metrics = results[condition]
                normal_em = metrics[ac.REC004AC_FRESH_NORMAL_VALIDATION]["j0_em"]
                length10_em = metrics[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]["j0_em"]
                minimum_o1 = min(m["o1_em"] for m in metrics.values())
                lines.append(f"| {condition} | {alpha} | {beta} | {normal_em:.8f} | "
                             f"{length10_em:.8f} | {minimum_o1:.8f} |")
            lines += ["", "All parameters were frozen; optimizer updates and selections were zero.",
                      "Baseline predictions and EMs match; O1 predictions are invariant.",
                      "The gate requires normal/length-10 EM and oracle controls each >= 0.95.",
                      "This tests four fixed inference settings on one saved endpoint only.",
                      "It does not determine whether training with different scales could succeed.",
                      "RG3 was not executed; REC-005 remains ineligible."]
            (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        write(
            output / "summary.json",
            {
                "execution_status": "FAIL",
                "reason": repr(exc),
                "research_gate": "NOT_EVALUATED",
                "rg3": "NOT_EXECUTED",
                "candidate_selected": None,
                "rec005_eligible": False,
            },
        )
        raise
    finally:
        after = {str(p): sha(p) for p in source_paths}
        write(output / "side_effect_audit.json", {"sources_unchanged": before == after})
        if before != after:
            write(output / "summary.json", {"execution_status": "FAIL",
                  "reason": "SOURCE_ARTIFACT_MUTATED", "research_gate": "NOT_EVALUATED",
                  "rg3": "NOT_EXECUTED", "candidate_selected": None, "rec005_eligible": False})
            raise RuntimeError("SOURCE_ARTIFACT_MUTATED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)
