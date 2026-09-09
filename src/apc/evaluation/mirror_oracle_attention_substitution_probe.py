"""B-C005REC-004F: Oracle-Attention Substitution Probe (MIRROR_HALVES).

Executes the SINGLE diagnostic probe proposed -- and only proposed -- in
B-C005REC-004E's `next_repair_contract.md`
(`runs/phase_b_b2_model_bundle_recovery/rec004e/run_001/next_repair_contract.md`,
`status: PROPOSED_NOT_AUTHORIZED`). Running it was authorized by an explicit
user instruction (2026-09-09) that named exactly this probe from among
several candidates the assistant surfaced; no other repair (B-C005REC-005,
B-C005R3-011, B-C006, Task Inference) is started here.

Question (from the contract): REC-004E found that at length 10 the bias-
augmented combined attention already ranks the teacher-aligned key
favorably (positive descriptive margin, mean rank 1.7-2.7) while sequence
EM stays low. Does substituting the ORACLE attention distribution (one-hot
over `pi_n`, MIRROR_HALVES's content-independent position map) in place of
the model's own combined attention -- on the SAME saved P/I01-I05
step=6000 checkpoints, through the SAME real value projection, `out_proj`,
`attn_norm`, `ffn`, and `readout` weights -- recover length-10 EM? If it
does, the residual sits in the attention distribution itself. If it does
not, it sits downstream (value projection, residual path, or readout).

**Deliberate, disclosed exception to REC-004E's own rule.** REC-004E never
fed `pi_n` into any score/bias/intervention forward input; the oracle
position map was used there strictly for post-hoc scoring. THIS probe is
the one place that rule is intentionally suspended, exactly as specified in
the contract it implements: the oracle position map IS the substituted
attention weight under test, not a scoring convenience. `mirror_halves_
position_map` is called from exactly one function in this module,
`_oracle_attention` -- verified by
`tests/test_mirror_oracle_attention_substitution_probe.py`'s source-scan
test -- and nowhere else: not in checkpoint loading, not in example
generation, not in any comparison/diagnosis function.

**Zero new optimizer updates.** No weights, Core, parent bundle, router,
ArgumentScorer, verifier, fixed 3 candidates, shared cache, or REC-004D/
REC-004E checkpoint or result file is ever modified. Uses the SAME 5 saved
P/I01-I05 step=6000 checkpoints and the SAME length-balanced diagnostic
suite as REC-004E (`mirror_position_initialization_diagnostic._generate_
length_balanced_diagnostic`) -- no new init, no new data, no new query set.
Never selects, adopts, or trains on any result. REC-004D's own adoption
rule (5/5 P inits >= 0.95 existing_validation EM) and RG3's non-SHIFT-15
floor are unchanged by this or any future task. `selected_init`,
`selected_intervention`, and `child_bundle` stay `null`; `rg3_recheck`
stays `NOT_EXECUTED`. STOP after this probe's result is reported -- no
auto-continuation to any further repair, even if length-10 EM recovers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_position_bias_repair as bias_repair
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004F_TASK_ID",
    "REC004F_SOURCE_TASK_ID",
    "REC004F_CONTRACT_FILE",
    "OracleAttentionSubstitutionProbeConfig",
    "run_oracle_attention_substitution_probe_task",
]

# =============================================================================
# Constants -- every sizing/selection constant is inherited from REC-004E
# (never re-derived) so the two tasks read literally the same checkpoints
# and the same suite; only the recovery-label thresholds below are new,
# and they are pre-registered here, before any probe result is seen.
# =============================================================================

REC004F_TASK_ID: Final = "B-C005REC-004F"
REC004F_SOURCE_TASK_ID: Final = "B-C005REC-004E"
REC004F_CONTRACT_FILE: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004e/run_001/next_repair_contract.md"
)
REC004F_EXPECTED_CONTRACT_STATUS: Final = "PROPOSED_NOT_AUTHORIZED"
REC004E_RUN_DIR: Final = resid_audit.REC004E_SOURCE_RUN_DIR.parent.parent / "rec004e" / "run_001"

REC004F_INIT_IDS: Final[tuple[str, ...]] = resid_audit.REC004E_INIT_IDS
REC004F_LEGAL_LENGTHS: Final[tuple[int, ...]] = resid_audit.REC004E_LEGAL_LENGTHS
REC004F_DECISIVE_STEP: Final = resid_audit.REC004E_DECISIVE_STEP
REC004F_TARGET_OPERATION: Final = resid_audit.REC004E_TARGET_OPERATION
REC004F_PER_LENGTH: Final = resid_audit.REC004E_INTERVENTION_PER_LENGTH  # 256, SAME suite

# Pre-registered labeling thresholds (task doc analogue of REC-004E's own
# section-3 tolerances) -- fixed before the probe runs, never widened after
# seeing a result. These only pick a descriptive LABEL; they never select,
# adopt, or gate anything.
REC004F_RECOVERY_DELTA_THRESHOLD: Final = 0.05
REC004F_RECOVERY_EM_THRESHOLD: Final = 0.95
REC004F_SOURCE_REPLAY_EM_TOL: Final = 1e-9


@dataclass(frozen=True)
class OracleAttentionSubstitutionProbeConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004f/run_001")
    seed: int = RECOVERY_PILOT_SEED


# =============================================================================
# Authorization trail -- confirms this run implements exactly the probe
# proposed in REC-004E's own contract file, not a reinterpretation of it.
# =============================================================================


def verify_authorization() -> dict[str, Any]:
    if not REC004F_CONTRACT_FILE.is_file():
        return {
            "task_id": REC004F_TASK_ID,
            "status": "CONTRACT_FILE_MISSING",
            "authorized": False,
        }
    text = REC004F_CONTRACT_FILE.read_text(encoding="utf-8")
    status_line = next(
        (line for line in text.splitlines() if line.startswith("status:")), ""
    )
    matches = REC004F_EXPECTED_CONTRACT_STATUS in status_line
    return {
        "task_id": REC004F_TASK_ID,
        "contract_file": str(REC004F_CONTRACT_FILE),
        "contract_file_sha256": mb.raw_file_sha256(REC004F_CONTRACT_FILE),
        "contract_status_line": status_line,
        "matches_expected_proposed_status": matches,
        "user_authorization": (
            "explicit user instruction, 2026-09-09: run exactly the oracle-attention "
            "substitution probe described in this contract file, and only that probe"
        ),
        "authorized": matches,
        "status": "AUTHORIZED" if matches else "CONTRACT_STATUS_MISMATCH",
    }


# =============================================================================
# The substitution itself -- value projection and out_proj are the SAME
# real, trained weights `primitive.cross_attn` uses in production; only the
# attention distribution is replaced by a one-hot `pi_n` map.
# =============================================================================


def _oracle_attention(
    primitive: CrossPositionLengthBiasPrimitive,
    query: torch.Tensor,
    kv: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replaces the model's own combined (S_other + B) softmax attention
    with a one-hot distribution over `mirror_halves_position_map(n)` -- THE
    substitution under test (see module docstring's disclosed exception).
    Value projection, `out_proj`, and every downstream weight (`attn_norm`/
    `ffn`/`readout`, applied by the caller via `_post_attention`) are the
    SAME real, trained weights `primitive` uses in its normal forward; only
    the attention distribution itself is replaced. The SAME distribution is
    shared across all `n_head` heads, since `pi_n` carries no per-head
    information. Requires a single-length, unpadded batch (`kv`/`query`
    both sized exactly `n`) -- this probe never mixes lengths in one batch,
    so no padding-aware masking is needed here."""
    resid_audit._assert_eval_only()
    mha = primitive.cross_attn
    embed_dim = primitive.d_operator
    n_head = primitive.n_head
    head_dim = embed_dim // n_head
    _, _, wv = mha.in_proj_weight.chunk(3, dim=0)
    bv = None
    if mha.in_proj_bias is not None:
        _, _, bv = mha.in_proj_bias.chunk(3, dim=0)
    batch, lmax, _ = kv.shape
    out_max = query.shape[1]
    if lmax != n or out_max != n:
        raise AssertionError(
            f"oracle substitution requires a single-length, unpadded batch "
            f"(got lmax={lmax}, out_max={out_max}, n={n})"
        )
    v = F.linear(kv, wv, bv)
    v = v.view(batch, lmax, n_head, head_dim).transpose(1, 2)  # [batch, n_head, lmax, head_dim]
    pi = mpid.mirror_halves_position_map(n)
    oracle = torch.zeros(n, n, device=kv.device, dtype=v.dtype)
    for i, j in enumerate(pi):
        oracle[i, j] = 1.0
    attn_probs = oracle.view(1, 1, n, n).expand(batch, n_head, n, n)
    attn_out_heads = torch.matmul(attn_probs, v)  # [batch, n_head, out_max, head_dim]
    attn_out = attn_out_heads.transpose(1, 2).reshape(batch, out_max, embed_dim)
    attn_out = mha.out_proj(attn_out)
    return attn_out, attn_probs


def run_oracle_forward(
    primitive: CrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, Any]:
    """Full oracle-substituted forward: real query/kv construction (`
    resid_audit._prepare_query_kv`, the SAME helper REC-004E's own J-
    conditions use), the oracle attention substitution above, then the
    SAME real `_post_attention` (attn_norm -> ffn -> readout) REC-004E's
    own J-conditions use. Requires every example in the batch to share the
    same real content length (no cross-length padding)."""
    resid_audit._assert_eval_only()
    lengths = set(content_lengths)
    if len(lengths) != 1:
        raise AssertionError(
            f"oracle substitution probe requires a single-length batch, got {sorted(lengths)}"
        )
    n = lengths.pop()
    query, kv, out_max, lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content_features, content_lengths, output_lengths
    )
    attn_out, attn_probs = _oracle_attention(primitive, query, kv, n)
    logits = resid_audit._post_attention(primitive, query, attn_out)
    return {"logits": logits, "attn_probs": attn_probs}


# =============================================================================
# Probe execution -- per init, per legal length, paired against a freshly
# computed J0 (REC-004E's own real-forward baseline) on the SAME examples.
# =============================================================================


def run_oracle_attention_substitution_probe(core: Any) -> dict[str, Any]:
    resid_audit._assert_eval_only()
    by_length = resid_audit._length_balanced_by_length()
    op = get_operation(REC004F_TARGET_OPERATION)
    rows: list[dict[str, Any]] = []
    per_init_summary: dict[str, Any] = {}

    for init_id in REC004F_INIT_IDS:
        primitive = resid_audit._load_p_primitive(core, init_id, REC004F_DECISIVE_STEP)
        if primitive is None:
            per_init_summary[init_id] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
            continue
        per_length: dict[str, Any] = {}
        for n in REC004F_LEGAL_LENGTHS:
            examples = by_length[n]
            content_lengths = [len(e.input_tokens) for e in examples]
            assert all(cl == n for cl in content_lengths)
            output_lengths = [op.output_length(cl) for cl in content_lengths]
            batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
            with torch.no_grad():
                h_content = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
                j0_out = resid_audit.run_intervention_forward(
                    primitive, h_content, content_lengths, output_lengths, "J0"
                )
                oracle_out = run_oracle_forward(
                    primitive, h_content, content_lengths, output_lengths
                )
                labels = _labels_for_examples(
                    examples, output_lengths, max(output_lengths), core.device
                )
                oracle_loss = F.cross_entropy(
                    oracle_out["logits"].reshape(-1, oracle_out["logits"].size(-1)),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                ).item()
            j0_preds = resid_audit._predict_from_logits(j0_out["logits"], output_lengths)
            oracle_preds = resid_audit._predict_from_logits(oracle_out["logits"], output_lengths)
            j0_correct = [
                tuple(p) == tuple(e.target_tokens[:no])
                for p, e, no in zip(j0_preds, examples, output_lengths, strict=True)
            ]
            oracle_correct = [
                tuple(p) == tuple(e.target_tokens[:no])
                for p, e, no in zip(oracle_preds, examples, output_lengths, strict=True)
            ]
            pairs = list(zip(j0_correct, oracle_correct, strict=True))
            j0_only = sum(1 for a, b in pairs if a and not b)
            oracle_only = sum(1 for a, b in pairs if b and not a)
            both_correct = sum(1 for a, b in pairs if a and b)
            both_wrong = sum(1 for a, b in pairs if not a and not b)
            oracle_token_correct = sum(
                int(p[k] == e.target_tokens[k])
                for p, e, no in zip(oracle_preds, examples, output_lengths, strict=True)
                for k in range(no)
            )
            token_total = sum(output_lengths)
            j0_em = sum(j0_correct) / len(j0_correct)
            oracle_em = sum(oracle_correct) / len(oracle_correct)
            row = {
                "init_id": init_id,
                "length": n,
                "n": len(examples),
                "j0_sequence_exact_match": j0_em,
                "oracle_sequence_exact_match": oracle_em,
                "oracle_token_accuracy": (
                    oracle_token_correct / token_total if token_total else None
                ),
                "oracle_loss": oracle_loss,
                "paired_delta_oracle_minus_j0": oracle_em - j0_em,
                "j0_only_correct": j0_only,
                "oracle_only_correct": oracle_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            }
            rows.append(row)
            per_length[str(n)] = row
        per_init_summary[init_id] = per_length

    return {"rows": rows, "per_init_summary": per_init_summary}


def cross_check_j0_against_rec004e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirms this probe's freshly recomputed J0 matches REC-004E's own
    saved `paired_intervention_summary.json` J0 numbers exactly -- both read
    the same checkpoints via the same deterministic suite, so any
    difference would mean the two tasks are not actually comparing the same
    baseline."""
    path = REC004E_RUN_DIR / "paired_intervention_summary.json"
    if not path.is_file():
        return {"status": "REC004E_ARTIFACT_UNAVAILABLE"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        init_id, n = row["init_id"], row["length"]
        rec = recorded.get(init_id, {}).get("J0", {}).get("per_length", {}).get(str(n))
        recorded_em = rec["sequence_exact_match"] if rec else None
        reproduced_em = row["j0_sequence_exact_match"]
        matches = (
            recorded_em is not None
            and abs(reproduced_em - recorded_em) < REC004F_SOURCE_REPLAY_EM_TOL
        )
        if not matches:
            mismatches.append(
                {
                    "init_id": init_id, "length": n,
                    "reproduced": reproduced_em, "recorded": recorded_em,
                }
            )
    return {
        "status": "VERIFIED" if not mismatches else "MISMATCH",
        "n_checked": len(rows),
        "mismatches": mismatches,
    }


def run_probe_nonmutation_audit(core: Any) -> dict[str, Any]:
    """Confirms running the full oracle probe against one loaded checkpoint
    leaves its parameters byte-identical, that the SAME checkpoint file on
    disk is untouched, and that a normal (unmodified) forward afterward
    reproduces the pre-probe prediction exactly."""
    resid_audit._assert_eval_only()
    primitive = resid_audit._load_p_primitive(core, "I01", REC004F_DECISIVE_STEP)
    if primitive is None:
        return {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
    ckpt_path = (
        resid_audit.REC004E_SOURCE_RUN_DIR
        / "I01" / resid_audit.REC004E_ARM_P / "checkpoints" / f"step{REC004F_DECISIVE_STEP}.pt"
    )
    ckpt_hash_before = mb.raw_file_sha256(ckpt_path)
    before_hash = mb.canonical_state_hash(primitive.state_dict())

    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    d_model = primitive.content_in_proj.in_features
    torch.manual_seed(2026)
    content = torch.randn(len(content_lengths), lmax, d_model, device=core.device)
    with torch.no_grad():
        pre_logits = primitive(content, content_lengths, output_lengths, None)
        pre_preds = resid_audit._predict_from_logits(pre_logits, output_lengths)
        for idx, n in enumerate(content_lengths):
            single_content = content[idx : idx + 1, :n, :]
            run_oracle_forward(primitive, single_content, [n], [n])
        post_logits = primitive(content, content_lengths, output_lengths, None)
        post_preds = resid_audit._predict_from_logits(post_logits, output_lengths)
    after_hash = mb.canonical_state_hash(primitive.state_dict())
    ckpt_hash_after = mb.raw_file_sha256(ckpt_path)
    return {
        "task_id": REC004F_TASK_ID,
        "weights_unchanged": before_hash == after_hash,
        "checkpoint_file_unchanged": ckpt_hash_before == ckpt_hash_after,
        "normal_forward_restored_after_probe": pre_preds == post_preds,
        "logit_diff_before_after": (pre_logits - post_logits).abs().max().item(),
    }


# =============================================================================
# Diagnosis -- mechanically derived from the pre-registered thresholds
# above; never a forced single conclusion, always reported per init.
# =============================================================================


def build_oracle_probe_diagnosis(results: dict[str, Any]) -> dict[str, Any]:
    per_length_rows: dict[int, list[dict[str, Any]]] = {n: [] for n in REC004F_LEGAL_LENGTHS}
    for per_length in results["per_init_summary"].values():
        if not isinstance(per_length, dict):
            continue
        for n in REC004F_LEGAL_LENGTHS:
            row = per_length.get(str(n))
            if row is not None:
                per_length_rows[n].append(row)

    def _label_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"label": "EVIDENCE_INSUFFICIENT", "n_inits": 0}
        deltas = [r["paired_delta_oracle_minus_j0"] for r in rows]
        oracle_ems = [r["oracle_sequence_exact_match"] for r in rows]
        all_recover = all(
            d >= REC004F_RECOVERY_DELTA_THRESHOLD for d in deltas
        ) and all(em >= REC004F_RECOVERY_EM_THRESHOLD for em in oracle_ems)
        none_recover = all(d < REC004F_RECOVERY_DELTA_THRESHOLD for d in deltas)
        if all_recover:
            label = "ATTENTION_DISTRIBUTION_IS_THE_BOTTLENECK"
        elif none_recover:
            label = "RESIDUAL_IS_DOWNSTREAM_OF_ATTENTION"
        else:
            label = "MIXED_ACROSS_INITS"
        return {
            "label": label,
            "n_inits": len(rows),
            "mean_paired_delta": sum(deltas) / len(deltas),
            "mean_oracle_em": sum(oracle_ems) / len(oracle_ems),
            "per_init": {r["init_id"]: r for r in rows},
        }

    per_length_diagnosis = {str(n): _label_for(per_length_rows[n]) for n in REC004F_LEGAL_LENGTHS}
    return {
        "task_id": REC004F_TASK_ID,
        "recovery_delta_threshold": REC004F_RECOVERY_DELTA_THRESHOLD,
        "recovery_em_threshold": REC004F_RECOVERY_EM_THRESHOLD,
        "per_length": per_length_diagnosis,
        "length_10_label": per_length_diagnosis["10"]["label"],
    }


# =============================================================================
# Orchestration
# =============================================================================


def _config_to_yaml_dict(config: OracleAttentionSubstitutionProbeConfig) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_oracle_attention_substitution_probe_task(
    config: OracleAttentionSubstitutionProbeConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_oracle_attention_substitution_probe_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004F reads REC-004D/REC-004E artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; got seed={seed}"
        )

    authorization = verify_authorization()
    _write_json(output_dir / "authorization.json", authorization)
    if not authorization["authorized"]:
        result: dict[str, Any] = {
            "implementation_status": "BLOCKED",
            "probe_status": "NOT_EXECUTED",
            "diagnosis": "EVIDENCE_INSUFFICIENT",
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_intervention": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
            "authorization": authorization,
        }
        _write_json(output_dir / "summary.json", result)
        return result

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    loaded = mb.load_bundle(parent_manifest, mode="diagnostic", expected_primitive_count=16)
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    protected_hashes_before = bias_repair._protected_scope_hashes(eval_bank, op_to_id)

    with torch.no_grad():
        results = run_oracle_attention_substitution_probe(core)
    with (output_dir / "oracle_attention_substitution_results.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for row in results["rows"]:
            fh.write(json.dumps(row, default=str) + "\n")
    _write_json(
        output_dir / "oracle_attention_substitution_summary.json", results["per_init_summary"]
    )

    j0_cross_check = cross_check_j0_against_rec004e(results["rows"])
    _write_json(output_dir / "j0_cross_check_against_rec004e.json", j0_cross_check)

    with torch.no_grad():
        nonmutation_audit = run_probe_nonmutation_audit(core)
    _write_json(output_dir / "probe_nonmutation_audit.json", nonmutation_audit)

    diagnosis = build_oracle_probe_diagnosis(results)
    _write_json(output_dir / "oracle_probe_diagnosis.json", diagnosis)

    protected_hashes_after = bias_repair._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004F_TASK_ID,
        "core_canonical_state_hash_before": parent_manifest.core.canonical_state_hash,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == parent_manifest.core.canonical_state_hash,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "loaded_bundle_checks_performed": list(loaded.checks_performed),
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004F_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    cost_accounting = {
        "task_id": REC004F_TASK_ID,
        "new_optimizer_updates": 0,
        "n_forward_predictions": len(REC004F_INIT_IDS)
        * REC004F_PER_LENGTH
        * len(REC004F_LEGAL_LENGTHS)
        * 2,  # J0 + oracle, per example
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=seed)
    _write_json(output_dir / "system.json", system_info)

    protocol = {
        "task_id": REC004F_TASK_ID,
        "source_task_id": REC004F_SOURCE_TASK_ID,
        "contract_file": str(REC004F_CONTRACT_FILE),
        "checkpoint_step": REC004F_DECISIVE_STEP,
        "legal_lengths": list(REC004F_LEGAL_LENGTHS),
        "per_length_examples": REC004F_PER_LENGTH,
        "recovery_delta_threshold": REC004F_RECOVERY_DELTA_THRESHOLD,
        "recovery_em_threshold": REC004F_RECOVERY_EM_THRESHOLD,
        "forbidden": [
            "new optimizer updates", "production feature/attention changes",
            "candidate adoption", "child assembly", "RG3 recheck",
            "any further repair auto-started from this probe's result",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)

    result = {
        "implementation_status": "COMPLETE",
        "probe_status": "COMPLETE",
        "j0_cross_check_status": j0_cross_check["status"],
        "diagnosis": diagnosis["length_10_label"],
        "diagnosis_full": diagnosis,
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_intervention": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "authorization": authorization,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "probe_nonmutation_audit": nonmutation_audit,
        "cost_accounting": cost_accounting,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "summary.json", result)
    return result
