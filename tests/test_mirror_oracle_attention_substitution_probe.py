"""CPU-only contract tests for Task B-C005REC-004F (Oracle-Attention
Substitution Probe).

Per AGENTS.md ("Preserve CPU-testable logic even when milestone runs use
CUDA"): the real probe against REC-004D's saved 5-init step=6000
checkpoints is never invoked from this test module (it is dispatched
separately via `scripts/run_phase_b_b2_model_bundle_recovery.py --task
B-C005REC-004F`). These tests exercise the real production functions
against freshly-constructed, tiny, CPU-only primitives and synthetic
inputs -- never optimizer/training code, since this task performs zero new
optimizer updates by design.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from apc.evaluation import mirror_oracle_attention_substitution_probe as m
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _tiny_primitive(seed: int = 0) -> CrossPositionLengthBiasPrimitive:
    torch.manual_seed(seed)
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        bias_hidden_dim=resid_audit.REC004E_BIAS_HIDDEN_DIM,
        length_ref=resid_audit.REC004E_LENGTH_REF,
    )
    primitive = CrossPositionLengthBiasPrimitive(resid_audit.REC004E_TARGET_PHYSICAL_ID, cfg)
    with torch.no_grad():
        primitive.position_bias_out.weight.normal_(mean=0.0, std=0.2)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def _tiny_batch(n: int, d_model: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, n, d_model)


# ---------------------------------------------------------------------------
# 1. Oracle substitution matches pi_n exactly and is a valid distribution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 7, 8, 9, 10])
def test_oracle_attention_is_one_hot_over_pi_n(n: int) -> None:
    primitive = _tiny_primitive()
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    query, kv, out_max, lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content, [n], [n]
    )
    with torch.no_grad():
        _attn_out, attn_probs = m._oracle_attention(primitive, query, kv, n)
    pi = mpid.mirror_halves_position_map(n)
    assert attn_probs.shape == (1, primitive.n_head, n, n)
    assert torch.allclose(attn_probs.sum(dim=-1), torch.ones(1, primitive.n_head, n))
    for i, j in enumerate(pi):
        row = attn_probs[0, :, i, :]
        assert torch.all(row[:, j] == 1.0)
        assert torch.all(row.sum(dim=-1) == 1.0)


def test_oracle_attention_shares_the_same_distribution_across_heads() -> None:
    primitive = _tiny_primitive()
    n = 10
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    query, kv, _out_max, _lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content, [n], [n]
    )
    with torch.no_grad():
        _attn_out, attn_probs = m._oracle_attention(primitive, query, kv, n)
    for h in range(1, primitive.n_head):
        assert torch.equal(attn_probs[0, 0], attn_probs[0, h])


# ---------------------------------------------------------------------------
# 2. Oracle attn_out is exactly the real V/out_proj at the pi_n-selected key
#    -- the only thing replaced is the attention distribution, nothing in
#    the value/out_proj path.
# ---------------------------------------------------------------------------


def test_oracle_attn_out_matches_real_value_projection_at_pi_n() -> None:
    primitive = _tiny_primitive()
    n = 10
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    query, kv, _out_max, _lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content, [n], [n]
    )
    with torch.no_grad():
        attn_out, _attn_probs = m._oracle_attention(primitive, query, kv, n)
        mha = primitive.cross_attn
        _, _, wv = mha.in_proj_weight.chunk(3, dim=0)
        bv = None
        if mha.in_proj_bias is not None:
            _, _, bv = mha.in_proj_bias.chunk(3, dim=0)
        v_full = F.linear(kv, wv, bv)  # [1, n, embed_dim]
        pi = mpid.mirror_halves_position_map(n)
        expected = torch.stack([mha.out_proj(v_full[0, j]) for j in pi], dim=0).unsqueeze(0)
    assert torch.allclose(attn_out, expected, atol=1e-5)


def test_oracle_attention_rejects_padded_or_mismatched_batch() -> None:
    primitive = _tiny_primitive()
    query = torch.randn(1, 10, primitive.d_operator)
    kv = torch.randn(1, 10, primitive.d_operator)
    with torch.no_grad(), pytest.raises(AssertionError):
        m._oracle_attention(primitive, query, kv, 9)


def test_run_oracle_forward_rejects_mixed_length_batch() -> None:
    primitive = _tiny_primitive()
    d_model = primitive.content_in_proj.in_features
    lmax = 10
    content = torch.randn(2, lmax, d_model)
    with torch.no_grad(), pytest.raises(AssertionError):
        m.run_oracle_forward(primitive, content, [9, 10], [9, 10])


@pytest.mark.parametrize("n", [6, 7, 8, 9, 10])
def test_run_oracle_forward_end_to_end_shape(n: int) -> None:
    primitive = _tiny_primitive()
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    with torch.no_grad():
        out = m.run_oracle_forward(primitive, content, [n], [n])
    assert out["logits"].shape[0] == 1
    assert out["logits"].shape[1] == n


# ---------------------------------------------------------------------------
# 3. Dynamic guard: every forward path here must run with gradients
#    disabled, exactly like REC-004E's own guard.
# ---------------------------------------------------------------------------


def test_oracle_attention_raises_under_grad_enabled() -> None:
    primitive = _tiny_primitive()
    n = 6
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    query, kv, _out_max, _lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content, [n], [n]
    )
    with torch.enable_grad(), pytest.raises(RuntimeError, match="REC004E_TRAINING_GUARD"):
        m._oracle_attention(primitive, query, kv, n)


def test_run_oracle_forward_raises_under_grad_enabled() -> None:
    primitive = _tiny_primitive()
    n = 6
    d_model = primitive.content_in_proj.in_features
    content = _tiny_batch(n, d_model)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="REC004E_TRAINING_GUARD"):
        m.run_oracle_forward(primitive, content, [n], [n])


# ---------------------------------------------------------------------------
# 4. Non-mutation: running the probe repeatedly never changes primitive
#    weights and a normal forward is restored exactly afterward.
# ---------------------------------------------------------------------------


def test_running_oracle_forward_never_mutates_primitive_weights() -> None:
    from apc.utils import model_bundle as mb

    primitive = _tiny_primitive()
    before_hash = mb.canonical_state_hash(primitive.state_dict())
    content_lengths = [6, 7, 8, 9, 10]
    d_model = primitive.content_in_proj.in_features
    torch.manual_seed(7)
    full_content = torch.randn(len(content_lengths), max(content_lengths), d_model)
    with torch.no_grad():
        pre_logits = primitive(full_content, content_lengths, content_lengths, None)
        pre_preds = resid_audit._predict_from_logits(pre_logits, content_lengths)
        for idx, n in enumerate(content_lengths):
            single = full_content[idx : idx + 1, :n, :]
            m.run_oracle_forward(primitive, single, [n], [n])
        post_logits = primitive(full_content, content_lengths, content_lengths, None)
        post_preds = resid_audit._predict_from_logits(post_logits, content_lengths)
    after_hash = mb.canonical_state_hash(primitive.state_dict())
    assert before_hash == after_hash
    assert pre_preds == post_preds
    assert torch.equal(pre_logits, post_logits)


# ---------------------------------------------------------------------------
# 5. Diagnosis labeling: mechanically derived from pre-registered
#    thresholds, evidence-backed, never a forced single conclusion.
# ---------------------------------------------------------------------------


def _fake_per_init_summary(delta10: float, em10: float, n_inits: int = 5) -> dict:
    return {
        f"I0{i}": {
            "10": {
                "init_id": f"I0{i}", "length": 10, "n": 256,
                "j0_sequence_exact_match": max(0.0, em10 - delta10),
                "oracle_sequence_exact_match": em10,
                "paired_delta_oracle_minus_j0": delta10,
                "oracle_token_accuracy": 0.9, "oracle_loss": 0.1,
                "j0_only_correct": 0, "oracle_only_correct": 0,
                "both_correct": 0, "both_wrong": 0,
            }
        }
        for i in range(1, n_inits + 1)
    }


def test_diagnosis_labels_bottleneck_when_all_inits_recover() -> None:
    results = {"per_init_summary": _fake_per_init_summary(delta10=0.5, em10=0.98)}
    diagnosis = m.build_oracle_probe_diagnosis(results)
    assert diagnosis["length_10_label"] == "ATTENTION_DISTRIBUTION_IS_THE_BOTTLENECK"


def test_diagnosis_labels_downstream_when_no_init_recovers() -> None:
    results = {"per_init_summary": _fake_per_init_summary(delta10=0.01, em10=0.3)}
    diagnosis = m.build_oracle_probe_diagnosis(results)
    assert diagnosis["length_10_label"] == "RESIDUAL_IS_DOWNSTREAM_OF_ATTENTION"


def test_diagnosis_labels_mixed_when_inits_disagree() -> None:
    summary = _fake_per_init_summary(delta10=0.5, em10=0.98)
    summary["I05"]["10"]["paired_delta_oracle_minus_j0"] = 0.01
    summary["I05"]["10"]["oracle_sequence_exact_match"] = 0.3
    diagnosis = m.build_oracle_probe_diagnosis({"per_init_summary": summary})
    assert diagnosis["length_10_label"] == "MIXED_ACROSS_INITS"


def test_diagnosis_is_evidence_insufficient_on_empty_results_not_a_fabricated_label() -> None:
    diagnosis = m.build_oracle_probe_diagnosis({"per_init_summary": {}})
    for n in resid_audit.REC004E_LEGAL_LENGTHS:
        assert diagnosis["per_length"][str(n)]["label"] == "EVIDENCE_INSUFFICIENT"
        assert diagnosis["per_length"][str(n)]["n_inits"] == 0
    assert diagnosis["length_10_label"] == "EVIDENCE_INSUFFICIENT"


def test_diagnosis_thresholds_are_pre_registered_not_derived() -> None:
    assert m.REC004F_RECOVERY_DELTA_THRESHOLD == 0.05
    assert m.REC004F_RECOVERY_EM_THRESHOLD == 0.95


# ---------------------------------------------------------------------------
# 6. J0 cross-check against REC-004E's own saved results.
# ---------------------------------------------------------------------------


def test_cross_check_reports_unavailable_when_rec004e_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004E_RUN_DIR", tmp_path / "does_not_exist")
    rows = [{"init_id": "I01", "length": 10, "j0_sequence_exact_match": 0.5}]
    out = m.cross_check_j0_against_rec004e(rows)
    assert out["status"] == "REC004E_ARTIFACT_UNAVAILABLE"


def test_cross_check_detects_a_fabricated_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004E_RUN_DIR", tmp_path)
    recorded = {"I01": {"J0": {"per_length": {"10": {"sequence_exact_match": 0.9999}}}}}
    (tmp_path / "paired_intervention_summary.json").write_text(json.dumps(recorded))
    rows = [{"init_id": "I01", "length": 10, "j0_sequence_exact_match": 0.1234}]
    out = m.cross_check_j0_against_rec004e(rows)
    assert out["status"] == "MISMATCH"
    assert out["mismatches"][0]["init_id"] == "I01"


def test_cross_check_verifies_a_matching_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "REC004E_RUN_DIR", tmp_path)
    recorded = {"I01": {"J0": {"per_length": {"10": {"sequence_exact_match": 0.325}}}}}
    (tmp_path / "paired_intervention_summary.json").write_text(json.dumps(recorded))
    rows = [{"init_id": "I01", "length": 10, "j0_sequence_exact_match": 0.325}]
    out = m.cross_check_j0_against_rec004e(rows)
    assert out["status"] == "VERIFIED"
    assert out["mismatches"] == []


# ---------------------------------------------------------------------------
# 7. Authorization trail.
# ---------------------------------------------------------------------------


def test_verify_authorization_against_the_real_contract_file() -> None:
    out = m.verify_authorization()
    assert out["authorized"] is True
    assert out["status"] == "AUTHORIZED"
    assert "PROPOSED_NOT_AUTHORIZED" in out["contract_status_line"]


def test_verify_authorization_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(m, "REC004F_CONTRACT_FILE", tmp_path / "missing.md")
    out = m.verify_authorization()
    assert out["authorized"] is False
    assert out["status"] == "CONTRACT_FILE_MISSING"


def test_verify_authorization_status_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "next_repair_contract.md"
    fake.write_text("status: SOMETHING_ELSE\n")
    monkeypatch.setattr(m, "REC004F_CONTRACT_FILE", fake)
    out = m.verify_authorization()
    assert out["authorized"] is False
    assert out["status"] == "CONTRACT_STATUS_MISMATCH"


# ---------------------------------------------------------------------------
# 8. The critical contract invariant: `mirror_halves_position_map` (the
#    oracle position map) is called from exactly one function in this
#    module -- `_oracle_attention` -- and nowhere else. This is the
#    module's disclosed, deliberate exception to REC-004E's rule against
#    feeding `pi_n` into any forward input; it must not leak into any other
#    function (checkpoint loading, example generation, diagnosis/scoring).
# ---------------------------------------------------------------------------


def test_pi_n_lookup_is_confined_to_oracle_attention_only() -> None:
    source_path = Path(inspect.getfile(m))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            calls_pi_n = any(
                isinstance(n, ast.Call)
                and (
                    (isinstance(n.func, ast.Name) and n.func.id == "mirror_halves_position_map")
                    or (
                        isinstance(n.func, ast.Attribute)
                        and n.func.attr == "mirror_halves_position_map"
                    )
                )
                for n in ast.walk(node)
            )
            if calls_pi_n and node.name != "_oracle_attention":
                offending.append(node.name)
    assert offending == []


def test_module_never_constructs_an_optimizer_or_calls_backward() -> None:
    source_path = Path(inspect.getfile(m))
    text = source_path.read_text(encoding="utf-8")
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert "requires_grad_(True)" not in text


def test_module_never_dispatches_or_imports_a_later_task() -> None:
    """The module docstring DISCLOSES that REC-005/R3-011/B-C006 remain
    blocked (documentation, not a violation) -- this test instead confirms
    there is no executable hook (function definition, import, or dispatch
    call) that would actually start one."""
    source_path = Path(inspect.getfile(m))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    defined_names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for forbidden in ("rec005", "r3_011", "b_c006"):
        assert not any(forbidden in name.lower() for name in defined_names)
        assert not any(forbidden in name.lower() for name in imported_names)


# ---------------------------------------------------------------------------
# 9. Same suite/checkpoints as REC-004E -- pre-registered, not re-derived.
# ---------------------------------------------------------------------------


def test_reuses_rec004e_constants_exactly() -> None:
    assert m.REC004F_INIT_IDS == resid_audit.REC004E_INIT_IDS
    assert m.REC004F_LEGAL_LENGTHS == resid_audit.REC004E_LEGAL_LENGTHS
    assert m.REC004F_DECISIVE_STEP == resid_audit.REC004E_DECISIVE_STEP
    assert m.REC004F_TARGET_OPERATION == resid_audit.REC004E_TARGET_OPERATION
    assert m.REC004F_PER_LENGTH == resid_audit.REC004E_INTERVENTION_PER_LENGTH


def test_no_query_or_teacher_argument_in_public_forward_signatures() -> None:
    for fn in (m.run_oracle_forward, m._oracle_attention):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"pi_n", "teacher", "query_target", "target_tokens", "labels"}
