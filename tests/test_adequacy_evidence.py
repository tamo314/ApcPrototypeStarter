"""Tests for Functional Adequacy Evidence Interface (Phase A.2 Task A2-C005).

Verifies:
1. Strict zero oracle leakage:
   - Stripping `oracle_metadata` and `program` produces bitwise identical evidence.
   - Mutating `oracle_metadata.label` has zero effect.
   - No oracle fields or held-out targets are exposed in `AdequacyEvidence`.
2. Deterministic execution:
   - Repeated calls on identical support sets produce deterministic evidence.
3. Behavioral evidence profiles across K/C/N/R fixtures:
   - K (Known): high direct EM (>= 0.95), low composition improvement (<= 0.05).
   - C (Composition): low direct EM (<= 0.20), high composition EM (>= 0.90),
     high composition improvement (>= 0.70), depth == 2.
   - N (Novel): low direct EM (<= 0.20), low composition EM (<= 0.20),
     low composition improvement (<= 0.10).
   - R (Recurrence): high direct EM (>= 0.95), low composition improvement (<= 0.05),
     high recurrence key similarity (>= 0.70).
4. Serialization and downstream controller inputs:
   - `to_feature_vector()` returns 13 finite numerical features (0 NaNs, 0 Infs).
   - `to_dict()` and `from_dict()` round-trip faithfully.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn as nn

from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import (
    TaskSpec,
    num_registered_operations,
    operation_id,
)
from apc.meta.adequacy import (
    AdequacyEvidence,
    AdequacyEvidenceConfig,
    compute_adequacy_evidence,
    extract_task_representations,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.primitives.router import Router, RouterConfig

# ---------------------------------------------------------------------------
# Test Fixture Helpers: Deterministic Transparent Core & Primitives
# ---------------------------------------------------------------------------


class _TransparentModel(nn.Module):
    """Simple deterministic encoder for CPU smoke testing."""

    def __init__(self, tokens: SharedCoreTokens, d_model: int = 64) -> None:
        super().__init__()
        self.tokens = tokens
        self.d_model = d_model
        # Embeddings: position 0 stores (token_id + 1.0)
        self.token_emb = nn.Embedding(tokens.model_vocab_size, d_model)
        with torch.no_grad():
            self.token_emb.weight.zero_()
            for tid in range(tokens.model_vocab_size):
                if tid < tokens.env_vocab_size:
                    self.token_emb.weight[tid, 0] = float(tid + 1)
                # For operation tokens, encode distinct feature in channels 16..31
                if tokens.op_base <= tid < tokens.arg_base:
                    op_idx = tid - tokens.op_base
                    self.token_emb.weight[tid, 16 + op_idx] = 5.0

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs: [batch, seq_len] -> [batch, seq_len, d_model]."""
        emb = self.token_emb(input_ids).clone()
        # In a transformer with causal attention, tokens at [TASK_END] pool prefix representations.
        # We simulate causal accumulation of task features in channels 16.. along dim=1:
        emb[:, :, 16:] = torch.cumsum(emb[:, :, 16:], dim=1)
        return emb


class _TransparentCore:
    """Lightweight deterministic Core wrapper."""

    def __init__(self, d_model: int = 64, vocab_size: int = 10) -> None:
        self.device = torch.device("cpu")
        num_ops = max(32, num_registered_operations())
        self.tokens = build_shared_core_tokens(
            env_vocab_size=vocab_size, num_operations=num_ops, arg_span=10
        )
        self.model = _TransparentModel(self.tokens, d_model=d_model)


class _ExactFunctionalPrimitive(CrossPositionPrimitive):
    """Test primitive that applies reference operation logic to recovered tokens."""

    def __init__(
        self,
        primitive_id: int,
        operation: str,
        config: CrossPositionPrimitiveConfig,
        vocab_size: int = 10,
    ) -> None:
        super().__init__(primitive_id=primitive_id, config=config, status=PrimitiveStatus.STABLE)
        self.operation = operation
        self.vocab_size = vocab_size

    def forward(
        self,
        h_content: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        arg_values: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        self.forward_call_count += 1
        batch_size = h_content.shape[0]
        max_out_len = max(output_lengths)
        logits = torch.full(
            (batch_size, max_out_len, self.vocab_size),
            fill_value=-20.0,
            device=h_content.device,
        )

        op_def = get_operation(self.operation)

        for b in range(batch_size):
            in_len = content_lengths[b]
            # Recover token IDs from channel 0
            recovered_tokens = tuple(
                int(round(h_content[b, pos, 0].item() - 1.0)) for pos in range(in_len)
            )

            # Resolve argument dictionary
            params: dict[str, Any] = {}
            if arg_values is not None and arg_values[b] is not None:
                arg_v = arg_values[b]
                if op_def.required_argument_names:
                    req_name = next(iter(op_def.required_argument_names))
                    params[req_name] = arg_v

            # Apply exact reference operation
            out_tokens = op_def.apply(recovered_tokens, self.vocab_size, params=params)

            # Produce high-confidence logits for correct output tokens
            for pos, tok in enumerate(out_tokens):
                if 0 <= tok < self.vocab_size and pos < max_out_len:
                    logits[b, pos, tok] = 20.0

        return logits


def _build_test_kcnr_environment() -> tuple[
    _TransparentCore,
    PrimitiveBank,
    Router,
    dict[str, int],
    dict[str, list[Example]],
]:
    """Construct an exact test environment with K, C, N, R tasks and calibrated router."""
    core = _TransparentCore(d_model=64, vocab_size=10)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}

    registered_ops = [
        "REVERSE",
        "NEGATE",
        "SWAP_PAIRS",  # Consolidated novel primitive (R)
        "SHIFT",
        "SELECT",
    ]

    for op in registered_ops:
        p_cfg = CrossPositionPrimitiveConfig(
            operation=op,
            d_model=64,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=32,
            arg_dim=8,
        )
        pid = len(bank)
        p = _ExactFunctionalPrimitive(primitive_id=pid, operation=op, config=p_cfg, vocab_size=10)
        bank.add_primitive(p)
        op_to_id[op] = pid

    # Calibrate router keys to match operation representations in channels 16..31
    router_cfg = RouterConfig(d_model=64, top_k=2, score_fn="dot")
    router = Router(router_cfg)

    # Initialize query_proj as identity
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(64))
        router.query_proj.bias.zero_()

    for pid in bank.ids():
        router.add_primitive_key(pid)
        op_name = [name for name, i in op_to_id.items() if i == pid][0]
        # In tokens: op_idx is given by operation_id(op_name)
        op_idx = operation_id(op_name)
        with torch.no_grad():
            key = torch.zeros(64)
            key[16 + op_idx] = 5.0
            router.key_parameter(pid).copy_(key)

    # Generate test support examples for K, C, N, R
    fixtures: dict[str, list[Example]] = {}

    # 1. K (Known task: REVERSE)
    k_prog = Program(steps=(ProgramStep("REVERSE"),))
    k_examples = []
    for seq in [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1), (0, 3, 7, 2, 9, 4), (5, 5, 1, 2, 8, 4)]:
        res = run_program(k_prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=k_prog,
            operation_graph=res.graph,
            category="known",
            split="train",
            vocab_size=10,
            task_spec=TaskSpec.from_program(k_prog),
            oracle_metadata=OracleMetadata(label="K", primitive_operations=("REVERSE",)),
        )
        k_examples.append(ex)
    fixtures["K"] = k_examples

    # 2. C (Composition task: REVERSE -> NEGATE)
    c_prog = Program(steps=(ProgramStep("REVERSE"), ProgramStep("NEGATE")))
    c_examples = []
    for seq in [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1), (0, 3, 7, 2, 9, 4), (5, 5, 1, 2, 8, 4)]:
        res = run_program(c_prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=c_prog,
            operation_graph=res.graph,
            category="novel_composition",
            split="test",
            vocab_size=10,
            task_spec=TaskSpec.from_program(c_prog),
            oracle_metadata=OracleMetadata(label="C", primitive_operations=("REVERSE", "NEGATE")),
        )
        c_examples.append(ex)
    fixtures["C"] = c_examples

    # 3. N (Novel task: CYCLE_FOUR, not in bank and not composable from registered ops)
    n_prog = Program(steps=(ProgramStep("CYCLE_FOUR"),))
    n_examples = []
    for seq in [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1), (0, 3, 7, 2, 9, 4), (5, 5, 1, 2, 8, 4)]:
        res = run_program(n_prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=n_prog,
            operation_graph=res.graph,
            category="novel_operation",
            split="test",
            vocab_size=10,
            task_spec=TaskSpec.from_program(n_prog),
            oracle_metadata=OracleMetadata(label="N", primitive_operations=("CYCLE_FOUR",)),
        )
        n_examples.append(ex)
    fixtures["N"] = n_examples

    # 4. R (Recurrence task: SWAP_PAIRS returning after consolidation)
    r_prog = Program(steps=(ProgramStep("SWAP_PAIRS"),))
    r_examples = []
    for seq in [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1), (0, 3, 7, 2, 9, 4), (5, 5, 1, 2, 8, 4)]:
        res = run_program(r_prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=r_prog,
            operation_graph=res.graph,
            category="novel_operation",
            split="test",
            vocab_size=10,
            task_spec=TaskSpec.from_program(r_prog),
            oracle_metadata=OracleMetadata(
                label="R",
                primitive_operations=("SWAP_PAIRS",),
                recurrence_operation="SWAP_PAIRS",
            ),
        )
        r_examples.append(ex)
    fixtures["R"] = r_examples

    return core, bank, router, op_to_id, fixtures


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_adequacy_evidence_dataclass_roundtrip() -> None:
    """Verify to_dict and from_dict roundtrip with numerical precision."""
    ev = AdequacyEvidence(
        direct_em=0.95,
        direct_loss=0.05,
        direct_token_acc=0.99,
        direct_primitive_id=1,
        direct_runner_up_em=0.10,
        direct_runner_up_loss=2.5,
        direct_margin=0.85,
        direct_candidates_evaluated=2,
        composition_em=0.95,
        composition_loss=0.05,
        composition_depth=1,
        composition_recipe=("REVERSE",),
        composition_improvement_em=0.0,
        composition_improvement_loss=0.0,
        composition_candidates_evaluated=5,
        composition_candidates_pruned=3,
        router_confidence=0.92,
        router_margin=0.88,
        router_entropy=0.15,
        proposed_primitive_id=1,
        runner_up_primitive_id=2,
        recurrence_key_similarity=0.95,
        prototype_similarity=0.88,
        support_size=16,
        compute_time_seconds=0.012,
    )

    d = ev.to_dict()
    assert isinstance(d, dict)
    assert d["direct_em"] == 0.95
    assert d["composition_recipe"] == ["REVERSE"]

    recovered = AdequacyEvidence.from_dict(d)
    assert recovered.direct_em == ev.direct_em
    assert recovered.composition_recipe == ev.composition_recipe
    assert recovered.router_confidence == ev.router_confidence
    assert recovered.prototype_similarity == ev.prototype_similarity


def test_adequacy_evidence_feature_vector_invariants() -> None:
    """Verify to_feature_vector returns 13 finite numeric features without NaNs or Infs."""
    ev = AdequacyEvidence(
        direct_em=1.0,
        direct_loss=float("inf"),  # test clamping
        direct_token_acc=1.0,
        direct_primitive_id=0,
        direct_margin=1.0,
        composition_em=0.0,
        composition_loss=float("inf"),  # test clamping
        composition_depth=1,
        composition_improvement_em=0.0,
        composition_improvement_loss=0.0,
        router_confidence=0.99,
        router_margin=0.98,
        router_entropy=0.05,
        recurrence_key_similarity=0.98,
    )

    features = ev.to_feature_vector()
    assert len(features) == 13
    assert all(isinstance(f, float) for f in features)
    assert all(math.isfinite(f) for f in features)
    # Check clamping on inf loss
    assert features[1] == 10.0
    assert features[5] == 10.0


def test_extract_task_representations_zero_oracle_leakage() -> None:
    """extract_task_representations must only consume task_spec."""
    core, _, _, _, fixtures = _build_test_kcnr_environment()
    examples = fixtures["K"]

    z1 = extract_task_representations(core, examples)

    # Redact oracle metadata and program completely
    redacted = [
        Example(
            input_tokens=ex.input_tokens,
            target_tokens=ex.target_tokens,
            program=None,  # type: ignore[arg-type]
            operation_graph=None,  # type: ignore[arg-type]
            category=ex.category,
            split=ex.split,
            vocab_size=ex.vocab_size,
            task_spec=ex.task_spec,
            oracle_metadata=None,
        )
        for ex in examples
    ]

    z2 = extract_task_representations(core, redacted)
    assert torch.equal(z1, z2)


def test_compute_adequacy_evidence_determinism() -> None:
    """Identical support sets must produce identical evidence records."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    k_examples = fixtures["K"]

    ev1 = compute_adequacy_evidence(core, bank, router, op_to_id, k_examples)
    ev2 = compute_adequacy_evidence(core, bank, router, op_to_id, k_examples)

    assert ev1.direct_em == ev2.direct_em
    assert ev1.direct_loss == ev2.direct_loss
    assert ev1.direct_margin == ev2.direct_margin
    assert ev1.composition_em == ev2.composition_em
    assert ev1.composition_depth == ev2.composition_depth
    assert ev1.composition_recipe == ev2.composition_recipe
    assert ev1.router_confidence == ev2.router_confidence
    assert ev1.router_margin == ev2.router_margin
    assert ev1.recurrence_key_similarity == ev2.recurrence_key_similarity


def test_compute_adequacy_evidence_leakage_prohibition() -> None:
    """Verify zero oracle leakage: redacting oracle metadata produces identical output."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    c_examples = fixtures["C"]

    ev_original = compute_adequacy_evidence(core, bank, router, op_to_id, c_examples)

    # Redact oracle metadata, program, and mutate category
    redacted = [
        Example(
            input_tokens=ex.input_tokens,
            target_tokens=ex.target_tokens,
            program=None,  # type: ignore[arg-type]
            operation_graph=None,  # type: ignore[arg-type]
            category="unknown",
            split="unknown",
            vocab_size=ex.vocab_size,
            task_spec=ex.task_spec,
            oracle_metadata=None,
        )
        for ex in c_examples
    ]

    ev_redacted = compute_adequacy_evidence(core, bank, router, op_to_id, redacted)

    assert ev_original.direct_em == ev_redacted.direct_em
    assert ev_original.direct_loss == ev_redacted.direct_loss
    assert ev_original.composition_em == ev_redacted.composition_em
    assert ev_original.composition_recipe == ev_redacted.composition_recipe
    assert ev_original.composition_improvement_em == ev_redacted.composition_improvement_em
    assert ev_original.router_confidence == ev_redacted.router_confidence

    # Verify no oracle fields exist in to_dict()
    d = ev_original.to_dict()
    for forbidden in ["oracle", "program", "label", "category", "target"]:
        assert not any(forbidden in k for k in d.keys())


def test_kcnr_fixtures_evidence_acceptance() -> None:
    """Acceptance criterion: K/C/N/R fixtures produce distinct, expected evidence profiles."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()

    # 1. K (Known task: REVERSE)
    ev_k = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["K"])
    assert ev_k.direct_em >= 0.95
    assert ev_k.direct_primitive_id == op_to_id["REVERSE"]
    assert ev_k.router_confidence >= 0.70
    assert ev_k.composition_improvement_em <= 0.05  # direct is already sufficient

    # 2. C (Composition task: REVERSE -> NEGATE)
    ev_c = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["C"])
    assert ev_c.direct_em <= 0.20  # direct primitive insufficient
    assert ev_c.composition_em >= 0.90  # composition solves it
    assert ev_c.composition_depth == 2  # recovered 2-step recipe
    assert ev_c.composition_improvement_em >= 0.70  # large improvement over direct
    assert ev_c.composition_recipe == ("REVERSE", "NEGATE")

    # 3. N (Novel task: CYCLE_FOUR)
    ev_n = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["N"])
    assert ev_n.direct_em <= 0.20  # direct primitive insufficient
    assert ev_n.composition_em <= 0.20  # composition insufficient
    assert ev_n.composition_improvement_em <= 0.10  # both fail

    # 4. R (Recurrence task: SWAP_PAIRS)
    ev_r = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["R"])
    assert ev_r.direct_em >= 0.95  # direct primitive solves it
    assert ev_r.direct_primitive_id == op_to_id["SWAP_PAIRS"]
    assert ev_r.composition_improvement_em <= 0.05  # direct sufficient
    assert ev_r.recurrence_key_similarity >= 0.70  # matches calibrated router key


def test_direct_margin_and_runner_up_evaluation() -> None:
    """Verify direct_margin correctly measures difference between top candidates."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    ev = compute_adequacy_evidence(
        core,
        bank,
        router,
        op_to_id,
        fixtures["K"],
        config=AdequacyEvidenceConfig(direct_eval_k=2),
    )

    assert ev.direct_candidates_evaluated == 2
    assert ev.direct_runner_up_em is not None
    assert ev.direct_margin == pytest.approx(ev.direct_em - ev.direct_runner_up_em, abs=1e-6)


def test_prototype_similarity_retrieval() -> None:
    """Verify prototype similarity computation when prototype embeddings are supplied."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    k_examples = fixtures["K"]

    z_k = extract_task_representations(core, k_examples).mean(dim=0)
    prototypes = {op_to_id["REVERSE"]: z_k}

    ev = compute_adequacy_evidence(
        core,
        bank,
        router,
        op_to_id,
        k_examples,
        prototypes_by_pid=prototypes,
    )

    assert ev.prototype_similarity is not None
    assert ev.prototype_similarity >= 0.99


def test_empty_support_examples_raises() -> None:
    """Passing empty support examples must raise ValueError."""
    core, bank, router, op_to_id, _ = _build_test_kcnr_environment()
    with pytest.raises(ValueError, match="support_examples must be non-empty"):
        compute_adequacy_evidence(core, bank, router, op_to_id, [])
