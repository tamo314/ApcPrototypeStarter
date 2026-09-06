"""Unit tests for argument compatibility scoring (Phase B Task B-C005R1;
extended by Task B-C005R3-007 for the SELECT sigmoid/softmax repair)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from apc.environments.task_spec import TaskSpec, TaskStepSpec
from apc.primitives.argument_scoring import (
    ArgumentScorer,
    ArgumentScorerConfig,
    extract_raw_argument_values,
)


def _select_example(indices: list[int]) -> Any:
    """Minimal duck-typed stand-in for `Example`: `train_on_examples` only
    ever reads `.task_spec.steps[0].arguments`."""
    step = TaskStepSpec(operation="SELECT", arguments={"indices": indices})
    return SimpleNamespace(task_spec=TaskSpec(steps=(step,)))


def test_extract_raw_argument_values() -> None:
    assert extract_raw_argument_values("SHIFT", {"amount": 3}) == [3]
    assert extract_raw_argument_values("COUNT", {"target": 5}) == [5]
    assert extract_raw_argument_values("BIND", {"query_key": 2}) == [2]
    assert extract_raw_argument_values("SELECT", {"indices": [1, 4]}) == [1, 4]
    assert extract_raw_argument_values("IDENTITY", {}) == []


def test_argument_scorer_forward_parameter_free_vs_parameterized() -> None:
    config = ArgumentScorerConfig(d_model=16, arg_vocab_size=10, lambda_weight=1.5)
    scorer = ArgumentScorer(config)

    z_task = torch.randn(2, 16)

    # Parameter-free operation returns exact zero
    score_free = scorer(z_task, "IDENTITY", None)
    assert torch.equal(score_free, torch.zeros(2))

    # Parameterized operation returns finite tensor
    score_param = scorer(z_task, "SHIFT", {"amount": 2})
    assert score_param.shape == (2,)
    assert not torch.isnan(score_param).any()


def test_argument_scorer_discrimination() -> None:
    """A trained scorer must assign higher score to matching argument than wrong argument."""
    config = ArgumentScorerConfig(d_model=32, arg_vocab_size=10, lr=0.05, steps=50)
    scorer = ArgumentScorer(config)

    # Create synthetic z_task representing distinct amount values
    # e.g. amount 2 has distinct representation from amount 5
    z_task_2 = torch.zeros(10, 32)
    z_task_2[:, 2] = 5.0  # signal at index 2

    # Dummy training: manually set head weights to discriminate
    scorer.heads["SHIFT"].weight.data.zero_()
    scorer.heads["SHIFT"].weight.data[:10, :10] = torch.eye(10) * 10.0
    scorer.heads["SHIFT"].bias.data.zero_()

    score_correct = scorer(z_task_2, "SHIFT", {"amount": 2})
    score_wrong = scorer(z_task_2, "SHIFT", {"amount": 5})

    assert (score_correct > score_wrong).all()


def test_argument_scorer_select_uses_independent_sigmoid_not_softmax() -> None:
    """SELECT trains under `BCEWithLogitsLoss` (independent per-index
    targets, see `train_on_examples` below); `forward` must score it with
    independent sigmoids, not a shared softmax that forces every
    simultaneously-true index to compete for one unit of probability mass
    (the ADR-0081 `ARGUMENT_ENCODING_FAILURE` train/inference mismatch,
    repaired by Task B-C005R3-007 / ADR-0088)."""
    config = ArgumentScorerConfig(d_model=8, arg_vocab_size=16)
    scorer = ArgumentScorer(config)
    z_task = torch.randn(3, 8)

    with torch.no_grad():
        logits = scorer.heads["SELECT"](z_task)
        expected_probs = torch.sigmoid(logits)
        # Softmax must differ from sigmoid here (logits are not degenerate) --
        # confirms this test actually distinguishes the two formulas.
        assert not torch.allclose(expected_probs, F.softmax(logits, dim=-1))
        val_idx = torch.tensor([[3]]).expand(3, 1)
        expected_p = expected_probs.gather(dim=-1, index=val_idx).squeeze(-1)
        expected_score = 2.0 * (expected_p - 0.5)

    score = scorer(z_task, "SELECT", {"indices": [3]})
    assert torch.allclose(score, expected_score, atol=1e-6)


def test_argument_scorer_non_select_ops_still_use_softmax() -> None:
    """Regression/freeze test: the SELECT-specific branch must not change
    SHIFT/COUNT/BIND's existing shared-softmax scoring formula."""
    config = ArgumentScorerConfig(d_model=8, arg_vocab_size=16)
    scorer = ArgumentScorer(config)
    z_task = torch.randn(3, 8)

    op_arg_val = (("SHIFT", "amount", 2), ("COUNT", "target", 5), ("BIND", "query_key", 7))
    for op, arg_key, arg_val in op_arg_val:
        with torch.no_grad():
            logits = scorer.heads[op](z_task)
            expected_probs = F.softmax(logits, dim=-1)
            val_idx = torch.tensor([[arg_val]]).expand(3, 1)
            expected_p = expected_probs.gather(dim=-1, index=val_idx).squeeze(-1)
            expected_score = 2.0 * (expected_p - 0.5)
        score = scorer(z_task, op, {arg_key: arg_val})
        assert torch.allclose(score, expected_score, atol=1e-6)


def test_argument_scorer_select_discriminates_multi_index_targets() -> None:
    """A trained SELECT head must assign higher score to the correct
    multi-index set than to a shifted-by-one wrong set (the exact competitor
    `hard_negative_routing_benchmark._wrong_arguments` builds for SELECT) --
    extending `test_argument_scorer_discrimination`'s SHIFT-only coverage to
    the one set-valued operation this repair targets."""
    config = ArgumentScorerConfig(d_model=32, arg_vocab_size=10, lr=0.05, steps=300)
    scorer = ArgumentScorer(config)

    z_a = torch.zeros(1, 32)
    z_a[0, 0] = 5.0
    z_b = torch.zeros(1, 32)
    z_b[0, 1] = 5.0
    indices_a = [1, 3]
    indices_b = [2, 5]

    z_batch = torch.cat([z_a, z_b], dim=0)
    examples = [_select_example(indices_a), _select_example(indices_b)]
    scorer.train_on_examples({"SELECT": z_batch}, {"SELECT": examples})

    score_a_correct = scorer(z_a, "SELECT", {"indices": indices_a})
    score_a_wrong = scorer(z_a, "SELECT", {"indices": [(i + 1) % 10 for i in indices_a]})
    assert (score_a_correct > score_a_wrong).all()

    score_b_correct = scorer(z_b, "SELECT", {"indices": indices_b})
    score_b_wrong = scorer(z_b, "SELECT", {"indices": [(i + 1) % 10 for i in indices_b]})
    assert (score_b_correct > score_b_wrong).all()


def test_select_sigmoid_scoring_avoids_legacy_softmax_dilution() -> None:
    """Characterizes the ADR-0081 bug directly, isolated from any training
    noise or benchmark-specific competitor difficulty: on a hand-built logit
    vector where several indices are simultaneously, maximally confident
    (well-separated logits, not a trained checkpoint), a shared softmax
    dilutes each true index's score because they must share one unit of
    total probability mass, while independent sigmoids do not."""
    arg_vocab_size = 32
    correct = [1, 5, 9, 13]  # 4 simultaneously "true" indices
    logits = torch.full((1, arg_vocab_size), -6.0)
    for i in correct:
        logits[0, i] = 6.0

    def legacy_softmax_score(logits: torch.Tensor, indices: list[int]) -> float:
        probs = F.softmax(logits, dim=-1)
        vals = [2.0 * (probs[0, i].item() - 0.5) for i in indices]
        return sum(vals) / len(vals)

    def sigmoid_score(logits: torch.Tensor, indices: list[int]) -> float:
        probs = torch.sigmoid(logits)
        vals = [2.0 * (probs[0, i].item() - 0.5) for i in indices]
        return sum(vals) / len(vals)

    assert sigmoid_score(logits, correct) > 0.99
    assert legacy_softmax_score(logits, correct) < 0.5
