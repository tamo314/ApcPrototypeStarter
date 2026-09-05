"""Unit tests for argument compatibility scoring (Phase B Task B-C005R1)."""

from __future__ import annotations

import torch

from apc.primitives.argument_scoring import (
    ArgumentScorer,
    ArgumentScorerConfig,
    extract_raw_argument_values,
)


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
