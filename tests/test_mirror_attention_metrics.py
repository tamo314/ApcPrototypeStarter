"""Semantic observer regressions independent of trained weights and optimizer loops."""

from types import SimpleNamespace

import pytest
import torch

from apc.environments.operations import get_operation
from apc.evaluation import mirror_parallel_score_residual_pilot as ac
from apc.evaluation.mirror_attention_metrics import matched_score_norms, position_statistics


@pytest.mark.parametrize("length", [6, 8, 10])
def test_oracle_attention_and_wrong_identity_at_every_position(length: int) -> None:
    expected = get_operation("MIRROR_HALVES").apply(tuple(range(length)), length, {})
    scores = torch.zeros(1, 4, length, length + 3)
    for output, source in enumerate(expected):
        scores[:, :, output, source] = 10.0
    scores[..., length:] = 1000  # A padded key may never outrank a valid key.
    probs = scores[..., :length].softmax(-1)
    probs = torch.nn.functional.pad(probs, (0, 3))
    for output, source in enumerate(expected):
        result = position_statistics(scores, probs, [length], output)
        assert torch.all(result["rank"] == 1)
        assert torch.all(result["margin"] == 10)
        assert torch.all(result["probability"] > 0.999)
        wrong = torch.zeros_like(scores)
        wrong[:, :, output, output] = 10.0
        observed = position_statistics(wrong, probs, [length], output)
        assert torch.all(observed["margin"] == (10 if source == output else -10))


def test_head_recall_and_ties_do_not_round_mean_rank() -> None:
    scores = torch.zeros(1, 4, 10, 10)
    scores[:, :, 4, 0] = 10
    scores[:, 1::2, 4, 1] = 11
    ranks = position_statistics(scores, scores.softmax(-1), [10], 4)["rank"]
    assert ranks.tolist() == [[1, 2, 1, 2]]
    assert (ranks <= 1).float().mean().item() == 0.5
    scores[:, :, 4, 1] = 10  # Tie with a wrong key: not certain top-1.
    ranks = position_statistics(scores, scores.softmax(-1), [10], 4)["rank"]
    assert ranks.tolist() == [[2, 2, 2, 2]]


def test_norm_is_head_count_and_row_offset_invariant() -> None:
    torch.manual_seed(42)
    base = torch.randn(2, 1, 10, 12)
    residual = torch.randn(2, 10, 12)
    lengths = [6, 10]
    a = matched_score_norms(base, residual, lengths)
    b = matched_score_norms(base.expand(-1, 4, -1, -1), residual, lengths)
    assert torch.allclose(a["centered"], b["centered"])
    c = matched_score_norms(base + 50, residual + 20, lengths)
    assert torch.allclose(a["centered"], c["centered"], atol=1e-5)
    base[:, :, :, 10:] = torch.nan
    residual[:, :, 10:] = torch.nan
    d = matched_score_norms(base, residual, lengths)
    assert torch.allclose(a["centered"], d["centered"])
    zero = matched_score_norms(torch.zeros_like(base), residual, lengths)
    assert torch.isnan(zero["centered"]).all()


def test_invalid_position_is_rejected() -> None:
    scores = torch.zeros(1, 4, 10, 10)
    with pytest.raises(ValueError, match="valid"):
        position_statistics(scores, scores, [6], 6)


def test_ac_evaluator_reports_correct_mapping_and_preserves_perfect_outputs(monkeypatch):
    """Exercise the actual AC aggregator, not just a helper's constants."""
    target = get_operation("MIRROR_HALVES").apply(tuple(range(10)), 10, {})
    examples = [SimpleNamespace(input_tokens=tuple(range(10)), target_tokens=target)] * 2
    scores = torch.zeros(2, 4, 10, 10)
    for i, j in enumerate(target):
        scores[:, :, i, j] = 10
    logits = torch.nn.functional.one_hot(torch.tensor([target, target]), 10).float() * 10
    stages = {
        "final_token_logits": logits, "score_logits": scores,
        "attn_probs": scores.softmax(-1), "s_base": scores,
        "delta_s": torch.zeros(2, 10, 10), "s_total": scores,
    }
    monkeypatch.setattr(ac, "collate_content_only_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(ac, "evaluate_parallel_score_residual_forward_with_stages",
                        lambda *args, **kwargs: stages)
    core = SimpleNamespace(device=torch.device("cpu"), tokens=None,
                           model=SimpleNamespace(encode=lambda _: torch.zeros(2, 11, 4)))
    primitive = SimpleNamespace(eval=lambda: None, n_head=4)
    before = scores.clone()
    observed = ac.evaluate_length10_metrics(core, primitive, examples)
    assert observed["j0_sequence_em"] == observed["o1_sequence_em"] == 1.0
    assert observed["j0_p4_score_margin_median"] == 10.0
    assert observed["j0_p4_top1_key_recall"] == 1.0
    assert observed["score_decomposition"]["residual_margin_contribution_mean"] == 0.0
    assert torch.equal(before, scores)
