from __future__ import annotations

import pytest
import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.utils.seed import set_seed


def _tiny_config(**overrides: object) -> TransformerConfig:
    defaults: dict[str, object] = {
        "vocab_size": 14,
        "max_seq_len": 32,
        "d_model": 16,
        "n_layer": 2,
        "n_head": 2,
        "d_ff": 32,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)  # type: ignore[arg-type]


def test_forward_output_shape() -> None:
    model = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (3, 7))
    logits = model(input_ids)
    assert logits.shape == (3, 7, 14)


def test_max_seq_len_enforced() -> None:
    model = DecoderOnlyTransformer(_tiny_config(max_seq_len=5))
    input_ids = torch.randint(0, 14, (1, 6))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids)


def test_d_model_must_be_divisible_by_n_head() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(d_model=17, n_head=2)


def test_causal_masking_future_tokens_do_not_affect_earlier_logits() -> None:
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    seq_a = torch.tensor([[1, 2, 3, 4, 5]])
    seq_b = torch.tensor([[1, 2, 3, 9, 9]])  # differs only after position 2
    with torch.no_grad():
        logits_a = model(seq_a)
        logits_b = model(seq_b)
    assert torch.equal(logits_a[:, :3, :], logits_b[:, :3, :])
    assert not torch.equal(logits_a, logits_b)


def test_same_seed_produces_identical_initialization() -> None:
    set_seed(123)
    model_a = DecoderOnlyTransformer(_tiny_config())
    set_seed(123)
    model_b = DecoderOnlyTransformer(_tiny_config())
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_num_parameters_trainable_only() -> None:
    model = DecoderOnlyTransformer(_tiny_config())
    total = model.num_parameters()
    assert model.num_parameters(trainable_only=True) == total
    for param in model.parameters():
        param.requires_grad_(False)
    assert model.num_parameters(trainable_only=True) == 0
    assert model.num_parameters() == total


def test_weight_tying_between_embedding_and_head() -> None:
    model = DecoderOnlyTransformer(_tiny_config())
    assert model.head.weight is model.token_emb.weight
