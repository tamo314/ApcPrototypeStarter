from __future__ import annotations

import pytest
import torch

from apc.core.model import DecoderOnlyTransformer, EncodedState, TransformerConfig
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


# --- encode_split (Phase A.1 Task A1-005) -----------------------------------


def test_encode_split_returns_encoded_state_with_expected_shapes() -> None:
    model = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (3, 7))
    encoded = model.encode_split(input_ids)
    assert isinstance(encoded, EncodedState)
    assert encoded.task_state.shape == (3, 7, 16)
    assert encoded.content_state.shape == (3, 7, 16)


def test_encode_split_content_state_matches_plain_encode() -> None:
    """Phase A/A.1 compatibility: `content_state` is exactly `encode`'s
    output, value-for-value, so callers that have not migrated to the
    split API see no behavior change."""
    set_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (2, 5))
    with torch.no_grad():
        expected = model.encode(input_ids)
        encoded = model.encode_split(input_ids)
    torch.testing.assert_close(encoded.content_state, expected)


def test_encode_split_task_state_is_independently_probeable() -> None:
    """`task_state` and `content_state` are distinct tensors carrying
    different values -- not aliases of one shared hidden state -- so each
    can be logged/probed on its own."""
    set_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (2, 5))
    with torch.no_grad():
        encoded = model.encode_split(input_ids)
    assert encoded.task_state.data_ptr() != encoded.content_state.data_ptr()
    assert not torch.allclose(encoded.task_state, encoded.content_state)


def test_encode_split_content_state_gradient_matches_plain_encode_gradient() -> None:
    """`task_head` is parameter-free and sits downstream of `content_state`
    (see `encode_split`'s docstring), so backpropagating through
    `content_state` alone must produce the exact same trunk gradient as
    backpropagating through plain `encode`'s output -- `task_head`'s
    existence changes nothing about that path."""
    set_seed(0)
    model_a = DecoderOnlyTransformer(_tiny_config())
    set_seed(0)
    model_b = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (2, 5))

    model_a.encode(input_ids).pow(2).sum().backward()
    model_b.encode_split(input_ids).content_state.pow(2).sum().backward()

    torch.testing.assert_close(model_a.token_emb.weight.grad, model_b.token_emb.weight.grad)


def test_encode_split_task_state_gradient_reaches_trunk() -> None:
    """`task_state` is a function of `content_state`, so a loss on
    `task_state` alone must still reach the trunk's parameters."""
    model = DecoderOnlyTransformer(_tiny_config())
    input_ids = torch.randint(0, 14, (2, 5))
    encoded = model.encode_split(input_ids)
    encoded.task_state.pow(2).sum().backward()
    assert model.token_emb.weight.grad is not None
