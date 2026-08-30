from __future__ import annotations

from pathlib import Path

import torch

from apc.core.checkpoint import load_checkpoint, save_checkpoint
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.utils.seed import set_seed


def _config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=14, max_seq_len=16, d_model=16, n_layer=2, n_head=2, d_ff=32
    )


def test_save_and_load_checkpoint_restores_weights_exactly(tmp_path: Path) -> None:
    set_seed(0)
    model = DecoderOnlyTransformer(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Take one optimization step so the checkpoint captures non-trivial state.
    input_ids = torch.randint(0, 14, (2, 5))
    loss = model(input_ids).sum()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint" / "model.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=1, config={"seed": 0})
    assert checkpoint_path.is_file()

    # A differently-initialized model must diverge before loading...
    set_seed(1)
    reloaded = DecoderOnlyTransformer(_config())
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    for (_, p_orig), (_, p_new) in zip(
        model.named_parameters(), reloaded.named_parameters(), strict=True
    ):
        assert not torch.equal(p_orig, p_new)

    # ...and match exactly after loading.
    checkpoint = load_checkpoint(checkpoint_path, reloaded, reloaded_optimizer)
    for (_, p_orig), (_, p_new) in zip(
        model.named_parameters(), reloaded.named_parameters(), strict=True
    ):
        assert torch.equal(p_orig, p_new)

    assert checkpoint["step"] == 1
    assert checkpoint["config"] == {"seed": 0}
    assert reloaded_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
