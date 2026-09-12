"""A saved content Core's vocabulary must survive unrelated operation imports."""

from types import SimpleNamespace

import pytest
import torch

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import shared_encoder_architecture_gate as architecture
from apc.utils import model_bundle as mb


def _loaded():
    return SimpleNamespace(
        vocabulary_state_dict={k: torch.tensor([v]) for k, v in {
            "vocab_size": 10, "num_operations": 18, "arg_span": 10,
            "op_base": 16, "arg_base": 34,
        }.items()},
        core_state_dict={"token_emb.weight": torch.zeros(44, 16),
                         "head.weight": torch.zeros(44, 16)},
    )


def test_saved_schema_preserves_core_shape_and_outputs_when_registry_grows(monkeypatch):
    tokens = ibc._tokens_from_loaded_bundle(_loaded())
    config = architecture.SharedEncoderArchitectureConfig(
        device="cpu", model={"d_model": 16, "n_layer": 1, "n_head": 2,
                             "d_ff": 32, "max_seq_len": 48, "dropout": 0.0},
    )
    source = architecture.build_shared_encoder_architecture(config, tokens=tokens).core
    monkeypatch.setattr(architecture, "num_registered_operations", lambda: 23)
    ambient = architecture.build_shared_encoder_architecture(config).core
    restored = architecture.build_shared_encoder_architecture(config, tokens=tokens).core
    assert ambient.tokens.model_vocab_size == 49
    assert restored.tokens == source.tokens
    assert restored.model.config.vocab_size == 44
    restored.model.load_state_dict(source.model.state_dict(), strict=True)
    content = torch.tensor([[tokens.bos, 0, 1, 2, 3]])
    source.model.eval()
    restored.model.eval()
    with torch.no_grad():
        assert torch.equal(source.model.encode(content), restored.model.encode(content))


@pytest.mark.parametrize("field,value", [
    ("num_operations", torch.tensor([23])),
    ("arg_base", torch.tensor([35])),
    ("op_base", torch.tensor([17])),
    ("arg_span", torch.tensor([-1])),
    ("vocab_size", torch.tensor([10.5])),
    ("num_operations", None),
])
def test_inconsistent_or_missing_saved_schema_fails_closed(field, value):
    loaded = _loaded()
    if value is None:
        loaded.vocabulary_state_dict.pop(field)
    else:
        loaded.vocabulary_state_dict[field] = value
    with pytest.raises(mb.SchemaMismatchError):
        ibc._tokens_from_loaded_bundle(loaded)
