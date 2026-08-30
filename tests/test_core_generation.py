from __future__ import annotations

import torch
from torch import nn

from apc.core.generation import evaluate_exact_match, generate_greedy
from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example
from apc.environments.interpreter import OperationGraph
from apc.environments.program import Program

VOCAB_SIZE = 6
SPECIALS = build_special_tokens(VOCAB_SIZE)  # pad=6 bos=7 sep=8 eos=9, model_vocab_size=10


class _ScriptedModel(nn.Module):
    """A fake model whose greedy argmax follows a fixed `seq_len -> next_id` script.

    Used to unit-test `generate_greedy`/`evaluate_exact_match` without a real
    training run: the model doesn't matter, only that generation correctly
    drives it step by step and stops on EOS.
    """

    def __init__(self, script: dict[int, int]) -> None:
        super().__init__()
        self.script = script
        self.dummy = nn.Parameter(torch.zeros(1))

    def eval(self) -> _ScriptedModel:  # noqa: D102
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        next_id = self.script[seq_len]
        logits = torch.full((1, seq_len, SPECIALS.model_vocab_size), -10.0)
        logits[0, -1, next_id] = 10.0
        return logits


def _example(input_tokens: tuple[int, ...], target_tokens: tuple[int, ...]) -> Example:
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=Program(steps=()),
        operation_graph=OperationGraph(nodes=()),
        category="known",
        split="train",
        vocab_size=VOCAB_SIZE,
    )


def test_generate_greedy_follows_script_and_stops_on_eos() -> None:
    # prompt = BOS 1 2 SEP -> len 4; script emits target (3, 4) then EOS.
    model = _ScriptedModel({4: 3, 5: 4, 6: SPECIALS.eos})
    prompt = torch.tensor([[SPECIALS.bos, 1, 2, SPECIALS.sep]])
    generated = generate_greedy(model, prompt, SPECIALS.eos, max_new_tokens=10)
    assert generated == (3, 4)


def test_generate_greedy_is_deterministic() -> None:
    model = _ScriptedModel({4: 3, 5: 4, 6: SPECIALS.eos})
    prompt = torch.tensor([[SPECIALS.bos, 1, 2, SPECIALS.sep]])
    first = generate_greedy(model, prompt, SPECIALS.eos, max_new_tokens=10)
    second = generate_greedy(model, prompt, SPECIALS.eos, max_new_tokens=10)
    assert first == second


def test_generate_greedy_respects_max_new_tokens_when_eos_never_emitted() -> None:
    model = _ScriptedModel({n: 0 for n in range(4, 20)})  # never emits EOS
    prompt = torch.tensor([[SPECIALS.bos, 1, 2, SPECIALS.sep]])
    generated = generate_greedy(model, prompt, SPECIALS.eos, max_new_tokens=3)
    assert generated == (0, 0, 0)


def test_evaluate_exact_match_perfect_model_scores_one() -> None:
    example = _example((1, 2), (3, 4))
    model = _ScriptedModel({4: 3, 5: 4, 6: SPECIALS.eos})
    exact_match, predictions = evaluate_exact_match(model, [example], SPECIALS)
    assert exact_match == 1.0
    assert predictions == [(3, 4)]


def test_evaluate_exact_match_wrong_prediction_scores_zero() -> None:
    example = _example((1, 2), (3, 4))
    model = _ScriptedModel({4: 5, 5: SPECIALS.eos})  # predicts (5,) then stops
    exact_match, predictions = evaluate_exact_match(model, [example], SPECIALS)
    assert exact_match == 0.0
    assert predictions == [(5,)]
