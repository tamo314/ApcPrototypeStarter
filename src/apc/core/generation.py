"""Greedy decoding and exact-match evaluation for the fixed dense baseline.

Generation is per-example (not batched) and fully deterministic (argmax,
no sampling, `model.eval()`), which keeps the reload-reproducibility check
in Task 003 ("reload reproduces outputs") simple: given the same weights
and the same prompt, two calls must return identical token sequences.
"""

from __future__ import annotations

import torch

from apc.core.data import build_prompt_tokens
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SpecialTokens
from apc.environments.generator import Example


@torch.no_grad()
def generate_greedy(
    model: DecoderOnlyTransformer,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
) -> tuple[int, ...]:
    """Greedily extend a single `[1, prompt_len]` prompt up to `max_new_tokens`
    new tokens, stopping early if `eos_id` is generated. Returns only the
    generated continuation (not the prompt), with any trailing `eos_id`
    stripped."""
    model.eval()
    generated = prompt_ids
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_id = int(logits[0, -1, :].argmax(dim=-1).item())
        new_tokens.append(next_id)
        if next_id == eos_id:
            break
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], dtype=torch.long, device=generated.device)],
            dim=1,
        )
    if new_tokens and new_tokens[-1] == eos_id:
        new_tokens = new_tokens[:-1]
    return tuple(new_tokens)


def evaluate_exact_match(
    model: DecoderOnlyTransformer,
    examples: list[Example],
    specials: SpecialTokens,
    device: torch.device | str = "cpu",
    max_extra_tokens: int = 2,
    *,
    include_task_spec: bool = False,
) -> tuple[float, list[tuple[int, ...]]]:
    """Greedily decode each example's target from its prompt and compute the
    fraction of examples reproduced exactly. Returns `(exact_match, predictions)`.

    `include_task_spec` mirrors `apc.core.data.encode_example`/`collate_batch`
    (Task A1-C003): when set, the prompt also carries each example's
    `TaskSpec` segment, and `specials` must be a `apc.core.tokens.
    SharedCoreTokens`. Off by default, so existing callers (Task A1-006's
    per-operation gate) see unchanged behavior.
    """
    predictions: list[tuple[int, ...]] = []
    correct = 0
    for example in examples:
        prompt = build_prompt_tokens(example, specials, include_task_spec=include_task_spec)
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        max_new_tokens = len(example.target_tokens) + max_extra_tokens
        prediction = generate_greedy(model, prompt_ids, specials.eos, max_new_tokens)
        predictions.append(prediction)
        if prediction == example.target_tokens:
            correct += 1
    return correct / len(examples), predictions
