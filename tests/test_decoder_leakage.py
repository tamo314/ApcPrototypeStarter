"""Tests for the decoder-leakage audit and no-primitive/identity execution
path (Phase A.1 Post-Correction Task A1-R002).

Two things are covered here, matching `apc.core.execution`'s "Decoder-input
audit and no-primitive/identity mode" module-docstring section:

1. **The audit itself** (Work items 1-2, "audit decoder inputs" / "remove
   task-conditioned bypasses in causal mode"): every causal-mode decode
   function in `apc.core.execution` -- bank-routed
   (`evaluate_exact_match_with_capacity`), oracle-routed
   (`evaluate_exact_match_with_oracle_calls`), and the new no-primitive
   function this task adds (`evaluate_exact_match_no_primitive`) -- is driven
   with a `SharedCoreTokens` vocabulary and `Example`s that carry a real
   `TaskSpec`, with `model.encode` monkeypatched to record every `input_ids`
   tensor it is called with. None of the recorded ids may ever fall inside
   `apc.core.tokens.task_token_id_set`'s task-segment/operation/
   argument-value range.
2. **The no-primitive/identity functions' own behavior** (Work item 3, "add
   no-primitive/identity mode"): they match plain `apc.core.generation`
   output when there's nothing to route through (an empty/no bank is exactly
   the identity transform), and structurally cannot accept a
   bank/router/workspace parameter at all.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch

from apc.core.execution import (
    evaluate_exact_match_no_primitive,
    evaluate_exact_match_with_capacity,
    evaluate_exact_match_with_oracle_calls,
    forward_logits_no_primitive,
    generate_greedy_no_primitive,
)
from apc.core.generation import evaluate_exact_match, generate_greedy
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import (
    SharedCoreTokens,
    build_shared_core_tokens,
    build_special_tokens,
    task_token_id_set,
)
from apc.environments.generator import Example, TaskGenerator, oracle_call_for_example
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.environments.task_spec import (
    default_argument_value_span,
    num_registered_operations,
    operation_id,
)
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus
from apc.primitives.router import Router, RouterConfig

D_MODEL = 8
SEQUENCE_LENGTH_RANGE = (4, 6)


def _tokens() -> SharedCoreTokens:
    return build_shared_core_tokens(
        DEFAULT_VOCAB_SIZE,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(DEFAULT_VOCAB_SIZE, SEQUENCE_LENGTH_RANGE),
    )


def _model(vocab_size: int) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        TransformerConfig(
            vocab_size=vocab_size, max_seq_len=48, d_model=D_MODEL, n_layer=2, n_head=2, d_ff=16
        )
    )


def _example_with_task_spec(operation: str, *, seed: int = 0) -> Example:
    generator = TaskGenerator(
        seed=seed,
        operation_names=(operation,),
        max_depth=1,
        vocab_size=DEFAULT_VOCAB_SIZE,
        sequence_length_range=SEQUENCE_LENGTH_RANGE,
    )
    example = generator.generate(1, "train")[0]
    assert example.task_spec is not None
    return example


def _bank_with_primitives_at(ids: list[int]) -> PrimitiveBank:
    bank = PrimitiveBank()
    for pid in ids:
        primitive = Primitive(
            pid, PrimitiveConfig(d_model=D_MODEL, rank=4), status=PrimitiveStatus.STABLE
        )
        bank.add_primitive(primitive)
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    return bank


def _spy_on_encode(model: DecoderOnlyTransformer) -> list[list[int]]:
    """Monkeypatch `model.encode` to record each call's flattened ids as one
    entry, while still delegating to the real implementation. Returns the
    list of per-call id lists, in call order -- the *first* entry is always
    the initial prompt every `generate_greedy*` variant seeds generation
    with, before any autoregressive self-extension.

    Only the initial prompt is a meaningful "decoder input" to audit for
    task-token leakage: a freshly-initialized (untrained, effectively
    random-output) model can legitimately emit a token whose id happens to
    fall inside the task-segment/operation/argument range as part of its
    own greedy *output*, which then gets appended and re-encoded on the next
    step -- that is the model talking to itself, not a task-information
    bypass, and checking later calls would make this audit fail on pure
    numerical coincidence rather than a real leak.
    """
    seen_calls: list[list[int]] = []
    real_encode = model.encode

    def _spy(input_ids: torch.Tensor) -> torch.Tensor:
        seen_calls.append([int(t) for t in input_ids.reshape(-1).tolist()])
        return real_encode(input_ids)

    model.encode = _spy  # type: ignore[method-assign]
    return seen_calls


# --- audit: no task token ever reaches model.encode -------------------------


def _run_capacity(
    model: DecoderOnlyTransformer, tokens: SharedCoreTokens, examples: list[Example]
) -> None:
    bank = PrimitiveBank()
    router = Router(RouterConfig(d_model=D_MODEL, top_k=1))
    evaluate_exact_match_with_capacity(model, examples, tokens, bank, router)


def _run_oracle(
    model: DecoderOnlyTransformer, tokens: SharedCoreTokens, examples: list[Example]
) -> None:
    bank_ids = [operation_id(name) for name in KNOWN_OPERATION_NAMES]
    bank = _bank_with_primitives_at(bank_ids)
    evaluate_exact_match_with_oracle_calls(model, examples, tokens, bank, oracle_call_for_example)


def _run_no_primitive(
    model: DecoderOnlyTransformer, tokens: SharedCoreTokens, examples: list[Example]
) -> None:
    evaluate_exact_match_no_primitive(model, examples, tokens)


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(_run_capacity, id="capacity"),
        pytest.param(_run_oracle, id="oracle"),
        pytest.param(_run_no_primitive, id="no_primitive"),
    ],
)
@pytest.mark.parametrize("operation", ["SHIFT", "COPY"])
def test_causal_mode_functions_never_feed_task_tokens_to_encode(
    runner: Callable[[DecoderOnlyTransformer, SharedCoreTokens, list[Example]], None],
    operation: str,
) -> None:
    tokens = _tokens()
    task_ids = task_token_id_set(tokens)
    model = _model(tokens.model_vocab_size)
    seen_calls = _spy_on_encode(model)

    example = _example_with_task_spec(operation, seed=0)
    runner(model, tokens, [example])

    assert seen_calls, "the audited function never called model.encode at all"
    prompt_call = seen_calls[0]  # the initial prompt, before any autoregressive self-extension
    assert prompt_call == [tokens.bos, *example.input_tokens, tokens.sep]
    leaked = sorted(set(prompt_call) & task_ids)
    assert leaked == [], f"task-segment token ids reached model.encode via the prompt: {leaked}"


def test_causal_mode_functions_do_see_formatting_and_content_tokens() -> None:
    """The audit above is not vacuous -- BOS/SEP and real content ids are
    genuinely present in what `model.encode` receives, so "no task tokens"
    is a real, non-trivial finding rather than an empty input."""
    tokens = _tokens()
    model = _model(tokens.model_vocab_size)
    seen_calls = _spy_on_encode(model)

    example = _example_with_task_spec("COPY", seed=0)
    _run_no_primitive(model, tokens, [example])

    prompt_call = seen_calls[0]
    assert tokens.bos in prompt_call
    assert tokens.sep in prompt_call
    assert any(tid in prompt_call for tid in example.input_tokens)


# --- forward_logits_no_primitive / generate_greedy_no_primitive / ----------
# --- evaluate_exact_match_no_primitive --------------------------------------


def test_forward_logits_no_primitive_equals_model_forward() -> None:
    model = _model(14)
    input_ids = torch.randint(0, 14, (2, 5))
    with torch.no_grad():
        expected = model(input_ids)
        actual = forward_logits_no_primitive(model, input_ids)
    torch.testing.assert_close(actual, expected)


def test_generate_greedy_no_primitive_matches_plain_generation() -> None:
    model = _model(10)
    specials = build_special_tokens(6)
    prompt = torch.tensor([[specials.bos, 1, 2, specials.sep]])

    plain = generate_greedy(model, prompt, specials.eos, max_new_tokens=6)
    no_primitive = generate_greedy_no_primitive(model, prompt, specials.eos, max_new_tokens=6)
    assert plain == no_primitive


def test_evaluate_exact_match_no_primitive_matches_plain_generation() -> None:
    specials = build_special_tokens(DEFAULT_VOCAB_SIZE)
    model = _model(specials.model_vocab_size)
    example = _example_with_task_spec("COPY", seed=0)
    plain_score, plain_preds = evaluate_exact_match(model, [example], specials)
    no_primitive_score, no_primitive_preds = evaluate_exact_match_no_primitive(
        model, [example], specials
    )
    assert plain_score == no_primitive_score
    assert plain_preds == no_primitive_preds


@pytest.mark.parametrize(
    "func",
    [evaluate_exact_match_no_primitive, generate_greedy_no_primitive, forward_logits_no_primitive],
)
def test_no_primitive_functions_never_accept_bank_router_or_workspace(func: Callable) -> None:
    """Work item 3, "add no-primitive/identity mode": primitive execution is
    structurally unreachable from these functions, not merely unused --
    mirrors `tests/test_oracle_routing.py::
    test_oracle_routing_functions_never_accept_a_router`'s convention for
    the oracle-routed siblings."""
    params = inspect.signature(func).parameters
    assert "bank" not in params
    assert "router" not in params
    assert "workspace" not in params
