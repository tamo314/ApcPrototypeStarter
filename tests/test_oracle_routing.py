"""Tests for the oracle `PrimitiveCall` routing adapter (Phase A.1
Correction Task A1-C007).

Two layers, matching the module split documented in
`apc.environments.generator`/`apc.core.execution`:

1. `oracle_calls_for_example`/`oracle_call_for_example` (environment layer):
   converting an `Example`'s latent `program` into a complete, executable
   `PrimitiveCall` -- covers both parameterized operations (SHIFT/SELECT/
   COUNT/BIND) and parameter-free ones (COPY/NEGATE/COMPARE/ACCUMULATE),
   per A1-C007 Work item 4.
2. `apply_bank_with_oracle_calls`/`forward_logits_with_oracle_calls`/
   `generate_greedy_with_oracle_call`/`evaluate_exact_match_with_oracle_calls`
   (core execution layer): forcing bank primitive selection from oracle
   calls, with the learned `Router` never appearing anywhere in the call
   graph (Work item 2).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from apc.core.execution import (
    apply_bank_with_oracle_calls,
    apply_bank_with_oracle_calls_trainable,
    evaluate_exact_match_with_oracle_calls,
    forward_logits_with_oracle_calls,
    forward_logits_with_oracle_calls_trainable,
    generate_greedy_with_oracle_call,
)
from apc.core.generation import evaluate_exact_match, generate_greedy
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.environments.generator import (
    NOVEL_COMPOSITION_SPLIT,
    TaskGenerator,
    oracle_call_for_example,
    oracle_calls_for_example,
)
from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES, KNOWN_OPERATION_NAMES
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import operation_id
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

PARAMETERIZED_OPERATIONS: tuple[str, ...] = ("SHIFT", "SELECT", "COUNT", "BIND")

D_MODEL = 8
ENV_VOCAB_SIZE = DEFAULT_VOCAB_SIZE
MODEL_VOCAB_SIZE = ENV_VOCAB_SIZE + 4


def _one_example(operation: str, *, seed: int = 0):
    generator = TaskGenerator(seed=seed, operation_names=(operation,), max_depth=1)
    return generator.generate(1, "train")[0]


# --- oracle_calls_for_example / oracle_call_for_example ---------------------


@pytest.mark.parametrize("operation", KNOWN_OPERATION_NAMES)
def test_oracle_call_reproduces_target_tokens_exactly(operation: str) -> None:
    """Acceptance: "oracle call fully determines the primitive execution" --
    for every known operation, parameterized or not."""
    example = _one_example(operation)
    call = oracle_call_for_example(example)
    assert call.operation == operation
    assert call.execute(example.input_tokens, example.vocab_size) == example.target_tokens


@pytest.mark.parametrize("operation", KNOWN_OPERATION_NAMES)
def test_oracle_calls_for_example_returns_one_call_for_depth_one_examples(operation: str) -> None:
    example = _one_example(operation)
    calls = oracle_calls_for_example(example)
    assert calls == (oracle_call_for_example(example),)


@pytest.mark.parametrize("operation", PARAMETERIZED_OPERATIONS)
def test_parameterized_operation_oracle_calls_carry_nonempty_arguments(operation: str) -> None:
    example = _one_example(operation)
    call = oracle_call_for_example(example)
    assert call.arguments != {}


@pytest.mark.parametrize("operation", DETERMINISTIC_OPERATION_NAMES)
def test_parameter_free_operation_oracle_calls_carry_empty_arguments(operation: str) -> None:
    example = _one_example(operation)
    call = oracle_call_for_example(example)
    assert call.arguments == {}


@pytest.mark.parametrize("operation", PARAMETERIZED_OPERATIONS)
def test_primitive_id_only_call_cannot_be_constructed_for_parameterized_operations(
    operation: str,
) -> None:
    """Regression guard for "hidden-parameter failures from A1-006 cannot
    recur in oracle mode": an oracle call for a parameterized operation
    cannot even be constructed without its argument -- there is no
    primitive-id-only shortcut available to (re)introduce the A1-006
    non-identifiability one level into oracle routing."""
    with pytest.raises(ValueError, match="missing required"):
        PrimitiveCall(operation=operation)


def test_oracle_calls_for_example_returns_multiple_calls_for_a_composition() -> None:
    generator = TaskGenerator(seed=0, max_depth=2)
    example = generator.generate(1, NOVEL_COMPOSITION_SPLIT)[0]
    assert len(example.program.steps) > 1

    calls = oracle_calls_for_example(example)

    assert len(calls) == len(example.program.steps)
    assert [c.operation for c in calls] == list(example.program.operation_sequence)


def test_oracle_call_for_example_raises_on_multi_step_composition() -> None:
    generator = TaskGenerator(seed=0, max_depth=2)
    example = generator.generate(1, NOVEL_COMPOSITION_SPLIT)[0]
    assert len(example.program.steps) > 1

    with pytest.raises(ValueError, match="exactly one"):
        oracle_call_for_example(example)


# --- core.execution oracle-forced routing: fixtures --------------------------


def _model() -> DecoderOnlyTransformer:
    set_seed(0)
    return DecoderOnlyTransformer(
        TransformerConfig(
            vocab_size=MODEL_VOCAB_SIZE,
            max_seq_len=32,
            d_model=D_MODEL,
            n_layer=2,
            n_head=2,
            d_ff=16,
        )
    )


def _bank_with_primitives_at(ids: list[int], *, perturb: bool = True) -> PrimitiveBank:
    bank = PrimitiveBank()
    for pid in ids:
        primitive = Primitive(
            pid, PrimitiveConfig(d_model=D_MODEL, rank=4), status=PrimitiveStatus.STABLE
        )
        bank.add_primitive(primitive)
        if perturb:
            with torch.no_grad():
                primitive.b_proj.weight.add_(1.0)
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    return bank


# --- structural isolation from the learned Router ----------------------------


@pytest.mark.parametrize(
    "func",
    [
        apply_bank_with_oracle_calls,
        apply_bank_with_oracle_calls_trainable,
        forward_logits_with_oracle_calls,
        forward_logits_with_oracle_calls_trainable,
        generate_greedy_with_oracle_call,
        evaluate_exact_match_with_oracle_calls,
    ],
)
def test_oracle_routing_functions_never_accept_a_router(func) -> None:
    """Work item 2, "the learned path never receives oracle calls": the
    learned `Router` is not merely unused at runtime by these functions, it
    cannot be passed to them at all."""
    params = inspect.signature(func).parameters
    assert "router" not in params
    assert "stable_ids" not in params


# --- apply_bank_with_oracle_calls --------------------------------------------


def test_apply_bank_with_oracle_calls_executes_only_the_forced_primitives() -> None:
    shift_id, count_id, bind_id = operation_id("SHIFT"), operation_id("COUNT"), operation_id("BIND")
    bank = _bank_with_primitives_at([shift_id, count_id, bind_id])
    calls = [
        PrimitiveCall(operation="SHIFT", arguments={"amount": 1}),
        PrimitiveCall(operation="COUNT", arguments={"target": 2}),
    ]
    hidden = torch.randn(2, 3, D_MODEL)

    result, routing_out = apply_bank_with_oracle_calls(hidden, bank, calls)

    assert routing_out.selected_ids == (shift_id, count_id)
    assert routing_out.executed_primitive_ids == tuple(sorted({shift_id, count_id}))
    assert bank.get(shift_id).forward_call_count == 1
    assert bank.get(count_id).forward_call_count == 1
    assert bank.get(bind_id).forward_call_count == 0

    expected = hidden.clone()
    expected[0] = bank.get(shift_id)(hidden[0])
    expected[1] = bank.get(count_id)(hidden[1])
    torch.testing.assert_close(result, expected)


def test_apply_bank_with_oracle_calls_output_never_requires_grad() -> None:
    pid = operation_id("COPY")
    bank = _bank_with_primitives_at([pid])
    for p in bank.parameters():
        p.requires_grad_(True)
    h = torch.randn(1, 3, D_MODEL, requires_grad=True)
    result, _ = apply_bank_with_oracle_calls(h, bank, [PrimitiveCall(operation="COPY")])
    assert not result.requires_grad


def test_apply_bank_with_oracle_calls_rejects_batch_length_mismatch() -> None:
    bank = _bank_with_primitives_at([operation_id("COPY")])
    h = torch.randn(2, 3, D_MODEL)
    with pytest.raises(ValueError, match="batch"):
        apply_bank_with_oracle_calls(h, bank, [PrimitiveCall(operation="COPY")])


def test_apply_bank_with_oracle_calls_raises_key_error_for_unregistered_primitive() -> None:
    bank = PrimitiveBank()
    h = torch.randn(1, 2, D_MODEL)
    with pytest.raises(KeyError):
        apply_bank_with_oracle_calls(h, bank, [PrimitiveCall(operation="COPY")])


def test_apply_bank_with_oracle_calls_raises_value_error_for_disabled_primitive() -> None:
    pid = operation_id("COPY")
    bank = _bank_with_primitives_at([pid])
    bank.disable(pid)
    h = torch.randn(1, 2, D_MODEL)
    with pytest.raises(ValueError, match="disabled"):
        apply_bank_with_oracle_calls(h, bank, [PrimitiveCall(operation="COPY")])


# --- apply_bank_with_oracle_calls_trainable (Task A1-R003) -------------------


def test_apply_bank_with_oracle_calls_trainable_matches_no_grad_sibling_numerically() -> None:
    """Same routing/execution logic, only the grad-tracking differs."""
    shift_id, count_id = operation_id("SHIFT"), operation_id("COUNT")
    bank = _bank_with_primitives_at([shift_id, count_id])
    calls = [
        PrimitiveCall(operation="SHIFT", arguments={"amount": 1}),
        PrimitiveCall(operation="COUNT", arguments={"target": 2}),
    ]
    hidden = torch.randn(2, 3, D_MODEL)

    with torch.no_grad():
        expected, expected_routing = apply_bank_with_oracle_calls(hidden, bank, calls)
    actual, actual_routing = apply_bank_with_oracle_calls_trainable(hidden, bank, calls)

    torch.testing.assert_close(actual, expected)
    assert actual_routing.selected_ids == expected_routing.selected_ids
    assert actual_routing.executed_primitive_ids == expected_routing.executed_primitive_ids


def test_apply_bank_with_oracle_calls_trainable_gradient_reaches_only_the_selected_primitive() -> (
    None
):
    """The whole point of the `_trainable` sibling: gradient must flow into
    the oracle-selected primitive's own weights, and never into a primitive
    family a given example's call did not select."""
    shift_id, count_id, bind_id = (
        operation_id("SHIFT"),
        operation_id("COUNT"),
        operation_id("BIND"),
    )
    bank = _bank_with_primitives_at([shift_id, count_id, bind_id])
    for p in bank.parameters():
        p.requires_grad_(True)
    calls = [
        PrimitiveCall(operation="SHIFT", arguments={"amount": 1}),
        PrimitiveCall(operation="COUNT", arguments={"target": 2}),
    ]
    hidden = torch.randn(2, 3, D_MODEL)

    result, _ = apply_bank_with_oracle_calls_trainable(hidden, bank, calls)
    assert result.requires_grad
    result.sum().backward()

    assert bank.get(shift_id).a_proj.weight.grad is not None
    assert bank.get(count_id).a_proj.weight.grad is not None
    assert bank.get(bind_id).a_proj.weight.grad is None


def test_apply_bank_with_oracle_calls_trainable_executes_only_the_forced_primitives() -> None:
    shift_id, count_id, bind_id = (
        operation_id("SHIFT"),
        operation_id("COUNT"),
        operation_id("BIND"),
    )
    bank = _bank_with_primitives_at([shift_id, count_id, bind_id])
    calls = [
        PrimitiveCall(operation="SHIFT", arguments={"amount": 1}),
        PrimitiveCall(operation="COUNT", arguments={"target": 2}),
    ]
    hidden = torch.randn(2, 3, D_MODEL)

    _, routing_out = apply_bank_with_oracle_calls_trainable(hidden, bank, calls)

    assert routing_out.selected_ids == (shift_id, count_id)
    assert routing_out.executed_primitive_ids == tuple(sorted({shift_id, count_id}))
    assert bank.get(shift_id).forward_call_count == 1
    assert bank.get(count_id).forward_call_count == 1
    assert bank.get(bind_id).forward_call_count == 0


# --- forward_logits_with_oracle_calls ----------------------------------------


def test_forward_logits_with_oracle_calls_shape() -> None:
    model = _model()
    pid = operation_id("COPY")
    bank = _bank_with_primitives_at([pid])
    input_ids = torch.randint(0, MODEL_VOCAB_SIZE, (2, 5))
    calls = [PrimitiveCall(operation="COPY"), PrimitiveCall(operation="COPY")]

    logits, routing_out = forward_logits_with_oracle_calls(model, input_ids, bank, calls)

    assert logits.shape == (2, 5, MODEL_VOCAB_SIZE)
    assert routing_out.executed_primitive_ids == (pid,)


def test_forward_logits_with_oracle_calls_matches_manual_apply_bank_plus_decode() -> None:
    model = _model()
    pid = operation_id("NEGATE")
    bank = _bank_with_primitives_at([pid])
    input_ids = torch.randint(0, MODEL_VOCAB_SIZE, (1, 4))
    calls = [PrimitiveCall(operation="NEGATE")]

    with torch.no_grad():
        content, _ = apply_bank_with_oracle_calls(model.encode(input_ids), bank, calls)
        expected = model.decode(content)
        actual, _ = forward_logits_with_oracle_calls(model, input_ids, bank, calls)

    torch.testing.assert_close(actual, expected)


def test_forward_logits_with_oracle_calls_trainable_gradient_reaches_primitive_through_decode() -> (
    None
):
    """End-to-end grad check for the A1-R003 training call pattern: freeze
    the model, leave the bank trainable, and confirm a loss on the logits
    backpropagates into the selected primitive."""
    model = _model()
    for p in model.parameters():
        p.requires_grad_(False)
    pid = operation_id("NEGATE")
    bank = _bank_with_primitives_at([pid])
    for p in bank.parameters():
        p.requires_grad_(True)
    input_ids = torch.randint(0, MODEL_VOCAB_SIZE, (1, 4))
    calls = [PrimitiveCall(operation="NEGATE")]

    logits, routing_out = forward_logits_with_oracle_calls_trainable(model, input_ids, bank, calls)
    assert logits.requires_grad
    assert routing_out.executed_primitive_ids == (pid,)

    logits.sum().backward()
    assert bank.get(pid).a_proj.weight.grad is not None
    assert model.token_emb.weight.grad is None


# --- generate_greedy_with_oracle_call ----------------------------------------


def test_generate_greedy_with_oracle_call_matches_plain_generation_for_an_identity_primitive() -> (
    None
):
    """A freshly created primitive is the identity function (`B` starts at
    zero, see `apc.primitives.primitive.Primitive`), so forcing it via an
    oracle call must reproduce plain, bank-free generation exactly."""
    model = _model()
    pid = operation_id("COPY")
    bank = _bank_with_primitives_at([pid], perturb=False)
    specials = build_special_tokens(ENV_VOCAB_SIZE)
    prompt = torch.tensor([[specials.bos, 1, 2, specials.sep]])

    plain = generate_greedy(model, prompt, specials.eos, max_new_tokens=6)
    oracle = generate_greedy_with_oracle_call(
        model, bank, PrimitiveCall(operation="COPY"), prompt, specials.eos, max_new_tokens=6
    )
    assert plain == oracle


def test_generate_greedy_with_oracle_call_rejects_batch_other_than_one() -> None:
    model = _model()
    pid = operation_id("COPY")
    bank = _bank_with_primitives_at([pid])
    prompt = torch.randint(0, MODEL_VOCAB_SIZE, (2, 4))
    with pytest.raises(ValueError, match="batch"):
        generate_greedy_with_oracle_call(
            model, bank, PrimitiveCall(operation="COPY"), prompt, eos_id=0, max_new_tokens=3
        )


# --- evaluate_exact_match_with_oracle_calls ----------------------------------


def test_evaluate_exact_match_with_oracle_calls_matches_plain_for_an_identity_primitive() -> None:
    model = _model()
    specials = build_special_tokens(ENV_VOCAB_SIZE)
    example = _one_example("SHIFT")
    pid = operation_id("SHIFT")
    bank = _bank_with_primitives_at([pid], perturb=False)

    plain_score, plain_preds = evaluate_exact_match(model, [example], specials)
    oracle_score, oracle_preds, calls_used = evaluate_exact_match_with_oracle_calls(
        model, [example], specials, bank
    )

    assert plain_score == oracle_score
    assert plain_preds == oracle_preds
    assert calls_used == [oracle_call_for_example(example)]


def test_evaluate_exact_match_with_oracle_calls_selected_ids_match_metadata() -> None:
    """A1-007's own accept criterion ("selected IDs exactly match
    metadata") holds by construction: the returned calls' `primitive_id`s
    are exactly each example's oracle operation id."""
    model = _model()
    specials = build_special_tokens(ENV_VOCAB_SIZE)
    examples = [_one_example(name, seed=idx) for idx, name in enumerate(KNOWN_OPERATION_NAMES)]
    bank_ids = [operation_id(name) for name in KNOWN_OPERATION_NAMES]
    bank = _bank_with_primitives_at(bank_ids, perturb=False)

    _, _, calls_used = evaluate_exact_match_with_oracle_calls(model, examples, specials, bank)

    expected_ids = [operation_id(example.program.operation_sequence[0]) for example in examples]
    assert [c.primitive_id for c in calls_used] == expected_ids


def test_evaluate_exact_match_with_oracle_calls_uses_the_injected_provider() -> None:
    """Work item 3, "accept an oracle call provider": a caller-supplied
    provider is actually threaded through, not silently ignored in favor
    of the default."""
    model = _model()
    specials = build_special_tokens(ENV_VOCAB_SIZE)
    examples = [_one_example("SHIFT", seed=0), _one_example("SHIFT", seed=1)]
    fixed_call = PrimitiveCall(operation="SHIFT", arguments={"amount": 0})
    bank = _bank_with_primitives_at([operation_id("SHIFT")], perturb=False)

    def stub_provider(example) -> PrimitiveCall:
        del example
        return fixed_call

    _, _, calls_used = evaluate_exact_match_with_oracle_calls(
        model, examples, specials, bank, stub_provider
    )

    assert calls_used == [fixed_call, fixed_call]
