"""Tests for CompositionLibrary and multi-step recipe execution (Task A1-B003)."""

from __future__ import annotations

import pytest

from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import TaskSpec
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import (
    CompositionLibrary,
    CompositionRecipe,
    execute_composition_recipe,
    extract_argument_value,
)
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
    ReverseRelativePrimitiveConfig,
    ShiftRelativePrimitiveConfig,
)


def test_composition_recipe_validation() -> None:
    call1 = PrimitiveCall("SHIFT", {"amount": 2})
    call2 = PrimitiveCall("SELECT", {"indices": (0, 2)})
    recipe = CompositionRecipe("shift_select", (call1, call2))

    assert len(recipe) == 2
    assert recipe.operations == ("SHIFT", "SELECT")
    assert recipe.name == "shift_select"

    with pytest.raises(ValueError, match="must contain at least one step"):
        CompositionRecipe("empty", ())


def test_composition_library_registration_and_retrieval() -> None:
    library = CompositionLibrary()
    recipe1 = CompositionRecipe(
        "shift_select",
        (
            PrimitiveCall("SHIFT", {"amount": 1}),
            PrimitiveCall("SELECT", {"indices": (0, 1)}),
        ),
    )
    library.register_recipe(recipe1)

    assert len(library) == 1
    assert library.has_recipe("shift_select")
    assert not library.has_recipe("non_existent")
    assert library.get_recipe("shift_select") == recipe1
    assert library.names() == ["shift_select"]

    # Reject duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        library.register_recipe(recipe1)

    with pytest.raises(KeyError, match="not found"):
        library.get_recipe("unknown")


def test_extract_argument_value() -> None:
    assert extract_argument_value("SHIFT", PrimitiveCall("SHIFT", {"amount": 3})) == 3
    assert extract_argument_value("SELECT", PrimitiveCall("SELECT", {"indices": (1, 2)})) == (1, 2)
    assert extract_argument_value("COUNT", PrimitiveCall("COUNT", {"target": 5})) == 5
    assert extract_argument_value("BIND", PrimitiveCall("BIND", {"query_key": 2})) == 2
    assert extract_argument_value("REVERSE", PrimitiveCall("REVERSE", {})) is None
    assert extract_argument_value("COPY", PrimitiveCall("COPY", {})) is None


def test_composition_execution_sparse_accounting() -> None:
    """Invariant test: unselected primitives must receive zero forward calls."""
    # Build a tiny shared encoder architecture
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=42,
        vocab_size=10,
        sequence_length_range=(6, 8),
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        model={
            "d_model": 32,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
    )
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core

    # Populate a bank with 3 primitives: SHIFT, SELECT, REVERSE
    bank = PrimitiveBank()
    p_shift = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
            arg_dim=8,
        ),
        status=PrimitiveStatus.STABLE,
    )
    p_select = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SELECT",
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
            arg_dim=8,
        ),
        status=PrimitiveStatus.STABLE,
    )
    p_reverse = bank.new_reverse_relative_primitive(
        ReverseRelativePrimitiveConfig(
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
        ),
        status=PrimitiveStatus.STABLE,
    )

    op_to_id = {
        "SHIFT": p_shift.primitive_id,
        "SELECT": p_select.primitive_id,
        "REVERSE": p_reverse.primitive_id,
    }

    # Reset forward call counts
    bank.reset_all_forward_call_counts()
    assert all(c == 0 for c in bank.forward_call_counts().values())

    # Create dummy examples for SHIFT(1) -> SELECT((0, 2))
    step1 = ProgramStep("SHIFT", {"amount": 1})
    step2 = ProgramStep("SELECT", {"indices": (0, 2, 4)})
    prog = Program(steps=(step1, step2))

    examples = []
    for seq in [(1, 2, 3, 4, 5, 6), (7, 8, 9, 0, 1, 2)]:
        res = run_program(prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="known",
            split="test",
            vocab_size=10,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(label="C", primitive_operations=("SHIFT", "SELECT")),
        )
        examples.append(ex)

    recipe = CompositionRecipe(
        "shift_select",
        (
            PrimitiveCall("SHIFT", {"amount": 1}),
            PrimitiveCall("SELECT", {"indices": (0, 2, 4)}),
        ),
    )

    logits = execute_composition_recipe(core, bank, op_to_id, examples, recipe=recipe)

    # Check output shape: batch_size=2, select output length = 6 // 2 = 3, vocab_size=10
    assert logits.shape == (2, 3, 10)

    # Check sparse forward call counts:
    # SHIFT was called once (for the batch)
    # SELECT was called once (for the batch)
    # REVERSE was NOT called (0 forward calls!)
    counts = bank.forward_call_counts()
    assert counts[p_shift.primitive_id] == 1
    assert counts[p_select.primitive_id] == 1
    assert counts[p_reverse.primitive_id] == 0

    # Check that no plastic capacity is used
    assert bank.total_parameter_count() == bank.persistent_parameter_count()


def test_composition_benchmark_config_validation() -> None:
    from apc.evaluation.composition_library_benchmark import (
        CompositionBenchmarkConfig,
        composition_benchmark_config_from_dict,
    )

    cfg = CompositionBenchmarkConfig(seed=123)
    d = cfg.to_dict()
    assert d["seed"] == 123
    restored = composition_benchmark_config_from_dict(d)
    assert restored == cfg

    with pytest.raises(ValueError, match="num_eval_examples_per_composition below minimum"):
        CompositionBenchmarkConfig(
            num_eval_examples_per_composition=10, min_eval_examples_per_composition=50
        )


def test_composition_benchmark_tiny_execution() -> None:
    """Smoke test running a tiny benchmark config on CPU."""
    from apc.evaluation.composition_library_benchmark import (
        CompositionBenchmarkConfig,
        run_composition_library_benchmark,
    )

    cfg = CompositionBenchmarkConfig(
        seed=42,
        vocab_size=10,
        sequence_length_range=(6, 8),
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        parameterized_train_steps=10,
        parameter_free_train_steps=10,
        core_train_steps=10,
        num_eval_examples_per_composition=5,
        min_eval_examples_per_composition=5,
        model={
            "d_model": 32,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
    )
    rep = run_composition_library_benchmark(cfg)
    assert rep.total_temporary_params == 0
    assert rep.total_unselected_calls == 0
    assert len(rep.results_by_composition) == 6
    assert 0.0 <= rep.mean_composition_exact_match <= 1.0

