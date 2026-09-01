"""Special tokens for the decoder-only baseline, layered on the environment vocab.

The symbolic environment (`apc.environments`) uses a closed vocabulary
`0 .. vocab_size - 1` with no reserved id for model-only control tokens.
`SpecialTokens` appends PAD/BOS/SEP/EOS after the environment vocabulary so
model input ids never collide with environment token ids.

`SharedCoreTokens` (Phase A.1 Correction Task A1-C003) extends that scheme
with a model-visible task-specification segment: TASK_START/TASK_END framing
tokens, one operation-identity token per registered
`apc.environments.operations.Operation`, and a shared span of argument-value
tokens. See `apc.core.data.encode_task_spec` for how a
`apc.environments.task_spec.TaskSpec` is rendered into this token space.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    """Control token ids appended after the environment vocabulary."""

    env_vocab_size: int
    pad: int
    bos: int
    sep: int
    eos: int

    @property
    def model_vocab_size(self) -> int:
        return self.env_vocab_size + 4


def build_special_tokens(env_vocab_size: int) -> SpecialTokens:
    """Derive PAD/BOS/SEP/EOS ids immediately after the environment vocabulary."""
    if env_vocab_size < 1:
        raise ValueError(f"env_vocab_size must be >= 1, got {env_vocab_size}")
    return SpecialTokens(
        env_vocab_size=env_vocab_size,
        pad=env_vocab_size,
        bos=env_vocab_size + 1,
        sep=env_vocab_size + 2,
        eos=env_vocab_size + 3,
    )


@dataclass(frozen=True)
class SharedCoreTokens(SpecialTokens):
    """`SpecialTokens` plus a model-visible task-specification token segment
    (Task A1-C003).

    PAD/BOS/SEP/EOS keep the exact ids `build_special_tokens` would assign
    for the same `env_vocab_size` -- `SharedCoreTokens` is a strict
    superset, not a parallel numbering scheme, so content/control token ids
    already trained under plain `SpecialTokens` (e.g. Task A1-006's
    per-operation gate) remain meaningful if ever reused here. The new
    range is, in order:

    - `task_start`, `task_end`: framing tokens around the task segment.
    - `op_base .. op_base + num_operations - 1`: one token per registered
      operation, indexed by `apc.environments.task_spec.operation_id`.
    - `arg_base .. arg_base + arg_span - 1`: a shared span of argument-value
      tokens, indexed by the raw integer argument value (see
      `apc.core.data.encode_task_spec`). One shared span rather than a
      separate token space per argument name -- `docs/design-docs/
      PARAMETERIZED_PRIMITIVE_CALLS.md` section 7 warns against
      over-engineering argument transport for this correction, and every
      current parameterized operation (`SHIFT.amount`, `SELECT.indices`,
      `COUNT.target`, `BIND.query_key`) is already a plain non-negative
      integer below `max(vocab_size, max content length)`.
    """

    task_start: int
    task_end: int
    op_base: int
    num_operations: int
    arg_base: int
    arg_span: int

    def __post_init__(self) -> None:
        if self.num_operations < 1:
            raise ValueError(f"num_operations must be >= 1, got {self.num_operations}")
        if self.arg_span < 1:
            raise ValueError(f"arg_span must be >= 1, got {self.arg_span}")

    @property
    def model_vocab_size(self) -> int:
        return self.arg_base + self.arg_span

    def operation_token(self, op_id: int) -> int:
        """The token id for operation identity `op_id` (see
        `apc.environments.task_spec.operation_id`)."""
        if not 0 <= op_id < self.num_operations:
            raise ValueError(
                f"op_id must be in [0, {self.num_operations}), got {op_id}"
            )
        return self.op_base + op_id

    def argument_value_token(self, value: int) -> int:
        """The token id for the raw integer argument value `value`."""
        if not 0 <= value < self.arg_span:
            raise ValueError(f"argument value must be in [0, {self.arg_span}), got {value}")
        return self.arg_base + value


def build_shared_core_tokens(
    env_vocab_size: int, *, num_operations: int, arg_span: int
) -> SharedCoreTokens:
    """Derive `SharedCoreTokens` after the environment vocabulary.

    `num_operations` and `arg_span` size the operation-identity and
    argument-value ranges -- see `apc.environments.task_spec.
    num_registered_operations` and `apc.environments.task_spec.
    default_argument_value_span` for the values Phase A.1 Correction call
    sites should normally pass.
    """
    base = build_special_tokens(env_vocab_size)
    task_start = base.eos + 1
    task_end = task_start + 1
    op_base = task_end + 1
    arg_base = op_base + num_operations
    return SharedCoreTokens(
        env_vocab_size=base.env_vocab_size,
        pad=base.pad,
        bos=base.bos,
        sep=base.sep,
        eos=base.eos,
        task_start=task_start,
        task_end=task_end,
        op_base=op_base,
        num_operations=num_operations,
        arg_base=arg_base,
        arg_span=arg_span,
    )
