# Parameterized Primitive Calls

## 1. Motivation

A1-006 revealed that several operations were not functions of visible content alone.

Examples include:

- SELECT requires an index,
- COUNT requires a target,
- SHIFT requires an offset,
- BIND requires a key or binding specification.

These are not necessarily separate primitives. They are better represented as a reusable primitive plus arguments.

## 2. PrimitiveCall abstraction

Introduce a model-facing execution abstraction:

```python
@dataclass(frozen=True)
class PrimitiveCall:
    primitive_id: int
    arguments: Mapping[str, Tensor | int | float | str]
```

The exact concrete schema may be stricter and tensor-oriented.

The important distinction is:

- `primitive_id` selects the reusable computation family,
- `arguments` configure the computation instance.

## 3. Examples

```text
PrimitiveCall(SHIFT, {"amount": 2})
PrimitiveCall(SELECT, {"index": 3})
PrimitiveCall(COUNT, {"target": token_x})
PrimitiveCall(BIND, {"key": token_k})
```

## 4. Task specification

Synthetic examples should expose an explicit task/control segment.

Conceptually:

```text
[TASK]
operation = SHIFT
amount = 2

[CONTENT]
a b c d
```

The wire format may use tokens rather than structured objects, but the information must be model-visible.

## 5. Representation split

The Stable Core should encode:

```text
task specification -> z_task
content -> h_content
```

`z_task` should support:

- operation identification,
- argument decoding/conditioning,
- later primitive routing.

`h_content` should support:

- operand/state representation,
- primitive execution.

## 6. Execution API

Preferred conceptual API:

```python
encoded = stable_core.encode_split(input_ids)

call = router_or_oracle(encoded.task_state)

output_state = primitive_executor(
    content_state=encoded.content_state,
    primitive_call=call,
)
```

For compositions:

```python
recipe = [
    PrimitiveCall(...),
    PrimitiveCall(...),
]
```

## 7. Argument transport

Phase A.1 may start with simple discrete arguments.

Possible implementations:

- special task tokens,
- learned argument embeddings,
- integer buckets,
- explicit scalar features.

Do not over-engineer argument polymorphism in the first correction.

## 8. Oracle interface update

A1-007 and later oracle-routing experiments should prefer oracle **PrimitiveCall** rather than only oracle primitive ID.

This prevents a false failure where the correct primitive family is selected but required arguments remain hidden.

## 9. Persistent-capacity implication

Parameterized primitives should reduce primitive-bank growth.

Example:

```text
bad:
SHIFT_1
SHIFT_2
SHIFT_3
SHIFT_4

preferred:
SHIFT(amount)
```

This is directly aligned with APC's goal of maximizing reusable capability per persistent parameter.

## 10. Out of scope

This correction does not require:

- arbitrary function signatures,
- natural-language tool calling,
- symbolic theorem proving,
- dynamic code generation.

Only the existing synthetic operations need to be represented cleanly.
