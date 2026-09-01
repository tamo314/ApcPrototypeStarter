"""Wire the stable core, primitive bank, router, and plastic workspace
together into one executable model (Phase A Milestone A9 / Task 012).

No previous task actually combined these pieces: `apc.core.model` produces
logits on its own, `apc.primitives.router.Router` only scores candidate ids
against a hidden state, and `apc.consolidation.distill`/`shadow` operate on
generic `[N, d_model]` hidden-state tensors divorced from any real model
(both modules document this explicitly). `docs/design-docs/ARCHITECTURE.md`
section 3 asks the stable core to "expose insertion points for primitive
transforms"; this module is that insertion point, added now because Task
012 is the first task that needs the model to actually execute through the
bank and workspace to produce real task predictions (real exact-match,
real novelty, real consolidation/shadow data) rather than synthetic tensors.

Design choices not pinned down by the architecture doc (recorded here per
AGENTS.md workflow rather than hidden):

- **Combining primitive deltas.** `docs/design-docs/ARCHITECTURE.md`
  section 4 defines a single primitive as `P_i(h) = h + s_i(h) * B_i A_i
  h`; it does not say how to combine several selected primitives at once.
  Following `apc.consolidation.distill._combined_teacher_delta` /
  `apc.consolidation.shadow._combined_output` (Tasks 010/011), every
  active primitive's residual delta is computed from the *same* input
  hidden state and summed (parallel, not chained), so consolidation's
  "reproduce the combined delta" objective matches what execution actually
  produces.

- **The router's gate `s_i(h)`.** `apc.primitives.primitive.Primitive.
  forward`'s docstring leaves `s_i(h)` for "whatever module wires the
  stable core to the bank" to supply (Task 005). Here it is the router's
  top-k softmax weight for that primitive at that position -- but a
  softmax over a *single* real candidate is always 1.0 regardless of
  whether that primitive is actually relevant (there is nothing to
  normalize against), which would force every persistent primitive onto
  every input once the bank holds only one or two of them. `NULL_PRIMITIVE_ID`
  is a permanent, keyed-but-unassigned "apply nothing" candidate that
  always competes for routing mass; a bank primitive is only ever applied
  where the router prefers it over the null candidate (and over every
  other bank primitive). This is what makes selective reuse (as opposed to
  "always apply the only primitive that exists") learnable at small bank
  sizes -- see `calibrate_router` in `apc.evaluation.sequential_benchmark`.

- **Bank/router gradients.** Per ADR-0004 and architecture doc section 3,
  persistent primitives and the stable core are frozen outside of the
  dedicated router-calibration step, so `apply_bank` always runs under
  `torch.no_grad()`: its output is treated as an ordinary (non-leaf but
  ungraphed) tensor that `apply_workspace`'s trainable temporary transforms
  can still read and backprop through -- only the workspace's own low-rank
  weights receive gradient from a task loss computed on the combined
  output.

- **Task/content split is opt-in here (Phase A.1 Task A1-005).**
  `apply_bank`'s `route_state` parameter and `forward_logits`'s
  `use_split_state` flag let a caller route on `apc.core.model.
  EncodedState.task_state` while primitives/workspace transform
  `content_state`, but default to the pre-A1-005 behavior (route on the
  same tensor primitives transform). `apc.evaluation.baselines` and
  `apc.evaluation.sequential_benchmark` calibrate the router and measure
  novelty against plain `encode` hidden states; flipping their routing
  input to `task_state` without also recalibrating against it would route
  through an untrained input space, so that migration is left to whichever
  later task actually needs learned routing to consume `task_state` (e.g.
  A1-007 oracle routing or A1-015 learned routing), not bundled here. See
  ADR-0015 for why `task_state` itself is a parameter-free view rather
  than a learned projection.

- **Oracle `PrimitiveCall` routing adapter (Phase A.1 Correction Task
  A1-C007).** `apply_bank_with_oracle_calls`/`forward_logits_with_oracle_calls`/
  `generate_greedy_with_oracle_call`/`evaluate_exact_match_with_oracle_calls`
  below are `apply_bank`/`forward_logits`/`generate_greedy_with_capacity`/
  `evaluate_exact_match_with_capacity`'s oracle-routed siblings: instead of
  scoring `stable_ids` through a `Router`, they force exactly the bank
  primitive each caller-supplied `apc.environments.primitive_call.
  PrimitiveCall` names (`call.primitive_id`, one call per batch item) and
  apply it unconditionally (`gate=1.0`) -- "environment supplies oracle
  primitive IDs ... learned router fully bypassed" (`docs/
  CODEX_TASKS_PHASE_A1.md` A1-007). None of these four functions accepts a
  `Router` -- the learned routing path is not merely unused at runtime, it
  is structurally unreachable from here (A1-C007 Work item 2). Using a full
  `PrimitiveCall` rather than a bare `primitive_id` (A1-C007 Work item 1,
  via `apc.environments.generator.oracle_call_for_example`) is what stops
  ADR-0017's hidden-parameter non-identifiability from recurring one level
  into oracle routing: knowing *that* `SHIFT` is the right family is not
  enough to know *which* `amount` -- see `docs/design-docs/
  PARAMETERIZED_PRIMITIVE_CALLS.md` section 8. Multi-step oracle recipes
  (`apc.environments.generator.oracle_calls_for_example` returning more
  than one call) are Task A1-008's Composition Library; these four
  functions accept only one call per batch item and are the scope A1-007's
  own "oracle-routed K" targets, not A1-008's "oracle C".

- **Decoder-input audit and no-primitive/identity mode (Phase A.1
  Post-Correction Task A1-R002).** Work item 1 ("audit decoder inputs"):
  every decode entry point in this module builds its prompt as exactly
  `(specials.bos,) + example.input_tokens + (specials.sep,)` -- never
  `apc.core.data.build_prompt_tokens`/`encode_example` with
  `include_task_spec=True`, and never `example.task_spec` at all -- so no
  task-specification token (`apc.core.tokens.SharedCoreTokens`' task-start/
  task-end/operation/argument-value ranges) ever reaches `model.encode`
  through `forward_logits`, `generate_greedy_with_capacity`,
  `evaluate_exact_match_with_capacity`, or any of the oracle-routed or
  no-primitive functions below (`tests/test_decoder_leakage.py::
  test_causal_mode_functions_never_feed_task_tokens_to_encode` audits this
  directly, over the bank-routed, oracle-routed, and no-primitive functions,
  by capturing the actual ids passed to `model.encode`). Only
  BOS/SEP/EOS/PAD (`apc.core.tokens.SpecialTokens`, formatting-only
  sequence-boundary markers, never
  operation-specific) and the example's own content tokens appear. The one
  latent risk this audit found (Work item 2, "remove task-conditioned
  bypasses in causal mode"): `forward_logits(..., use_split_state=True)`
  computes `content_state` via `model.encode_split(input_ids)` -- a *single*
  causal pass whose `content_state` is literally `encode(input_ids)`, so if
  a caller ever passed a task-spec-carrying `input_ids` through this branch,
  `content_state` would have already attended over the task tokens (exactly
  ADR-0022's finding for `encode_split`, and exactly what Task A1-R001's
  `encode_task_content_split` two-pass factorization exists to avoid). No
  current caller does this (`use_split_state=True` is exercised only by
  Task A1-005's own unit tests, always with content-only `input_ids`), so
  there is no live bypass to remove; the fix recorded here is a documented
  restriction rather than a code deletion, since `encode_split`/
  `use_split_state` remain valid for A1-005's original task_state-routing
  purpose on content-only input. The causal *primitive* path (this module's
  bank/workspace/oracle-routed functions) must never source `content_state`
  from `encode_split`; only `encode` (on content-only ids) or, once wired,
  `encode_task_content_split`'s `content_state` is safe. Work items 3-4
  ("add no-primitive/identity mode", "evaluate decoder-only task
  performance"): `forward_logits_no_primitive`/`generate_greedy_no_primitive`/
  `evaluate_exact_match_no_primitive` below are the causal ablation matrix's
  "None" arm (`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 8) --
  no bank, router, or workspace parameter exists on any of them, so
  primitive execution is structurally unreachable, not merely unused. The
  A1-R002 STOP GATE (`apc.evaluation.decoder_leakage_gate`) trains a shared
  core with no task segment ever visible (mirroring Task A1-C004's
  `include_task_spec=False` negative control, ADR-0021) and evaluates it
  through `evaluate_exact_match_no_primitive`, checking that decoder-only
  performance stays materially below the future primitive-causality Correct
  target (0.95) rather than already solving the task through some bypass.

- **Trainable oracle-forced primitive execution (Phase A.1 Post-Correction
  Task A1-R003).** `apply_bank_with_oracle_calls_trainable`/
  `forward_logits_with_oracle_calls_trainable` are gradient-enabled siblings
  of A1-C007's `apply_bank_with_oracle_calls`/`forward_logits_with_
  oracle_calls`: identical oracle-forced per-example routing (now factored
  into a shared `_route_and_apply_oracle_calls` helper), but not wrapped in
  `torch.no_grad()`, so a task loss can actually train the oracle-selected
  primitives' `a_proj`/`b_proj` weights. The no-grad originals stay exactly
  as A1-C007 left them (same public signature, same behavior) -- they are
  the evaluation-time entry points for a bank whose primitives are already
  trained; the new `_trainable` siblings exist because A1-R003 is the first
  task that needs to *train* a primitive via oracle routing rather than only
  evaluate one. `apc.evaluation.parameter_free_primitive_gate` (the A1-R003
  STOP GATE) freezes a pretrained, task-blind `DecoderOnlyTransformer`
  (`apc.evaluation.shared_core_generalization.train_shared_core`,
  `include_task_spec=False`, restricted to `apc.environments.operations.
  DETERMINISTIC_OPERATION_NAMES`), registers one `Primitive` per operation
  family, and trains those primitives with the frozen core's `encode`
  output as input -- gradient reaches only the oracle-selected primitive
  for each example, never the frozen core (whose output already carries no
  grad of its own) and never an unselected primitive family.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SpecialTokens
from apc.environments.generator import Example, oracle_call_for_example
from apc.environments.primitive_call import PrimitiveCall
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveStatus
from apc.primitives.router import Router, RouterOutput

NULL_PRIMITIVE_ID = -1


def stable_candidate_ids(bank: PrimitiveBank) -> list[int]:
    """Enabled `STABLE` bank primitives: the candidate set `apply_bank`
    routes among during ordinary (non-PLASTIC) execution."""
    return [pid for pid in bank.ids_by_status(PrimitiveStatus.STABLE) if bank.get(pid).enabled]


def ensure_null_key(router: Router) -> None:
    """Register the permanent "apply nothing" routing candidate if absent."""
    if not router.has_primitive(NULL_PRIMITIVE_ID):
        router.add_primitive_key(NULL_PRIMITIVE_ID)


def sum_primitive_deltas(transforms: Sequence[Primitive], h: torch.Tensor) -> torch.Tensor:
    """`sum_i B_i(A_i(h))` over every *enabled* transform in `transforms`,
    each computed from the same `h` (see module docstring)."""
    total = torch.zeros_like(h)
    for transform in transforms:
        if transform.enabled:
            total = total + transform.b_proj(transform.a_proj(h))
    return total


def apply_bank(
    hidden: torch.Tensor,
    bank: PrimitiveBank,
    router: Router,
    stable_ids: Sequence[int] | None = None,
    *,
    route_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RouterOutput]:
    """Route among `stable_ids` (default: `stable_candidate_ids(bank)`) plus
    the permanent null candidate, and add each selected primitive's delta
    (computed from `hidden`, `[..., d_model]`) weighted by its routed weight.

    Always executed under `torch.no_grad()` (see module docstring); the
    returned hidden state is safe to feed into a trainable module (e.g.
    `apply_workspace`) afterwards.

    Only primitives selected at one or more batch/sequence positions are
    executed.  The selected hidden states are gathered per primitive, the
    low-rank transform is evaluated once for that gathered tensor, and its
    weighted delta is scattered back to the corresponding positions.  This
    preserves the prior parallel-delta semantics while making the primitive
    computation genuinely top-k sparse (Phase A.1 task A1-002).

    `route_state` (Phase A.1 Task A1-005): the tensor the *router* scores
    candidates against, e.g. a Stable Core's `task_state` (`z_task`,
    `apc.core.model.EncodedState`) rather than `hidden` itself. Must share
    `hidden`'s leading (batch/sequence) shape. Defaults to `hidden`, which
    reproduces every pre-A1-005 call site exactly -- routing and primitive
    execution read the same undivided tensor.
    """
    ensure_null_key(router)
    ids = list(stable_ids) if stable_ids is not None else stable_candidate_ids(bank)
    candidate_ids = [NULL_PRIMITIVE_ID, *ids]
    if route_state is None:
        route_state = hidden
    elif route_state.shape[:-1] != hidden.shape[:-1]:
        raise ValueError(
            f"route_state leading shape {tuple(route_state.shape[:-1])} must match "
            f"hidden leading shape {tuple(hidden.shape[:-1])}"
        )

    with torch.no_grad():
        router_out = router(route_state, candidate_ids)
        if not ids:
            return hidden, router_out

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_selected_ids = router_out.selected_ids.reshape(-1, router_out.selected_ids.shape[-1])
        flat_weights = router_out.weights.reshape(-1, router_out.weights.shape[-1])
        flat_delta = torch.zeros_like(flat_hidden)

        candidate_id_set = set(ids)
        selected_ids = {
            int(pid)
            for pid in flat_selected_ids.reshape(-1).tolist()
            if int(pid) in candidate_id_set and bank.get(int(pid)).enabled
        }
        executed_ids = tuple(sorted(selected_ids))
        router_out.executed_primitive_ids = executed_ids

        for pid in executed_ids:
            primitive = bank.get(pid)
            selected_mask = flat_selected_ids == pid
            position_mask = selected_mask.any(dim=-1)
            positions = position_mask.nonzero(as_tuple=False).squeeze(-1)
            selected_hidden = flat_hidden.index_select(0, positions)
            # Call Primitive.forward rather than its projections directly so
            # execution instrumentation reflects actual sparse work.
            delta = primitive(selected_hidden) - selected_hidden
            gates = (selected_mask.to(flat_weights.dtype) * flat_weights).sum(dim=-1)
            weighted_delta = gates.index_select(0, positions).unsqueeze(-1) * delta
            flat_delta.index_add_(0, positions, weighted_delta)

        return hidden + flat_delta.reshape_as(hidden), router_out


def apply_workspace(
    hidden: torch.Tensor,
    workspace: PlasticWorkspace,
    active_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """`hidden + sum_i B_i(A_i(hidden))` over `active_ids` (default: every
    currently allocated temporary transform). Gradient-enabled: this is the
    path PLASTIC training backpropagates through."""
    if not workspace.is_allocated:
        return hidden
    ids = list(active_ids) if active_ids is not None else workspace.ids()
    if not ids:
        return hidden
    transforms = workspace.get_many(ids)
    return hidden + sum_primitive_deltas(transforms, hidden)


def forward_logits(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    bank: PrimitiveBank,
    router: Router,
    *,
    stable_ids: Sequence[int] | None = None,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
    use_split_state: bool = False,
) -> tuple[torch.Tensor, RouterOutput]:
    """`input_ids -> logits`, executed through the stable core, the
    persistent bank (always applied), and the plastic workspace (applied
    only when `workspace` is given -- PLASTIC/CONSOLIDATE/SHADOW).

    `use_split_state` (Phase A.1 Task A1-005): when True, encodes via
    `model.encode_split` and routes the bank on `task_state` while the
    bank/workspace transform and `decode` read `content_state` -- see
    `apc.core.model.EncodedState`. Default False keeps every existing
    caller (`apc.evaluation.baselines`, `apc.evaluation.sequential_
    benchmark`, `apc.core.generation`-style callers) byte-for-byte
    identical to Phase A.1 up to A1-004: those modules calibrate routing
    and measure novelty against the plain `encode` hidden state, and
    switching their routing input to `task_state` without also
    recalibrating against it is a separate, not-yet-scoped change (see
    `docs/DECISIONS.md`).
    """
    if use_split_state:
        encoded = model.encode_split(input_ids)
        content, router_out = apply_bank(
            encoded.content_state, bank, router, stable_ids, route_state=encoded.task_state
        )
    else:
        content, router_out = apply_bank(model.encode(input_ids), bank, router, stable_ids)
    if workspace is not None:
        content = apply_workspace(content, workspace, workspace_ids)
    return model.decode(content), router_out


@torch.no_grad()
def generate_greedy_with_capacity(
    model: DecoderOnlyTransformer,
    bank: PrimitiveBank,
    router: Router,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
    *,
    stable_ids: Sequence[int] | None = None,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """`apc.core.generation.generate_greedy`, routed through the bank/
    workspace instead of calling `model` directly."""
    model.eval()
    generated = prompt_ids
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = forward_logits(
            model,
            generated,
            bank,
            router,
            stable_ids=stable_ids,
            workspace=workspace,
            workspace_ids=workspace_ids,
        )
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


def evaluate_exact_match_with_capacity(
    model: DecoderOnlyTransformer,
    examples: Sequence[Example],
    specials: SpecialTokens,
    bank: PrimitiveBank,
    router: Router,
    *,
    stable_ids: Sequence[int] | None = None,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    max_extra_tokens: int = 2,
) -> tuple[float, list[tuple[int, ...]]]:
    """`apc.core.generation.evaluate_exact_match`, routed through the bank/
    workspace."""
    predictions: list[tuple[int, ...]] = []
    correct = 0
    for example in examples:
        prompt = (specials.bos,) + example.input_tokens + (specials.sep,)
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        max_new_tokens = len(example.target_tokens) + max_extra_tokens
        prediction = generate_greedy_with_capacity(
            model,
            bank,
            router,
            prompt_ids,
            specials.eos,
            max_new_tokens,
            stable_ids=stable_ids,
            workspace=workspace,
            workspace_ids=workspace_ids,
        )
        predictions.append(prediction)
        if prediction == example.target_tokens:
            correct += 1
    return correct / len(examples), predictions


def forward_logits_no_primitive(
    model: DecoderOnlyTransformer, input_ids: torch.Tensor
) -> torch.Tensor:
    """`input_ids -> logits` with no bank, router, or workspace involved at
    all (Task A1-R002): the causal ablation matrix's "None"/identity arm
    (`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 8, `AGENTS.md`'s
    causal primitive evidence rule) -- `model.decode(model.encode(input_ids))`,
    i.e. exactly `model.forward`, given a name and a place in this module so
    future causal-primitive gates (A1-R003 onward) can call "Correct"
    (`forward_logits_with_oracle_calls`), "Wrong" (the same function with a
    deliberately incorrect call), and "None" (this function) through one
    consistent `apc.core.execution` surface instead of reaching into
    `apc.core.generation` for the third arm alone.

    There is no `stable_ids`/`bank`/`router`/`workspace` parameter to accept
    here -- unlike `forward_logits`, which always applies the bank -- so a
    caller cannot smuggle primitive execution back in through this entry
    point (`tests/test_decoder_leakage.py::
    test_no_primitive_functions_never_accept_bank_router_or_workspace`).
    """
    return model.decode(model.encode(input_ids))


@torch.no_grad()
def generate_greedy_no_primitive(
    model: DecoderOnlyTransformer,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
) -> tuple[int, ...]:
    """`generate_greedy_with_capacity`'s no-primitive sibling (Task A1-R002):
    greedy decoding through `forward_logits_no_primitive` alone."""
    model.eval()
    generated = prompt_ids
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits = forward_logits_no_primitive(model, generated)
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


def evaluate_exact_match_no_primitive(
    model: DecoderOnlyTransformer,
    examples: Sequence[Example],
    specials: SpecialTokens,
    *,
    device: torch.device | str = "cpu",
    max_extra_tokens: int = 2,
) -> tuple[float, list[tuple[int, ...]]]:
    """`evaluate_exact_match_with_capacity`'s no-primitive sibling (Task
    A1-R002): decodes every example's target from a content-only prompt
    (`bos + input_tokens + sep`, matching every other function in this
    module -- never `example.task_spec`, never rendered through
    `apc.core.data.encode_task_spec`/`build_prompt_tokens(include_task_spec=
    True)`) with no bank, router, or workspace anywhere in the call graph.
    This is the A1-R002 STOP GATE's "no-primitive path" acceptance measure
    (`apc.evaluation.decoder_leakage_gate`); numerically identical to
    `apc.core.generation.evaluate_exact_match(..., include_task_spec=False)`
    given the same model/examples (`tests/test_decoder_leakage.py::
    test_evaluate_exact_match_no_primitive_matches_plain_generation`), kept
    as a distinct entry point in this module rather than re-exported from
    `apc.core.generation` so a future Correct/Wrong/None benchmark can call
    all three arms through one consistent `apc.core.execution` surface.
    """
    predictions: list[tuple[int, ...]] = []
    correct = 0
    for example in examples:
        prompt = (specials.bos,) + example.input_tokens + (specials.sep,)
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        max_new_tokens = len(example.target_tokens) + max_extra_tokens
        prediction = generate_greedy_no_primitive(
            model, prompt_ids, specials.eos, max_new_tokens
        )
        predictions.append(prediction)
        if prediction == example.target_tokens:
            correct += 1
    return correct / len(examples), predictions


@dataclass
class OracleRoutingOutput:
    """Result of one `apply_bank_with_oracle_calls` call: the oracle
    `PrimitiveCall`s that forced this batch's routing, the primitive id each
    named (`selected_ids`, one per batch item, always exactly
    `tuple(call.primitive_id for call in calls)` -- A1-007's "selected IDs
    exactly match metadata" holds by construction here, not by measurement),
    and which of those ids actually had a bank entry to execute
    (`executed_primitive_ids`, unique and sorted, mirroring `RouterOutput.
    executed_primitive_ids`'s execution-instrumentation role for the
    learned path)."""

    calls: tuple[PrimitiveCall, ...]
    selected_ids: tuple[int, ...]
    executed_primitive_ids: tuple[int, ...]


def _route_and_apply_oracle_calls(
    hidden: torch.Tensor,
    bank: PrimitiveBank,
    calls: Sequence[PrimitiveCall],
) -> tuple[torch.Tensor, OracleRoutingOutput]:
    """Shared implementation behind `apply_bank_with_oracle_calls` (no-grad,
    evaluation/inference -- Task A1-C007) and `apply_bank_with_oracle_calls_
    trainable` (gradient-enabled, primitive training -- Task A1-R003): force
    exactly one bank primitive per batch item -- `calls[b].primitive_id` --
    instead of scoring candidates through a `Router`. Applied unconditionally
    (`gate=1.0`; there is no competing candidate to softmax against, unlike
    the learned router's permanent null candidate) at every non-batch
    position of that batch item.

    This function itself does not decide whether gradient is tracked --
    every op here is an ordinary differentiable tensor op (`Tensor.
    index_add`, not the in-place `index_add_`, specifically so this same
    implementation is safe to reuse from a gradient-enabled call site); each
    public wrapper below chooses via `torch.no_grad()` or not.

    `hidden`: `[batch, ..., d_model]`; `len(calls)` must equal `hidden`'s
    batch dimension (`hidden.shape[0]`) -- one oracle call per example, not
    per position, matching the current online generator's depth-1 "one
    operation for the whole sequence" semantics (multi-step recipes are
    Task A1-008's Composition Library, see module docstring).

    Raises:
        ValueError: `len(calls) != hidden.shape[0]`, or a call names a bank
            primitive that exists but is disabled.
        KeyError: a call names a primitive id absent from `bank` entirely
            (via `PrimitiveBank.get`) -- e.g. no bank primitive has yet been
            assigned that operation's family id.
    """
    if hidden.dim() < 2:
        raise ValueError(
            f"hidden must have at least 2 dims (batch, ..., d_model), got shape "
            f"{tuple(hidden.shape)}"
        )
    batch = hidden.shape[0]
    if len(calls) != batch:
        raise ValueError(
            f"len(calls) ({len(calls)}) must equal hidden's batch dimension ({batch}) -- "
            "apply_bank_with_oracle_calls forces exactly one oracle PrimitiveCall per batch item"
        )

    primitive_ids = [call.primitive_id for call in calls]
    for call, primitive_id in zip(calls, primitive_ids, strict=True):
        primitive = bank.get(primitive_id)
        if not primitive.enabled:
            raise ValueError(
                f"Oracle call selected disabled primitive id {primitive_id} "
                f"(operation {call.operation!r}); a disabled family cannot be oracle-executed"
            )

    id_tensor = torch.tensor(primitive_ids, dtype=torch.long, device=hidden.device)
    broadcast_shape = (batch,) + (1,) * (hidden.dim() - 2)
    flat_ids = id_tensor.view(broadcast_shape).expand(hidden.shape[:-1]).reshape(-1)

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_delta = torch.zeros_like(flat_hidden)
    executed_ids: list[int] = []
    for primitive_id in sorted(set(primitive_ids)):
        primitive = bank.get(primitive_id)
        positions = (flat_ids == primitive_id).nonzero(as_tuple=False).squeeze(-1)
        selected_hidden = flat_hidden.index_select(0, positions)
        # Call Primitive.forward (not its projections directly) so
        # execution instrumentation reflects actual oracle-forced work,
        # matching apply_bank's A1-002 sparse-execution convention.
        delta = primitive(selected_hidden) - selected_hidden
        flat_delta = flat_delta.index_add(0, positions, delta)
        executed_ids.append(primitive_id)

    result = hidden + flat_delta.reshape_as(hidden)

    return result, OracleRoutingOutput(
        calls=tuple(calls),
        selected_ids=tuple(primitive_ids),
        executed_primitive_ids=tuple(sorted(executed_ids)),
    )


def apply_bank_with_oracle_calls(
    hidden: torch.Tensor,
    bank: PrimitiveBank,
    calls: Sequence[PrimitiveCall],
) -> tuple[torch.Tensor, OracleRoutingOutput]:
    """`apply_bank`'s oracle-routed sibling (Task A1-C007) -- see
    `_route_and_apply_oracle_calls` for the shared routing/execution logic.

    Always executed under `torch.no_grad()`, matching `apply_bank`: the
    returned hidden state is an ordinary (non-leaf but ungraphed) tensor,
    safe to feed into a trainable module (e.g. `apply_workspace`) afterwards.
    This is the evaluation/inference entry point -- a bank whose primitives
    are already trained (e.g. `apply_bank_with_oracle_calls_trainable`'s
    result, Task A1-R003). To train primitive parameters via oracle-forced
    routing, use `apply_bank_with_oracle_calls_trainable` instead.

    Raises: see `_route_and_apply_oracle_calls`.
    """
    with torch.no_grad():
        return _route_and_apply_oracle_calls(hidden, bank, calls)


def apply_bank_with_oracle_calls_trainable(
    hidden: torch.Tensor,
    bank: PrimitiveBank,
    calls: Sequence[PrimitiveCall],
) -> tuple[torch.Tensor, OracleRoutingOutput]:
    """`apply_bank_with_oracle_calls`'s gradient-enabled sibling (Task
    A1-R003, "train primitives"): identical oracle-forced routing/execution
    (`_route_and_apply_oracle_calls`), but *not* wrapped in `torch.
    no_grad()` -- the returned hidden state carries a gradient path back
    into whichever selected primitives' `a_proj`/`b_proj` weights require
    grad, so a task loss computed on top of it can train those primitives
    directly (mirroring `apply_workspace`'s gradient-enabled role for
    PLASTIC training, but with oracle-forced per-example selection instead
    of one shared `active_ids` set for the whole batch).

    Intended call pattern (Task A1-R003's parameter-free oracle primitive
    benchmark): freeze a pretrained `DecoderOnlyTransformer`'s parameters,
    register one `Primitive` per operation family in `bank`, and train via
    `forward_logits_with_oracle_calls_trainable` + a token-level
    cross-entropy loss -- gradient reaches only the oracle-selected
    primitive's own parameters for each example, never the frozen Stable
    Core (whose `hidden` input carries no grad of its own) and never any
    primitive family not selected by that example's oracle call.

    Raises: see `_route_and_apply_oracle_calls`.
    """
    return _route_and_apply_oracle_calls(hidden, bank, calls)


def forward_logits_with_oracle_calls(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    bank: PrimitiveBank,
    calls: Sequence[PrimitiveCall],
    *,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, OracleRoutingOutput]:
    """`forward_logits`'s oracle-routed sibling (Task A1-C007): `input_ids
    -> logits`, executed through the stable core, the persistent bank
    (oracle-forced via `apply_bank_with_oracle_calls`, always applied), and
    the plastic workspace (applied only when `workspace` is given).

    `model.encode(input_ids)` is used directly (not `encode_split`):
    `apc.core.model.DecoderOnlyTransformer.encode_split`'s `content_state`
    is documented as exactly `encode`'s output, and oracle routing needs no
    `route_state` at all (selection is forced, not scored), so there is no
    `use_split_state` flag to thread through here.
    """
    content, routing_out = apply_bank_with_oracle_calls(model.encode(input_ids), bank, calls)
    if workspace is not None:
        content = apply_workspace(content, workspace, workspace_ids)
    return model.decode(content), routing_out


def forward_logits_with_oracle_calls_trainable(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    bank: PrimitiveBank,
    calls: Sequence[PrimitiveCall],
    *,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, OracleRoutingOutput]:
    """`forward_logits_with_oracle_calls`'s gradient-enabled sibling (Task
    A1-R003): identical composition (`encode` -> oracle-forced bank ->
    optional workspace -> `decode`), but routes through
    `apply_bank_with_oracle_calls_trainable` instead of the no-grad
    evaluation entry point, so the returned logits carry a gradient path
    into the oracle-selected primitives.

    `model.encode(input_ids)` is called with no explicit `torch.no_grad()`
    here: the intended caller (Task A1-R003's primitive-training loop) has
    already frozen every `model` parameter (`requires_grad=False`), so the
    encode forward produces a plain, gradient-free `hidden` tensor either
    way -- explicit `no_grad()` would only be an optimization, not a
    correctness requirement -- and leaving it out keeps this function usable
    if a future caller ever wants gradient to reach the Stable Core too.
    """
    content, routing_out = apply_bank_with_oracle_calls_trainable(
        model.encode(input_ids), bank, calls
    )
    if workspace is not None:
        content = apply_workspace(content, workspace, workspace_ids)
    return model.decode(content), routing_out


@torch.no_grad()
def generate_greedy_with_oracle_call(
    model: DecoderOnlyTransformer,
    bank: PrimitiveBank,
    call: PrimitiveCall,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
    *,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """`generate_greedy_with_capacity`'s oracle-routed sibling (Task
    A1-C007): `prompt_ids` must be a single (`batch == 1`) prompt -- `call`
    is the one oracle `PrimitiveCall` forced at every generation step for
    that whole example."""
    if prompt_ids.shape[0] != 1:
        raise ValueError(
            f"generate_greedy_with_oracle_call takes one prompt at a time (batch == 1), "
            f"got batch {prompt_ids.shape[0]}"
        )
    model.eval()
    generated = prompt_ids
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = forward_logits_with_oracle_calls(
            model, generated, bank, (call,), workspace=workspace, workspace_ids=workspace_ids
        )
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


def evaluate_exact_match_with_oracle_calls(
    model: DecoderOnlyTransformer,
    examples: Sequence[Example],
    specials: SpecialTokens,
    bank: PrimitiveBank,
    oracle_call_provider: Callable[[Example], PrimitiveCall] = oracle_call_for_example,
    *,
    workspace: PlasticWorkspace | None = None,
    workspace_ids: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    max_extra_tokens: int = 2,
) -> tuple[float, list[tuple[int, ...]], list[PrimitiveCall]]:
    """`evaluate_exact_match_with_capacity`'s oracle-routed sibling: the
    A1-007 evaluation API entry point Task A1-C007 prepares (`docs/
    CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C007 Work item 3, "Update A1-007
    evaluation API to accept an oracle call provider").

    `oracle_call_provider` converts each `Example` to the single oracle
    `PrimitiveCall` that forces this example's routing -- defaults to
    `apc.environments.generator.oracle_call_for_example` (Work item 1,
    "Convert environment oracle metadata into PrimitiveCall"), but any
    `Callable[[Example], PrimitiveCall]` may be substituted (e.g. a future
    A1-007 benchmark that also validates against `Example.oracle_metadata`).
    `Router` never appears anywhere in this call graph: the learned routing
    path is not merely unused at runtime here, it is structurally
    unreachable (Work item 2, "the learned path never receives oracle
    calls").

    Returns `(exact_match, predictions, calls_used)` -- `calls_used[i]` is
    the oracle call `oracle_call_provider` produced for `examples[i]`, so a
    caller can check "selected IDs exactly match metadata" (A1-007 Accept)
    via `[c.primitive_id for c in calls_used]` against the examples' own
    ground truth.
    """
    predictions: list[tuple[int, ...]] = []
    calls_used: list[PrimitiveCall] = []
    correct = 0
    for example in examples:
        call = oracle_call_provider(example)
        calls_used.append(call)
        prompt = (specials.bos,) + example.input_tokens + (specials.sep,)
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        max_new_tokens = len(example.target_tokens) + max_extra_tokens
        prediction = generate_greedy_with_oracle_call(
            model,
            bank,
            call,
            prompt_ids,
            specials.eos,
            max_new_tokens,
            workspace=workspace,
            workspace_ids=workspace_ids,
        )
        predictions.append(prediction)
        if prediction == example.target_tokens:
            correct += 1
    return correct / len(examples), predictions, calls_used
