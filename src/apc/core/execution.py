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
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SpecialTokens
from apc.environments.generator import Example
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
) -> tuple[torch.Tensor, RouterOutput]:
    """Route `hidden` (`[..., d_model]`) among `stable_ids` (default:
    `stable_candidate_ids(bank)`) plus the permanent null candidate, and
    add each selected primitive's delta weighted by its routed weight.

    Always executed under `torch.no_grad()` (see module docstring); the
    returned hidden state is safe to feed into a trainable module (e.g.
    `apply_workspace`) afterwards.

    Not currently sparse in compute (Task 014 review finding): every
    enabled primitive in `ids` has its delta computed unconditionally --
    the router's top-k selection only zeroes out the *weight* of
    non-selected primitives in the sum below, it does not skip computing
    them. `stable_candidate_ids(bank)`-sized routing cost is paid every
    forward pass regardless of `router.config.top_k`. This means a
    caller-side "active parameters per inference step" figure derived from
    `stable_ids` (as `apc.evaluation.sequential_benchmark.EventReport`
    currently does) is not distinguishable from persistent parameter
    count -- see `docs/exec-plans/completed/PHASE_A_RESULT.md` section 3.2.
    """
    ensure_null_key(router)
    ids = list(stable_ids) if stable_ids is not None else stable_candidate_ids(bank)
    candidate_ids = [NULL_PRIMITIVE_ID, *ids]

    with torch.no_grad():
        router_out = router(hidden, candidate_ids)
        if not ids:
            return hidden, router_out

        combined_delta = torch.zeros_like(hidden)
        for pid in ids:
            primitive = bank.get(pid)
            if not primitive.enabled:
                continue
            weight = ((router_out.selected_ids == pid).float() * router_out.weights).sum(
                dim=-1, keepdim=True
            )
            combined_delta = combined_delta + weight * primitive.b_proj(primitive.a_proj(hidden))
        return hidden + combined_delta, router_out


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
) -> tuple[torch.Tensor, RouterOutput]:
    """`input_ids -> logits`, executed through the stable core, the
    persistent bank (always applied), and the plastic workspace (applied
    only when `workspace` is given -- PLASTIC/CONSOLIDATE/SHADOW)."""
    hidden = model.encode(input_ids)
    hidden, router_out = apply_bank(hidden, bank, router, stable_ids)
    if workspace is not None:
        hidden = apply_workspace(hidden, workspace, workspace_ids)
    return model.decode(hidden), router_out


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
