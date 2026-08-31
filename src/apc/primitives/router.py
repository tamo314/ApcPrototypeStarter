"""Top-k primitive router (Phase A Milestone A3 / Task 005).

Implements the router from `docs/design-docs/ARCHITECTURE.md` section 6:
a learned query projection scores a hidden state against a learned key
per candidate primitive, hard-selects the top-k candidates, and produces
soft gate weights inside that selected top-k.

The router exposes everything section 6 requires:
- selected IDs (`RouterOutput.selected_ids`);
- score distribution (`RouterOutput.probs`, softmax over every candidate);
- entropy (`RouterOutput.entropy`);
- per-primitive usage (`Router.usage_counts`).

Gradient-isolation design choice (Task 005 acceptance criterion): the
top-k index selection (`torch.topk`) is a non-differentiable operation,
and the gate `weights` are a softmax computed only over the *selected*
candidates' scores. A loss built from `weights`/`selected_ids` therefore
back-propagates only into the query projection and the selected
candidates' key vectors -- unselected candidates get no gradient from
that path. `probs` (the full distribution used for entropy/novelty
logging) is a separate softmax over every candidate and does receive
gradient at every candidate key when used directly in a loss; it is a
logging signal, not the routed compute path.

The router only knows about candidate ids and learned key vectors, not
about `Primitive`/`PrimitiveBank` instances -- combining the returned
gate weights with actual primitive forward passes belongs to whatever
module wires the stable core to the bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

_EPS = 1e-9


@dataclass(frozen=True)
class RouterConfig:
    """Explicit, serializable configuration for a `Router`."""

    d_model: int
    top_k: int = 2
    score_dim: int | None = None
    score_fn: str = "dot"  # "dot" | "cosine"

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.score_dim is not None and self.score_dim < 1:
            raise ValueError(f"score_dim must be >= 1, got {self.score_dim}")
        if self.score_fn not in ("dot", "cosine"):
            raise ValueError(f"score_fn must be 'dot' or 'cosine', got {self.score_fn!r}")

    @property
    def resolved_score_dim(self) -> int:
        return self.score_dim if self.score_dim is not None else self.d_model


@dataclass
class RouterOutput:
    """Result of one `Router.forward` call.

    Shapes below use `...` for whatever leading (batch/sequence) shape
    `h` had; `k_eff = min(top_k, len(candidate_ids))` and `C =
    len(candidate_ids)`.
    """

    candidate_ids: list[int]
    selected_ids: torch.Tensor  # long, [..., k_eff]
    weights: torch.Tensor  # float, [..., k_eff], softmax over the top-k, sums to 1
    probs: torch.Tensor  # float, [..., C], softmax over every candidate
    entropy: torch.Tensor  # float, [...], entropy of `probs`


class Router(nn.Module):
    """Scores a hidden state against a candidate primitive set and
    hard-selects the top-k, per `docs/design-docs/ARCHITECTURE.md` section 6."""

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_proj = nn.Linear(config.d_model, config.resolved_score_dim)
        self._keys = nn.ParameterDict()
        self.usage_count: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def ids(self) -> list[int]:
        return sorted(int(key) for key in self._keys)

    def has_primitive(self, primitive_id: int) -> bool:
        return str(primitive_id) in self._keys

    def add_primitive_key(self, primitive_id: int) -> None:
        """Register a learned score-space key vector for `primitive_id`."""
        key = str(primitive_id)
        if key in self._keys:
            raise ValueError(f"Router already has a key for primitive id {primitive_id}")
        vector = torch.empty(self.config.resolved_score_dim)
        nn.init.normal_(vector, mean=0.0, std=0.02)
        self._keys[key] = nn.Parameter(vector)
        self.usage_count.setdefault(primitive_id, 0)

    def key_parameter(self, primitive_id: int) -> nn.Parameter:
        """The learned key vector for `primitive_id`, for callers that need
        to build a targeted optimizer over specific keys (e.g. a router
        calibration step run only over newly-registered ids)."""
        key = str(primitive_id)
        if key not in self._keys:
            raise KeyError(f"No router key registered for primitive id {primitive_id}")
        param = self._keys[key]
        assert isinstance(param, nn.Parameter)
        return param

    def remove_primitive_key(self, primitive_id: int) -> None:
        key = str(primitive_id)
        if key not in self._keys:
            raise KeyError(f"No router key for primitive id {primitive_id}")
        del self._keys[key]
        self.usage_count.pop(primitive_id, None)

    def usage_counts(self) -> dict[int, int]:
        return dict(self.usage_count)

    def _stacked_keys(self, candidate_ids: list[int]) -> torch.Tensor:
        vectors = []
        for primitive_id in candidate_ids:
            key = str(primitive_id)
            if key not in self._keys:
                raise KeyError(f"No router key registered for primitive id {primitive_id}")
            vectors.append(self._keys[key])
        return torch.stack(vectors, dim=0)

    def _record_usage(self, selected_ids: torch.Tensor) -> None:
        for primitive_id in selected_ids.reshape(-1).tolist():
            self.usage_count[primitive_id] = self.usage_count.get(primitive_id, 0) + 1

    def forward(self, h: torch.Tensor, candidate_ids: list[int]) -> RouterOutput:
        """Route `h` (`[..., d_model]`) to at most `top_k` of `candidate_ids`.

        `candidate_ids` must all have a registered key (see
        `add_primitive_key`). If fewer candidates than `top_k` are given,
        all of them are selected -- "at most k", never an error.
        """
        if not candidate_ids:
            raise ValueError("candidate_ids must be non-empty")

        keys = self._stacked_keys(candidate_ids)  # [C, score_dim]
        query = self.query_proj(h)  # [..., score_dim]

        if self.config.score_fn == "cosine":
            query = F.normalize(query, dim=-1, eps=_EPS)
            keys = F.normalize(keys, dim=-1, eps=_EPS)

        scores = query @ keys.transpose(0, 1)  # [..., C]
        probs = F.softmax(scores, dim=-1)
        entropy = -(probs * (probs + _EPS).log()).sum(dim=-1)

        k_eff = min(self.config.top_k, len(candidate_ids))
        topk_scores, topk_idx = torch.topk(scores, k_eff, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        id_lookup = torch.tensor(candidate_ids, dtype=torch.long, device=h.device)
        selected_ids = id_lookup[topk_idx]

        self._record_usage(selected_ids)

        return RouterOutput(
            candidate_ids=list(candidate_ids),
            selected_ids=selected_ids,
            weights=weights,
            probs=probs,
            entropy=entropy,
        )
