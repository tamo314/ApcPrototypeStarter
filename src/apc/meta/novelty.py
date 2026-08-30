"""Novelty estimator: error + router-entropy signal (Phase A Milestone A6 / Task 008).

Implements the first slice of the novelty score from
`docs/design-docs/ARCHITECTURE.md` section 8:

    N = alpha*E + beta*U + gamma*H + delta*G

Task 008 only implements the `E` (task error) and `H` (router entropy)
terms, per Milestone A6: "First controller may use only error + router
entropy; gradient novelty can be added after the loop works." `beta`
(predictive uncertainty `U`) and `delta` (gradient novelty `G`) do not
exist yet -- they are not stubbed to zero, they are simply absent from
`NoveltyConfig`, so a future task adding them is a visible, additive
change rather than flipping an existing default.

`NoveltySignals` holds the raw, already-observable measurements (current
task error and the current `Router` step's entropy, see
`apc.primitives.router.RouterOutput.entropy`) that `NoveltyEstimator`
combines into the single scalar score consumed by
`apc.meta.controller.Controller`. Router entropy has no fixed scale (it
depends on the number of routed candidates), so it is optionally
normalized by `max_router_entropy` (typically `log(num_candidates)`,
i.e. the entropy of a uniform distribution over every candidate) before
being weighted -- this keeps `gamma` a comparable weight across banks of
different sizes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoveltyConfig:
    """Weights for the observable terms of the Milestone A6 novelty score."""

    alpha: float = 1.0  # weight on task error E
    gamma: float = 1.0  # weight on router entropy H

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")


@dataclass(frozen=True)
class NoveltySignals:
    """One step's raw, observable novelty measurements."""

    error: float  # task error/loss proxy, e.g. 1 - exact_match or normalized loss
    router_entropy: float  # e.g. RouterOutput.entropy for this step
    max_router_entropy: float | None = None  # e.g. log(num_candidates), for normalization

    def to_dict(self) -> dict[str, float | None]:
        return {
            "error": self.error,
            "router_entropy": self.router_entropy,
            "max_router_entropy": self.max_router_entropy,
        }


class NoveltyEstimator:
    """Combines `NoveltySignals` into the scalar novelty score `Controller` reads."""

    def __init__(self, config: NoveltyConfig | None = None) -> None:
        self.config = config if config is not None else NoveltyConfig()

    def normalized_entropy(self, signals: NoveltySignals) -> float:
        """`router_entropy` scaled to roughly `[0, 1]` by `max_router_entropy`
        when given; returned unnormalized (and un-scale-comparable across
        bank sizes) otherwise."""
        if signals.max_router_entropy is None or signals.max_router_entropy <= 0:
            return signals.router_entropy
        return signals.router_entropy / signals.max_router_entropy

    def score(self, signals: NoveltySignals) -> float:
        """`alpha*E + gamma*H` (H normalized to `[0, 1]` when
        `max_router_entropy` is given)."""
        return self.config.alpha * signals.error + self.config.gamma * self.normalized_entropy(
            signals
        )

    def log_entry(self, step: int, signals: NoveltySignals) -> dict[str, float | int | None]:
        """A flat, JSON-serializable record of this step's signals and the
        combined score, for the transition/metrics log (architecture doc
        section 8: novelty signals must be "observable and logged")."""
        return {
            "step": step,
            "error": signals.error,
            "router_entropy": signals.router_entropy,
            "max_router_entropy": signals.max_router_entropy,
            "score": self.score(signals),
        }
